"""CUDA batches for the paper's W=0, lr=1 ESN/DESN configurations.

ReservoirPy still initializes the weights. Reservoir features use float32;
the dual ridge system uses float64 because the paper's ridge is only 5e-6.
"""
import numpy as np
import torch
from scipy import sparse


def _dense(value, device, dtype=torch.float32):
    if sparse.issparse(value):
        value = value.toarray()
    return torch.as_tensor(value, device=device, dtype=dtype)


class GPUESN:
    """CPU-stored checkpoint with a ReservoirPy-compatible run method."""

    def __init__(self, layers, ridge, seed, increment=False):
        self.layers = layers
        self.ridge = ridge
        self.seed = seed
        self.increment = increment
        self.Wout = None

    def run(self, X):
        with torch.inference_mode():
            batch = _DeviceBatch([self], "cuda")
            x = _dense(np.asarray(X)[None], "cuda")
            return batch.predict(x)[0].cpu().numpy()


def initialize_models(hypers, seeds, input_dim, output_dim):
    from RCENSO import get_esn_from_hypers

    units = hypers['units'] if hypers.get('deep', False) else [hypers['units']]
    lr = hypers['lr'] if hypers.get('deep', False) else [hypers['lr']]
    rc = hypers['rc_connectivity'] if hypers.get('deep', False) else [hypers['rc_connectivity']]
    # shortcut: only the paper's memoryless reservoirs, use ReservoirPy for W!=0 or lr!=1.
    if any(c != 0 for c in rc) or any(a != 1 for a in lr) or hypers.get('online', False):
        raise ValueError('CUDA batch backend requires rc_connectivity=0, lr=1 and Ridge readout')
    if not units or any(u <= 0 for u in units) or hypers['ridge'] <= 0:
        raise ValueError('Positive reservoir sizes and ridge are required')
    models = []
    for seed in seeds:
        h = dict(hypers, seed=int(seed), sr=[None] * len(units) if hypers.get('deep') else None)
        model = get_esn_from_hypers(h)
        # A zero matrix has no spectral radius to rescale; skip ARPACK on it.
        model.initialize(np.zeros((1, input_dim)), np.zeros((1, output_dim)))
        layers = []
        for node in model.nodes:
            if type(node).__name__ not in ('Reservoir', 'IPReservoir'):
                continue
            ip = type(node).__name__ == 'IPReservoir'
            layers.append(dict(
                Win=node.Win.astype(np.float32),
                bias=(node.bias.toarray() if sparse.issparse(node.bias) else np.asarray(node.bias)).ravel().astype(np.float32),
                a=np.ones(node.output_dim, dtype=np.float32),
                b=np.zeros(node.output_dim, dtype=np.float32), ip=ip,
                noise_rc=float(node.noise_rc), noise_in=float(node.noise_in),
                learning_rate=float(node.learning_rate) if ip else 0.,
                mu=float(node.mu) if ip else 0., sigma=float(node.sigma) if ip else 1.,
                epochs=int(node.epochs) if ip else 0,
            ))
        models.append(GPUESN(layers, h['ridge'], seed, h.get('predict_increment', False)))
    return models


class _DeviceBatch:
    def __init__(self, models, device):
        self.models = models
        self.device = device
        self.layers = []
        self.generators = [torch.Generator(device=device).manual_seed(int(m.seed)) for m in models]
        for i in range(len(models[0].layers)):
            source = [m.layers[i] for m in models]
            layer = dict(source[0])
            for key in ('Win', 'bias', 'a', 'b'):
                layer[key] = torch.stack([_dense(s[key], device) for s in source])
            self.layers.append(layer)
        self.Wout = None if models[0].Wout is None else torch.stack([
            _dense(m.Wout, device, torch.float64) for m in models])

    def noise(self, shape, gain):
        return torch.stack([torch.randn(shape, generator=g, device=self.device)
                            for g in self.generators]) * gain

    def features(self, x, train=False, warmup=0):
        states = []
        for layer in self.layers:
            if train and layer['noise_in']:
                x = x + self.noise(x.shape[1:], layer['noise_in'])
            z = torch.bmm(x, layer['Win'].transpose(1, 2)) + layer['bias'][:, None]
            if layer['ip']:
                # IPReservoir uses external noise (before tanh), standard Reservoir after tanh.
                if train:
                    for _ in range(layer['epochs']):
                        pre = z + self.noise(z.shape[1:], layer['noise_rc']) if layer['noise_rc'] else z
                        a, b = layer['a'], layer['b']
                        eta, mu, sig2 = layer['learning_rate'], layer['mu'], layer['sigma'] ** 2
                        for t in range(warmup, pre.shape[1]):
                            u = pre[:, t]
                            y = torch.tanh(a * u + b)
                            db = -eta * (-mu / sig2 + y / sig2 * (2 * sig2 + 1 - y*y + mu*y))
                            a = a + eta / a + db * u
                            b = b + db
                        layer['a'], layer['b'] = a, b
                    if layer['noise_rc']:
                        z = z + self.noise(z.shape[1:], layer['noise_rc'])
                x = torch.tanh(z * layer['a'][:, None] + layer['b'][:, None])
            else:
                x = torch.tanh(z)
                if train and layer['noise_rc']:
                    x = x + self.noise(x.shape[1:], layer['noise_rc'])
            states.append(x)
        return torch.cat(states, dim=2)

    def fit(self, X, Y, warmup=0):
        x = _dense(X, self.device).expand(len(self.models), -1, -1)
        features = self.features(x, train=True, warmup=warmup)[:, warmup:].double()
        features = torch.cat((torch.ones_like(features[:, :, :1]), features), dim=2)
        y = _dense(Y[warmup:], self.device, torch.float64).expand(len(self.models), -1, -1)
        if self.models[0].increment:
            y = y - x[:, warmup:, :y.shape[2]].double()
        gram = torch.bmm(features, features.transpose(1, 2))
        gram.diagonal(dim1=1, dim2=2).add_(self.models[0].ridge)
        alpha = torch.linalg.solve(gram, y)
        self.Wout = torch.bmm(features.transpose(1, 2), alpha)
        for i, model in enumerate(self.models):
            model.Wout = self.Wout[i].cpu().numpy()
            for source, layer in zip(model.layers, self.layers):
                source['a'] = layer['a'][i].cpu().numpy()
                source['b'] = layer['b'][i].cpu().numpy()
        residual = torch.bmm(gram, alpha) - y
        if not torch.isfinite(self.Wout).all() or residual.abs().max().item() > 1e-5:
            raise RuntimeError('Non-finite or inaccurate ridge solve')

    def predict(self, x):
        features = self.features(x).double()
        y = torch.bmm(features, self.Wout[:, 1:]) + self.Wout[:, :1]
        if self.models[0].increment:
            y = y + x[:, :, :y.shape[2]].double()
        return y


def iter_train_batches(X, Y, hypers, seeds, batch_size=2, warmup=0, device='cuda'):
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('A CUDA-enabled PyTorch installation and GPU are required')
    if batch_size < 1 or not 0 <= warmup < len(X) or len(X) != len(Y):
        raise ValueError('Invalid batch size, training length or warmup')
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError('Training inputs and targets must be finite')
    # Disable reduced-precision products: tiny ridge needs a well-resolved Gram matrix.
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        for start in range(0, len(seeds), batch_size):
            models = initialize_models(hypers, seeds[start:start + batch_size], X.shape[1], Y.shape[1])
            batch = _DeviceBatch(models, device)
            batch.fit(X, Y, warmup)
            del batch
            yield models


def forecast_members(models, Ytest, TP, steps=22, batch_size=2, device='cuda'):
    """Parallel independent forecast origins; lead zero is the observation.

    Forecast noise is disabled; training noise is preserved. This makes the
    deterministic forecast repeatable and avoids stochastic boundary kinks.
    """
    if not models or steps < 1 or not len(Ytest) or len(TP) < len(Ytest) + steps - 1 or batch_size < 1:
        raise ValueError('Models, positive horizon/batch size and complete TP are required')
    if not np.isfinite(Ytest).all() or not np.isfinite(TP).all():
        raise ValueError('Forecast inputs must be finite')
    predictions = np.empty((len(models), len(Ytest), Ytest.shape[1], steps))
    groups = {}
    for index, model in enumerate(models):
        signature = (model.increment, tuple((l['Win'].shape, l['ip']) for l in model.layers))
        groups.setdefault(signature, []).append(index)
    with torch.inference_mode():
        for compatible in groups.values():
            for start in range(0, len(compatible), batch_size):
                indices = compatible[start:start + batch_size]
                group = [models[i] for i in indices]
                batch = _DeviceBatch(group, device)
                state = _dense(Ytest, device).expand(len(group), -1, -1)
                predictions[indices, :, :, 0] = Ytest
                for lead in range(1, steps):
                    forcing = _dense(TP[lead - 1:len(Ytest) + lead - 1], device).expand(len(group), -1, -1)
                    state = batch.predict(torch.cat((state.float(), forcing), dim=2))
                    predictions[indices, :, :, lead] = state.cpu().numpy()
                del batch
    return predictions


def select_members(predictions, observations, count=20, max_lead=19, min_lead=1):
    """Greedy addition then one-swap refinement on validation correlation.

    Uses the same trailing 3-month mean and endpoint alignment as ndforecast_skill.
    Only Niño3.4 (column zero) is needed for member selection.
    """
    if not 1 <= count <= len(predictions) or not np.isfinite(predictions).all() or not np.isfinite(observations).all():
        raise ValueError('Invalid ensemble size or non-finite predictions')
    if observations[:, 0].std() == 0:
        raise ValueError('Validation observations must have nonzero variance')
    def smooth(x):
        sums = np.cumsum(x, axis=-1)
        return (sums - np.concatenate((np.zeros_like(sums[..., :3]), sums[..., :-3]), axis=-1)) / np.minimum(np.arange(x.shape[-1])+1, 3)
    lead_max = min(max_lead, predictions.shape[-1] - 1, len(observations) - 3)
    if not 1 <= min_lead <= lead_max:
        raise ValueError('Validation window is too short')
    # Different leads have different endpoints; keep those valid sample sets intact.
    pairs = []
    for lead in range(min_lead, lead_max + 1):
        p = smooth(predictions[:, :-lead, 0, lead])
        o = smooth(observations[lead:, 0])
        p -= p.mean(axis=1, keepdims=True)
        o -= o.mean()
        pairs.append((p, o))
    def scores(summed):
        values = []
        for (p, o), s in zip(pairs, summed):
            numerator = s @ o
            denominator = np.linalg.norm(s, axis=-1) * np.linalg.norm(o)
            values.append(np.divide(numerator, denominator, out=np.full_like(numerator, -np.inf), where=denominator > 0))
        return np.mean(values, axis=0)
    selected = []
    summed = [np.zeros(p.shape[1]) for p, _ in pairs]
    while len(selected) < count:
        candidate_scores = scores([s[None] + p for s, (p, _) in zip(summed, pairs)])
        candidate_scores[selected] = -np.inf
        chosen = int(np.argmax(candidate_scores))
        selected.append(chosen)
        summed = [s + p[chosen] for s, (p, _) in zip(summed, pairs)]
    best = float(scores([s[None] for s in summed])[0])
    for _ in range(count):
        improved = False
        for slot, old in enumerate(selected):
            candidates = scores([s[None] - p[old] + p for s, (p, _) in zip(summed, pairs)])
            candidates[selected] = -np.inf
            new = int(np.argmax(candidates))
            if candidates[new] > best + 1e-8:
                summed = [s - p[old] + p[new] for s, (p, _) in zip(summed, pairs)]
                selected[slot], best, improved = new, float(candidates[new]), True
        if not improved:
            break
    return selected, best
