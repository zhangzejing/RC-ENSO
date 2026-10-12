import unittest
import numpy as np
import reservoirpy as rpy
import torch
from threadpoolctl import threadpool_limits

from RCENSO import (get_esn_from_hypers, ndforecast_skill, TPRC_Train_Ensemble,
                    TPRC_Forecast_Ensemble, TPRC_Forecast_Train_Test_Ensemble,
                    realtime_forecast, get_RCTP)
from RCENSO_gpu import iter_train_batches, forecast_members, select_members


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU required')
class GPUChecks(unittest.TestCase):
    def setUp(self):
        rpy.verbosity(0)
        self.X = np.random.default_rng(9).normal(size=(60, 6))
        self.Y = np.sin(self.X[:, :2])

    def hypers(self, deep=False, ip=False):
        h = dict(units=48, lr=1, sr=None, rc_connectivity=0,
                 input_connectivity=.5, input_scaling=1, ridge=.01,
                 noise_rc=0, noise_in=0, output_dim=2, seed=19, ip_reservoir=ip)
        if deep:
            for key in ['units', 'lr', 'sr', 'rc_connectivity', 'input_connectivity',
                        'input_scaling', 'noise_rc', 'noise_in']:
                h[key] = [h[key], h[key]]
            h['units'] = [48, 31]
            h.update(deep=True, deep_ip_list=[ip, False])
        return h

    def test_cpu_cuda_equivalence(self):
        for deep, ip, warmup in [(False, False, 0), (False, True, 0), (True, True, 0),
                                 (False, True, 7), (True, True, 7)]:
            with self.subTest(deep=deep, ip=ip, warmup=warmup), threadpool_limits(limits=1):
                h = self.hypers(deep, ip)
                reference = get_esn_from_hypers(h).fit(self.X, self.Y, warmup=warmup)
                actual = next(iter_train_batches(self.X, self.Y, h, [19], warmup=warmup))[0]
                np.testing.assert_allclose(actual.run(self.X), reference.run(self.X), atol=2e-5, rtol=2e-5)

    def test_batch_size_and_lead_alignment(self):
        h = self.hypers(True, True)
        h['noise_rc'] = [.05, 0]
        a = [m for batch in iter_train_batches(self.X, self.Y, h, [19, 20], 2) for m in batch]
        b = [m for batch in iter_train_batches(self.X, self.Y, h, [19, 20], 1) for m in batch]
        p = forecast_members(a, self.X[:20, :2], self.X[:24, 2:], 5)
        q = forecast_members(b, self.X[:20, :2], self.X[:24, 2:], 5, batch_size=1)
        np.testing.assert_allclose(p, q, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(p[:, :, :, 0], np.broadcast_to(self.X[:20, :2], (2, 20, 2)))
        np.testing.assert_allclose(p[0, :, :, 1], a[0].run(self.X[:20]), atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(p, forecast_members(a, self.X[:20, :2], self.X[:24, 2:], 5))

    def test_mixed_ensemble_and_combined_api(self):
        a = next(iter_train_batches(self.X, self.Y, self.hypers(), [19]))[0]
        h = self.hypers(True, True)
        h['predict_increment'] = True
        b = next(iter_train_batches(self.X, self.Y, h, [20]))[0]
        mixed = forecast_members([a, b], self.X[:20, :2], self.X[:24, 2:], 5)
        for i, model in enumerate([a, b]):
            np.testing.assert_array_equal(mixed[i], forecast_members([model], self.X[:20, :2], self.X[:24, 2:], 5)[0])
        ts = self.X[:, :2]
        tp = get_RCTP(ts, steps=5, order=2)
        rt, mean, std = realtime_forecast([dict(esn=a), dict(esn=b)], ts[-1], tp, len(ts), 4)
        self.assertEqual(rt.shape, (2, 5, 2))
        np.testing.assert_array_equal(rt[:, 0], np.broadcast_to(ts[-1], (2, 2)))
        pred, obs, members = TPRC_Forecast_Train_Test_Ensemble(ts, 35, hypers=self.hypers(),
                                                            nmember=2, steps=5, wl=3, backend='cuda')
        self.assertEqual(members.shape, (2, 22, 6, 5))
        np.testing.assert_array_equal(pred, members.mean(axis=0))
        np.testing.assert_allclose(members[:, :, :, 0], np.broadcast_to(obs, (2, 22, 6)))
        np.testing.assert_allclose(members[0, :, 2:, 2], tp[40:62])

    def test_increment_readout_and_api(self):
        h = self.hypers()
        h['predict_increment'] = True
        m = next(iter_train_batches(self.X, self.X[:, :2], h, [19]))[0]
        np.testing.assert_allclose(m.run(self.X), self.X[:, :2], atol=1e-7)
        ts = self.X[:, :2]
        models, tp = TPRC_Train_Ensemble(ts, len(ts), hypers=h, nmember=2, backend='cuda')
        mean, members = TPRC_Forecast_Ensemble(models, ts[:20], TP_test=tp[:25], steps=5)
        self.assertEqual(members.shape, (2, 20, 2, 5))
        np.testing.assert_array_equal(mean, members.mean(axis=0))

    def test_validation_selection_matches_existing_skill(self):
        rng = np.random.default_rng(4)
        obs = rng.normal(size=(70, 2))
        predictions = rng.normal(size=(25, 70, 2, 20))
        for lead in range(1, 20):
            predictions[0, :-lead, :, lead] = obs[lead:]
        indices, score = select_members(predictions, obs, count=20)
        self.assertEqual(len(set(indices)), 20)
        corr, _ = ndforecast_skill(predictions[indices].mean(axis=0), obs, wl=0)
        self.assertAlmostEqual(score, corr[1:20].mean(), places=12)
        self.assertIn(0, indices)
        indices, score = select_members(predictions, obs, count=20, min_lead=12)
        corr, _ = ndforecast_skill(predictions[indices].mean(axis=0), obs, wl=0)
        self.assertAlmostEqual(score, corr[12:20].mean(), places=12)

    def test_reject_unsupported_and_nan(self):
        h = self.hypers()
        h['lr'] = .9
        with self.assertRaisesRegex(ValueError, 'lr=1'):
            next(iter_train_batches(self.X, self.Y, h, [19]))
        h['lr'] = 1
        x = self.X.copy()
        x[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            next(iter_train_batches(x, self.Y, h, [19]))

    def test_centered_error_keeps_serial_covariance(self):
        from train_gpu import centered_forecast_rmse
        obs = np.zeros((30, 1))
        pred = np.full((30, 1, 5), 3.)
        pred[:, :, 0] = 0
        error = centered_forecast_rmse(pred, obs)
        # Perfectly correlated errors retain RMSE=3, rather than 3/sqrt(3).
        np.testing.assert_allclose(error[:4], [1., 2., 3., 3.])
        self.assertTrue(np.isnan(error[4]))

    def test_candidate_overrides_and_checkpoint_reload(self):
        import pickle
        from train_gpu import train
        configs = [dict(deep=True, increment=False, seed=19,
                        overrides=dict(units=[37, 23], noise_rc=[0., 0.],
                                       input_scaling=[1., .25])),
                   dict(deep=False, increment=True, seed=20,
                        overrides=dict(units=41, noise_rc=0.))]
        models = [None] * 2
        ts = self.X[:, :2]
        for indices, batch in train(ts, configs, batch_size=2):
            for i, model in zip(indices, batch):
                models[i] = model
        self.assertEqual(models[0].layers[1]['Win'].shape, (23, 37))
        self.assertEqual(models[1].layers[0]['Win'].shape, (41, 6))
        tp = get_RCTP(ts, steps=5, order=2)
        restored = pickle.loads(pickle.dumps(models))
        np.testing.assert_array_equal(forecast_members(models, ts, tp, 5),
                                      forecast_members(restored, ts, tp, 5))
        from train_gpu import paper_hypers
        for config, model in zip(configs, models):
            h = paper_hypers(config['deep'], config['increment'])
            h.update(config['overrides'])
            ref = next(iter_train_batches(np.hstack((ts[:-1], tp[:len(ts)-1])), ts[1:], h,
                                          [config['seed']]))[0]
            np.testing.assert_allclose(forecast_members([model], ts, tp, 5),
                                       forecast_members([ref], ts, tp, 5), atol=2e-5, rtol=2e-5)

    def test_v3_saved_models_without_external_json(self):
        import hashlib
        import io
        import pickle
        import tempfile
        from contextlib import redirect_stdout
        from pathlib import Path
        from unittest.mock import patch
        import pandas as pd
        import xarray as xr
        import train_gpu
        from forecast import forecast_real_time_v3 as v3
        from RCENSO_gpu import initialize_models

        with tempfile.TemporaryDirectory(prefix='rcenso-v3-test-') as folder:
            root = Path(folder).resolve()
            self.assertEqual(root.parent, Path(tempfile.gettempdir()).resolve())
            source = root/'forecast/oras5_indices_1958-2026-07.nc'
            source.parent.mkdir()
            names = ['Nino34', 'WWV'] + [f'mode{i}' for i in range(2, 10)]
            data = np.random.default_rng(5).normal(size=(571, 10))
            xr.Dataset({name: ('time', data[:, i]) for i, name in enumerate(names)},
                       coords={'time': pd.date_range('1979-01-01', periods=571, freq='MS')}).to_netcdf(source)
            h = dict(self.hypers(), units=4, output_dim=10, predict_increment=True)
            models = initialize_models(h, list(range(19, 39)), 14, 10)
            for model in models:
                model.Wout = np.zeros((5, 10))
            selection = dict(indices=list(range(20)), configs=[dict(deep=False, increment=True, seed=m.seed) for m in models])
            manifest = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            selection=selection, train_start='1979-01', train_end='2026-07',
                            mode_names=names, forecast_steps=21, forecast_noise=False, tp_order=2)
            output = root/'forecast/desn_forecast_v3_2026-07'
            (output/'models').mkdir(parents=True)
            with (output/'models/selected_members.pkl').open('wb') as f:
                pickle.dump(dict(manifest=manifest, selected_models=[dict(esn=m) for m in models]), f)
            benchmark = root/'gpu_runs/benchmark'
            benchmark.mkdir(parents=True)
            np.savez(benchmark/'independent_predictions.npz', observations=data[:70],
                     predictions=np.broadcast_to(data[None, :70, :, None], (20, 70, 10, 22)))
            (root/'data/sourcedata').mkdir(parents=True)
            with (root/'data/sourcedata/fs_ds_to_plot_end.pickle').open('wb') as f:
                pickle.dump(dict(DESN=np.linspace(1., .2, 22), ESN=np.linspace(1., .1, 22)), f)
            self.assertFalse(list(root.rglob('*.json')))
            with patch.object(v3, 'ROOT', root), patch.object(train_gpu, 'ROOT', root), redirect_stdout(io.StringIO()):
                summary = v3.run(benchmark)
            self.assertEqual(summary['n_members'], 20)
            with xr.open_dataset(output/'realtime_forecast.nc') as forecast:
                np.testing.assert_allclose(forecast.forecast_mean, np.broadcast_to(data[-1], (22, 10)), atol=2e-6)
                self.assertEqual(str(forecast.time.values[-1])[:10], '2028-04-01')
            self.assertTrue((output/'first_month_diagnosis.png').is_file())


if __name__ == '__main__':
    unittest.main()
