"""Train on 1958–1999 and select 20 members on the 2002–2023 benchmark validation."""
import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import reservoirpy as rpy
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from RCENSO import get_RCTP, ndforecast_skill, standardize_time_to_month_start
from RCENSO_gpu import (GPUESN, _DeviceBatch, initialize_models,
                       forecast_members, select_members)


def paper_hypers(deep, increment=False):
    h = dict(units=20000, lr=1, sr=1, rc_connectivity=0., input_connectivity=.1,
             input_scaling=1., ridge=5e-6, noise_rc=.05, noise_in=0., output_dim=10,
             seed=None, ip_reservoir=True, predict_increment=increment)
    if deep:
        for key in ('units', 'lr', 'sr', 'rc_connectivity', 'input_connectivity',
                    'input_scaling', 'noise_rc', 'noise_in'):
            h[key] = [h[key], h[key]]
        h.update(units=[20000, 12000], noise_rc=[.05, 0.], deep=True,
                 deep_ip_list=[True, False])
    return h


def arrays(ds, start, end):
    return ds.sel(time=slice(start, end)).to_array().values.T


def train(X, configs, batch_size):
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA-enabled PyTorch installation and GPU are required')
    if batch_size < 1 or len(X) < 3 or not np.isfinite(X).all():
        raise ValueError('Finite monthly data and a positive batch size are required')
    torch.backends.cuda.matmul.allow_tf32 = False
    forcing = get_RCTP(X, steps=0, order=2)
    input_values = np.hstack((X[:-1], forcing[:-1]))
    groups = {}
    for i, config in enumerate(configs):
        h = paper_hypers(config['deep'], config['increment'])
        h.update(config.get('overrides', {}))
        groups.setdefault(json.dumps(h, sort_keys=True), []).append((i, config))
    weights = {}
    for key, group in groups.items():
        h = json.loads(key)
        base_h = dict(h, predict_increment=False, noise_rc=[0., 0.] if h.get('deep') else 0.,
                      input_scaling=[1., 1.] if h.get('deep') else 1.)
        base_key = json.dumps(base_h, sort_keys=True)
        for start in range(0, len(group), batch_size):
            chunk = group[start:start + batch_size]
            models = []
            for _, config in chunk:
                seed = config['seed']
                cache_key = (base_key, seed)
                if cache_key not in weights:
                    weights[cache_key] = initialize_models(base_h, [seed], input_values.shape[1], X.shape[1])[0]
                layers = [dict(layer) for layer in weights[cache_key].layers]
                for j, layer in enumerate(layers):
                    scale = h['input_scaling'][j] if h.get('deep') else h['input_scaling']
                    layer['noise_rc'] = h['noise_rc'][j] if h.get('deep') else h['noise_rc']
                    if scale != 1:
                        layer['Win'] = layer['Win'] * scale
                models.append(GPUESN(layers, h['ridge'], seed, h.get('predict_increment', False)))
            with torch.inference_mode():
                device_batch = _DeviceBatch(models, 'cuda')
                device_batch.fit(input_values, X[1:])
                del device_batch
            yield [i for i, _ in chunk], models


def skill(pred, obs):
    corr, rmse = ndforecast_skill(pred, obs, wl=0, ismv3=True)
    return dict(correlation=corr.tolist(), rmse=rmse.tolist(),
                mean_correlation_1_19=float(corr[1:20].mean()),
                mean_rmse_1_19=float(rmse[1:20].mean()))


def centered_forecast_rmse(pred, obs):
    """Empirical error of centered forecast paths, retaining serial covariance."""
    steps = pred.shape[2]
    result = np.full(steps, np.nan)
    for lead in range(steps - 1):
        start = 1 if lead == 0 else 0
        stop = len(obs) - lead - 1
        if lead == 0:
            forecast = (obs[start-1:stop-1, 0] + pred[start:stop, 0, 0] + pred[start:stop, 0, 1]) / 3
        else:
            forecast = pred[start:stop, 0, lead-1:lead+2].mean(axis=1)
        target = (obs[start+lead-1:stop+lead-1, 0] + obs[start+lead:stop+lead, 0]
                  + obs[start+lead+1:stop+lead+1, 0]) / 3
        result[lead] = np.sqrt(np.mean((forecast - target) ** 2))
    return result


def benchmark_figure(out, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    leads = np.arange(1, 20)
    for key, label, color in [('selected_test', 'GPU selected 20', '#0072B2'),
                              ('baseline_desn', 'plot_result DESN', '#D55E00'),
                              ('baseline_esn', 'plot_result ESN', '#CC79A7')]:
        values = report[key]['correlation'] if isinstance(report[key], dict) else report[key]
        ax.plot(leads, np.asarray(values)[1:20], label=label, color=color, marker='o', ms=3)
    ax.axhline(.5, color='gray', linestyle='--', lw=.8)
    ax.set(xlabel='Lead (months)', ylabel='Niño3.4 correlation', ylim=(0, 1.02),
           title='Niño3.4 forecast skill (2002–2023)')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'benchmark_skill.png', dpi=180); plt.close(fig)


def figures(out, report, latest, old_forecast, realtime):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    benchmark_figure(out, report)

    observed = latest.Nino34.isel(time=slice(-8, None))
    init = pd.Timestamp(latest.time.values[-1])
    dates = pd.date_range(init, periods=len(realtime), freq='MS')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, centered in zip(axes, (False, True)):
        history = observed.values
        values = np.r_[history, realtime[1:, 0]]
        combined_dates = pd.DatetimeIndex(observed.time.values).append(dates[1:])
        if centered:
            values = pd.Series(values).rolling(3, center=True).mean().values
        ax.plot(combined_dates[:len(history)], values[:len(history)], 'ko-', ms=3, label='Observed / boundary')
        ax.plot(combined_dates[len(history)-1:], values[len(history)-1:], 'o-', ms=3, color='#0072B2', label='GPU selected 20')
        if old_forecast is not None:
            old = np.r_[history, old_forecast[1:]]
            if centered:
                old = pd.Series(old).rolling(3, center=True).mean().values
            ax.plot(combined_dates[len(history)-1:len(old)], old[len(history)-1:], '--',
                    color='#D55E00', label='Original saved forecast')
        ax.axvline(init, color='gray', lw=.8, ls='--')
        ax.set(title='Centered 3-month mean' if centered else 'Monthly forecast', ylabel='Niño3.4 (°C)')
        ax.legend(fontsize=8); ax.tick_params(axis='x', rotation=35)
    fig.tight_layout(); fig.savefig(out/'first_month_diagnosis.png', dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--members-per-architecture', type=int, default=32,
                        help='For each architecture and each readout target (pool=4*N)')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--tune', action='store_true',
                        help='Search training noise and second-layer input scaling on validation only')
    parser.add_argument('--selection-min-lead', type=int, default=12,
                        help='First lead included in the validation selection score (last=19)')
    parser.add_argument('--output', type=Path, default=ROOT/'gpu_runs/2026-10-11-validation-2002-2023')
    args = parser.parse_args()
    if args.members_per_architecture < 5 or args.batch_size < 1 or not 1 <= args.selection_min_lead <= 19:
        parser.error('At least 20 candidates and a positive batch size are required')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    cache = out/'candidates'; cache.mkdir(exist_ok=True)
    model_dir = out/'models'; model_dir.mkdir(exist_ok=True)
    rpy.verbosity(0)
    source = ROOT/'data/oras5_indices_1958-2025.nc'
    ds = standardize_time_to_month_start(xr.open_dataset(source)).load()
    ds = ds[['Nino34', 'WWV'] + [v for v in ds if v not in ('Nino34', 'WWV')]]
    configs = [dict(deep=deep, increment=inc, seed=20261011 + i + 100000*deep + 200000*inc)
               for deep in (False, True) for inc in (False, True)
               for i in range(args.members_per_architecture)]
    if args.tune:
        configs = []
        for deep in (False, True):
            for noise in (.025, .05, .075, .1):
                for scale in ((.25, 1.) if deep else (1.,)):
                    for i in range(args.members_per_architecture):
                        overrides = dict(noise_rc=[noise, 0.] if deep else noise,
                                         input_scaling=[1., scale] if deep else scale)
                        configs.append(dict(deep=deep, increment=False, overrides=overrides,
                                            seed=20261011 + i + 100000 * deep))
    manifest = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    configs=configs, selection_train='1958–1999', validation='2002–2023',
                    benchmark_train='1958–1999', test='2002–2023', steps=22,
                    selection_uses_benchmark=True,
                    selection_min_lead=args.selection_min_lead,
                    forecast_noise=False, feature_dtype='float32', ridge_dtype='float64')
    manifest_file = out/'manifest.json'
    if manifest_file.exists() and json.loads(manifest_file.read_text()) != manifest:
        raise RuntimeError('Output directory contains a different experiment; use a new output path')
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    validation = arrays(ds, '2002-01', '2023-12')
    validation_tp = get_RCTP(validation, steps=22, order=2)
    predictions = np.empty((len(configs), len(validation), 10, 22))
    missing = [i for i in range(len(configs)) if not (cache/f'{i:04d}.npz').exists()]
    candidate_start = time.perf_counter()
    for indices, models in train(arrays(ds, '1958-01', '1999-12'), [configs[i] for i in missing], args.batch_size):
        indices = [missing[i] for i in indices]
        started = time.perf_counter()
        pred = forecast_members(models, validation, validation_tp, steps=22, batch_size=args.batch_size)
        for index, values in zip(indices, pred):
            np.savez_compressed(cache/f'{index:04d}.npz', predictions=values)
        print(f'Candidates {indices}: saved; validation {time.perf_counter()-started:.2f}s; '
              f'total {time.perf_counter()-candidate_start:.1f}s', flush=True)
        del models
    for i in range(len(configs)):
        with np.load(cache/f'{i:04d}.npz') as f:
            predictions[i] = f['predictions']
    selected, score = select_members(predictions, validation, count=20, min_lead=args.selection_min_lead)
    selected_configs = [configs[i] for i in selected]
    selection = dict(indices=selected, configs=selected_configs, validation_score=score)
    (out/'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    print(f'Selected exactly 20: {selected}; validation ACC{args.selection_min_lead}–19={score:.4f}', flush=True)
    report = dict(manifest, gpu=torch.cuda.get_device_name(0), pool_size=len(configs),
                  selected_pool_indices=selected, selected_configs=selected_configs,
                  candidate_seconds=time.perf_counter()-candidate_start,
                  selected_validation=skill(predictions[selected].mean(axis=0), validation))
    del predictions

    test = arrays(ds, '2002-01', '2023-12')
    test_tp = get_RCTP(test, steps=22, order=2)
    test_preds = np.empty((20, len(test), 10, 22))
    benchmark_models = [None] * 20
    started = time.perf_counter()
    for indices, models in train(arrays(ds, '1958-01', '1999-12'), selected_configs, args.batch_size):
        for index, model in zip(indices, models):
            benchmark_models[index] = dict(member=index, esn=model, **selected_configs[index])
        del models
    # Use the same member grouping as a subsequent checkpoint reload.
    test_preds = forecast_members([item['esn'] for item in benchmark_models], test, test_tp, 22, args.batch_size)
    report['benchmark_seconds'] = time.perf_counter() - started
    report['selected_test'] = skill(test_preds.mean(axis=0), test)
    np.savez_compressed(out/'independent_predictions.npz', predictions=test_preds, observations=test)
    with open(ROOT/'data/sourcedata/fs_ds_to_plot_end.pickle', 'rb') as f:
        baseline = pickle.load(f)
    report['baseline_desn'] = baseline['DESN'].tolist()
    report['baseline_esn'] = baseline['ESN'].tolist()
    diff = np.array(report['selected_test']['correlation'])[1:20] - baseline['DESN'][1:20]
    report['parity'] = dict(mean_difference_1_19=float(diff.mean()), worst_lead_difference=float(diff.min()),
                            long_mean_difference_12_19=float(diff[11:19].mean()),
                            leads_within_0_01=int((diff >= -.01).sum()),
                            pass_mean=bool(diff.mean() >= -.01), pass_every_lead=bool(diff.min() >= -.01),
                            tolerance=.01)
    print('Benchmark validation after member selection:', report['selected_test'], 'parity', report['parity'], flush=True)
    report['versions'] = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__)
    (out/'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    benchmark_figure(out, report)
    if not report['parity']['pass_every_lead']:
        print('Long-lead benchmark target not met; no release prepared.', flush=True)
        return
    reforecast_obs = arrays(ds, '1979-01', '2023-12')
    reforecast_tp = get_RCTP(reforecast_obs, steps=22, order=2)
    reforecast_preds = np.empty((20, len(reforecast_obs), 10, 22))
    for indices, models in train(reforecast_obs, selected_configs, args.batch_size):
        reforecast_preds[indices] = forecast_members(models, reforecast_obs, reforecast_tp, 22, args.batch_size)
    report['reforecast'] = skill(reforecast_preds.mean(axis=0), reforecast_obs)
    with open(ROOT/'data/sourcedata/rfs_ds_to_plot_end.pickle', 'rb') as f:
        reforecast_baseline = pickle.load(f)
    report['reforecast']['baseline_desn'] = reforecast_baseline['DESN'].tolist()
    rdiff = np.array(report['reforecast']['correlation'])[1:22] - reforecast_baseline['DESN'][1:22]
    report['reforecast']['mean_difference_1_21'] = float(rdiff.mean())
    report['reforecast']['worst_lead_difference_1_21'] = float(rdiff.min())
    np.savez_compressed(out/'reforecast_predictions.npz', predictions=reforecast_preds,
                        observations=reforecast_obs)
    (out/'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    leads = np.arange(1, 22)
    ax.plot(leads, report['reforecast']['correlation'][1:22], 'o-', label='GPU selected 20')
    ax.plot(leads, reforecast_baseline['DESN'][1:22], 'o-', label='plot_result DESN')
    ax.set(xlabel='Lead (months)', ylabel='Niño3.4 correlation',
           title='In-sample reforecast skill (1979–2023)', ylim=(0, 1.02))
    ax.legend(); fig.tight_layout(); fig.savefig(out/'reforecast_skill.png', dpi=180); plt.close(fig)
    if rdiff.min() < -.01:
        print('In-sample reforecast target not met; no release prepared.', flush=True)
        return
    benchmark_checkpoint = model_dir/'benchmark_members.pkl'
    with open(benchmark_checkpoint.with_suffix('.tmp'), 'wb') as f:
        pickle.dump(dict(selected_models=benchmark_models, selection=selection,
                         train_period='1958–1999', mode_names=list(ds.data_vars)), f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    benchmark_checkpoint.with_suffix('.tmp').replace(benchmark_checkpoint)
    del benchmark_models
    from forecast.forecast_real_time_v3 import run
    run(out, args.batch_size)


if __name__ == '__main__':
    main()
