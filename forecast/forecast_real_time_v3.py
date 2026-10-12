"""Refit the selected 20 members on 1979-01 through 2026-07 and forecast 21 months."""
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
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from RCENSO import get_RCTP, ndforecast_skill, standardize_time_to_month_start
from RCENSO_gpu import forecast_members
from train_gpu import train, figures, centered_forecast_rmse, benchmark_report


def run(benchmark_dir, batch_size=2, output=None):
    benchmark_dir = Path(benchmark_dir).resolve()
    out = Path(output).resolve() if output else ROOT/'forecast/desn_forecast_v3_2026-07'
    checkpoint = out/'models/selected_members.pkl'
    checkpoint_source = checkpoint
    saved = None
    started = time.perf_counter()
    if checkpoint.exists():
        with checkpoint.open('rb') as f:
            saved = pickle.load(f)
    else:
        member_files = sorted(checkpoint.parent.glob('member_*.pkl'))
        if member_files:
            if member_files != [checkpoint.parent/f'member_{i:02d}.pkl' for i in range(1, 21)]:
                raise ValueError('All 20 member files (member_01.pkl through member_20.pkl) are required')
            members = []
            for i, path in enumerate(member_files):
                with path.open('rb') as f:
                    member = pickle.load(f)
                if member['member'] != i or (members and member['manifest'] != members[0]['manifest']):
                    raise ValueError('Member files have inconsistent metadata or ordering')
                members.append(member)
            saved = dict(manifest=members[0]['manifest'],
                         selected_models=[member['selected_model'] for member in members])
            checkpoint_source = checkpoint.parent
    if (benchmark_dir/'selection.json').exists():
        selection = json.loads((benchmark_dir/'selection.json').read_text(encoding='utf-8'))
    elif saved is not None:
        selection = saved['manifest']['selection']
    else:
        with (benchmark_dir/'models/benchmark_members.pkl').open('rb') as f:
            selection = pickle.load(f)['selection']
    configs = selection['configs']
    if len(configs) != 20 or len(selection['indices']) != 20 or len(set(selection['indices'])) != 20 or batch_size < 1:
        raise ValueError('Exactly 20 distinct selected members and a positive batch size are required')
    source = ROOT/'forecast/oras5_indices_1958-2026-07.nc'
    with xr.open_dataset(source) as ds:
        latest = standardize_time_to_month_start(ds).load()
    latest = latest[['Nino34', 'WWV'] + [v for v in latest if v not in ('Nino34', 'WWV')]]
    history_ds = latest.sel(time=slice('1979-01', '2026-07'))
    dates = pd.DatetimeIndex(history_ds.time.values)
    if not dates.equals(pd.date_range('1979-01-01', '2026-07-01', freq='MS')):
        raise ValueError('Training requires every month from 1979-01 through 2026-07')
    history = history_ds.to_array().values.T
    if not np.isfinite(history).all():
        raise ValueError('Training data contains missing or non-finite climate indices')
    out.mkdir(parents=True, exist_ok=True)
    model_dir = out/'models'; model_dir.mkdir(exist_ok=True)
    manifest = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    selection=selection, train_start='1979-01', train_end='2026-07',
                    mode_names=list(latest.data_vars), forecast_steps=21,
                    forecast_noise=False, tp_order=2)
    manifest_file = out/'manifest.json'
    if manifest_file.exists() and json.loads(manifest_file.read_text(encoding='utf-8')) != manifest:
        raise RuntimeError('Output contains a different model selection; choose a new output directory')
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if saved is not None:
        if saved['manifest'] != manifest or len(saved['selected_models']) != 20:
            raise ValueError('Saved checkpoint does not match the requested selection and training period')
        selected_models = saved['selected_models']
    else:
        rpy.verbosity(0)
        selected_models = [None] * 20
        for indices, models in train(history, configs, batch_size):
            for index, model in zip(indices, models):
                selected_models[index] = dict(member=index, pool_index=selection['indices'][index],
                    arch='DESN' if configs[index]['deep'] else 'ESN', esn=model, **configs[index])
            print(f'V3 refit members {indices}; elapsed {time.perf_counter()-started:.1f}s', flush=True)
        temp_checkpoint = checkpoint.with_suffix('.tmp')
        with temp_checkpoint.open('wb') as f:
            pickle.dump(dict(selected_models=selected_models, manifest=manifest), f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        temp_checkpoint.replace(checkpoint)
    tp = get_RCTP(history, steps=22, order=2)[len(history)-1:]
    models = [item['esn'] for item in selected_models]
    members = forecast_members(models, history[-1:], tp, 22, batch_size)[:, 0].transpose(0, 2, 1)
    np.testing.assert_array_equal(members[:, 0], np.broadcast_to(history[-1], (20, 10)))
    if not np.isfinite(members).all():
        raise RuntimeError('Non-finite realtime forecast')
    mean = members.mean(axis=0)
    with np.load(benchmark_dir/'independent_predictions.npz') as f:
        historical_pred = f['predictions'].mean(axis=0)
        historical_obs = f['observations']
    if (benchmark_dir/'results.json').exists():
        report = json.loads((benchmark_dir/'results.json').read_text(encoding='utf-8'))
    else:
        report = benchmark_report(historical_pred, historical_obs)
        report['selection_uses_benchmark'] = True
    _, monthly_rmse = ndforecast_skill(historical_pred, historical_obs, wl=0, ismv3=False)
    centered_rmse = centered_forecast_rmse(historical_pred, historical_obs)
    old_path = ROOT/'forecast/desn_forecast_2026-07/desn_forecast_2026-07.nc'
    original = None
    if old_path.exists():
        with xr.open_dataset(old_path) as ds:
            original = ds.forecast_mean.sel(mode='Nino34').values
    summary = dict(train_period='1979-01 through 2026-07', n_members=20,
                   checkpoint=str(checkpoint_source), elapsed_seconds=time.perf_counter()-started,
                   historical_benchmark=report['parity'],
                   first_month=dict(observation=float(history[-1, 0]),
                       last_observed_increment=float(history[-1, 0]-history[-2, 0]),
                       original_forecast_increment=float(original[1]-original[0]) if original is not None else None,
                       new_forecast_increment=float(mean[1, 0]-mean[0, 0])))
    times = pd.date_range('2026-07-01', periods=22, freq='MS')
    forecast = xr.Dataset(dict(forecast_members=(('member', 'time', 'mode'), members),
                               forecast_mean=(('time', 'mode'), mean),
                               forecast_std=(('time', 'mode'), members.std(axis=0)),
                               nino34_historical_monthly_rmse=(('time',), monthly_rmse),
                               nino34_historical_centered_rmse=(('time',), centered_rmse)),
                          coords=dict(member=np.arange(20), time=times, mode=list(latest.data_vars)),
                          attrs=dict(train_period=summary['train_period'], n_members_selected=20,
                                     forecast_noise='disabled', historical_error_period='2002-2023',
                                     historical_error_training='1958-1999',
                                     error_period_used_for_selection=int(report.get('selection_uses_benchmark', False)),
                                     uncertainty_description='Historical RMSE and ensemble spread are separate quantities'))
    forecast.to_netcdf(out/'realtime_forecast.nc')
    frame = pd.DataFrame(mean, index=times, columns=list(latest.data_vars))
    # Use one continuous observed/forecast path across the initialization boundary.
    path = pd.Series(np.r_[history[-2, 0], mean[:, 0]])
    frame['Nino34_centered_3month'] = path.rolling(3, center=True).mean().values[1:]
    frame.index.name = 'time'; frame.to_csv(out/'forecast_mean.csv')
    (out/'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    figures(out, report, latest, original, mean, monthly_rmse, centered_rmse)
    print('V3 saved:', out, summary['first_month'], flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-dir', type=Path, default=ROOT/'gpu_runs/2026-10-11-validation-2002-2023')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.benchmark_dir, args.batch_size, args.output)
