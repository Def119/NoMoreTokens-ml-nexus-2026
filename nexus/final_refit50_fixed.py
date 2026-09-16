"""Frozen 50-fold refit using the completed final20 recipe.

This is deliberately not a new hyperparameter search.  It tests only whether
training each fixed component on 98% of the rows and averaging 50 fits helps
the current 20-fold champion.

Run from the repository root:
    .venv\\Scripts\\python.exe -u -m nexus.final_refit50_fixed
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from .features import ID, TARGET
from .models import fit_model

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = ROOT / 'runs' / 'final20_v11'
OUT = ROOT / 'runs' / 'final50_fixed_v12'
SEED = 20260914
FOLDS = 50
THREADS = 4
FAMILIES = ('lr', 'spline', 'lgb', 'cb', 'ebm')


def clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)


def metrics(y, p):
    p = clip(p)
    return {
        'log_loss': float(log_loss(y, p)),
        'brier': float(brier_score_loss(y, p)),
        'auc': float(roc_auc_score(y, p)),
    }


def load_recipe():
    spec = json.loads((SOURCE_RUN / 'resolved_config.json').read_text())
    configs = spec['candidates']
    if [c['family'] for c in configs] != list(FAMILIES):
        raise ValueError('final20 recipe family order changed')
    expected = ['lr_engineered_c0.1_l0.8', 'spline_k2_c0.01', 'lgb_raw_d1_r5',
                'cb_raw_d3_r20', 'ebm_bins16']
    if [c['name'] for c in configs] != expected:
        raise ValueError('final20 recipe names changed')
    results = []
    for path in sorted(SOURCE_RUN.glob('fold_*_results.json')):
        results.append(json.loads(path.read_text()))
    if len(results) != 20:
        raise ValueError('Expected 20 completed final20 fold results')
    iterations = {}
    for family in ('lgb', 'cb'):
        vals = [r['iterations'][family] for r in results]
        iterations[family] = int(round(float(np.median(vals))))
    return configs, iterations


def fit_fixed_calibrator():
    """Freeze one calibrator from the prior completed campaign."""
    oof = pd.read_csv(SOURCE_RUN / 'oof.csv')
    model = LogisticRegression(C=1.0, max_iter=2000)
    model.fit(logit(clip(oof['equal'])).reshape(-1, 1), oof[TARGET])
    return model


def run(resume=False):
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / 'train.csv')
    test = pd.read_csv(ROOT / 'test.csv')
    sample = pd.read_csv(ROOT / 'sample_submission.csv')
    if not train[ID].is_unique or not test[ID].equals(sample[ID]):
        raise ValueError('Invalid input IDs')

    configs, iterations = load_recipe()
    calibrator = fit_fixed_calibrator()
    manifest = {
        'source_run': str(SOURCE_RUN.relative_to(ROOT)),
        'outer_folds': FOLDS,
        'seed': SEED,
        'threads': THREADS,
        'training_fraction': 1 - 1 / FOLDS,
        'configs': configs,
        'frozen_iterations': iterations,
        'calibrator': {
            'C': 1.0,
            'coef': float(calibrator.coef_[0, 0]),
            'intercept': float(calibrator.intercept_[0]),
        },
        'note': 'Fixed-recipe 50-fold refit; no new hyperparameter selection.',
    }
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        if not resume or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Existing run differs; use --resume only with identical inputs')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    X = train.drop(columns=[TARGET])
    y = train[TARGET]
    splits = list(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(X, y))
    fold_ids = np.full(len(train), -1, dtype=int)
    for f, (_, va) in enumerate(splits):
        fold_ids[va] = f

    oof = np.full(len(train), np.nan)
    test_parts = []
    results = []
    start = time.monotonic()
    backend = 'GPU'
    for f, (tr_idx, va_idx) in enumerate(splits):
        cache_path = OUT / f'fold_{f:02d}.npz'
        if resume and cache_path.exists():
            z = np.load(cache_path)
            if not np.array_equal(z['validation_indices'], va_idx):
                raise ValueError(f'Fold {f} indices changed')
            oof[va_idx] = z['oof']
            test_parts.append(z['test'])
            results.append(json.loads((OUT / f'fold_{f:02d}.json').read_text()))
            print(f'RESUME fold {f + 1}/{FOLDS}', flush=True)
            continue

        xtr, ytr = X.iloc[tr_idx].reset_index(drop=True), y.iloc[tr_idx].reset_index(drop=True)
        val_parts, test_family_parts = [], []
        for config in configs:
            family = config['family']
            print(f'FIT fold {f + 1}/{FOLDS} {config["name"]} train={len(tr_idx)}', flush=True)
            with threadpool_limits(limits=THREADS):
                model = fit_model(
                    config, xtr, ytr,
                    seed=SEED + f,
                    threads=THREADS,
                    backend=backend,
                    iterations=iterations.get(family),
                )
                val_parts.append(model.predict(X.iloc[va_idx]))
                test_family_parts.append(model.predict(test))

        raw_val = clip(np.mean(val_parts, axis=0))
        raw_test = clip(np.mean(test_family_parts, axis=0))
        cal_val = clip(calibrator.predict_proba(logit(raw_val).reshape(-1, 1))[:, 1])
        cal_test = clip(calibrator.predict_proba(logit(raw_test).reshape(-1, 1))[:, 1])
        oof[va_idx] = cal_val
        test_parts.append(cal_test)
        result = dict(
            fold=f,
            train_n=len(tr_idx),
            validation_n=len(va_idx),
            metrics=metrics(y.iloc[va_idx], cal_val),
        )
        results.append(result)
        np.savez_compressed(cache_path, validation_indices=va_idx, oof=cal_val, test=cal_test)
        (OUT / f'fold_{f:02d}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        elapsed = (time.monotonic() - start) / 60
        print(f'FOLD {f + 1}/{FOLDS} ll={result["metrics"]["log_loss"]:.7f} elapsed={elapsed:.1f}m', flush=True)

    if not np.isfinite(oof).all():
        raise RuntimeError('Incomplete 50-fold run')
    test_pred = clip(np.mean(test_parts, axis=0))
    oof_frame = pd.DataFrame({ID: train[ID], TARGET: y, 'fold': fold_ids, 'equal_cal': oof})
    oof_frame.to_csv(OUT / 'oof.csv', index=False)
    pd.DataFrame({ID: sample[ID], TARGET: test_pred}).to_csv(OUT / 'submission_equal_cal.csv', index=False)

    # A pre-specified hedge against the known v6 local/LB mismatch.
    v6 = pd.read_csv(ROOT / 'submission_v6.csv')
    if not v6[ID].equals(sample[ID]):
        raise ValueError('v6 submission IDs do not match sample order')
    hedge = clip(0.75 * test_pred + 0.25 * v6[TARGET].to_numpy())
    pd.DataFrame({ID: sample[ID], TARGET: hedge}).to_csv(OUT / 'submission_50_plus_v6_25.csv', index=False)

    summary = {
        'complete': True,
        'metrics': metrics(y, oof),
        'folds': results,
        'outer_folds': FOLDS,
        'training_fraction': 1 - 1 / FOLDS,
        'elapsed_seconds': time.monotonic() - start,
        'submission_policy': 'Frozen 50-fold calibrated recipe plus a pre-specified 25% v6 hedge.',
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.resume)
