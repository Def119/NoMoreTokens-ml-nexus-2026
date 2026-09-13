"""Non-v6 20-fold seed-bag experiment.

Uses the frozen final20 recipe and fold design. Only LightGBM and CatBoost
receive three fixed seeds per outer fold; their probabilities are averaged.
No target encoding, feature pruning, v6 predictions, or new tuning is used.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from .features import ID, TARGET
from .models import fit_model

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs' / 'final20_v11'
OUT = ROOT / 'runs' / 'final20_seedbag_v14'
SEED = 20260914
FOLDS = 20
SOURCE_FOLDS = 20
THREADS = 4
FAMILIES = ('lr', 'spline', 'lgb', 'cb', 'ebm')
TREE_SEEDS = (0, 1, 2)


def clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)


def metrics(y, p):
    p = clip(p)
    return {
        'log_loss': float(log_loss(y, p)),
        'brier': float(brier_score_loss(y, p)),
        'auc': float(roc_auc_score(y, p)),
    }


def recipe_and_iterations():
    spec = json.loads((SOURCE / 'resolved_config.json').read_text())
    configs = spec['candidates']
    if [c['family'] for c in configs] != list(FAMILIES):
        raise ValueError('Unexpected final20 family order')
    expected = ['lr_engineered_c0.1_l0.8', 'spline_k2_c0.01', 'lgb_raw_d1_r5',
                'cb_raw_d3_r20', 'ebm_bins16']
    if [c['name'] for c in configs] != expected:
        raise ValueError('Unexpected final20 configurations')
    results = [json.loads(p.read_text()) for p in SOURCE.glob('fold_*_results.json')]
    if len(results) != SOURCE_FOLDS:
        raise ValueError('final20 run is incomplete')
    iterations = {}
    for family in ('lgb', 'cb'):
        iterations[family] = int(round(np.median([r['iterations'][family] for r in results])))
    return configs, iterations


def fixed_calibrator():
    oof = pd.read_csv(SOURCE / 'oof.csv')
    model = LogisticRegression(C=1.0, max_iter=2000)
    model.fit(logit(clip(oof['equal'])).reshape(-1, 1), oof[TARGET])
    return model


def run(resume=False):
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / 'train.csv')
    test = pd.read_csv(ROOT / 'test.csv')
    sample = pd.read_csv(ROOT / 'sample_submission.csv')
    if not train[ID].is_unique or not test[ID].equals(sample[ID]):
        raise ValueError('Invalid input ID alignment')
    configs, iterations = recipe_and_iterations()
    calibrator = fixed_calibrator()
    manifest = {
        'source': str(SOURCE.relative_to(ROOT)),
        'outer_folds': FOLDS,
        'seed': SEED,
        'tree_seeds': list(TREE_SEEDS),
        'configs': configs,
        'frozen_iterations': iterations,
        'calibrator': {'C': 1.0, 'coef': float(calibrator.coef_[0, 0]),
                       'intercept': float(calibrator.intercept_[0])},
        'note': 'Non-v6 seed bag; only target-independent tree seed averaging added.',
    }
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        if not resume or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Existing run differs; use --resume with identical inputs')
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
    for f, (tr_idx, va_idx) in enumerate(splits):
        cache = OUT / f'fold_{f:02d}.npz'
        result_path = OUT / f'fold_{f:02d}.json'
        if resume and cache.exists() and result_path.exists():
            z = np.load(cache)
            if not np.array_equal(z['validation_indices'], va_idx):
                raise ValueError(f'Fold {f} indices changed')
            oof[va_idx] = z['oof']
            test_parts.append(z['test'])
            results.append(json.loads(result_path.read_text()))
            print(f'RESUME fold {f + 1}/{FOLDS}', flush=True)
            continue

        xtr = X.iloc[tr_idx].reset_index(drop=True)
        ytr = y.iloc[tr_idx].reset_index(drop=True)
        family_val, family_test = [], []
        for config in configs:
            family = config['family']
            seeds = TREE_SEEDS if family in ('lgb', 'cb') else (0,)
            vals, tests = [], []
            for s in seeds:
                print(f'FIT fold {f + 1}/{FOLDS} {config["name"]} seed={s} train={len(tr_idx)}', flush=True)
                with threadpool_limits(limits=THREADS):
                    model = fit_model(
                        config, xtr, ytr,
                        seed=SEED + f * 100 + s,
                        threads=THREADS,
                        backend='GPU',
                        iterations=iterations.get(family),
                    )
                    vals.append(model.predict(X.iloc[va_idx]))
                    tests.append(model.predict(test))
            family_val.append(np.mean(vals, axis=0))
            family_test.append(np.mean(tests, axis=0))

        raw_val = clip(np.mean(family_val, axis=0))
        raw_test = clip(np.mean(family_test, axis=0))
        cal_val = clip(calibrator.predict_proba(logit(raw_val).reshape(-1, 1))[:, 1])
        cal_test = clip(calibrator.predict_proba(logit(raw_test).reshape(-1, 1))[:, 1])
        oof[va_idx] = cal_val
        test_parts.append(cal_test)
        result = {'fold': f, 'train_n': len(tr_idx), 'validation_n': len(va_idx),
                  'metrics': metrics(y.iloc[va_idx], cal_val)}
        results.append(result)
        np.savez_compressed(cache, validation_indices=va_idx, oof=cal_val, test=cal_test)
        result_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(f'FOLD {f + 1}/{FOLDS} ll={result["metrics"]["log_loss"]:.7f}', flush=True)

    if not np.isfinite(oof).all():
        raise RuntimeError('Incomplete seed-bag OOF predictions')
    test_pred = clip(np.mean(test_parts, axis=0))
    pd.DataFrame({ID: train[ID], TARGET: y, 'fold': fold_ids, 'equal_cal': oof}).to_csv(OUT / 'oof.csv', index=False)
    pd.DataFrame({ID: sample[ID], TARGET: test_pred}).to_csv(OUT / 'submission_seedbag.csv', index=False)
    summary = {'complete': True, 'metrics': metrics(y, oof), 'folds': results,
               'outer_folds': FOLDS, 'tree_seeds': list(TREE_SEEDS),
               'elapsed_seconds': time.monotonic() - start,
               'submission_policy': 'Non-v6 20-fold seed-bag; no v6 inputs.'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.resume)
