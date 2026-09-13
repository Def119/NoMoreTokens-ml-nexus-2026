"""Nested, non-v6 engineered LightGBM/CatBoost branch for final20.

The final20 recipe remains the base prediction.  This script trains only an
engineered-feature tree branch within the same 20 outer folds and evaluates a
pre-specified 80% base + 20% branch probability blend.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import joblib
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
BASE = ROOT / 'runs' / 'final20_v11'
OUT = ROOT / 'runs' / 'engineered_tree_v16'
SEED = 20260914
OUTER_FOLDS = 20
INNER_FOLDS = 3
THREADS = 4
BASE_WEIGHT = 0.80
BRANCH_WEIGHT = 0.20


def clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)


def metrics(y, p):
    p = clip(p)
    return {
        'log_loss': float(log_loss(y, p)),
        'brier': float(brier_score_loss(y, p)),
        'auc': float(roc_auc_score(y, p)),
    }


def configs():
    spec = json.loads((BASE / 'resolved_config.json').read_text())
    selected = [c for c in spec['candidates'] if c['family'] in ('lgb', 'cb')]
    if [c['family'] for c in selected] != ['lgb', 'cb']:
        raise ValueError('Expected exactly one LGBM and CatBoost final20 configuration')
    result = []
    for source in selected:
        cfg = copy.deepcopy(source)
        cfg['name'] = cfg['name'].replace('_raw_', '_engineered_')
        cfg['mode'] = 'engineered'
        result.append(cfg)
    return result


def base_fold(fold, indices):
    saved = joblib.load(BASE / 'cache' / f'outer_{fold}.joblib')
    if not np.array_equal(saved['indices'], indices):
        raise ValueError(f'Final20 fold {fold} indices do not match')
    return clip(saved['outer']['equal_cal']), clip(saved['test']['equal_cal'])


def run(resume=False):
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / 'train.csv')
    test = pd.read_csv(ROOT / 'test.csv')
    sample = pd.read_csv(ROOT / 'sample_submission.csv')
    if not train[ID].is_unique or not test[ID].equals(sample[ID]):
        raise ValueError('Invalid input ID alignment')
    tree_configs = configs()
    manifest = {
        'base_run': str(BASE.relative_to(ROOT)),
        'outer_folds': OUTER_FOLDS,
        'inner_folds': INNER_FOLDS,
        'seed': SEED,
        'configs': tree_configs,
        'base_weight': BASE_WEIGHT,
        'branch_weight': BRANCH_WEIGHT,
        'note': 'No v6 input. Engineered tree branch; calibration is fit only on inner OOF per outer fold.',
    }
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        if not resume or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Existing run differs; use --resume only with identical inputs')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    X = train.drop(columns=[TARGET])
    y = train[TARGET]
    splits = list(StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    fold_ids = np.full(len(train), -1, dtype=int)
    for fold, (_, va) in enumerate(splits):
        fold_ids[va] = fold
    branch_oof = np.full(len(train), np.nan)
    blend_oof = np.full(len(train), np.nan)
    branch_test_parts, blend_test_parts, results = [], [], []
    start = time.monotonic()

    for fold, (tr_idx, va_idx) in enumerate(splits):
        cache = OUT / f'fold_{fold:02d}.joblib'
        if resume and cache.exists():
            saved = joblib.load(cache)
            if not np.array_equal(saved['indices'], va_idx):
                raise ValueError(f'Fold {fold} indices changed')
            branch_oof[va_idx] = saved['branch_outer']
            blend_oof[va_idx] = saved['blend_outer']
            branch_test_parts.append(saved['branch_test'])
            blend_test_parts.append(saved['blend_test'])
            results.append(saved['result'])
            print(f'RESUME fold {fold + 1}/{OUTER_FOLDS}', flush=True)
            continue

        tx = X.iloc[tr_idx].reset_index(drop=True)
        ty = y.iloc[tr_idx].reset_index(drop=True)
        inner = list(StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED + fold + 1).split(tx, ty))
        inner_predictions, selected_iterations = [], {}
        for config in tree_configs:
            family = config['family']
            ip = np.full(len(tx), np.nan)
            iters = []
            for inner_fold, (it, iv) in enumerate(inner):
                print(f'INNER fold {fold + 1}/{OUTER_FOLDS} {config["name"]} {inner_fold + 1}/{INNER_FOLDS}', flush=True)
                with threadpool_limits(limits=THREADS):
                    model = fit_model(
                        config, tx.iloc[it], ty.iloc[it],
                        seed=SEED + fold * 100 + inner_fold,
                        threads=THREADS,
                        backend='GPU',
                        validation=(tx.iloc[iv], ty.iloc[iv]),
                    )
                ip[iv] = model.predict(tx.iloc[iv])
                iters.append(model.iterations)
            if not np.isfinite(ip).all() or any(v is None for v in iters):
                raise RuntimeError(f'Incomplete inner predictions for {config["name"]}')
            inner_predictions.append(clip(ip))
            selected_iterations[family] = int(round(np.median(iters)))

        inner_branch = clip(np.mean(inner_predictions, axis=0))
        calibrator = LogisticRegression(C=1.0, max_iter=2000).fit(logit(inner_branch).reshape(-1, 1), ty)
        outer_parts, test_parts = [], []
        for config in tree_configs:
            family = config['family']
            print(f'REFIT fold {fold + 1}/{OUTER_FOLDS} {config["name"]} iterations={selected_iterations[family]}', flush=True)
            with threadpool_limits(limits=THREADS):
                model = fit_model(
                    config, tx, ty,
                    seed=SEED + fold * 100,
                    threads=THREADS,
                    backend='GPU',
                    iterations=selected_iterations[family],
                )
            outer_parts.append(model.predict(X.iloc[va_idx]))
            test_parts.append(model.predict(test))
        raw_outer = clip(np.mean(outer_parts, axis=0))
        raw_test = clip(np.mean(test_parts, axis=0))
        branch_outer = clip(calibrator.predict_proba(logit(raw_outer).reshape(-1, 1))[:, 1])
        branch_test = clip(calibrator.predict_proba(logit(raw_test).reshape(-1, 1))[:, 1])
        base_outer, base_test = base_fold(fold, va_idx)
        blend_outer = clip(BASE_WEIGHT * base_outer + BRANCH_WEIGHT * branch_outer)
        blend_test = clip(BASE_WEIGHT * base_test + BRANCH_WEIGHT * branch_test)
        result = {
            'fold': fold,
            'train_n': len(tr_idx),
            'validation_n': len(va_idx),
            'iterations': selected_iterations,
            'branch_metrics': metrics(y.iloc[va_idx], branch_outer),
            'blend_metrics': metrics(y.iloc[va_idx], blend_outer),
        }
        saved = {
            'indices': va_idx,
            'branch_outer': branch_outer,
            'blend_outer': blend_outer,
            'branch_test': branch_test,
            'blend_test': blend_test,
            'result': result,
        }
        joblib.dump(saved, cache)
        (OUT / f'fold_{fold:02d}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        branch_oof[va_idx] = branch_outer
        blend_oof[va_idx] = blend_outer
        branch_test_parts.append(branch_test)
        blend_test_parts.append(blend_test)
        results.append(result)
        print(f'OUTER {fold + 1}/{OUTER_FOLDS} branch={result["branch_metrics"]["log_loss"]:.7f} blend={result["blend_metrics"]["log_loss"]:.7f}', flush=True)

    if not np.isfinite(branch_oof).all() or not np.isfinite(blend_oof).all():
        raise RuntimeError('Incomplete OOF predictions')
    branch_test = clip(np.mean(branch_test_parts, axis=0))
    blend_test = clip(np.mean(blend_test_parts, axis=0))
    pd.DataFrame({ID: train[ID], TARGET: y, 'fold': fold_ids,
                  'engineered_tree': branch_oof, 'blend_80_20': blend_oof}).to_csv(OUT / 'oof.csv', index=False)
    pd.DataFrame({ID: sample[ID], TARGET: branch_test}).to_csv(OUT / 'submission_engineered_tree.csv', index=False)
    pd.DataFrame({ID: sample[ID], TARGET: blend_test}).to_csv(OUT / 'submission_blend_80_20.csv', index=False)
    summary = {
        'complete': True,
        'metrics': {'engineered_tree': metrics(y, branch_oof), 'blend_80_20': metrics(y, blend_oof)},
        'folds': results,
        'outer_folds': OUTER_FOLDS,
        'inner_folds': INNER_FOLDS,
        'base_weight': BASE_WEIGHT,
        'branch_weight': BRANCH_WEIGHT,
        'elapsed_seconds': time.monotonic() - start,
        'submission_policy': 'Pre-specified 80% final20 base + 20% non-v6 engineered tree branch.',
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.resume)
