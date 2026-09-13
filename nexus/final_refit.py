"""Frozen five-family, 10-fold campaign. Run: python -B -u -m nexus.final_refit.

Only runs/final10_v9 is written. --resume requires identical provenance.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from .features import ID, TARGET
from .models import fit_model
from .ensemble import Combiner
from .train import dump, digest, metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/final10_v9'
BASE = ROOT / 'runs/smooth_v8'
SEED = 20260914
FAMILIES = ('lr', 'spline', 'lgb', 'cb', 'ebm')
NAMES = ('lr_engineered_c0.1_l0.8', 'spline_k2_c0.01', 'lgb_raw_d1_r5',
         'cb_raw_d3_r20', 'ebm_bins16')
METHODS = ('equal', 'equal_cal')
NOTE = ('Adaptive reuse: fixed configurations and the 10-fold design were chosen after '
        'examining smooth_v8 inner selections and learning curves on these same labels. '
        'Outer labels in this campaign are scoring-only, but this is not pristine independent '
        'validation. Paired patient-bootstrap intervals condition on fitted OOF predictions; '
        'they exclude training, tuning, adaptive-selection and overlapping-fold uncertainty. '
        'Averaging models trained on 90% rather than 80% may help test performance, '
        'but a deployment advantage is not established by this comparison.')


def fixed_configs(spec):
    configs = []
    for family, name in zip(FAMILIES, NAMES):
        matches = [c for c in spec['candidates'] if c['name'] == name]
        if len(matches) != 1 or matches[0]['family'] != family:
            raise ValueError(f'Missing, duplicate or wrong-family fixed config: {name}')
        configs.append(copy.deepcopy(matches[0]))
    return configs


def outer_splits(y):
    return list(StratifiedKFold(10, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def probabilities(p, n):
    p = np.asarray(p, dtype=float)
    if p.shape != (n,) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid probability shape, finiteness or bounds')
    return p


def aligned_baseline(train, baseline):
    if not train[ID].is_unique or not baseline[ID].is_unique:
        raise ValueError('Duplicate OOF IDs')
    if not train[ID].equals(baseline[ID]) or not train[TARGET].equals(baseline[TARGET]):
        raise ValueError('Baseline OOF ID/label order differs from training data')
    return probabilities(baseline['equal_cal'], len(train))


def submission(test, sample, p):
    if sample.columns.tolist() != [ID, TARGET] or not test[ID].is_unique or not sample[ID].is_unique:
        raise ValueError('Invalid submission schema or duplicate IDs')
    if not test[ID].equals(sample[ID]):
        raise ValueError('Sample/test ID order mismatch')
    return pd.DataFrame({ID: test[ID], TARGET: probabilities(p, len(test))})


def atomic_joblib(value, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    joblib.dump(value, tmp)
    tmp.replace(path)


def atomic_csv(frame, path):
    tmp = path.with_suffix('.csv.tmp')
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def row_loss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(np.asarray(y) * np.log(p) + (1 - np.asarray(y)) * np.log1p(-p))


def paired_delta(y, p, reference):
    delta = row_loss(y, p) - row_loss(y, reference)
    rng = np.random.default_rng(SEED)
    means = [float(delta[rng.integers(len(delta), size=len(delta))].mean()) for _ in range(5000)]
    return dict(delta_log_loss=float(delta.mean()), ci95=np.quantile(means, [.025, .975]).tolist(),
                repeats=5000, seed=SEED, interpretation=NOTE)


class Progress:
    def __init__(self, deadline):
        self.start = time.monotonic()
        self.deadline = datetime.fromisoformat(deadline)
        if self.deadline.tzinfo is None:
            raise ValueError('Deadline must include a timezone')
        self.done = 0
        self.times = {}
        self.state = dict(fold=0, trial=0, inner_fold=None)

    def log(self, message, **state):
        self.state.update(state)
        elapsed = time.monotonic() - self.start
        avg = np.mean([v for values in self.times.values() for v in values] or [20.0])
        eta = max(0, 200 - self.done) * float(avg)
        now = datetime.now().astimezone()
        line = f'[{now:%H:%M:%S}] elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m {message}'
        print(line, flush=True)
        with (OUT / 'training.log').open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')
        dump(OUT / 'status.json', dict(timestamp=now.isoformat(), elapsed_seconds=elapsed,
             eta_seconds=eta, deadline=self.deadline.isoformat(), completed_fits=self.done,
             total_fits=200, message=message, **self.state))

    def check(self, family):
        estimate = max(self.times.get(family, [60.0]))
        if (self.deadline - datetime.now().astimezone()).total_seconds() < estimate + 120:
            self.log('Deadline cutoff; all finished fits retained. Resume with a new deadline.', phase='paused')
            return False
        return True

    def completed(self, family, seconds):
        self.done += 1
        self.times.setdefault(family, []).append(seconds)


def run(resume=False, deadline='2026-09-13T18:45:00+05:30'):
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ('cache', 'models'):
        (OUT / name).mkdir(exist_ok=True)
    # Exclusive OS-held lock; released even after a crash, with no stale-PID bypass.
    import msvcrt
    with (OUT / 'campaign.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        return _run(resume, deadline)


def _run(resume, deadline):
    progress = Progress(deadline)
    tr = pd.read_csv(ROOT / 'train.csv')
    te = pd.read_csv(ROOT / 'test.csv')
    sample = pd.read_csv(ROOT / 'sample_submission.csv')
    if not tr[ID].is_unique or set(tr[TARGET]) != {0, 1} or TARGET in te:
        raise ValueError('Invalid training/test schema')
    submission(te, sample, np.full(len(te), .5))
    base_summary = json.loads((BASE / 'summary.json').read_text())
    if not base_summary.get('complete') or base_summary.get('smoke'):
        raise ValueError('Baseline must be complete and non-smoke')
    reference = aligned_baseline(tr, pd.read_csv(BASE / 'oof.csv'))
    configs = fixed_configs(json.loads((BASE / 'resolved_config.json').read_text()))
    frozen = dict(candidates=configs, methods=list(METHODS), equal_cal_C=1.0)
    config_hash = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    manifest = dict(data={p: digest(ROOT / p) for p in ['train.csv', 'test.csv', 'sample_submission.csv']},
        source={p: digest(ROOT / p) for p in ['nexus/final_refit.py', 'nexus/features.py',
                'nexus/models.py', 'nexus/ensemble.py', 'nexus/train.py']},
        base={p: digest(BASE / p) for p in ['resolved_config.json', 'oof.csv', 'summary.json']},
        fixed_config_sha256=config_hash, fixed_config=frozen, seed=SEED, outer_folds=10,
        inner_folds=3, threads=4, backend='GPU', python=sys.version,
        packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn',
                  'scipy', 'lightgbm', 'catboost', 'interpret-core', 'joblib', 'threadpoolctl']},
        evaluation_note=NOTE, nondeterminism='CatBoost GPU floating-point operations may be nondeterministic.')
    path = OUT / 'manifest.json'
    if path.exists():
        if not resume or json.loads(path.read_text()) != manifest:
            raise ValueError('Existing campaign requires --resume and identical manifest')
    else:
        dump(path, manifest)
    dump(OUT / 'resolved_config.json', frozen)
    X, y = tr.drop(columns=TARGET), tr[TARGET]
    splits = outer_splits(y)
    fold_ids = np.full(len(tr), -1, dtype=int)
    for f, (_, ov) in enumerate(splits):
        fold_ids[ov] = f
    atomic_csv(pd.DataFrame({ID: tr[ID], 'fold': fold_ids}), OUT / 'folds.csv')
    predictions = {m: np.full(len(tr), np.nan) for m in METHODS}
    tests = {m: [] for m in METHODS}
    results = []
    progress.log('Frozen 10-fold campaign: 150 inner fits + 50 outer refits; GPU CatBoost; threads=4', phase='starting')
    for f, (ot, ov) in enumerate(splits):
        progress.state.update(fold=f + 1)
        finish = OUT / 'cache' / f'outer_{f}.joblib'
        model_path = OUT / 'models' / f'fold_{f}.joblib'
        if finish.exists():
            saved = joblib.load(finish)
            if digest(model_path) != saved['model_sha256']:
                raise ValueError('Completed fold model hash mismatch')
            np.testing.assert_array_equal(saved['indices'], ov)
            progress.done += 20
        else:
            tx, ty = X.iloc[ot].reset_index(drop=True), y.iloc[ot].reset_index(drop=True)
            inner = list(StratifiedKFold(3, shuffle=True, random_state=SEED + f + 1).split(tx, ty))
            inner_ids = np.full(len(ot), -1, dtype=int)
            for j, (_, iv) in enumerate(inner):
                inner_ids[iv] = j
            atomic_csv(pd.DataFrame({ID: tx[ID], 'inner_fold': inner_ids}), OUT / 'cache' / f'inner_folds_{f}.csv')
            inner_matrix, val_matrix, test_matrix = [], [], []
            fitted, iterations, inner_scores = {}, {}, {}
            for ci, config in enumerate(configs):
                family = config['family']
                ip, its = np.full(len(tx), np.nan), []
                for j, (it, iv) in enumerate(inner):
                    cache = OUT / 'cache' / f'o{f}_{family}_i{j}.joblib'
                    progress.state.update(trial=ci + 1, inner_fold=j + 1)
                    if cache.exists():
                        z = joblib.load(cache)
                        np.testing.assert_array_equal(z['indices'], iv)
                    else:
                        if not progress.check(family):
                            return False
                        progress.log(f'FIT outer={f+1}/10 {config["name"]} inner={j+1}/3', phase='inner')
                        with threadpool_limits(limits=4):
                            model = fit_model(config, tx.iloc[it], ty.iloc[it], seed=SEED + f * 100 + j,
                                threads=4, backend='GPU', validation=(tx.iloc[iv], ty.iloc[iv])
                                if family in ('lgb', 'cb') else None)
                            p = model.predict(tx.iloc[iv])
                        z = dict(indices=iv, prediction=probabilities(p, len(iv)), iterations=model.iterations,
                                 seconds=model.seconds)
                        atomic_joblib(z, cache)
                    ip[iv] = z['prediction']
                    if z['iterations'] is not None:
                        its.append(z['iterations'])
                    progress.completed(family, z['seconds'])
                inner_matrix.append(probabilities(ip, len(tx)))
                inner_scores[family] = metrics(ty, ip)
                iterations[family] = int(np.median(its)) if family in ('lgb', 'cb') else None
                cache = OUT / 'cache' / f'o{f}_{family}_refit.joblib'
                progress.state.update(inner_fold=None)
                if cache.exists():
                    z = joblib.load(cache)
                    np.testing.assert_array_equal(z['indices'], ov)
                else:
                    if not progress.check(family):
                        return False
                    progress.log(f'REFIT outer={f+1}/10 {config["name"]} n={len(tx)} iterations={iterations[family]}', phase='refit')
                    with threadpool_limits(limits=4):
                        model = fit_model(config, tx, ty, seed=SEED + f * 100, threads=4,
                                          backend='GPU', iterations=iterations[family])
                        vp, tp = model.predict(X.iloc[ov]), model.predict(te)
                    z = dict(model=model, indices=ov, outer=probabilities(vp, len(ov)),
                             test=probabilities(tp, len(te)), seconds=model.seconds)
                    atomic_joblib(z, cache)
                progress.completed(family, z['seconds'])
                fitted[config['name']] = z['model']
                val_matrix.append(z['outer'])
                test_matrix.append(z['test'])
            a, b, t = map(np.column_stack, (inner_matrix, val_matrix, test_matrix))
            combiners = {m: Combiner('equal', m == 'equal_cal').fit(a, ty) for m in METHODS}
            pp = {m: probabilities(combiners[m].predict(b), len(ov)) for m in METHODS}
            tp = {m: probabilities(combiners[m].predict(t), len(te)) for m in METHODS}
            result = dict(fold=f, train_n=len(ot), validation_n=len(ov), iterations=iterations,
                inner_metrics=inner_scores, outer_metrics={m: metrics(y.iloc[ov], pp[m]) for m in METHODS})
            atomic_joblib(dict(models=fitted, chosen=dict(zip(FAMILIES, configs)), combiners=combiners,
                outer_train=ot, outer_validation=ov, family_order=FAMILIES), model_path)
            saved = dict(indices=ov, outer=pp, test=tp, result=result, model_sha256=digest(model_path))
            atomic_joblib(saved, finish)  # Commit marker last, after the model bundle.
        for m in METHODS:
            predictions[m][ov] = saved['outer'][m]
            tests[m].append(saved['test'][m])
        results.append(saved['result'])
        dump(OUT / f'fold_{f}_results.json', saved['result'])
        progress.log(f'OUTER {f+1}/10 {saved["result"]["outer_metrics"]}', phase='outer_complete')
    for m in METHODS:
        probabilities(predictions[m], len(tr))
        frame = submission(te, sample, np.mean(tests[m], axis=0))
        atomic_csv(frame, OUT / f'submission_{m}.csv')
        # Validate the serialized artifact as delivered.
        delivered = pd.read_csv(OUT / f'submission_{m}.csv')
        if not delivered[ID].equals(sample[ID]):
            raise ValueError('Serialized submission IDs changed')
        probabilities(delivered[TARGET], len(sample))
    atomic_csv(pd.DataFrame({ID: tr[ID], TARGET: y, 'fold': fold_ids, **predictions}), OUT / 'oof.csv')
    variation = {m: {metric: dict(mean=float(np.mean(vals)), std=float(np.std(vals, ddof=1)),
                 min=float(np.min(vals)), max=float(np.max(vals)))
                 for metric in ('log_loss', 'brier', 'auc')
                 for vals in [[r['outer_metrics'][m][metric] for r in results]]} for m in METHODS}
    summary = dict(complete=True, metrics={m: metrics(y, predictions[m]) for m in METHODS},
        folds=results, fold_variation=variation, baseline='runs/smooth_v8/oof.csv:equal_cal',
        baseline_metrics=metrics(y, reference), paired_bootstrap={m: paired_delta(y, predictions[m], reference) for m in METHODS},
        evaluation_note=NOTE, outer_folds=10, inner_folds=3, seed=SEED,
        submission_policy='Both prespecified recipes delivered; no outer-score winner selection.',
        model_training_fraction=.9, elapsed_seconds=time.monotonic() - progress.start)
    dump(OUT / 'summary.json', summary)
    progress.log(f'COMPLETE {summary["metrics"]}', phase='complete')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--deadline', default='2026-09-13T18:45:00+05:30')
    args = parser.parse_args()
    try:
        complete = run(args.resume, args.deadline)
    except Exception:
        import traceback
        OUT.mkdir(parents=True, exist_ok=True)
        dump(OUT / 'error.json', dict(traceback=traceback.format_exc(), timestamp=datetime.now().astimezone().isoformat()))
        raise
    sys.exit(0 if complete else 2)
