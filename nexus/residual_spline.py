"""Frozen six-configuration residual-spline experiment; CPU-only standalone runner.

Run with .venv/Scripts/python.exe -u -m nexus.residual_spline [--resume].
All artifacts are confined to runs/residual_v8. No other campaign is inspected.
"""
import os

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from .features import CATS, LABS, ID, TARGET, engineer, linear_preprocessor

OUT = Path('runs/residual_v8')
SEED = 20260914
CONTINUOUS = ['age', 'socioeconomic_index', 'length_of_stay_days'] + LABS
CONFIGS = [dict(name=f'residual_c{c}_a{a}', family='residual_spline', mode='raw',
                params=dict(C=c, multiplier=a))
           for c in (.01, .03, .1) for a in (0., .25)]
DEADLINE = '2026-09-13T18:45:00+05:30'


class ResidualBasis:
    """Training-only projection of spline coordinates away from linear effects."""
    def fit(self, X):
        self.blocks_ = {}
        for col in CONTINUOUS:
            values = X[col].to_numpy(dtype=float)
            observed = values[np.isfinite(values)]
            median = float(np.median(observed)) if len(observed) else 0.
            filled = np.where(np.isfinite(values), values, median)
            # Degenerate quantile knots cannot define a useful three-knot basis.
            if len(np.unique(np.quantile(filled, [0, .5, 1]))) < 3:
                self.blocks_[col] = None
                continue
            linear_scale = StandardScaler().fit(filled[:, None])
            z = linear_scale.transform(filled[:, None])[:, 0]
            design = np.column_stack([np.ones(len(z)), z, ~np.isfinite(values)])
            spline = SplineTransformer(n_knots=3, degree=3, knots='quantile',
                                       extrapolation='linear', include_bias=False)
            b = spline.fit_transform(filled[:, None])
            projection = np.linalg.lstsq(design, b, rcond=None)[0]
            residual = b - design @ projection
            keep = residual.std(axis=0) > 1e-10
            scaler = StandardScaler().fit(residual[:, keep]) if keep.any() else None
            self.blocks_[col] = (median, linear_scale, spline, projection, keep, scaler)
        return self

    def transform(self, X):
        result = []
        for col, block in self.blocks_.items():
            if block is None or block[-1] is None:
                continue
            median, linear_scale, spline, projection, keep, scaler = block
            values = X[col].to_numpy(dtype=float)
            filled = np.where(np.isfinite(values), values, median)
            z = linear_scale.transform(filled[:, None])[:, 0]
            design = np.column_stack([np.ones(len(z)), z, ~np.isfinite(values)])
            residual = spline.transform(filled[:, None]) - design @ projection
            result.append(scaler.transform(residual[:, keep]))
        return np.column_stack(result) if result else np.empty((len(X), 0))


class ResidualModel:
    def __init__(self, config):
        self.config = copy.deepcopy(config)

    def transform(self, X):
        x = engineer(X, 'raw').replace([np.inf, -np.inf], np.nan)
        linear = self.preprocessor.transform(x)
        if self.basis is None:
            return linear
        return np.column_stack([linear, self.multiplier * self.basis.transform(x)])

    def predict(self, X):
        with threadpool_limits(limits=2):
            return np.clip(self.model.predict_proba(self.transform(X))[:, 1], 1e-7, 1-1e-7)


def fit_residual(config, X, y, seed):
    """Fit from the supplied training rows only; compatible with model.predict(X)."""
    if config not in CONFIGS:
        raise ValueError('Configuration is outside the frozen six-config specification')
    result = ResidualModel(config)
    result.multiplier = config['params']['multiplier']
    x = engineer(X, 'raw').replace([np.inf, -np.inf], np.nan)
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        result.preprocessor = linear_preprocessor(x).fit(x)
        result.basis = ResidualBasis().fit(x) if result.multiplier else None
        result.model = LogisticRegression(C=config['params']['C'], solver='lbfgs',
                                          l1_ratio=0., max_iter=4000, random_state=seed)
        result.model.fit(result.transform(X), y)
    result.iterations = int(result.model.n_iter_.max())
    return result


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def save_model(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    joblib.dump(value, tmp)
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, p):
    return dict(log_loss=float(log_loss(y, p)), brier=float(brier_score_loss(y, p)),
                auc=float(roc_auc_score(y, p)))


def log(message, **state):
    line = f'[{datetime.now().astimezone().isoformat()}] {message}'
    print(line, flush=True)
    with (OUT / 'training.log').open('a', encoding='utf-8', buffering=1) as stream:
        stream.write(line + '\n')
    dump(OUT / 'status.json', dict(message=message, timestamp=datetime.now().astimezone().isoformat(),
                                  pid=os.getpid(), **state))


def check_deadline():
    if datetime.now().astimezone() >= datetime.fromisoformat(DEADLINE):
        raise TimeoutError('18:45 deadline reached')


def validate_data(train, test, sample):
    if len(train) != 7000 or len(test) != 3000 or len(sample) != 3000:
        raise ValueError('Expected 7000 train and 3000 test/sample rows')
    if any(not frame[ID].is_unique or frame[ID].isna().any() for frame in (train, test, sample)):
        raise ValueError('IDs must be unique and nonmissing')
    if set(test[ID]) != set(sample[ID]) or set(train[ID]) & set(test[ID]):
        raise ValueError('ID set mismatch or train/test overlap')
    if list(sample.columns) != [ID, TARGET] or TARGET in test:
        raise ValueError('Unexpected target/schema')
    if set(train[TARGET]) != {0, 1} or int(train[TARGET].sum()) != 881:
        raise ValueError('Unexpected target distribution')
    if set(train.columns) - {TARGET} != set(test.columns):
        raise ValueError('Train/test features differ')


def run(resume=False):
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ('cache', 'models'):
        (OUT / sub).mkdir(exist_ok=True)
    check_deadline()
    train, test, sample = [pd.read_csv(f) for f in ('train.csv', 'test.csv', 'sample_submission.csv')]
    validate_data(train, test, sample)
    manifest = dict(data={f: digest(f) for f in ('train.csv', 'test.csv', 'sample_submission.csv')},
                    code={f: digest(f) for f in ('nexus/residual_spline.py', 'nexus/features.py',
                                                 'tests/test_residual_spline.py')},
                    candidates=CONFIGS, seed=SEED, outer_folds=5, inner_folds=3, threads=2,
                    deadline=DEADLINE, python=sys.version, platform=platform.platform(),
                    versions={p: importlib.metadata.version(p) for p in
                              ('numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl')})
    manifest['version_hash'] = hashlib.sha256(json.dumps(manifest['versions'], sort_keys=True).encode()).hexdigest()
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        if not resume or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Existing run requires --resume and identical manifest')
    else:
        dump(manifest_path, manifest)
    dump(OUT / 'resolved_config.json', dict(candidates=CONFIGS, projection='intercept+linear+missing',
                                           knots=3, degree=3, multiplier_after_scaling=True))
    log('Frozen specification; CPU threads=2; nested 5x3; no other campaign read', phase='running')
    X, y = train.drop(columns=TARGET), train[TARGET]
    outer = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    folds = np.empty(len(y), dtype=int)
    for f, (_, ov) in enumerate(outer):
        folds[ov] = f
    pd.DataFrame({ID: train[ID], 'fold': folds}).to_csv(OUT / 'folds.csv', index=False)
    names = ('selected', 'linear_control')
    oof = {name: np.full(len(y), np.nan) for name in names}
    test_preds = {name: [] for name in names}
    results = []
    for f, (ot, ov) in enumerate(outer):
        tx, ty = X.iloc[ot], y.iloc[ot]
        inner = list(StratifiedKFold(3, shuffle=True, random_state=SEED + f + 1).split(tx, ty))
        assignments = np.empty(len(ot), dtype=int)
        for j, (_, iv) in enumerate(inner):
            assignments[iv] = j
        pd.DataFrame({ID: tx[ID].to_numpy(), 'row_index': ot, 'inner_fold': assignments}).to_csv(
            OUT / f'inner_folds_{f}.csv', index=False)
        scores, inner_frame = {}, pd.DataFrame({ID: tx[ID].to_numpy(), TARGET: ty.to_numpy()})
        for config in CONFIGS:
            ip = np.full(len(ot), np.nan)
            for j, (it, iv) in enumerate(inner):
                check_deadline()
                path = OUT / 'cache' / f'outer{f}_{config["name"]}_inner{j}.joblib'
                if path.exists():
                    checkpoint = joblib.load(path)
                    if not np.array_equal(checkpoint['validation_rows'], ot[iv]):
                        raise ValueError('Checkpoint rows do not match')
                else:
                    start = time.monotonic()
                    model = fit_residual(config, tx.iloc[it], ty.iloc[it], SEED + f * 100 + j)
                    checkpoint = dict(model=model, validation_rows=ot[iv], prediction=model.predict(tx.iloc[iv]),
                                      train_log_loss=float(log_loss(ty.iloc[it], model.predict(tx.iloc[it]))))
                    save_model(path, checkpoint)
                    log(f'outer={f+1}/5 {config["name"]} inner={j+1}/3 seconds={time.monotonic()-start:.2f}',
                        phase='running', outer_fold=f, candidate=config['name'], inner_fold=j)
                ip[iv] = checkpoint['prediction']
            scores[config['name']] = float(log_loss(ty, ip))
            inner_frame[config['name']] = ip
        inner_frame.to_csv(OUT / f'inner_oof_{f}.csv', index=False)
        selected = min(CONFIGS, key=lambda c: scores[c['name']])
        control = min((c for c in CONFIGS if c['params']['multiplier'] == 0), key=lambda c: scores[c['name']])
        fold_result = dict(fold=f, inner_scores=scores, chosen={}, metrics={})
        fitted = {}
        for name, config in zip(names, (selected, control)):
            check_deadline()
            if config['name'] not in fitted:
                fitted[config['name']] = fit_residual(config, tx, ty, SEED + f * 100)
            model = fitted[config['name']]
            save_model(OUT / 'models' / f'outer{f}_{name}.joblib', model)
            p = model.predict(X.iloc[ov])
            oof[name][ov] = p
            test_preds[name].append(model.predict(test))
            fold_result['chosen'][name] = config
            fold_result['metrics'][name] = metrics(y.iloc[ov], p)
        fold_result['delta_log_loss'] = fold_result['metrics']['selected']['log_loss'] - fold_result['metrics']['linear_control']['log_loss']
        results.append(fold_result)
        dump(OUT / f'fold_{f}_results.json', fold_result)
        frame = pd.DataFrame({ID: train[ID], TARGET: y, 'fold': folds, **oof})
        frame.to_csv(OUT / 'oof.csv', index=False)
        dump(OUT / 'summary.json', dict(complete=False, folds=results))
        log(f'Outer fold {f+1} complete', phase='running', completed_outer_folds=f+1)
    for name in names:
        p = np.mean(test_preds[name], axis=0)
        ordered = pd.Series(p, index=test[ID]).reindex(sample[ID]).to_numpy()
        if not np.isfinite(ordered).all() or not ((ordered > 0) & (ordered < 1)).all():
            raise ValueError('Invalid submission probabilities')
        submission = sample.copy()
        submission[TARGET] = ordered
        submission.to_csv(OUT / f'submission_{name}.csv', index=False)
        saved = pd.read_csv(OUT / f'submission_{name}.csv')
        if len(saved) != 3000 or not saved[ID].equals(sample[ID]) or list(saved) != list(sample):
            raise ValueError('Submission round-trip validation failed')
        pd.DataFrame(np.array(test_preds[name]).T, columns=[f'fold_{f}' for f in range(5)]).assign(
            **{ID: test[ID]}).to_csv(OUT / f'test_predictions_{name}.csv', index=False)
    all_metrics = {name: metrics(y, oof[name]) for name in names}
    losses = {}
    for name in names:
        p = oof[name]
        losses[name] = -(y.to_numpy() * np.log(p) + (1-y.to_numpy()) * np.log1p(-p))
    difference = losses['selected'] - losses['linear_control']
    rng = np.random.default_rng(SEED)
    bootstrap = [float(difference[rng.integers(0, len(y), len(y))].mean()) for _ in range(2000)]
    summary = dict(complete=True, seed=SEED, metrics=all_metrics, folds=results,
                   delta_log_loss=float(difference.mean()),
                   delta_ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
                   submission_checks=dict(rows=3000, sample_id_order=True, finite_probabilities=True),
                   evaluation_note='Inner-selected six-config policy vs inner-selected linear-only policy. '
                   'Bootstrap conditions on saved OOF predictions; reused development labels are not independent confirmation. '
                   'No smooth/reference comparison performed. Submissions average five outer models.')
    dump(OUT / 'summary.json', summary)
    pd.DataFrame([dict(method=name, **all_metrics[name]) for name in names]).to_csv(OUT / 'comparison.csv', index=False)
    dump(OUT / 'artifact_hashes.json', {str(p.relative_to(OUT)): digest(p) for p in OUT.rglob('*')
                                      if p.is_file() and p.name not in ('training.log', 'status.json', 'artifact_hashes.json')})
    log(f'COMPLETE metrics={all_metrics} delta={difference.mean():.9f}', phase='complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        try:
            run(args.resume)
        except Exception as exc:
            if OUT.exists():
                log(f'FAILED: {type(exc).__name__}: {exc}', phase='failed')
            raise
        return
    remaining = (datetime.fromisoformat(DEADLINE) - datetime.now().astimezone()).total_seconds()
    if remaining <= 0:
        raise SystemExit('18:45 deadline already passed; no training started')
    command = [sys.executable, '-B', '-u', '-m', 'nexus.residual_spline', '--worker']
    if args.resume:
        command.append('--resume')
    try:
        completed = subprocess.run(command, timeout=remaining)
        raise SystemExit(completed.returncode)
    except subprocess.TimeoutExpired:
        # subprocess.run kills and waits for its worker; numerical libraries use threads only.
        log('HARD DEADLINE reached; worker stopped; atomic checkpoints retained', phase='deadline')
        raise SystemExit(2)


if __name__ == '__main__':
    # Canonical import is essential: persisted classes must never live in __main__.
    from nexus.residual_spline import main as canonical_main
    canonical_main()
