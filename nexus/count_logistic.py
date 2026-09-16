"""Frozen count one-hot deviations alongside raw logistic; standalone CPU runner.

python -B -u -m nexus.count_logistic [--resume]
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from .features import ID, TARGET, engineer, linear_preprocessor

OUT = Path('runs/count_v10')
SEED = 20260915
DEADLINE = '2026-09-13T18:45:00+05:30'
COUNTS = ('prior_admissions_12m', 'comorbidity_count', 'missed_appointments_12m')
CONFIGS = [dict(name=f'count_{basis}_c{c}', basis=basis, C=c)
           for basis in ('A', 'B') for c in (.01, .03, .1)]
CONTROLS = [dict(name=f'linear_c{c}', basis='linear', C=c) for c in (.01, .03, .1)]


class CountDesign(TransformerMixin, BaseEstimator):
    """No target encoding. Unscaled one-hot deviations retain the raw linear design.

The added basis is deliberately not residualized: ridge on the redundant linear
and categorical count columns permits penalized departures from a linear trend.
"""
    def __init__(self, basis='A'):
        self.basis = basis

    def fit(self, X, y=None):
        if self.basis not in ('A', 'B', 'linear'):
            raise ValueError('Unknown count basis')
        x = engineer(X, 'raw').replace([np.inf, -np.inf], np.nan)
        self.raw_ = linear_preprocessor(x).fit(x)
        self.count_columns_ = list(COUNTS[:1] if self.basis == 'A' else COUNTS if self.basis == 'B' else ())
        if self.count_columns_:
            self.imputer_ = SimpleImputer(strategy='median', keep_empty_features=True).fit(x[self.count_columns_])
            self.encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(
                self.imputer_.transform(x[self.count_columns_]))
        return self

    def transform(self, X):
        check_is_fitted(self, 'raw_')
        x = engineer(X, 'raw').replace([np.inf, -np.inf], np.nan)
        raw = self.raw_.transform(x)
        if not self.count_columns_:
            return raw
        deviations = self.encoder_.transform(self.imputer_.transform(x[self.count_columns_]))
        return np.column_stack([raw, deviations])


class CountModel:
    def __init__(self, config):
        self.config = copy.deepcopy(config)

    def predict(self, X):
        with threadpool_limits(limits=2):
            return np.clip(self.model.predict_proba(self.preprocessor.transform(X))[:, 1], 1e-7, 1-1e-7)


def fit_count(config, X, y, seed):
    if config not in CONFIGS + CONTROLS:
        raise ValueError('Not a frozen configuration')
    result = CountModel(config)
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        result.preprocessor = CountDesign(config['basis']).fit(X)
        result.model = LogisticRegression(C=config['C'], solver='lbfgs', l1_ratio=0.,
                                          max_iter=4000, random_state=seed)
        result.model.fit(result.preprocessor.transform(X), y)
    result.iterations = int(result.model.n_iter_.max())
    return result


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    joblib.dump(value, temp)
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log(message, phase='running'):
    timestamp = datetime.now().astimezone().isoformat()
    line = f'[{timestamp}] {message}'
    print(line, flush=True)
    with (OUT / 'training.log').open('a', encoding='utf-8', buffering=1) as stream:
        stream.write(line + '\n')
    dump(OUT / 'status.json', dict(timestamp=timestamp, message=message, phase=phase, pid=os.getpid()))


def check_deadline():
    if datetime.now().astimezone() >= datetime.fromisoformat(DEADLINE):
        raise TimeoutError('18:45 deadline reached')


def make_submission(test_ids, sample_ids, p):
    if not test_ids.is_unique or not sample_ids.is_unique or test_ids.isna().any() or sample_ids.isna().any():
        raise ValueError('Invalid IDs')
    if len(test_ids) != len(sample_ids) or set(test_ids) != set(sample_ids):
        raise ValueError('ID set mismatch')
    p = np.asarray(p)
    if p.shape != (len(test_ids),) or not np.isfinite(p).all() or not ((p > 0) & (p < 1)).all():
        raise ValueError('Invalid probabilities')
    return pd.DataFrame({ID: sample_ids.to_numpy(), TARGET: pd.Series(p, index=test_ids).reindex(sample_ids).to_numpy()})


def metrics(y, p):
    return dict(log_loss=float(log_loss(y, p)), brier=float(brier_score_loss(y, p)), auc=float(roc_auc_score(y, p)))


def run(resume=False):
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ('models', 'cache'):
        (OUT / sub).mkdir(exist_ok=True)
    check_deadline()
    specification = dict(candidates=CONFIGS, controls=CONTROLS, seed=SEED, outer_folds=5, inner_folds=3,
        count_basis='Unscaled full one-hot after training median imputation; unknown ignored; retain all raw linear columns',
        target_encoding=False, cpu_threads=2, deadline=DEADLINE)
    frozen = OUT / 'frozen_config.json'
    if frozen.exists():
        if not resume or json.loads(frozen.read_text()) != specification:
            raise ValueError('Existing run requires --resume and unchanged specification')
    else:
        dump(frozen, specification)
    tr, te, sample = [pd.read_csv(p) for p in ('train.csv', 'test.csv', 'sample_submission.csv')]
    if len(tr) != 7000 or len(te) != 3000 or len(sample) != 3000 or int(tr[TARGET].sum()) != 881:
        raise ValueError('Unexpected dataset size/distribution')
    if not tr[ID].is_unique or tr[ID].isna().any() or set(tr[ID]) & set(te[ID]):
        raise ValueError('Invalid training IDs')
    if set(tr[TARGET]) != {0, 1} or TARGET in te or list(sample) != [ID, TARGET]:
        raise ValueError('Invalid target/schema')
    if set(tr.columns) - {TARGET} != set(te.columns):
        raise ValueError('Train/test feature schema mismatch')
    make_submission(te[ID], sample[ID], np.full(len(te), .5))
    manifest = dict(specification=specification,
        data={p:digest(p) for p in ('train.csv', 'test.csv', 'sample_submission.csv')},
        code={p:digest(p) for p in ('nexus/count_logistic.py', 'nexus/features.py', 'tests/test_count_logistic.py')},
        python=sys.version, platform=platform.platform(),
        versions={p:importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'threadpoolctl', 'joblib')})
    manifest['version_hash'] = hashlib.sha256(json.dumps(manifest['versions'], sort_keys=True).encode()).hexdigest()
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        if not resume or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Resume provenance mismatch')
    else:
        dump(manifest_path, manifest)
    log('Frozen six count candidates plus three matched linear controls; CPU=2; seed=20260915')
    X, y = tr.drop(columns=TARGET), tr[TARGET].to_numpy()
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    fold_ids = np.empty(len(y), dtype=int)
    for f, (_, ov) in enumerate(splits):
        fold_ids[ov] = f
    pd.DataFrame({ID:tr[ID], 'fold':fold_ids}).to_csv(OUT/'folds.csv', index=False)
    names = ('selected', 'linear_control')
    oof = {name:np.full(len(y), np.nan) for name in names}
    test_p = {name:[] for name in names}
    results = []
    for f, (ot, ov) in enumerate(splits):
        inner_splits = list(StratifiedKFold(3, shuffle=True, random_state=SEED+f+1).split(ot, y[ot]))
        inner_ids = np.empty(len(ot), dtype=int)
        for j, (_, iv) in enumerate(inner_splits):
            inner_ids[iv] = j
        inner_frame = pd.DataFrame({ID:tr[ID].iloc[ot].to_numpy(), TARGET:y[ot], 'row_index':ot, 'inner_fold':inner_ids})
        inner_frame.to_csv(OUT/f'inner_folds_{f}.csv', index=False)
        scores = {}
        for config in CONFIGS + CONTROLS:
            predictions = np.full(len(ot), np.nan)
            for j, (it, iv) in enumerate(inner_splits):
                check_deadline()
                cache = OUT/'cache'/f'o{f}_{config["name"]}_i{j}.joblib'
                if cache.exists():
                    item = joblib.load(cache)
                    np.testing.assert_array_equal(item['validation_rows'], ot[iv])
                    np.testing.assert_array_equal(item['training_rows'], ot[it])
                    if item['model'].config != config:
                        raise ValueError('Cached model configuration mismatch')
                else:
                    model = fit_count(config, X.iloc[ot[it]], y[ot[it]], SEED+f*100+j)
                    item = dict(model=model, prediction=model.predict(X.iloc[ot[iv]]),
                        training_rows=ot[it], validation_rows=ot[iv],
                        train_log_loss=float(log_loss(y[ot[it]], model.predict(X.iloc[ot[it]]))))
                    save(cache, item)
                predictions[iv] = item['prediction']
                log(f'outer={f+1}/5 {config["name"]} inner={j+1}/3')
            scores[config['name']] = float(log_loss(y[ot], predictions))
            inner_frame[config['name']] = predictions
        inner_frame.to_csv(OUT/f'inner_oof_{f}.csv', index=False)
        chosen = {name:min(grid, key=lambda c:scores[c['name']]) for name,grid in zip(names,(CONFIGS,CONTROLS))}
        result = dict(fold=f, chosen=chosen, inner_scores=scores, metrics={})
        for name in names:
            check_deadline()
            model = fit_count(chosen[name], X.iloc[ot], y[ot], SEED+f*100)
            save(OUT/'models'/f'outer{f}_{name}.joblib', model)
            oof[name][ov] = model.predict(X.iloc[ov])
            test_p[name].append(model.predict(te))
            result['metrics'][name] = metrics(y[ov], oof[name][ov])
        result['delta_log_loss'] = result['metrics']['selected']['log_loss'] - result['metrics']['linear_control']['log_loss']
        results.append(result)
        dump(OUT/f'fold_{f}.json', result)
        pd.DataFrame({ID:tr[ID], TARGET:y, 'fold':fold_ids, **oof}).to_csv(OUT/'oof.csv', index=False)
        dump(OUT/'summary.json', dict(complete=False, folds=results))
        log(f'Outer fold {f+1}/5 complete')
    all_metrics = {name:metrics(y,oof[name]) for name in names}
    difference = (-y*np.log(oof['selected'])-(1-y)*np.log1p(-oof['selected'])) - (
        -y*np.log(oof['linear_control'])-(1-y)*np.log1p(-oof['linear_control']))
    rng = np.random.default_rng(SEED)
    bootstrap = [float(difference[rng.integers(len(y),size=len(y))].mean()) for _ in range(2000)]
    for name in names:
        frame = make_submission(te[ID], sample[ID], np.mean(test_p[name],axis=0))
        frame.to_csv(OUT/f'submission_{name}.csv', index=False)
        saved = pd.read_csv(OUT/f'submission_{name}.csv')
        np.testing.assert_array_equal(saved[ID], sample[ID])
        make_submission(saved[ID],sample[ID],saved[TARGET])
        pd.DataFrame(np.array(test_p[name]).T, columns=[f'fold_{f}' for f in range(5)]).assign(
            **{ID:te[ID]}).to_csv(OUT/f'test_predictions_{name}.csv', index=False)
    # Read the completed smooth leader only after the frozen experiment is finished.
    smooth_path = Path('runs/smooth_v8/summary.json')
    context = None
    if smooth_path.exists():
        smooth = json.loads(smooth_path.read_text())
        if smooth.get('complete'):
            smooth_oof_path = smooth_path.parent/'oof.csv'
            smooth_oof = pd.read_csv(smooth_oof_path)
            if not smooth_oof[ID].is_unique or set(smooth_oof[ID]) != set(tr[ID]):
                raise ValueError('Smooth context IDs do not match')
            aligned = smooth_oof.set_index(ID).loc[tr[ID]]
            np.testing.assert_array_equal(aligned[TARGET], y)
            context = dict(method=smooth['winner'], metrics=metrics(y,aligned[smooth['winner']]),
                source_hashes={str(p):digest(p) for p in (smooth_path,smooth_oof_path)},
                note='Adaptive cross-run context on aligned IDs, different outer partition; not a matched-fold comparison or independent confirmation.')
    summary = dict(complete=True, seed=SEED, metrics=all_metrics, folds=results,
        delta_log_loss=float(difference.mean()), delta_ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
        smooth_context=context, submission_checks=dict(rows=3000, exact_sample_id_order=True, finite_bounded=True),
        evaluation_note='Six-candidate selection and separate three-C linear control use only inner OOF labels. '
        'Paired bootstrap conditions on saved predictions; does not include training/selection variance. '
        'Fresh partition of previously inspected labels is still adaptive development.')
    dump(OUT/'summary.json',summary)
    pd.DataFrame([dict(method=name,**all_metrics[name]) for name in names]).to_csv(OUT/'comparison.csv',index=False)
    dump(OUT/'artifact_hashes.json',{str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file()
        and p.name not in ('artifact_hashes.json','training.log','status.json')})
    log(f'COMPLETE metrics={all_metrics}; delta={difference.mean():.12f}',phase='complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        try:
            run(args.resume)
        except Exception as exc:
            if OUT.exists():
                log(f'FAILED {type(exc).__name__}: {exc}',phase='failed')
            raise
        return
    remaining = (datetime.fromisoformat(DEADLINE)-datetime.now().astimezone()).total_seconds()
    if remaining <= 0:
        raise SystemExit('18:45 deadline passed; no training started')
    command = [sys.executable,'-B','-u','-m','nexus.count_logistic','--worker']
    if args.resume:
        command.append('--resume')
    try:
        result = subprocess.run(command,timeout=remaining)
        raise SystemExit(result.returncode)
    except subprocess.TimeoutExpired:
        log('HARD DEADLINE: worker stopped; completed checkpoints retained',phase='deadline')
        raise SystemExit(2)


if __name__ == '__main__':
    from nexus.count_logistic import main as canonical_main
    canonical_main()
