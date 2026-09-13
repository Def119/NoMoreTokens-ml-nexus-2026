"""Standalone frozen 20-fold campaign: python -B -u -m nexus.final_refit20.

Reuses pure helpers and the progress interface from final_refit; never redirects
its globals or invokes its runner. All writes remain in runs/final20_v11.
"""
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from .final_refit import (ROOT, SEED, FAMILIES, NAMES, METHODS, Progress as BaseProgress,
                         fixed_configs, probabilities, aligned_baseline, submission,
                         atomic_csv, atomic_joblib, row_loss, digest, dump, metrics)
from .features import ID, TARGET
from .models import fit_model
from .ensemble import Combiner

OUT = ROOT / 'runs/final20_v11'
TEN = ROOT / 'runs/final10_v9'
SMOOTH = ROOT / 'runs/smooth_v8'
OUTER_FOLDS = 20
TOTAL_FITS = 400
DEADLINE = '2026-09-13T23:45:00+05:30'
NOTE = ('Post-hoc adaptive training-fraction expansion: this 20-fold/95% experiment '
        'was requested after inspecting smooth_v8 and final10_v9 results on the same labels. '
        'Five configurations, equal weights and equal_cal C=1 are frozen from final10. '
        'Outer labels are scoring-only within this run. Conditional paired patient-bootstrap '
        'intervals omit training/tuning uncertainty, overlapping-fold dependence and adaptive '
        'selection. A CI containing zero does not establish statistical superiority. Even a '
        'CI excluding zero is conditional evidence, not independent confirmation after this '
        'post-hoc expansion. Higher training fraction does not guarantee test improvement.')


def outer_splits(y):
    return list(StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def frozen_recipe():
    frozen = json.loads((TEN / 'resolved_config.json').read_text())
    old = json.loads((TEN / 'manifest.json').read_text())
    actual = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    if actual != old['fixed_config_sha256'] or frozen != old['fixed_config']:
        raise ValueError('Final10 frozen recipe provenance mismatch')
    if frozen != dict(candidates=fixed_configs(frozen), methods=list(METHODS), equal_cal_C=1.0):
        raise ValueError('Wrong fixed recipe or combiner settings')
    smooth = fixed_configs(json.loads((SMOOTH / 'resolved_config.json').read_text()))
    if frozen['candidates'] != smooth:
        raise ValueError('Final10 and smooth fixed configurations differ')
    return frozen


def paired_delta(y, p, reference):
    delta = row_loss(y, p) - row_loss(y, reference)
    rng = np.random.default_rng(SEED)
    draws = [float(delta[rng.integers(len(delta), size=len(delta))].mean()) for _ in range(5000)]
    ci = np.quantile(draws, [.025, .975]).tolist()
    return dict(delta_log_loss=float(delta.mean()), ci95=ci, repeats=5000, seed=SEED,
                conditional_interval_excludes_zero=bool(ci[1] < 0 or ci[0] > 0), interpretation=NOTE)


class Progress(BaseProgress):
    def log(self, message, **state):
        self.state.update(state)
        elapsed = time.monotonic() - self.start
        avg = np.mean([v for values in self.times.values() for v in values] or [2.2])
        eta = max(0, TOTAL_FITS - self.done) * float(avg)
        now = datetime.now().astimezone()
        line = f'[{now:%H:%M:%S}] elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m {message}'
        print(line, flush=True)
        with (OUT / 'training.log').open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')
        dump(OUT / 'status.json', dict(timestamp=now.isoformat(), elapsed_seconds=elapsed,
             eta_seconds=eta, deadline=self.deadline.isoformat(), completed_fits=self.done,
             total_fits=TOTAL_FITS, message=message, **self.state))


def save_checkpoint(value, path):
    atomic_joblib(value, path)
    dump(path.with_suffix('.sha256.json'), {'sha256': digest(path)})


def load_checkpoint(path):
    expected = json.loads(path.with_suffix('.sha256.json').read_text())['sha256']
    if digest(path) != expected:
        raise ValueError(f'Checkpoint checksum mismatch: {path.name}')
    return joblib.load(path)


def run(resume=False, deadline=DEADLINE):
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ('cache', 'models'):
        (OUT / folder).mkdir(exist_ok=True)
    import msvcrt
    with (OUT / 'campaign.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        return _run(resume, deadline)


def _run(resume, deadline):
    progress = Progress(deadline)
    tr, te, sample = [pd.read_csv(ROOT / p) for p in ('train.csv', 'test.csv', 'sample_submission.csv')]
    if not tr[ID].is_unique or set(tr[TARGET]) != {0, 1} or TARGET in te:
        raise ValueError('Invalid training/test schema')
    submission(te, sample, np.full(len(te), .5))
    frozen = frozen_recipe()
    original = json.loads((TEN / 'manifest.json').read_text())
    for group in ('data', 'source'):
        for filename, expected in original[group].items():
            if digest(ROOT / filename) != expected:
                raise ValueError(f'Final10 {group} dependency changed: {filename}')
    for package, expected in original['packages'].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f'Final10 runtime changed: {package}')
    references = {}
    for label, base in [('final10_equal_cal', TEN), ('smooth5_leader_equal_cal', SMOOTH)]:
        summary = json.loads((base / 'summary.json').read_text())
        if not summary.get('complete') or summary.get('smoke'):
            raise ValueError('Both baselines must be complete, non-smoke')
        if base == SMOOTH and summary['winner'] != 'equal_cal':
            raise ValueError('Smooth leader differs from frozen equal_cal reference')
        references[label] = aligned_baseline(tr, pd.read_csv(base / 'oof.csv'))
    manifest = dict(data=original['data'], source={p: digest(ROOT / p) for p in
        ['nexus/final_refit20.py', 'nexus/final_refit.py', 'nexus/features.py', 'nexus/models.py',
         'nexus/ensemble.py', 'nexus/train.py', 'tests/test_final_refit20.py']},
        baselines={str((base / filename).relative_to(ROOT)): digest(base / filename)
                   for base in (TEN, SMOOTH) for filename in ('manifest.json', 'resolved_config.json', 'summary.json', 'oof.csv')},
        fixed_config=frozen, fixed_config_sha256=original['fixed_config_sha256'],
        outer_folds=OUTER_FOLDS, inner_folds=3, seed=SEED, threads=4, backend='GPU',
        total_fits=TOTAL_FITS, packages=original['packages'], python=sys.version,
        evaluation_note=NOTE, nondeterminism='CatBoost GPU floating-point operations may be nondeterministic.')
    path = OUT / 'manifest.json'
    if path.exists():
        if not resume or json.loads(path.read_text()) != manifest:
            raise ValueError('Resume requires an identical manifest')
    else:
        dump(path, manifest)
    dump(OUT / 'resolved_config.json', frozen)
    configs = frozen['candidates']
    X, y = tr.drop(columns=TARGET), tr[TARGET]
    splits = outer_splits(y)
    fold_ids = np.full(len(tr), -1, dtype=int)
    for f, (_, ov) in enumerate(splits):
        fold_ids[ov] = f
    atomic_csv(pd.DataFrame({ID: tr[ID], 'fold': fold_ids}), OUT / 'folds.csv')
    predictions = {m: np.full(len(tr), np.nan) for m in METHODS}
    tests, results = {m: [] for m in METHODS}, []
    progress.log('Frozen 20-fold campaign: 300 inner fits + 100 refits, one GPU, threads=4', phase='starting')
    for f, (ot, ov) in enumerate(splits):
        progress.state.update(fold=f + 1)
        finish = OUT / 'cache' / f'outer_{f}.joblib'
        model_path = OUT / 'models' / f'fold_{f}.joblib'
        if finish.exists():
            saved = load_checkpoint(finish)
            if digest(model_path) != saved['model_sha256']:
                raise ValueError('Completed fold model checksum mismatch')
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
                        z = load_checkpoint(cache)
                        np.testing.assert_array_equal(z['indices'], iv)
                    else:
                        if not progress.check(family):
                            return False
                        progress.log(f'FIT outer={f+1}/20 {config["name"]} inner={j+1}/3', phase='inner')
                        with threadpool_limits(limits=4):
                            model = fit_model(config, tx.iloc[it], ty.iloc[it], seed=SEED + f * 100 + j,
                                threads=4, backend='GPU', validation=(tx.iloc[iv], ty.iloc[iv])
                                if family in ('lgb', 'cb') else None)
                            p = model.predict(tx.iloc[iv])
                        z = dict(indices=iv, prediction=probabilities(p, len(iv)), iterations=model.iterations,
                                 seconds=model.seconds)
                        save_checkpoint(z, cache)
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
                    z = load_checkpoint(cache)
                    np.testing.assert_array_equal(z['indices'], ov)
                else:
                    if not progress.check(family):
                        return False
                    progress.log(f'REFIT outer={f+1}/20 {config["name"]} n={len(tx)} iterations={iterations[family]}', phase='refit')
                    with threadpool_limits(limits=4):
                        model = fit_model(config, tx, ty, seed=SEED + f * 100, threads=4,
                                          backend='GPU', iterations=iterations[family])
                        vp, tp = model.predict(X.iloc[ov]), model.predict(te)
                    z = dict(model=model, indices=ov, outer=probabilities(vp, len(ov)),
                             test=probabilities(tp, len(te)), seconds=model.seconds)
                    save_checkpoint(z, cache)
                progress.completed(family, z['seconds'])
                fitted[config['name']] = z['model']
                val_matrix.append(z['outer'])
                test_matrix.append(z['test'])
            a, b, t = map(np.column_stack, (inner_matrix, val_matrix, test_matrix))
            combiners = {m: Combiner('equal', m == 'equal_cal').fit(a, ty) for m in METHODS}
            if combiners['equal_cal'].calibrator_.C != 1.0:
                raise ValueError('Unexpected calibration C')
            pp = {m: probabilities(combiners[m].predict(b), len(ov)) for m in METHODS}
            tp = {m: probabilities(combiners[m].predict(t), len(te)) for m in METHODS}
            result = dict(fold=f, train_n=len(ot), validation_n=len(ov), iterations=iterations,
                inner_metrics=inner_scores, outer_metrics={m: metrics(y.iloc[ov], pp[m]) for m in METHODS})
            save_checkpoint(dict(models=fitted, chosen=dict(zip(FAMILIES, configs)), combiners=combiners,
                outer_train=ot, outer_validation=ov, family_order=FAMILIES), model_path)
            saved = dict(indices=ov, outer=pp, test=tp, result=result, model_sha256=digest(model_path))
            save_checkpoint(saved, finish)  # Fold committed only after model and checksum exist.
        for m in METHODS:
            predictions[m][ov] = saved['outer'][m]
            tests[m].append(saved['test'][m])
        results.append(saved['result'])
        dump(OUT / f'fold_{f}_results.json', saved['result'])
        progress.log(f'OUTER {f+1}/20 {saved["result"]["outer_metrics"]}', phase='outer_complete')
    if progress.done != TOTAL_FITS:
        raise ValueError('Unexpected fit count')
    for m in METHODS:
        probabilities(predictions[m], len(tr))
        atomic_csv(submission(te, sample, np.mean(tests[m], axis=0)), OUT / f'submission_{m}.csv')
        delivered = pd.read_csv(OUT / f'submission_{m}.csv')
        if not delivered[ID].equals(sample[ID]):
            raise ValueError('Serialized submission IDs changed')
        submission(te, sample, delivered[TARGET])
    atomic_csv(pd.DataFrame({ID: tr[ID], TARGET: y, 'fold': fold_ids, **predictions}), OUT / 'oof.csv')
    variation = {m: {metric: dict(mean=float(np.mean(vals)), std=float(np.std(vals, ddof=1)),
                 min=float(np.min(vals)), max=float(np.max(vals)))
                 for metric in ('log_loss', 'brier', 'auc')
                 for vals in [[r['outer_metrics'][m][metric] for r in results]]} for m in METHODS}
    summary = dict(complete=True, metrics={m: metrics(y, predictions[m]) for m in METHODS},
        folds=results, fold_variation=variation, baseline_metrics={n: metrics(y, p) for n, p in references.items()},
        paired_bootstrap={n: {m: paired_delta(y, predictions[m], p) for m in METHODS} for n, p in references.items()},
        evaluation_note=NOTE, outer_folds=OUTER_FOLDS, inner_folds=3, seed=SEED,
        total_fits=progress.done, submission_policy='Both frozen recipes; no outer-score winner selection.',
        model_training_fraction=.95, elapsed_seconds=time.monotonic() - progress.start)
    # Include all finished artifact checksums; summary is the final campaign commit.
    dump(OUT / 'artifact_checksums.json', {str(p.relative_to(OUT)): digest(p) for p in
         sorted(OUT.rglob('*')) if p.is_file() and p.suffix in ('.joblib', '.csv')})
    dump(OUT / 'summary.json', summary)
    progress.log(f'COMPLETE {summary["metrics"]}', phase='complete')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--deadline', default=DEADLINE)
    args = parser.parse_args()
    try:
        complete = run(args.resume, args.deadline)
    except Exception:
        import traceback
        OUT.mkdir(parents=True, exist_ok=True)
        dump(OUT / 'error.json', dict(traceback=traceback.format_exc(), timestamp=datetime.now().astimezone().isoformat()))
        raise
    sys.exit(0 if complete else 2)
