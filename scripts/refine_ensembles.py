"""Frozen CPU-only ensemble refinement of completed smooth_v8 caches.

Run: python -B -u -m scripts.refine_ensembles
No base models are trained. Existing artifacts are never overwritten.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '2'

import ast
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from nexus.ensemble import Combiner, clipped

BASE = Path('runs/smooth_v8')
OUT = Path('runs/ensemble_v8')
FAMILIES = ['lr', 'spline', 'lgb', 'cb', 'ebm']
RECIPES = [
    dict(name='equal_additive', weights=[1/3, 1/3, 0., 0., 1/3], calibration_C=None),
    dict(name='equal_all_cal_c0.1', weights=[.2]*5, calibration_C=.1),
    dict(name='equal_all_cal_c10', weights=[.2]*5, calibration_C=10.),
    dict(name='convex_shrink50', weights='convex_shrink50', calibration_C=None),
    dict(name='equal_additive_cal_c1', weights=[1/3, 1/3, 0., 0., 1/3], calibration_C=1.),
    dict(name='weighted_prior_cal_c1', weights=[.3, .3, .1, .1, .2], calibration_C=1.),
]
BASELINES = ['equal_cal', 'reference']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def validate_hashes(expected):
    for path, value in expected.items():
        if digest(path) != value:
            raise ValueError(f'Provenance hash mismatch: {path}')


def probabilities(p, rows=None):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError('Nonfinite or out-of-bounds probabilities')
    if rows is not None and len(p) != rows:
        raise ValueError('Prediction row count mismatch')
    return p


class Refinement:
    def __init__(self, recipe):
        self.recipe = recipe

    def fit(self, inner_predictions, inner_labels):
        p = probabilities(inner_predictions)
        y = np.asarray(inner_labels)
        if p.ndim != 2 or p.shape[1] != 5 or len(y) != len(p) or set(y) != {0, 1}:
            raise ValueError('Expected five ordered families and binary inner labels')
        with threadpool_limits(limits=2):
            if self.recipe['weights'] == 'convex_shrink50':
                convex = Combiner('convex').fit(p, y)
                self.convex_weights_ = convex.weights_.copy()
                self.weights_ = .5 * self.convex_weights_ + .5 * np.full(5, .2)
            else:
                self.weights_ = np.array(self.recipe['weights'], dtype=float)
            self.calibrator_ = None
            if self.recipe['calibration_C'] is not None:
                self.calibrator_ = LogisticRegression(C=self.recipe['calibration_C'], max_iter=2000).fit(
                    logit(clipped(p @ self.weights_))[:, None], y)
                if int(self.calibrator_.n_iter_.max()) >= 2000:
                    raise RuntimeError('Calibration did not converge')
        return self

    def predict(self, p):
        p = probabilities(p)
        if p.ndim != 2 or p.shape[1] != 5:
            raise ValueError('Wrong family dimensions')
        q = clipped(p @ self.weights_)
        with threadpool_limits(limits=2):
            if self.calibrator_ is not None:
                q = self.calibrator_.predict_proba(logit(q)[:, None])[:, 1]
        return clipped(q)

    def parameters(self):
        return dict(family_order=FAMILIES, weights=self.weights_.tolist(),
                    convex_weights=getattr(self, 'convex_weights_', np.array([])).tolist(),
                    calibration_C=self.recipe['calibration_C'],
                    calibration_coefficient=None if self.calibrator_ is None else float(self.calibrator_.coef_[0, 0]),
                    calibration_intercept=None if self.calibrator_ is None else float(self.calibrator_.intercept_[0]),
                    calibration_iterations=None if self.calibrator_ is None else int(self.calibrator_.n_iter_.max()))


def submission(test_ids, sample_ids, p):
    if len(test_ids) != len(sample_ids) or not test_ids.is_unique or not sample_ids.is_unique:
        raise ValueError('Duplicate IDs or row count mismatch')
    if test_ids.isna().any() or sample_ids.isna().any() or set(test_ids) != set(sample_ids):
        raise ValueError('ID sets differ')
    p = probabilities(p, len(test_ids))
    return pd.DataFrame({'patient_id': sample_ids.to_numpy(),
                         'readmitted_30d': pd.Series(p, index=test_ids).reindex(sample_ids).to_numpy()})


def loss_rows(y, p):
    return -(y * np.log(p) + (1-y) * np.log1p(-p))


def paired_bootstrap(y, p, reference, seed=20260914):
    differences = loss_rows(np.asarray(y), p) - loss_rows(np.asarray(y), reference)
    rng = np.random.default_rng(seed)
    samples = [float(differences[rng.integers(len(y), size=len(y))].mean()) for _ in range(2000)]
    return dict(delta_log_loss=float(differences.mean()), ci95=np.quantile(samples, [.025, .975]).tolist())


def metrics(y, p):
    return dict(log_loss=float(log_loss(y, p)), brier=float(brier_score_loss(y, p)),
                auc=float(roc_auc_score(y, p)))


def log(message):
    line = f'[{datetime.now().astimezone().isoformat()}] {message}'
    print(line, flush=True)
    with (OUT / 'training.log').open('a', encoding='utf-8', buffering=1) as stream:
        stream.write(line + '\n')


def run():
    if OUT.exists() and any(p.name != 'frozen_recipes.json' for p in OUT.iterdir()):
        raise ValueError('Output directory is not empty; refusing to overwrite artifacts')
    OUT.mkdir(parents=True, exist_ok=True)
    # This specification is persisted before any new recipe is fitted or scored.
    frozen_path = OUT / 'frozen_recipes.json'
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        if frozen['recipes'] != RECIPES or frozen['family_order'] != FAMILIES:
            raise ValueError('Previously frozen specification differs')
    else:
        dump(frozen_path, dict(recipes=RECIPES, family_order=FAMILIES,
        omission='Optional calibration of convex_shrink50 omitted to cap new candidates at six',
        baseline='equal_cal', promotion_rule='Only a strictly lower OOF log loss may replace equal_cal',
            frozen_at=datetime.now().astimezone().isoformat()))
    base_manifest = json.loads((BASE / 'manifest.json').read_text())
    base_summary = json.loads((BASE / 'summary.json').read_text())
    if not base_summary['complete'] or base_summary.get('smoke', False):
        raise ValueError('Base campaign must be complete and nonsmoke')
    expected = dict(base_manifest['data'])
    expected.update({str(Path('nexus') / name): value for name, value in base_manifest['code'].items() if name != 'train.py'})
    validate_hashes(expected)
    runner = Path('nexus/train.py').read_bytes()
    runner_origin = 'working_tree'
    if hashlib.sha256(runner).hexdigest() != base_manifest['code']['train.py']:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        runner = subprocess.check_output(['git', 'show', f'{revision}:nexus/train.py'])
        runner_origin = f'git:{revision}:nexus/train.py'
    if hashlib.sha256(runner).hexdigest() != base_manifest['code']['train.py']:
        raise ValueError('Cannot recover hash-matching base runner')
    assignments = [node for node in ast.parse(runner.decode('utf-8')).body
                   if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FAMILIES' for t in node.targets)]
    if len(assignments) != 1 or ast.literal_eval(assignments[0].value) != FAMILIES:
        raise ValueError('Base family ordering mismatch')
    train, test, sample = [pd.read_csv(f) for f in ('train.csv', 'test.csv', 'sample_submission.csv')]
    y = train.readmitted_30d.to_numpy()
    if len(train) != 7000 or len(test) != 3000 or len(sample) != 3000 or y.sum() != 881:
        raise ValueError('Unexpected dataset size/distribution')
    if not train.patient_id.is_unique or set(train.patient_id) & set(test.patient_id):
        raise ValueError('Train ID duplication/overlap')
    submission(test.patient_id, sample.patient_id, np.full(len(test), .5))
    seed = base_manifest['seed']
    if (seed, base_manifest['outer_folds'], base_manifest['inner_folds']) != (20260914, 5, 3):
        raise ValueError('Unexpected base split specification')
    old_oof = pd.read_csv(BASE / 'oof.csv')
    base_folds = pd.read_csv(BASE / 'folds.csv')
    np.testing.assert_array_equal(old_oof.patient_id, train.patient_id)
    np.testing.assert_array_equal(old_oof.readmitted_30d, y)
    np.testing.assert_array_equal(base_folds.patient_id, train.patient_id)
    paths = [BASE / n for n in ('manifest.json', 'summary.json', 'folds.csv', 'oof.csv')]
    paths += [Path('scripts/refine_ensembles.py'), Path('tests/test_refine_ensembles.py'), OUT / 'frozen_recipes.json']
    provenance = {str(p): digest(p) for p in paths}
    provenance.update(expected)
    names = [r['name'] for r in RECIPES] + BASELINES
    oof = {name: np.full(len(y), np.nan) for name in names}
    tests = {name: [] for name in names}
    folds = np.empty(len(y), dtype=int)
    results = []
    for f, (ot, ov) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(train, y)):
        folds[ov] = f
        np.testing.assert_array_equal(base_folds.loc[ov, 'fold'], f)
        np.testing.assert_array_equal(old_oof.loc[ov, 'fold'], f)
        completion = BASE / 'cache' / f'outer_{f}.joblib'
        provenance[str(completion)] = digest(completion)
        saved = joblib.load(completion)
        result = saved['result']
        if result['fold'] != f:
            raise ValueError('Wrong outer completion marker')
        splits = list(StratifiedKFold(3, shuffle=True, random_state=seed+f+1).split(ot, y[ot]))
        inner = np.full((len(ot), 5), np.nan)
        inner_fold = np.empty(len(ot), dtype=int)
        for i, family in enumerate(FAMILIES):
            chosen = result['chosen'][family]
            candidates = [c for c in base_manifest['candidates'] if c['family'] == family and not c['name'].startswith('reference')]
            if min(candidates, key=lambda c: result['inner_scores'][c['name']])['name'] != chosen:
                raise ValueError('Selected family does not match inner-only score selection')
            for j, (_, iv) in enumerate(splits):
                cache = BASE / 'cache' / f'o{f}_{chosen}_i{j}.npz'
                provenance[str(cache)] = digest(cache)
                with np.load(cache, allow_pickle=False) as z:
                    inner[iv, i] = probabilities(z['prediction'], len(iv))
                inner_fold[iv] = j
            np.testing.assert_allclose(log_loss(y[ot], inner[:, i]), result['inner_scores'][chosen], atol=1e-12, rtol=0)
        pd.DataFrame(inner, columns=FAMILIES).assign(patient_id=train.patient_id.iloc[ot].to_numpy(),
            row_index=ot, inner_fold=inner_fold, readmitted_30d=y[ot]).to_csv(OUT / f'inner_predictions_{f}.csv', index=False)
        outer = np.column_stack([probabilities(saved['outer'][family], len(ov)) for family in FAMILIES])
        test_p = np.column_stack([probabilities(saved['test'][family], len(test)) for family in FAMILIES])
        np.testing.assert_allclose(outer, old_oof.loc[ov, FAMILIES], atol=1e-14, rtol=0)
        # Reproduce existing calibration to validate reconstruction before new scoring.
        audit = Refinement(dict(weights=[.2]*5, calibration_C=1.)).fit(inner, y[ot])
        np.testing.assert_allclose(audit.predict(outer), saved['outer']['equal_cal'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(audit.predict(test_p), saved['test']['equal_cal'], atol=1e-12, rtol=0)
        parameters = {}
        for recipe in RECIPES:
            combiner = Refinement(recipe).fit(inner, y[ot])
            name = recipe['name']
            oof[name][ov] = combiner.predict(outer)
            tests[name].append(combiner.predict(test_p))
            parameters[name] = combiner.parameters()
        for name in BASELINES:
            oof[name][ov] = probabilities(saved['outer'][name], len(ov))
            tests[name].append(probabilities(saved['test'][name], len(test)))
            np.testing.assert_allclose(oof[name][ov], old_oof.loc[ov, name], atol=1e-14, rtol=0)
        fold_result = dict(fold=f, outer_training_rows=ot.tolist(), outer_validation_rows=ov.tolist(),
            chosen_base=result['chosen'], combiner_parameters=parameters,
            metrics={name: metrics(y[ov], oof[name][ov]) for name in names})
        dump(OUT / f'fold_{f}.json', fold_result)
        results.append(fold_result)
        log(f'Fold {f+1}/5 reconstructed, base calibration reproduced, six recipes evaluated')
    validate_hashes(provenance)
    manifest = dict(base=str(BASE), source_hashes=provenance, recipes=RECIPES, family_order=FAMILIES,
        base_runner=dict(origin=runner_origin, sha256=hashlib.sha256(runner).hexdigest()),
        seed=seed, outer_folds=5, inner_folds=3, cpu_threads=2, python=sys.version, platform=platform.platform(),
        versions={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'pandas', 'scikit-learn', 'joblib', 'threadpoolctl')},
        provenance_note='NPZ caches lack embedded row IDs; alignment reconstructed from hash-verified base runner and data, '
        'exact splits, selected inner scores, saved outer predictions, and reproduced equal_cal predictions.')
    dump(OUT / 'manifest.json', manifest)
    pd.DataFrame(dict(patient_id=train.patient_id, readmitted_30d=y, fold=folds, **oof)).to_csv(OUT / 'oof.csv', index=False)
    pd.DataFrame(dict(patient_id=train.patient_id, fold=folds)).to_csv(OUT / 'folds.csv', index=False)
    all_metrics = {name: metrics(y, oof[name]) for name in names}
    boot = {name: paired_bootstrap(y, oof[name], oof['equal_cal']) for name in names}
    comparison = pd.DataFrame([dict(method=name, **all_metrics[name],
        delta_vs_equal_cal=boot[name]['delta_log_loss'], ci95_low=boot[name]['ci95'][0], ci95_high=boot[name]['ci95'][1]) for name in names]).sort_values('log_loss')
    comparison.to_csv(OUT / 'comparison.csv', index=False)
    for name in names:
        frame = submission(test.patient_id, sample.patient_id, np.mean(tests[name], axis=0))
        frame.to_csv(OUT / f'submission_{name}.csv', index=False)
        saved_submission = pd.read_csv(OUT / f'submission_{name}.csv')
        np.testing.assert_array_equal(saved_submission.patient_id, sample.patient_id)
        probabilities(saved_submission.readmitted_30d, 3000)
        pd.DataFrame(np.array(tests[name]).T, columns=[f'fold_{f}' for f in range(5)]).assign(
            patient_id=test.patient_id).to_csv(OUT / f'test_predictions_{name}.csv', index=False)
    best_new = min(RECIPES, key=lambda r: all_metrics[r['name']]['log_loss'])['name']
    recommended = best_new if all_metrics[best_new]['log_loss'] < all_metrics['equal_cal']['log_loss'] else 'equal_cal'
    submission(test.patient_id, sample.patient_id, np.mean(tests[recommended], axis=0)).to_csv(OUT / 'submission_recommended.csv', index=False)
    dump(OUT / 'summary.json', dict(complete=True, metrics=all_metrics, paired_bootstrap=boot,
        folds=results, best_new=best_new, recommended=recommended, promoted=recommended != 'equal_cal',
        evaluation_note='All weights/calibrations fitted only on outer-training inner OOF labels. '
        'Outer labels used for evaluation and final recipe ranking only. Choosing among six outer scores adds selection optimism. '
        'Conditional paired bootstrap omits model-fitting and adaptive-selection uncertainty; intervals are not multiplicity-adjusted.'))
    log(f'COMPLETE recommended={recommended}; best_new={best_new}')


if __name__ == '__main__':
    run()
