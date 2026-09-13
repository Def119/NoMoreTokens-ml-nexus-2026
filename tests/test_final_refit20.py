"""Contract and saved-model replay checks; no model training."""
import json
import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from nexus.final_refit20 import (ROOT, OUT, TEN, SMOOTH, NAMES, METHODS, SEED,
    frozen_recipe, outer_splits, submission, aligned_baseline, load_checkpoint, digest, metrics)
from nexus.features import ID, TARGET


class FinalRefit20Tests(unittest.TestCase):
    def test_frozen_recipe_is_identical_to_final10(self):
        recipe = frozen_recipe()
        self.assertEqual(recipe, json.loads((TEN / 'resolved_config.json').read_text()))
        self.assertEqual(tuple(c['name'] for c in recipe['candidates']), NAMES)
        self.assertEqual(recipe['methods'], ['equal', 'equal_cal'])
        self.assertEqual(recipe['equal_cal_C'], 1.0)
        self.assertEqual(SEED, 20260914)

    def test_twentyfold_coverage_and_reproducibility(self):
        y = pd.read_csv(ROOT / 'train.csv')[TARGET]
        splits = outer_splits(y)
        self.assertEqual(len(splits), 20)
        coverage = np.zeros(len(y), int)
        for (ot, ov), (ot2, ov2) in zip(splits, outer_splits(y)):
            self.assertEqual(len(np.intersect1d(ot, ov)), 0)
            self.assertEqual(len(ot) + len(ov), len(y))
            self.assertEqual(len(ot), 6650)
            self.assertEqual(len(ov), 350)
            self.assertEqual(set(y.iloc[ov]), {0, 1})
            np.testing.assert_array_equal(ot, ot2)
            np.testing.assert_array_equal(ov, ov2)
            coverage[ov] += 1
        np.testing.assert_array_equal(coverage, np.ones(len(y), int))

    def test_baseline_alignment(self):
        train = pd.read_csv(ROOT / 'train.csv')
        for base in (TEN, SMOOTH):
            frame = pd.read_csv(base / 'oof.csv')
            self.assertEqual(len(aligned_baseline(train, frame)), len(train))
            with self.assertRaises(ValueError):
                aligned_baseline(train, frame.iloc[::-1].reset_index(drop=True))
            frame.loc[0, TARGET] = 1 - frame.loc[0, TARGET]
            with self.assertRaises(ValueError):
                aligned_baseline(train, frame)

    def test_submission_contract(self):
        test = pd.DataFrame({ID: [11, 12]})
        sample = pd.DataFrame({ID: [11, 12], TARGET: [0, 0]})
        frame = submission(test, sample, [.1, .9])
        self.assertEqual(frame.columns.tolist(), sample.columns.tolist())
        self.assertTrue(frame[ID].equals(sample[ID]))
        for bad in ([np.nan, .2], [-.1, .2], [.5, 1.1], [.1]):
            with self.assertRaises(ValueError):
                submission(test, sample, bad)
        with self.assertRaises(ValueError):
            submission(test, sample.iloc[::-1].reset_index(drop=True), [.1, .2])

    @unittest.skipUnless((OUT / 'summary.json').exists(), 'Campaign not yet complete')
    def test_completed_artifact_checksums_model_replay_and_average(self):
        train = pd.read_csv(ROOT / 'train.csv')
        test = pd.read_csv(ROOT / 'test.csv')
        sample = pd.read_csv(ROOT / 'sample_submission.csv')
        oof = pd.read_csv(OUT / 'oof.csv')
        summary = json.loads((OUT / 'summary.json').read_text())
        manifest = json.loads((OUT / 'manifest.json').read_text())
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['total_fits'], 400)
        self.assertEqual(len(summary['folds']), 20)
        self.assertEqual(len(list((OUT / 'cache').glob('o*_i*.joblib'))), 300)
        self.assertEqual(len(list((OUT / 'cache').glob('o*_refit.joblib'))), 100)
        for path, expected in manifest['source'].items():
            self.assertEqual(digest(ROOT / path), expected)
        for path, expected in json.loads((OUT / 'artifact_checksums.json').read_text()).items():
            self.assertEqual(digest(OUT / path), expected)
        self.assertTrue(oof[ID].equals(train[ID]))
        self.assertTrue(oof[TARGET].equals(train[TARGET]))
        tests = {m: [] for m in METHODS}
        with threadpool_limits(limits=4):
            for f, (ot, ov) in enumerate(outer_splits(train[TARGET])):
                path = OUT / 'models' / f'fold_{f}.joblib'
                saved = load_checkpoint(OUT / 'cache' / f'outer_{f}.joblib')
                self.assertEqual(saved['model_sha256'], digest(path))
                bundle = load_checkpoint(path)
                np.testing.assert_array_equal(bundle['outer_train'], ot)
                np.testing.assert_array_equal(bundle['outer_validation'], ov)
                self.assertTrue((oof.iloc[ov]['fold'] == f).all())
                self.assertEqual(bundle['combiners']['equal_cal'].calibrator_.C, 1.0)
                for family in ('lgb', 'cb'):
                    its = [load_checkpoint(OUT / 'cache' / f'o{f}_{family}_i{j}.joblib')['iterations'] for j in range(3)]
                    self.assertEqual(saved['result']['iterations'][family], int(np.median(its)))
                b = np.column_stack([bundle['models'][name].predict(train.iloc[ov]) for name in NAMES])
                t = np.column_stack([bundle['models'][name].predict(test) for name in NAMES])
                for m in METHODS:
                    np.testing.assert_allclose(bundle['combiners'][m].predict(b), oof.iloc[ov][m], rtol=0, atol=1e-12)
                    tp = bundle['combiners'][m].predict(t)
                    np.testing.assert_allclose(tp, saved['test'][m], rtol=0, atol=1e-12)
                    tests[m].append(tp)
        for m in METHODS:
            delivered = pd.read_csv(OUT / f'submission_{m}.csv')
            self.assertTrue(submission(test, sample, delivered[TARGET])[ID].equals(delivered[ID]))
            np.testing.assert_allclose(delivered[TARGET], np.mean(tests[m], axis=0), rtol=0, atol=1e-12)
            self.assertAlmostEqual(metrics(train[TARGET], oof[m])['log_loss'], summary['metrics'][m]['log_loss'], places=12)


if __name__ == '__main__':
    unittest.main()
