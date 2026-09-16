"""No model training: frozen recipe, split coverage, provenance and delivery guards."""
import json
import unittest
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from nexus.final_refit import (ROOT, BASE, OUT, NAMES, FAMILIES, METHODS, digest, metrics,
                               fixed_configs, outer_splits, aligned_baseline, submission)
from nexus.features import ID, TARGET


class FinalRefitTests(unittest.TestCase):
    def test_exact_frozen_configs_and_copy(self):
        spec = json.loads((BASE / 'resolved_config.json').read_text())
        configs = fixed_configs(spec)
        self.assertEqual(tuple(c['name'] for c in configs), NAMES)
        self.assertEqual(tuple(c['family'] for c in configs), FAMILIES)
        for config in configs:
            self.assertEqual(config, next(c for c in spec['candidates'] if c['name'] == config['name']))
        self.assertEqual(configs[0]['params'], {'C': .1, 'l1_ratio': .8})
        self.assertEqual(configs[1]['params'], {'knots': 2, 'C': .01, 'l1_ratio': 0})
        configs[0]['params']['C'] = 999
        self.assertEqual(fixed_configs(spec)[0]['params']['C'], .1)
        spec['candidates'].append(fixed_configs(spec)[0])
        with self.assertRaises(ValueError):
            fixed_configs(spec)

    def test_tenfold_coverage_and_reproducibility(self):
        y = pd.read_csv(ROOT / 'train.csv')[TARGET]
        splits = outer_splits(y)
        self.assertEqual(len(splits), 10)
        coverage = np.zeros(len(y), int)
        for (ot, ov), (ot2, ov2) in zip(splits, outer_splits(y)):
            self.assertEqual(len(np.intersect1d(ot, ov)), 0)
            self.assertEqual(len(ot) + len(ov), len(y))
            self.assertAlmostEqual(len(ot) / len(y), .9, places=3)
            self.assertEqual(set(y.iloc[ov]), {0, 1})
            np.testing.assert_array_equal(ot, ot2)
            np.testing.assert_array_equal(ov, ov2)
            coverage[ov] += 1
        np.testing.assert_array_equal(coverage, np.ones(len(y), int))

    def test_baseline_alignment_rejects_shuffled_ids_and_labels(self):
        train = pd.read_csv(ROOT / 'train.csv')
        base = pd.read_csv(BASE / 'oof.csv')
        self.assertEqual(len(aligned_baseline(train, base)), len(train))
        with self.assertRaises(ValueError):
            aligned_baseline(train, base.iloc[::-1].reset_index(drop=True))
        base.loc[0, TARGET] = 1 - base.loc[0, TARGET]
        with self.assertRaises(ValueError):
            aligned_baseline(train, base)

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

    @unittest.skipUnless((OUT / 'summary.json').exists(), 'Full campaign not complete yet')
    def test_completed_artifacts_replay_and_average(self):
        train = pd.read_csv(ROOT / 'train.csv')
        test = pd.read_csv(ROOT / 'test.csv')
        sample = pd.read_csv(ROOT / 'sample_submission.csv')
        oof = pd.read_csv(OUT / 'oof.csv')
        summary = json.loads((OUT / 'summary.json').read_text())
        manifest = json.loads((OUT / 'manifest.json').read_text())
        self.assertTrue(summary['complete'])
        self.assertEqual(len(summary['folds']), 10)
        for path, expected in manifest['source'].items():
            self.assertEqual(digest(ROOT / path), expected)
        self.assertTrue(oof[ID].equals(train[ID]))
        self.assertTrue(oof[TARGET].equals(train[TARGET]))
        test_predictions = {m: [] for m in METHODS}
        with threadpool_limits(limits=4):
            for f, (ot, ov) in enumerate(outer_splits(train[TARGET])):
                model_path = OUT / 'models' / f'fold_{f}.joblib'
                saved = joblib.load(OUT / 'cache' / f'outer_{f}.joblib')
                self.assertEqual(saved['model_sha256'], digest(model_path))
                bundle = joblib.load(model_path)
                np.testing.assert_array_equal(bundle['outer_train'], ot)
                np.testing.assert_array_equal(bundle['outer_validation'], ov)
                self.assertTrue((oof.iloc[ov]['fold'] == f).all())
                self.assertEqual(bundle['combiners']['equal_cal'].calibrator_.C, 1.0)
                b = np.column_stack([bundle['models'][name].predict(train.iloc[ov]) for name in NAMES])
                t = np.column_stack([bundle['models'][name].predict(test) for name in NAMES])
                for m in METHODS:
                    np.testing.assert_allclose(bundle['combiners'][m].predict(b), oof.iloc[ov][m], rtol=0, atol=1e-12)
                    tp = bundle['combiners'][m].predict(t)
                    np.testing.assert_allclose(tp, saved['test'][m], rtol=0, atol=1e-12)
                    test_predictions[m].append(tp)
        for m in METHODS:
            delivered = pd.read_csv(OUT / f'submission_{m}.csv')
            validated = submission(test, sample, delivered[TARGET])
            self.assertTrue(validated[ID].equals(delivered[ID]))
            np.testing.assert_allclose(delivered[TARGET], np.mean(test_predictions[m], axis=0), rtol=0, atol=1e-12)
            self.assertAlmostEqual(metrics(train[TARGET], oof[m])['log_loss'], summary['metrics'][m]['log_loss'], places=12)


if __name__ == '__main__':
    unittest.main()
