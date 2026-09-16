import pickle
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.refine_ensembles import RECIPES, FAMILIES, Refinement, submission, probabilities, validate_hashes, paired_bootstrap


class EnsembleRefinementTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.p = rng.uniform(.01, .6, (120, 5))
        self.y = np.array([0, 0, 0, 1] * 30)

    def test_frozen_recipes(self):
        self.assertEqual(len(RECIPES), 6)
        self.assertEqual(FAMILIES, ['lr', 'spline', 'lgb', 'cb', 'ebm'])
        self.assertEqual(RECIPES[-1]['weights'], [.3, .3, .1, .1, .2])
        self.assertEqual(RECIPES[0]['weights'], [1/3, 1/3, 0., 0., 1/3])

    def test_determinism_bounds_and_prediction_does_not_fit(self):
        for recipe in RECIPES:
            a = Refinement(recipe).fit(self.p, self.y)
            b = Refinement(recipe).fit(self.p, self.y)
            np.testing.assert_array_equal(a.predict(self.p), b.predict(self.p))
            before = pickle.dumps(a)
            probabilities(a.predict(self.p))
            self.assertEqual(before, pickle.dumps(a))
            self.assertAlmostEqual(a.weights_.sum(), 1.)
            self.assertTrue((a.weights_ >= 0).all())
            if recipe['name'] == 'convex_shrink50':
                np.testing.assert_allclose(a.weights_, .5*a.convex_weights_ + .1)

    def test_calibration_uses_only_passed_training_rows(self):
        for recipe in RECIPES:
            model = Refinement(recipe).fit(self.p[:80], self.y[:80])
            before = pickle.dumps(model)
            held = self.p[80:].copy()
            held[:] = .999
            model.predict(held)
            self.assertEqual(before, pickle.dumps(model))
            np.testing.assert_allclose(model.predict(held[:1]), model.predict(held)[:1])

    def test_id_order_and_invalid_ids(self):
        test = pd.Series(['b', 'a', 'c'])
        sample = pd.Series(['c', 'b', 'a'])
        result = submission(test, sample, [.2, .3, .4])
        np.testing.assert_array_equal(result.patient_id, sample)
        np.testing.assert_array_equal(result.readmitted_30d, [.4, .2, .3])
        for invalid in (pd.Series(['a', 'a', 'b']), pd.Series(['a', 'b', 'd'])):
            with self.assertRaises(ValueError):
                submission(test, invalid, [.2, .3, .4])

    def test_probability_validation(self):
        for p in ([0., .2], [1., .2], [np.nan, .2], [np.inf, .2]):
            with self.assertRaises(ValueError):
                probabilities(p)

    def test_provenance_mismatch_rejected(self):
        with patch('scripts.refine_ensembles.digest', return_value='correct'):
            validate_hashes({'source': 'correct'})
            with self.assertRaises(ValueError):
                validate_hashes({'source': 'tampered'})

    def test_bootstrap_deterministic_and_paired(self):
        p = self.p.mean(axis=1)
        result = paired_bootstrap(self.y, p, p)
        self.assertEqual(result, dict(delta_log_loss=0., ci95=[0., 0.]))
        self.assertEqual(paired_bootstrap(self.y, p, p*.9), paired_bootstrap(self.y, p, p*.9))


if __name__ == '__main__':
    unittest.main()
