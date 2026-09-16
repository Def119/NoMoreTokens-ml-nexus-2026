import pickle
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from nexus.features import LABS, linear_preprocessor, engineer
from nexus.residual_spline import CONFIGS, ResidualBasis, fit_residual, validate_data


class ResidualSplineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv('train.csv').iloc[:400].copy()
        cls.model = fit_residual(CONFIGS[1], cls.data.iloc[:300], cls.data.readmitted_30d.iloc[:300], 42)

    def test_frozen_grid(self):
        self.assertEqual({(c['params']['C'], c['params']['multiplier']) for c in CONFIGS},
                         {(c, a) for c in (.01, .03, .1) for a in (0., .25)})
        with self.assertRaises(ValueError):
            fit_residual({}, self.data, self.data.readmitted_30d, 42)

    def test_unseen_missing_and_no_prediction_mutation(self):
        held = self.data.iloc[300:].copy()
        before = pickle.dumps(self.model)
        held['region'] = 'UNSEEN'
        held['age'] = 99999.
        for col in LABS:
            held[col] = np.nan
        p = self.model.predict(held)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(((p > 0) & (p < 1)).all())
        self.assertEqual(before, pickle.dumps(self.model))
        np.testing.assert_allclose(p[:1], self.model.predict(held.iloc[:1]), rtol=1e-12)

    def test_id_and_target_never_enter_features(self):
        changed = self.data.copy()
        changed['patient_id'] = 'changed'
        changed['readmitted_30d'] = 1 - changed.readmitted_30d
        np.testing.assert_array_equal(self.model.predict(changed), self.model.predict(self.data))
        other = fit_residual(CONFIGS[1], changed.iloc[:300], self.data.readmitted_30d.iloc[:300], 42)
        np.testing.assert_array_equal(other.predict(self.data), self.model.predict(self.data))

    def test_training_only_statistics_and_projection(self):
        train = self.data.iloc[:300]
        basis = self.model.basis
        for col, block in basis.blocks_.items():
            if block is None:
                continue
            median, scale, spline, projection, keep, scaler = block
            self.assertAlmostEqual(median, train[col].median())
            filled = train[col].fillna(median).to_numpy()[:, None]
            design = np.column_stack([np.ones(len(train)), scale.transform(filled)[:, 0], train[col].isna()])
            residual = spline.transform(filled) - design @ projection
            np.testing.assert_allclose(design.T @ residual, 0, atol=1e-9)
            np.testing.assert_allclose(spline.bsplines_[0].t[3:-3], np.quantile(filled, [0, .5, 1]))
        x = engineer(train)
        expected = .25 * basis.transform(x)
        np.testing.assert_allclose(self.model.transform(train)[:, -expected.shape[1]:], expected)

    def test_zero_multiplier_matches_raw_logistic(self):
        train = self.data.iloc[:300]
        fitted = fit_residual(CONFIGS[0], train, train.readmitted_30d, 42)
        x = engineer(train)
        pre = linear_preprocessor(x).fit(x)
        with threadpool_limits(limits=2):
            baseline = LogisticRegression(C=.01, solver='lbfgs', l1_ratio=0., max_iter=4000, random_state=42)
            baseline.fit(pre.transform(x), train.readmitted_30d)
        np.testing.assert_allclose(fitted.predict(train), baseline.predict_proba(pre.transform(x))[:, 1], atol=1e-12)
        self.assertIsNone(fitted.basis)

    def test_degenerate_training_columns_and_serialization(self):
        train = self.data.iloc[:300].copy()
        train['hemoglobin_g_dl'] = np.nan
        train['age'] = 50.
        fitted = fit_residual(CONFIGS[1], train, train.readmitted_30d, 42)
        self.assertIsNone(fitted.basis.blocks_['age'])
        self.assertIsNone(fitted.basis.blocks_['hemoglobin_g_dl'])
        restored = pickle.loads(pickle.dumps(fitted))
        self.assertEqual(type(restored).__module__, 'nexus.residual_spline')
        self.assertGreater(restored.iterations, 0)
        self.assertEqual(restored.config, CONFIGS[1])
        np.testing.assert_array_equal(restored.predict(self.data), fitted.predict(self.data))

    def test_submission_id_contract(self):
        train, test, sample = [pd.read_csv(f) for f in ('train.csv', 'test.csv', 'sample_submission.csv')]
        validate_data(train, test, sample)
        broken = sample.copy()
        broken.loc[0, 'patient_id'] = 'UNKNOWN'
        with self.assertRaises(ValueError):
            validate_data(train, test, broken)


if __name__ == '__main__':
    unittest.main()
