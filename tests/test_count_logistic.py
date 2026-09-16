import pickle
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from nexus.features import engineer,linear_preprocessor,LABS
from nexus.count_logistic import CONFIGS,CONTROLS,COUNTS,CountDesign,fit_count,make_submission


class CountLogisticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv('train.csv').iloc[:400].copy()
        cls.model = fit_count(CONFIGS[3],cls.data.iloc[:300],cls.data.readmitted_30d.iloc[:300],42)

    def test_frozen_grid_and_sklearn_clone(self):
        self.assertEqual(len(CONFIGS),6)
        self.assertEqual({(c['basis'],c['C']) for c in CONFIGS},{(b,c) for b in ('A','B') for c in (.01,.03,.1)})
        self.assertEqual([c['C'] for c in CONTROLS],[.01,.03,.1])
        self.assertEqual(clone(CountDesign('B')).get_params(),{'basis':'B'})

    def test_label_and_id_isolation(self):
        changed = self.data.copy()
        changed['patient_id'] = 'ignored'
        changed['readmitted_30d'] = 1-changed.readmitted_30d
        np.testing.assert_array_equal(self.model.predict(changed),self.model.predict(self.data))
        other = fit_count(CONFIGS[3],changed.iloc[:300],self.data.readmitted_30d.iloc[:300],42)
        np.testing.assert_array_equal(other.predict(self.data),self.model.predict(self.data))

    def test_training_categories_imputers_and_unknown_basis(self):
        train = self.data.iloc[:300].copy()
        train.loc[train.index[:10],COUNTS[0]] = np.nan
        design = CountDesign('B').fit(train)
        np.testing.assert_allclose(design.imputer_.statistics_,train[list(COUNTS)].median())
        for i,col in enumerate(COUNTS):
            np.testing.assert_array_equal(design.encoder_.categories_[i],np.unique(train[col].fillna(train[col].median())))
        raw_imp = design.raw_.named_transformers_['numeric'].named_steps['impute']
        nums = design.raw_.transformers_[0][2]
        self.assertAlmostEqual(raw_imp.statistics_[nums.index('age')],train.age.median())
        before = pickle.dumps(design)
        held = self.data.iloc[300:].copy()
        held[list(COUNTS)] = 999
        held['region'] = 'NEVER_SEEN'
        result = design.transform(held)
        np.testing.assert_array_equal(result[:,-sum(len(c) for c in design.encoder_.categories_):],0.)
        self.assertEqual(before,pickle.dumps(design))

    def test_missing_unseen_predictions_and_serialization(self):
        held = self.data.iloc[300:].copy()
        held[list(COUNTS)+LABS] = np.nan
        held['region'] = 'NEW'
        p = self.model.predict(held)
        self.assertTrue(np.isfinite(p).all() and ((p>0)&(p<1)).all())
        np.testing.assert_allclose(p[:1],self.model.predict(held.iloc[:1]),atol=1e-14)
        restored = pickle.loads(pickle.dumps(self.model))
        self.assertEqual(type(restored).__module__,'nexus.count_logistic')
        self.assertEqual(restored.config,CONFIGS[3])
        self.assertGreater(restored.iterations,0)
        np.testing.assert_array_equal(p,restored.predict(held))

    def test_retains_raw_design_and_linear_control(self):
        train = self.data.iloc[:300]
        raw = linear_preprocessor(engineer(train)).fit(engineer(train))
        expected = raw.transform(engineer(train))
        np.testing.assert_array_equal(self.model.preprocessor.transform(train)[:,:expected.shape[1]],expected)
        control = fit_count(CONTROLS[0],train,train.readmitted_30d,42)
        with threadpool_limits(limits=2):
            baseline = LogisticRegression(C=.01,solver='lbfgs',l1_ratio=0.,max_iter=4000,random_state=42).fit(expected,train.readmitted_30d)
        np.testing.assert_allclose(control.predict(train),baseline.predict_proba(expected)[:,1],atol=1e-14)

    def test_exact_submission_and_probability_rejection(self):
        ids = pd.Series(['b','a','c'])
        order = pd.Series(['c','b','a'])
        result = make_submission(ids,order,[.1,.2,.3])
        np.testing.assert_array_equal(result.patient_id,order)
        np.testing.assert_array_equal(result.readmitted_30d,[.3,.1,.2])
        for invalid in ([0.,.2,.3],[np.nan,.2,.3],[1.,.2,.3]):
            with self.assertRaises(ValueError):
                make_submission(ids,order,invalid)
        with self.assertRaises(ValueError):
            make_submission(ids,pd.Series(['a','a','b']),[.1,.2,.3])
        te = pd.read_csv('test.csv'); sample = pd.read_csv('sample_submission.csv')
        full = make_submission(te.patient_id,sample.patient_id,np.full(3000,.2))
        self.assertEqual(len(full),3000)
        np.testing.assert_array_equal(full.patient_id,sample.patient_id)


if __name__ == '__main__':
    unittest.main()
