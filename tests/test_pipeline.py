import unittest
import numpy as np
import pandas as pd
from nexus.features import engineer, FrameAdapter, linear_preprocessor, CATS, LABS
from nexus.models import fit_model
from nexus.ensemble import Combiner


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=pd.read_csv('train.csv').iloc[:300].copy()

    def test_features_are_row_local_and_ignore_labels(self):
        a=engineer(self.data,'engineered')
        changed=self.data.copy();changed['readmitted_30d']=1-changed.readmitted_30d
        pd.testing.assert_frame_equal(a,engineer(changed,'engineered'))
        pd.testing.assert_frame_equal(a.iloc[:1],engineer(self.data.iloc[:1],'engineered'))
        self.assertEqual(len(a.columns),54)
        self.assertNotIn('patient_id',a)

    def test_thresholds_use_correct_lab_columns(self):
        x=self.data.iloc[:1].copy();x['sodium_mmol_l']=140;x['systolic_bp_mmhg']=170
        f=engineer(x,'engineered')
        self.assertEqual(f.hypertension_urgency.iloc[0],1)
        self.assertEqual(f.hyponatremia.iloc[0],0)

    def test_fitted_categories_do_not_learn_validation(self):
        adapter=FrameAdapter().fit(self.data)
        before={k:v.copy() for k,v in adapter.categories_.items()}
        held=self.data.iloc[:2].copy();held['region']='NEVER_SEEN'
        transformed=adapter.transform(held)
        self.assertTrue(transformed.region.isna().all())
        self.assertEqual(adapter.categories_,before)

    def test_missing_and_unseen_values_predict(self):
        cfg=dict(name='test',family='spline',mode='raw',params=dict(C=.1,knots=3,l1_ratio=0.))
        model=fit_model(cfg,self.data,self.data.readmitted_30d,threads=1)
        held=self.data.iloc[:5].copy()
        for c in LABS:held[c]=np.nan
        held['region']='NEVER_SEEN'
        p=model.predict(held)
        self.assertTrue(np.isfinite(p).all());self.assertTrue(((p>0)&(p<1)).all())

    def test_validation_labels_cannot_affect_fixed_refit(self):
        cfg=dict(name='test',family='lr',mode='raw',params=dict(C=.1,l1_ratio=0.))
        model=fit_model(cfg,self.data.iloc[:200],self.data.readmitted_30d.iloc[:200],threads=1)
        held=self.data.iloc[200:].copy();a=model.predict(held)
        held['readmitted_30d']=1-held.readmitted_30d
        np.testing.assert_array_equal(a,model.predict(held))

    def test_imputer_statistics_are_training_only(self):
        cfg=dict(name='test',family='lr',mode='raw',params=dict(C=.1,l1_ratio=0.))
        model=fit_model(cfg,self.data.iloc[:200],self.data.readmitted_30d.iloc[:200],threads=1)
        imputer=model.preprocessor.named_transformers_['numeric'].named_steps['impute']
        before=imputer.statistics_.copy()
        held=self.data.iloc[200:].copy();held['age']=99999
        model.predict(held)
        np.testing.assert_array_equal(before,imputer.statistics_)
        cols=[c for c in engineer(self.data) if c not in CATS]
        self.assertAlmostEqual(before[cols.index('age')],self.data.age.iloc[:200].median())

    def test_convex_weights_and_model_serialization(self):
        import pickle
        rng=np.random.default_rng(42);p=rng.uniform(.01,.7,(100,4));y=rng.binomial(1,.2,100)
        model=Combiner('convex',True).fit(p,y)
        self.assertAlmostEqual(model.weights_.sum(),1)
        self.assertTrue((model.weights_>=0).all())
        np.testing.assert_array_equal(model.predict(p),pickle.loads(pickle.dumps(model)).predict(p))

if __name__=='__main__':unittest.main()
