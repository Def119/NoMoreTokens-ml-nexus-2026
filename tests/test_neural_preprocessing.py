"""Optional neural preprocessing checks do not allocate GPU memory."""
import importlib.util
import unittest
import numpy as np
import pandas as pd


@unittest.skipUnless(importlib.util.find_spec('tabm'), 'Optional TabM not installed')
class NeuralPreprocessingTests(unittest.TestCase):
    def test_training_statistics_unknown_categories_and_label_isolation(self):
        from nexus.neural import NeuralPreprocessor
        tr=pd.read_csv('train.csv').iloc[:100]
        prep=NeuralPreprocessor().fit(tr)
        held=tr.iloc[:4].copy();held['region']='UNSEEN_REGION';held['creatinine_mg_dl']=np.nan
        before=prep.imputer.statistics_.copy()
        num,cat=prep.transform(held)
        self.assertTrue(np.isfinite(num).all())
        self.assertTrue((cat>=0).all())
        for col,card in enumerate(prep.cardinalities):self.assertTrue((cat[:,col]<card).all())
        held['readmitted_30d']=1-held.readmitted_30d;held['patient_id']=-1
        n2,c2=prep.transform(held)
        np.testing.assert_array_equal(num,n2);np.testing.assert_array_equal(cat,c2)
        np.testing.assert_array_equal(before,prep.imputer.statistics_)


if __name__=='__main__':unittest.main()
