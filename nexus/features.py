"""Row-local features and training-only preprocessing."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer

TARGET = 'readmitted_30d'
ID = 'patient_id'
CATS = ['sex', 'rurality', 'hospital_type', 'region', 'discharge_disposition', 'care_pathway']
LABS = ['hemoglobin_g_dl', 'creatinine_mg_dl', 'sodium_mmol_l', 'heart_rate_bpm', 'systolic_bp_mmhg', 'followup_days']


def engineer(df, mode='raw'):
    """No fitted statistics, labels, IDs, or information from other rows."""
    x = df.drop(columns=[TARGET, ID], errors='ignore').copy()
    if mode == 'raw':
        return x
    for c in LABS:
        x['missing_' + c] = x[c].isna().astype('int8')
    x['total_missing_labs'] = x[['missing_' + c for c in LABS]].sum(axis=1)
    hb, cr, na, hr, sbp = (x[c] for c in LABS[:5])
    x['anemia_flag'] = np.where(hb.isna(), np.nan, hb < np.where(x.sex.eq('Male'), 13, 12)).astype(float)
    x['renal_impairment'] = np.where(cr.isna(), np.nan, cr > 1.3)
    x['renal_ckd_interaction'] = x.renal_impairment * x.chronic_kidney_disease
    for name, cutoff in [('hypertension_stage1',130),('hypertension_stage2',140),('hypertension_urgency',160)]:
        x[name] = np.where(sbp.isna(), np.nan, sbp >= cutoff)
    x['tachycardia'] = np.where(hr.isna(), np.nan, hr > 100)
    x['bradycardia'] = np.where(hr.isna(), np.nan, hr < 60)
    x['vitals_instability'] = np.where(hr.isna(), np.nan, (hr > 100) | (hr < 60))
    x['hyponatremia'] = np.where(na.isna(), np.nan, na < 135)
    x['total_bed_days'] = x.prior_admissions_12m * x.length_of_stay_days
    x['comorbidity_per_age'] = x.comorbidity_count / (x.age + 1e-5)
    x['meds_per_condition'] = x.medication_count / (x.comorbidity_count + 1)
    x['access_barrier'] = x.missed_appointments_12m * x.rurality.map({'Urban':0,'Semi-urban':1,'Rural':2}).fillna(1)
    x['missed_x_socioeconomic'] = x.missed_appointments_12m * x.socioeconomic_index
    x['discharge_vulnerability'] = x.discharge_disposition.eq('Home_with_support') * (1-x.socioeconomic_index.clip(-2,2)/2)
    x['chronic_burden'] = x[['diabetes','hypertension','chronic_kidney_disease','heart_failure']].sum(axis=1)
    x['age_bucket'] = pd.cut(x.age, [0,40,50,60,70,80,120], labels=False, right=False).astype(float)
    x['has_prior_admission'] = x.prior_admissions_12m.gt(0).astype('int8')
    x['readmission_frequency'] = x.prior_admissions_12m / (x.age + 1e-5)
    x['meds_per_los'] = x.medication_count / (x.length_of_stay_days + 1e-5)
    x['followup_missing_or_long'] = (x.followup_days.isna() | x.followup_days.gt(30)).astype('int8')
    x['abnormal_vitals_count'] = x[['tachycardia','bradycardia','hypertension_stage1','hyponatremia','anemia_flag','renal_impairment']].fillna(0).sum(axis=1)
    x['los_x_comorbidity'] = x.length_of_stay_days * x.comorbidity_count
    return x


class FrameAdapter(BaseEstimator, TransformerMixin):
    def __init__(self, family='lgb', mode='raw'):
        self.family = family
        self.mode = mode

    def fit(self, X, y=None):
        x = engineer(X, self.mode)
        self.categories_ = {c: sorted(x[c].dropna().astype(str).unique()) for c in CATS}
        self.feature_names_in_ = np.asarray(x.columns)
        return self

    def transform(self, X):
        x = engineer(X, self.mode)
        for c in CATS:
            if self.family == 'lgb':
                x[c] = pd.Categorical(x[c].where(x[c].isin(self.categories_[c])), categories=self.categories_[c])
            else:
                x[c] = x[c].fillna('__MISSING__').astype(str)
        return x


def linear_preprocessor(x, spline=False, knots=4, legacy=False):
    nums = [c for c in x if c not in CATS]
    # Only continuous raw variables get splines. Binary/count variables remain linear.
    continuous = [c for c in nums if c in ['age','socioeconomic_index','length_of_stay_days'] + LABS]
    numeric = Pipeline([('impute', SimpleImputer(strategy='constant' if legacy else 'median', fill_value=-999,
                                                add_indicator=not legacy, keep_empty_features=True)),
                        ('scale', StandardScaler())])
    transforms = [('numeric', numeric, [c for c in nums if not spline or c not in continuous]),
                  ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first' if legacy else None), CATS)]
    if spline:
        transforms.append(('smooth', Pipeline([
            ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('spline', SplineTransformer(n_knots=knots, degree=3, knots='quantile', extrapolation='linear', include_bias=False)),
            ('scale', StandardScaler())]), continuous))
        # Explicit missingness columns remain separate from spline coordinates.
        from sklearn.impute import MissingIndicator
        transforms.append(('missing_flags', MissingIndicator(features='all'), continuous))
    return ColumnTransformer(transforms, sparse_threshold=0)
