import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('submission');args=p.parse_args()
    path=Path(args.submission);df=pd.read_csv(path);sample=pd.read_csv('sample_submission.csv')
    assert df.columns.tolist()==sample.columns.tolist(), 'Wrong columns/order'
    assert len(df)==len(sample)==3000, 'Wrong row count'
    assert df.patient_id.is_unique and df.patient_id.equals(sample.patient_id), 'Wrong IDs/order'
    assert np.isfinite(df.readmitted_30d).all(), 'Missing/nonfinite predictions'
    assert df.readmitted_30d.between(0,1,inclusive='neither').all(), 'Probabilities must lie strictly between zero and one'
    print('VALID',path)
    print('Rows:',len(df),'Mean:',df.readmitted_30d.mean(),'Range:',df.readmitted_30d.min(),df.readmitted_30d.max())
    print('SHA256:',hashlib.sha256(path.read_bytes()).hexdigest())
