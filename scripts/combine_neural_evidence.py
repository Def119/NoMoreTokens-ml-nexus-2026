"""Join compatible nested base/neural runs without learning from outer labels."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
import joblib
import pandas as pd

# Direct script execution otherwise exposes only scripts/ to pickle imports.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--neural',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();base=Path(a.base);neural=Path(a.neural);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'summary.json').exists(), 'Use a fresh combined output directory'
    (out/'models').mkdir(exist_ok=True)
    bs=json.loads((base/'summary.json').read_text());ns=json.loads((neural/'summary.json').read_text())
    nm=json.loads((neural/'manifest.json').read_text())
    assert bs.get('complete') and ns.get('complete'), 'Both runs must be complete'
    assert nm['source_manifest_sha256']==digest(base/'manifest.json'), 'Neural run used a different base campaign'
    for filename,expected in nm['base_artifacts'].items():
        assert digest(base/filename)==expected, f'Base artifact differs from neural provenance: {filename}'
    bo=pd.read_csv(base/'oof.csv');no=pd.read_csv(neural/'oof.csv')
    assert bo.patient_id.equals(no.patient_id) and bo.fold.equals(no.fold)
    assert bo.readmitted_30d.equals(no.readmitted_30d)
    for c in no:
        if c.startswith('tabm'):bo[c]=no[c]
    bo.to_csv(out/'oof.csv',index=False)
    bs['metrics'].update(ns['metrics']);bs['winner']=min(bs['metrics'],key=lambda n:bs['metrics'][n]['log_loss'])
    for f in range(len(bs['folds'])):
        bs['folds'][f]['outer_metrics'].update(ns['folds'][f]['outer_metrics'])
        bs['folds'][f]['neural_selection']=ns['folds'][f]
        b=joblib.load(base/'models'/f'fold_{f}.joblib');n=joblib.load(neural/'models'/f'fold_{f}.joblib')
        b.update(n);joblib.dump(b,out/'models'/f'fold_{f}.joblib')
    bs['source_runs']=[str(base),str(neural)]
    (out/'summary.json').write_text(json.dumps(bs,indent=2),encoding='utf-8')
    for source in [base,neural]:
        for file in source.glob('submission*.csv'):shutil.copy2(file,out/file.name)
    shutil.copy2(out/f'submission_{bs["winner"]}.csv',out/'submission_recommended.csv')
    for file in ['manifest.json','backend.json','folds.csv','resolved_config.json']:
        shutil.copy2(base/file,out/file)
    shutil.copy2(neural/'manifest.json',out/'neural_manifest.json')
    pd.DataFrame(bs['metrics']).T.sort_values('log_loss').to_csv(out/'comparison.csv',index_label='method')
    print('Combined nested recipe winner:',bs['winner'],bs['metrics'][bs['winner']])
