"""Package source plus small evidence artifacts; never includes environments or caches."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--destination',default='deliverables')
    args=p.parse_args();run=Path(args.run);dest=Path(args.destination);dest.mkdir(exist_ok=True)
    required=['submission_recommended.csv','TRUST_CARD.md','submission_notebook.ipynb','comparison.csv']
    for name in required:
        assert (run/name).exists(), f'Missing deliverable {name}'
        shutil.copy2(run/name,dest/name)
    shutil.copy2('submission_v4.csv',dest/'submission_historical_v4.csv')
    reference=run/'submission_reference.csv'
    shutil.copy2(reference if reference.exists() else Path('submission_v4.csv'),dest/'submission_reference.csv')
    diagnostics=run/'report' if (run/'report').exists() else run/'component_diagnostics'
    if diagnostics.exists():
        shutil.copytree(diagnostics,dest/'report',dirs_exist_ok=True)
    include=[Path(p) for p in ['README.md','requirements-lock.txt','requirements-research.txt','requirements-neural.txt',
                              'requirements.txt','train.csv','test.csv','sample_submission.csv','submission_v4.csv','data_dictionary.csv','best_params_optuna.json']]
    include+=list(Path('nexus').glob('*.py'))+list(Path('scripts').glob('*.py'))+list(Path('scripts').glob('*.ps1'))
    include+=list(Path('tests').glob('*.py'))+list(Path('configs').glob('*.json'))
    include+=[p for p in run.rglob('*') if p.is_file() and not {'cache','models'}&set(p.relative_to(run).parts) and p.suffix!='.tmp']
    for source in json.loads((run/'summary.json').read_text()).get('source_runs',[]):
        source=Path(source)
        include+=[p for p in source.rglob('*') if p.is_file() and not {'cache','models'}&set(p.relative_to(source).parts) and p.suffix!='.tmp']
    if Path('runs/review').exists():include+=list(Path('runs/review').glob('*.*'))
    archive=dest/'nexus_submission_bundle.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(include)):
            z.write(path,path.as_posix())
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file() and p.name!='checksums.json'}
    (dest/'checksums.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print('Packaged',archive,'bytes',archive.stat().st_size)
