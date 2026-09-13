"""Summarize completed runs without fitting or adapting to leaderboard scores."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def review(paths, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];capacity=[]
    for path in paths:
        run=Path(path)
        if not (run/'summary.json').exists():continue
        summary=json.loads((run/'summary.json').read_text())
        if not summary.get('complete'):continue
        manifest=json.loads((run/'manifest.json').read_text())
        for name,metric in summary['metrics'].items():
            folds=[f['outer_metrics'][name]['log_loss'] for f in summary['folds']]
            rows.append(dict(run=str(run),seed=manifest['seed'],method=name,**metric,
                             fold_sd=float(np.std(folds,ddof=1)),fold_min=min(folds),fold_max=max(folds),
                             delta_vs_same_run_reference=metric['log_loss']-summary['metrics']['reference']['log_loss']
                             if 'reference' in summary['metrics'] else None))
        for fold in summary['folds']:
            for name,loss in fold.get('inner_scores',{}).items():
                train=fold.get('inner_train_losses',{}).get(name)
                capacity.append(dict(run=str(run),fold=fold['fold'],candidate=name,inner_logloss=loss,
                                     train_logloss=train,gap=loss-train if train is not None else None))
    frame=pd.DataFrame(rows).sort_values('log_loss')
    frame.to_csv(output/'campaign_comparison.csv',index=False)
    pd.DataFrame(capacity).to_csv(output/'capacity_diagnostics.csv',index=False)
    winners=frame.sort_values('log_loss').groupby('run',sort=False).head(1)
    text=['# Completed experiment comparison',
          'Lower log loss is better. Runs with different seeds have different held-out partitions; compare each improvement with its own reference. Repeated use of these labels and selection among outer results introduce optimism. These are local validation scores, not Kaggle scores.',
          '```csv',winners.to_csv(index=False).strip(),'```',
          'Training/inner-validation gaps are descriptive: inner scores also guide tuning and are not final evaluation. Learning curves and paired conditional bootstrap intervals are in each run report. Increasing model capacity is justified only if held-out performance supports it.']
    (output/'EXPERIMENTS.md').write_text('\n\n'.join(text),encoding='utf-8')
    print(winners.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--output',default='runs/review')
    a=p.parse_args();review(a.runs,a.output)
