"""Prepare and score fixed, prespecified blends from completed OOF campaigns."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs' / 'final50_scaled_v13'
ID = 'patient_id'
TARGET = 'readmitted_30d'


def clip(p):
    return np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)


def row_loss(y, p):
    p = clip(p)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def main():
    train = pd.read_csv(ROOT / 'train.csv')
    sample = pd.read_csv(ROOT / 'sample_submission.csv')
    scaled = pd.read_csv(OUT / 'oof.csv')
    current = pd.read_csv(ROOT / 'runs/final20_v11/oof.csv')
    v6 = pd.read_csv(ROOT / 'oof_v6.csv').rename(columns={'actual': TARGET})
    test_scaled = pd.read_csv(OUT / 'submission_equal_cal.csv')
    test_v6 = pd.read_csv(ROOT / 'submission_v6.csv')
    for frame in (scaled, current, v6):
        if not frame[ID].equals(train[ID]):
            raise ValueError('OOF ID order mismatch')
    if not test_scaled[ID].equals(sample[ID]) or not test_v6[ID].equals(sample[ID]):
        raise ValueError('Test ID order mismatch')

    y = train[TARGET].to_numpy()
    p50 = clip(scaled['equal_cal'])
    p20 = clip(current['equal_cal'])
    pv6 = clip(v6['pred_final'])
    t50 = clip(test_scaled[TARGET])
    tv6 = clip(test_v6[TARGET])
    baseline = row_loss(y, p20)
    candidates = {
        'scaled50_v6_25': (0.75, 0.25),
        'scaled50_v6_30': (0.70, 0.30),
        'scaled50_v6_50': (0.50, 0.50),
    }
    summary = {'baseline_final20_log_loss': float(baseline.mean()), 'candidates': {}}
    rng = np.random.default_rng(20260914)
    for name, (w50, w6) in candidates.items():
        p = clip(w50 * p50 + w6 * pv6)
        delta = row_loss(y, p) - baseline
        draws = np.array([delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(5000)])
        test_p = clip(w50 * t50 + w6 * tv6)
        pd.DataFrame({ID: sample[ID], TARGET: test_p}).to_csv(OUT / f'submission_{name}.csv', index=False)
        summary['candidates'][name] = {
            'w_scaled50': w50,
            'w_v6': w6,
            'oof_log_loss': float(log_loss(y, p)),
            'delta_vs_final20': float(delta.mean()),
            'bootstrap_ci95_delta': np.quantile(draws, [0.025, 0.975]).tolist(),
            'test_mean': float(test_p.mean()),
            'test_std': float(test_p.std()),
        }
    summary['recommended'] = 'scaled50_v6_30'
    (OUT / 'comparison_next.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
