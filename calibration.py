"""How to read the probability: out-of-fold calibration by confidence bucket."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd, numpy as np
from train import pipeline, CATEGORICAL, HORIZON_CONFIG, MODEL_EVENTS, apply_platt, fit_platt
ROOT = Path(__file__).resolve().parents[1]

def run():
    d = pd.read_csv(ROOT/'data/processed/event_study.csv', parse_dates=['release_time'])
    d = d[d.event_type.isin(MODEL_EVENTS)]
    rows = []
    for year in [2023, 2024, 2025, 2026]:
        s = pd.Timestamp(f'{year}-01-01', tz='UTC'); e = pd.Timestamp(f'{year+1}-01-01', tz='UTC')
        for h in ['1h', '4h', '24h']:
            x = d.dropna(subset=[f'return_{h}', 'signed_surprise_z'])
            tr, te = x[x.release_time < s], x[(x.release_time >= s) & (x.release_time < e)]
            if len(tr) < 100 or len(te) < 15: continue
            y = te[f'return_{h}'].gt(0)
            if y.nunique() < 2: continue
            cfg = HORIZON_CONFIG[h]; cols = [c for c in cfg['numeric'] if c in x.columns]
            m = pipeline('direction', cols, cfg['C']); m.fit(tr[cols+CATEGORICAL], tr[f'return_{h}'].gt(0).astype(int))
            cal = fit_platt(tr, cols, h, cfg)
            p = apply_platt(m.predict_proba(te[cols+CATEGORICAL])[:, 1], cal)
            rows += [{'year': year, 'horizon': h, 'p': pi, 'up': int(yi)} for pi, yi in zip(p, y)]
    o = pd.DataFrame(rows)
    out = []
    for h in ['1h', '4h', '24h']:
        x = o[o.horizon == h].copy()
        base = x.up.mean()
        brier = ((x.p - x.up) ** 2).mean()
        brier_base = (base * (1 - base))
        x['bucket'] = pd.cut(x.p, [0, .4, .5, .6, .7, .8, 1.0])
        g = x.groupby('bucket', observed=True).agg(
            n=('up', 'size'), predicted=('p', 'mean'), actual=('up', 'mean')).reset_index()
        g['horizon'] = h; g['base_rate'] = base
        g['brier_skill_vs_base_rate'] = 1 - brier / brier_base
        out.append(g)
    res = pd.concat(out)
    res['predicted'] = (res.predicted * 100).round(1)
    res['actual'] = (res.actual * 100).round(1)
    res['base_rate'] = (res.base_rate * 100).round(1)
    res['bucket'] = res.bucket.astype(str)
    res.to_csv(ROOT/'outputs/calibration.csv', index=False)
    print(res.to_string(index=False))
    return res

if __name__ == '__main__':
    run()
