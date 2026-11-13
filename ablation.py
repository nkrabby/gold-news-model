"""Ablation: does each extra feed earn its place? All point-in-time, no in-progress bars."""
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score
from train import pipeline,CATEGORICAL,HORIZONS
from market_context import FEATURES as MKT
from live_gold_feed import FEATURES as PAXG
ROOT=Path(__file__).resolve().parents[1]
BASE=['signed_surprise_z','abs_surprise_z','pre_return_24h','pre_volatility_24h','release_hour_sin','release_hour_cos']
SETS={'base (surprise+gold trend)':BASE,
      'base + macro (DXY/oil/vol/yields)':BASE+MKT,
      'base + macro + 24/7 PAXG gold':BASE+MKT+PAXG,
      'base + 24/7 PAXG gold only':BASE+PAXG}
def run():
    d=pd.read_csv(ROOT/'data/processed/event_study.csv',parse_dates=['release_time']);rows=[]
    for year in [2023,2024,2025,2026]:
        s=pd.Timestamp(f'{year}-01-01',tz='UTC');e=pd.Timestamp(f'{year+1}-01-01',tz='UTC')
        for h in HORIZONS:
            x=d.dropna(subset=[f'return_{h}','signed_surprise_z']);tr=x[x.release_time<s];te=x[(x.release_time>=s)&(x.release_time<e)]
            if len(tr)<100 or len(te)<15:continue
            y=te[f'return_{h}'].gt(0)
            if y.nunique()<2:continue
            for name,cols in SETS.items():
                cols=[c for c in cols if c in x]
                m=pipeline('direction',cols);m.fit(tr[cols+CATEGORICAL],tr[f'return_{h}'].gt(0).astype(int))
                p=m.predict_proba(te[cols+CATEGORICAL])[:,1]
                rows.append({'fold':year,'horizon':h,'features':name,'auc':roc_auc_score(y,p),'accuracy':accuracy_score(y,p>=.5),'majority':max(y.mean(),1-y.mean())})
    out=pd.DataFrame(rows);out.to_csv(ROOT/'outputs/ablation_feed_value.csv',index=False)
    piv=out.groupby(['features','horizon'])[['auc','accuracy','majority']].mean().round(4)
    print(piv.to_string())
    print('\nMean AUC by feature set:',out.groupby('features').auc.mean().round(4).to_dict())
if __name__=='__main__':run()
