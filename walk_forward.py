"""Expanding-window validation across several chronological folds."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score,roc_auc_score,mean_absolute_error
from train import pipeline,CATEGORICAL,HORIZONS,HORIZON_CONFIG,MODEL_EVENTS
ROOT=Path(__file__).resolve().parents[1]
def run():
    data=pd.read_csv(ROOT/'data/processed/event_study.csv',parse_dates=['release_time'])
    data=data[data.event_type.isin(MODEL_EVENTS)];rows=[]
    for year in [2023,2024,2025,2026]:
        start=pd.Timestamp(f'{year}-01-01',tz='UTC');end=pd.Timestamp(f'{year+1}-01-01',tz='UTC')
        for h in HORIZONS:
            d=data.dropna(subset=[f'return_{h}','signed_surprise_z']);train=d[d.release_time<start]
            test=d[(d.release_time>=start)&(d.release_time<end)]
            if len(train)<100 or len(test)<15 or test[f'return_{h}'].gt(0).nunique()<2:continue
            cfg=HORIZON_CONFIG[h];cols=[c for c in cfg['numeric'] if c in d.columns]
            m=pipeline('direction',cols,cfg['C']);m.fit(train[cols+CATEGORICAL],train[f'return_{h}'].gt(0).astype(int))
            p=m.predict_proba(test[cols+CATEGORICAL])[:,1];y=test[f'return_{h}'].gt(0)
            r=pipeline('return',cols);r.fit(train[cols+CATEGORICAL],train[f'return_{h}'])
            rows.append({'fold':year,'horizon':h,'train_events':len(train),'test_events':len(test),
                         'accuracy':accuracy_score(y,p>=.5),'majority_accuracy':max(y.mean(),1-y.mean()),
                         'auc':roc_auc_score(y,p),'mae_bps':mean_absolute_error(test[f'return_{h}'],r.predict(test[cols+CATEGORICAL]))*10000})
    out=pd.DataFrame(rows);out.to_csv(ROOT/'outputs/walk_forward_metrics.csv',index=False)
    print(out.groupby('horizon')[['accuracy','majority_accuracy','auc','mae_bps']].mean())
if __name__=='__main__':run()
