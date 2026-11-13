"""Append immutable pre-release forecasts and fill outcomes when available."""
from pathlib import Path
import json,sys
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pre_release import forecast

def update():
    history=pd.read_csv(ROOT/'data/processed/event_study.csv',parse_dates=['release_time'])
    upcoming=pd.read_csv(ROOT/'data/processed/upcoming_events.csv',parse_dates=['release_time'])
    pre=joblib.load(ROOT/'outputs/pre_release_models.joblib');reaction=joblib.load(ROOT/'outputs/models.joblib')
    context=json.loads((ROOT/'outputs/market_context.json').read_text())
    # Live 5-minute chart features for the forecast row (last fully closed bar).
    try:
        from chart_features import features_at, FEATURES as CH
        live=features_at([pd.Timestamp.utcnow()]).iloc[0]
        context.update({c:float(live[c]) for c in ['ret_1h','rsi_14','range_pos_24h','vol_ratio']
                        if c in live.index and pd.notna(live[c])})
        context.update({'ch_'+k:context.pop(k) for k in ['ret_1h','rsi_14','range_pos_24h','vol_ratio'] if k in context})
    except Exception as e:print('chart context unavailable:',e)
    research=json.loads((ROOT/'outputs/research_intelligence.json').read_text())
    now=pd.Timestamp.utcnow();rows=[]
    for _,u in upcoming.iterrows():
        event=u.event_type;hist=history[history.event_type.eq(event)]
        if hist.empty:continue
        forecast_value=u.forecast if pd.notna(u.forecast) else u.previous
        previous=u.previous if pd.notna(u.previous) else hist.iloc[-1].previous
        if pd.isna(forecast_value) or pd.isna(previous):continue
        hour=u.release_time.hour+u.release_time.minute/60
        result=forecast(pre,reaction,history,event,float(forecast_value),float(previous),
                        float(u.get('suggested_pre_return_24h',0) or 0),hour,0,'1h',context=context)
        rows.append({'forecast_id':now.isoformat()+'|'+event+'|'+str(u.release_time),
                     'forecast_time':now.isoformat(),'event_type':event,'release_time':u.release_time,
                     'horizon':'1h','direction':'Bullish' if result['probability_up']>=.5 else 'Bearish',
                     'probability_up':result['probability_up'],'expected_return':result['expected_return'],
                     'lower_80':result['lower_80'],'upper_80':result['upper_80'],'consensus':forecast_value,
                     'expected_actual':result['expected_actual'],'probability_beat':result['probability_beat'],
                     'research_tilt':0,'research_headline_signal':research.get(event,{}).get('signal'),
                     **{k:context.get(k) for k in ['dxy_return_24h','oil_return_24h','dgs2_change_5d','dgs10_change_5d','real_yield_change_5d']}})
    path=ROOT/'outputs/forecast_log.csv'
    old=pd.read_csv(path) if path.exists() else pd.DataFrame()
    log=pd.concat([old,pd.DataFrame(rows)],ignore_index=True).drop_duplicates('forecast_id',keep='first')
    # Outcomes are owned exclusively by src/score_forecasts.py, which refuses to
    # write a horizon that has not fully elapsed. Importing them here allowed a
    # partially formed return to enter the log before scoring ran.
    log.to_csv(path,index=False)
    print(f'Forecast log: {len(log)} snapshots; {log.actual.notna().sum()} with outcomes')
if __name__=='__main__':update()
