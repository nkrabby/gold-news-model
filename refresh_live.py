"""Incremental, keyless refresh for the hosted dashboard and standalone export."""
from pathlib import Path
from datetime import datetime, timezone
import json, subprocess, sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fetch_calendar import fetch

def run(*args):
    print('+',' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),cwd=ROOT,check=True)

def latest_gold_context():
    p=ROOT/'data/raw/xauusd_m5.csv'
    if not p.exists():return {}
    # Reading only the timestamp/close columns keeps the scheduled job lightweight.
    prices=pd.read_csv(p,usecols=['timestamp','close'])
    prices['timestamp']=pd.to_datetime(prices.timestamp,unit='ms',utc=True)
    prices=prices.sort_values('timestamp')
    latest=prices.iloc[-1];target=latest.timestamp-pd.Timedelta(hours=24)
    pos=prices.timestamp.searchsorted(target,side='left')-1
    previous=prices.close.iloc[max(0,pos)]
    return {'price_timestamp':latest.timestamp.isoformat(),'last_bid':float(latest.close),
            'pre_return_24h':float(np.log(latest.close/previous))}

def main():
    now=pd.Timestamp.utcnow();start='2020-01-01'
    price_path=ROOT/'data/raw/xauusd_m5.csv'
    if price_path.exists():
        existing=pd.read_csv(price_path,usecols=['timestamp'])
        last=pd.to_datetime(existing.timestamp.iloc[-1],unit='ms',utc=True)
        increment_start=(last-pd.Timedelta(days=3)).strftime('%Y-%m-%d')
    else:
        existing=None;increment_start=start
    end=(now+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    run('npx','--yes','dukascopy-node','-i','xauusd','-from',increment_start,'-to',end,'-t','m5','-f','csv')
    downloads=list((ROOT/'download').glob('xauusd-m5-bid-*.csv'))
    if not downloads:raise RuntimeError('Dukascopy produced no CSV')
    new=pd.read_csv(max(downloads,key=lambda p:p.stat().st_mtime))
    if existing is not None:
        old=pd.read_csv(price_path)
        new=pd.concat([old,new]).drop_duplicates('timestamp').sort_values('timestamp')
    new.to_csv(price_path,index=False)
    for p in downloads:p.unlink()
    fetch(start,(now+pd.Timedelta(days=90)).strftime('%Y-%m-%d'),ROOT/'data/raw/us_calendar.csv')
    run(sys.executable,'src/fetch_forexfactory.py','--recent')
    from market_context import download_market,latest as latest_market
    download_market('2019-01-01',end)
    run(sys.executable,'src/build_dataset.py')
    run(sys.executable,'src/train.py')
    run(sys.executable,'src/pre_release.py')
    run(sys.executable,'src/walk_forward.py')
    run(sys.executable,'src/calibration.py')
    run(sys.executable,'src/upcoming_research.py')
    try:
        from live_gold_feed import download as download_paxg,latest_quotes
        download_paxg('2020-01-01')
        (ROOT/'outputs/live_quotes.json').write_text(json.dumps(latest_quotes(),indent=2,default=str))
    except Exception as e:print('Live gold feed unavailable:',e)
    context=latest_gold_context()
    market=latest_market()
    try:
        from chart_features import build as build_chart
        build_chart()
    except Exception as e:print('chart feature rebuild failed:',e)
    status={'generated_at':datetime.now(timezone.utc).isoformat(),**context,
            'calendar_last_updated':datetime.now(timezone.utc).isoformat(),
            'research_last_updated':datetime.now(timezone.utc).isoformat(),
            'source_note':'Free public sources; timestamps may lag and scheduled jobs can be delayed.'}
    (ROOT/'outputs/live_status.json').write_text(json.dumps(status,indent=2))
    # Make the current market trend available as a dashboard default.
    upcoming_path=ROOT/'data/processed/upcoming_events.csv'
    upcoming=pd.read_csv(upcoming_path)
    upcoming['suggested_pre_return_24h']=context.get('pre_return_24h',0)
    upcoming.to_csv(upcoming_path,index=False)
    run(sys.executable,'src/research_intelligence.py')
    run(sys.executable,'src/forecast_log.py')
    run(sys.executable,'src/score_forecasts.py')
    run(sys.executable,'src/export_standalone.py')
    print(json.dumps(status,indent=2))

if __name__=='__main__':main()
