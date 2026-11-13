"""24/7 tokenized-gold feed (PAXG) with cross-venue price check.

PAXG is a proxy, not XAUUSD: it trades around the clock, is thinner, and carries a
premium/discount versus spot. It is used only for point-in-time context features.
"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import requests
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/raw/market/paxg_h1.csv'
SYMBOLS={'binance':'https://data-api.binance.vision/api/v3/klines?symbol=PAXGUSDT&interval=1h',
         'kraken':'https://api.kraken.com/0/public/OHLC?pair=PAXGUSD&interval=60',
         'coinbase':'https://api.exchange.coinbase.com/products/PAXG-USD/candles?granularity=3600',
         'okx':'https://www.okx.com/api/v5/market/history-candles?instId=PAXG-USDT&bar=1H'}
FEATURES=['paxg_return_24h','paxg_return_6h','paxg_vol_24h','xau_paxg_premium']

def _binance(session,start_ms):
    rows=[];end=int(time.time()*1000)
    while True:
        r=session.get(SYMBOLS['binance'],params={'startTime':start_ms,'endTime':end,'limit':1000},timeout=20)
        r.raise_for_status();batch=r.json()
        if not batch:break
        rows+=[[int(k[0])/1000,float(k[1]),float(k[2]),float(k[3]),float(k[4])] for k in batch]
        nxt=batch[-1][0]+3_600_000
        if nxt>=end or len(batch)<1000:break
        start_ms=nxt
    return pd.DataFrame(rows,columns=['ts','open','high','low','close'])

def download(start='2020-01-01'):
    s=requests.Session();s.headers['User-Agent']='gold-news-model/1.0'
    start_ms=int(pd.Timestamp(start,tz='UTC').timestamp()*1000)
    df=_binance(s,start_ms)
    if df.empty:raise RuntimeError('No PAXG data returned')
    df['timestamp']=pd.to_datetime(df.ts,unit='s',utc=True)
    df=df.drop(columns=['ts',]).drop_duplicates('timestamp').sort_values('timestamp')
    OUT.parent.mkdir(parents=True,exist_ok=True);df.to_csv(OUT,index=False)
    print(f'PAXG {len(df)} hourly rows {df.timestamp.min()} -> {df.timestamp.max()}')
    return df

def latest_quotes():
    """Near-live price across venues; a disagreement check, not a tradeable quote."""
    out={}
    try:
        r=requests.get(SYMBOLS['binance']+'&limit=1',timeout=10);k=r.json()[-1]
        out['binance']={'price':float(k[4]),'timestamp':pd.to_datetime(int(k[0]),unit='ms',utc=True)}
    except Exception as e:out['binance']={'error':str(e)[:90]}
    try:
        r=requests.get(SYMBOLS['kraken'],timeout=10);k=list(r.json()['result'].values())[0][-1]
        out['kraken']={'price':float(k[4]),'timestamp':pd.to_datetime(int(k[0]),unit='s',utc=True)}
    except Exception as e:out['kraken']={'error':str(e)[:90]}
    try:
        r=requests.get(SYMBOLS['coinbase'],timeout=10);k=r.json()[0]
        out['coinbase']={'price':float(k[4]),'timestamp':pd.to_datetime(int(k[0]),unit='s',utc=True)}
    except Exception as e:out['coinbase']={'error':str(e)[:90]}
    try:
        r=requests.get(SYMBOLS['okx'],params={'instId':'PAXG-USDT','bar':'1H','limit':'1'},timeout=10);k=r.json()['data'][0]
        out['okx']={'price':float(k[4]),'timestamp':pd.to_datetime(int(k[0]),unit='ms',utc=True)}
    except Exception as e:out['okx']={'error':str(e)[:90]}
    prices=[v['price'] for v in out.values() if 'price' in v]
    return {'venues':out,'median':float(np.median(prices)) if prices else None,
            'spread_pct':float((max(prices)-min(prices))/np.median(prices)*100) if len(prices)>1 else None,
            'fetched_at':pd.Timestamp.utcnow().isoformat()}

def features_at(ts,frame=None):
    """Strictly earlier-bar features for a release time."""
    df=pd.read_csv(OUT,parse_dates=['timestamp']) if frame is None else frame
    prior=df[df.timestamp<=ts-pd.Timedelta(hours=1)]  # bar opens must be closed before release
    if len(prior)<25:return {k:np.nan for k in FEATURES}
    c=prior.close
    ret24=c.iloc[-1]/c.iloc[-25]-1;ret6=c.iloc[-1]/c.iloc[-7]-1
    vol=prior.close.pct_change().iloc[-24:].std()
    return {'paxg_return_24h':ret24,'paxg_return_6h':ret6,'paxg_vol_24h':vol}

def attach(dataset):
    df=pd.read_csv(OUT,parse_dates=['timestamp']);ts=pd.to_datetime(dataset.release_time,utc=True)-pd.Timedelta(hours=1)
    feats=[features_at(t,df) for t in ts]
    for k in ['paxg_return_24h','paxg_return_6h','paxg_vol_24h']:
        dataset[k]=[f.get(k,np.nan) for f in feats]
    # premium: XAUUSD close vs PAXG close at the same hour
    merged=pd.merge_asof(pd.DataFrame({'timestamp':ts}).sort_values('timestamp'),
                         df[['timestamp','close']].rename(columns={'close':'paxg'}),
                         on='timestamp',direction='backward',tolerance=pd.Timedelta('2h'))  # closed bars only
    anchor='pre_close' if 'pre_close' in dataset else 'price_at_release'
    if anchor in dataset:
        dataset['xau_paxg_premium']=(dataset[anchor].values/merged.paxg.values-1)*100
    return dataset

if __name__=='__main__':
    import sys
    if '--quotes' in sys.argv:print(json.dumps(latest_quotes(),indent=2,default=str))
    else:download()
