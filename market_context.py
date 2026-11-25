"""Point-in-time market context from Dukascopy and FRED public sources."""
from pathlib import Path
import json, subprocess
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
INSTRUMENTS={"dxy":"dollaridxusd","oil":"brentcmdusd","volatility":"volidxusd"}
FEATURES=["dxy_return_24h","oil_return_24h","volatility_change_24h","dgs2_change_5d","dgs10_change_5d","real_yield_change_5d"]

def download_market(start="2019-01-01", end=None):
    """Reuse saved history and fetch only recent market bars."""
    end = end or (pd.Timestamp.utcnow() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    out = ROOT / "data/raw/market"
    out.mkdir(parents=True, exist_ok=True)

    for name, instrument in INSTRUMENTS.items():
        path = out / f"{name}_h1.csv"
        old = None
        fetch_start = start

        if path.exists():
            old = pd.read_csv(path)
            if not old.empty and "timestamp" in old.columns:
                last = pd.to_datetime(old["timestamp"], unit="ms", utc=True).max()
                fetch_start = max(
                    pd.Timestamp(start, tz="UTC"),
                    last - pd.Timedelta(days=3),
                ).strftime("%Y-%m-%d")

        subprocess.run(
            [
                "npx", "--yes", "dukascopy-node",
                "-i", instrument,
                "-from", fetch_start,
                "-to", end,
                "-t", "h1",
                "-f", "csv",
            ],
            cwd=ROOT,
            check=True,
        )

        found = list((ROOT / "download").glob(f"{instrument}-h1-bid-*.csv"))
        if not found:
            raise RuntimeError(f"No data for {instrument}")

        fresh = pd.read_csv(max(found, key=lambda f: f.stat().st_mtime))
        for file in found:
            file.unlink(missing_ok=True)

        merged = pd.concat([old, fresh], ignore_index=True) if old is not None else fresh
        merged = merged.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
        merged.to_csv(path, index=False)
        print(f"{name}: {len(merged)} bars; downloaded from {fetch_start}")

def load_market():
    frames={}
    for name in INSTRUMENTS:
        p=ROOT/f'data/raw/market/{name}_h1.csv'
        if not p.exists():continue
        x=pd.read_csv(p);x['timestamp']=pd.to_datetime(x.timestamp,unit='ms',utc=True)
        frames[name]=x.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    fred=pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2,DGS10,DFII10')
    fred.columns=['date','dgs2','dgs10','real_yield'];fred['date']=pd.to_datetime(fred.date,utc=True)
    for col in ['dgs2','dgs10','real_yield']:
        fred[col]=pd.to_numeric(fred[col],errors='coerce')
    fred=fred.sort_values('date')
    return frames,fred

def at_or_before(x,t):  # t must already exclude the in-progress bar
    pos=x.timestamp.searchsorted(t,side='left')-1
    return pos if pos>=0 else None

def values_at(frames,fred,t):
    out={}
    for name in ['dxy','oil']:
        if name not in frames:out[f'{name}_return_24h']=np.nan;continue
        x=frames[name];i=at_or_before(x,t-pd.Timedelta(hours=1))
        if i is None:out[f'{name}_return_24h']=np.nan;continue
        j=at_or_before(x,x.timestamp.iloc[i]-pd.Timedelta(hours=24))
        out[f'{name}_return_24h']=np.log(x.close.iloc[i]/x.close.iloc[j]) if j is not None else np.nan
    if 'volatility' in frames:
        x=frames['volatility'];i=at_or_before(x,t-pd.Timedelta(hours=1))
        j=at_or_before(x,x.timestamp.iloc[i]-pd.Timedelta(hours=24)) if i is not None else None
        out['volatility_change_24h']=x.close.iloc[i]-x.close.iloc[j] if i is not None and j is not None else np.nan
    prior=fred[fred.date<t.tz_convert(None).tz_localize('UTC')].dropna(subset=['dgs2','dgs10'],how='all')
    for col,feature in [('dgs2','dgs2_change_5d'),('dgs10','dgs10_change_5d'),('real_yield','real_yield_change_5d')]:
        s=prior[col].dropna();out[feature]=float(s.iloc[-1]-s.iloc[-6]) if len(s)>=6 else np.nan
    return out

def attach(events):
    frames,fred=load_market();values=[values_at(frames,fred,t) for t in events.release_time]
    return pd.concat([events.reset_index(drop=True),pd.DataFrame(values)],axis=1)

def latest():
    frames,fred=load_market();t=max(x.timestamp.max() for x in frames.values())
    values=values_at(frames,fred,t+pd.Timedelta(minutes=1))
    result={'as_of':t.isoformat(),**values}
    for name,x in frames.items():result[name+'_last']=float(x.close.iloc[-1]);result[name+'_timestamp']=x.timestamp.iloc[-1].isoformat()
    for col in ['dgs2','dgs10','real_yield']:
        s=fred[['date',col]].dropna();result[col+'_last']=float(s[col].iloc[-1]);result[col+'_date']=s.date.iloc[-1].date().isoformat()
    (ROOT/'outputs/market_context.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    download_market();print(json.dumps(latest(),indent=2))
