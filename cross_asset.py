"""5-minute cross-asset context: DXY, Brent, VIX, gold-silver ratio.

These are different dimensions, not higher resolution of the gold series itself.
Alignment rule: an hourly bar stamped T covers (T, T+1h], so a release at 13:30 may
only use bars whose close is <= 13:30, i.e. bars stamped <= 12:00.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MARKET = ROOT / 'data/raw/market'
CACHE = ROOT / 'data/processed/cross_asset_5m.csv'
BAR = pd.Timedelta(minutes=5)
HOURLY = pd.Timedelta(hours=1)

FEATURES = ['x_dxy_r1h', 'x_dxy_r4h', 'x_dxy_r24h', 'x_oil_r1h', 'x_oil_r24h',
            'x_vix_r24h', 'x_vix_level', 'x_gold_dxy_corr_5d', 'x_rel_strength',
            'x_beta_dxy_5d']


def _load_hourly(name):
    path = MARKET / f'{name}_h1.csv'
    if not path.exists():
        return None
    d = pd.read_csv(path)
    d['timestamp'] = pd.to_datetime(d.timestamp, unit='ms', utc=True)
    return d.drop_duplicates('timestamp').sort_values('timestamp').set_index('timestamp')


def _resample_close(hourly, rule='5min'):
    """Hold the last completed hourly close forward across the finer grid."""
    if hourly is None:
        return None
    # shift by one bar so a value is only visible after its hour has closed
    s = hourly.close.shift(1)
    return s.resample(rule).ffill()


def build():
    gold = pd.read_csv(ROOT/'data/raw/xauusd_m5.csv')
    gold['timestamp'] = pd.to_datetime(gold.timestamp, unit='ms', utc=True)
    gold = gold.drop_duplicates('timestamp').sort_values('timestamp').set_index('timestamp')
    idx = gold.index
    g = gold.close

    out = pd.DataFrame(index=idx)
    caches = {n: _resample_close(_load_hourly(n)) for n in ['dxy', 'oil', 'volatility']}
    dxy, oil, vix = caches['dxy'], caches['oil'], caches['volatility']

    def ret(s, bars):
        return s / s.shift(bars) - 1 if s is not None else pd.Series(np.nan, index=idx)

    h = 12  # 12 five-minute bars = 1 hour
    out['x_dxy_r1h'] = ret(dxy.reindex(idx).ffill(), h)
    out['x_dxy_r4h'] = ret(dxy.reindex(idx).ffill(), h*4)
    out['x_dxy_r24h'] = ret(dxy.reindex(idx).ffill(), h*24)
    out['x_oil_r1h'] = ret(oil.reindex(idx).ffill(), h)
    out['x_oil_r24h'] = ret(oil.reindex(idx).ffill(), h*24)
    out['x_vix_r24h'] = np.log(vix.reindex(idx).ffill()).diff(h*24)
    out['x_vix_level'] = vix.reindex(idx).ffill()

    # rolling gold/dollar co-movement and relative strength (5 trading days ~ 5*288 bars)
    n = 5*288
    gr = np.log(g).diff()
    dr = np.log(dxy.reindex(idx).ffill()).diff()
    out['x_gold_dxy_corr_5d'] = gr.rolling(n).corr(dr)
    out['x_beta_dxy_5d'] = gr.rolling(n).cov(dr) / dr.rolling(n).var()
    out['x_rel_strength'] = (g / g.shift(n)) / (dxy.reindex(idx).ffill() / dxy.reindex(idx).ffill().shift(n))

    out = out.replace([np.inf, -np.inf], np.nan)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE)
    print(f'Cross-asset 5m: {len(out):,} rows -> {CACHE.name}')
    return out


def load():
    if not CACHE.exists():
        return build()
    return pd.read_csv(CACHE, parse_dates=['timestamp']).set_index('timestamp')


def features_at(times, frame=None):
    f = load() if frame is None else frame
    ts = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    cutoff = ts - BAR
    idx = np.searchsorted(f.index.values, cutoff.values, side='right') - 1
    out = pd.DataFrame(index=range(len(ts)))
    valid = idx >= 0
    for c in FEATURES:
        vals = np.full(len(ts), np.nan)
        if c in f.columns:
            vals[valid] = f[c].values[idx[valid]]
        out[c] = vals
    return out


if __name__ == '__main__':
    build()
