"""Point-in-time technical features from the 5-minute XAUUSD chart.

A bar stamped at time T contains prices up to T+5min, so a feature set for a
release at 13:30 may only use bars whose close is <= 13:30. Every lookup uses
`bar_timestamp + 5min <= release_time`, enforced in `features_at`.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / 'data/raw/xauusd_m5.csv'
CACHE = ROOT / 'data/processed/chart_features.csv'
BAR = pd.Timedelta(minutes=5)

FEATURES = [
    'ret_15m', 'ret_1h', 'ret_4h', 'ret_24h', 'ret_5d',
    'vol_1h', 'vol_24h', 'vol_ratio', 'atr_24h_pct',
    'range_pos_24h', 'range_width_24h', 'compression',
    'ema20_dist', 'ema50_dist', 'ema200_dist',
    'rsi_14', 'pct_bars_up_1h', 'drawdown_24h', 'runup_24h',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
]


def _rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build():
    p = pd.read_csv(PRICES)
    p['timestamp'] = pd.to_datetime(p.timestamp, unit='ms', utc=True)
    p = p.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    c = p.close
    f = pd.DataFrame({'timestamp': p.timestamp})

    def ret(span):
        return c / c.shift(span) - 1

    f['ret_15m'] = ret(3)
    f['ret_1h'] = ret(12)
    f['ret_4h'] = ret(48)
    f['ret_24h'] = ret(288)
    f['ret_5d'] = ret(288 * 5)
    r = c.pct_change()
    f['vol_1h'] = r.rolling(12).std()
    f['vol_24h'] = r.rolling(288).std()
    f['vol_ratio'] = f.vol_1h / f.vol_24h.replace(0, np.nan)
    tr = pd.concat([p.high - p.low, (p.high - c.shift()).abs(), (p.low - c.shift()).abs()], axis=1).max(axis=1)
    f['atr_24h_pct'] = tr.rolling(288).mean() / c
    hi = p.high.rolling(288).max()
    lo = p.low.rolling(288).min()
    f['range_pos_24h'] = (c - lo) / (hi - lo).replace(0, np.nan)
    f['range_width_24h'] = (hi - lo) / c
    f['compression'] = (p.high.rolling(12).max() - p.low.rolling(12).min()) / (hi - lo).replace(0, np.nan)
    for n, col in [(20, 'ema20_dist'), (50, 'ema50_dist'), (200, 'ema200_dist')]:
        ema = c.ewm(span=n * 12, adjust=False).mean()   # n hours in 5-min bars
        f[col] = (c - ema) / c
    f['rsi_14'] = _rsi(c, 14)
    f['pct_bars_up_1h'] = (r > 0).rolling(12).mean()
    f['drawdown_24h'] = c / hi - 1
    f['runup_24h'] = c / lo - 1
    hour = p.timestamp.dt.hour + p.timestamp.dt.minute / 60
    f['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    f['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    dow = p.timestamp.dt.dayofweek
    f['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    f['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # Release-time features override the bar's own clock features.
    f = f.replace([np.inf, -np.inf], np.nan)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(CACHE, index=False)
    print(f'Chart features: {len(f):,} bars -> {CACHE.name}')
    return f


def load():
    if not CACHE.exists():
        return build()
    f = pd.read_csv(CACHE, parse_dates=['timestamp'])
    return f


def features_at(times, frame=None):
    """Features from the last bar that has fully closed before each timestamp."""
    f = load() if frame is None else frame
    ts = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    cutoff = (ts - BAR).values                       # bar must close by release time
    idx = np.searchsorted(f.timestamp.values, cutoff, side='right') - 1
    out = pd.DataFrame(index=range(len(ts)))
    valid = idx >= 0
    for col in FEATURES:
        vals = np.full(len(ts), np.nan)
        vals[valid] = f[col].values[idx[valid]]
        out[col] = vals
    # Clock features describe the release, not the bar.
    hour = ts.hour + ts.minute / 60
    out['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    out['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    dow = ts.dayofweek
    out['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    out['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    return out


def attach(dataset):
    feats = features_at(dataset.release_time)
    for c in FEATURES:
        dataset['ch_' + c] = feats[c].values
    return dataset


CHART = ['ch_' + c for c in FEATURES]

if __name__ == '__main__':
    build()
