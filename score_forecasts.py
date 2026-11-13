"""Fill outcomes for logged forecasts and report live accuracy.

Forecasts are immutable: only the outcome columns (actual, return_1h, return_4h,
return_24h, scored_at) are ever written. A horizon is filled only once the price
window has fully elapsed, so a partially-formed return can never be recorded.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'outputs/forecast_log.csv'
STUDY = ROOT / 'data/processed/event_study.csv'
FF = ROOT / 'data/raw/forexfactory_usd.csv'
PRICES = ROOT / 'data/raw/xauusd_m5.csv'
SUMMARY = ROOT / 'outputs/live_scoring.csv'
BAR = pd.Timedelta(minutes=5)
HORIZONS = {'return_1h': pd.Timedelta(hours=1),
            'return_4h': pd.Timedelta(hours=4),
            'return_24h': pd.Timedelta(hours=24)}
OUTCOME_COLS = ['actual', 'return_1h', 'return_4h', 'return_24h', 'scored_at']


def load_prices(path=None):
    p = pd.read_csv(path or PRICES)
    p['timestamp'] = pd.to_datetime(p.timestamp, unit='ms', utc=True)
    p = p.drop_duplicates('timestamp').sort_values('timestamp')
    return p.set_index('timestamp').close


def price_at(series, t):
    """Last bar that has fully closed at or before t (bars are stamped at open)."""
    idx = series.index.searchsorted(t - BAR, side='right') - 1
    return (series.index[idx], series.iloc[idx]) if idx >= 0 else (None, np.nan)


def realised_returns(release_time, close, last_ts):
    out = {}
    _, p0 = price_at(close, release_time)
    for col, span in HORIZONS.items():
        end = release_time + span
        if end > last_ts:                 # horizon has not fully elapsed yet
            out[col] = np.nan
            continue
        _, p1 = price_at(close, end)
        out[col] = float(np.log(p1 / p0)) if p0 and p1 and p0 > 0 and p1 > 0 else np.nan
    return out


def actual_lookup():
    """Actual released value, keyed by (event_type, release date)."""
    frames = []
    if STUDY.exists():
        d = pd.read_csv(STUDY, parse_dates=['release_time'])
        frames.append(pd.DataFrame({'event_type': d.event_type, 'date': d.release_time.dt.date,
                                    'actual': d.actual}))
    if FF.exists():
        f = pd.read_csv(FF, parse_dates=['date'])
        try:
            from fetch_forexfactory import TITLES
            f['event_type'] = f.title.map(TITLES).fillna(f.get('event_type'))
        except Exception:
            pass
        f = f[f.event_type.notna() & f.ff_actual.notna()]
        frames.append(pd.DataFrame({'event_type': f.event_type, 'date': f.date.dt.date,
                                    'actual': f.ff_actual}))
    if not frames:
        return pd.DataFrame(columns=['event_type', 'date', 'actual'])
    a = pd.concat(frames).dropna(subset=['actual'])
    return a.drop_duplicates(['event_type', 'date'], keep='first')


def score(verbose=True, log_path=None, prices_path=None):
    log_path = Path(log_path) if log_path else LOG
    if not log_path.exists():
        print('No forecast log yet')
        return pd.DataFrame()
    log = pd.read_csv(log_path)
    log['release_time'] = pd.to_datetime(log.release_time, utc=True, format='mixed')
    log['forecast_time'] = pd.to_datetime(log.forecast_time, utc=True, format='mixed')
    for c in OUTCOME_COLS:
        if c not in log.columns:
            log[c] = np.nan
        log[c] = log[c].astype(object) if c == 'scored_at' else pd.to_numeric(log[c], errors='coerce')

    close = load_prices(prices_path)
    last_ts = close.index.max()
    actuals = actual_lookup()
    akey = actuals.set_index(['event_type', 'date']).actual

    filled = 0
    for i, row in log.iterrows():
        rt = row.release_time
        if rt + BAR > last_ts:
            continue                                     # release not yet printed
        if pd.isna(row.actual):
            val = akey.get((row.event_type, rt.date()), np.nan)
            if pd.notna(val):
                log.at[i, 'actual'] = val
                filled += 1
        rets = realised_returns(rt, close, last_ts)
        for col, v in rets.items():
            # Authoritative: overwrite whatever is present. A value written by another
            # stage may have been derived from an in-progress bar.
            log.at[i, col] = v
        if log.loc[i, ['return_1h', 'return_4h', 'return_24h']].notna().any():
            log.at[i, 'scored_at'] = pd.Timestamp.utcnow().isoformat()

    log.to_csv(log_path, index=False)

    # ---- live scoreboard -------------------------------------------------
    rows = []
    for h in ['1h', '4h', '24h']:
        col = f'return_{h}'
        s = log.dropna(subset=[col])
        if s.empty:
            rows.append({'horizon': h, 'n_scored': 0})
            continue
        up = (s[col] > 0).astype(int)
        pred_up = (s.probability_up >= .5).astype(int)
        p_up = s.probability_up.clip(.001, .999)
        p_used = np.where(pred_up == 1, p_up, 1 - p_up)          # probability assigned to the call made
        correct = (pred_up == up)
        rows.append({
            'horizon': h, 'n_scored': len(s),
            'hit_rate': correct.mean(),
            'mean_stated_confidence': p_used.mean(),
            'brier': float(((p_up - up) ** 2).mean()),
            'brier_base_rate': float(((up.mean()) * (1 - up.mean()))),
            'base_up_rate': up.mean(),
            'bullish_calls': int(pred_up.sum()),
            'accuracy_when_confident': correct[p_used >= .6].mean() if (p_used >= .6).any() else np.nan,
        })
    summary = pd.DataFrame(rows)
    if log_path == LOG:
        summary.to_csv(SUMMARY, index=False)

    if verbose:
        print(f'Forecast log: {len(log)} snapshots | '
              f'{log.actual.notna().sum()} with actuals | '
              f'{log.return_1h.notna().sum()} with 1h outcomes')
        scored = summary[summary.n_scored > 0]
        if scored.empty:
            print('No horizons have elapsed yet - nothing to score.')
        else:
            print(scored.round(3).to_string(index=False))
            over = (scored.mean_stated_confidence - scored.hit_rate).mean()
            print(f'\nOver/under-confidence (stated - realised hit rate): {over:+.3f}')
    return summary


if __name__ == '__main__':
    score()
