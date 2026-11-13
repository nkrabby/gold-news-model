"""The scorer must fill only fully elapsed horizons, and only correct outcomes."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import score_forecasts as sf


def make_prices(path, start='2024-01-01', hours=72, base=2000.0, step=1.0):
    """Monotonic ramp so the correct sign of every return is known."""
    ts = pd.date_range(start, periods=hours * 12, freq='5min', tz='UTC')
    close = [base + i * step for i in range(len(ts))]
    pd.DataFrame({'timestamp': ts, 'open': close, 'high': close,
                  'low': close, 'close': close}).assign(
        timestamp=lambda d: (d.timestamp.astype('int64') // 10**6)).to_csv(path, index=False)


def make_log(path, release_time, prob=0.7, horizon_ok=True):
    pd.DataFrame([{
        'forecast_id': 'test|PPI', 'forecast_time': str(pd.Timestamp('2024-01-02 10:00', tz='UTC')),
        'event_type': 'PPI', 'release_time': str(release_time), 'horizon': '1h',
        'direction': 'Bullish', 'probability_up': prob, 'expected_return': 0.001,
        'lower_80': -0.004, 'upper_80': 0.006, 'consensus': 0.4,
        'actual': np.nan, 'return_1h': np.nan, 'return_4h': np.nan, 'return_24h': np.nan,
        'scored_at': np.nan,
    }]).to_csv(path, index=False)


def test_fills_only_elapsed_horizons(tmp_path):
    prices, log = tmp_path / 'p.csv', tmp_path / 'l.csv'
    make_prices(prices, hours=72)                      # 72h of data from 2024-01-01
    release = pd.Timestamp('2024-01-02 13:30', tz='UTC')
    make_log(log, release)
    sf.score(verbose=False, log_path=log, prices_path=prices)
    out = pd.read_csv(log).iloc[0]
    # 1h and 4h fit inside 72h of data; 24h after 13:30 on Jan 2 also fits (Jan 3 13:30)
    assert np.isfinite(out.return_1h)
    assert out.return_1h > 0                            # rising ramp => positive return
    assert np.isfinite(out.return_24h)
    assert pd.notna(out.scored_at)


def test_leaves_unevaluated_when_horizon_incomplete(tmp_path):
    prices, log = tmp_path / 'p.csv', tmp_path / 'l.csv'
    make_prices(prices, hours=8)                        # only 8h of data
    release = pd.Timestamp('2024-01-01 03:30', tz='UTC')
    make_log(log, release)
    sf.score(verbose=False, log_path=log, prices_path=prices)
    out = pd.read_csv(log).iloc[0]
    assert np.isfinite(out.return_1h)                   # 1h elapsed
    assert np.isnan(out.return_24h)                     # 24h has not


def test_does_not_touch_future_or_rewrite_forecast(tmp_path):
    prices, log = tmp_path / 'p.csv', tmp_path / 'l.csv'
    make_prices(prices, hours=6)
    future = pd.Timestamp('2024-01-05 13:30', tz='UTC')
    make_log(log, future, prob=0.66)
    sf.score(verbose=False, log_path=log, prices_path=prices)
    out = pd.read_csv(log).iloc[0]
    assert out.return_1h != out.return_1h               # still NaN
    assert abs(out.probability_up - 0.66) < 1e-9        # forecast unchanged
    assert out.direction == 'Bullish'


def test_scorer_overwrites_fabricated_outcome(tmp_path):
    """A pre-filled but unrealised return must be discarded, not trusted."""
    prices, log = tmp_path / 'p.csv', tmp_path / 'l.csv'
    make_prices(prices, hours=8)
    release = pd.Timestamp('2024-01-01 03:30', tz='UTC')
    make_log(log, release)
    d = pd.read_csv(log)
    d.loc[0, 'return_24h'] = 0.123          # fabricated: 24h cannot have elapsed
    d.loc[0, 'return_1h'] = 0.999           # wrong value that should be recomputed
    d.to_csv(log, index=False)
    sf.score(verbose=False, log_path=log, prices_path=prices)
    out = pd.read_csv(log).iloc[0]
    assert np.isnan(out.return_24h)         # discarded
    assert np.isfinite(out.return_1h)
    assert abs(out.return_1h - 0.999) > 1e-6  # recomputed from prices
