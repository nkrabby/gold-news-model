"""Construct point-in-time event features and XAUUSD reaction labels.

The calendar endpoint is public but unofficial and may change. Downloaded raw
files are retained so a completed study remains reproducible.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# sign: +1 means a higher-than-expected release is normally bullish for gold.
EVENTS = {
    "Non Farm Payrolls": ("NFP", -1),
    "Unemployment Rate": ("Unemployment", 1),
    "Average Hourly Earnings MoM": ("Wages", -1),
    "Inflation Rate MoM": ("CPI", -1),
    "Core Inflation Rate MoM": ("Core CPI", -1),
    "PPI MoM": ("PPI", -1),
    "Core PCE Price Index MoM": ("Core PCE", -1),
    "Retail Sales MoM": ("Retail sales", -1),
    "GDP Growth Rate QoQ Adv": ("GDP", -1),
    "Interest Rate": ("FOMC", -1),
    # Expanded universe (2026-11-13): adds ~1,080 observations so the model is not
    # trained on 721 events. Sign convention: +1 = higher print normally bullish gold.
    "ISM Manufacturing PMI": ("ISM Manufacturing", -1),
    "ISM Services PMI": ("ISM Services", -1),
    "CB Consumer Confidence": ("Consumer Confidence", -1),
    "Durable Goods Orders MoM": ("Durable Goods", -1),
    "Core PPI MoM": ("Core PPI", -1),
    "Existing Home Sales": ("Existing Home Sales", -1),
    "Factory Orders MoM": ("Factory Orders", -1),
    "Chicago PMI": ("Chicago PMI", -1),
    "NAHB Housing Market Index": ("NAHB Housing", -1),
    "Challenger Job Cuts": ("Challenger Layoffs", 1),
    "ADP Employment Change": ("ADP", -1),
    "JOLTs Job Openings": ("JOLTS", -1),
    "NY Empire State Manufacturing Index": ("Empire State", -1),
    "Unemployment Claims": ("Jobless Claims", 1),
}
HORIZONS = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "24h": 1440}


def numeric_value(row, field):
    """Raw values are preferred because they normalize K/M scaling."""
    raw = pd.to_numeric(row.get(field + "Raw"), errors="coerce")
    if pd.notna(raw):
        return float(raw)
    return pd.to_numeric(row.get(field), errors="coerce")


def drop_ambiguous_duplicates(events: pd.DataFrame) -> pd.DataFrame:
    """Remove days where the calendar lists the same release twice.

    Some days carry two rows with the same title but different actual/forecast
    values (calendar duplication around time-zone or revision boundaries).
    Identical rows are collapsed to the earliest; conflicting rows are dropped
    because neither value can be trusted and both would enter training as labels.
    """
    events = events.copy()
    events["release_date"] = pd.to_datetime(events.release_time, utc=True).dt.date
    key = ["event_type", "release_date"]
    events = events.sort_values("release_time")
    same = events.groupby(key).actual.transform("nunique").le(1) & \
           events.groupby(key).forecast.transform("nunique").le(1)
    dropped = int((~same).sum())
    events = events[same].drop_duplicates(key, keep="first")
    if dropped:
        print(f"Dropped {dropped} calendar rows with conflicting duplicate values")
    return events.drop(columns=["release_date"])


def apply_forex_factory_consensus(events: pd.DataFrame,
                                  path: Path | None = None) -> pd.DataFrame:
    """Prefer the independent Forex Factory consensus, above all for FOMC.

    The TradingView calendar has no FOMC consensus, so those forecasts previously
    fell back to "no change" (proxy_consensus). Forex Factory publishes an
    expected Fed funds rate, which removes the proxy for most FOMC meetings.
    """
    events["consensus_source"] = np.where(events.proxy_consensus, "proxy (no change)", "TradingView")
    events["consensus_ff"] = np.nan
    events["consensus_disagreement"] = np.nan
    if path is None or not Path(path).exists():
        return events
    ff = pd.read_csv(path, parse_dates=["date"])
    try:
        from fetch_forexfactory import TITLES
        ff["event_type"] = ff.title.map(TITLES).fillna(ff.get("event_type"))
    except Exception:
        pass
    ff = ff[ff.event_type.notna() & ff.ff_forecast.notna()]
    ff["release_date"] = ff.date.dt.date
    ff = ff.drop_duplicates(["release_date", "event_type"], keep="first")[
        ["release_date", "event_type", "ff_forecast"]
    ]
    events["release_date"] = pd.to_datetime(events.release_time, utc=True).dt.tz_convert(
        "America/New_York").dt.date
    merged = events.merge(ff, on=["release_date", "event_type"], how="left")
    have = merged.ff_forecast.notna()
    merged.loc[have, "consensus_disagreement"] = (
        (merged.loc[have, "ff_forecast"] - merged.loc[have, "forecast"]).abs()
        / merged.loc[have, "forecast"].abs().clip(lower=1e-6))
    merged.loc[have, "consensus_source"] = "Forex Factory"
    merged.loc[have, "forecast"] = merged.loc[have, "ff_forecast"]
    merged.loc[have, "proxy_consensus"] = False
    merged.loc[have, "consensus_ff"] = merged.loc[have, "ff_forecast"]
    before = len(merged)
    merged = merged[merged.forecast.notna()]
    if before - len(merged):
        print(f"Dropped {before - len(merged)} events with no consensus from either source")
    merged["surprise"] = merged.actual - merged.forecast
    merged = merged.drop(columns=["release_date", "ff_forecast"])
    print(f"Consensus: {have.sum()}/{len(merged)} from Forex Factory "
          f"({merged.proxy_consensus.sum()} proxies remain)")
    return merged


def prepare_events(calendar: pd.DataFrame, consensus_path: Path | None = None) -> pd.DataFrame:
    rows = []
    for _, row in calendar[calendar.title.isin(EVENTS)].iterrows():
        event_type, sign = EVENTS[row.title]
        actual = numeric_value(row, "actual")
        forecast = numeric_value(row, "forecast")
        previous = numeric_value(row, "previous")
        proxy = False
        if pd.isna(forecast) and event_type == "FOMC":
            # No reliable free consensus: unchanged-rate expectation is only a proxy.
            forecast, proxy = previous, True
        # A missing TradingView consensus is not fatal: Forex Factory may supply one
        # below. The row is dropped only if no consensus exists after that merge.
        if pd.isna(actual):
            continue
        surprise = actual - forecast
        rows.append({
            "event_id": str(row.id), "release_time": pd.to_datetime(row.date, utc=True),
            "event_type": event_type, "source_title": row.title,
            "reference_period": row.get("referencePeriod", row.get("period")),
            "actual": actual, "forecast": forecast, "previous": previous,
            "surprise": surprise, "gold_surprise_sign": sign,
            "proxy_consensus": proxy, "source_url": row.get("source_url"),
        })
    events = pd.DataFrame(rows).sort_values("release_time").reset_index(drop=True)
    events = apply_forex_factory_consensus(events, consensus_path)
    events = drop_ambiguous_duplicates(events)
    # Robust rolling standardization uses only earlier surprises. Median absolute
    # error prevents the extraordinary 2020 payroll prints from dominating years
    # of subsequent NFP surprise scores, as a conventional standard deviation would.
    events["surprise_sigma"] = np.nan
    for event_type, idx in events.groupby("event_type").groups.items():
        idx = list(idx)
        typical_error = events.loc[idx, "surprise"].abs().rolling(24, min_periods=4).median().shift(1)
        robust_sigma = typical_error / 0.67448975
        fallback_sigma = events.loc[idx, "surprise"].expanding(min_periods=4).std().shift(1)
        # A run of exactly in-line releases can make the median error zero.
        # Use a historical standard deviation and a measurement-aware floor rather
        # than allowing a normal 0.1% surprise to become an astronomical z-score.
        floors = {"NFP": 5000, "FOMC": .125, "GDP": .25}
        floor = floors.get(event_type, .05)
        sigma = robust_sigma.where(robust_sigma > 1e-9, fallback_sigma).clip(lower=floor)
        events.loc[idx, "surprise_sigma"] = sigma.to_numpy()
    # A fully priced FOMC hold is not a surprise, but a change in the policy rate
    # still moves gold. Carry the stance change separately so removing the
    # "no change" consensus proxy does not delete that information.
    events["fomc_rate_change"] = np.where(events.event_type.eq("FOMC"),
                                          events.actual - events.previous, 0.0)
    events["surprise_z"] = events.surprise / events.surprise_sigma
    events["signed_surprise_z"] = events.gold_surprise_sign * events.surprise_z
    events["abs_surprise_z"] = events.signed_surprise_z.abs()
    return events


def price_at_or_before(prices: pd.DataFrame, timestamp):
    # Candle timestamps mark the bar OPEN. A bar opening exactly at the target
    # closes five minutes later, so it must not be used (strictly earlier bars only).
    pos = prices.timestamp.searchsorted(timestamp, side="left") - 1
    if pos < 0:
        return np.nan, pd.NaT
    return prices.close.iloc[pos], prices.timestamp.iloc[pos]


def attach_prices(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    prices["timestamp"] = pd.to_datetime(prices.timestamp, unit="ms", utc=True)
    prices = prices.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    close = prices.set_index("timestamp").close
    log_price = np.log(close)
    ret_24h = log_price.diff(288)
    vol_24h = log_price.diff().rolling(288).std() * np.sqrt(288)
    records = []
    for _, event in events.iterrows():
        t = event.release_time
        p0, actual_t = price_at_or_before(prices, t)
        if pd.isna(p0) or (t - actual_t) > pd.Timedelta(minutes=20):
            continue
        rec = event.to_dict()
        rec.update({"price_at_release": p0, "price_timestamp": actual_t})
        pre_t = t - pd.Timedelta(hours=24)
        pre_pos = prices.timestamp.searchsorted(pre_t, side="left") - 1
        rec["pre_return_24h"] = ret_24h.iloc[pre_pos] if pre_pos >= 0 else np.nan
        rec["pre_volatility_24h"] = vol_24h.iloc[pre_pos] if pre_pos >= 0 else np.nan
        last_bar = prices.timestamp.iloc[-1]
        for label, minutes in HORIZONS.items():
            end_t = t + pd.Timedelta(minutes=minutes)
            # If the window extends beyond the available data, the outcome does not
            # exist yet. Recording the last bar here would fabricate a realised return
            # from a partially formed move - the same look-ahead bug class caught
            # earlier in the chart features.
            if end_t > last_bar:
                rec[f"return_{label}"] = np.nan
                rec[f"price_{label}"] = np.nan
                rec[f"mfe_{label}"] = np.nan
                rec[f"mae_{label}"] = np.nan
                continue
            p1, end_actual = price_at_or_before(prices, end_t)
            rec[f"return_{label}"] = np.log(p1 / p0) if pd.notna(p1) else np.nan
            rec[f"price_{label}"] = p1
            rec[f"mfe_{label}"] = np.nan
            rec[f"mae_{label}"] = np.nan
            window = prices[(prices.timestamp >= t) & (prices.timestamp < end_t)]
            if len(window):
                rec[f"mfe_{label}"] = window.high.max() / p0 - 1
                rec[f"mae_{label}"] = window.low.min() / p0 - 1
        # A delayed label is a less unrealistic proxy for an executable reaction.
        p5, _ = price_at_or_before(prices, t + pd.Timedelta(minutes=5))
        for label in ("1h", "4h", "24h"):
            rec[f"delayed_return_{label}"] = np.log(rec[f"price_{label}"] / p5) if pd.notna(p5) else np.nan
        records.append(rec)
    result = pd.DataFrame(records)
    hour = result.release_time.dt.hour + result.release_time.dt.minute / 60
    result["release_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["release_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return result


def main(calendar_path, price_path, output):
    calendar = pd.read_csv(calendar_path)
    events = prepare_events(calendar, consensus_path=Path(__file__).resolve().parents[1] / "data/raw/forexfactory_usd.csv")
    prices = pd.read_csv(price_path)
    dataset = attach_prices(events, prices)
    try:
        from market_context import attach
        dataset = attach(dataset)
        print("Attached DXY, oil, volatility and Treasury context")
    except FileNotFoundError:
        print("Market-context files not found; continuing without them")
    try:
        from cross_asset import features_at as xfeatures, FEATURES as XF
        xf = xfeatures(dataset.release_time)
        for c in XF:
            dataset[c] = xf[c].values
        print("Attached 5-minute cross-asset context")
    except FileNotFoundError:
        print("Cross-asset cache missing; continuing without it")
    try:
        from chart_features import attach as attach_chart
        dataset = attach_chart(dataset)
        print("Attached 5-minute chart features")
    except FileNotFoundError:
        print("Chart feature cache missing; continuing without it")
    try:
        from live_gold_feed import attach as attach_paxg
        dataset = attach_paxg(dataset)
        print("Attached 24/7 tokenized-gold (PAXG) context")
    except FileNotFoundError:
        print("PAXG feed not downloaded; continuing without it")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Integrity guard: a source file caught mid-write can silently collapse the
    # event count. Never let that pass unreported.
    if output.exists():
        previous = len(pd.read_csv(output))
        if previous > 0 and len(dataset) < 0.9 * previous:
            print(f"WARNING: dataset shrank from {previous} to {len(dataset)} rows. "
                  f"A source file may be incomplete (check data/raw/forexfactory_usd.csv). "
                  f"Retry the refresh before trusting these models.")
    if len(dataset) < 700:
        raise SystemExit(f"Aborting: only {len(dataset)} events built; refusing to overwrite "
                         f"a working dataset with a partial one.")
    dataset.to_csv(output, index=False)
    print(f"Saved {len(dataset):,} matched events to {output}")
    print(dataset.groupby("event_type").size().sort_values(ascending=False).to_string())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--calendar", type=Path, default=Path("data/raw/us_calendar.csv"))
    p.add_argument("--prices", type=Path, default=Path("data/raw/xauusd_m5.csv"))
    p.add_argument("--output", type=Path, default=Path("data/processed/event_study.csv"))
    a = p.parse_args()
    main(a.calendar, a.prices, a.output)
