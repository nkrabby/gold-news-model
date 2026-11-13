"""Download US economic calendar history from TradingView's public calendar endpoint."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import pandas as pd
import requests

URL = "https://economic-calendar.tradingview.com/events"

def month_starts(start: str, end: str):
    return pd.period_range(start, end, freq="M")

def fetch(start: str, end: str, output: Path) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Origin": "https://www.tradingview.com"})
    rows, failures = [], []
    periods = month_starts(start, end)
    for i, period in enumerate(periods, 1):
        first, last = period.start_time, min(period.end_time, pd.Timestamp(end))
        params = {
            "from": first.strftime("%Y-%m-%dT00:00:00.000Z"),
            "to": last.strftime("%Y-%m-%dT23:59:59.999Z"),
            "countries": "US",
        }
        for attempt in range(4):
            try:
                response = session.get(URL, params=params, timeout=45)
                response.raise_for_status()
                payload = response.json()
                if payload.get("status") != "ok":
                    raise RuntimeError(str(payload)[:300])
                rows.extend(payload["result"])
                print(f"[{i}/{len(periods)}] {period}: {len(payload['result'])} events", flush=True)
                break
            except Exception as exc:
                if attempt == 3:
                    failures.append({"period": str(period), "error": str(exc)})
                    print(f"FAILED {period}: {exc}", flush=True)
                else:
                    time.sleep(2 ** attempt)
        time.sleep(0.08)
    if not rows:
        raise RuntimeError("No calendar records downloaded")
    frame = pd.DataFrame(rows).drop_duplicates(subset=["id"]).sort_values("date")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    if failures:
        output.with_suffix(".failures.json").write_text(json.dumps(failures, indent=2))
        raise RuntimeError(f"Some months failed; see {output.with_suffix('.failures.json')}")
    print(f"Saved {len(frame):,} unique events to {output}")
    return frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/us_calendar.csv"))
    args = parser.parse_args()
    fetch(args.start, args.end, args.output)
