"""Build an upcoming-event calendar and a linked, timestamped research feed.

Headlines are discovery aids, not verified facts or model features. No article text
is republished. Users should open and assess the original source.
"""
from __future__ import annotations
import argparse, time, urllib.parse, xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import requests
from build_dataset import EVENTS, numeric_value

QUERIES = {
    "NFP": 'gold (payrolls OR "jobs report")',
    "Unemployment": 'gold (payrolls OR unemployment OR "jobs report")',
    "Wages": 'gold (wages OR "jobs report")',
    "CPI": 'gold (CPI OR inflation)',
    "Core CPI": 'gold ("core CPI" OR inflation)',
    "PPI": 'gold (PPI OR "producer prices")',
    "Core PCE": 'gold (PCE OR inflation)',
    "Retail sales": 'gold "retail sales"',
    "GDP": 'gold (GDP OR "economic growth")',
    "FOMC": 'gold (FOMC OR "Federal Reserve")',
}
OFFICIAL = {
    "NFP": "https://www.bls.gov/news.release/empsit.toc.htm",
    "Unemployment": "https://www.bls.gov/news.release/empsit.toc.htm",
    "Wages": "https://www.bls.gov/news.release/empsit.toc.htm",
    "CPI": "https://www.bls.gov/news.release/cpi.toc.htm",
    "Core CPI": "https://www.bls.gov/news.release/cpi.toc.htm",
    "PPI": "https://www.bls.gov/news.release/ppi.toc.htm",
    "Core PCE": "https://www.bea.gov/data/personal-consumption-expenditures-price-index",
    "Retail sales": "https://www.census.gov/retail/index.html",
    "GDP": "https://www.bea.gov/data/gdp/gross-domestic-product",
    "FOMC": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
    "Core PPI": "https://www.bls.gov/news.release/ppi.toc.htm",
    "ISM Manufacturing": "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/",
    "ISM Services": "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/services/",
    "Consumer Confidence": "https://www.conference-board.org/topics/consumer-confidence",
    "ADP": "https://adpemploymentreport.com/",
    "JOLTS": "https://www.bls.gov/news.release/jolts.toc.htm",
    "Durable Goods": "https://www.census.gov/manufacturing/m3/index.html",
    "Jobless Claims": "https://www.dol.gov/ui/data.pdf",
}

# Only the modelled high-impact releases are surfaced; the rest stay in the dataset.
MODELLED = ["NFP", "Unemployment", "Wages", "CPI", "Core CPI", "PPI", "Core PCE",
            "Retail sales", "GDP", "FOMC"]


def upcoming(calendar_path: Path, output: Path):
    cal = pd.read_csv(calendar_path)
    cal["release_time"] = pd.to_datetime(cal.date, utc=True)
    now = pd.Timestamp.utcnow()
    rows = []
    cal = cal[cal.title.isin(EVENTS)]
    for _, row in cal[cal.release_time.ge(now)].iterrows():
        event, sign = EVENTS[row.title]
        rows.append({"event_id": str(row.id), "event_type": event,
                     "release_time": row.release_time, "forecast": numeric_value(row, "forecast"),
                     "previous": numeric_value(row, "previous"), "unit": row.get("unit"),
                     "official_url": OFFICIAL.get(event, "")})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame[frame.event_type.isin(MODELLED)]
    frame = frame.drop_duplicates(["event_type", "release_time"]).sort_values("release_time")
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame)} upcoming releases")
    return frame


def google_news(session, query, event, limit=8):
    q = f"({query}) when:45d"
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode({"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    response = session.get(url, timeout=35)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    rows = []
    for item in root.findall(".//item")[:limit]:
        source = item.find("source")
        rows.append({"event_type": event, "title": item.findtext("title", ""),
                     "url": item.findtext("link", ""), "source": source.text if source is not None else "Google News",
                     "published_at": item.findtext("pubDate", ""), "feed": "Google News",
                     "search_url": url})
    return rows


def research(events, output):
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (compatible; GoldNewsResearch/1.0)"
    rows = []
    for event in events.event_type.unique():
        try:
            found = google_news(session, QUERIES.get(event, f"gold {event} forecast"), event)
            rows.extend(found)
            print(f"{event}: {len(found)} research links")
        except Exception as exc:
            print(f"Research warning for {event}: {exc}")
        time.sleep(.2)
    frame = pd.DataFrame(rows).drop_duplicates(["event_type", "title", "source"])
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame)} research links")
    return frame


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--calendar", type=Path, default=Path("data/raw/us_calendar.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = upcoming(args.calendar, args.output_dir/"upcoming_events.csv")
    if len(events):
        research(events, args.output_dir/"research_links.csv")
