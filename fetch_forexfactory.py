"""Forex Factory consensus history.

The public calendar pages are HTML with session-dependent times, so this module
ingests dated event rows and uses only the values (actual/forecast/previous).
Release timestamps remain those already validated from the TradingView calendar;
Forex Factory supplies an independent consensus for cross-checking and filling
FOMC gaps, where the project previously fell back to "no change" as a proxy.
"""
from pathlib import Path
import re,sys,time
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/raw/forexfactory_usd.csv'
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'
# Forex Factory titles -> project event types (same taxonomy as build_dataset.EVENTS).
TITLES={
 'Non-Farm Employment Change':'NFP','Nonfarm Payrolls':'NFP',
 'Unemployment Rate':'Unemployment',
 'Average Hourly Earnings m/m':'Wages',
 'CPI m/m':'CPI','Core CPI m/m':'Core CPI',
 'PPI m/m':'PPI',
 'Core PCE Price Index m/m':'Core PCE',
 'Retail Sales m/m':'Retail sales',
 'Advance GDP q/q':'GDP','Prelim GDP q/q':'GDP','Final GDP q/q':'GDP',
 'Federal Funds Rate':'FOMC',
 'ISM Manufacturing PMI':'ISM Manufacturing','ISM Services PMI':'ISM Services',
 'CB Consumer Confidence':'Consumer Confidence','Durable Goods Orders m/m':'Durable Goods',
 'Core PPI m/m':'Core PPI','Existing Home Sales':'Existing Home Sales',
 'Factory Orders m/m':'Factory Orders','Chicago PMI':'Chicago PMI',
 'NAHB Housing Market Index':'NAHB Housing','Challenger Job Cuts y/y':'Challenger Layoffs',
 'ADP Non-Farm Employment Change':'ADP','JOLTS Job Openings':'JOLTS',
 'Empire State Manufacturing Index':'Empire State','Unemployment Claims':'Jobless Claims',
}
def parse_number(text,event):
    if text is None:return np.nan
    t=str(text).strip().replace('\u2212','-').replace(',','')
    if not t or t in {'-','—'}:return np.nan
    m=re.match(r'^(-?\d*\.?\d+)\s*([KMB%]?)$',t)
    if not m:return np.nan
    v=float(m.group(1));suf=m.group(2)
    if suf=='K':v*=1_000
    elif suf=='M':v*=1_000_000
    elif suf=='B':v*=1_000_000_000
    return v
MONTHS={m:i+1 for i,m in enumerate(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])}
def resolve_date(label,page_year,page_month):
    """'Tue Nov 13' -> date. The dateline attribute is midnight in a US timezone and
    shifts by a day when converted to UTC, so the visible label is authoritative."""
    m=re.search(r'([A-Z][a-z]{2})\s+(\d{1,2})',label)
    if not m:return None
    mon=MONTHS.get(m.group(1));day=int(m.group(2))
    if mon is None:return None
    year=page_year
    if mon-page_month>6:year-=1
    elif page_month-mon>6:year+=1
    try:return pd.Timestamp(year=year,month=mon,day=day).date()
    except ValueError:return None
def parse_page(html,year,month=None):
    soup=BeautifulSoup(html,'lxml');rows=[];cur=None
    for tr in soup.select('tr.calendar__row'):
        date_td=tr.select_one('td.calendar__date')
        if date_td:
            cur=resolve_date(date_td.get_text(' ',strip=True),year,month)
        if cur is None:continue
        cur_td=tr.select_one('td.calendar__currency')
        if not cur_td or cur_td.get_text(strip=True)!='USD':continue
        title_el=tr.select_one('span.calendar__event-title')
        if not title_el:continue
        title=title_el.get_text(' ',strip=True)
        rows.append({'date':cur,'title':title,
                     'impact':(tr.select_one('td.calendar__impact') or {}).get('class',[''])[-1] if tr.select_one('td.calendar__impact') else '',
                     'actual':(tr.select_one('td.calendar__actual') or BeautifulSoup('','lxml')).get_text(strip=True),
                     'forecast':(tr.select_one('td.calendar__forecast') or BeautifulSoup('','lxml')).get_text(strip=True),
                     'previous':(tr.select_one('td.calendar__previous') or BeautifulSoup('','lxml')).get_text(strip=True)})
    return rows
def download(start='2020-01',end='2026-12'):
    s=requests.Session();s.headers['User-Agent']=UA
    months=pd.period_range(start,end,freq='M');allrows=[]
    for i,per in enumerate(months,1):
        url=f'https://www.forexfactory.com/calendar?month={per.strftime("%b").lower()}.{per.year}'
        try:
            r=s.get(url,timeout=30);r.raise_for_status()
            rows=parse_page(r.content.decode('windows-1252','replace'),per.year,per.month)
        except Exception as e:
            print(f'[{i}/{len(months)}] {per} failed: {str(e)[:60]}');continue
        for row in rows:
            row['page_month']=str(per)
        allrows+=rows
        if i%12==0:print(f'[{i}/{len(months)}] through {per}: {len(allrows)} USD rows')
        time.sleep(0.4)
    df=pd.DataFrame(allrows)
    if df.empty:raise RuntimeError('No Forex Factory rows parsed')
    df['event_type']=df.title.map(TITLES)
    df['ff_actual']=df.apply(lambda r:parse_number(r.actual,r.title),axis=1)
    df['ff_forecast']=df.apply(lambda r:parse_number(r.forecast,r.title),axis=1)
    df['ff_previous']=df.apply(lambda r:parse_number(r.previous,r.title),axis=1)
    df=df.drop_duplicates(subset=['date','title'])
    OUT.parent.mkdir(parents=True,exist_ok=True);df.to_csv(OUT,index=False)
    print(f'Saved {len(df)} USD rows ({df.event_type.notna().sum()} matched to project events) -> {OUT}')
    return df
def consensus_frame():
    """One consensus per (date,event_type) for joining to the UTC calendar."""
    df=pd.read_csv(OUT,parse_dates=['date'])
    df=df[df.event_type.notna()&df.ff_forecast.notna()]
    # Prefer the row with an actual value: the released event, not a preview entry.
    df=df.sort_values(['date','event_type','ff_actual'],na_position='last')
    df=df.drop_duplicates(subset=['date','event_type'],keep='first')
    return df[['date','event_type','ff_forecast','ff_actual','ff_previous','title']]
def download_recent():
    """Refresh the months that can still change, backfilling fully on a fresh repo.

    A new deployment has no history file. Fetching only three months would leave most
    events without a consensus, and build_dataset would (correctly) refuse to publish a
    partial dataset - so every scheduled run would fail. Detect that case and backfill.
    """
    now=pd.Timestamp.utcnow()
    have_full = OUT.exists() and len(pd.read_csv(OUT)) > 5000
    if not have_full:
        print('No usable consensus history found; downloading full range 2020-01 onwards')
        return download('2020-01', (now+pd.DateOffset(months=1)).strftime('%Y-%m'))
    start=(now-pd.DateOffset(months=2)).strftime('%Y-%m')
    tmp=OUT.with_name('forexfactory_recent.csv')
    old=pd.read_csv(OUT,parse_dates=['date']) if OUT.exists() else pd.DataFrame()
    globals()['OUT']=tmp
    new=download(start,(now+pd.DateOffset(months=1)).strftime('%Y-%m'))
    globals()['OUT']=OUT
    if not old.empty and not new.empty:
        cutoff=pd.Timestamp(start+'-01')
        keep=old[old.date<pd.Timestamp(cutoff)]
        merged=pd.concat([keep,new]).drop_duplicates(subset=['date','title'])
        merged.to_csv(OUT,index=False)
        print(f'Recent refresh merged: {len(merged)} rows')
    tmp.unlink(missing_ok=True)


if __name__=='__main__':
    if '--recent' in sys.argv:download_recent()
    elif '--parse-test' in sys.argv:
        h=requests.get('https://www.forexfactory.com/calendar?month=sep.2026',headers={'User-Agent':UA},timeout=30).content.decode('windows-1252','replace')
        d=pd.DataFrame(parse_page(h,2026,9));print(d[d.title.str.contains('CPI|Payroll|PPI|PCE|Retail|Unemployment|Earnings|Funds|GDP',case=False,na=False)].to_string())
    else:download()
