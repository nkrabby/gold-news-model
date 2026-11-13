"""Transparent, cited headline-context scores; deliberately not auto-traded.

This is a reproducible keyword baseline, not an LLM comprehension system. Scores are
context indicators until point-in-time validation demonstrates incremental value.
"""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RULES={
 'safe_haven':(['war','missile','attack','conflict','escalation','sanction','ukraine','iran','geopolitical'],1),
 'deescalation':(['ceasefire','peace deal','de-escalation','truce'], -1),
 'hawkish':(['rate hike','hike bets','hawkish','higher rates','tightening'], -1),
 'dovish':(['rate cut','cut bets','dovish','lower rates','easing'],1),
 'official_demand':(['central bank buying','central-bank buying','etf inflow','gold demand'],1),
 'yield_pressure':(['yields rise','dollar rallies','stronger dollar','hot cpi','hot ppi'], -1),
 'relief':(['yields fall','dollar weakens','softer dollar','soft cpi','soft ppi'],1),
}

def analyze():
    p=ROOT/'data/processed/research_links.csv'
    articles=pd.read_csv(p).fillna('') if p.exists() else pd.DataFrame()
    now=pd.Timestamp.utcnow();output={}
    for event,g in articles.groupby('event_type'):
        evidence=[];weighted=[]
        for _,a in g.iterrows():
            text=a.title.lower();score=0.;categories=[]
            for category,(words,weight) in RULES.items():
                if any(w in text for w in words):score+=weight;categories.append(category)
            try:published=pd.Timestamp(a.published_at)
            except:published=now
            if published.tzinfo is None:published=published.tz_localize('UTC')
            age=max(0,(now-published).total_seconds()/86400)
            recency=.5**(age/10)
            if score:weighted.append(score*recency)
            evidence.append({'title':a.title,'url':a.url,'source':a.source,'published_at':a.published_at,
                             'headline_score':score,'categories':categories,'recency_weight':recency})
        raw=float(np.clip(np.sum(weighted),-2,2)) if weighted else 0.
        output[event]={'score':raw,'signal':'Bullish context' if raw>.25 else 'Bearish context' if raw<-.25 else 'Mixed/neutral',
                       'confidence':'Low—unvalidated headline rules','applied_to_probability':False,'evidence':evidence}
    (ROOT/'outputs/research_intelligence.json').write_text(json.dumps(output,indent=2))
    return output
if __name__=='__main__':print(json.dumps(analyze(),indent=2))
