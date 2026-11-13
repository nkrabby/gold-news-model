from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from src.pre_release import forecast as pre_release_forecast

ROOT = Path(__file__).parent
st.set_page_config(page_title="Gold News Lab", page_icon="🟡", layout="wide")
st.title("Gold News Lab")
st.caption("Event-study research and scenario analysis for US macro releases → XAUUSD")

@st.cache_data
def load_data():
    return pd.read_csv(ROOT/"data/processed/event_study.csv", parse_dates=["release_time"])

@st.cache_resource
def load_models():
    return joblib.load(ROOT/"outputs/models.joblib")

data, bundle = load_data(), load_models()
pre_bundle = joblib.load(ROOT/"outputs/pre_release_models.joblib")
status_path = ROOT/"outputs/live_status.json"
live_status = json.loads(status_path.read_text()) if status_path.exists() else {}
market_path = ROOT/"outputs/market_context.json"
market_context = json.loads(market_path.read_text()) if market_path.exists() else {}
intelligence_path = ROOT/"outputs/research_intelligence.json"
research_intelligence = json.loads(intelligence_path.read_text()) if intelligence_path.exists() else {}
if live_status:
    st.caption(f"Data refreshed: {live_status.get('generated_at', 'unknown')} · XAUUSD bid timestamp: {live_status.get('price_timestamp', 'unknown')} · Free sources may be delayed")
SIGNS = data.groupby("event_type").gold_surprise_sign.median().to_dict()
EVENT_UNITS = {
    "NFP": "jobs, e.g. 165000", "Unemployment": "%", "Wages": "% m/m", "CPI": "% m/m",
    "Core CPI": "% m/m", "PPI": "% m/m", "Core PCE": "% m/m", "Retail sales": "% m/m",
    "GDP": "% annualized q/q", "FOMC": "% policy rate",
}

tab0, tab1, tab2, tab3 = st.tabs(["Upcoming event", "Scenario analyzer", "Historical reactions", "Validation"])

with tab0:
    st.subheader("Pre-event analyst workspace")
    upcoming = pd.read_csv(ROOT/"data/processed/upcoming_events.csv", parse_dates=["release_time"])
    research = pd.read_csv(ROOT/"data/processed/research_links.csv")
    labels = [f"{r.release_time.strftime('%d %b %Y %H:%M UTC')} — {r.event_type}" for _, r in upcoming.iterrows()]
    selected = st.selectbox("Next scheduled release", labels)
    scheduled = upcoming.iloc[labels.index(selected)]
    event = scheduled.event_type
    subset = data[data.event_type.eq(event)]
    unit = EVENT_UNITS.get(event, "release units")
    default_forecast = scheduled.forecast if pd.notna(scheduled.forecast) else scheduled.previous
    if pd.isna(default_forecast):
        default_forecast = subset.iloc[-1].actual
    c1, c2, c3 = st.columns(3)
    c1.metric("Release", event)
    c2.metric("Previous", "Awaiting data" if pd.isna(scheduled.previous) else f"{scheduled.previous:g}")
    c3.metric("Calendar consensus", "Not yet published" if pd.isna(scheduled.forecast) else f"{scheduled.forecast:g}")
    st.markdown(f"[Official release/source]({scheduled.official_url})")
    if pd.isna(scheduled.forecast):
        st.warning("Consensus is not published in the feed yet. Enter a working consensus from your preferred research source; the previous value is only a placeholder.")
    forecast_input = st.number_input(f"Working consensus ({unit})", value=float(default_forecast), key="upcoming_consensus", format="%.4f")
    suggested_trend = float(scheduled.get("suggested_pre_return_24h", 0.0) or 0.0) * 100
    pre_input = st.number_input("Current XAUUSD prior-24h return (%)", value=suggested_trend, step=0.1, key="upcoming_trend") / 100
    st.write("### Pre-release forecast")
    f1, f2 = st.columns(2)
    forecast_horizon = f1.selectbox("Forecast horizon", ["1h", "4h", "24h"], index=0)
    research_tilt = f2.slider("Analyst data-outcome tilt (σ)", -1.0, 1.0, 0.0, .1,
        help="Your explicit research assumption: positive means you expect a stronger release than consensus; negative means weaker. Leave at zero for the statistical baseline.")
    previous_input = scheduled.previous if pd.notna(scheduled.previous) else subset.iloc[-1].previous
    hour = scheduled.release_time.hour + scheduled.release_time.minute/60
    forecast = pre_release_forecast(pre_bundle, bundle, data, event, forecast_input, float(previous_input),
                                    pre_input, hour, research_tilt, forecast_horizon, context=market_context)
    direction_probability = max(forecast["probability_up"], forecast["probability_down"])
    final_direction = "BULLISH" if forecast["probability_up"] >= .5 else "BEARISH"
    research_signal = research_intelligence.get(event, {}).get("signal", "Mixed/neutral")
    research_conflict = (final_direction == "BULLISH" and research_signal == "Bearish context") or (final_direction == "BEARISH" and research_signal == "Bullish context")
    confidence = "Moderate" if direction_probability >= .60 and not research_conflict else "Low"
    st.markdown(f"## FINAL PRE-RELEASE: {final_direction} · {direction_probability:.1%}")
    st.write(f"**Confidence: {confidence}.** Combined pre-release surprise distribution, historical event reactions, DXY/oil/yields, gold trend and analyst research tilt. Headline context: **{research_signal}** (unvalidated and not automatically applied)." + (" It conflicts with the numerical forecast, reducing confidence." if research_conflict else ""))
    with st.expander("Current market context"):
        c = st.columns(5)
        c[0].metric("DXY 24h", f"{market_context.get('dxy_return_24h', 0):+.2%}")
        c[1].metric("Brent 24h", f"{market_context.get('oil_return_24h', 0):+.2%}")
        c[2].metric("US 2Y 5d", f"{market_context.get('dgs2_change_5d', 0):+.2f} pp")
        c[3].metric("US 10Y 5d", f"{market_context.get('dgs10_change_5d', 0):+.2f} pp")
        c[4].metric("Real 10Y 5d", f"{market_context.get('real_yield_change_5d', 0):+.2f} pp")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("P(gold up)", f"{forecast['probability_up']:.1%}")
    m2.metric("P(gold down)", f"{forecast['probability_down']:.1%}")
    m3.metric("Expected move", f"{forecast['expected_return']*100:+.2f}%")
    m4.metric("80% model range", f"{forecast['lower_80']*100:+.2f}% to {forecast['upper_80']*100:+.2f}%")
    d1, d2, d3 = st.columns(3)
    d1.metric("Expected actual", f"{forecast['expected_actual']:.4g}")
    d2.metric("Working consensus", f"{forecast['consensus']:.4g}")
    d3.metric("P(actual beats consensus)", f"{forecast['probability_beat']:.1%}")
    st.caption(f"Forecast for {forecast_horizon} after the scheduled release. This integrates a pre-release surprise distribution with historical reaction models. The range is a model interval, not a guarantee. Out-of-time pre-release direction performance remains weak; use this as decision support, not a trade command.")
    st.write("#### Scenario stress test")
    sigma = float(subset.surprise_sigma.iloc[-1])
    st.write("#### Conditional scenarios—not a prediction of the actual number")
    scenario_rows = []
    scenarios = [("Much stronger than consensus", 1.5), ("Stronger", .75), ("In line", 0), ("Weaker", -.75), ("Much weaker than consensus", -1.5)]
    for label, raw_z in scenarios:
        signed_z = SIGNS[event] * raw_z
        actual_scenario = forecast_input + raw_z*sigma
        hour = scheduled.release_time.hour + scheduled.release_time.minute/60
        features = pd.DataFrame([{"signed_surprise_z": signed_z, "abs_surprise_z": abs(signed_z),
            "pre_return_24h": pre_input, "pre_volatility_24h": np.nan,
            "release_hour_sin": np.sin(2*np.pi*hour/24), "release_hour_cos": np.cos(2*np.pi*hour/24),
            "event_type": event, **{k: market_context.get(k, np.nan) for k in bundle["numeric_features"] if k in market_context}}])
        row = {"Scenario": label, f"Hypothetical actual ({unit})": actual_scenario}
        for h, pair in bundle["models"].items():
            row[f"{h}: P(up)"] = pair["direction"].predict_proba(features)[0, 1]*100
            row[f"{h}: expected bps"] = pair["return"].predict(features)[0]*10000
        scenario_rows.append(row)
    st.dataframe(pd.DataFrame(scenario_rows), hide_index=True, use_container_width=True,
                 column_config={f"{h}: P(up)": st.column_config.NumberColumn(format="%.1f%%") for h in bundle["models"]})
    st.caption("Each row assumes a different unpublished actual. Expected bps are model means, not targets; scenario outputs do not incorporate the research headlines below.")
    st.write("#### Recent research and market commentary")
    st.info("These are discovery links, not verified model facts. Headlines can be wrong, biased, duplicated or stale. Open the original, check its date and evidence, and distinguish reporting from opinion.")
    articles = research[research.event_type.eq(event)].copy()
    if len(articles):
        st.dataframe(articles[["published_at", "source", "title", "url"]], hide_index=True, use_container_width=True,
                     column_config={"url": st.column_config.LinkColumn("Open article", display_text="Read source")})
    else:
        st.write("No recent links collected for this event.")
    with st.expander("Research library and analyst checklist"):
        st.markdown("""
- [World Gold Council research](https://www.gold.org/goldhub/research): real yields, dollar, ETF flows, central-bank demand and regime context.
- [CME FedWatch](https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html): market-implied policy expectations; check how they change after the release.
- [Federal Reserve research and FOMC materials](https://www.federalreserve.gov/econres.htm): policy framework, speeches and working papers.
- [BLS release schedules](https://www.bls.gov/schedule/news_release/): official CPI, PPI and employment timing and tables.

Before forming a view, record: consensus and any 'whisper' number; previous-month revisions; DXY and 2-year/10-year real-yield trend; Fed pricing; gold trend/positioning; and simultaneous releases. For employment, assess payrolls, unemployment, wages and revisions together rather than independently.
""")

with tab1:
    st.subheader("What does a new release imply?")
    st.info("Enter the actual and consensus after publication. This estimates historical reaction probabilities; it does not know breaking headlines, Fed rhetoric, positioning, or your broker's execution costs.")
    left, right = st.columns([1, 2])
    with left:
        event = st.selectbox("Release", sorted(data.event_type.unique()))
        subset = data[data.event_type.eq(event)]
        unit = EVENT_UNITS.get(event, "release units")
        latest = subset.iloc[-1]
        actual = st.number_input(f"Actual ({unit})", value=float(latest.actual), format="%.4f")
        forecast = st.number_input(f"Consensus forecast ({unit})", value=float(latest.forecast), format="%.4f")
        pre_trend = st.number_input("XAUUSD return in prior 24h (%)", value=0.0, step=0.1, format="%.3f") / 100
        sigma = float(subset.surprise_sigma.iloc[-1])
        surprise = actual - forecast
        z = surprise / sigma if sigma and np.isfinite(sigma) and sigma > 0 else 0.0
        signed_z = SIGNS[event] * z
        st.metric("Standardized surprise", f"{z:+.2f}σ", help="Standard deviation is estimated only from earlier surprises for this release.")
        st.write("First-order macro interpretation:", "**gold-bullish**" if signed_z > .25 else "**gold-bearish**" if signed_z < -.25 else "**approximately in line**")
    with right:
        rows = []
        for h, pair in bundle["models"].items():
            t = latest.release_time
            hour = 18 if event == "FOMC" else 13.5
            features = pd.DataFrame([{
                "signed_surprise_z": signed_z, "abs_surprise_z": abs(signed_z),
                "pre_return_24h": pre_trend, "pre_volatility_24h": np.nan,
                "release_hour_sin": np.sin(2*np.pi*hour/24), "release_hour_cos": np.cos(2*np.pi*hour/24),
                "event_type": event,
                **{k: market_context.get(k, np.nan) for k in bundle["numeric_features"] if k in market_context},
            }])
            probability = pair["direction"].predict_proba(features)[0, 1]
            expected = pair["return"].predict(features)[0]
            rows.append({"Horizon": h, "P(XAUUSD up)": probability*100, "Expected return (bps)": expected*10000,
                         "Direction": "Up" if probability >= .5 else "Down"})
        result = pd.DataFrame(rows)
        st.dataframe(result, hide_index=True, use_container_width=True,
                     column_config={"P(XAUUSD up)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
                                    "Expected return (bps)": st.column_config.NumberColumn(format="%.1f")})
        st.caption("Expected returns are model means, not price targets. Probabilities near 50% mean low conviction.")
        analogs = subset[(subset.signed_surprise_z-signed_z).abs() < .75].copy()
        post_horizon = st.selectbox("Final post-release forecast horizon", ["1h", "4h", "24h"], index=0)
        model_p = float(result.loc[result.Horizon.eq(post_horizon), "P(XAUUSD up)"].iloc[0] / 100)
        valid_analogs = analogs.dropna(subset=[f"return_{post_horizon}"])
        if len(valid_analogs) >= 3:
            analog_p = (valid_analogs[f"return_{post_horizon}"] > 0).mean()
            final_p = .5*model_p + .5*analog_p
            final_direction = "BULLISH" if final_p >= .5 else "BEARISH"
            macro_direction = "BEARISH" if signed_z > .25 else "BULLISH" if signed_z < -.25 else "MIXED"
            model_direction = "BULLISH" if model_p >= .5 else "BEARISH"
            agreement = macro_direction == model_direction
            confidence = "Moderate" if agreement and abs(final_p-.5) >= .10 else "Low"
            st.markdown(f"## FINAL POST-RELEASE: {final_direction} · {max(final_p, 1-final_p):.1%}")
            st.write(f"**Confidence: {confidence}.** Final probability blends the reaction model ({model_p:.1%} up) with {len(valid_analogs)} similar historical releases ({analog_p:.1%} up). Macro interpretation: {macro_direction.lower()}; model: {model_direction.lower()}." + (" Signals agree." if agreement else " Signals conflict, so conviction is reduced."))
        if len(analogs) >= 3:
            st.write(f"**Closest broad analogs:** {len(analogs)} {event} releases with a similar signed surprise")
            show = analogs[["release_time", "actual", "forecast", "surprise_z", "return_1h", "return_4h", "return_24h"]].tail(10).copy()
            show.release_time = show.release_time.dt.strftime("%Y-%m-%d %H:%M UTC")
            for c in ["return_1h", "return_4h", "return_24h"]:
                show[c] = (show[c]*100).map(lambda x: f"{x:+.2f}%")
            st.dataframe(show.rename(columns={"surprise_z": "surprise σ"}), hide_index=True, use_container_width=True)

with tab2:
    st.subheader("Historical event explorer")
    c1, c2, c3 = st.columns(3)
    event_filter = c1.selectbox("Event", sorted(data.event_type.unique()), key="explorer")
    horizon = c2.selectbox("Reaction horizon", ["5m", "15m", "1h", "4h", "24h"], index=2)
    regime = c3.selectbox("Surprise regime", ["All", "Gold-bullish", "In line", "Gold-bearish"])
    frame = data[data.event_type.eq(event_filter)].copy()
    if regime == "Gold-bullish": frame = frame[frame.signed_surprise_z > .25]
    if regime == "Gold-bearish": frame = frame[frame.signed_surprise_z < -.25]
    if regime == "In line": frame = frame[frame.signed_surprise_z.abs() <= .25]
    frame = frame.dropna(subset=[f"return_{horizon}"])
    a, b, c, d = st.columns(4)
    a.metric("Events", len(frame))
    b.metric("Mean reaction", f"{frame[f'return_{horizon}'].mean()*10000:+.1f} bps")
    c.metric("Median reaction", f"{frame[f'return_{horizon}'].median()*10000:+.1f} bps")
    d.metric("XAUUSD up frequency", f"{(frame[f'return_{horizon}']>0).mean():.1%}")
    fig = px.scatter(frame, x="signed_surprise_z", y=frame[f"return_{horizon}"]*10000,
                     hover_data=["release_time", "actual", "forecast"], trendline="ols",
                     labels={"x": "Gold-direction signed surprise (σ)", "y": f"XAUUSD {horizon} return (bps)"})
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    explorer = frame[["release_time", "actual", "forecast", "previous", "surprise_z", f"return_{horizon}", f"mfe_{horizon}", f"mae_{horizon}"]].sort_values("release_time", ascending=False).copy()
    explorer.release_time = explorer.release_time.dt.strftime("%Y-%m-%d %H:%M UTC")
    st.dataframe(explorer, hide_index=True, use_container_width=True)

with tab3:
    st.subheader("Out-of-time validation (2025 onward)")
    st.write("**Pre-release forecast model**")
    pre_metrics = pd.read_csv(ROOT/"outputs/pre_release_models.gold_metrics.csv")
    st.dataframe(pre_metrics, hide_index=True, use_container_width=True)
    st.write("**Walk-forward market-context model (multiple chronological folds)**")
    walk_path = ROOT/"outputs/walk_forward_metrics.csv"
    if walk_path.exists():
        st.dataframe(pd.read_csv(walk_path), hide_index=True, use_container_width=True)
    calib_path = ROOT/"outputs/calibration.csv"
    if calib_path.exists():
        st.write("**What a probability actually means**")
        st.caption("Out-of-fold: when the model said X%, how often did gold actually rise?")
        st.dataframe(pd.read_csv(calib_path), hide_index=True, use_container_width=True)
    st.write("**Post-release reaction model**")
    metrics = pd.read_csv(ROOT/"outputs/metrics.csv")
    st.dataframe(metrics, hide_index=True, use_container_width=True)
    st.warning("No performance claim should be based on this small sample alone. Accuracy near 50% and AUC near 0.5 indicate no demonstrated direction edge. The trade columns are diagnostics, not a portfolio backtest.")
    preds = pd.read_csv(ROOT/"outputs/out_of_sample_predictions.csv", parse_dates=["release_time"])
    selected_h = st.selectbox("Prediction horizon", sorted(preds.horizon.unique()))
    p = preds[preds.horizon.eq(selected_h)]
    fig = px.scatter(p, x="predicted_return", y=f"return_{selected_h}", color="event_type",
                     hover_data=["release_time"], labels={"predicted_return": "Predicted log return", f"return_{selected_h}": "Actual log return"})
    fig.add_hline(y=0, line_dash="dot").add_vline(x=0, line_dash="dot")
    st.plotly_chart(fig, use_container_width=True)
    st.write("**Timestamped forecast log**")
    log_path = ROOT/"outputs/forecast_log.csv"
    if log_path.exists():
        log = pd.read_csv(log_path)
        st.dataframe(log.sort_values("forecast_time", ascending=False), hide_index=True, use_container_width=True)

st.caption("Sources: Dukascopy public historical XAUUSD feed; TradingView public US economic calendar. Built for research, not financial advice.")
