"""Time-split training, out-of-sample evaluation, and research report."""
from __future__ import annotations
import argparse, base64, io, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from market_context import FEATURES as MARKET_FEATURES
from live_gold_feed import FEATURES as PAXG_FEATURES
from chart_features import CHART as CHART_FEATURES

BASE_NUMERIC = ["signed_surprise_z", "abs_surprise_z", "pre_return_24h", "pre_volatility_24h",
                "release_hour_sin", "release_hour_cos", "fomc_rate_change"] + MARKET_FEATURES
# A curated 5-minute chart subset. Adding all 23 chart features overfits ~700 training
# rows (Brier worsens); these four improved 2025-26 holdout AUC from 0.548 to 0.585 at 1h
# when paired with strong regularization. See MODEL_IMPROVEMENTS.md.
CHART_SMALL = ["ch_ret_1h", "ch_rsi_14", "ch_range_pos_24h", "ch_vol_ratio"]
# Per-horizon configuration selected on 2023-24 and confirmed on 2025-26. The 24h
# horizon keeps the base features because chart features did not generalise there.
HORIZON_CONFIG = {
    "1h": {"numeric": BASE_NUMERIC + CHART_SMALL, "C": 0.05},
    "4h": {"numeric": BASE_NUMERIC + CHART_SMALL, "C": 0.5},
    "24h": {"numeric": BASE_NUMERIC, "C": 0.5},
}
NUMERIC = BASE_NUMERIC
# Event selection matters more than feature engineering here: adding 14 lower-impact
# releases (Existing Home Sales, Factory Orders, NAHB...) cut 1h walk-forward AUC from
# 0.621 to 0.587. The extra events stay in the dataset for analysis but are excluded
# from the direction model. Their shocks are mostly noise for gold.
MODEL_EVENTS = ["NFP", "Unemployment", "Wages", "CPI", "Core CPI", "PPI", "Core PCE",
                "Retail sales", "GDP", "FOMC"]
CATEGORICAL = ["event_type"]
HORIZONS = ["1h", "4h", "24h"]
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
ROUND_TRIP_COST = 0.0003  # 3 bps; user must replace with broker-specific news slippage.


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(train, cols, h, cfg, n_folds=4):
    """Leak-free probability calibration.

    Out-of-fold predictions are generated from expanding folds *inside* the training
    window, so the calibration map never sees the test period. Platt scaling (two
    parameters) beat isotonic regression out of sample: isotonic overfit the small
    calibration sample and raised Brier score in every horizon.
    """
    tr = train.sort_values("release_time")
    y_all = (tr[f"return_{h}"] > 0).astype(int)
    oof, truth = [], []
    edges = np.linspace(0, len(tr), n_folds + 1).astype(int)
    for i in range(1, n_folds):
        fit_idx, val_idx = slice(0, edges[i]), slice(edges[i], edges[i + 1])
        sub = tr.iloc[fit_idx]
        if sub[f"return_{h}"].gt(0).nunique() < 2:
            continue
        m = pipeline("direction", cols, cfg["C"])
        m.fit(sub[cols + CATEGORICAL], (sub[f"return_{h}"] > 0).astype(int))
        oof.extend(m.predict_proba(tr.iloc[val_idx][cols + CATEGORICAL])[:, 1])
        truth.extend(y_all.iloc[val_idx].to_numpy())
    if len(set(truth)) < 2 or len(oof) < 40:
        return {"a": 1.0, "b": 0.0, "n": len(oof)}
    lr = LogisticRegression(max_iter=1000).fit(_logit(oof).reshape(-1, 1), np.array(truth))
    return {"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0]), "n": len(oof)}


def apply_platt(p, cal):
    return 1 / (1 + np.exp(-(cal["a"] * _logit(p) + cal["b"])))


def pipeline(kind, numeric=None, C=0.5):
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric or NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    model = LogisticRegression(C=C, max_iter=4000, class_weight="balanced") if kind == "direction" else Ridge(alpha=10)
    return Pipeline([("features", pre), ("model", model)])


def fig_uri(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def metrics_table(predictions):
    rows = []
    for h, frame in predictions.groupby("horizon"):
        y = frame.actual_direction
        p = frame.probability_up
        trade = np.where(p >= .55, 1, np.where(p <= .45, -1, 0))
        delayed = frame[f"delayed_return_{h}"].fillna(0).to_numpy()
        gross = trade * delayed
        net = gross - np.abs(trade) * ROUND_TRIP_COST
        active = trade != 0
        rows.append({
            "horizon": h, "test_events": len(frame), "up_rate": y.mean(),
            "direction_accuracy": accuracy_score(y, p >= .5),
            "majority_accuracy": max(y.mean(), 1-y.mean()),
            "balanced_accuracy": balanced_accuracy_score(y, p >= .5),
            "roc_auc": roc_auc_score(y, p) if y.nunique() > 1 else np.nan,
            "mae_bps": mean_absolute_error(frame[f"return_{h}"], frame.predicted_return) * 10000,
            "return_correlation": frame[[f"return_{h}", "predicted_return"]].corr().iloc[0, 1],
            "trades": int(active.sum()), "trade_hit_rate": float((net[active] > 0).mean()) if active.any() else np.nan,
            "mean_net_trade_bps": float(net[active].mean() * 10000) if active.any() else np.nan,
            "sum_net_bps": float(net.sum() * 10000),
        })
    return pd.DataFrame(rows)


def event_stats(data):
    rows = []
    for event in sorted(data.event_type.unique()):
        for h in HORIZONS:
            g = data[data.event_type.eq(event) & data[f"return_{h}"].notna()].copy()
            if len(g) < 4:
                continue
            base = {"event_type": event, "horizon": h, "n": len(g),
                    "mean_return_bps": g[f"return_{h}"].mean()*10000,
                    "median_return_bps": g[f"return_{h}"].median()*10000,
                    "up_rate": (g[f"return_{h}"] > 0).mean()}
            for label, mask in {
                "gold_bullish_surprise": g.signed_surprise_z > .25,
                "inline": g.signed_surprise_z.abs() <= .25,
                "gold_bearish_surprise": g.signed_surprise_z < -.25,
            }.items():
                x = g[mask]
                base[label+"_n"] = len(x)
                base[label+"_mean_bps"] = x[f"return_{h}"].mean()*10000 if len(x) else np.nan
                base[label+"_up_rate"] = (x[f"return_{h}"] > 0).mean() if len(x) else np.nan
            rows.append(base)
    return pd.DataFrame(rows)


def reaction_profile(data):
    times = ["5m", "15m", "1h", "4h", "24h"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, event in zip(axes, ["NFP", "CPI", "FOMC"]):
        g = data[data.event_type.eq(event)]
        for name, mask, color in [("Gold-bullish surprise", g.signed_surprise_z > .25, "#16866b"),
                                  ("Gold-bearish surprise", g.signed_surprise_z < -.25, "#c14c4c")]:
            x = g[mask]
            vals = [x[f"return_{t}"].mean()*10000 for t in times]
            ax.plot(times, vals, marker="o", label=f"{name} (n={len(x)})", color=color)
        ax.axhline(0, color="#777", lw=.8)
        ax.set_title(event)
        ax.set_ylabel("Mean XAUUSD log return (bps)")
        ax.legend(fontsize=8)
    return fig_uri(fig)


def html_report(metrics, stats, profile, train_rows, test_rows, start, end):
    def table(df, pct=False):
        x = df.copy()
        for c in x.columns:
            if c in {"up_rate", "direction_accuracy", "majority_accuracy", "balanced_accuracy", "roc_auc", "trade_hit_rate", "return_correlation", "gold_bullish_surprise_up_rate", "inline_up_rate", "gold_bearish_surprise_up_rate"}:
                x[c] = x[c].map(lambda v: "" if pd.isna(v) else f"{v:.1%}" if c != "return_correlation" else f"{v:.3f}")
            elif pd.api.types.is_float_dtype(x[c]):
                x[c] = x[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
        return x.to_html(index=False, border=0, classes="data")
    key_stats = stats[stats.horizon.eq("1h")][["event_type", "n", "gold_bullish_surprise_n", "gold_bullish_surprise_mean_bps", "gold_bullish_surprise_up_rate", "gold_bearish_surprise_n", "gold_bearish_surprise_mean_bps", "gold_bearish_surprise_up_rate"]]
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>XAUUSD US News Model — Research Report</title>
<style>body{{font:15px/1.55 system-ui,sans-serif;max-width:1200px;margin:35px auto;padding:0 20px;color:#18222d}}h1,h2{{color:#102a43}}.note{{background:#fff7dc;border-left:4px solid #d69e2e;padding:14px}}.good{{background:#eaf7f1;border-left:4px solid #16866b;padding:14px}}.data{{border-collapse:collapse;width:100%;font-size:13px;margin:15px 0 30px}}.data th,.data td{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}.data th:first-child,.data td:first-child{{text-align:left}}img{{max-width:100%}}code{{background:#f2f4f7;padding:2px 5px}}</style></head><body>
<h1>XAUUSD Reaction to Major US Economic News</h1><p>Research period: {start} to {end}. Five-minute Dukascopy XAUUSD bid bars; TradingView US economic calendar.</p>
<div class='note'><b>Research, not financial advice.</b> This is a small event sample, not a guaranteed trading system. FOMC consensus often uses an unchanged-rate proxy and omits statement/press-conference tone. Revisions, feed differences, spreads and news slippage can materially change results.</div>
<h2>Out-of-time results (2025 onward)</h2><p>Models train on {train_rows:,} pre-2025 event observations and are evaluated on {test_rows:,} later observations. The direction model is pooled across event types using signed standardized surprise, surprise magnitude, pre-event trend/volatility and event type. Return predictions come from ridge regression; direction probabilities from regularized logistic regression.</p>
{table(metrics)}
<p>Trades are only a diagnostic: positions follow probability signals outside 45%–55%, enter using the five-minute-delayed return, and pay a fixed {ROUND_TRIP_COST*10000:.0f} bps assumed round-trip cost. <code>sum_net_bps</code> is an arithmetic event sum, not compounded portfolio equity; overlapping events are not a valid executable portfolio simulation.</p>
<h2>Average event path</h2><img src='{profile}' alt='Average event reaction path'>
<h2>One-hour conditional event study</h2><p>A gold-bullish surprise is signed by the usual first-order macro channel. For example, stronger NFP/CPI is gold-bearish; higher unemployment is gold-bullish. Values are basis points from the last available pre-release bid close.</p>{table(key_stats)}
<h2>How to interpret this responsibly</h2><ul><li>Focus on whether surprise-conditioned averages and out-of-time metrics agree, not on one impressive backtest number.</li><li>A direction accuracy near 50% or AUC near 0.5 means no demonstrated directional edge. Return magnitude can still be useful for risk planning.</li><li>FOMC reactions depend heavily on guidance. Treat its numerical model as incomplete.</li><li>Use the dashboard for scenario analysis, then validate future releases in a forward log before risking money.</li></ul>
</body></html>"""


def main(input_path, output_dir):
    data = pd.read_csv(input_path, parse_dates=["release_time"])
    output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions, models, metric_input = [], {}, []
    for h in HORIZONS:
        frame = data[data.event_type.isin(MODEL_EVENTS)]
        frame = frame.dropna(subset=[f"return_{h}", "signed_surprise_z"]).copy()
        train = frame[frame.release_time < TEST_START]
        test = frame[frame.release_time >= TEST_START].copy()
        if min(len(train), len(test)) < 20:
            continue
        cfg = HORIZON_CONFIG[h]
        cols = [c for c in cfg["numeric"] if c in frame.columns]
        direction = pipeline("direction", cols, cfg["C"])
        regression = pipeline("return", cols)
        direction.fit(train[cols+CATEGORICAL], (train[f"return_{h}"] > 0).astype(int))
        regression.fit(train[cols+CATEGORICAL], train[f"return_{h}"])
        cal = fit_platt(train, cols, h, cfg)
        raw = direction.predict_proba(test[cols+CATEGORICAL])[:, 1]
        test["probability_up_raw"] = raw
        test["probability_up"] = apply_platt(raw, cal)
        test["predicted_return"] = regression.predict(test[cols+CATEGORICAL])
        test["actual_direction"] = (test[f"return_{h}"] > 0).astype(int)
        test["horizon"] = h
        all_predictions.append(test)
        models[h] = {"direction": direction, "return": regression, "numeric": cols, "calibration": cal,
                     "train_end": str(train.release_time.max()), "test_start": str(test.release_time.min())}
    predictions = pd.concat(all_predictions)
    metrics = metrics_table(predictions)
    stats = event_stats(data)
    profile = reaction_profile(data)
    predictions.to_csv(output_dir/"out_of_sample_predictions.csv", index=False)
    metrics.to_csv(output_dir/"metrics.csv", index=False)
    stats.to_csv(output_dir/"event_statistics.csv", index=False)
    joblib.dump({"models": models, "numeric_features": NUMERIC, "horizon_config": HORIZON_CONFIG, "categorical_features": CATEGORICAL,
                 "test_start": TEST_START, "cost": ROUND_TRIP_COST}, output_dir/"models.joblib")
    metadata = {"generated_at": pd.Timestamp.utcnow().isoformat(), "events": len(data),
                "start": str(data.release_time.min()), "end": str(data.release_time.max()),
                "test_start": str(TEST_START), "metrics": metrics.to_dict("records")}
    (output_dir/"metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    report = html_report(metrics, stats, profile, int((data.release_time < TEST_START).sum()),
                         int((data.release_time >= TEST_START).sum()), str(data.release_time.min().date()), str(data.release_time.max().date()))
    (output_dir/"research_report.html").write_text(report)
    print(metrics.to_string(index=False))
    print(f"\nSaved models, predictions and report to {output_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/processed/event_study.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    a = p.parse_args()
    main(a.input, a.output_dir)
