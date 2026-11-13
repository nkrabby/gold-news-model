"""Controlled comparison of feature sets, model classes, and event universes.

Every configuration runs on identical expanding folds. All features are
point-in-time; chart features use only bars closed before the release.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from market_context import FEATURES as MKT
from live_gold_feed import FEATURES as PAXG
from chart_features import CHART

ROOT = Path(__file__).resolve().parents[1]
CORE = ['signed_surprise_z', 'abs_surprise_z', 'pre_return_24h', 'pre_volatility_24h',
        'release_hour_sin', 'release_hour_cos', 'fomc_rate_change']
CAT = ['event_type']
HORIZONS = ['1h', '4h', '24h']
ORIGINAL_TYPES = ['NFP', 'Unemployment', 'Wages', 'CPI', 'Core CPI', 'PPI',
                  'Core PCE', 'Retail sales', 'GDP', 'FOMC']


def make(kind, cols):
    pre = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]) if kind == 'linear'
                else SimpleImputer(strategy='median'), cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CAT)])
    model = (LogisticRegression(C=0.5, max_iter=4000, class_weight='balanced') if kind == 'linear'
             else HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.05,
                                                l2_regularization=1.0, min_samples_leaf=30,
                                                random_state=42))
    return Pipeline([('f', pre), ('m', model)])


def run():
    d = pd.read_csv(ROOT / 'data/processed/event_study.csv', parse_dates=['release_time'])
    d_all, d_orig = d, d[d.event_type.isin(ORIGINAL_TYPES)]
    sets = {
        'A baseline (surprise+macro)|all': ('linear', CORE + MKT + PAXG, d_all),
        'A0 baseline|original 10 types': ('linear', CORE + MKT + PAXG, d_orig),
        'B +5m chart|all': ('linear', CORE + MKT + CHART, d_all),
        'C chart only|all': ('linear', CORE + CHART, d_all),
        'D all features|all': ('linear', CORE + MKT + CHART + PAXG, d_all),
        'A gbm|all': ('gbm', CORE + MKT + PAXG, d_all),
        'D gbm|all': ('gbm', CORE + MKT + CHART + PAXG, d_all),
        'B gbm|all': ('gbm', CORE + MKT + CHART, d_all),
    }
    rows = []
    for year in [2023, 2024, 2025, 2026]:
        s = pd.Timestamp(f'{year}-01-01', tz='UTC'); e = pd.Timestamp(f'{year + 1}-01-01', tz='UTC')
        for h in HORIZONS:
            for name, (kind, cols, data) in sets.items():
                x = data.dropna(subset=[f'return_{h}', 'signed_surprise_z'])
                tr, te = x[x.release_time < s], x[(x.release_time >= s) & (x.release_time < e)]
                if len(tr) < 200 or len(te) < 30:
                    continue
                ytr = tr[f'return_{h}'].gt(0).astype(int); yte = te[f'return_{h}'].gt(0)
                if yte.nunique() < 2:
                    continue
                cols_ok = [c for c in cols if c in x.columns]
                try:
                    m = make(kind, cols_ok)
                    m.fit(tr[cols_ok + CAT], ytr)
                    p = m.predict_proba(te[cols_ok + CAT])[:, 1]
                except Exception as exc:
                    print('skip', name, h, year, type(exc).__name__, exc); continue
                rows.append({'fold': year, 'horizon': h, 'config': name, 'n_train': len(tr),
                             'n_test': len(te), 'auc': roc_auc_score(yte, p),
                             'accuracy': accuracy_score(yte, p >= .5),
                             'brier': brier_score_loss(yte, p), 'base_up': yte.mean()})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / 'outputs/experiment_results.csv', index=False)
    print(f'folds evaluated: {len(out)}\n')
    print('=== mean AUC ===')
    print(out.pivot_table(index='config', columns='horizon', values='auc', aggfunc='mean').round(4).to_string())
    print('\n=== mean accuracy ===')
    print(out.pivot_table(index='config', columns='horizon', values='accuracy', aggfunc='mean').round(4).to_string())
    print('\n=== mean Brier (lower better) ===')
    print(out.pivot_table(index='config', columns='horizon', values='brier', aggfunc='mean').round(4).to_string())
    print('\n=== overall ===')
    print(out.groupby('config')[['auc', 'accuracy', 'brier']].mean().sort_values('auc', ascending=False).round(4).to_string())
    return out


if __name__ == '__main__':
    run()
