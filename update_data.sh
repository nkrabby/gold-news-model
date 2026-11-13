#!/usr/bin/env bash
# Refresh all data, retrain, and leave the generated files ready for the dashboard.
set -euo pipefail
cd "$(dirname "$0")"

START=${1:-2020-01-01}
END=${2:-$(date -u +%F)}
CALENDAR_END=$(python -c "import pandas as pd; print((pd.Timestamp.utcnow()+pd.Timedelta(days=90)).strftime('%Y-%m-%d'))")

echo "[1/5] Downloading US calendar through $CALENDAR_END"
python src/fetch_calendar.py --start "$START" --end "$CALENDAR_END"

echo "[2/5] Downloading XAUUSD five-minute prices through $END"
bash src/fetch_prices.sh "$START" "$END"

echo "[3/5] Building event dataset"
python src/build_dataset.py

echo "[4/5] Training models and regenerating report"
python src/train.py

echo "[5/7] Refreshing upcoming releases and research links"
python src/upcoming_research.py

echo "[6/7] Training pre-release probabilistic forecast model"
python src/pre_release.py

echo "[7/7] Exporting standalone browser dashboard"
python src/export_standalone.py

echo "Update complete. Restart the dashboard if it is already running."
