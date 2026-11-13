#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! python -c "import streamlit" 2>/dev/null; then
  echo "Installing Python dependencies..."
  python -m pip install -r requirements.txt
fi

echo "Starting Gold News Lab at http://localhost:8501"
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
