#!/usr/bin/env bash
# Dukascopy timestamps are Unix milliseconds UTC. End date is exclusive.
set -euo pipefail
START=${1:-2020-01-01}
END=${2:-2026-11-11}
npx --yes dukascopy-node -i xauusd -from "$START" -to "$END" -t m5 -f csv -v
mkdir -p data/raw
mv download/xauusd-m5-bid-*.csv data/raw/xauusd_m5.csv
