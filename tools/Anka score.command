#!/bin/bash
# Anka score – double-click. Scores every run's thought log and shows the comparison table
# (each scenario x condition: average and lowest-highest per measure). Free – no cloud brain is used.
cd "$(dirname "$0")/.." || exit 1
PY="$HOME/anka-env/bin/python"
"$PY" tools/score_run.py
"$PY" tools/score_run.py --compare
echo
read -r -p "Press Enter to close this window..."
