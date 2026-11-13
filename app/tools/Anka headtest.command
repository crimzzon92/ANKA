#!/bin/bash
# Anka headtest – double-click (Anka NOT running, simulator running). Measures the head/camera tilt standing and walking. ~30 s.
cd "$(dirname "$0")/../ankasim" || exit 1
mkdir -p ../test_logs
LOG="../test_logs/headtest_$(date +%Y-%m-%d_%H%M).txt"
"$HOME/anka-env/bin/python" headtest.py 2>&1 | tee "$LOG"
echo "Saved in $LOG"
echo
read -r -p "Press Enter to close this window..."
