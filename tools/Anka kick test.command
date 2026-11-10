#!/bin/bash
# Anka kick test – double-click (simulator running: "Start Anka sim.command"; Anka NOT running). ~40 s.
# Kicks with each foot, measures how much the leg joints move, and tries two duck sounds.
cd "$(dirname "$0")/../ankasim" || exit 1
mkdir -p ../test_logs
LOG="../test_logs/kicktest_$(date +%Y-%m-%d_%H%M).txt"
"$HOME/anka-env/bin/python" -u kicktest.py 2>&1 | tee "$LOG"
echo "Saved in $LOG"
echo
read -r -p "Press Enter to close this window..."
