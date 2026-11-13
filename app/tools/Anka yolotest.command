#!/bin/bash
# Anka yolotest - double-click. Simulator must be running (Start Anka sim.command), Anka must NOT be running.
# Installs ultralytics (YOLO) the first time (a few hundred MB) and shows what YOLO sees in the simulator's picture.
cd "$(dirname "$0")/../ankasim" || exit 1
PYTHON="$HOME/anka-env/bin/python"
if ! "$PYTHON" -c "import ultralytics" 2>/dev/null; then
  echo "Installing ultralytics (YOLO) - this takes a few minutes the first time ..."
  "$PYTHON" -m pip install ultralytics
fi
"$PYTHON" yolotest.py
echo
read -r -p "Press Enter to close this window..."
