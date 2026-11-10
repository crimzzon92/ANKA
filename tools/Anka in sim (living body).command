#!/bin/bash
# Anka in sim WITH the living body – double-click (needs "Start Anka sim.command" running).
cd "$(dirname "$0")/.." || exit 1
PYTHON="$HOME/anka-env/bin/python"
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then open -a Ollama; sleep 8; fi
"$PYTHON" run_anka.py --simulate --mujoco "$@"
echo
read -r -p "Press Enter to close this window..."
