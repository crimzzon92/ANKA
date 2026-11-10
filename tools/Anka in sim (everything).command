#!/bin/bash
# Anka in sim – EVERYTHING he can do, in one day (~17 min + the night). Needs "Start Anka sim.command" running first.
# Look around, see, walk to the sock, walk to the ball and kick it, find it, explore alone, tell what he found,
# go to the guitar, look up, sit, get shouted at + apology, best part of the day, sleep.
cd "$(dirname "$0")/.." || exit 1
PYTHON="$HOME/anka-env/bin/python"
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then open -a Ollama; sleep 8; fi
"$PYTHON" run_anka.py --simulate --mujoco --scenario all "$@"
echo
read -r -p "Press Enter to close this window..."
