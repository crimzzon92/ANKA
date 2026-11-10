#!/bin/bash
# Anka in sim WITHOUT the living body (--no-living): the comparison run – the still meter runs, the idle movements don't.
cd "$(dirname "$0")/.." || exit 1
PYTHON="$HOME/anka-env/bin/python"
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then open -a Ollama; sleep 8; fi
"$PYTHON" run_anka.py --simulate --mujoco --no-living "$@"
echo
read -r -p "Press Enter to close this window..."
