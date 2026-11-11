#!/bin/bash
# Anka's start menu (macOS: double-click). Starts Anka, checks and installs what he needs, connects to a real duck.
# On Linux: bash "Anka menu.command"  (or python3 anka_menu.py)
cd "$(dirname "$0")" || exit 1
if [ -x "$HOME/anka-env/bin/python" ]; then PY="$HOME/anka-env/bin/python"
elif command -v python3 >/dev/null 2>&1; then PY="python3"
else echo "❌ Python 3 isn't installed. Get it from https://www.python.org/downloads/ and try again."; read -r -p "Press Enter..."; exit 1
fi
"$PY" anka_menu.py
