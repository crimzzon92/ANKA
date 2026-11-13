#!/bin/bash
# Anka's start menu (macOS: double-click). Starts Anka, checks and installs what he needs, connects to a real duck.
# On Linux use "Anka menu (linux).sh"; on Windows "Anka menu (win).bat".
cd "$(dirname "$0")" || exit 1
PY=""
for p in "$HOME/anka-env/bin/python"; do
    if [ -x "$p" ]; then PY="$p"; break; fi
done
if [ -z "$PY" ]; then
    if command -v python3 >/dev/null 2>&1; then PY="python3"
    else echo "❌ Python 3 isn't installed. Get it from https://www.python.org/downloads/ and try again."; read -r -p "Press Enter..."; exit 1
    fi
fi
"$PY" app/anka_menu.py
