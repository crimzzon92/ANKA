#!/bin/bash
# Anka's start menu on Linux: run it in a terminal ( ./"Anka menu (linux).sh" ), or right-click it -> "Run as a program"
# in your file manager. The menu's option 5 can also add "Anka" to your apps menu. (Untested on Linux so far.)
cd "$(dirname "$0")" || exit 1
PY=""
for p in "$HOME/anka-env/bin/python"; do
    if [ -x "$p" ]; then PY="$p"; break; fi
done
[ -z "$PY" ] && PY="$(command -v python3)"
if [ -z "$PY" ]; then echo "❌ Python 3 isn't installed (sudo apt install python3 python3-venv)."; read -r -p "Press Enter..."; exit 1; fi
"$PY" anka_menu.py
read -r -p "Press Enter to close..." 2>/dev/null
