#!/bin/bash
# Anka turntest – double-click (Anka NOT running, simulator running). Finds a command that makes the duck turn. ~40 s.
cd "$(dirname "$0")/../ankasim" || exit 1
"$HOME/anka-env/bin/python" turntest.py
echo
read -r -p "Press Enter to close this window..."
