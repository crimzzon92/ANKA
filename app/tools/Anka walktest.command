#!/bin/bash
# Anka walktest – double-click (with Anka NOT running, simulator running). Checks that the duck really walks.
cd "$(dirname "$0")/../ankasim" || exit 1
"$HOME/anka-env/bin/python" walktest.py
echo
read -r -p "Press Enter to close this window..."
