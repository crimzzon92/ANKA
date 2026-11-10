#!/bin/bash
cd "$(dirname "$0")/../ankasim" && "$HOME/anka-env/bin/python" faketest2.py
echo; echo "Done - press Enter"; read
