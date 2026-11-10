#!/bin/bash
# Teach Anka a face – double-click this file.
# It must be in the same folder as run_anka.py.

cd "$(dirname "$0")" || exit 1
PYTHON="$HOME/anka-env/bin/python"

echo "👤 Teach Anka a face"
echo

if pgrep -f "run_anka.py" >/dev/null 2>&1; then
    echo "⚠️  Anka is running right now and is using the camera."
    echo "   Put him to sleep first (Ctrl + C in his window), then try again."
    read -r -p "Press Enter to close..."
    exit 1
fi
if [ ! -x "$PYTHON" ]; then
    echo "❌ Can't find Anka's Python at $PYTHON"
    read -r -p "Press Enter to close..."
    exit 1
fi

read -r -p "Whose face should Anka learn? (e.g. Alex): " NAME
if [ -z "$NAME" ]; then
    echo "No name given – nothing to do."
    read -r -p "Press Enter to close..."
    exit 1
fi

echo
echo "Sit in front of the camera, alone in the picture. Turn your head a little, smile, look serious..."
echo
"$PYTHON" run_anka.py --learn-face "$NAME"
echo
"$PYTHON" run_anka.py --faces

echo
read -r -p "Done. Press Enter to close this window..."
