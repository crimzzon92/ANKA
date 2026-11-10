#!/bin/bash
# Start Anka – double-click this file.
# It must be in the same folder as run_anka.py.

cd "$(dirname "$0")" || exit 1
PYTHON="$HOME/anka-env/bin/python"

echo "🦆 Starting Anka..."
echo

# 1) Is everything there?
if [ ! -f "run_anka.py" ]; then
    echo "❌ Can't find run_anka.py in this folder: $(pwd)"
    echo "   Put 'Start Anka.command' in the same folder as run_anka.py."
    read -r -p "Press Enter to close..."
    exit 1
fi
if [ ! -x "$PYTHON" ]; then
    echo "❌ Can't find Anka's Python at $PYTHON"
    echo "   Run the setup steps again (python3.12 -m venv ~/anka-env ...)."
    read -r -p "Press Enter to close..."
    exit 1
fi

# 2) Enough disk space? (a full disk makes the Mac freeze)
FREE_GB=$(df -g . | awk 'NR==2 {print $4}')
if [ -n "$FREE_GB" ] && [ "$FREE_GB" -lt 10 ]; then
    echo "⚠️  Only ${FREE_GB} GB free on the disk. Anka works best with at least 10 GB free."
    echo "   Free up some space, or your Mac may freeze."
    echo
fi

# 3) Is Ollama running? Otherwise start it.
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "🧠 Starting Ollama..."
    open -a Ollama
    for i in $(seq 1 30); do
        sleep 1
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            break
        fi
    done
    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "❌ Ollama didn't start. Open the Ollama app yourself and try again."
        read -r -p "Press Enter to close..."
        exit 1
    fi
fi
echo "🧠 Ollama is running"
echo

# 4) Start Anka
"$PYTHON" run_anka.py "$@"

echo
read -r -p "Anka is asleep. Press Enter to close this window..."
