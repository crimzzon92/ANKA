#!/bin/bash
# Pollen go back – double-click to go back to the Pollen version you had before "Update Pollen".
REPO="$HOME/Pollen/microduck"
SAVED="$HOME/Pollen/microduck_before_update.txt"
cd "$REPO" || exit 1
[ -f "$SAVED" ] || { echo "❌ No saved version ($SAVED)."; read -r -p "Press Enter..."; exit 1; }
"$HOME/Pollen/microduck/scripts/duck-sim" down >/dev/null 2>&1
git checkout --quiet "$(cat "$SAVED")" && echo "✅ Back at: $(git log -1 --format='%h %ci %s')"
echo "(To follow Pollen's main again later: git checkout main && git pull)"
read -r -p "Press Enter to close this window..."
