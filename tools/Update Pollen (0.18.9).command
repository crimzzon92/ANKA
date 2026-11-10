#!/bin/bash
# Update Pollen – double-click ONCE (the simulator should NOT be running).
# Moves ~/Pollen/microduck from the 30 Oct version to release 0.18.9 (6 Nov 2026): the 13 ready-made emotions
# (angry, curious, defiant, devastated, excited, impatient, laugh, mmh, mock, no, sad, yes, yes_fast),
# the v9 walking policies, the simulator twin with backlash. NOT the newest main (10 Nov: big security changes
# for camera/connections that might disturb the simulator).
# The version you had is saved first - "Pollen go back.command" goes back to it.
REPO="$HOME/Pollen/microduck"
TARGET="da34a22"     # Merge pull request #553 – release 0.18.9
cd "$REPO" || { echo "❌ Can't find $REPO"; read -r -p "Press Enter..."; exit 1; }
if [ -n "$(git status --porcelain)" ]; then
    echo "❌ There are changes in $REPO that are not saved in git - nothing was changed. (git status shows them.)"
    read -r -p "Press Enter..."; exit 1
fi
if [ ! -f "$HOME/Pollen/microduck_before_update.txt" ]; then
    git rev-parse HEAD > "$HOME/Pollen/microduck_before_update.txt"
    echo "💾 Saved the version you had: $(cat "$HOME/Pollen/microduck_before_update.txt")"
fi
"$HOME/Pollen/microduck/scripts/duck-sim" down >/dev/null 2>&1
echo "⬇️  Fetching Pollen's code..."
git fetch origin || { echo "❌ git fetch failed (network?) - nothing was changed."; read -r -p "Press Enter..."; exit 1; }
git checkout --quiet "$TARGET" || { echo "❌ couldn't switch version - nothing was changed."; read -r -p "Press Enter..."; exit 1; }
echo "✅ Pollen is now at: $(git log -1 --format='%h %ci %s')"
echo "   Emotions in this version: $(ls robotd/emotions 2>/dev/null | grep json | sed 's/.json//' | tr '\n' ' ')"
echo
echo "Next: double-click 'Start Anka sim.command' – it rebuilds the duck's programs the first time (can take"
echo "several minutes) and then starts as usual."
read -r -p "Press Enter to close this window..."
