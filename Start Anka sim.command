#!/bin/bash
# Start Anka sim – double-click. Starts Pollen's MuJoCo simulator with ANKA'S ROOM (instead of the plain scene).
# Leave this window open while you use Anka. Then start "Anka in sim.command" (his brain) and/or "Anka eyes.command".
# Use "--robot groundcontact" if the duck can't walk:   ./"Start Anka sim.command" --robot groundcontact
cd "$(dirname "$0")" || exit 1
PYTHON="$HOME/anka-env/bin/python"
POLLEN="$HOME/Pollen/microduck"
RL="$HOME/Pollen/microduck_rl"
ROBOT="allcollisions"; [ "$1" = "--robot" ] && ROBOT="$2"

if [ ! -x "$POLLEN/scripts/duck-sim" ]; then echo "❌ Can't find $POLLEN/scripts/duck-sim"; read -r -p "Press Enter..."; exit 1; fi
if ! command -v gst-launch-1.0 >/dev/null 2>&1; then
    echo "⚠️  GStreamer is missing – the duck's camera needs it. Install it once with:"
    echo "      brew install gstreamer libnice-gstreamer"
    echo "   Without it the simulator may not start (or has no camera). Continuing anyway..."
    echo
fi

echo "🏠 Building Anka's room ($ROBOT robot model)..."
"$PYTHON" ankasim/make_scene.py --rl "$RL" --robot "$ROBOT" || { read -r -p "Press Enter..."; exit 1; }

"$POLLEN/scripts/duck-sim" down >/dev/null 2>&1
export DUCK_SIM_SCENE="$(pwd)/ankasim/scene/anka_room.xml"
export DUCK_SIM_CAMERAS=a
echo "🦆 Starting the simulator (a window with the duck should open)..."
"$POLLEN/scripts/duck-sim"
echo
read -r -p "Press Enter to close this window (the simulator keeps running)..."
