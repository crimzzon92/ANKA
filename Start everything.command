#!/bin/bash
# Start everything – double-click. Does everything in order:
#   1. builds Anka's room and starts Pollen's MuJoCo simulator
#   2. waits until the simulator is up and loads the walking policy (it must be loaded by hand after every start)
#   3. starts Anka's brain with the simulated body (and Ollama if it is not running)
# Options:  ./"Start everything.command" --robot groundcontact   (if the duck can't walk)
#           ./"Start everything.command" --newborn               (Anka forgets the room: no old room memory)
#           ./"Start everything.command" --no-eyes               (don't open the "Anka eyes" camera window)
cd "$(dirname "$0")" || exit 1
PYTHON="$HOME/anka-env/bin/python"
POLLEN="$HOME/Pollen/microduck"
RL="$HOME/Pollen/microduck_rl"
POLICY="$HOME/.cache/duck-sim/policies/current/velstand.onnx"
SOCK="${DUCK_SIM_STATE:-$HOME/.cache/duck-sim}/duck-a.sock"
ROBOT="allcollisions"; EXTRA=(); EYES=1
while [ $# -gt 0 ]; do
    case "$1" in
        --robot) ROBOT="$2"; shift 2 ;;
        --no-eyes) EYES=0; shift ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

if [ ! -x "$POLLEN/scripts/duck-sim" ]; then echo "❌ Can't find $POLLEN/scripts/duck-sim"; read -r -p "Press Enter..."; exit 1; fi
if [ ! -x "$PYTHON" ]; then echo "❌ Can't find $PYTHON"; read -r -p "Press Enter..."; exit 1; fi
if ! command -v gst-launch-1.0 >/dev/null 2>&1; then
    echo "⚠️  GStreamer is missing (the camera needs it): brew install gstreamer libnice-gstreamer"
fi

echo "🏠 1/3  Building Anka's room ($ROBOT robot model)..."
"$PYTHON" ankasim/make_scene.py --rl "$RL" --robot "$ROBOT" || { read -r -p "Press Enter..."; exit 1; }

echo "🦆 1/3  Starting the simulator (a window with the duck opens; the first start is slow)..."
"$POLLEN/scripts/duck-sim" down >/dev/null 2>&1
export DUCK_SIM_SCENE="$(pwd)/ankasim/scene/anka_room.xml"
export DUCK_SIM_CAMERAS=a
"$POLLEN/scripts/duck-sim"

echo "⏳ 2/3  Waiting for the simulator to be ready..."
for i in $(seq 1 90); do [ -S "$SOCK" ] && break; sleep 2; done
if [ ! -S "$SOCK" ]; then echo "❌ The simulator did not come up (no $SOCK). Look at its window."; read -r -p "Press Enter..."; exit 1; fi
sleep 4

echo "🦿 2/3  Loading the walking policy..."
OK=0
for i in 1 2 3 4 5 6 7 8; do
    if "$POLLEN/scripts/duck-sim" ctl policy load walk "$POLICY"; then OK=1; break; fi
    echo "   (not ready yet, trying again...)"; sleep 3
done
if [ $OK -ne 1 ]; then echo "❌ Could not load the walking policy ($POLICY)."; read -r -p "Press Enter..."; exit 1; fi
sleep 3
"$POLLEN/scripts/duck-sim" realtime 2>/dev/null | head -1

if [ $EYES -eq 1 ]; then
    echo "👀 Opening Anka's eyes (his camera, with boxes on what he finds) in a new Terminal window..."
    chmod +x "tools/Anka eyes.command" 2>/dev/null
    open -a Terminal "tools/Anka eyes.command"
fi

echo "🧠 3/3  Starting Anka..."
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then open -a Ollama; sleep 8; fi
"$PYTHON" run_anka.py --simulate --mujoco "${EXTRA[@]}"
echo
echo "Anka is asleep. The simulator keeps running in the background (stop it with: $POLLEN/scripts/duck-sim down)."
read -r -p "Press Enter to close this window..."
