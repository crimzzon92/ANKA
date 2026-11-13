#!/bin/bash
# Camera floor test – Anka's floor plan from the Mac's REAL webcam (no Gemini, costs nothing).
# 1. Put the MacBook on the FLOOR with the screen standing straight up, so the webcam is at a duck's eye height
#    (about 21 cm). Clear floor right in front of it.
# 2. Double-click this file. A window shows: the camera · the depth AI's guess (near = bright) · a radar view from
#    above (white = free floor, black = something in the way, rings at 1, 2 and 3 m).
# 3. Keys in the window: q = quit, s = save a picture, c = calibrate (put a box exactly 1 m straight ahead first),
#    n = switch to the next camera (it starts with the same camera Anka uses: CAMERA in settings.py).
# A picture is saved every 5 seconds in camera_floor_test.png. Stops by itself after 5 minutes.
# (The first time, macOS asks if Terminal may use the camera – say yes.)

cd "$(dirname "$0")" || exit 1
PYTHON="$HOME/anka-env/bin/python"

echo "📷 Camera floor test – put the Mac on the floor, screen straight up."
echo

if [ ! -x "$PYTHON" ]; then
    echo "❌ Can't find Anka's Python at $PYTHON"
    read -r -p "Press Enter to close..."
    exit 1
fi

# camera height in metres, camera number ("auto" = the one in settings.py), how many degrees the camera looks down
"$PYTHON" camera_floor_test.py 0.21 auto 0

echo
read -r -p "Done – the latest picture is camera_floor_test.png. Press Enter to close..."
