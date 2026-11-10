#!/bin/bash
# Anka eyes – double-click. Shows what the simulated duck sees and senses (camera, depth sensor, IMU, odometry).
# You can drive him by hand with the keys written at the bottom of the window.
cd "$(dirname "$0")/../ankasim" || exit 1
"$HOME/anka-env/bin/python" eyes.py "$@"
read -r -p "Press Enter to close this window..."
