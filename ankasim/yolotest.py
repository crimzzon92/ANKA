"""yolotest: what does YOLO see in the simulator's camera picture? (Simulator running, Anka NOT running.)
Takes four pictures (ahead, left, right, down), runs YOLO on each, prints what it found and saves yolo_test.png."""
import math, os, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from duck_link import DuckLink
import yolo_eyes as Y

link = DuckLink().start()
print("connecting to the simulator ...", link.wait_ready(10))
time.sleep(1)
eyes = Y.YoloEyes(link, lambda: (0.0, 0.0))
print("models loaded:", list(eyes.models) or "NONE", "| can find:", sorted(eyes.names))
for p in eyes.problems:
    print("  problem:", p)
if not eyes.ready:
    sys.exit("No YOLO model could be loaded - see the problems above.")
shots = []
for label, target, pitch in (("ahead", (1.5, 0, 0.13), 0), ("left", (1.0, 0.85, 0.13), 0), ("right", (1.0, -0.85, 0.13), 0), ("down", (0.40, 0, -0.10), -28)):
    link.look_at(*target); time.sleep(2.0)
    frame = link.frame()
    if frame is None:
        print(label, ": no camera picture"); continue
    t = time.time(); res = eyes.see(frame, pitch, extra=True); dt = time.time() - t
    print(f"\n{label}: {len(eyes.last_boxes)} boxes in {dt*1000:.0f} ms")
    for name, conf, x0, y0, x1, y1 in eyes.last_boxes:
        print(f"   {name:8s} {conf:.2f}   box ({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f})")
    for name, kind, b, d, conf in res:
        print(f"   -> {name}: {abs(b):.0f}° to the {'left' if b > 0 else 'right'}, {d:.2f} m")
    img = frame.copy()
    for name, conf, x0, y0, x1, y1 in eyes.last_boxes:
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
        cv2.putText(img, f"{name} {conf:.2f}", (int(x0), max(14, int(y0) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    shots.append(img)
link.head(0, 0, 0, 0)
if shots:
    h = min(s.shape[0] for s in shots)
    cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "yolo_test.png"), np.hstack([s[:h] for s in shots]))
    print("\npicture saved: yolo_test.png (in the Duck folder)")
link.close()
