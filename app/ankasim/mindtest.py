"""Offline test of mind.py + pretend.py (no simulator, no brain): a pretend duck walks around the room with a drifting
odometry; do the landmarks keep his position estimate right?   python3 mindtest.py [runs]"""
import math, random, sys
import numpy as np
import room_layout as R
import pretend
from mind import Mind, wrap

def free(x, y, r=0.18):
    if not (r < x < R.ROOM_W - r and r < y < R.ROOM_H - r):
        return False
    return not any(x1 - r < x < x2 + r and y1 - r < y < y2 + r for (x1, y1, x2, y2, h) in R.FURNITURE.values())

def run(seed, steps=120, use_landmarks=True):
    rnd = random.Random(seed)
    x, y, hd = R.START[0], R.START[1], math.radians(R.START[2])        # truth, in room coordinates
    ox, oy, oth = 0.0, 0.0, 0.0                                         # odometry in his own frame (x forward, y left)
    m = Mind()
    m.predict((ox, oy, oth))
    for _ in range(steps):
        turn = rnd.uniform(-1.2, 1.2)
        hd += turn
        oth += turn * rnd.gauss(1.0, 0.03) + math.radians(rnd.gauss(0, 0.5))    # the gyro is good but not perfect
        step = rnd.choice([0.3, 0.5, 0.8])
        nx, ny = x + step * math.cos(hd), y + step * math.sin(hd)
        if not free(nx, ny):
            continue
        real = step * rnd.gauss(0.93, 0.05)                              # legs walk a bit short, odometry counts the order
        x, y = x + real * math.cos(hd), y + real * math.sin(hd)
        ox, oy = ox + step * math.cos(oth), oy + step * math.sin(oth)
        m.predict((ox, oy, oth))
        if use_landmarks:
            for look in (0.0, math.radians(-20), math.radians(20)):
                s = pretend.detect(x, y, math.degrees(hd), math.degrees(look), 0.0, rng=rnd)
                m.see(s, look, now=0)
                m.predict((ox, oy, oth))
    m.predict((ox, oy, oth))                                            # (the turns of the last blocked tries)
    # truth in his own frame
    x0, y0, h0 = R.START[0], R.START[1], math.radians(R.START[2])
    c, s_ = math.cos(h0), math.sin(h0)
    tx, ty = c * (x - x0) + s_ * (y - y0), -s_ * (x - x0) + c * (y - y0)
    est = m.pose() if use_landmarks else (ox, oy, oth)
    return math.hypot(est[0] - tx, est[1] - ty), math.degrees(wrap(est[2] - wrap(hd - h0))), m

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    for use in (False, True):
        e = [run(s, use_landmarks=use) for s in range(n)]
        pos = np.array([a[0] for a in e]); hdg = np.abs([a[1] for a in e])
        print(("with landmarks   " if use else "step counting only") + f": end error mean {pos.mean()*100:.0f} cm, worst {pos.max()*100:.0f} cm; heading mean {hdg.mean():.1f}°, worst {hdg.max():.1f}°"
              + (f"; landmarks known {np.mean([len(a[2].index) for a in e]):.1f}, corrections {np.mean([a[2].fixes for a in e]):.0f}" if use else ""))
