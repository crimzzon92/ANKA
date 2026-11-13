"""faketest: runs SimBody against a FAKE simulator (no MuJoCo needed) to check the new mind / walk-to / explore code.

The fake duck moves kinematically in Anka's room, its odometry is a bit wrong (legs 6 % short, turns 4 % too big - like
real step counting), the depth sensor is a ray cast against the walls and furniture. Time runs FAST x SPEED.

    python faketest.py            (about a minute)
"""
import math, os, sys, threading, time
import numpy as np

SPEED = 8.0
_t0 = time.time()
class FastTime:                                    # a clock that runs SPEED times faster, so tests are short
    strftime = staticmethod(time.strftime)
    @staticmethod
    def time(): return _t0 + (time.time() - _t0) * SPEED
    @staticmethod
    def sleep(s): time.sleep(max(0.0, s) / SPEED)

import room_layout as R
import mind as M
import pretend as P
import simbody as S
import duck_link as D
for mod in (M, P, S, D):
    if hasattr(mod, "time"):
        mod.time = FastTime
sys.modules["time"] = FastTime if False else sys.modules["time"]

BOXES = [(x1, y1, x2, y2) for (x1, y1, x2, y2, h) in R.FURNITURE.values()]
def ray(x, y, a, maxd=4.0):
    """distance along direction a (rad) to the walls / furniture (room coordinates)"""
    dx, dy = math.cos(a), math.sin(a)
    best = maxd
    for d in np.arange(0.05, maxd, 0.03):
        px, py = x + dx * d, y + dy * d
        if not (0.0 < px < R.ROOM_W and 0.0 < py < R.ROOM_H) or any(b[0] <= px <= b[2] and b[1] <= py <= b[3] for b in BOXES):
            return d
    return best

def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return (w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2)

class FakeLink:
    def __init__(self):
        self.x, self.y, self.h = R.START[0], R.START[1], math.radians(R.START[2])
        self.ox = self.oy = self.oyaw = 0.0
        self.cmd = (0.0, 0.0, 0.0); self.head_yaw = 0.0; self.head_pitch = 0.0
        self._lock = threading.Lock(); self.sitting = False
        threading.Thread(target=self._run, daemon=True).start()
    def start(self): return self
    def wait_ready(self, t=1): return {"robot": True, "tof": True, "camera": True}
    def errors(self): return {}
    def _run(self):
        last = FastTime.time()
        while True:
            time.sleep(0.01 / 1)
            now = FastTime.time(); dt = now - last; last = now
            vx, vy, w = self.cmd
            if abs(vx) < 0.3 and vx > 0: vx = 0.0          # the policy's dead zones
            if abs(w) < 0.9: w = 0.0
            self.h += w * dt
            self.x += math.cos(self.h) * vx * dt
            self.y += math.sin(self.h) * vx * dt
            # keep out of walls / furniture
            if not (0.1 < self.x < R.ROOM_W - 0.1 and 0.1 < self.y < R.ROOM_H - 0.1) or any(b[0]-0.1 <= self.x <= b[2]+0.1 and b[1]-0.1 <= self.y <= b[3]+0.1 for b in BOXES):
                self.x -= math.cos(self.h) * vx * dt; self.y -= math.sin(self.h) * vx * dt
            # odometry in the simulator's frame: 6 % short legs, 4 % too much turn
            self.oyaw += w * dt * 1.04
            self.ox += math.cos(self.oyaw) * vx * dt * 0.94
            self.oy += math.sin(self.oyaw) * vx * dt * 0.94
    # --- the interface SimBody uses
    def move(self, vx=0.0, vy=0.0, vyaw=0.0, seconds=None):
        self.cmd = (vx, vy, vyaw)
        if seconds:   # (2026-11-10: like the real link - stop after `seconds`)
            t0 = FastTime.time()
            def _end(c=self.cmd):
                while FastTime.time() < t0 + seconds: time.sleep(0.002)
                if self.cmd == c: self.cmd = (0.0, 0.0, 0.0)
            threading.Thread(target=_end, daemon=True).start()
    def stop(self): self.cmd = (0.0, 0.0, 0.0)
    def head(self, neck_pitch=0.0, head_pitch=0.0, head_yaw=0.0, head_roll=0.0, seconds=None): self.head_yaw, self.head_pitch = head_yaw, head_pitch + neck_pitch
    def head_free(self): self.head_yaw = self.head_pitch = 0.0
    def pose(self, z=0.0, roll=0.0, pitch=0.0, active=True): self.body_pose = (z, roll, pitch, active)
    def mouth(self, open_=0.0): self.beak = open_
    def look_at(self, x, y, z, neck_pitch=0.0):
        self.head_yaw = math.atan2(y, x); self.head_pitch = math.atan2(z - 0.20 + 0.12, math.hypot(x, y))
        return {"head": {}, "clamped": False}
    def do(self, skill): return {"ok": True}
    def enable(self, on=True): return True
    def fallen(self): return False
    def skills(self): return ["sit_toggle", "ball_kick_left", "ball_kick_right"]
    def odom(self): return self.ox, self.oy, self.oyaw
    def odom_anka(self): return self.x, self.y, math.degrees(self.h) % 360
    def truth(self): return {"x": self.x, "y": self.y, "heading": math.degrees(self.h) % 360}
    def realtime(self): return 1.0
    def frame(self): return None
    def frame_info(self): return {"age": 0.1, "count": 1}
    def call(self, *a, **k): return {}
    def state(self):
        c = math.cos((math.pi / 2 - self.head_pitch) / 2); s = math.sin((math.pi / 2 - self.head_pitch) / 2)
        cam = qmul((math.cos(self.head_yaw / 2), 0, 0, math.sin(self.head_yaw / 2)), (c, 0, s, 0))
        return {"odom": {"position": [self.ox, self.oy, 0.0], "yaw": self.oyaw},
                "imu": {"quat": [1, 0, 0, 0]}, "safety": {"fallen": False, "gravity": [0.0, 0.0, -1.0]},
                "frames": {"camera": {"quat": list(cam), "pos": [0.05, 0, 0.1]},
                           "tof": {"quat": [1, 0, 0, 0], "pos": [0.06, 0, 0.0]}}}
    def tof(self):
        m = np.full((8, 8), np.nan, np.float32)
        half = math.radians(45) / 2
        edges = np.linspace(-half, half, 9); cen = (edges[:-1] + edges[1:]) / 2
        height = S.TRUNK_H
        for c in range(8):
            d = ray(self.x, self.y, self.h - cen[c])           # wall ahead of this column (horizontal distance)
            for r in range(8):
                el = -cen[r]
                # level sensor: the rows look from slightly up to slightly down; down rays hit the floor
                if el < 0:
                    floor_d = height / math.tan(-el)
                    dist = min(d / math.cos(el), floor_d / math.cos(el)) if floor_d < d else d / math.cos(el)
                else:
                    dist = d / math.cos(el)
                m[r, c] = dist if dist < 4.0 else np.nan
        return {"m": m, "age": 0.0}

def say(body, text):
    t0 = time.time()
    print(f"\n>>> {text}")
    r = body.do(text)
    print("   ", str(r)[:400])
    return r

def err(body):
    t = body.link.truth(); p = body.mind.pose()
    tx, ty = body._room_to_own(t["x"], t["y"]); th = S._wrap(math.radians(t["heading"]) - math.radians((body._wake or R.START)[2]))
    return math.hypot(p[0] - tx, p[1] - ty) * 100, math.degrees(S._wrap(p[2] - th))

if __name__ == "__main__":
    S.MEMORY_FILE = "/tmp/faketest_memory.json"
    if os.path.exists(S.MEMORY_FILE): os.remove(S.MEMORY_FILE)
    link = FakeLink()
    body = S.SimBody(link=link, wait=0.1)
    time.sleep(0.5)
    for cmd in ("look around", "walk to the sofa", "explore", "walk to the ball", "look around"):
        say(body, cmd)
        print(f"    mind error {err(body)[0]:.0f} cm / {err(body)[1]:+.1f}°   landmarks known: {sorted(body.mind.index)}")
    print("\n--- describe() ---")
    print(body.describe() if hasattr(body, "describe") else "(no describe)")
    print("\n--- report ---")
    print("\n".join(body.report()))
    try:
        import cv2
        cv2.imwrite("/tmp/faketest_draw.png", body.draw("faketest"))
        print("picture: /tmp/faketest_draw.png")
    except Exception as e:
        print("draw failed:", type(e).__name__, e)
