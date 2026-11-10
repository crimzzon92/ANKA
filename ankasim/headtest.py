"""Why is his head crooked while he walks?  Run with Anka NOT running, simulator running:  python headtest.py
Prints the head joints and how much the CAMERA is tilted sideways (roll) and up/down (pitch), standing and walking,
with the head free, held level, and held level with the neck pitched."""
import math, time
try:
    from .duck_link import DuckLink
except ImportError:
    from duck_link import DuckLink


def rot(q, v):
    w, x, y, z = q
    return (
        (1 - 2*(y*y+z*z))*v[0] + 2*(x*y - w*z)*v[1] + 2*(x*z + w*y)*v[2],
        2*(x*y + w*z)*v[0] + (1 - 2*(x*x+z*z))*v[1] + 2*(y*z - w*x)*v[2],
        2*(x*z - w*y)*v[0] + 2*(y*z + w*x)*v[1] + (1 - 2*(x*x+y*y))*v[2])


def camera_tilt(st):
    """(roll, pitch) of the camera in degrees against gravity. roll: the picture's horizon tilt. pitch: looking up (+) / down (-)."""
    cam = ((st.get("frames") or {}).get("camera") or {}).get("quat")
    g = (st.get("safety") or {}).get("gravity")
    if not cam or not any(cam) or not g:
        return None
    n = math.sqrt(sum(c * c for c in g)) or 1.0
    up = (-g[0] / n, -g[1] / n, -g[2] / n)               # 'up' in the trunk frame
    right = rot(cam, (1, 0, 0)); fwd = rot(cam, (0, 0, 1))
    roll = math.degrees(math.asin(max(-1, min(1, sum(a * b for a, b in zip(right, up))))))
    pitch = math.degrees(math.asin(max(-1, min(1, sum(a * b for a, b in zip(fwd, up))))))
    return roll, pitch


def sample(link, label, seconds, move=None):
    t_end = time.time() + seconds
    rows = []
    if move:
        link.move(*move, seconds=seconds)
    while time.time() < t_end:
        st = link.state() or {}
        h = st.get("head") or [0, 0, 0, 0]
        ct = camera_tilt(st)
        rows.append((h, ct))
        time.sleep(0.25)
    link.stop()
    def avg(i, f): 
        v = [f(r) for r in rows if f(r) is not None]
        return (sum(v) / len(v), min(v), max(v)) if v else (0, 0, 0)
    print(f"\\n{label}")
    for i, nm in enumerate(("neck", "pitch", "yaw", "roll")):
        a = avg(i, lambda r: math.degrees(r[0][i]))
        print(f"  head {nm:5}: mean {a[0]:+5.1f}  range {a[1]:+5.1f} .. {a[2]:+5.1f} deg")
    a = avg(0, lambda r: r[1][0] if r[1] else None); b = avg(0, lambda r: r[1][1] if r[1] else None)
    print(f"  CAMERA horizon tilt (roll): mean {a[0]:+5.1f}  range {a[1]:+5.1f} .. {a[2]:+5.1f} deg")
    print(f"  CAMERA looking up(+)/down(-): mean {b[0]:+5.1f}  range {b[1]:+5.1f} .. {b[2]:+5.1f} deg")


link = DuckLink().start()
print("connected:", link.wait_ready(6)); link.enable(True); time.sleep(2)
link.head_free(); sample(link, "1. standing, head NOT commanded", 3)
sample(link, "2. walking 0.3 m/s, head NOT commanded", 4, (0.3, 0, 0))
link.head(0.0, 0.0, 0.0, 0.0); time.sleep(1.5)
sample(link, "3. standing, head held at 0,0,0,0", 3)
sample(link, "4. walking 0.3 m/s, head held at 0,0,0,0", 4, (0.3, 0, 0))
link.look_at(1.5, 0.0, 0.13); time.sleep(1.5)
sample(link, "5. standing, head held by look_at(1.5, 0, 0.13)", 3)
sample(link, "6. walking 0.3 m/s, head held by look_at(1.5, 0, 0.13)", 4, (0.3, 0, 0))
print("\\nerrors:", link.errors())
link.close()
