"""Finds out how this duck can be made to turn.  Run with Anka NOT running, simulator running:  python turntest.py
It tries several 'forward + turn' commands for 3 s each, measures the real heading change and writes the best one to
turn_calibration.json, which SimBody then uses."""
import json, math, os, time
try:
    from .duck_link import DuckLink
except ImportError:
    from duck_link import DuckLink

HERE = os.path.dirname(os.path.abspath(__file__))
link = DuckLink().start()
print("connected:", link.wait_ready(6))
link.enable(True)
time.sleep(2)
tests = [(0.0, 0.5), (0.0, 1.0), (0.0, 1.5), (0.1, 1.0), (0.15, 1.0), (0.2, 1.0), (0.3, 0.8), (0.3, 1.2)]
rows = []
for vx, vyaw in tests:
    o0 = link.odom()
    link.move(vx, 0, vyaw, seconds=3.0)
    time.sleep(3.6)
    link.stop()
    time.sleep(1.5)
    o1 = link.odom()
    if not o0 or not o1:
        print("no odometry"); continue
    dyaw = math.degrees((o1[2] - o0[2] + math.pi) % (2 * math.pi) - math.pi)
    dist = math.hypot(o1[0] - o0[0], o1[1] - o0[1])
    fell = link.fallen()
    rows.append((vx, vyaw, dyaw, dist, fell))
    print(f"forward {vx:.2f} m/s + turn {vyaw:.1f} rad/s for 3 s -> turned {dyaw:+.0f} deg, moved {dist*100:.0f} cm" + ("  FELL" if fell else ""))
    if fell:
        link.enable(True); time.sleep(5)
cal_path = os.path.join(HERE, "turn_calibration.json")
try:
    cal = json.load(open(cal_path))
except Exception:
    cal = {}
good = [r for r in rows if not r[4] and r[2] > 20]
if good:
    best = min(good, key=lambda r: (r[3] / max(r[2], 1)))   # the one that turns well with the least walking around
    cal.update({"vx": best[0], "rate": best[1], "deg_per_s": best[2] / 3.0})
    print("\nTurning:", best)
else:
    print("\nNo command turned him more than 20 degrees in 3 s. Paste this list back to Claude.")

print("\nBackwards:")
link.enable(True); time.sleep(1)
back_ok = None
for speed in (-0.2, -0.3, -0.4, -0.5):
    o0 = link.odom(); link.move(speed, 0, 0, seconds=3.0); time.sleep(3.6); link.stop(); time.sleep(1.5)
    o1 = link.odom()
    # distance along the way he faced
    d = ((o1[0] - o0[0]) * math.cos(o0[2]) + (o1[1] - o0[1]) * math.sin(o0[2])) * 100
    print(f"  {speed:+.1f} m/s for 3 s -> moved {d:+.0f} cm along his facing" + ("  FELL" if link.fallen() else ""))
    if link.fallen():
        link.enable(True); time.sleep(5)
    elif d < -15 and back_ok is None:
        back_ok = abs(speed)
if back_ok:
    cal["back"] = back_ok
    print("Backwards works from", back_ok, "m/s - saved.")
else:
    print("He did not walk backwards at any speed - Anka will be told that backing up does not work.")
    cal["back"] = 0
json.dump(cal, open(cal_path, "w"))
print("saved", cal)
link.close()
