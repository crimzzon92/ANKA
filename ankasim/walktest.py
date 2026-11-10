"""Does the duck really walk when asked?  Run it with Anka NOT running.  python walktest.py"""
import time, math
try:
    from .duck_link import DuckLink
except ImportError:
    from duck_link import DuckLink

link = DuckLink().start()
print("connected:", link.wait_ready(6))
time.sleep(1)
s = link.state() or {}
print("policy:", s.get("policy"), "| safety:", s.get("safety"), "| loop:", s.get("loop"))
print("skills:", link.skills())
print("world speed (1.0 = real time):", link.realtime())
print("enable ->", link.enable(True))
time.sleep(2)
for name, mv, secs in (("walk forward 0.30 m/s", (0.30, 0, 0), 4), ("turn left 0.5 rad/s", (0, 0, 0.5), 4), ("walk backward 0.20 m/s", (-0.20, 0, 0), 3)):
    o0, t0 = link.odom_anka(), link.truth()
    link.move(*mv, seconds=secs)
    time.sleep(secs + 1)
    o1, t1 = link.odom_anka(), link.truth()
    f = lambda o: "none" if not o else f"x {o[0]:.2f} y {o[1]:.2f} heading {o[2]:.0f}"
    print(f"\n{name} for {secs} s:\n  odometry  {f(o0)}  ->  {f(o1)}")
    if t0 and t1:
        print(f"  truth     x {t0['x']:.2f} y {t0['y']:.2f} heading {t0['heading']:.0f}  ->  x {t1['x']:.2f} y {t1['y']:.2f} heading {t1['heading']:.0f}")
    print("  fallen:", link.fallen(), "| policy:", (link.state() or {}).get("policy"), "| move applied:", ((link.state() or {}).get("move") or {}))
print("\nerrors:", link.errors())
link.close()
