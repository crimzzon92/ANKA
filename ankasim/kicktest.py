"""Does the kick really move his legs?  (2026-11-10)  Run it with the simulator running and Anka NOT running.
Double-click tools/"Anka kick test.command".  It kicks with each foot (with and without the body pose switched on
first, like the living body does), and measures how much each leg joint moves; it also tries a duck sound."""
import math, time
try:
    from .duck_link import DuckLink
except ImportError:
    from duck_link import DuckLink


def joints(st):
    """The joint positions in robot.state, whatever the key is called."""
    for key in ("positions", "joints", "q", "joint_positions"):
        v = st.get(key)
        if isinstance(v, dict):
            v = v.get("positions") or v.get("q") or list(v.values())
        if isinstance(v, (list, tuple)) and v and all(isinstance(x, (int, float)) for x in v):
            return list(v), key
    return None, None


link = DuckLink().start()
print("connected:", link.wait_ready(6))
time.sleep(1)
st = link.state() or {}
print("robot.state keys:", sorted(st.keys()))
print("skills:", link.skills(), "| world speed (1.0 = real time):", link.realtime())
print("enable ->", link.enable(True))
time.sleep(3)
q0, key = joints(st)
print("joint positions found under:", key, "| count:", len(q0) if q0 else 0)


def measure(label, before=None):
    if before:
        before()
    st = link.state() or {}
    start, _ = joints(st)
    t0 = link.truth()
    biggest = [0.0] * (len(start) if start else 0)
    labels = set()
    end = time.time() + 4.0
    answer = None
    sent = False
    while time.time() < end:
        if not sent:
            answer = link.do(label)
            sent = True
        s = link.state() or {}
        labels.add(str(s.get("policy", ""))[:60])
        q, _ = joints(s)
        if q and start:
            biggest = [max(b, abs(a - c)) for b, a, c in zip(biggest, q, start)]
        time.sleep(0.05)
    t1 = link.truth()
    big = sorted(((round(math.degrees(b), 1), i) for i, b in enumerate(biggest)), reverse=True)[:5]
    print(f"\n{label}{(' – ' + before.__name__) if before else ''}:")
    print("   robot answered:", answer)
    print("   biggest joint moves (degrees, joint number):", big)
    print("   what the robot said it was running:", labels)
    if t0 and t1:
        print(f"   trunk moved {math.hypot(t1['x'] - t0['x'], t1['y'] - t0['y']) * 100:.1f} cm, fallen: {link.fallen()}")
    time.sleep(2)


def pose_on_then_off():
    for _ in range(10):
        link.pose(-0.01, 0.05, 0.0, True)
        time.sleep(0.1)
    link.pose(0.0, 0.0, 0.0, True); time.sleep(0.4)
    link.pose(0.0, 0.0, 0.0, False); time.sleep(0.2)


def head_down():
    """Like Anka before a kick: the head held looking down at his feet (re-sent all the time by the link)."""
    link.look_at(0.40, 0.0, -0.10)
    time.sleep(1.0)


def head_down_and_pose():
    pose_on_then_off()
    head_down()


have = link.skills()
kicks = [k for k in have if "kick" in k.lower()]
for name in kicks:
    measure(name)
for name in kicks[:1]:
    measure(name, before=pose_on_then_off)
    measure(name, before=head_down)
    measure(name, before=head_down_and_pose)
    link.head(0.0, 0.0, 0.0, 0.0)

print("\nduck sound 'chirp' ->", link.call("robot.sound", {"tag": "chirp"}))
time.sleep(1.5)
print("duck sound 'greet' ->", link.call("robot.sound", {"tag": "greet"}))
print("\nerrors:", link.errors())
link.close()
import os
os._exit(0)      # (the link threads must not keep the window waiting)
