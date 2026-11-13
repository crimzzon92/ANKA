"""livingtest: the living body against the FAKE duck (no MuJoCo): idle movements, a voice, a rolling ball, a move."""
import math, os, sys, time
import faketest as F
import living as L
L.time = F.FastTime
S = F.S
S.MEMORY_FILE = os.path.expanduser("~/livingtest_memory.json")
if os.path.exists(S.MEMORY_FILE): os.remove(S.MEMORY_FILE)
b = S.SimBody(link=F.FakeLink(), wait=0.1)
def wait(s):
    end = F.FastTime.time() + s
    while F.FastTime.time() < end:
        F.FastTime.sleep(0.5)
print(">> look around:", b.do("look around")[0][:60])
wait(30); print("after 30 s idle:", dict(b.living.glances))
b.hear("Alex"); wait(6); print("after a voice:", dict(b.living.glances))
print(">> walk forward 50 cm:", b.do("walk forward 50 cm"))
# a rolling ball: put fake sightings into his eyes, 1.2 m ahead, rolling to the left
t0 = F.FastTime.time()
real_detect = b._detect; b._detect = lambda yaw, pitch: []   # (only the made-up ball, no other sightings)
for i in range(30):
    now = F.FastTime.time()
    b._visible["ball"] = {"b": math.degrees(math.atan2(0.06 * i, 1.2)) - b.living.cur["head_yaw"] * 57.3 if b.living.cur else math.degrees(math.atan2(0.06 * i, 1.2)),
                          "d": math.hypot(1.2, 0.06 * i), "t": now, "head": math.degrees(b.living.cur["head_yaw"]) if b.living.cur else 0.0}
    F.FastTime.sleep(0.1)
print("after a rolling ball:", dict(b.living.glances), "head yaw now %.0f°" % math.degrees((b.living.cur or {}).get("head_yaw", 0)))
b._detect = real_detect
wait(10)
print(">> turn left 90:", b.do("turn left 90"), "| head after the move goes back home:", b.living.moved_head)
wait(20)
print("\n".join(b.living.report()))
