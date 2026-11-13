import os, time
import faketest as F
S = F.S
S.MEMORY_FILE = "/tmp/faketest_memory.json"
if os.path.exists(S.MEMORY_FILE): os.remove(S.MEMORY_FILE)
b = S.SimBody(link=F.FakeLink(), wait=0.1); time.sleep(0.3)
b.do("look around")
for i in range(8):
    r = b.do("explore"); print(i, str(r)[:150], "| err %.0f cm %+.1f°" % F.err(b), sorted(b.mind.index))
for cmd in ("walk to the sofa", "walk to the table", "walk to the bookshelf", "walk to the sock", "walk to the ball", "turn to the window"):
    r = b.do(cmd); print(">>", cmd, "->", str(r)[:220], "| err %.0f cm %+.1f°" % F.err(b))
print("\n".join(b.report()))
import cv2; cv2.imwrite("/tmp/faketest_draw2.png", b.draw("faketest2"))
