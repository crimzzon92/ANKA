"""Draws the TRUE path of a replayed body test with different walking settings side by side (no Gemini).
    python3 tools/trail_plot.py test_logs/simulation_log_body_....txt [seed]   -> test_logs/trail_compare.png"""
import sys, os
LOG = sys.argv[1]; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
sys.argv = [sys.argv[0], LOG, "1"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_test as R
B = R.B
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
configs = [("before (sidestep up to 8, tight routes)", dict(ROUTE_DODGE=True, ROUTE_ROOM=0.03)),
           ("now (routes with room, max 2 sidesteps)", dict(ROUTE_DODGE=False, ROUTE_ROOM=0.12, ROUTE_DODGES=2))]
events = R.parse(LOG)
fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 5))
for ax, (name, sets) in zip(axes, configs):
    for k, v in sets.items():
        setattr(B, k, v)
    b, err, *_ = R.run(events, SEED)
    for x1, y1, x2, y2 in [v[:4] for v in B.FURNITURE.values()]:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, color="#d8cfc0"))
    ax.plot([0, B.ROOM_W, B.ROOM_W, 0, 0], [0, 0, B.ROOM_H, B.ROOM_H, 0], color="#555")
    xs, ys = zip(*b.trail)
    ax.plot(xs, ys, color="#f0b040", lw=1.2)
    ax.set_title(f"{name}\nwalked {sum(((xs[i+1]-xs[i])**2+(ys[i+1]-ys[i])**2)**.5 for i in range(len(xs)-1)):.1f} m", fontsize=9)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
out = os.path.join("test_logs", "trail_compare.png")
plt.tight_layout(); plt.savefig(out, dpi=90)
print("saved", out)
