"""Offline test bench for Anka's body (no Gemini, no window, no cost).

Replays the moves (ACT lines) and the people's movements from a test log, many times with different random
seeds, and measures how far off his idea of where he is (and which way he faces) ends up.

Run from the Duck folder:
    python3 offline_tests/replay_test.py test_logs/simulation_log_body_2026-11-01_1125.txt 30
(second number = how many random seeds; add  --depth  to switch the depth AI on, --verbose for the worst run)
"""
import math, os, re, sys, types, random

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.chdir(os.path.dirname(HERE))
sys.argv = [sys.argv[0], "--simulate", "--body"] + [a for a in sys.argv[1:]]
RAW = sys.argv[3:]
SETS = {}
PROCS = 4
for i, a in enumerate(RAW):
    if a == "--set":
        k, _, v = RAW[i + 1].partition("=")
        SETS[k] = eval(v)
    if a == "--procs":
        PROCS = int(RAW[i + 1])
ARGS = [a for i, a in enumerate(RAW) if not a.startswith("--") and not (i > 0 and RAW[i - 1] in ("--set", "--procs"))]
LOG = ARGS[0]
SEEDS = int(ARGS[1]) if len(ARGS) > 1 else 20
SEED0 = int(ARGS[2]) if len(ARGS) > 2 else 0   # first seed (so several runs can work side by side)
USE_DEPTH = "--depth" in sys.argv
VERBOSE = "--verbose" in sys.argv

fw = types.ModuleType("faster_whisper"); fw.WhisperModel = object; sys.modules["faster_whisper"] = fw
import settings
settings.DEPTH_AI = USE_DEPTH
import anka.body as B

# a fake clock: every look takes 0.2 s
clock = [1000.0]
B.time = types.SimpleNamespace(time=lambda: clock[0], sleep=lambda s: None, monotonic=lambda: clock[0],
                               perf_counter=lambda: clock[0])
if hasattr(B, "DEPTH_AI"):
    B.DEPTH_AI = USE_DEPTH
for _k, _v in SETS.items():
    assert hasattr(B, _k), _k
    setattr(B, _k, _v)


def parse(path):
    events = []
    for line in open(path, encoding="utf-8"):
        m = re.search(r"in the room: (.*)$", line)
        if m and "🎬" in line:
            events.append(("room", m.group(1).strip()))
            continue
        m = re.match(r"🦶 \(Anka: (.+?) → ", line)
        if m:
            events.append(("act", m.group(1).strip()))
    return events


def spot_of(text):
    people = {}
    if text != "nobody":
        for part in text.split(","):
            m = re.match(r"\s*(\w+) by the (\w+)", part)
            if m:
                people[m.group(1)] = m.group(2)
    return people


def run(events, seed, looks_per_move=12):
    random.seed(seed)
    clock[0] = 1000.0
    b = B.RealSensesBody()
    b.place_person("Alex", "sofa")
    worst, worst_at, series = 0.0, None, []
    n = 0
    for kind, what in events:
        if kind == "room":
            want = spot_of(what)
            for p in list(b.people):
                if p not in want:
                    b.place_person(p, None)
            for p, s in want.items():
                b.place_person(p, s)
            continue
        res = ""
        try:
            r_ = b.do(what)
            res = r_[0] if isinstance(r_, tuple) and r_[0] else ""
        except Exception as e:           # (a move the bench can't follow – skip it)
            print("  (skipped:", what, e, ")")
        n += 1
        for _ in range(looks_per_move):
            clock[0] += 0.2
            try:
                b.remember_view()
            except Exception as e:
                print("  look failed:", repr(e)); break
        err = math.hypot(b.est_x - b.x, b.est_y - b.y)
        herr = B._wrap(b.est_heading() - b.heading)
        series.append((n, what, err, herr, res))
        if err > worst:
            worst, worst_at = err, (n, what)
    err = math.hypot(b.est_x - b.x, b.est_y - b.y)
    herr = B._wrap(b.est_heading() - b.heading)
    return b, err, herr, worst, worst_at, series


def map_quality(b):
    import numpy as np
    boxes = [(0, 0, B.ROOM_W, B.ROOM_H)]
    solid = [v[:4] for v in B.FURNITURE.values()]
    def dist_to_solid(x, y):
        d = min(x, B.ROOM_W - x, y, B.ROOM_H - y)           # the walls (inside the room)
        d = abs(d)
        for x1, y1, x2, y2 in solid:
            dx = max(x1 - x, 0, x - x2); dy = max(y1 - y, 0, y - y2)
            d = min(d, math.hypot(dx, dy))
        return d
    ii, kk = np.nonzero(b.floor > B.MAP_SURE)
    if not len(ii):
        return 0.0, 0.0
    xs = B.MAP_ORIGIN[0] + (ii + 0.5) * B.MAP_CELL
    ys = B.MAP_ORIGIN[1] + (kk + 0.5) * B.MAP_CELL
    good = np.mean([dist_to_solid(x, y) < 0.10 for x, y in zip(xs, ys)])
    pts = []
    for t in np.arange(0, B.ROOM_W, 0.05):
        pts += [(t, 0), (t, B.ROOM_H)]
    for t in np.arange(0, B.ROOM_H, 0.05):
        pts += [(0, t), (B.ROOM_W, t)]
    for x1, y1, x2, y2 in solid:
        for t in np.arange(x1, x2, 0.05):
            pts += [(t, y1), (t, y2)]
        for t in np.arange(y1, y2, 0.05):
            pts += [(x1, t), (x2, t)]
    found = np.mean([np.min(np.hypot(xs - px, ys - py)) < 0.10 for px, py in pts])
    return float(good), float(found)


def one(seed):
    import io, contextlib
    events = parse(LOG)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        b, err, herr, worst, worst_at, series = run(events, seed)
    problems = [l for l in buf.getvalue().splitlines() if "skipped" in l or "look failed" in l]
    if problems:
        print(f"  !! seed {seed}: {len(problems)} problem(s), first: {problems[0][:200]}", flush=True)
    return seed, err, herr, worst, worst_at, series, b.fixes, b.scan_fixes, len(b.walls), b.falls, map_quality(b)


if __name__ == "__main__":
    import multiprocessing as mp
    events = parse(LOG)
    print(f"{sum(1 for k, _ in events if k == 'act')} moves in {LOG}; seeds {SEED0}..{SEED0+SEEDS-1}; "
          f"depth AI {'on' if USE_DEPTH else 'off'}; settings {SETS or 'as in body.py'}", flush=True)
    results = []
    with mp.Pool(PROCS) as pool:
        for r in pool.imap_unordered(one, range(SEED0, SEED0 + SEEDS)):
            results.append(r)
            s_, err, herr, worst, worst_at, series, fx, sf, nw, fl, mq = r
            print(f"seed {s_:2d}: end {err*100:5.0f} cm  heading {herr:+6.1f}°  worst {worst*100:5.0f} cm at move {worst_at}  "
                  f"fixes {fx}  scan_fixes {sf}  walls {nw}  falls {fl}  map right {mq[0]*100:3.0f}% edges found {mq[1]*100:3.0f}%", flush=True)
    errs = sorted(r[1] for r in results)
    herrs = sorted(abs(r[2]) for r in results)
    print(f"\nend error:     median {errs[len(errs)//2]*100:.0f} cm · mean {sum(errs)/len(errs)*100:.0f} cm · "
          f"worst {errs[-1]*100:.0f} cm · over 50 cm in {sum(e > .5 for e in errs)} of {len(errs)}")
    print(f"heading error: median {herrs[len(herrs)//2]:.1f}° · worst {herrs[-1]:.1f}° · over 8° in {sum(h > 8 for h in herrs)} of {len(herrs)}")
    import re as _re
    dod, stops, bumps_, walks = [], [], [], []
    for r in results:
        d = st = bu = w = 0
        for _, what, _, _, res in r[5]:
            m = _re.findall(r"stepp(?:ing|ed) around something (\d+)", res)
            d += sum(int(x) for x in m)
            if _re.search(r"(?i)\b(walk to|explore)\b", what):
                w += 1
                st += bool(_re.search(r"stopped|NOT at", res))
            bu += len(_re.findall(r"(?i)bump", res))
        dod.append(d); stops.append(st); bumps_.append(bu); walks.append(w)
    print(f"route walks: sidesteps per run {sum(dod)/len(dod):.1f} (max {max(dod)}) · walks that ended stopped "
          f"{sum(stops)}/{sum(walks)} · bump words per run {sum(bumps_)/len(bumps_):.1f} · falls per run {sum(r[9] for r in results)/len(results):.2f}")
    print(f"floor plan:   black squares that are really something {sum(r[10][0] for r in results)/len(results)*100:.0f} % · real edges found {sum(r[10][1] for r in results)/len(results)*100:.0f} %")
    if VERBOSE:
        w = max(results, key=lambda r: r[1])
        w = (w[0], w[1], w[2], w[3], w[4], w[5])
        print(f"\nworst run: seed {w[0]} – how the error grew:")
        last = 0
        for n, what, err, herr, res in w[5]:
            if abs(err - last) > 0.08 or n % 15 == 0:
                print(f"  move {n:3d} {what[:40]:40s} error {err*100:5.0f} cm  heading {herr:+6.1f}°")
            last = err
    if "--blame" in sys.argv:
        # which kinds of moves make the error jump? (sum of increases of more than 4 cm, per kind)
        from collections import defaultdict
        tot, cnt, allc = defaultdict(float), defaultdict(int), defaultdict(int)
        for r in results:
            prev = 0.0
            for n, what, err, herr, res in r[5]:
                kind = what.split()[0] + (" (to)" if re.match(r"(walk|go) (to|towards)", what) else "")
                tag = ("bump" if "bump" in res else "legs-stopped" if "stopped" in res else "dodge" if "stepped around" in res
                       else "fell" if "fell" in res else "")
                key = f"{kind} {tag}".strip()
                allc[key] += 1
                if err - prev > 0.04:
                    tot[key] += err - prev
                    cnt[key] += 1
                prev = err
        print("\nerror growth per kind of move (jumps > 4 cm): total metres · jumps · moves of that kind")
        for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {k:28s} {v:6.2f} m · {cnt[k]:3d} · {allc[k]}")
    # bumps per kind of move (all seeds), and how many seeds fell
    from collections import Counter
    bumps, kinds = Counter(), Counter()
    for r in results:
        for n, what, err, herr, res in r[5]:
            k = what.split()[0]
            kinds[k] += 1
            if "bump" in res or "BUMP" in res:
                bumps[k] += 1
    print("bumps per seed: " + " · ".join(f"{k} {v/len(results):.2f}" for k, v in bumps.most_common()) +
          f"  (total {sum(bumps.values())/len(results):.2f})")
