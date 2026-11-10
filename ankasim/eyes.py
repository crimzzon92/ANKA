"""Anka's eyes: a window that shows what the simulated duck senses, and lets you drive him by hand to try it.

    python eyes.py                  # opens the window (the simulator must be running: see "Start Anka sim.command")
    python eyes.py --once eyes.png  # saves one picture of what he senses and quits (handy for checking)

Left: the camera picture, upright and clean. Boxes show what Anka finds in it right now (while his brain runs):
  green = the colour finder found it (a real box in the picture: the ball, the sock)
  blue  = the stand-in detector says it is there (only angle + distance, so the box is where that puts it)
  red   = the colour finder looked at it and said no (only shown after pressing r)
The depth sensor's 8x8 grid can be laid over the picture with key o.
Right: the depth sensor as a heat map (red = close, blue = far, grey = nothing seen), then the numbers: odometry,
the IMU, the head, and - for checking only - where the duck REALLY is in the simulator.

Keys:  w / s  walk forward / back      a / d  turn left / right      space  stop
       i  look straight ahead          k  look down at the floor     j / l  look left / right
       t  sit down / stand up          u  switch the walking policy on      g  pick up from the floor
       o  depth grid on/off            r  rejected boxes on/off       p  save a picture      q  quit
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np

import cv2

try:
    from .duck_link import DuckLink
except ImportError:
    from duck_link import DuckLink

PANEL_W, HEIGHT = 440, 640
WHITE, GREY, GREEN, RED, AMBER = (240, 240, 240), (150, 150, 150), (110, 220, 110), (90, 90, 240), (60, 190, 250)


def _colour(d):
    """Distance in metres -> BGR (close = red, far = blue)."""
    t = 1.0 - min(max((d - 0.1) / 3.9, 0.0), 1.0)
    px = cv2.applyColorMap(np.array([[int(t * 255)]], np.uint8), cv2.COLORMAP_JET)[0, 0]
    return int(px[0]), int(px[1]), int(px[2])


def _text(img, s, x, y, colour=WHITE, scale=0.5, bold=1):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, bold, cv2.LINE_AA)


SEEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seen_now.json")
BOX_COLOURS = {"colour": (60, 220, 60), "stand-in": (240, 170, 60), "yolo": (240, 170, 60), "rejected": (70, 70, 230)}


def what_he_finds(max_age=1.5):
    """What Anka's body last found in the camera picture (written by simbody.py while the brain runs), or None."""
    try:
        with open(SEEN_FILE) as f:
            seen = json.load(f)
        if time.time() - seen.get("t", 0) > max_age:
            return None
        return seen
    except (OSError, ValueError):
        return None


def draw_finds(img, scale, seen, rejected=False):
    """Boxes and names on the (scaled) camera picture."""
    for it in seen.get("items", []):
        how = it.get("how", "stand-in")
        if how == "rejected" and not rejected:
            continue
        box = it.get("box")
        if not box:
            continue
        colour = BOX_COLOURS.get(how, WHITE)
        x0, y0, bw, bh = [int(round(v * scale)) for v in box]
        x1, y1 = x0 + bw, y0 + bh
        H, W = img.shape[:2]
        if x1 < 0 or y1 < 0 or x0 > W or y0 > H:
            continue
        cv2.rectangle(img, (x0, y0), (x1, y1), colour, 3 if how == "colour" else 2)
        label = f"{it['name']} {it['d']:.1f} m" + {"colour": "", "rejected": " (no)", "yolo": " (yolo)"}.get(how, " (stand-in)")
        tx, ty = min(max(2, x0), W - 8 * len(label)), y0 - 8 if y0 > 24 else min(H - 6, y1 + 20)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (tx - 3, ty - th - 5), (tx + tw + 3, ty + 5), (20, 20, 20), -1)
        _text(img, label, tx, ty, colour, 0.55, 2)


def camera_panel(link, overlay, rejected=False):
    """The camera picture scaled to the window's height, with the depth grid on top."""
    frame = link.frame()
    if frame is None:
        w = 360
        img = np.full((HEIGHT, w, 3), 40, np.uint8)
        _text(img, "no camera picture", 70, HEIGHT // 2 - 10, GREY, 0.6)
        err = link.errors().get("camera")
        if err:
            for i, line in enumerate(_wrap(err, 38)[:5]):
                _text(img, line, 12, HEIGHT // 2 + 20 + i * 18, AMBER, 0.42)
        return img
    h, w = frame.shape[:2]
    scale = HEIGHT / h
    img = cv2.resize(frame, (int(round(w * scale)), HEIGHT), interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)
    tof = link.tof()
    if overlay and tof is not None:
        # the depth sensor looks at the picture's middle square (45 x 45 degrees; the camera is 45 degrees wide)
        side = img.shape[1]
        y0, cell = (HEIGHT - side) // 2, side / tof["cols"]
        layer = img.copy()
        for r in range(tof["rows"]):
            for c in range(tof["cols"]):
                d = tof["m"][r, c]
                x1, y1 = int(c * cell), int(y0 + r * cell)
                x2, y2 = int((c + 1) * cell), int(y0 + (r + 1) * cell)
                if not math.isnan(d):
                    cv2.rectangle(layer, (x1, y1), (x2, y2), _colour(d), -1)
        img = cv2.addWeighted(layer, 0.28, img, 0.72, 0)
        for r in range(tof["rows"]):
            for c in range(tof["cols"]):
                d = tof["m"][r, c]
                x1, y1 = int(c * cell), int(y0 + r * cell)
                cv2.rectangle(img, (x1, y1), (int((c + 1) * cell), int(y0 + (r + 1) * cell)), (255, 255, 255), 1)
                if not math.isnan(d):
                    _text(img, f"{d * 100:.0f}", x1 + 3, int(y1 + cell / 2 + 4), WHITE, 0.32)
    seen = what_he_finds()
    if seen and seen.get("w"):
        draw_finds(img, img.shape[1] / seen["w"], seen, rejected)
    return img


def _wrap(s, n):
    words, lines, cur = s.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > n:
            lines.append(cur)
            cur = wd
        else:
            cur = (cur + " " + wd).strip()
    return lines + ([cur] if cur else [])


def tof_panel(link, size=336):
    img = np.full((size + 28, PANEL_W, 3), 24, np.uint8)
    _text(img, "depth sensor (8x8, cm)", 12, 18, GREY, 0.45)
    tof = link.tof()
    cell = size // 8
    for r in range(8):
        for c in range(8):
            x1, y1 = 12 + c * cell, 28 + r * cell
            d = float("nan") if tof is None else float(tof["m"][r, c])
            colour = (50, 50, 50) if math.isnan(d) else _colour(d)
            cv2.rectangle(img, (x1, y1), (x1 + cell - 2, y1 + cell - 2), colour, -1)
            if not math.isnan(d):
                _text(img, f"{d * 100:.0f}", x1 + 5, y1 + cell // 2 + 4, (20, 20, 20) if 0.5 < d < 3.0 else WHITE, 0.4)
    return img


def facts(link):
    """The numbers, as (text, colour) lines."""
    lines = []
    ok = link.connected()
    names = [("robot", "robot"), ("tof", "depth"), ("camera", "camera"), ("truth", "truth")]
    lines.append(("  ".join(f"{label} {'ok' if ok[k] else '--'}" for k, label in names), GREEN if all(ok.values()) else AMBER))
    s = link.state()
    if s:
        loop = s.get("loop", {})
        rt = link.realtime()
        lines.append((f"policy {s.get('policy', '?')}   loop {loop.get('hz', 0):.0f} Hz"
                      + (f"   world x{rt:.2f}" if rt else ""), WHITE if not rt or rt > 0.95 else AMBER))
        o = link.odom_anka()
        if o:
            lines.append((f"odom   x {o[0]:.2f}  y {o[1]:.2f}  heading {o[2]:.0f} deg", WHITE))
        t = link.truth()
        if t and o:
            err = math.hypot(t["x"] - o[0], t["y"] - o[1]) * 100
            dh = (o[2] - t["heading"] + 180) % 360 - 180
            lines.append((f"truth  x {t['x']:.2f}  y {t['y']:.2f}  heading {t['heading']:.0f} deg", GREY))
            lines.append((f"his odometry is off by {err:.0f} cm and {abs(dh):.0f} deg", GREY))
        elif t:
            lines.append((f"truth  x {t['x']:.2f}  y {t['y']:.2f}  heading {t['heading']:.0f} deg", GREY))
        h = s.get("head")
        if h and len(h) == 4:
            lines.append((f"head   neck {math.degrees(h[0]):.0f}  pitch {math.degrees(h[1]):.0f}  yaw {math.degrees(h[2]):.0f}"
                          f"  roll {math.degrees(h[3]):.0f} deg", WHITE))
        saf = s.get("safety", {})
        g = saf.get("gravity") or [0, 0, 0]
        lines.append((f"IMU    gravity {g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f}   " + ("FALLEN" if saf.get("fallen") else "upright"),
                      RED if saf.get("fallen") else WHITE))
        mv = s.get("move", {}).get("applied")
        if mv:
            lines.append((f"moving fwd {mv[0]:+.2f} m/s  left {mv[1]:+.2f} m/s  turn {mv[2]:+.2f} rad/s", WHITE))
    tof = link.tof()
    if tof is not None and not np.all(np.isnan(tof["m"])):
        m = tof["m"]
        mid = m[2:6, 2:6]
        lines.append((f"depth  closest {np.nanmin(m):.2f} m   middle {np.nanmedian(mid) if not np.all(np.isnan(mid)) else float('nan'):.2f} m", WHITE))
    seen = what_he_finds()
    if seen is None:
        lines.append(("Anka finds: (his brain is not running)", GREY))
    else:
        found = [f"{i['name']} {i['d']:.1f} m" for i in seen.get("items", []) if i.get("how") != "rejected"]
        lines.append(("Anka finds: " + (", ".join(found) if found else "nothing")[:58], GREEN if found else WHITE))
    for k, v in link.errors().items():
        if k in ("robot", "tof"):
            lines.append((f"{k}: {v}"[:60], AMBER))
    return lines


def compose(link, overlay=False, rejected=False):
    cam = camera_panel(link, overlay, rejected)
    panel = np.full((HEIGHT, PANEL_W, 3), 24, np.uint8)
    t = tof_panel(link)
    panel[:t.shape[0]] = t
    y = t.shape[0] + 24
    for text, colour in facts(link):
        _text(panel, text, 12, y, colour, 0.46)
        y += 22
    _text(panel, "w/s walk  a/d turn  space stop  i/k/j/l look  t sit  u enable  o grid  r rejected  p photo  q quit", 8, HEIGHT - 10, GREY, 0.31)
    return np.hstack([cam, panel])


def main():
    ap = argparse.ArgumentParser(description="Anka's eyes: what the simulated duck senses")
    ap.add_argument("--state-dir", default=None, help="where the simulator keeps its sockets (default ~/.cache/duck-sim)")
    ap.add_argument("--duck", default="duck-a")
    ap.add_argument("--no-truth", action="store_true", help="don't ask the simulator where the duck really is")
    ap.add_argument("--once", metavar="FILE.png", help="save one picture and quit")
    ap.add_argument("--wait", type=float, default=6.0, help="seconds to wait for the simulator with --once")
    args = ap.parse_args()

    link = DuckLink(state_dir=args.state_dir, duck=args.duck, truth=not args.no_truth).start()
    if args.once:
        link.wait_ready(args.wait)
        time.sleep(0.6)
        cv2.imwrite(args.once, compose(link))
        print(f"saved {args.once}; connected: {link.connected()}")
        if link.errors():
            print("problems:", link.errors())
        link.close()
        return

    overlay, rejected, title = False, False, "Anka's eyes"
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            cv2.imshow(title, compose(link, overlay, rejected))
            k = cv2.waitKey(50) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("w"):
                link.move(0.30, 0, 0, seconds=0.4)
            elif k == ord("s"):
                link.move(-0.20, 0, 0, seconds=0.4)
            elif k == ord("a"):
                link.move(0, 0, 0.8, seconds=0.4)
            elif k == ord("d"):
                link.move(0, 0, -0.8, seconds=0.4)
            elif k == ord(" "):
                link.stop()
            elif k == ord("i"):
                link.look_at(1.0, 0.0, 0.08)
            elif k == ord("k"):
                link.look_at(0.35, 0.0, -0.10)
            elif k == ord("j"):
                link.look_at(0.8, 0.8, 0.05)
            elif k == ord("l"):
                link.look_at(0.8, -0.8, 0.05)
            elif k == ord("t"):
                link.do("sit_toggle")
            elif k == ord("g"):
                link.do("ground_pick")
            elif k == ord("u"):
                link.enable(True)
            elif k == ord("o"):
                overlay = not overlay
            elif k == ord("r"):
                rejected = not rejected
            elif k == ord("p"):
                name = datetime.now().strftime("eyes_%Y%m%d_%H%M%S.png")
                cv2.imwrite(name, compose(link, overlay, rejected))
                print("saved", name)
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        link.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
