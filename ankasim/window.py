"""Anka's ONE window in the 3D simulator (2026-11-11): the room from above and his own floor plan on top, his camera and
depth sensor and the last lines of the log below - scaled to fit the screen.

The window runs in its OWN program (ankasim/eyes.py --one-window, started by "Start everything.command"), so drawing it
never slows his brain or his walking commands down, and his camera is shown live. Anka's program only hands over what
it alone knows, through small files next to this one, a few times a second:
    live_room.jpg   the room picture (as the room window used to show it)
    live_room.json  where the mute button is in that picture
    live_log.txt    the last lines of the log
    live_cmd.txt    written by the window when you click the mute button (or press M); Anka's program reads it
    live_status.txt what "Start everything" is doing while it starts (the window shows it until Anka is awake)
This file has the drawing that both programs share.
"""
import os
import re
import subprocess
import sys
import textwrap

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE_ROOM = os.path.join(HERE, "live_room.jpg")
LIVE_INFO = os.path.join(HERE, "live_room.json")
LIVE_LOG = os.path.join(HERE, "live_log.txt")
LIVE_CMD = os.path.join(HERE, "live_cmd.txt")
LIVE_STATUS = os.path.join(HERE, "live_status.txt")   # "seconds|what is happening" lines, written while starting

SYMBOLS = [("💭", "thinks:"), ("🦆", "says:"), ("❤️", "feels:"), ("🫀", "body:"), ("🦶", "move:"), ("😮", "surprised:"),
           ("🧑", ""), ("🎬", "script:"), ("🦉", "wise brain:"), ("🔎", "checked:"), ("📔", "diary:"), ("🎯", "goal:"),
           ("🏁", "goal done:"), ("🫳", "gave up:"), ("🧭", "mindful:"), ("🖼️", "kept a picture:"), ("🗺️", "map:"),
           ("🔊", "sound:"), ("👀", "sees:"), ("😴", ""), ("🌙", "night:"), ("🪞", "new about me:"), ("📝", "lesson:"),
           ("💡", "idea:")]
COLOURS = {"thinks:": (230, 200, 150), "says:": (120, 230, 255), "feels:": (170, 140, 255), "body:": (230, 150, 90),
           "move:": (150, 220, 150), "surprised:": (90, 160, 255), "script:": (200, 200, 200)}
BG = 24


def plain(line):
    """A log line in plain letters (the window's font has no emoji), with a word for the symbol it started with."""
    line = line.strip()
    label = None
    for sym, word in SYMBOLS:
        if line.startswith(sym):
            line, label = line[len(sym):].strip(), word
            break
    for a, b in (("–", "-"), ("—", "-"), ("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("→", "->"), ("…", "..."),
                 ("°", " deg"), ("·", "|"), ("×", "x")):
        line = line.replace(a, b)
    line = line.encode("ascii", "ignore").decode().strip()
    line = re.sub(r"\s{2,}", "  ", line)
    return ((label + " ") if label else "") + line, label


_log_cache = {"key": None, "img": None}


def log_panel(lines, w, h):
    """The last log lines, coloured by kind and word-wrapped (redrawn only when something changed)."""
    key = (w, h, len(lines), lines[-1] if lines else "")
    if _log_cache["key"] == key:
        return _log_cache["img"]
    img = np.full((h, w, 3), BG, np.uint8)
    cv2.putText(img, "what is happening (the log)", (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    step, cols, rows = 17, max(20, int(w / 7.6)), []
    for raw in list(lines)[-40:]:
        text, label = plain(raw)
        if text:
            colour = COLOURS.get(label, (210, 210, 210))
            rows += [(part, colour) for part in textwrap.wrap(text, cols, subsequent_indent="   ")]
    rows = rows[-max(1, (h - 34) // step):]
    for i, (t, c) in enumerate(rows):
        cv2.putText(img, t, (12, 40 + i * step), cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)
    _log_cache.update(key=key, img=img)
    return img


def fit(img, w, h):
    """Scales a picture to fit inside w x h (keeping its shape), top left on a dark background."""
    out = np.full((h, w, 3), BG, np.uint8)
    if img is None or img.size == 0:
        return out
    f = min(w / img.shape[1], h / img.shape[0])
    small = cv2.resize(img, (max(1, int(img.shape[1] * f)), max(1, int(img.shape[0] * f))), interpolation=cv2.INTER_LINEAR)
    out[:small.shape[0], :small.shape[1]] = small
    return out


_SCREEN = []


def _mac_screen():
    """On a Mac: the usable part of the screen (without the menu bar and the Dock) in PICTURE pixels. A Retina screen
    shows one picture pixel per screen pixel, and has 2 screen pixels per point - so a window measured in points came
    out half as big as meant (2026-11-12: "the window is too small")."""
    js = ("ObjC.import('AppKit'); var s = $.NSScreen.mainScreen; var f = s.visibleFrame;"
          "f.size.width + ',' + f.size.height + ',' + s.backingScaleFactor")
    out = subprocess.run(["osascript", "-l", "JavaScript", "-e", js], capture_output=True, text=True, timeout=5).stdout
    w, h, k = [float(v) for v in out.strip().split(",")[:3]]
    return int(w * k), int((h - 30) * k)          # (30 points for the window's own title bar)


def screen():
    """The screen size the window fits into, in picture pixels. On a Mac it is measured; elsewhere ANKA_SCREEN
    ("width,height", set by the start menu) or a guess."""
    if not _SCREEN:
        size = None
        if sys.platform == "darwin":
            try:
                size = _mac_screen()
            except Exception:
                size = None
        if size is None:
            try:
                size = tuple(int(float(v)) for v in os.environ.get("ANKA_SCREEN", "1440,860").split(",")[:2])
            except Exception:
                size = (1440, 860)
        _SCREEN.append(size)
    return _SCREEN[0]


def read_status():
    """The start steps from live_status.txt: [(seconds since the epoch, text), ...]."""
    steps = []
    try:
        with open(LIVE_STATUS, encoding="utf-8") as f:
            for line in f:
                t, _, text = line.strip().partition("|")
                if text:
                    steps.append((float(t), text))
    except (OSError, ValueError):
        pass
    return steps


def waiting_room(w=936, h=460, steps=None, now=None):
    """Shown until the first room picture arrives (2026-11-13: it took a minute before anything happened and it
    looked broken) - what is starting, step by step, and how long it has taken so far."""
    import time as _time
    now = now or _time.time()
    img = np.full((h, w, 3), 245, np.uint8)
    dark, grey, green = (60, 60, 60), (130, 130, 130), (60, 150, 60)
    cv2.putText(img, "Anka is waking up" + "." * (int(now * 2) % 4), (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.3, dark, 2,
                cv2.LINE_AA)
    steps = steps if steps is not None else read_status()
    if steps:
        secs = int(now - steps[0][0])
        cv2.putText(img, f"{secs // 60}:{secs % 60:02d}", (w - 130, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, grey, 2,
                    cv2.LINE_AA)
    y = 130
    for i, (t, text) in enumerate(steps[-8:]):
        last = i == len(steps[-8:]) - 1
        mark, colour = ("...", dark) if last else ("done", green)
        cv2.putText(img, mark, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2 if last else 1, cv2.LINE_AA)
        cv2.putText(img, text, (110, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2 if last else 1, cv2.LINE_AA)
        y += 36
    cv2.putText(img, "This takes about a minute. Everything starts by itself - nothing is wrong.", (40, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, grey, 1, cv2.LINE_AA)
    return img


def layout(room, eyes, lines):
    """The whole window: the room on top, his eyes and the log side by side below. Returns (picture, scale)."""
    room = room if room is not None else waiting_room()
    rh, rw = room.shape[:2]
    bottom_h = int(rh * 1.15)          # 2026-11-12: his eyes and the log a bit BIGGER than the room and the floor plan
    eyes_w = int(bottom_h * (eyes.shape[1] / eyes.shape[0])) if eyes is not None else int(rw * 0.5)
    eyes_w = min(eyes_w, int(rw * 0.55))
    whole = np.vstack([room, np.hstack([fit(eyes, eyes_w, bottom_h), log_panel(lines, rw - eyes_w, bottom_h)])])
    sw, sh = screen()
    # As big as fits on the screen (keeping its shape). ANKA_WINDOW_SIZE makes it smaller: 0.8 = 80 % of that.
    try:
        size = max(0.3, min(1.0, float(os.environ.get("ANKA_WINDOW_SIZE", "1.0"))))
    except ValueError:
        size = 1.0
    f = size * min(0.99 * sw / whole.shape[1], 0.99 * sh / whole.shape[0])
    if abs(f - 1.0) > 0.01:
        whole = cv2.resize(whole, (int(whole.shape[1] * f), int(whole.shape[0] * f)), interpolation=cv2.INTER_LINEAR)
    return whole, f


def write_atomic(path, data):
    """Writes a file so the other program never reads half of it."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
