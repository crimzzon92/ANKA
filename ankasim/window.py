"""Anka's ONE window in the 3D simulator (2026-11-11): the room from above and his own floor plan on top, his camera and
depth sensor and the last lines of the log below - scaled to fit the screen.

The window runs in its OWN program (ankasim/eyes.py --one-window, started by "Start everything.command"), so drawing it
never slows his brain or his walking commands down, and his camera is shown live. Anka's program only hands over what
it alone knows, through small files next to this one, a few times a second:
    live_room.jpg   the room picture (as the room window used to show it)
    live_room.json  where the mute button is in that picture
    live_log.txt    the last lines of the log
    live_cmd.txt    written by the window when you click the mute button (or press M); Anka's program reads it
This file has the drawing that both programs share.
"""
import os
import re
import textwrap

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE_ROOM = os.path.join(HERE, "live_room.jpg")
LIVE_INFO = os.path.join(HERE, "live_room.json")
LIVE_LOG = os.path.join(HERE, "live_log.txt")
LIVE_CMD = os.path.join(HERE, "live_cmd.txt")

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


def screen():
    """The screen size the window fits into (ANKA_SCREEN="width,height", set by the start menu)."""
    try:
        w, h = [int(float(v)) for v in os.environ.get("ANKA_SCREEN", "1440,860").split(",")[:2]]
        return w, h
    except Exception:
        return 1440, 860


def waiting_room(w=1240, h=620):
    img = np.full((h, w, 3), 245, np.uint8)
    cv2.putText(img, "waiting for Anka to wake up...", (40, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 120), 2, cv2.LINE_AA)
    return img


def layout(room, eyes, lines):
    """The whole window: the room on top, his eyes and the log side by side below. Returns (picture, scale)."""
    room = room if room is not None else waiting_room()
    rh, rw = room.shape[:2]
    bottom_h = int(rh * 0.65)
    eyes_w = int(bottom_h * (eyes.shape[1] / eyes.shape[0])) if eyes is not None else int(rw * 0.45)
    eyes_w = min(eyes_w, int(rw * 0.6))
    whole = np.vstack([room, np.hstack([fit(eyes, eyes_w, bottom_h), log_panel(lines, rw - eyes_w, bottom_h)])])
    sw, sh = screen()
    # 2026-11-12: 20 % bigger than before. ANKA_WINDOW_SIZE changes it (1.0 = the old size); never bigger than
    # the screen, so on a small screen it grows less than 20 %.
    try:
        size = float(os.environ.get("ANKA_WINDOW_SIZE", "1.2"))
    except ValueError:
        size = 1.2
    f = min(1.2 * size, 0.98 * sw / whole.shape[1], 0.90 * size * sh / whole.shape[0], 0.95 * sh / whole.shape[0])
    if abs(f - 1.0) > 0.01:
        whole = cv2.resize(whole, (int(whole.shape[1] * f), int(whole.shape[0] * f)), interpolation=cv2.INTER_LINEAR)
    return whole, f


def write_atomic(path, data):
    """Writes a file so the other program never reads half of it."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
