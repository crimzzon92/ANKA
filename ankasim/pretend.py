"""pretend: a stand-in for "an object detector (YOLO) + the depth sensor" in the MuJoCo room.

The simulated robot's camera sees plain coloured boxes, which no ready-made detector would call "sofa" or "table". So -
exactly as in the 2D simulator - a pretend detector says what a real one would have said: which of the room's known
things are in the picture, at which angle (accurate to ~2 degrees) and how far away (+-8 % within 4 m, +-20 % beyond).

It uses the simulator's TRUTH (where the duck really is) only to work out what is in the picture - his brain is told what
is detected, with these errors, never the truth. On the real duck this one function is replaced by YOLO on the camera
picture (yolo11n.pt for person / couch / table / ball / cup, yoloe-26n-seg.pt for "guitar" / "sock") + the depth sensor.

    detect(x, y, heading_deg, cam_yaw_deg, cam_pitch_deg)  ->  [(name, kind, bearing_deg, distance_m)]
        x, y, heading: the duck's true place in the room (Anka's coordinates), cam_yaw / cam_pitch: where his head points
        (relative to the body; yaw positive = left, pitch positive = up). bearing is relative to where the camera points
        (positive = left).
"""
import math
import random

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

try:
    from . import room_layout as R
except ImportError:
    import room_layout as R

CAM_H = 0.20            # his eyes are about 20 cm above the floor
FOV_H, FOV_V = 45.0, 73.0
MAX_RANGE = 5.0
MIN_RANGE = 0.25
CLOSE_RANGE = 0.04        # the colour finder sees the ball / sock this close (looking down at his feet)


def _landmarks():
    """Fixed points he can use to correct his own position: (x, y, height of the point)."""
    out = {}
    f = R.FURNITURE
    if "sofa" in f:
        x1, y1, x2, y2, h = f["sofa"]
        out["sofa"] = ((x1 + x2) / 2, y2, 0.30)
    if "table" in f:
        x1, y1, x2, y2, h = f["table"]
        out["table"] = ((x1 + x2) / 2, (y1 + y2) / 2, 0.40)
    if "bookshelf" in f:
        x1, y1, x2, y2, h = f["bookshelf"]
        out["bookshelf"] = (x1, (y1 + y2) / 2, 0.90)
    if "door" in R.WALL_THINGS:
        out["door"] = (*R.WALL_THINGS["door"], 1.0)
    if "window" in R.WALL_THINGS:
        out["window"] = (*R.WALL_THINGS["window"], 1.3)
    return out


LANDMARKS = _landmarks()
THING_HEIGHT = {"ball": 0.035, "sock": 0.01, "guitar": 0.45, "cup": 0.80}   # how high up each small thing is


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _seg_hits_box(ax, ay, bx, by, x1, y1, x2, y2):
    """Does the straight line from (ax, ay) to (bx, by) pass through the box? (Liang-Barsky)"""
    t0, t1 = 0.0, 1.0
    for p, q in ((-(bx - ax), ax - x1), (bx - ax, x2 - ax), (-(by - ay), ay - y1), (by - ay, y2 - ay)):
        if abs(p) < 1e-12:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def _hidden(ax, ay, px, py, top, own=None):
    """Is the point (px, py) of height `top` hidden from (ax, ay) by a piece of furniture that is at least as high?"""
    for name, (x1, y1, x2, y2, h) in R.FURNITURE.items():
        if name == own or h / 100.0 < top:
            continue
        if _seg_hits_box(ax, ay, px, py, x1 + 0.02, y1 + 0.02, x2 - 0.02, y2 - 0.02):
            return True
    return False


def detect(x, y, heading_deg, cam_yaw_deg=0.0, cam_pitch_deg=0.0, ball_at=None, rng=random, skip=()):
    """What the pretend detector reports right now. `ball_at` = where the ball is if it is known to have moved
    (None = still at its start, False = it was kicked somewhere unknown, so it is not reported)."""
    cam = heading_deg + cam_yaw_deg
    targets = []
    for name, (px, py, top) in LANDMARKS.items():
        targets.append((name, "landmark", px, py, top, name))
    for name, (px, py) in R.THINGS.items():
        if name == "ball" and ball_at is False:
            continue
        if name == "ball" and ball_at:
            px, py = ball_at
        targets.append((name, "thing", px, py, THING_HEIGHT.get(name, 0.2), "table" if name == "cup" else None))
    seen = []
    for name, kind, px, py, top, own in targets:
        dx, dy = px - x, py - y
        d = math.hypot(dx, dy)
        if not (MIN_RANGE < d < MAX_RANGE):
            continue
        bearing = _wrap(math.degrees(math.atan2(dy, dx)) - cam)
        if abs(bearing) > FOV_H / 2 - 1.0:
            continue
        elevation = math.degrees(math.atan2(top - CAM_H, d))
        if abs(elevation - cam_pitch_deg) > FOV_V / 2 - 2.0:
            continue
        if _hidden(x, y, px, py, top, own):
            continue
        seen.append((name, kind, bearing + rng.gauss(0, 2.0), d * rng.gauss(1.0, 0.08 if d < 4 else 0.2)))
    return seen


# ---------------------------------------------------------------------------------------------------------------
# The ball and the sock are loose (they get kicked), so nobody can know where they are from a table of places:
# they are found in the camera picture itself, by colour. (A real duck: YOLO / yoloe does this.)
# colours in the room: ball orange (0.95 0.45 0.10), sock red (0.85 0.15 0.20); floor/walls/furniture are less saturated.
COLOURS = {   # name: list of (H low, H high, S low, V low) in OpenCV's units (H 0-179, S and V 0-255)
    "ball": [(5, 17, 210, 110)],          # the ball is S~228; the guitar / bookshelf wood is S~160-175 (brown, same hue)
    "sock": [(0, 4, 140, 70), (168, 179, 140, 70)],
}
MIN_BLOB = {"ball": 5, "sock": 5}        # pixels (in the half-size picture)
BALL_DIAMETER = 0.07                      # m (the ball is 3.5 cm in radius)
LAST = []                                 # what the colour finder looked at last time (for the pictures saved during a test)
MAX_RATIO = {"ball": 3.5, "sock": 14.0}  # how wide compared to high a blob may be (a sock lying down seen low is wide and flat)


def find_by_colour(frame, cam_pitch_deg=0.0, rng=random):
    """frame: the camera picture (BGR, upright: 45 degrees wide, 73 tall). Returns [(name, "thing", bearing_deg (+ left),
    distance_m)] for the ball and the sock if they are in the picture. The distance comes from where the thing touches the
    floor in the picture (the camera is CAM_H above the floor and tilted cam_pitch_deg), like the brain is told it can."""
    if cv2 is None or frame is None or frame.size == 0:
        return []
    small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, w = small.shape[:2]
    fy = (h / 2) / math.tan(math.radians(FOV_V / 2))
    fx = (w / 2) / math.tan(math.radians(FOV_H / 2))
    out = []
    del LAST[:]
    for name, ranges in COLOURS.items():
        mask = np.zeros((h, w), np.uint8)
        for lo_h, hi_h, lo_s, lo_v in ranges:
            mask |= cv2.inRange(hsv, (lo_h, lo_s, lo_v), (hi_h, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, labels, stats, cent = cv2.connectedComponentsWithStats(mask)
        best = None
        for i in range(1, n):
            x0, y0, bw, bh, area = stats[i]
            if area < MIN_BLOB[name] or bw > MAX_RATIO[name] * bh + 6:
                continue
            if name == "ball" and area >= 12 and (area < 0.5 * bw * bh or not 0.5 < bw / max(bh, 1) < 2.0):
                continue                                   # not round: a handle, a book, a stripe - not the ball
            if best is None or area > best[0]:
                best = (area, x0, y0, bw, bh)
        if best is None:
            continue
        area, x0, y0, bw, bh = best
        cx = x0 + bw / 2
        bearing = -math.degrees(math.atan2(cx - w / 2, fx))
        down = math.degrees(math.atan2((y0 + bh) - h / 2, fy)) - cam_pitch_deg      # below the horizon, at the bottom of the blob
        if down < 1.0:
            continue                                                                  # at or above the horizon: not on the floor
        d = CAM_H / math.tan(math.radians(down))
        if not (CLOSE_RANGE < d < MAX_RANGE):    # (2026-11-10: was MIN_RANGE/2 = 12.5 cm - the ball vanished right where a kick needs it)
            continue
        if name == "ball" and 0 < x0 and x0 + bw < w and y0 > 0 and y0 + bh < h - 1:
            # a whole ball at distance d must be about this wide in the picture; a big orange-brown patch on the guitar or
            # the bookshelf (whose "floor distance" is nonsense) is far too wide for that
            expect = fx * BALL_DIAMETER / math.hypot(d, CAM_H)
            if not (0.4 * expect - 1 <= bw <= 2.2 * expect + 2):
                LAST.append((name, int(2 * x0), int(2 * y0), int(2 * bw), int(2 * bh), d, "rejected: wrong size for a ball at that distance"))
                continue
        LAST.append((name, int(2 * x0), int(2 * y0), int(2 * bw), int(2 * bh), d, "accepted"))
        out.append((name, "thing", bearing + rng.gauss(0, 1.0), d * rng.gauss(1.0, 0.05)))
    return out
