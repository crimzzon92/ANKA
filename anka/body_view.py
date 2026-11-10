"""The virtual duck's senses, drawn from his own eyes – the same kind of signals the real Microduck has:
  • a camera picture, seen from 20 cm above the floor (a simple 3D drawing of the room)
  • the 8 × 8 depth sensor (ToF): 64 distances straight ahead
The brain gets the picture (like a real camera picture) and a short summary of the depth sensor."""
import math
import os
import urllib.request

import numpy as np

from anka.shared import cv2

CAM_H = 0.20          # metres: his eyes are this high above the floor
FOV = 70              # degrees: how wide the camera sees
W, H = 320, 240       # camera picture size
TOF_FOV = 45          # degrees: the depth sensor sees a 45° × 45° square in front of his head
TOF_RANGE = 4.0       # metres: further away than this = "nothing"
WALL_TOP = 2.4        # the ceiling

FLOOR = np.array((120, 160, 196), np.float32)     # (colours are BGR)
CEILING = np.array((232, 234, 236), np.float32)
WALLS = {"north": (214, 214, 206), "east": (200, 210, 220), "south": (205, 214, 205), "west": (222, 210, 200)}
WINDOW = (240, 210, 150)
DOOR = (60, 105, 150)
BASEBOARD = (110, 110, 110)
BOOKS = [(50, 60, 160), (150, 90, 40), (60, 140, 70), (40, 160, 200), (120, 60, 120), (200, 200, 200), (30, 90, 150)]


def _ray_box(px, py, dx, dy, x1, y1, x2, y2):
    """Where a ray (on the floor plan) goes into and out of a box: (t_in, t_out, side) or None."""
    t_in, t_out, side = -math.inf, math.inf, 0
    for p, d, lo, hi, ax in ((px, dx, x1, x2, 0), (py, dy, y1, y2, 1)):
        if abs(d) < 1e-9:
            if p < lo or p > hi:
                return None
        else:
            a, b = (lo - p) / d, (hi - p) / d
            if a > b:
                a, b = b, a
            if a > t_in:
                t_in, side = a, ax
            t_out = min(t_out, b)
    if t_out < max(t_in, 0):
        return None
    return t_in, t_out, side


def _ray_wall(px, py, dx, dy, room_w, room_h):
    """Where a ray from inside the room meets a wall: (t, wall name, x, y)."""
    tx = ((room_w if dx > 0 else 0) - px) / dx if abs(dx) > 1e-9 else math.inf
    ty = ((room_h if dy > 0 else 0) - py) / dy if abs(dy) > 1e-9 else math.inf
    if tx < ty:
        return tx, ("east" if dx > 0 else "west"), px + tx * dx, py + tx * dy
    return ty, ("north" if dy > 0 else "south"), px + ty * dx, py + ty * dy


def _ray_circle(px, py, dx, dy, cx, cy, r):
    fx, fy = px - cx, py - cy
    b = fx * dx + fy * dy
    c = fx * fx + fy * fy - r * r
    disc = b * b - c
    if disc < 0:
        return None
    t = -b - math.sqrt(disc)
    return t if t > 0 else None


def render(world, x, y, yaw, pitch=0.0, fallen=False, with_depth=False):
    """What his camera sees from (x, y), looking towards 'yaw' (degrees), head tilted 'pitch' degrees up.
    Returns (picture, seen) – seen = {name: {"pixels": n, "face": n}} for the things and people in it."""
    f = (W / 2) / math.tan(math.radians(FOV / 2))
    yh = H / 2 + f * math.tan(math.radians(pitch))           # the horizon moves down when he looks up
    rows = np.arange(H, dtype=np.float32)
    img = np.zeros((H, W, 3), np.float32)
    zbuf = np.full((H, W), np.inf, np.float32)
    below = rows > yh + 0.5
    dfloor = np.where(below, CAM_H * f / np.maximum(rows - yh, 0.5), np.inf)
    shade_floor = 1 / (1 + 0.16 * np.minimum(dfloor, 8))
    img[below] = (FLOOR[None, :] * shade_floor[below][:, None])[:, None, :]
    img[~below] = CEILING
    zbuf[below] = (dfloor[below] + 0.03)[:, None]
    boxes = world.solid_boxes()
    room_w, room_h = world.room_size()
    yaw_r = math.radians(yaw)
    for c in range(W):
        ang = yaw_r + math.atan((W / 2 - c) / f)
        dx, dy = math.cos(ang), math.sin(ang)
        cosfix = math.cos(ang - yaw_r)
        t, wall, hx, hy = _ray_wall(x, y, dx, dy, room_w, room_h)
        hits = [(t, 0.0, WALL_TOP, WALLS[wall])]
        hits.append((t - 1e-3, 0.0, 0.07, BASEBOARD))
        if wall == "north" and 1.2 < hx < 2.8:
            hits.append((t - 2e-3, 0.9, 2.0, WINDOW))
        if wall == "east" and 0.8 < hy < 1.8:
            hits.append((t - 2e-3, 0.0, 2.0, DOOR))
        if wall == "north" and 1.2 < hx < 2.8:   # window frame
            if abs(hx - 2.0) < 0.03 or abs(hx - 1.22) < 0.03 or abs(hx - 2.78) < 0.03:
                hits.append((t - 3e-3, 0.9, 2.0, (245, 245, 245)))
            hits.append((t - 3e-3, 1.43, 1.47, (245, 245, 245)))
            hits.append((t - 3e-3, 0.88, 0.93, (245, 245, 245)))
        if wall == "east" and 1.55 < hy < 1.62:   # door handle
            hits.append((t - 3e-3, 0.95, 1.0, (60, 190, 220)))
        for (x1, y1, x2, y2, z0, z1, colour, name) in boxes:
            r = _ray_box(x, y, dx, dy, x1, y1, x2, y2)
            if r and r[0] > 0.01:
                col = tuple(v * (0.82 if r[2] == 0 else 1.0) for v in colour)
                t0 = r[0]
                hits.append((t0, z0, z1, col))
                u = (y + t0 * dy) if r[2] == 0 else (x + t0 * dx)   # where on the face the ray hits
                dark = tuple(v * 0.6 for v in col)
                if name == "sofa" and z1 < 0.5:      # seat: cushions, seams, little feet
                    hits.append((t0 - 1e-4, z0, 0.07, (40, 50, 70)))
                    hits.append((t0 - 1e-4, 0.07, 0.10, dark))
                    hits.append((t0 - 1e-4, 0.40, 0.42, dark))
                    if (u - 0.3) % 0.66 < 0.025:
                        hits.append((t0 - 1e-4, 0.10, z1, dark))
                elif name == "sofa":                 # backrest cushions
                    if (u - 0.3) % 0.66 < 0.025:
                        hits.append((t0 - 1e-4, 0.45, z1, dark))
                    hits.append((t0 - 1e-4, z1 - 0.03, z1, tuple(min(255, v * 1.15) for v in col)))
                elif name == "bookshelf":            # shelves with books
                    for lo, hi in ((0.05, 0.33), (0.38, 0.68), (0.73, 1.03), (1.08, 1.38), (1.43, 1.75)):
                        k = int(u * 16 + lo * 7)
                        if k % 7 == 3:
                            continue
                        book = BOOKS[k % len(BOOKS)]
                        top = hi - (0.05 if k % 3 == 0 else 0.0) - (0.04 if k % 5 == 1 else 0.0)
                        hits.append((t0 - 1e-4, lo, top, tuple(v * (0.82 if r[2] == 0 else 1.0) for v in book)))
                elif name == "table" and z0 > 0.5:   # the edge of the table top
                    hits.append((t0 - 1e-4, z0, z0 + 0.01, dark))
        hits.sort(key=lambda h: -h[0])
        for t, z0, z1, col in hits:
            d = max(t * cosfix, 0.02)
            top = int(math.floor(yh - (z1 - CAM_H) * f / d))
            bot = int(math.ceil(yh - (z0 - CAM_H) * f / d))
            a, b = max(0, top), min(H, bot + 1)
            if a >= b:
                continue
            column, depth = img[a:b, c], zbuf[a:b, c]
            near = depth > d
            column[near] = np.array(col, np.float32) / (1 + 0.16 * d)
            depth[near] = d
    # things and people (drawn standing on the floor, hidden behind what's in front of them)
    seen = {}
    cy, sy = math.cos(yaw_r), math.sin(yaw_r)
    for s in world.sprites():
        vx, vy = s["x"] - x, s["y"] - y
        depth = vx * cy + vy * sy
        if depth < 0.08:
            continue
        right = vx * sy - vy * cy
        cx = W / 2 + right * f / depth
        scale = f / depth
        half = s["w"] / 2 * scale
        if cx + half < 0 or cx - half >= W:
            continue
        Y = lambda z: int(round(yh - (z - CAM_H) * scale))
        layer = np.zeros((H, W, 3), np.uint8)
        mask = np.zeros((H, W), np.uint8)
        face = np.zeros((H, W), np.uint8)
        col = s["color"]
        ix, ihalf = int(round(cx)), max(1, int(round(half)))
        if s["shape"] == "ball":
            r = max(1, int(round(s["w"] / 2 * scale)))
            centre = (ix, Y(s["z0"] + s["w"] / 2))
            for m, v in ((layer, col), (mask, 255)):
                cv2.circle(m, centre, r, v, -1, cv2.LINE_AA)
            cv2.circle(layer, (centre[0] - r // 3, centre[1] - r // 3), max(1, r // 4), (200, 200, 255), -1)
        elif s["shape"] == "person":
            legs, body_, head_r = (int(round(0.075 * scale)), int(round(0.2 * scale)), max(1, int(round(0.12 * scale))))
            gap = max(1, int(round(0.025 * scale)))
            trousers = s.get("trousers", (140, 95, 55))
            shoe = int(round(0.05 * scale))
            parts = [((ix - gap - legs * 2, Y(0.85)), (ix - gap, Y(0.07)), trousers),          # legs
                     ((ix + gap, Y(0.85)), (ix + gap + legs * 2, Y(0.07)), trousers),
                     ((ix - gap - legs * 2 - shoe // 2, Y(0.08)), (ix - gap + shoe // 3, Y(0.0)), (35, 35, 35)),
                     ((ix + gap - shoe // 3, Y(0.08)), (ix + gap + legs * 2 + shoe // 2, Y(0.0)), (35, 35, 35)),
                     ((ix - body_, Y(1.45)), (ix + body_, Y(0.82)), col),                          # shirt
                     ((ix - body_ - legs, Y(1.42)), (ix - body_, Y(0.9)), col),                    # arms
                     ((ix + body_, Y(1.42)), (ix + body_ + legs, Y(0.9)), col),
                     ((ix - body_ - legs, Y(0.9)), (ix - body_, Y(0.8)), (150, 180, 225)),         # hands
                     ((ix + body_, Y(0.9)), (ix + body_ + legs, Y(0.8)), (150, 180, 225))]
            for p1, p2, c in parts:
                cv2.rectangle(layer, p1, p2, c, -1)
                cv2.rectangle(mask, p1, p2, 255, -1)
            head = (ix, Y(1.6))
            cv2.circle(layer, head, head_r, (150, 180, 225), -1, cv2.LINE_AA)
            cv2.circle(layer, (head[0], head[1] - head_r // 2), head_r, s.get("hair", (40, 50, 70)), -1, cv2.LINE_AA)
            cv2.circle(layer, (head[0], head[1] + head_r // 4), int(head_r * 0.8), (150, 180, 225), -1, cv2.LINE_AA)
            cv2.circle(mask, head, head_r, 255, -1)
            cv2.circle(face, head, head_r, 255, -1)
        elif s["shape"] == "guitar":
            bw = max(1, int(round(0.17 * scale)))
            for m, v in ((layer, col), (mask, 255)):
                cv2.ellipse(m, (ix, Y(0.22)), (bw, max(1, int(round(0.2 * scale)))), 0, 0, 360, v, -1, cv2.LINE_AA)
                cv2.rectangle(m, (ix - max(1, bw // 4), Y(1.0)), (ix + max(1, bw // 4), Y(0.35)), v, -1)
            cv2.circle(layer, (ix, Y(0.25)), max(1, bw // 3), (30, 30, 30), -1, cv2.LINE_AA)
        else:   # a small box: sock, cup
            for m, v in ((layer, col), (mask, 255)):
                cv2.rectangle(m, (ix - ihalf, Y(s["z1"])), (ix + ihalf, Y(s["z0"])), v, -1)
        test = (mask > 0) & (zbuf > depth)
        if not test.any():
            continue
        img[test] = layer[test].astype(np.float32) / (1 + 0.1 * depth)
        zbuf[test] = depth
        seen[s["name"]] = {"pixels": int(test.sum()), "face": int(((face > 0) & test).sum()), "kind": s["kind"]}
    picture = np.clip(img, 0, 255).astype(np.uint8)
    if with_depth:
        # The floor trick: his eyes are always 20 cm above the floor, so where the floor ENDS in each column of the
        # picture tells how far it is free that way (the first thing standing on the floor). Per column: metres.
        floor_z = np.where(below, dfloor + 0.03, np.inf)
        with np.errstate(invalid="ignore"):
            on_floor = below[:, None] & (np.abs(zbuf - floor_z[:, None]) < 1e-4)
        start = int(max(0, math.ceil(yh + 1)))
        col_dist = np.full(W, np.inf)
        if start < H:
            part = ~on_floor[start:, :]                    # not floor, below the horizon
            any_hit = part.any(axis=0)
            lowest = H - 1 - np.argmax(part[::-1, :], axis=0)  # the lowest such pixel = where the floor ends
            cols = np.arange(W)
            angles = np.arctan((W / 2 - cols) / f)
            dist = zbuf[lowest, cols] / np.cos(angles)
            col_dist = np.where(any_hit & np.isfinite(dist), dist, np.inf)
    if fallen:
        picture = cv2.rotate(picture, cv2.ROTATE_90_CLOCKWISE)
        picture = cv2.resize(picture, (W, H))
    if with_depth == "zbuf":
        return picture, seen, col_dist, zbuf
    if with_depth:
        return picture, seen, col_dist
    return picture, seen


def panorama(world, x, y, yaw, pitch=0.0):
    """'Look around': three pictures with the head turned left, ahead and right, side by side."""
    pics, seen = [], {}
    for turn, label in ((60, "left"), (0, "ahead"), (-60, "right")):
        pic, s = render(world, x, y, yaw + turn, pitch)
        cv2.putText(pic, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(pic, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        pics.append(pic)
        for k, v in s.items():
            if k not in seen or v["pixels"] > seen[k]["pixels"]:
                seen[k] = v
    sep = np.full((H, 4, 3), 30, np.uint8)
    return np.hstack([pics[0], sep, pics[1], sep, pics[2]]), seen


def depth_sensor(world, x, y, yaw, pitch=0.0):
    """The 8 × 8 depth sensor (row 0 = top, column 0 = left). Like the real one, each zone reports the closest
    thing anywhere inside it (3 × 3 rays per zone – thin things like table legs aren't missed).
    Returns (distances in metres, floor?)."""
    dist = np.full((8, 8), np.inf)
    floor = np.zeros((8, 8), bool)
    boxes = world.solid_boxes()
    sprites = [s for s in world.sprites() if s["kind"] != "flat"]
    room_w, room_h = world.room_size()
    step = TOF_FOV / 8

    def ray(a_deg, p_deg):
        p = math.radians(p_deg)
        tp, cp = math.tan(p), math.cos(p)
        z = lambda t: CAM_H + t * tp
        a = math.radians(a_deg)
        dx, dy = math.cos(a), math.sin(a)
        best, is_floor = TOF_RANGE * cp, False
        if tp < -1e-6 and CAM_H / -tp < best:
            best, is_floor = CAM_H / -tp, True
        if tp > 1e-6:
            best = min(best, (WALL_TOP - CAM_H) / tp)
        t, *_ = _ray_wall(x, y, dx, dy, room_w, room_h)
        if t < best:
            best, is_floor = t, False
        for (x1, y1, x2, y2, z0, z1, _, _) in boxes:
            r = _ray_box(x, y, dx, dy, x1, y1, x2, y2)
            if not r:
                continue
            te, tx = max(r[0], 0.0), r[1]
            if te >= best:
                continue
            if z0 <= z(te) <= z1:
                hit = te
            elif z(te) < z0 and tp > 0:
                hit = (z0 - CAM_H) / tp
            elif z(te) > z1 and tp < 0:
                hit = (z1 - CAM_H) / tp
            else:
                continue
            if te <= hit <= tx and hit < best:
                best, is_floor = hit, False
        for s in sprites:
            t = _ray_circle(x, y, dx, dy, s["x"], s["y"], s["w"] / 2)
            if t is not None and t < best and s["z0"] <= z(t) <= s["z1"]:
                best, is_floor = t, False
        return best / cp, is_floor

    for i in range(8):
        for j in range(8):
            for di in (-1 / 3, 0, 1 / 3):
                for dj in (-1 / 3, 0, 1 / 3):
                    d, fl = ray(yaw + (3.5 - j + dj) * step, pitch + (3.5 - i + di) * step)
                    if d < TOF_RANGE and (d < dist[i, j] - 0.02 or (fl is False and floor[i, j] and d < dist[i, j] + 0.02)):
                        dist[i, j], floor[i, j] = d, fl
    return dist, floor


def depth_words(dist, floor):
    """A short summary for the brain: the closest thing to the left, ahead and to the right (not the floor)."""
    def part(cols):
        d = np.where(floor[:, cols], np.inf, dist[:, cols]).min()
        return "nothing within 4 m" if not np.isfinite(d) else (f"{d * 100:.0f} cm" if d < 1 else f"{d:.1f} m")
    return f"left {part(slice(0, 3))} · ahead {part(slice(3, 5))} · right {part(slice(5, 8))}"


def depth_picture(dist, floor, size=160):
    """The 8 × 8 grid as a small picture for the window (near = warm colours, far = blue, floor = grey)."""
    cell = size // 8
    img = np.zeros((cell * 8, cell * 8, 3), np.uint8)
    for i in range(8):
        for j in range(8):
            d = dist[i, j]
            if not np.isfinite(d):
                col = (60, 30, 20)
            elif floor[i, j]:
                col = (110, 110, 110)
            else:
                k = min(1.0, d / 2.5)
                col = (int(255 * k), int(120 + 80 * (1 - abs(k - 0.5) * 2)), int(255 * (1 - k)))
            cv2.rectangle(img, (j * cell, i * cell), ((j + 1) * cell - 2, (i + 1) * cell - 2), col, -1)
            if np.isfinite(d):
                cv2.putText(img, f"{d:.1f}", (j * cell + 2, i * cell + cell // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ---------- the depth AI: Depth Anything V2 Small (guesses how far every pixel of a picture is) ----------
_depth_ai = {"session": None, "failed": None}
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def depth_ai_model():
    """Loads the depth AI (downloads it the first time). None if it can't be used – then he does without it."""
    from settings import DEPTH_AI, DEPTH_AI_FILE, DEPTH_AI_URL
    if _depth_ai["session"] is not None or _depth_ai["failed"] or not DEPTH_AI:
        return _depth_ai["session"]
    try:
        import onnxruntime as ort
        if not os.path.exists(DEPTH_AI_FILE):
            os.makedirs(os.path.dirname(DEPTH_AI_FILE), exist_ok=True)
            print("📥 Downloading the depth AI (Depth Anything V2 Small, about 100 MB – only the first time)...")
            urllib.request.urlretrieve(DEPTH_AI_URL, DEPTH_AI_FILE + ".part")
            os.replace(DEPTH_AI_FILE + ".part", DEPTH_AI_FILE)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2     # (leave the rest of the Mac to the brain and the window)
        _depth_ai["session"] = ort.InferenceSession(DEPTH_AI_FILE, options, providers=["CPUExecutionProvider"])
        print("🔭 Depth AI ready: it guesses distances in the camera picture")
    except Exception as error:
        _depth_ai["failed"] = str(error)
        print(f"🔭 (the depth AI can't be used: {str(error)[:120]} – using the depth sensor and the floor trick)")
    return _depth_ai["session"]


def depth_ai(picture):
    """The depth AI's guess for a camera picture: one number per pixel, BIGGER = CLOSER (relative, no unit).
    None if the depth AI isn't available."""
    session = depth_ai_model()
    if session is None:
        return None
    rgb = cv2.cvtColor(cv2.resize(picture, (336, 252)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    out = session.run(None, {session.get_inputs()[0].name: x})[0]
    out = np.squeeze(np.asarray(out, np.float32))
    if out.ndim == 3:
        out = out[0]
    return cv2.resize(out, (picture.shape[1], picture.shape[0]))
