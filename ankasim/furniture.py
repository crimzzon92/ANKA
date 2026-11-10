"""furniture: detailed models of Anka's room things, built from MuJoCo's own shapes (boxes, cylinders, ellipsoids).

Everything is placed in Anka's room coordinates (see room_layout.py); the footprints and heights match
room_layout.FURNITURE / WALL_THINGS / THINGS, so Anka's map, the pretend detector and the 2D picture stay right.
Each function returns a list of <geom .../> lines. `MATERIALS` lists the colours used (make_scene.py writes them).

The ball is NOT here (it is loose and has a joint: make_scene.py).
"""
import random

import room_layout as R

MATERIALS = {                          # name: rgba
    "sofa": (0.35, 0.45, 0.68, 1), "sofa_cushion": (0.42, 0.52, 0.76, 1), "sofa_dark": (0.20, 0.26, 0.40, 1),
    "wood": (0.55, 0.38, 0.20, 1), "wood_dark": (0.40, 0.27, 0.14, 1),
    "shelf": (0.40, 0.28, 0.17, 1), "shelf_dark": (0.28, 0.18, 0.10, 1),
    "door": (0.42, 0.26, 0.14, 1), "door_panel": (0.48, 0.31, 0.18, 1), "brass": (0.72, 0.74, 0.78, 1),   # (chrome grey: a golden handle can look orange to the ball finder)
    "frame_white": (0.93, 0.93, 0.90, 1), "glass": (0.75, 0.90, 1.00, 1),
    "guitar_side": (0.60, 0.38, 0.19, 1), "guitar_front": (0.74, 0.52, 0.28, 1), "guitar_black": (0.08, 0.06, 0.05, 1),
    "cup": (0.95, 0.95, 0.95, 1), "coffee": (0.25, 0.14, 0.08, 1),
    "sock": (0.85, 0.15, 0.20, 1), "sock_cuff": (0.95, 0.95, 0.95, 1),
    "ball": (0.95, 0.45, 0.10, 1),
    # book colours: no red / orange (those are the sock's and the ball's colours)
    "b1": (0.20, 0.35, 0.60, 1), "b2": (0.25, 0.50, 0.30, 1), "b3": (0.80, 0.72, 0.32, 1), "b4": (0.50, 0.32, 0.58, 1),
    "b5": (0.90, 0.90, 0.85, 1), "b6": (0.20, 0.22, 0.28, 1), "b7": (0.15, 0.45, 0.50, 1), "b8": (0.35, 0.22, 0.12, 1),
}
BOOKS = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"]


# ---------------------------------------------------------------- small helpers (Anka's coordinates in, MuJoCo's out)
def _geom(kind, name, size, pos, mat, collide=True, euler=None):
    e = f' euler="{euler}"' if euler else ""
    c = 'contype="1" conaffinity="1"' if collide else 'contype="0" conaffinity="0"'
    return f'    <geom name="{name}" type="{kind}" size="{size}" pos="{pos}"{e} material="{mat}" {c}/>'


def box(name, x1, y1, x2, y2, z0, z1, mat, collide=True):
    (ax, ay), (bx, by) = R.to_mj(x1, y1), R.to_mj(x2, y2)
    hx, hy, hz = abs(ax - bx) / 2, abs(ay - by) / 2, (z1 - z0) / 2
    return _geom("box", name, f"{hx:.4f} {hy:.4f} {hz:.4f}", f"{(ax + bx) / 2:.4f} {(ay + by) / 2:.4f} {z0 + hz:.4f}", mat, collide)


def vcyl(name, cx, cy, z0, z1, r, mat, collide=True):
    mx, my = R.to_mj(cx, cy)
    return _geom("cylinder", name, f"{r:.4f} {(z1 - z0) / 2:.4f}", f"{mx:.4f} {my:.4f} {(z0 + z1) / 2:.4f}", mat, collide)


def hcyl(name, cx, cy, cz, r, half, axis, mat, collide=True):
    """A cylinder lying down; its axis runs along Anka's x or y."""
    mx, my = R.to_mj(cx, cy)
    return _geom("cylinder", name, f"{r:.4f} {half:.4f}", f"{mx:.4f} {my:.4f} {cz:.4f}", mat, collide,
                 euler="90 0 0" if axis == "x" else "0 90 0")


def ell(name, cx, cy, cz, ax, ay, az, mat, collide=True):
    """An ellipsoid with half-sizes ax, ay, az along Anka's x, y and up."""
    mx, my = R.to_mj(cx, cy)
    return _geom("ellipsoid", name, f"{ay:.4f} {ax:.4f} {az:.4f}", f"{mx:.4f} {my:.4f} {cz:.4f}", mat, collide)


def sph(name, cx, cy, cz, r, mat, collide=True):
    mx, my = R.to_mj(cx, cy)
    return _geom("sphere", name, f"{r:.4f}", f"{mx:.4f} {my:.4f} {cz:.4f}", mat, collide)


# ---------------------------------------------------------------- the things
def sofa():
    x1, y1, x2, y2, _ = R.FURNITURE["sofa"]            # against the south wall, facing north (+y)
    o = []
    o.append(box("sofa_plinth", x1, y1, x2, y2, 0.0, 0.14, "sofa_dark"))
    o.append(box("sofa_back", x1 + 0.17, y1, x2 - 0.17, y1 + 0.20, 0.14, 0.80, "sofa"))
    for i, (a, b) in enumerate(((x1, x1 + 0.17), (x2 - 0.17, x2))):
        o.append(box(f"sofa_arm{i}", a, y1, b, y2, 0.14, 0.58, "sofa"))
        o.append(hcyl(f"sofa_armtop{i}", (a + b) / 2, (y1 + y2) / 2, 0.58, 0.085, (y2 - y1) / 2, "y", "sofa"))
    mid = (x1 + x2) / 2
    for i, (a, b) in enumerate(((x1 + 0.19, mid - 0.01), (mid + 0.01, x2 - 0.19))):
        o.append(box(f"sofa_seat{i}", a, y1 + 0.20, b, y2 - 0.02, 0.14, 0.40, "sofa_cushion"))
        o.append(box(f"sofa_cushion{i}", a + 0.02, y1 + 0.20, b - 0.02, y1 + 0.34, 0.40, 0.76, "sofa_cushion", collide=False))
        o.append(ell(f"sofa_roll{i}", (a + b) / 2, y2 - 0.02, 0.28, (b - a) / 2 - 0.01, 0.012, 0.12, "sofa", collide=False))
    return o


def table():
    x1, y1, x2, y2, h = R.FURNITURE["table"]
    top = h / 100.0
    o = [box("table_top", x1, y1, x2, y2, top - 0.04, top, "wood")]
    o.append(box("table_apron_n", x1 + 0.06, y2 - 0.09, x2 - 0.06, y2 - 0.06, top - 0.11, top - 0.04, "wood_dark"))
    o.append(box("table_apron_s", x1 + 0.06, y1 + 0.06, x2 - 0.06, y1 + 0.09, top - 0.11, top - 0.04, "wood_dark"))
    o.append(box("table_apron_w", x1 + 0.06, y1 + 0.06, x1 + 0.09, y2 - 0.06, top - 0.11, top - 0.04, "wood_dark"))
    o.append(box("table_apron_e", x2 - 0.09, y1 + 0.06, x2 - 0.06, y2 - 0.06, top - 0.11, top - 0.04, "wood_dark"))
    for i, (lx, ly) in enumerate(((x1 + 0.05, y1 + 0.05), (x2 - 0.05, y1 + 0.05), (x1 + 0.05, y2 - 0.05), (x2 - 0.05, y2 - 0.05))):
        o.append(box(f"table_leg{i}", lx - 0.03, ly - 0.03, lx + 0.03, ly + 0.03, 0.0, top - 0.04, "wood_dark"))
    return o


def bookshelf():
    x1, y1, x2, y2, h = R.FURNITURE["bookshelf"]       # against the east wall; open side faces west (-x)
    H = h / 100.0
    t = 0.025
    o = [box("shelf_back", x2 - 0.02, y1, x2, y2, 0.0, H, "shelf_dark"),
         box("shelf_side0", x1, y1, x2, y1 + t, 0.0, H, "shelf"),
         box("shelf_side1", x1, y2 - t, x2, y2, 0.0, H, "shelf"),
         box("shelf_cabinet", x1, y1 + t, x2, y2 - t, 0.0, 0.50, "shelf"),
         box("shelf_top", x1, y1, x2, y2, H - 0.03, H, "shelf")]
    mid = (y1 + y2) / 2
    for i, (a, b) in enumerate(((y1 + t + 0.01, mid - 0.01), (mid + 0.01, y2 - t - 0.01))):      # cabinet doors
        o.append(box(f"shelf_door{i}", x1 - 0.015, a, x1, b, 0.04, 0.46, "shelf_dark", collide=False))
        o.append(sph(f"shelf_knob{i}", x1 - 0.025, mid - 0.06 if i == 0 else mid + 0.06, 0.26, 0.015, "brass", collide=False))
    rng = random.Random(7)
    levels = [0.50, 0.95, 1.40]
    for li, z in enumerate(levels):
        o.append(box(f"shelf_board{li}", x1, y1 + t, x2, y2 - t, z, z + 0.02, "shelf"))
        top = (levels[li + 1] if li + 1 < len(levels) else H - 0.03)
        space = top - (z + 0.02)
        y = y1 + t + 0.03
        end = y2 - t - 0.03 - (0.35 if li == 1 else 0.0)            # the middle shelf is not full
        n = 0
        while y < end - 0.03:
            w = rng.uniform(0.022, 0.050)
            hh = space * rng.uniform(0.72, 0.94)
            dp = rng.uniform(0.17, 0.25)
            if y + w > end:
                break
            o.append(box(f"book{li}_{n}", x2 - 0.02 - dp, y, x2 - 0.02, y + w, z + 0.02, z + 0.02 + hh,
                         rng.choice(BOOKS), collide=False))
            y += w + 0.002
            n += 1
        if li == 2:                                                   # something standing on the top shelf: a small plant pot
            o.append(vcyl("pot", x2 - 0.15, y2 - 0.15, z + 0.02, z + 0.12, 0.05, "wood_dark", collide=False))
            o.append(ell("plant", x2 - 0.15, y2 - 0.15, z + 0.20, 0.09, 0.09, 0.09, "b2", collide=False))
    return o


def door():
    dx, dy = R.WALL_THINGS["door"]                      # on the east wall (x = 5.0), leaf stands slightly out of the wall
    o = [box("door_leaf", dx - 0.03, dy - 0.45, dx, dy + 0.45, 0.0, 2.0, "door"),
         box("door_casing_l", dx - 0.04, dy - 0.51, dx, dy - 0.45, 0.0, 2.06, "frame_white"),
         box("door_casing_r", dx - 0.04, dy + 0.45, dx, dy + 0.51, 0.0, 2.06, "frame_white"),
         box("door_casing_top", dx - 0.04, dy - 0.51, dx, dy + 0.51, 2.0, 2.06, "frame_white")]
    for i, (z0, z1) in enumerate(((0.12, 0.90), (1.10, 1.88))):
        for j, (a, b) in enumerate(((dy - 0.36, dy - 0.04), (dy + 0.04, dy + 0.36))):
            o.append(box(f"door_panel{i}{j}", dx - 0.036, a, dx - 0.03, b, z0, z1, "door_panel", collide=False))
    o.append(box("door_plate", dx - 0.035, dy + 0.34, dx - 0.03, dy + 0.40, 0.95, 1.10, "brass", collide=False))
    o.append(sph("door_handle", dx - 0.055, dy + 0.37, 1.02, 0.022, "brass", collide=False))
    return o


def window():
    wx, wy = R.WALL_THINGS["window"]                    # on the north wall (y = 4.0)
    x1, x2, z0, z1, f = wx - 0.6, wx + 0.6, 0.9, 1.7, 0.05
    o = [box("window_glass", x1, wy - 0.01, x2, wy, z0, z1, "glass", collide=False),
         box("window_frame_b", x1, wy - 0.05, x2, wy, z0, z0 + f, "frame_white"),
         box("window_frame_t", x1, wy - 0.05, x2, wy, z1 - f, z1, "frame_white"),
         box("window_frame_l", x1, wy - 0.05, x1 + f, wy, z0, z1, "frame_white"),
         box("window_frame_r", x2 - f, wy - 0.05, x2, wy, z0, z1, "frame_white"),
         box("window_bar_v", wx - 0.015, wy - 0.04, wx + 0.015, wy, z0, z1, "frame_white", collide=False),
         box("window_bar_h", x1, wy - 0.04, x2, wy, (z0 + z1) / 2 - 0.015, (z0 + z1) / 2 + 0.015, "frame_white", collide=False),
         box("window_sill", x1 - 0.06, wy - 0.10, x2 + 0.06, wy, z0 - 0.04, z0, "frame_white")]
    return o


def guitar():
    gx, gy = R.THINGS["guitar"]                         # stands on the floor, its front faces west (-x), towards the room
    o = []
    # body: two round bouts, a front plate with a sound hole, a bridge
    o.append(hcyl("guitar_lower", gx, gy, 0.105, 0.105, 0.040, "x", "guitar_side"))
    o.append(hcyl("guitar_upper", gx, gy, 0.265, 0.080, 0.040, "x", "guitar_side"))
    o.append(box("guitar_waist", gx - 0.04, gy - 0.07, gx + 0.04, gy + 0.07, 0.17, 0.23, "guitar_side"))
    o.append(hcyl("guitar_front_lower", gx - 0.041, gy, 0.105, 0.102, 0.003, "x", "guitar_front", collide=False))
    o.append(hcyl("guitar_front_upper", gx - 0.041, gy, 0.265, 0.077, 0.003, "x", "guitar_front", collide=False))
    o.append(box("guitar_front_waist", gx - 0.044, gy - 0.067, gx - 0.038, gy + 0.067, 0.17, 0.23, "guitar_front", collide=False))
    o.append(hcyl("guitar_hole", gx - 0.045, gy, 0.215, 0.028, 0.002, "x", "guitar_black", collide=False))
    o.append(box("guitar_bridge", gx - 0.047, gy - 0.04, gx - 0.041, gy + 0.04, 0.085, 0.100, "guitar_black", collide=False))
    # neck, fretboard, headstock
    o.append(box("guitar_neck", gx - 0.015, gy - 0.014, gx + 0.015, gy + 0.014, 0.33, 0.60, "guitar_side"))
    o.append(box("guitar_fretboard", gx - 0.021, gy - 0.016, gx - 0.015, gy + 0.016, 0.33, 0.60, "guitar_black", collide=False))
    o.append(box("guitar_head", gx - 0.018, gy - 0.022, gx + 0.018, gy + 0.022, 0.60, 0.68, "guitar_black"))
    return o


def cup():
    cx, cy = R.THINGS["cup"]                            # stands on the table
    z0 = R.FURNITURE["table"][4] / 100.0
    o = [vcyl("cup", cx, cy, z0, z0 + 0.09, 0.04, "cup"),
         vcyl("cup_coffee", cx, cy, z0 + 0.086, z0 + 0.089, 0.035, "coffee", collide=False),
         box("cup_handle", cx - 0.006, cy + 0.04, cx + 0.006, cy + 0.062, z0 + 0.025, z0 + 0.07, "cup", collide=False)]
    return o


def sock():
    sx, sy = R.THINGS["sock"]                           # an L-shaped red sock lying flat, with a white cuff
    o = [box("sock_leg", sx - 0.045, sy - 0.025, sx + 0.025, sy + 0.025, 0.0, 0.014, "sock", collide=False),
         box("sock_cuff", sx - 0.060, sy - 0.026, sx - 0.045, sy + 0.026, 0.0, 0.016, "sock_cuff", collide=False),
         box("sock_foot", sx + 0.00, sy + 0.02, sx + 0.04, sy + 0.07, 0.0, 0.014, "sock", collide=False),
         ell("sock_toe", sx + 0.02, sy + 0.07, 0.007, 0.022, 0.014, 0.007, "sock", collide=False),
         ell("sock_heel", sx + 0.027, sy, 0.007, 0.014, 0.027, 0.007, "sock", collide=False)]
    return o


def everything():
    out = []
    for f in (sofa, table, bookshelf, door, window, guitar, cup, sock):
        out.extend(f())
    return out
