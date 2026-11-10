"""Camera floor test – Anka's floor plan from a REAL camera (the Mac's webcam), before he has a real body.

Put the MacBook on the floor with the screen open, so the webcam is about as high as Anka's eyes (~20 cm).
The depth AI (Depth Anything V2 Small) guesses how far away everything in the picture is, but only in its own
units. Because we know how high the camera is, the floor right in front of it tells us the scale: the script
finds the camera's tilt and the scale that make the bottom of the picture a flat floor, turns every pixel into a
point in the room, and draws a "radar" view from above: free floor, things in the way, and how far away they are.

Run:  python camera_floor_test.py [camera height in metres, default 0.21] [camera number, default 0]
                                   [how many degrees the camera looks down, default 0]
Keys in the window: q = quit, s = save a picture, c = calibrate (put a box or a book exactly 1 m straight ahead
first – the script then finds how much the camera looks up or down). A picture is also saved every 5 seconds:
camera_floor_test.png (in this folder).
"""
import math
import os
import sys
import time

import cv2
import numpy as np

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FOLDER)
import settings  # noqa: E402

settings.DEPTH_AI = True                       # (the test always uses the depth AI, even if Anka's body doesn't)
from anka.body_view import depth_ai  # noqa: E402

CAM_H = float(sys.argv[1]) if len(sys.argv) > 1 else 0.21     # how high the webcam is above the floor (metres)
# the camera: the same one Anka uses (CAMERA in settings.py – on this Mac the webcam is number 1, number 0 is often the
# iPhone). Give a number as 2nd argument to use another one; press n in the window to switch while it runs.
CAMERA = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].lstrip("-").isdigit() else settings.CAMERA
TILT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0       # how much the camera looks DOWN (degrees; up = minus)
HFOV = 65.0                  # the MacBook webcam sees about this wide (degrees)
W, H = 320, 240              # pictures are made this small first (enough, and fast)
CELL = 0.05                  # radar squares: 5 cm
AHEAD, SIDE = 3.0, 2.0       # radar: 3 m ahead, 2 m to each side
OBSTACLE = (0.04, 0.5)       # something at a small duck's height: 4–50 cm above the floor
OUT = os.path.join(FOLDER, "camera_floor_test.png")


def measure(picture, cam_h=CAM_H, hfov=HFOV, tilt=None, disp=None):
    """Depth AI + floor scaling. Returns (metric depth along the camera's axis per pixel, tilt in degrees,
    fit quality 0–1, the depth AI's own numbers) – or None when the floor in front can't be found. The depth AI's
    numbers have an unknown scale AND offset, so the floor can't also tell the tilt – it is given (TILT, or found
    with the 'c' calibration); the floor gives the scale."""
    tilt = settings_tilt[0] if tilt is None else tilt
    disp = depth_ai(picture) if disp is None else disp
    if disp is None:
        return None
    h, w = disp.shape
    f = (w / 2) / math.tan(math.radians(hfov / 2))
    # the floor sample: the bottom fifth of the picture, the middle half – right in front of him
    rows = np.arange(int(h * 0.8), h - 1)
    cols = np.arange(int(w * 0.25), int(w * 0.75), 2)
    V, U = np.meshgrid(rows, cols, indexing="ij")
    d = disp[V, U].ravel()
    y = ((V - h / 2) / f).ravel()
    s, c = math.sin(math.radians(tilt)), math.cos(math.radians(tilt))
    inv = (s + c * y) / cam_h                             # 1 / distance to the floor along the camera's axis
    ok = inv > 0.05
    if ok.sum() < 50:
        return None
    a, b = np.polyfit(d[ok], inv[ok], 1)
    if a <= 0:
        return None
    res = inv[ok] - (a * d[ok] + b)
    quality = 1.0 - float(np.var(res) / max(np.var(inv[ok]), 1e-12))
    inv_z = a * disp + b
    z = np.where(inv_z > 1e-3, 1.0 / np.maximum(inv_z, 1e-3), np.inf)
    return z, tilt, quality, disp


settings_tilt = [TILT]


def calibrate(picture, distance=1.0, cam_h=CAM_H, hfov=HFOV):
    """Something is exactly `distance` m straight ahead: find the camera tilt that makes it so."""
    disp = depth_ai(picture)
    best = None
    for t in np.arange(-15.0, 15.5, 0.5):
        m = measure(picture, cam_h, hfov, t, disp)
        if m is None:
            continue
        _, closest = radar(m[0], t, cam_h, hfov)
        if closest and (best is None or abs(closest - distance) < best[0]):
            best = (abs(closest - distance), t)
    return None if best is None else best[1]


def radar(z, tilt, cam_h=CAM_H, hfov=HFOV):
    """Every pixel as a point in the room, seen from above: free floor (white), something in the way (black),
    not seen (grey). Also the closest thing in the way straight ahead."""
    h, w = z.shape
    f = (w / 2) / math.tan(math.radians(hfov / 2))
    V, U = np.mgrid[0:h, 0:w].astype(np.float32)
    x_cam = (U - w / 2) / f * z                           # right
    y_cam = (V - h / 2) / f * z                           # down
    s, c = math.sin(math.radians(tilt)), math.cos(math.radians(tilt))
    forward = c * z - s * y_cam
    height = cam_h - (s * z + c * y_cam)
    ok = np.isfinite(z) & (forward > 0.05) & (forward < AHEAD) & (np.abs(x_cam) < SIDE)
    gw, gh = int(2 * SIDE / CELL), int(AHEAD / CELL)
    gi = ((x_cam + SIDE) / CELL).astype(int)
    gk = (forward / CELL).astype(int)
    floor_pts = ok & (np.abs(height) < OBSTACLE[0])
    obst_pts = ok & (height >= OBSTACLE[0]) & (height < OBSTACLE[1])
    free = np.zeros((gh, gw), np.float32)
    hit = np.zeros((gh, gw), np.float32)
    np.add.at(free, (gk[floor_pts], gi[floor_pts]), 1)
    np.add.at(hit, (gk[obst_pts], gi[obst_pts]), 1)
    grid = np.full((gh, gw), 150, np.uint8)
    grid[free > 0] = 245
    grid[hit >= 3] = 25
    # the closest thing in the way straight ahead (a 30 cm wide strip)
    strip = obst_pts & (np.abs(x_cam) < 0.15)
    closest = float(np.percentile(forward[strip], 5)) if strip.sum() > 20 else None
    return grid, closest


def picture_of(frame, z, grid, tilt, quality, closest):
    """One picture: the camera, the depth (near = bright), and the radar view from above."""
    cam = cv2.resize(frame, (W, H))
    dz = np.clip(z, 0.1, 4.0)
    depth = cv2.applyColorMap((255 * (1 - (dz - 0.1) / 3.9)).astype(np.uint8), cv2.COLORMAP_INFERNO)
    scale = 6
    top = cv2.resize(grid[::-1], (grid.shape[1] * scale // 2, grid.shape[0] * scale // 2), interpolation=cv2.INTER_NEAREST)
    top = cv2.cvtColor(top, cv2.COLOR_GRAY2BGR)
    for m in (1, 2, 3):                                   # distance rings, 1–3 m
        y = top.shape[0] - int(m / CELL * scale / 2)
        cv2.line(top, (0, y), (top.shape[1], y), (180, 200, 220), 1)
        cv2.putText(top, f"{m} m", (4, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 110, 140), 1, cv2.LINE_AA)
    cx = top.shape[1] // 2
    cv2.circle(top, (cx, top.shape[0] - 4), 5, (40, 200, 250), -1)
    height = max(H, top.shape[0])
    out = np.full((height + 40, W * 2 + top.shape[1] + 30, 3), (236, 240, 244), np.uint8)
    out[:H, 10:10 + W] = cam
    out[:H, 20 + W:20 + 2 * W] = depth
    out[:top.shape[0], 30 + 2 * W:30 + 2 * W + top.shape[1]] = top
    text = (f"tilt {tilt:+.1f} deg  floor fit {quality:.2f}  "
            + (f"closest thing straight ahead: {closest:.2f} m" if closest else "nothing in the way straight ahead"))
    cv2.putText(out, text, (10, height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    return out


def open_camera(nr):
    """Opens camera nr and checks that it really gives a picture. Returns the capture or None."""
    cap = cv2.VideoCapture(nr)
    if cap.isOpened():
        for _ in range(10):                      # (the first frames can be empty while the camera wakes up)
            ok, _f = cap.read()
            if ok:
                return cap
            time.sleep(0.1)
    cap.release()
    return None


def main():
    cam = CAMERA
    print(f"📷 Camera floor test – camera {cam}, {CAM_H * 100:.0f} cm above the floor. "
          "q = quit, s = save, c = calibrate (something exactly 1 m straight ahead), n = next camera.")
    cap = open_camera(cam)
    if cap is None:                              # (that one doesn't work: try the others)
        for nr in (0, 1, 2, 3):
            if nr != cam:
                cap = open_camera(nr)
                if cap is not None:
                    print(f"📷 Camera {cam} gave no picture – using camera {nr} instead (n = next camera).")
                    cam = nr
                    break
    if cap is None:
        print("📷 Couldn't open any camera. (macOS: allow Terminal to use the camera in System Settings → "
              "Privacy & Security → Camera, then try again.)")
        return
    last_save, started = 0.0, time.time()
    try:
        while time.time() - started < 300:                # (stops by itself after 5 minutes)
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            small = cv2.resize(frame, (W, H))
            m = measure(small)
            if m is None:
                print("🔭 (no depth this time)")
                continue
            z, tilt, quality, _ = m
            grid, closest = radar(z, tilt)
            view = picture_of(small, z, grid, tilt, quality, closest)
            cv2.putText(view, f"camera {cam}   (n = next camera, q = quit)", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 200), 1, cv2.LINE_AA)
            cv2.imshow("Anka camera floor test", view)
            if time.time() - last_save > 5:
                cv2.imwrite(OUT, view)
                last_save = time.time()
                print(f"📷 tilt {tilt:+.1f}° · floor fit {quality:.2f} · "
                      + (f"closest thing ahead {closest:.2f} m" if closest else "nothing straight ahead"))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("n"):                          # the next camera that works
                for k in range(1, 6):
                    nxt = (cam + k) % 6
                    c2 = open_camera(nxt)
                    if c2 is not None:
                        cap.release()
                        cap, cam = c2, nxt
                        print(f"📷 Now using camera {cam}. (Anka's own setting is CAMERA = {settings.CAMERA} in settings.py)")
                        break
                else:
                    print("📷 (no other camera found)")
            if key == ord("c"):
                t = calibrate(small)
                if t is None:
                    print("📏 (couldn't calibrate – is something straight ahead?)")
                else:
                    settings_tilt[0] = t
                    print(f"📏 Calibrated: the camera looks {t:+.1f}° down (with something 1 m straight ahead)")
            if key == ord("s"):
                path = os.path.join(FOLDER, f"camera_floor_test_{time.strftime('%H%M%S')}.png")
                cv2.imwrite(path, view)
                print("📷 saved", path)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"📷 Done. The latest picture: {OUT}")


if __name__ == "__main__":
    main()
