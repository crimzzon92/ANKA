"""yolo_eyes: Anka's learned eyes - YOLO looks at the camera picture and says what it recognises, where and how far.

Replaces the "pretend detector" for the small things (ball, cup, sock, guitar). Two models, both optional:
    yolo11n.pt        knows 80 everyday things: here "sports ball" -> ball, "cup" -> cup (also couch, dining table, person)
    yoloe-26n-seg.pt  you tell it WORDS to look for ("sock", "guitar") - needs the text model mobileclip2_b.ts the first time
If ultralytics or a model is missing, that part is simply not available and SimBody falls back (colour finder in pretend.py).

For each thing: bearing from the picture column, distance from where the thing touches the floor (the camera sits CAM_H
above the floor and is tilted by the head), or - for a cup on a table - from how big it looks (a cup is ~8 cm wide).
The work runs in its own thread (about 5 pictures a second at most) so the legs never wait for it.

    eyes = YoloEyes(link, gaze_fn)          # gaze_fn() -> (head yaw rad, head pitch rad)
    eyes.start(); eyes.latest(max_age=0.8)  # [(name, "thing", bearing_deg (+left), distance_m)]  (+ eyes.names = what it can find)
"""
import math
import os
import threading
import time

try:
    import cv2
    import numpy as np
except ImportError:                                # (no camera tools: no eyes)
    cv2 = np = None

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDERS = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "models"), HERE]
CAM_H = 0.20
FOV_H, FOV_V = 45.0, 73.0
COCO = {"sports ball": "ball", "cup": "cup"}                 # what yolo11n's words mean in Anka's room
COCO_EXTRA = {"couch": "sofa", "dining table": "table", "person": "person", "bed": "bed", "chair": "chair"}   # (only shown in the test)
WORDS = ["sock", "guitar"]                                    # what yoloe is asked to find
KNOWN_WIDTH = {"cup": 0.08, "ball": 0.07}                    # m - used when the floor is hidden (a cup stands on a table)
ON_FLOOR = {"ball", "sock", "guitar"}
CONF = {"yolo": 0.30, "yoloe": 0.20}


def find_weights(name):
    for f in FOLDERS:
        p = os.path.join(f, name)
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def look(frame, boxes, pitch_deg, rng=None):
    """boxes: [(name, conf, x0, y0, x1, y1)] in the frame's pixels -> [(name, "thing", bearing_deg, distance_m, conf)]."""
    h, w = frame.shape[:2]
    fy = (h / 2) / math.tan(math.radians(FOV_V / 2))
    fx = (w / 2) / math.tan(math.radians(FOV_H / 2))
    out = []
    for name, conf, x0, y0, x1, y1 in boxes:
        cx = (x0 + x1) / 2
        bearing = -math.degrees(math.atan2(cx - w / 2, fx))
        d = None
        if name in ON_FLOOR:
            down = math.degrees(math.atan2(y1 - h / 2, fy)) - pitch_deg
            if down > 1.0:
                d = CAM_H / math.tan(math.radians(down))
        if d is None and name in KNOWN_WIDTH and x1 - x0 > 3:
            d = fx * KNOWN_WIDTH[name] / (x1 - x0)
        if d is None or not (0.12 < d < 6.0):
            continue
        out.append((name, "thing", bearing, d, conf))
    return out


class YoloEyes:
    def __init__(self, link, gaze_fn, use_yoloe=True, folder_note=True):
        self.link, self.gaze_fn = link, gaze_fn
        self.models, self.names, self.problems = {}, set(), []
        self._result, self._t = [], 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.fps = 0.0
        self.last_boxes = []
        if cv2 is None:
            self.problems.append("opencv is not installed")
            return
        try:
            from ultralytics import YOLO, YOLOE        # noqa: F401  (YOLOE needs a recent ultralytics)
        except ImportError as e:
            try:
                from ultralytics import YOLO           # an older ultralytics: only yolo11n
                YOLOE = None
            except ImportError:
                self.problems.append(f"ultralytics is not installed ({e}) - pip install ultralytics")
                return
        path = find_weights("yolo11n.pt") or "yolo11n.pt"      # (a name alone: ultralytics downloads it)
        try:
            self.models["yolo"] = YOLO(path)
            self.names |= set(COCO.values())
        except Exception as e:
            self.problems.append(f"yolo11n: {type(e).__name__}: {e}")
        if use_yoloe and YOLOE is not None:
            try:
                path = find_weights("yoloe-26n-seg.pt") or "yoloe-26n-seg.pt"
                m = YOLOE(path)
                m.set_classes(WORDS, m.get_text_pe(WORDS))
                self.models["yoloe"] = m
                self.names |= set(WORDS)
            except Exception as e:
                self.problems.append(f"yoloe: {type(e).__name__}: {e}")

    @property
    def ready(self):
        return bool(self.models)

    def start(self):
        if self.ready:
            threading.Thread(target=self._loop, daemon=True).start()
        return self

    def stop(self):
        self._stop.set()

    def latest(self, max_age=0.8):
        with self._lock:
            return [(n, k, b, d) for (n, k, b, d, c) in self._result] if time.time() - self._t <= max_age else []

    def see(self, frame, pitch_deg, extra=False):
        """One picture -> [(name, "thing", bearing, distance, conf)]; also keeps every box (for the test picture)."""
        boxes, shown = [], []
        for key, model in self.models.items():
            try:
                r = model.predict(frame, imgsz=640, conf=CONF[key], verbose=False)[0]
            except Exception as e:
                self.problems.append(f"{key} failed on a picture: {type(e).__name__}: {e}")
                continue
            for b in r.boxes:
                word = r.names[int(b.cls[0])]
                x0, y0, x1, y1 = [float(v) for v in b.xyxy[0].tolist()]
                name = word if key == "yoloe" else COCO.get(word) or (COCO_EXTRA.get(word) if extra else None)
                if name:
                    shown.append((name, float(b.conf[0]), x0, y0, x1, y1))
        self.last_boxes = shown
        return look(frame, [s for s in shown if s[0] in self.names], pitch_deg)

    def _loop(self):
        last_count, t0, n = -1, time.time(), 0
        while not self._stop.is_set():
            info = self.link.frame_info()
            if info["count"] == last_count or info.get("age") is None or info["age"] > 1.0:
                time.sleep(0.05)
                continue
            last_count = info["count"]
            frame = self.link.frame()
            if frame is None:
                continue
            try:
                _, pitch = self.gaze_fn()
                res = self.see(frame, math.degrees(pitch))
            except Exception as e:
                self.problems.append(f"{type(e).__name__}: {e}")
                time.sleep(0.5)
                continue
            with self._lock:
                self._result, self._t = res, time.time()
            n += 1
            if time.time() - t0 > 5:
                self.fps, t0, n = n / (time.time() - t0), time.time(), 0
