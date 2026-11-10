"""SimBody: Anka's body when it is the Microduck in Pollen's MuJoCo simulator (and, later, the real Microduck).

It has the same face towards his brain as the virtual 2D body in anka/body.py - do(), describe(), camera_picture(),
note_place() - but everything comes from the duck's own senses and everything he does is a real movement:

    senses    camera picture (upright, from his eyes)      depth sensor 8x8        IMU / fall detection
              step counting (odometry)                     the head's own angles
    actions   walk forward / back <cm>   turn left / right <degrees> / around   look left / right / up / down / straight
              look around   sit down / stand up / get up   kick left / right

Since 2026-11-02 it also has a MIND (mind.py, ported from the 2D simulator): he keeps track of where he is by counting his
own steps and correcting himself when he sees a landmark, he remembers where he saw things, he draws his own floor plan from
the depth sensor, and he can "walk to <thing>", "turn to <thing>" and "explore". Which things the camera recognises comes from
pretend.py (a stand-in for YOLO). It still tells the brain FACTS (what the picture and the depth sensor measured, how far he
walked, where he remembers things) and the brain decides what to do with them.

Used from Anka with   python run_anka.py --mujoco   (see the README in this folder).
"""
import json
import math
import os
import random
import re
import sys
import threading
import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from . import room_layout as R
    from . import mind as M
    from . import pretend as P
    from . import yolo_eyes as Y
    from . import living as L
    from .duck_link import DuckLink
except ImportError:                      # run from inside this folder (tests, scripts)
    import room_layout as R
    import mind as M
    import pretend as P
    import yolo_eyes as Y
    import living as L
    from duck_link import DuckLink

WALK_SPEED = 0.30          # m/s walking forward (the walking policy does not get going at 0.15)
BACK_SPEED = 0.20          # m/s backwards
TURN_VX = 0.0             # forward speed while turning (turntest.py finds out what the duck needs)
TURN_MIN = 0.9
TURN_SPEED = 1.4           # rad/s at most (the walking policy ignores slow turns)
STOP_BEFORE = 0.28         # the depth sensor stops him this far (m) before something in front of him
MAX_BACK = 0.30            # m
MAX_WALK = 2.0             # m per action
TRUNK_H = 0.12             # the trunk's height above the floor when standing (m); the sensor pose is given relative to the trunk
SOUNDS = {"chirp": "a short quack", "greet": "a happy hello-quack, sometimes a double wak-wak", "inquire": "a rising, asking quack",
          "coo": "a soft, drowsy, content sound", "alarm": "a sharp honk", "peck": "a low little 'tock'",
          "wheee": "a long happy 'wheee'"}
SOUND_ALIASES = {"quack": "chirp", "honk": "alarm", "whee": "wheee", "hello": "greet", "question": "inquire", "tock": "peck"}
SOUND_WORDS = r"\b(quack|chirp|coo|greet|inquire|honk|alarm|tock|wheee?|make (a|your) \w+ sound)\b"   # ('peck' = the move now)
# Pollen's ready-made emotions (release 0.18.9, robot.do <name>): head, body and beak keyframes + a sound. 2026-11-10
EMOTIONS = ("angry", "curious", "defiant", "devastated", "excited", "impatient", "laugh", "mmh", "mock", "no", "sad",
            "yes", "yes_fast")
EMOTION_ALIASES = {"nod": "yes", "nods": "yes", "shake": "no", "laughing": "laugh", "giggle": "laugh", "sigh": "sad",
                   "hmm": "mmh", "mm": "mmh", "pout": "defiant", "sulk": "defiant", "tease": "mock", "cry": "devastated",
                   "heartbroken": "devastated", "furious": "angry", "eager": "excited", "wonder": "curious"}
EMOTION_DIR = os.path.expanduser("~/Pollen/microduck/robotd/emotions")
EMOTION_WORDS = (r"\b(act out|act|show|express|do|perform)\b.*\b(" + "|".join(EMOTIONS + tuple(EMOTION_ALIASES)) + r")\b"
                 r"|^\W*(" + "|".join(EMOTIONS) + r"|nod|shake (my|your) head|laugh|giggle|sigh)\W*$")
PECK_WORDS = r"\b(peck|bend down|beak (down|to the floor)|nibble)\b"
SOUND_BANK = os.path.expanduser("~/.cache/duck-sim/sounds/duck-a")
LOOKS = {                  # where the head points for each look: (x forward, y left, z up) in the trunk frame
    "left": (1.0, 0.85, 0.13), "right": (1.0, -0.85, 0.13),
    "up": (0.8, 0.0, 0.75), "down": (0.40, 0.0, -0.10),
}
LEVEL = (1.5, 0.0, 0.13)   # straight ahead, level with his eyes
SENSE_EVERY = 0.15         # s: how often he looks at what his eyes and depth sensor say (position, landmarks, floor plan)
STOP_AT = {"sofa": 0.50, "table": 1.0, "bookshelf": 0.50, "door": 0.50, "window": 0.50,    # 'walk to' stops this far from the thing
           "ball": 0.20, "sock": 0.20, "guitar": 0.45, "cup": 0.45}      # (the table's spot is its middle: it is 1 m wide)
SMALL_THINGS = ("ball", "sock")   # things he may walk right up to (the depth sensor must not stop him at the thing itself)
SEGMENT = 1.0              # m: a walk to something is made of pieces this long, so he can look where he is in between
MAX_SEGMENTS = 14
WORDS = {"sofa": ("sofa", "couch"), "table": ("table",), "bookshelf": ("bookshelf", "bookcase", "shelf", "books"),
         "door": ("door",), "window": ("window",), "ball": ("ball",), "sock": ("sock",), "guitar": ("guitar",),
         "cup": ("cup", "mug")}


try:    # what turntest.py measured for this duck
    import json as _json, os as _os
    _cal = _json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "turn_calibration.json")))
    TURN_VX, TURN_MIN, TURN_SPEED = float(_cal["vx"]), float(_cal["rate"]), float(_cal["rate"])
    if "back" in _cal:
        BACK_SPEED = float(_cal["back"])     # 0 = he cannot walk backwards at all
except Exception:
    pass


# what he remembers about the room - a separate file for the simulator, so a real duck never inherits the simulator's room
_DUCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # (the Anka folder)
MEMORY_FILE = (os.path.join(_DUCK, "memory", "simulation", "room_memory_sim.json") if "--simulate" in sys.argv
               else os.path.join(_DUCK, "memory", "real", "room_memory.json"))   # his room memory (kept between days)
os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
for _old in (os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(MEMORY_FILE)),):
    if os.path.exists(_old) and not os.path.exists(MEMORY_FILE):   # (moved here from ankasim/ on 2026-11-04)
        os.replace(_old, MEMORY_FILE)


class _Halt(Exception):
    """He is going to sleep: whatever he was doing stops at once."""


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _metres(d):
    return f"{d * 100:.0f} cm" if d < 1 else f"{d:.1f} m"


def _side(rel_deg):
    a = abs(rel_deg)
    side = "left" if rel_deg > 0 else "right"
    if a <= 12:
        return "straight ahead"
    if a <= 35:
        return f"ahead, a bit to the {side}"
    if a <= 80:
        return f"to your {side}"
    if a <= 140:
        return f"behind you on the {side}"
    return "behind you"


def _quat_rotate(q, v):
    w, x, y, z = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return r @ np.asarray(v, float)


class SimBody:
    real = True                       # the brain treats a body with real senses differently (anka/eyes.py checks this)

    def __init__(self, link=None, wait=8.0):
        self.link = link or DuckLink().start()
        self.lock = threading.RLock()
        self.people = {}              # (nobody is in the simulator's room yet)
        self.bumped = None
        self.actions = []             # (what he chose, what happened) - shown after a test
        self.walked, self.falls = 0.0, 0
        self.notes = []               # [(x, y, text, time)] - what he chose to write down about places (odometry frame)
        self.old_notes = []           # notes from earlier days, still waiting to be slept on
        self.room_text = []           # what he knows about his room, in his own words (written by him while dreaming)
        self._places0 = {}            # where things were last time, in words (loaded from the memory file)
        self._load_memory()
        self.mind = M.Mind(places=self._places0)      # his own position, mental map, floor plan and places
        self._mind_lock = threading.RLock()
        self._sense_t, self._sense_errors = 0.0, 0
        self._visible = {}            # name -> {"b": bearing, "d": distance, "t": when} for what the detector reported lately
        self._ball_at = None          # None = the ball is where it started; False = he kicked it away, nobody knows where
        self._last_moved, self._last_stop = 0.0, None
        self._halted = False          # set when he goes to sleep: every move stops at once
        self._wake = None             # where he REALLY stood when his own coordinates began (x, y, heading deg) - for checking only
        self._audit = []              # (kind, asked, odometry says, truth says ...) for every turn and walk - for the test report only
        self._turn_lag = 0.15         # s: the legs keep turning this long after "stop" (he learns it from his own turns)
        self._stop_bg = threading.Event()
        self.sitting = False
        self.stopped_in_row = 0
        self.trail = []               # Anka-room coordinates from his odometry
        self._pano = None             # the three-views picture after "look around"
        self._gaze = "straight ahead"
        self._was_fallen = False
        ready = self.link.wait_ready(wait)
        self._start = self._odom()
        if not (ready["robot"] and ready["tof"]):
            print(f"[SimBody] the simulator is not (fully) there yet: {self.link.errors()}")
        self.yolo = None                # learned eyes (YOLO), if ultralytics + the models are there; otherwise a colour finder
        if "--yolo" in sys.argv:                  # (off for now: YOLO does not recognise the simulator's props)
            try:
                eyes = Y.YoloEyes(self.link, self._gaze_angles)
                for problem in eyes.problems:
                    print(f"[SimBody] YOLO: {problem}")
                if eyes.ready:
                    self.yolo = eyes.start()
                    print(f"[SimBody] his learned eyes (YOLO) find: {', '.join(sorted(eyes.names))}")
            except Exception as e:
                print(f"[SimBody] YOLO could not start: {type(e).__name__}: {e}")
        if self.yolo is None:
            print("[SimBody] the ball and the sock are found by colour in the picture (pretend.py); YOLO is off (--yolo turns it on)")
        threading.Thread(target=self._background, daemon=True).start()     # his eyes keep working between his moves
        self._home_point = LEVEL         # where the BRAIN last pointed his head (the living body drifts around it)
        self.sounds = {}                 # how often he made each duck sound
        self.emotions, self.pecks = {}, 0    # Pollen's ready-made emotions he chose, and pecks
        self.showing = None                  # (text, until): what drives his body right now - for the room window
        self._pulse_extra = 0.15             # the kick approach learns how long a step pulse must be to take a step
        self.kicks = []                  # every kick: the skill, the robot's answer, how far his eyes saw the ball move
        self.living = L.Living(self, on="--no-living" not in sys.argv).start()   # what his body does by itself between thoughts
        print(f"[SimBody] the moves this body has besides walking: {', '.join(self.link.skills()) or '(none reported)'}")

    # ------------------------------------------------------------------ what he remembers between days
    def _load_memory(self):
        if "--newborn" in sys.argv:    # a newly hatched duck knows nothing about the room
            return
        try:
            data = json.load(open(MEMORY_FILE, encoding="utf-8"))
            self.room_text = [str(t) for t in data.get("room", [])][:8]
            self.old_notes = [(float(n["x"]), float(n["y"]), str(n["text"]), 0.0) for n in data.get("notes", [])]
            self._places0 = {str(k): v for k, v in (data.get("places") or {}).items() if isinstance(v, dict) and v.get("where")}
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def _save_memory(self):
        try:
            notes = [{"x": x, "y": y, "text": t} for (x, y, t, _) in (self.old_notes + self.notes)][-20:]
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump({"room": self.room_text, "notes": notes, "places": self.mind.places}, f, ensure_ascii=False, indent=1)
        except OSError as e:
            print(f"[SimBody] could not save the room memory: {e}")

    def sleep_on_it(self, ask, events=()):
        """Called when he goes to sleep: he turns what he knows about the room, his notes and what happened today into a
        short, updated description of the room - in his own words. `ask(messages, temperature)` is the night brain."""
        o = self.link.odom()
        notes = [t for (_, _, t, _) in self.old_notes + self.notes]
        if not (notes or self.room_text or self.actions or self.mind.places):
            return
        went = [f"{a} -> {r}" for a, r in self.actions[-25:]]
        text = ("You are " + "Anka" + ", a small robot duck, going to sleep. Write what you now know about your room, "
                "for yourself, tomorrow: at most 6 short plain lines, one thing per line, each saying what it is and where it is "
                "(from where you wake up, or next to what). Use your own words. Keep what is still true from before, "
                "correct what you found out was wrong, and leave out what you are not sure about. Invent nothing: use only "
                "the notes and what happened.\n\n"
                + ("What you knew before:\n- " + "\n- ".join(self.room_text) + "\n\n" if self.room_text else "")
                + ("Your notes about places:\n- " + "\n- ".join(notes) + "\n\n" if notes else "")
                + ("Where you saw things (from your own map, in words):\n- "
                   + "\n- ".join(f"the {k} – {v['where']}" for k, v in self.mind.places.items()) + "\n\n" if self.mind.places else "")
                + ("What your body did today (last moves):\n- " + "\n- ".join(went) + "\n\n" if went else "")
                + ("What happened and what you thought:\n" + "\n".join(list(events)[-30:]) + "\n\n" if events else "")
                + "Answer with only the lines, nothing else.")
        try:
            answer = ask([{"role": "user", "content": text}], temperature=0.3)
        except Exception as e:
            print(f"[SimBody] could not sleep on the room: {e}")
            return
        lines = [re.sub(r"^[\s\-*•\d.)]+", "", ln).strip() for ln in (answer or "").splitlines()]
        lines = [ln[:160] for ln in lines if len(ln.split()) >= 3][:6]
        if lines:
            self.room_text = lines
            self.old_notes, self.notes = [], []
            self._save_memory()
            print("🏠 What he knows about his room now:\n   " + "\n   ".join(lines))

    # ------------------------------------------------------------------ small helpers
    def _odom(self):
        return self.link.odom() or (0.0, 0.0, 0.0)

    def _sample_trail(self):
        t = self.link.truth()
        o = (t["x"], t["y"]) if t else self.link.odom_anka()
        if o:
            self.trail.append((o[0], o[1]))

    def _sleep(self, s):
        """Waits - but keeps his eyes working (position, landmarks, floor plan) while he waits."""
        end = time.time() + s
        while True:
            if self._halted:
                raise _Halt()
            time.sleep(max(0.0, min(end - time.time(), SENSE_EVERY)))
            self._sample_trail()
            self._sense()
            if time.time() >= end - 1e-3:
                break

    def _background(self):
        while not self._stop_bg.is_set():
            time.sleep(SENSE_EVERY * 2)
            self._sense()

    # ------------------------------------------------------------------ eyes and mind
    def _up(self):
        """Which way is up, in the trunk's own frame (from gravity, which the robot feels). While he walks the trunk leans
        forward a bit - without this every distance read off the picture or the depth sensor would be wrong by that much."""
        g = ((self.link.state() or {}).get("safety") or {}).get("gravity")
        if g and len(g) == 3:
            n = math.sqrt(sum(c * c for c in g))
            if n > 0.5:
                return (-g[0] / n, -g[1] / n, -g[2] / n)
        return (0.0, 0.0, 1.0)

    def _gaze_angles(self):
        """Where his head points, from the robot's own kinematics: (turned left in radians, tilted up in radians)."""
        cam = ((self.link.state() or {}).get("frames") or {}).get("camera")
        if cam and cam.get("quat") and any(cam["quat"]):
            axis = _quat_rotate(cam["quat"], [0, 0, 1])           # the camera's optical axis in the trunk frame
            up = self._up()
            tilt = math.asin(max(-1.0, min(1.0, sum(a * b for a, b in zip(axis, up)))))     # up/down against the real horizon
            return math.atan2(axis[1], axis[0]), tilt                                         # (turned left, tilted up)
        return 0.0, 0.0

    def _detect(self, yaw, pitch):
        """The pretend detector (see pretend.py): what is in the picture, at which angle and how far away."""
        t = self.link.truth()
        if not t:
            return []
        by_yolo = self.yolo.latest(0.8) if self.yolo is not None else []     # what YOLO recognised in the picture
        named = {d[0] for d in by_yolo}
        # whatever YOLO names comes from YOLO; the rest still comes from the stand-in (ball/sock: the colour finder below)
        seen = P.detect(t["x"], t["y"], t["heading"], math.degrees(yaw), math.degrees(pitch), ball_at=self._ball_at,
                        skip=tuple(named | ({"ball", "sock"} if P.cv2 is not None else set())))
        seen += by_yolo
        frame, boxes = None, []
        if P.cv2 is not None:                       # the ball and the sock: if YOLO did not name them, look for their colour
            missing = {"ball", "sock"} - {d[0] for d in by_yolo}
            info = self.link.frame_info()
            if missing and info.get("age") is not None and info["age"] < 0.6:
                frame = self.link.frame()
                seen += [d for d in P.find_by_colour(frame, math.degrees(pitch)) if d[0] in missing]
                boxes = list(P.LAST)
                self._save_colour_picture(frame, math.degrees(pitch))
        self._publish_seen(seen, boxes, frame, math.degrees(pitch), {d[0] for d in by_yolo})
        return seen

    SEEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seen_now.json")

    def _publish_seen(self, seen, boxes, frame, pitch_deg, by_yolo=()):
        """For the eyes window only (eyes.py draws it on the camera picture): what he found in the picture right now.
        The colour finder's boxes are real boxes in the picture. Things from the stand-in detector (or YOLO) have only an
        angle and a distance, so their box is where that angle and distance would put them. The brain never reads this."""
        try:
            if frame is not None:
                fh, fw = frame.shape[:2]
            else:
                f = self.link.frame()
                fh, fw = (f.shape[:2] if f is not None else (0, 0))
            items = []
            for name, x0, y0, bw, bh, d, verdict in boxes:
                items.append({"name": name, "d": round(float(d), 2), "box": [x0, y0, bw, bh],
                              "how": "colour" if verdict == "accepted" else "rejected"})
            found = {i["name"] for i in items if i["how"] == "colour"}
            if fw and fh:
                fx = (fw / 2) / math.tan(math.radians(P.FOV_H / 2))
                fy = (fh / 2) / math.tan(math.radians(P.FOV_V / 2))
            for name, kind, bearing, d in seen:
                if name in found:
                    continue
                item = {"name": name, "d": round(float(d), 2), "how": "yolo" if name in by_yolo else "stand-in"}
                if fw and fh and d > 0.05:
                    top = P.LANDMARKS[name][2] if name in P.LANDMARKS else P.THING_HEIGHT.get(name, 0.2)
                    bottom = max(0.0, top - 0.12) if name in ("cup", "window") else 0.0
                    cx = fw / 2 - fx * math.tan(math.radians(bearing))
                    y_top = fh / 2 - fy * math.tan(math.radians(math.degrees(math.atan2(top - P.CAM_H, d)) - pitch_deg))
                    y_bot = fh / 2 - fy * math.tan(math.radians(math.degrees(math.atan2(bottom - P.CAM_H, d)) - pitch_deg))
                    half = max(8, fx * (0.08 if kind == "thing" else 0.25) / d)
                    item["box"] = [int(cx - half), int(min(y_top, y_bot)), int(2 * half), int(max(4, abs(y_bot - y_top)))]
                items.append(item)
            tmp = self.SEEN_FILE + ".tmp"
            with open(tmp, "w") as fh_out:
                json.dump({"t": time.time(), "w": fw, "h": fh, "pitch": round(pitch_deg, 1), "items": items}, fh_out)
            os.replace(tmp, self.SEEN_FILE)
        except Exception:
            pass

    def _save_colour_picture(self, frame, pitch_deg):
        """For checking afterwards: a picture (with the boxes the colour finder drew) every few seconds while it sees a
        ball or a sock, in test_logs/frames_<start time>/ - so a false ball can be seen with our own eyes."""
        try:
            if not P.LAST or frame is None:
                return
            now = time.time()
            if now - getattr(self, "_pic_t", 0) < 2.5 or getattr(self, "_pic_n", 0) >= 120:
                return
            self._pic_t, self._pic_n = now, getattr(self, "_pic_n", 0) + 1
            if not hasattr(self, "_pic_dir"):
                self._pic_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_logs",
                                             "frames_" + time.strftime("%Y-%m-%d_%H%M"))
                os.makedirs(self._pic_dir, exist_ok=True)
            img = frame.copy()
            tag = ""
            for name, x0, y0, bw, bh, d, verdict in P.LAST:
                ok = verdict == "accepted"
                P.cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (0, 255, 0) if ok else (0, 0, 255), 2)
                P.cv2.putText(img, f"{name} {d:.2f}m" + ("" if ok else " NO"), (x0, max(12, y0 - 4)), P.cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (0, 255, 0) if ok else (0, 0, 255), 1)
                tag += f"_{name}{'' if ok else 'X'}{d:.1f}"
            P.cv2.imwrite(os.path.join(self._pic_dir, f"{self._pic_n:03d}{tag}_pitch{pitch_deg:.0f}.png"), img)
        except Exception:
            pass

    def _rays(self):
        """The depth sensor as rays for his floor plan: [(angle (+ = left, in the body's frame), how far the floor is free,
        how far to the first thing standing there or None)] - one per column of the 8x8 picture."""
        tof, st = self.link.tof(), self.link.state()
        if tof is None or tof["age"] > 2.0:
            return []
        tp = ((st or {}).get("frames") or {}).get("tof")
        if not tp or not tp.get("quat") or not any(tp["quat"]):
            return []
        m = tof["m"]
        rows, cols = m.shape
        half = math.radians(45.0) / 2
        edges = np.linspace(-half, half, cols + 1)
        centres = (edges[:-1] + edges[1:]) / 2
        height = TRUNK_H + tp["pos"][2]
        up = self._up()
        out = []
        for c in range(cols):
            angs, hit, free = [], None, 0.0
            for r in range(rows):
                el, az = -centres[r], -centres[c]
                v = _quat_rotate(tp["quat"], [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
                vz = sum(a * b for a, b in zip(v, up))                # how much it points up, against the real world
                horiz = math.sqrt(max(0.0, 1.0 - vz * vz))
                if horiz < 0.05:
                    continue
                angs.append(math.atan2(v[1], v[0]))
                meas = m[r, c]
                if math.isnan(meas):
                    if vz <= 0.02:                        # level or downward and nothing within 4 m: open floor ahead
                        free = max(free, M.FREE_NO_HIT)
                    continue
                dh, z_hit = meas * horiz, height + meas * vz
                if z_hit > 0.06:                          # something standing there (not the floor)
                    hit = dh if hit is None else min(hit, dh)
                else:
                    free = max(free, dh)
            if angs and (hit is not None or free > 0):
                out.append((float(np.mean(angs)), hit if hit is not None else free, hit))
        return out

    def _sense(self, force=False):
        """One look at the world: step counting -> position, the detector -> landmarks and things, the depth sensor -> floor
        plan. Called all the time (while he walks, turns, looks, waits)."""
        now = time.time()
        if not force and now - self._sense_t < SENSE_EVERY:
            return
        with self._mind_lock:
            self._sense_t = now
            try:
                if self.sitting or self.link.fallen():
                    return                                # (sitting or lying, his eyes see something else)
                o = self.link.odom()
                if o is None:
                    return
                self.mind.predict(o)
                if self._wake is None and self.mind.mu is not None:
                    tw = self.link.truth()
                    if tw:
                        self._wake = (tw["x"], tw["y"], tw["heading"])
                yaw, pitch = self._gaze_angles()
                det = self._detect(yaw, pitch)
                self.mind.see(det, yaw, now)
                for name, kind, bearing, d in det:
                    self._visible[name] = {"b": bearing, "d": d, "t": now, "head": math.degrees(yaw)}
                rays = self._rays()
                if rays:
                    self.mind.add_scan(rays, now)
                    self.mind.wall_heading(rays, now)
                if self.mind.places_changed:
                    self.mind.places_changed = False
                    if self.mind.moved:
                        print(f"🗺️  (Anka noticed the {self.mind.moved[0]} has moved: {self.mind.moved[1]} → {self.mind.moved[2]})")
                        self.mind.moved = None
                    self._save_memory()
            except Exception as e:                        # (a bug in the mind must never stop his legs)
                self._sense_errors += 1
                if self._sense_errors <= 3:
                    print(f"[SimBody] his mind had a problem: {type(e).__name__}: {e}")

    def _visible_name(self, name, max_age=1.0):
        v = self._visible.get(name)
        return v if v and time.time() - v["t"] <= max_age else None

    def _check_fall(self):
        f = self.link.fallen()
        if f and not self._was_fallen:
            self.falls += 1
        self._was_fallen = f
        return f

    def battery(self):
        return 100

    # ------------------------------------------------------------------ the depth sensor, in words
    def _zones(self):
        """Splits the depth sensor's 8x8 zones into 'floor' and 'something standing there', using where the sensor points
        (from the robot's own kinematics). Returns (obstacle distance per zone in m or nan, floor mask, sensor height) or None."""
        tof = self.link.tof()
        st = self.link.state()
        if tof is None or tof["age"] > 2.0:
            return None
        m = tof["m"]
        rows, cols = m.shape
        frames = (st or {}).get("frames") or {}
        tp = frames.get("tof")
        z0 = TRUNK_H
        if not tp or not tp.get("quat") or not any(tp["quat"]):
            return m.copy(), np.zeros_like(m, bool), None
        half = math.radians(45.0) / 2
        edges = np.linspace(-half, half, cols + 1)
        centres = (edges[:-1] + edges[1:]) / 2
        height = z0 + tp["pos"][2]
        obstacle = np.full_like(m, np.nan)
        floor = np.zeros_like(m, bool)
        for r in range(rows):
            for c in range(cols):
                el, az = -centres[r], -centres[c]
                d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
                dz = sum(a * b for a, b in zip(_quat_rotate(tp["quat"], d), self._up()))   # how much it points up, against the real world
                meas = m[r, c]
                if math.isnan(meas):
                    continue
                if dz < -0.02:
                    expect = height / -dz
                    if abs(meas - expect) < 0.12 * expect + 0.05:
                        floor[r, c] = True
                        continue
                obstacle[r, c] = meas
        return obstacle, floor, height

    def _ahead(self):
        """Distance (m) to the nearest thing straight ahead that is not floor, or nan."""
        z = self._zones()
        if z is None:
            return float("nan")
        obstacle = z[0]
        band = obstacle[:, 2:6] if z[2] is not None else obstacle[:4, 2:6]    # without kinematics: ignore the lower rows
        return float(np.nanmin(band)) if not np.all(np.isnan(band)) else float("nan")

    def depth_words(self):
        z = self._zones()
        if z is None:
            return "no reading right now"
        obstacle, floor, _ = z
        groups = {"to the left": obstacle[:, 0:3], "straight ahead": obstacle[:, 3:5], "to the right": obstacle[:, 5:8]}
        parts = []
        for name, g in groups.items():
            parts.append(f"{name}: {_metres(float(np.nanmin(g)))}" if not np.all(np.isnan(g)) else f"{name}: nothing within 4 m")
        words = "; ".join(parts)
        if np.any(floor):
            lowest = np.where(floor.any(axis=1))[0].max()
            fl = self.link.tof()["m"][lowest, 3:5]
            if not np.all(np.isnan(fl)):
                words += f". Bare floor starts about {_metres(float(np.nanmin(fl)))} in front of you"
        return words

    # ------------------------------------------------------------------ his body in words
    def _gaze_words(self):
        st = self.link.state() or {}
        cam = (st.get("frames") or {}).get("camera")
        if cam and cam.get("quat") and any(cam["quat"]):
            axis = _quat_rotate(cam["quat"], [0, 0, 1])           # the camera's optical axis in the trunk frame
            yaw = math.degrees(math.atan2(axis[1], axis[0]))
            pitch = math.degrees(self._gaze_angles()[1])          # against the real horizon
            out = []
            if abs(yaw) > 6:
                out.append(f"turned {abs(yaw):.0f}° to the {'left' if yaw > 0 else 'right'}")
            if pitch > 12:
                out.append(f"tilted up about {pitch:.0f}°")
            elif pitch < -12:
                out.append(f"tilted down about {-pitch:.0f}°")
            return " and ".join(out)
        return "" if self._gaze == "straight ahead" else f"looking {self._gaze}"

    def describe(self, last=None):
        fallen = self.link.fallen()
        posture = "lying on your side – you fell over" if fallen else ("sitting" if self.sitting else "standing")
        head = self._gaze_words()
        lines = [f"[Your body: you are {posture}" + (f", your head is {head}" if head else "") + ". "
                 "Your beak is empty (you can't pick things up yet).]",
                 "[The attached picture: your camera right now – what your eyes really see.]",
                 f"[Depth sensor (the middle of your picture, about 45° wide): {self.depth_words()}.]"]
        news = self.living.take_news() if self.living else []
        if news:   # his feet's reflexes (living.py) - facts he should know, told once
            lines.append("[While you were thinking: " + "; ".join(news[-3:]) + ".]")
        o, s = self.link.odom(), self._start
        if o:
            dx, dy = o[0] - s[0], o[1] - s[1]
            far = math.hypot(dx, dy)
            turned = math.degrees(_wrap(o[2] - s[2]))
            lines.append(f"[Your own step counting (it drifts): you are about {_metres(far)} from where you woke up, and you face "
                         + ("the way you started" if abs(turned) < 10 else f"{abs(turned):.0f}° to the {'left' if turned > 0 else 'right'} of the way you started")
                         + ". You can't see behind you.]")
            if self.mind.pose() is not None:
                words = self.mind.things_words(head_pitch=self._gaze_angles()[1])
                if words:
                    lines.append("[Your mental map – where you remember seeing things, guessed from counting your own steps and "
                                 "from landmarks you saw (it gets less exact the more you walk): " + "; ".join(words) + ".]")
                lines.append(f"[Your own floor plan (a robot-vacuum-style map from your depth sensor): {self.mind.floor_words()}.]")
                earlier = [f"the {k} – {v['where']} (seen {v.get('date', 'earlier')})" for k, v in self.mind.places.items()
                           if self.mind.pos_of(k) is None]
                if earlier:
                    lines.append("[Where things were when you last saw them (they may have been moved): " + "; ".join(earlier) + ".]")
            if self.room_text:
                lines.append("[What you know about your room from earlier days (your own words, written while you slept): "
                             + " | ".join(self.room_text) + "]")
            if self.old_notes:
                spots = []
                for (x, y, text, t) in self.old_notes[-5:]:
                    rel = math.degrees(_wrap(math.atan2(y - o[1], x - o[0]) - o[2]))
                    spots.append(f"'{text}' ({_metres(math.hypot(x - o[0], y - o[1]))} {_side(rel)})")
                lines.append("[Notes from earlier days, measured from where you woke up then (that may not be where you woke up today): "
                             + "; ".join(spots) + ".]")
            if self.notes and o:
                spots = []
                for (x, y, text, t) in self.notes[-5:]:
                    rel = math.degrees(_wrap(math.atan2(y - o[1], x - o[0]) - o[2]))
                    d = math.hypot(x - o[0], y - o[1])
                    spots.append(f"'{text}' ({'right here' if d < 0.3 else _metres(d) + ' ' + _side(rel)})")
                lines.append("[Your own notes about places: " + "; ".join(spots) + ".]")
        if self.stopped_in_row >= 3:
            lines.append(f"[Note: your last {self.stopped_in_row} tries to walk were stopped almost at once – something keeps blocking you there.]")
        if last:
            lines.append(f"[Your last move: {last}]")
        return "\n".join(lines)

    def note_place(self, text):
        o = self.link.odom()
        text = (text or "").strip().strip('"').strip()
        if not o or not text or re.fullmatch(r"(?i)(none|no|nothing)\W*", text):
            return
        # a note close to an earlier one replaces it (one note per spot - not the same note again and again)
        self.notes = [n for n in self.notes if math.hypot(n[0] - o[0], n[1] - o[1]) >= 0.5]
        self.notes.append((o[0], o[1], text[:80], time.time()))
        self.notes = self.notes[-12:]
        self._save_memory()

    def camera_picture(self):
        """The picture the brain gets: what the camera sees - after 'look around', the three views side by side (once)."""
        with self.lock:
            if self._pano is not None:
                pic, self._pano = self._pano, None
                return pic
        return self._frame_now()

    def _frame_now(self, fresh=False):
        if fresh:
            n0 = self.link.frame_info()["count"]
            end = time.time() + 2.0
            while time.time() < end and self.link.frame_info()["count"] < n0 + 2:
                time.sleep(0.03)
        f = self.link.frame()
        if f is None:
            f = np.full((640, 360, 3), 40, np.uint8)
            if cv2 is not None:
                cv2.putText(f, "no camera picture", (60, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        return f

    # ------------------------------------------------------------------ the compatibility bits the old 2D body had
    def place_person(self, name, spot):
        """Nobody has a body in the 3D room - but the script still says who is in the room (so he can HEAR someone go
        out or come in, and a voice counts as someone being around). 2026-11-10"""
        if spot is None:
            self.people.pop(name, None)
        else:
            try:
                from anka.body import SPOTS
                self.people[name] = tuple(SPOTS.get(spot, (0.0, 0.0)))[:2]
            except Exception:
                self.people[name] = (0.0, 0.0)

    def voice_from(self, name):
        return "from somewhere in the room"

    def hear(self, name):
        self.living.heard()          # (his body lifts his head at a voice - see living.py)

    def visible(self):
        return []

    def seen_names(self, min_pixels=30):
        return {}

    def remember_view(self, things=True):
        pass

    def draw(self, live=False, caption="", **kw):
        """A top-down picture of his walk (for the file saved after a test): the room, his step counting, and where he really was."""
        scale = 100
        W, H = int(R.ROOM_W * scale) + 40, int(R.ROOM_H * scale) + 60
        img = np.full((H, W, 3), 245, np.uint8)
        if cv2 is None:
            return img

        def px(x, y):
            return int(20 + x * scale), int(H - 40 - y * scale)

        cv2.rectangle(img, px(0, R.ROOM_H), px(R.ROOM_W, 0), (60, 60, 60), 2)
        for name, (x1, y1, x2, y2, h) in R.FURNITURE.items():
            cv2.rectangle(img, px(x1, y2), px(x2, y1), (200, 190, 170), -1)
            cv2.putText(img, name, (px(x1, y2)[0] + 4, px(x1, y2)[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 80, 60), 1, cv2.LINE_AA)
        if len(self.trail) > 1:
            cv2.polylines(img, [np.array([px(x, y) for x, y in self.trail], np.int32)], False, (0, 170, 255), 2)
        pose = self.mind.pose()
        if pose is not None:
            for name, pos in self.mind.landmarks().items():      # where HE thinks the landmarks are (purple squares)
                rx, ry = self._own_to_room(*pos)
                cv2.rectangle(img, (px(rx, ry)[0] - 4, px(rx, ry)[1] - 4), (px(rx, ry)[0] + 4, px(rx, ry)[1] + 4), (160, 60, 160), 1)
                cv2.putText(img, name + "?", (px(rx, ry)[0] + 7, px(rx, ry)[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 60, 160), 1, cv2.LINE_AA)
            for name, t in self.mind.things.items():             # where HE thinks the small things are (purple rings)
                if t.get("kind") == "thing" and t.get("pos"):
                    rx, ry = self._own_to_room(*t["pos"])
                    cv2.circle(img, px(rx, ry), 8, (160, 60, 160), 2)
                    cv2.putText(img, name + ("?" if t.get("kicked") else ""), (px(rx, ry)[0] + 10, px(rx, ry)[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 60, 160), 1, cv2.LINE_AA)
            rx, ry = self._own_to_room(pose[0], pose[1])         # where he thinks he is (purple ring)
            cv2.circle(img, px(rx, ry), 9, (160, 60, 160), 2)
        t = self.link.truth()
        if t:
            cv2.circle(img, px(t["x"], t["y"]), 6, (60, 160, 60), -1)
            cv2.putText(img, "him (truth)", (px(t["x"], t["y"])[0] + 8, px(t["x"], t["y"])[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 160, 60), 1, cv2.LINE_AA)
        if isinstance(caption, (list, tuple)):          # the room window passes several lines
            caption = "  |  ".join(str(c) for c in caption)
        caption = str(caption).encode("ascii", "ignore").decode()   # (cv2's font has no emoji or non-ASCII letters)
        cv2.putText(img, caption[:90], (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        try:
            return np.hstack([img, self.floor_picture(H)])
        except Exception:
            return img

    def _own_to_room(self, x, y):
        """His own coordinates (x forward, y left, from where he woke up) -> the room's (for the picture only)."""
        wx, wy, wh = self._wake or R.START
        h0 = math.radians(wh)
        return wx + math.cos(h0) * x - math.sin(h0) * y, wy + math.sin(h0) * x + math.cos(h0) * y

    def _room_to_own(self, rx, ry):
        wx, wy, wh = self._wake or R.START
        h0 = math.radians(wh)
        dx, dy = rx - wx, ry - wy
        return math.cos(h0) * dx + math.sin(h0) * dy, -math.sin(h0) * dx + math.cos(h0) * dy

    def floor_picture(self, height):
        """His own floor plan as a picture: grey = never looked there, white = free floor, black = something there; up =
        the way he faced when he woke up. Purple ring = where he thinks he is, green = the landmarks he knows."""
        g = self.mind.floor
        img = np.full(g.shape + (3,), 150, np.uint8)
        img[g < -0.5] = (255, 255, 255)
        img[g > M.BLOCKED] = (20, 20, 20)
        img = img[::-1, ::-1].copy()           # rows: x forward = up, columns: y left = left
        # crop to what he has seen (+ a margin) so the plan is not a tiny dot in the middle
        seen = np.argwhere(np.abs(g[::-1, ::-1]) > 0.05)
        cell = M.CELL

        def at(x, y):
            i, k = self.mind.cell(x, y)
            return g.shape[1] - 1 - k, g.shape[0] - 1 - i          # (column, row) in the flipped picture

        pose = self.mind.pose()
        marks = []                              # (column, row, text, colour) - the words are drawn after the picture is enlarged
        if pose is not None:
            for name, pos in self.mind.landmarks().items():
                c, r = at(*pos)
                cv2.circle(img, (c, r), 3, (60, 160, 60), -1)
                marks.append((c, r, name, (40, 120, 40)))
            for name, t in self.mind.things.items():            # the small things he remembers (orange; "?" = he kicked it since)
                if t.get("kind") == "thing" and t.get("pos"):
                    c, r = at(*t["pos"])
                    cv2.circle(img, (c, r), 3, (30, 130, 240), -1)
                    marks.append((c, r, name + ("?" if t.get("kicked") else ""), (20, 100, 220)))
            fr = self.mind.frontier()
            if fr:
                c, r = at(*fr[0])
                cv2.drawMarker(img, (c, r), (40, 160, 40), cv2.MARKER_CROSS, 6, 1)
            c, r = at(pose[0], pose[1])
            cv2.circle(img, (c, r), 5, (160, 60, 160), 1)
            c2, r2 = at(pose[0] + 0.3 * math.cos(pose[2]), pose[1] + 0.3 * math.sin(pose[2]))
            cv2.line(img, (c, r), (c2, r2), (160, 60, 160), 1)
        r0 = c0 = 0
        if len(seen):
            r0, c0 = np.maximum(seen.min(axis=0) - 12, 0)
            r1, c1 = np.minimum(seen.max(axis=0) + 12, np.array(g.shape) - 1)
            img = img[r0:r1 + 1, c0:c1 + 1]
        scale = max(1, int(height / max(img.shape[:2])))
        img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        for c, r, text, colour in marks:
            x, y = int((c - c0 + 0.5) * scale), int((r - r0 + 0.5) * scale)
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.putText(img, text, (x + 7, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        out = np.full((height, max(img.shape[1], 200), 3), 245, np.uint8)
        out[:min(height, img.shape[0]), :img.shape[1]] = img[:height]
        cv2.putText(out, "his own floor plan", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 60, 160), 1, cv2.LINE_AA)
        cv2.putText(out, "green = landmarks  orange = things he remembers  purple = him", (6, height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1, cv2.LINE_AA)
        return out

    # ---- checking only: how right was his step counting, move by move (uses the simulator's truth; the brain never sees this)
    def _audit_start(self):
        t, o = self.link.truth(), self.link.odom()
        return (t, o, getattr(self.link, "gyro_yaw", lambda: None)()) if t and o else None

    def _audit_end(self, kind, asked, a0):
        try:
            t1, o1 = self.link.truth(), self.link.odom()
            if a0 is None or not t1 or not o1:
                return
            (t0, o0, g0) = a0
            g1 = getattr(self.link, "gyro_yaw", lambda: None)()
            self._audit.append({
                "kind": kind, "asked": asked,
                "odom_turn": math.degrees(_wrap(o1[2] - o0[2])), "true_turn": math.degrees(_wrap(math.radians(t1["heading"] - t0["heading"]))),
                "gyro_turn": math.degrees(g1 - g0) if g0 is not None and g1 is not None else None,
                "odom_dist": math.hypot(o1[0] - o0[0], o1[1] - o0[1]), "true_dist": math.hypot(t1["x"] - t0["x"], t1["y"] - t0["y"]),
            })
        except Exception:
            pass

    def _aligned_report(self):
        """His landmark map compared with the truth AFTER turning and shifting it to the best fit: tells a map that is only
        turned as a whole (harmless: everything in it still fits together) from one that is really crooked inside."""
        pairs = []
        for name, (lx, ly, _h) in P.LANDMARKS.items():
            est = self.mind.pos_of(name) if name in self.mind.index else None
            if est:
                pairs.append((est, self._room_to_own(lx, ly)))
        if len(pairs) < 2:
            return []
        B = np.array([p[0] for p in pairs]); A = np.array([p[1] for p in pairs])      # B = what he believes, A = the truth
        bc, ac = B.mean(0), A.mean(0)
        b, a = B - bc, A - ac
        ang = math.atan2(float((b[:, 0] * a[:, 1] - b[:, 1] * a[:, 0]).sum()), float((b * a).sum()))
        c, s = math.cos(ang), math.sin(ang)
        Bf = b @ np.array([[c, s], [-s, c]]) + ac
        res = np.hypot(*(Bf - A).T)
        shift = math.hypot(*(ac - bc))
        return [f"best fit of his map onto the truth: the whole map is turned {math.degrees(-ang):+.1f}° (shifted {shift * 100:.0f} cm); "
                f"after that the landmarks are on average {res.mean() * 100:.0f} cm from the truth (worst {res.max() * 100:.0f} cm) – "
                f"that last number is what is really crooked INSIDE his map"]

    def _wall_report(self):
        m = self.mind
        if m.wall_ref is None:
            return ["walls: he has not seen enough straight walls to tell the room's main direction yet"]
        out = [f"walls: the room's main direction was decided as {m.wall_ref:.1f}° (in his own coordinates, mod 90) from {len(m._wall_runs)} straight runs; "
               f"{len(m.wall_log)} wall measurements" + (f", {m.wall_fixes} used to correct his heading" if m.wall_fixes else " (not used to steer – measuring only)")]
        t, pose = self.link.truth(), m.pose()
        if m.wall_log and t and pose is not None and self._wake:
            th = _wrap(math.radians(t["heading"]) - math.radians(self._wake[2]))
            truth_err = math.degrees(_wrap(pose[2] - th))                      # how far off his heading really is now
            recent = [d for _, d, _ in m.wall_log[-15:]]
            out.append(f"   the last walls said his heading is off by {np.median(recent):+.1f}° (median of {len(recent)}); in truth it is {truth_err:+.1f}° off "
                       f"– they agree if the two numbers are close (the truth's number includes the room's own start direction, so compare changes)")
        return out

    def _audit_report(self):
        out = []
        turns = [a for a in self._audit if a["kind"] == "turn" and abs(a["odom_turn"]) > 8]
        if turns:
            ot, tt = sum(abs(a["odom_turn"]) for a in turns), sum(abs(a["true_turn"]) for a in turns)
            worst = max(turns, key=lambda a: abs(a["odom_turn"] - a["true_turn"]))
            out.append(f"turns ({len(turns)}): the odometry said {ot:.0f}° in all, the truth was {tt:.0f}° (truth/odometry = {tt / max(ot, 1):.2f}); "
                       f"worst: asked {math.degrees(worst['asked']):+.0f}°, odometry {worst['odom_turn']:+.0f}°, truth {worst['true_turn']:+.0f}°")
            out.append("   each turn (asked / odometry / truth / gyro, degrees): " + "; ".join(
                f"{math.degrees(a['asked']):+.0f}/{a['odom_turn']:+.0f}/{a['true_turn']:+.0f}/"
                + (f"{a['gyro_turn']:+.0f}" if a.get("gyro_turn") is not None else "-") for a in turns))
            gy = [a for a in turns if a.get("gyro_turn") is not None]
            if gy:
                out.append(f"   the IMU gyro added up said {sum(abs(a['gyro_turn']) for a in gy):.0f}° in all (the truth {sum(abs(a['true_turn']) for a in gy):.0f}°, "
                           f"the odometry {sum(abs(a['odom_turn']) for a in gy):.0f}°) – the better of the two is the one to steer by")
            else:
                out.append("   (the robot sends no gyro readings in robot.state, so no gyro heading to compare)")
        walks = [a for a in self._audit if a["kind"] in ("walk", "walk back") and a["odom_dist"] > 0.1]
        if walks:
            od, td = sum(a["odom_dist"] for a in walks), sum(a["true_dist"] for a in walks)
            drift = sum(a["true_turn"] - a["odom_turn"] for a in walks)
            out.append(f"walks ({len(walks)}): the odometry said {od:.1f} m in all, the truth was {td:.1f} m (truth/odometry = {td / max(od, 0.01):.2f}); "
                       f"while walking the odometry's heading and the truth's differed by {drift:+.0f}° in all")
        return out

    def report(self):
        """Lines for the test report: how well did his own senses work, compared with the truth (checking only)."""
        lines = list(self.living.report())
        if self.emotions or self.pecks:
            lines.append("emotions he acted out: " + (", ".join(f"{k} ×{v}" for k, v in self.emotions.items()) or "none")
                         + f"; pecks: {self.pecks}")
        if self.sounds:
            lines.append("duck sounds he chose: " + ", ".join(f"{k} ×{v}" for k, v in self.sounds.items()))
        if self.kicks:
            lines.append(f"kicks ({len(self.kicks)}): " + "; ".join(
                f"{k['skill']} → robot said {str(k['answer'])[:60]}, ball at (fwd, left) {k.get('ball_at')} m before" + (f", ball moved {k['moved_m']} m by his eyes" if 'moved_m' in k else "")
                for k in self.kicks[-8:]))
        t, pose = self.link.truth(), self.mind.pose()
        # numbers for the thought log's end row (tools/score_run.py scores them) - 2026-11-10
        self.summary_numbers = dict(self.living.numbers(), falls=self.falls, kicks=len(self.kicks),
                                    kicks_moved_ball=sum(1 for k in self.kicks if k.get("moved_m", 0) >= 0.08),
                                    sounds=sum(self.sounds.values()), map_error_cm=None,
                                    emotions=sum(self.emotions.values()), pecks=self.pecks)
        if t and pose is not None:
            tx, ty = self._room_to_own(t["x"], t["y"])
            th = _wrap(math.radians(t["heading"]) - math.radians((self._wake or R.START)[2]))
            err, herr = math.hypot(pose[0] - tx, pose[1] - ty), math.degrees(_wrap(pose[2] - th))
            self.summary_numbers["map_error_cm"] = round(err * 100)
            lines.append(f"where he thought he was at the end: {err * 100:.0f} cm off, facing {herr:+.1f}° off "
                         f"(landmarks corrected him {self.mind.fixes} times; he knows {len(self.mind.index)} landmarks)")
            o = self.link.odom()
            if o is not None:
                ox, oy, oyaw = self.mind._own(o)
                lines.append(f"(his raw step counting alone: {math.hypot(ox - tx, oy - ty) * 100:.0f} cm off, "
                             f"facing {math.degrees(_wrap(oyaw - th)):+.1f}° off)")
            errs = []
            for name, (lx, ly, _h) in P.LANDMARKS.items():
                est = self.mind.pos_of(name) if name in self.mind.index else None
                if est:
                    ex, ey = self._room_to_own(lx, ly)
                    errs.append(math.hypot(est[0] - ex, est[1] - ey))
            if errs:
                lines.append(f"his landmark map is on average {np.mean(errs) * 100:.0f} cm from the truth (worst {max(errs) * 100:.0f} cm)")
            lines += self._aligned_report()
        if self._wake and abs(_wrap(math.radians(self._wake[2] - R.START[2]))) > math.radians(2):
            lines.append(f"(he woke up facing {self._wake[2]:.0f}°, not the {R.START[2]:.0f}° of the start spot – his own coordinates start from there)")
        lines += self._audit_report()
        lines += self._wall_report()
        lines.append(f"his own floor plan: explored about {self.mind.explored():.0f} m²; "
                     f"{self._sense_errors} problems in his mind")
        if self.mind.places:
            lines.append("where things are, kept for tomorrow: " + "; ".join(f"{k} {v['where']}" for k, v in self.mind.places.items()))
        return lines

    # ------------------------------------------------------------------ what he does
    def halt(self):
        """He is going to sleep: stop the legs and every move that is still going on (called from another thread)."""
        self._halted = True
        try:
            self.link.stop()
        except Exception:
            pass

    def do(self, act):
        """Carries out one action the brain chose. Returns (what happened, did it go well?)."""
        if self._halted:
            return "you are going to sleep, so you stay still", False
        try:
            return self._do(act)
        except _Halt:
            try:
                self.link.stop()
            except Exception:
                pass
            return "you stopped – you are going to sleep", False

    def _do(self, act):
        with self.lock:
            a = act.strip().strip(".").lower()
            if not a or re.fullmatch(r"(none|nothing|no|wait|stay|stay here|stay still|stay where i am|rest)\b.*", a):
                return None, True
            self.living.acting = True          # his legs and head belong to the brain's move now
            try:
                self.living.before_move()
                return self._do_move(act, a)
            finally:
                self.living.after_move()
                self.living.acting = False

    def _do_move(self, act, a):
        if True:
            num = re.search(r"(-?\d+(?:\.\d+)?)", a)
            n = float(num.group(1)) if num else None
            self.bumped = None
            fallen = self._check_fall()
            if re.search(r"\b(get up|stand up|stand|rise)\b", a):
                result, ok = self._stand_up()
            elif fallen:
                result, ok = "you are lying on your side after a fall – say 'get up' first", False
            elif re.search(r"\bsit\b", a):
                result, ok = self._sit_down()
            elif re.search(EMOTION_WORDS, a):           # a ready-made emotion (whole body + sound)
                result, ok = self._emotion(a)
            elif re.search(PECK_WORDS, a):
                result, ok = self._peck()
            elif re.search(SOUND_WORDS, a):            # (sounds and looking work sitting too - 2026-11-10)
                result, ok = self._sound(a)
            elif self.sitting and re.search(r"\blook around\b|\blook behind\b|\bscan\b", a):
                result, ok = self._look_around()
            elif self.sitting and re.search(r"\blook\b", a) and re.search(r"\b(left|right|up|down|straight|ahead|forward|level)\b", a):
                result, ok = self._look(a)
            elif self.sitting:
                result, ok = "you are sitting – stand up first", False
            elif re.search(r"\bexplore\b", a):
                result, ok = self._explore()
            elif (re.search(r"\b(walk|go|move|waddle|run|head|come|march)\b", a) and re.search(r"\b(to|towards|toward)\b", a)
                  and not re.search(r"\d", a)):
                thing = self._thing_in(a)
                result, ok = self._walk_to(thing) if thing else (self._unknown_thing(a), False)
            elif re.search(r"\b(turn|look|face)\b", a) and re.search(r"\b(to|towards|toward|at)\b", a) and not re.search(r"\d", a) \
                    and self._thing_in(a):
                result, ok = self._turn_to(self._thing_in(a))
            elif re.search(r"\blook around\b|\blook behind\b|\bscan\b", a):
                result, ok = self._look_around()
            elif re.search(r"\blook\b", a) and re.search(r"\b(left|right|up|down|straight|ahead|forward|level)\b", a):
                result, ok = self._look(a)
            elif re.search(r"\b(turn|rotate|spin)\b", a):
                if "around" in a:
                    deg = 180.0
                else:
                    deg = n if n is not None else 90.0
                    deg = -abs(deg) if "right" in a else abs(deg)
                result, ok = self._turn(deg)
            elif re.search(r"\bkick\b", a):
                result, ok = self._kick("left" if "left" in a else "right")
            elif re.search(r"\b(walk|go|move|waddle|step)\b", a):
                back = bool(re.search(r"\b(back|backward|backwards)\b", a))
                if n is not None and re.search(r"\bsteps?\b", a):
                    dist = n * 0.15
                else:
                    dist = (n if n is not None else 50) / (100 if (n is None or n > 5 or "cm" in a) else 1)
                result, ok = self._walk(min(abs(dist), MAX_BACK if back else MAX_WALK), back)
            elif re.search(r"\b(pick up|grab|take|bite|put down|drop|let go)\b", a):
                result, ok = "you can't pick things up or put them down yet", False
            else:
                result, ok = f"your body doesn't know how to '{act}'", False
            self.actions.append((act, result))
            return result, ok

    # --- walking and turning: close the loop on the duck's own odometry, stop when the depth sensor says so
    def _walk(self, dist, back, target=None):
        a0 = self._audit_start()
        out = self._walk_raw(dist, back, target)
        self._audit_end("walk back" if back else "walk", dist, a0)
        return out

    def _walk_raw(self, dist, back, target=None):
        if back and BACK_SPEED <= 0:
            return "your legs can't walk backwards – turn around and walk forward instead", False
        x0, y0, yaw0 = self._odom()
        speed = -BACK_SPEED if back else WALK_SPEED
        timeout = dist / abs(speed) * 1.8 + 4.0
        end, moved, why = time.time() + timeout, 0.0, None
        dirx, diry = math.cos(yaw0), math.sin(yaw0)
        while time.time() < end:
            self.link.move(speed, 0, 0, seconds=0.3)
            self._sleep(0.05)
            x, y, _ = self._odom()
            moved = ((x - x0) * dirx + (y - y0) * diry) * (-1 if back else 1)
            if self._check_fall():
                why = "you fell over"
                break
            if not back:
                d = self._ahead()
                stop_before = STOP_BEFORE
                if target in SMALL_THINGS and self.mind.pos_of(target) is not None:
                    # walking up to a ball: the depth sensor sees the ball itself - that is no reason to stop, only something
                    # nearer than the ball is (it is only 7 cm high, the sensor and his own step counting say where it is)
                    dt, rel_t = self.mind.from_me(self.mind.pos_of(target))
                    if abs(rel_t) < 35 and dt < 1.0:
                        stop_before = min(STOP_BEFORE, max(0.06, dt - 0.12))
                if not math.isnan(d) and d < stop_before:
                    why = f"your depth sensor stopped you: something is only {_metres(d)} in front of you"
                    break
            if moved >= dist - 0.02:
                break
        self.link.stop()
        self._sleep(0.5)
        x, y, _ = self._odom()
        moved = max(0.0, ((x - x0) * dirx + (y - y0) * diry) * (-1 if back else 1))
        self.walked += moved
        self._last_moved, self._last_stop = moved, why
        word = "walked backwards" if back else "walked forward"
        if why:
            self.stopped_in_row = self.stopped_in_row + 1 if moved < 0.1 else 0
            return f"{word} only {_metres(moved)} of the {_metres(dist)} – {why}", False
        self.stopped_in_row = 0
        if moved < dist * 0.6:
            return f"{word}, but only about {_metres(moved)} (your legs didn't take you the whole way)", False
        return f"{word} about {_metres(moved)}", True

    def _turn(self, deg):
        a0 = self._audit_start()
        out = self._turn_raw(deg)
        self._audit_end("turn", math.radians(deg), a0)
        return out

    def _turn_raw(self, deg):
        _, _, yaw = self._odom()
        total, last, target = 0.0, yaw, math.radians(deg)
        end = time.time() + abs(deg) / 25.0 + 4.0
        rate_now = TURN_MIN
        stopped_at = None
        while time.time() < end:
            remaining = target - total
            if abs(remaining) < max(math.radians(3), min(abs(rate_now) * self._turn_lag, 0.3 * abs(target))):
                stopped_at = total   # the legs keep turning ~0.15 s after the command stops (but a small turn must still happen)
                break
            rate = math.copysign(min(TURN_SPEED, max(TURN_MIN, abs(remaining) * 2.0)), remaining)
            rate_now = rate
            self.link.move(TURN_VX, 0, rate, seconds=0.3)
            self._sleep(0.05)
            now = self._odom()[2]
            total += _wrap(now - last)
            last = now
            if self._check_fall():
                break
        self.link.stop()
        self._sleep(0.5)
        total += _wrap(self._odom()[2] - last)
        if stopped_at is not None and abs(rate_now) > 0.5 and abs(target) > math.radians(20):
            lag = (total - stopped_at) / rate_now             # how long the legs kept turning after the stop, in seconds
            self._turn_lag = min(0.8, max(0.05, 0.6 * self._turn_lag + 0.4 * lag))
        shown = math.degrees(total)
        self._last_turned = shown
        if self._check_fall():
            return "you fell over while turning", False
        if abs(shown - deg) > max(10.0, abs(deg) * 0.25):
            return f"turned {'left' if shown > 0 else 'right'} about {abs(shown):.0f}° (you meant {abs(deg):.0f}°)", False
        return f"turned {'left' if shown > 0 else 'right'} about {abs(shown):.0f}°", True

    # --- going to things: his legs follow HIS OWN map (mind.py), a piece at a time, and look where they are in between
    def _thing_in(self, a):
        for name, words in WORDS.items():
            if any(re.search(rf"\b{w}s?\b", a) for w in words):
                return name
        return None

    def _unknown_thing(self, a):
        known = [n for n in self.mind.things if self.mind.pos_of(n) is not None]
        what = re.sub(r"^.*?\b(?:to|towards|toward)\b\s+(?:the\s+|a\s+|an\s+)?", "", a).strip() or "that"
        if known:
            return (f"your body doesn't know a '{what}' yet – you can walk to things in your mental map: "
                    + ", ".join(f"the {n}" for n in known))
        return f"you have not seen anything you can walk to yet ('{what}') – look around first"

    def _go(self, goal_fn, stop, label, name=None):
        """Walks towards a spot (goal_fn() gives where he believes it is, in his own coordinates) until he is `stop` m from
        it: plans a route on his own floor plan, turns, walks a piece (the depth sensor still stops him), looks where he is
        now (landmarks correct his position), plans again. Returns (what he did in words, why it ended):
        arrived / blocked / fell / far / lost."""
        first_turn, walked, why, blocked, looked = None, 0.0, "far", 0, 0
        for _ in range(MAX_SEGMENTS):
            self._sense(force=True)
            goal = goal_fn()
            if goal is None or self.mind.pose() is None:
                why = "lost"
                break
            d, rel = self.mind.from_me(goal)
            if d <= stop + 0.08:
                why = "arrived"
                break
            if name and d < 2.5 and not self._visible_name(name, 1.0) and looked < 3:
                looked += 1                           # close to where he thinks it is, but it isn't in sight: look about
                if self._scan_for(name):
                    continue
                self._sense(force=True)
                goal = goal_fn()
                if goal is None:
                    why = "lost"
                    break
                d, rel = self.mind.from_me(goal)
                if d <= stop + 0.08:
                    why = "arrived"
                    break
            route = self.mind.plan(goal, stop) or [goal]
            wp = next((w for w in route if self.mind.from_me(w)[0] >= 0.25), route[-1])
            d1, rel1 = self.mind.from_me(wp)
            last = wp == route[-1]
            length = min(max(d - stop, 0.1) if last else d1, SEGMENT)
            if abs(rel1) > 6:
                self._last_turned = 0.0
                self._turn(rel1)
                first_turn = self._last_turned if first_turn is None else first_turn
            if self._check_fall():
                why = "fell"
                break
            self._last_moved, self._last_stop = 0.0, None
            self._walk(length, False, name)
            walked += self._last_moved
            if self._check_fall():
                why = "fell"
                break
            if self._last_stop:
                blocked = blocked + 1 if self._last_moved < 0.1 else 0
                if blocked >= 3:
                    why = "blocked"
                    break
            else:
                blocked = 0
        parts = []
        if first_turn is not None and abs(first_turn) >= 8:
            parts.append(f"turned {'left' if first_turn > 0 else 'right'} about {abs(first_turn):.0f}°")
        if walked > 0.1:
            parts.append(f"walked {_metres(walked)}")
        return (" and ".join(parts) or "stayed where you were"), why

    def _scan_for(self, name):
        """Not in sight: he turns his head left and right to look for it; if he sees it, he turns his body towards it."""
        found = None
        for word in ("left", "right"):
            self._look(f"look {word}")
            self._sense(force=True)
            found = self._visible_name(name, 1.0)
            if found:
                break
        self._look("look straight")
        if found:
            self._last_turned = 0.0
            self._turn(found["head"] + found["b"])
            self._sense(force=True)
        return bool(found)

    def _walk_to(self, name):
        label = f"the {name}"
        if self.mind.pos_of(name) is None:
            return self._walk_to_usual(name)
        from_memory = self._visible_name(name, 2.0) is None
        text, why = self._go(lambda: self.mind.pos_of(name), STOP_AT.get(name, 0.45), label, name)
        self._sense(force=True)
        if why == "fell":
            return f"{text} – you fell over on the way", False
        if why == "blocked":
            return f"{text} towards {label} – but something keeps blocking your way: {self.depth_words()}", False
        seen = self._visible_name(name, 1.5)
        if why == "far":
            pos = self.mind.pos_of(name)
            d = self.mind.from_me(pos)[0] if pos else 0
            return f"{text} towards {label} – still {M.about(d)} away (say it again to go on)", True
        if why == "lost":
            return f"{text} – you lost track of where {label} is", False
        where = "where you remember" if from_memory else "to"
        if seen:
            return f"{text} {where} {label} – you can see {label} now, {M.about(seen['d'])} {M.side(seen['b'] + seen['head'])}", True
        pos = self.mind.pos_of(name)
        ahead = self._ahead()
        if pos is not None and name in self.mind.things and abs(self.mind.from_me(pos)[1]) < 35 and not math.isnan(ahead) and ahead < 0.55:
            return (f"{text} {where} {label} – {label} is right in front of you, but too close for your camera to see it with "
                    f"your head level; your depth sensor feels something {_metres(ahead)} straight ahead and your step counting says it is {M.about(self.mind.from_me(pos)[0])} away (look down to see it)"), True
        if self._scan_for(name):
            return f"{text} {where} {label} – looked left and right and spotted {label}", True
        return (f"{text} {where} {label} – but you are NOT at {label}: you can't see it from here (your map was off, or it was moved). "
                "Look around before you do anything else here"), False

    def _walk_to_usual(self, name):
        """He hasn't seen it today, but remembers where it was last time: he goes to that landmark and looks there."""
        place = self.mind.places.get(name)
        if not place:
            return f"you haven't seen a {name} yet – look around to find it", False
        near = place.get("near")
        if not near or self.mind.pos_of(near) is None:
            return (f"the {name} was {place['where']} last time, but you don't know where "
                    f"{'the ' + near if near else 'that'} is from here – look around to find it"), False
        text, why = self._go(lambda: self.mind.pos_of(near), STOP_AT.get(near, 0.45), f"the {near}")
        text = f"went to look {place['where']}, where the {name} was last time: {text}"
        if why in ("fell", "blocked", "lost"):
            return text + f" – but you didn't get there ({why})", False
        self._sense(force=True)
        if self._visible_name(name, 1.5) or self._scan_for(name):
            last, ok = self._walk_to(name)
            return text + f" – and there is the {name}! you {last}", ok
        return text + f" – but you can't see the {name} there now; maybe someone moved it", False

    def _turn_to(self, name):
        pos = self.mind.pos_of(name)
        if pos is None:
            return f"you haven't seen a {name} yet – look around to find it", False
        d, rel = self.mind.from_me(pos)
        if abs(rel) < 4:
            text = f"you are already facing the {name}"
        else:
            self._last_turned = 0.0
            res, ok = self._turn(rel)
            if not ok and self._check_fall():
                return res, False
            text = f"{res} towards the {name}"
        self._sense(force=True)
        seen = self._visible_name(name, 1.5)
        return text + (f" – you can see it now, {M.about(seen['d'])} away" if seen else " – you can't see it from here (your map may be off)"), True

    def _explore(self):
        f = self.mind.frontier()
        if f is None:
            return f"you don't know of any unexplored places nearby ({self.mind.floor_words()})", False
        (gx, gy), d, rel = f
        text, why = self._go(lambda: (gx, gy), 0.3, "the unexplored place")
        if why in ("fell", "blocked", "lost"):
            return f"{text} towards the nearest place you haven't explored – but you didn't get there ({why})", False
        res, ok = self._look_around()
        return f"{text} to the nearest place you hadn't explored, then {res}", ok

    # --- the head
    def _look(self, a):
        for word in ("left", "right", "up", "down"):
            if re.search(rf"\b{word}\b", a):
                already = self._gaze == word and self._home_point == LOOKS[word]
                self.link.look_at(*LOOKS[word])
                self._gaze, self._home_point = word, LOOKS[word]
                self._sleep(1.0)
                # a fact about his neck (2026-11-10): "look up" is one head position - the furthest his head goes that way
                yaw, pitch = (math.degrees(v) for v in self._gaze_angles())
                angle = {"left": yaw, "right": -yaw, "up": pitch, "down": -pitch}[word]
                where = f"about {abs(angle):.0f}°, " if abs(angle) >= 3 else ""
                if already:
                    return (f"your head was already looking {word} – that is as far {word} as your head goes "
                            f"({where.rstrip(', ') or 'its limit'}), so it didn't move and you see the same view as before"), False
                return f"looked {word} ({where}as far {word} as your head goes)", True
        self.link.head(0.0, 0.0, 0.0, 0.0)   # level, and held
        self._gaze, self._home_point = "straight ahead", LEVEL
        self._sleep(1.0)
        return "looked straight ahead", True

    def _look_around(self):
        """Five looks in one picture: left, ahead, right on the top row; up and down (the floor at his feet) below."""
        shots = {}
        for name, target in (("left", LOOKS["left"]), ("ahead", LEVEL), ("right", LOOKS["right"]),
                             ("up", LOOKS["up"]), ("down", LOOKS["down"])):
            self.link.look_at(*target)
            self._sleep(1.0 if name in ("left", "ahead", "right") else 0.9)
            shots[name] = self._frame_now(fresh=True).copy()
        self.link.head(0.0, 0.0, 0.0, 0.0)   # level, and held
        self._gaze, self._home_point = "straight ahead", LEVEL
        self._sleep(0.8)
        h = min(s.shape[0] for s in shots.values())
        w = min(s.shape[1] for s in shots.values())
        tiles = {k: v[:h, :w].copy() for k, v in shots.items()}
        if cv2 is not None:
            for k, label in (("left", "left"), ("ahead", "ahead"), ("right", "right"), ("up", "up"), ("down", "down (your feet)")):
                cv2.putText(tiles[k], label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(tiles[k], label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        blank = np.full((h, w, 3), 40, np.uint8)
        top = np.hstack([tiles["left"], tiles["ahead"], tiles["right"]])
        bottom = np.hstack([tiles["up"], tiles["down"], blank])
        self._pano = np.vstack([top, bottom])
        return ("looked all around (your next picture shows, top row: left, ahead, right; bottom row: up, then down at the "
                "floor right in front of your feet)"), True

    # --- posture and skills
    def _sit_down(self):
        if self.sitting:
            return "you are already sitting", False
        self.link.do("sit_toggle")
        self._sleep(3.0)
        self.sitting = True
        return "sat down", True

    def _stand_up(self):
        if self.link.fallen():
            self.link.enable(True)
            end = time.time() + 8.0
            while time.time() < end and self.link.fallen():
                self._sleep(0.2)
            if self.link.fallen():
                return "you tried to get up but you are still lying down", False
            self.sitting = False
            return "got up again after the fall", True
        if not self.sitting:
            return "you are already standing", False
        self.link.do("sit_toggle")
        self._sleep(3.5)
        self.sitting = False
        return "stood up", True

    def _play_wav(self, path):
        """Plays a sound file on the Mac (robotd's aplay can't) - not on top of his voice, and not when muted."""
        try:
            from anka import voice
            from anka.shared import S as _S
            from settings import SIM_SOUND as _SIM_SOUND
            if not (_SIM_SOUND and _S.sound_on):
                return False
            end = time.time() + 15
            while voice.speaking() and time.time() < end:
                self._sleep(0.2)
        except Exception:
            pass
        if sys.platform == "darwin" and os.path.isfile(path):
            import subprocess
            subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        return False

    def _emotion(self, a):
        """One of Pollen's ready-made emotions, his choice (release 0.18.9 and later)."""
        words = re.findall(r"[a-z_]+", a)
        name = next((w for w in words if w in EMOTIONS), None) or next((EMOTION_ALIASES[w] for w in words if w in EMOTION_ALIASES), None)
        if "shake" in words and "head" in words:
            name = "no"
        have = self.link.skills()
        if not name:
            return "your body didn't understand which feeling to show", False
        if have and name not in have:
            return (f"your body can't act out '{name}' yet (Pollen's emotions need the newer duck software – "
                    f"'Update Pollen (0.18.9).command')"), False
        length = 3.0
        try:
            with open(os.path.join(EMOTION_DIR, name + ".json")) as f:
                length = max(k.get("t", 0.0) for k in json.load(f).get("keyframes", [{}])) or 3.0
        except Exception:
            pass
        self.link.head(0.0, 0.0, 0.0, 0.0)
        self._sleep(0.3)
        res = self.link.do(name)
        if isinstance(res, dict) and (res.get("accepted") is False or res.get("error")):
            return f"your body didn't act it out (it said: {str(res.get('reason') or res.get('error'))[:100]})", False
        self._play_wav(os.path.join(EMOTION_DIR, name + ".wav"))
        self.emotions[name] = self.emotions.get(name, 0) + 1
        self.showing = (f"POLLEN'S ANIMATION: {name}", time.time() + min(8.0, length + 0.4))   # (shown in the room window)
        self._sleep(min(8.0, length + 0.4))
        return f"your whole body acted out '{name}' (head, body and beak, with its sound)", True

    def _peck(self):
        """Bend down until the beak almost touches the floor (Pollen's ground_pick) - a peck. In the simulator the beak
        has no motor, so nothing can be picked up; the real duck's beak can grip."""
        self.link.head(0.0, 0.0, 0.0, 0.0)
        self._sleep(0.3)
        self.showing = ("POLLEN'S ANIMATION: ground pick (peck)", time.time() + 4.3)
        res = self.link.do("ground_pick")
        if isinstance(res, dict) and (res.get("accepted") is False or res.get("error")):
            return f"your body didn't bend down (it said: {str(res.get('reason') or res.get('error'))[:100]})", False
        self.pecks += 1
        self._sleep(4.3)
        return ("you bent down and pecked – your beak came right down to the floor in front of your feet (your beak "
                "can't grab anything in this body), then you stood up again"), True

    def _sound(self, a):
        """One of the duck's own sounds (2026-11-10). The robot plays them itself (robot.sound); in the simulator on the
        Mac robotd can't (it plays with aplay - Linux), so the Mac plays the same voice-bank file. Never on top of his
        voice: if he is talking, the sound waits until he is done."""
        tag = next((SOUND_ALIASES.get(w, w) for w in re.findall(r"[a-z]+", a) if SOUND_ALIASES.get(w, w) in SOUNDS), "chirp")
        try:
            from anka import voice
            end = time.time() + 15
            while voice.speaking() and time.time() < end:
                self._sleep(0.2)
        except Exception:
            pass
        try:
            self.link.call("robot.sound", {"tag": tag})
        except Exception:
            pass
        played = False
        try:
            from anka.shared import S as _S
            from settings import SIM_SOUND as _SIM_SOUND
            muted = not (_SIM_SOUND and _S.sound_on)
        except Exception:
            muted = False
        if sys.platform == "darwin" and not muted:
            import glob, subprocess
            files = glob.glob(os.path.join(SOUND_BANK, tag, "*.wav"))
            if files:
                subprocess.Popen(["afplay", random.choice(files)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                played = True
        self.sounds[tag] = self.sounds.get(tag, 0) + 1
        self._sleep(0.8)
        return f"you made your '{tag}' sound ({SOUNDS[tag]})" + ("" if played or muted or sys.platform != "darwin" else " – but no sound came out"), True

    KICK_X, KICK_Y = 0.09, 0.042     # where Pollen's kick wants the ball: 9 cm ahead, 4.2 cm to the side of the foot

    def _fresh_ball(self, after, wait=1.5):
        """A sighting of the ball made by his eyes AFTER a given moment (not an old one from before a step)."""
        end = time.time() + wait
        while time.time() < end:
            v = self._visible.get("ball")
            if v and v["t"] > after:
                return v
            self._sleep(0.1)
        return None

    def _kick_approach(self, side):
        fy = -self.KICK_Y if side == "right" else self.KICK_Y
        self.link.look_at(*LOOKS["down"])                   # the ball at his feet is below his level view
        self._gaze, self._home_point = "down", LOOKS["down"]
        self._sleep(0.8)
        steps, after = [], time.time()
        for _ in range(6):
            v = self._fresh_ball(after)
            if not v:
                return ("you looked down at your feet to kick, but you can't see the ball there – "
                        + ("after stepping " + ", ".join(steps) if steps else "it isn't right in front of you")), False
            rel = math.radians(v["b"] + v.get("head", 0.0))
            cam = ((self.link.state() or {}).get("frames") or {}).get("camera") or {}
            cx, cy = (cam["pos"][0], cam["pos"][1]) if cam.get("pos") else (0.03, 0.0)   # the camera sits ahead of his feet
            bx, by = cx + v["d"] * math.cos(rel), cy + v["d"] * math.sin(rel)           # the ball, measured from his feet
            if v["d"] > 0.7:
                return f"the ball is about {_metres(v['d'])} away – too far to kick from here (walk up to it first)", False
            if bx - self.KICK_X > 0.10:                       # far: walk most of the way first (walking turns a little)
                self._walk(bx - self.KICK_X - 0.04, False, target="ball")
                steps.append(f"forward {_metres(bx - self.KICK_X - 0.04)}")
            elif abs(by - fy) > 0.025:                        # line the ball up with the kicking foot: a short turn pulse
                ang = math.atan2(by, bx) - math.asin(max(-1.0, min(1.0, fy / max(v["d"], 0.05))))
                ang *= 0.6                                    # (a bit less each time - an overshoot loses the ball from view)
                secs = max(0.08, abs(ang) / TURN_MIN - 0.5 * self._turn_lag)    # (the legs go on a moment after the stop)
                self.link.move(TURN_VX, 0, math.copysign(TURN_MIN, ang), seconds=secs)
                self._sleep(secs + 0.6)
                steps.append(f"{'left' if ang > 0 else 'right'} {abs(math.degrees(ang)):.0f}°")
                if self._fresh_ball(time.time() - 0.05, wait=1.0) is None:    # lost it: turn back the same way and look again
                    self.link.move(TURN_VX, 0, -math.copysign(TURN_MIN, ang), seconds=secs)
                    self._sleep(secs + 0.6)
                    steps.append("back again")
            elif bx - self.KICK_X > 0.025:                    # the last few cm: a short step pulse (a walk overshoots
                dx = bx - self.KICK_X                         # and pushes the ball away with his feet)
                secs = dx / WALK_SPEED + self._pulse_extra    # (the legs need a moment to start stepping at all)
                o0 = self._odom()
                self.link.move(WALK_SPEED, 0, 0, seconds=secs)
                self._sleep(secs + 0.6)
                o1 = self._odom()
                moved = math.hypot(o1[0] - o0[0], o1[1] - o0[1])
                if moved < 0.01:                              # no step at all: a longer pulse next time
                    self._pulse_extra = min(0.7, self._pulse_extra + 0.12)
                elif moved > dx * 1.6:                        # too far: a shorter one
                    self._pulse_extra = max(0.0, self._pulse_extra - 0.08)
                steps.append(f"a small step forward ({_metres(moved)})")
            else:
                self.kick_lineup = (round(bx, 3), round(by, 3))
                return ("your body stepped up to the ball (" + ", ".join(steps) + ")" if steps
                        else "the ball was right in front of your foot"), True
            self._sleep(0.3)
            after = time.time()
        self.kick_lineup = (round(bx, 3), round(by, 3))
        return "your body tried to step up to the ball (" + ", ".join(steps) + ") but couldn't line it up exactly", True

    def _kick(self, side):
        have = self.link.skills()
        skill = next((k for k in have if "kick" in k.lower() and side in k.lower()), None)
        if skill is None:
            return (f"your body has no kick with the {side} foot (the moves your body has: "
                    f"{', '.join(have) if have else 'none'})"), False
        # the approach (2026-11-10): Pollen's kick is BLIND and trained with the ball 9 cm in front of and 4 cm to the
        # side of the kicking foot - from 13-20 cm away the foot swings at nothing. So, like a footballer, his body steps
        # up to the ball first (a motor skill of the kick, from what his eyes see when he looks down at his feet).
        approach, ok = self._kick_approach(side)
        if not ok:
            return approach, False
        before = ""
        pos = self.mind.pos_of("ball")
        if pos is not None and self.mind.pose() is not None:
            bd, brel = self.mind.from_me(pos)
            before = f" (by your map the ball was {M.about(bd)} {M.side(brel)} when you kicked)"
        seen0 = self._visible_name("ball", max_age=1.5)        # what his EYES said just before the kick
        # the head must be straight for the kick (kicktest 2026-11-10: with the head looking down the kick policy
        # moves the legs only 6-10° instead of 23-33° - it was trained with the head at home)
        self.link.head(0.0, 0.0, 0.0, 0.0)
        self._sleep(0.7)
        res = self.link.do(skill)
        self.kicks.append({"skill": skill, "answer": res, "ball_at": getattr(self, "kick_lineup", None)})
        if isinstance(res, dict) and (res.get("accepted") is False or res.get("error")):
            why = res.get("reason") or res.get("error") or "no reason given"
            return f"your body didn't do the kick (it said: {str(why)[:120]})", False
        self._sleep(3.0)
        self.link.look_at(*LOOKS["down"])        # then look down again to see where the ball went
        self._gaze, self._home_point = "down", LOOKS["down"]
        self._sleep(0.8)
        self.mind.note_kicked("ball")         # he still remembers where it was - and that he kicked it after that
        # what his eyes see now (2026-11-10: before this he was always told "look to see where the ball went", also
        # when the ball never moved - he then believed he had kicked it away while it lay in front of him)
        seen1 = self._visible_name("ball", max_age=0.6)
        if seen1 is not None:
            self._ball_at = None
            where = f"about {_metres(seen1['d'])} {_side(seen1['b'] + seen1.get('head', 0.0))}"
            if seen0 is not None:
                a0 = math.radians(seen0["b"] + seen0.get("head", 0.0)); a1 = math.radians(seen1["b"] + seen1.get("head", 0.0))
                moved = math.hypot(seen1["d"] * math.cos(a1) - seen0["d"] * math.cos(a0),
                                   seen1["d"] * math.sin(a1) - seen0["d"] * math.sin(a0))
                self.kicks[-1]["moved_m"] = round(moved, 2)
                if moved < 0.08:
                    return (f"{approach}, then did the kick with your {side} foot{before}, but your eyes see the ball still where it was: "
                            f"{where} – it didn't move"), False
                return f"{approach}, then kicked with your {side} foot{before} – your eyes see the ball now {where} (it moved about {_metres(moved)})", True
            return f"{approach}, then kicked with your {side} foot{before} – your eyes see the ball now {where}", True
        self._ball_at = False                 # nobody knows where it rolled to - he has to look
        return f"{approach}, then kicked with your {side} foot{before} – the ball is not in your camera picture now: look to see where it went", True
