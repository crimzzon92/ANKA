"""The living body: what Anka's body does BY ITSELF between his thoughts (2026-11-09, livelier 2026-11-10).

A thought takes 5-40 s (the cloud brain). Before this file his body stood frozen all that time. Now a small loop runs
on the duck itself, 10 times a second, and does what a body does without deciding anything:

  * his eyes follow something that MOVES in front of him (a rolling ball, a sock someone moves)
  * a voice makes him lift his head towards where people's faces are, beak a little open
  * when nothing happens he looks about like an animal: a glance somewhere (mostly left or right, a little up or
    down, sometimes back to the middle, sometimes at something he remembers nearby or at the floor), held for a
    second or few while the gaze keeps re-fixating a few degrees around the spot; the neck leans a little with each
    look, the beak opens a touch, now and then the curious head tilt
  * he never quite freezes: a slow drift of a degree or two on every head axis, and his body BREATHES (a few
    millimetres up and down) and now and then shifts its weight / leans a little (robot.pose)
  * he calms down: right after his own move he is lively; the longer he stands, the slower and smaller it gets

The numbers come from Pollen's own idle behaviour for this robot (robotd/src/idle_head.rs, [idle] in robotd.toml) -
robotd's version can't run while Anka's head is held, so this is the same idea, centred on where the BRAIN last
pointed his head (not on the straight head) and with the Anka parts added (moving things, voices, remembered things).

What it NEVER does: ask the brain, pick a goal, walk, turn his body, or decide anything. His legs belong to his brain.
When the brain chooses a move, head, beak and body pose go back to normal before the move starts.

It also MEASURES: how much of the time between his moves he stands completely still (the "still meter"; the time
he is asleep is not counted). --no-living turns the movements off (the meter still runs).
"""
import math
import os
import random
import re
import threading
import time
from collections import Counter, deque

TICK = 0.1                 # s
STILL_GAZE = math.radians(1.2)   # less than this change of gaze in half a second ...
STILL_MOVE = 0.005               # ... and less than 5 mm / 0.5° of walking  = standing completely still
EYE_Z = -0.085             # a thing on the floor, seen from the trunk (the floor is 0.12 m below; a ball's middle is 3.5 cm up)
FACE_POINT = (1.2, 0.0, 0.85)    # where a standing person's face roughly is, from his trunk (he can't tell where a voice is from)
MOVING = 0.25              # m: something counts as moving if it went this far in about a second (seen twice)
SMALL = ("ball", "sock")   # things that can roll or be moved

# looking about (radians / seconds) - Pollen's defaults for this robot
GLANCE_YAW, GLANCE_PITCH, GLANCE_NECK = 0.6, 0.2, 0.15
GLANCE_MOUTH = 0.4         # each look opens the beak between half and all of this (below ~0.15 it doesn't show)
TILT_ROLL, TILT_CHANCE = 0.3, 0.25
CENTRE_CHANCE = 0.25       # a look goes back to the middle (where the brain pointed his head)
REMEMBERED_CHANCE = 0.20   # ... at something on his mental map nearby
FLOOR_CHANCE = 0.12        # ... at the floor in front of his feet
GLANCE_MIN = 0.2           # a new look is at least this far from the last one (yaw + pitch)
HOLD = (1.0, 3.0)          # how long a look is held, at full energy (dozing stretches it)
MOVE_MIN_S, MOVE_S_PER_RAD = 0.3, 1.0
NUDGE_YAW, NUDGE_PITCH, NUDGE_EVERY = 0.08, 0.05, (0.4, 1.2)
DRIFT, DRIFT_HZ = 0.04, (0.15, 0.13, 0.11, 0.17)
# the body (robot.pose) - kept small: the standing network keeps the balance, the head carries the sensors
BREATH_M, BREATH_PERIOD = 0.004, 3.6      # (Pollen: 1 mm; the trained height range is -25 / +10 mm)
SHIFT_CHANCE = 0.4         # a new look sometimes comes with a small weight shift / lean
SHIFT_ROLL, SHIFT_PITCH, SHIFT_CROUCH = 0.12, 0.12, 0.015   # about half of the trained range (±12°, -25 mm)
POSE_EASE = 0.08           # per tick: the body glides, it never jumps
# calming down after his own move
DOZE_TAU, DOZE_FLOOR = 90.0, 0.5
# his feet (2026-11-10, Alex: "he still stops and is kind of still"): small reflexes, never walking anywhere
LEGS = True
TURN_TO_GAZE_AFTER = 2.5   # s: looking this far to the side for this long ...
TURN_TO_GAZE_MIN = 0.45    # rad (~26°) ... and his body turns a little that way (like an animal), at most:
TURN_TO_GAZE_MAX = 0.35    # rad (~20°) per turn
TURN_REST = (6.0, 14.0)    # s between two such turns
LEG_TURN_RATE = 1.0        # rad/s (the walking policy ignores slower turns - simbody TURN_MIN 0.9)
SHUFFLE_CHANCE = 0.12      # a new look sometimes comes with a little shuffle of the feet (turn a bit, turn back)
SHUFFLE = 0.14             # rad each way
FOLLOW_EASE = 0.35         # following a moving thing: each tick this share of the remaining way
AXES = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")

# HOW HE FEELS, IN HIS BODY (2026-11-10, step 3 - the BODY field): the brain writes in its own words how its feelings
# show in its body ("crouched low, head down, trembling, looking away from Alex"); these words set the STYLE of the
# living body until the next BODY (it fades back to normal after STYLE_LASTS s). The body only interprets the words -
# it never decides what he feels or does.
STYLE_LASTS = 90.0
_H = r"\bhead\W+(?:\w+\W+){0,3}?"          # "head ... <word>" (up to three words between)
STYLE_WORDS = [   # (words, what they change) - his own words, read loosely (2026-11-10: bigger, and more phrasings)
    (r"\b(crouch\w*|low|cower\w*|hunch\w*|shrink\w*|huddl\w*|curl\w*|small)\b", {"z": -0.025}),
    (r"\b(tall|upright|proud\w*|puff\w*)\b|\bstretch\w*", {"z": 0.010}),
    (_H + r"(down|low|tucked|bowed|hanging|drooping|lowered)\b|\b(tuck\w*|bow\w*|hang\w*|droop\w*|lower\w*)\W+(?:\w+\W+)?head\b",
     {"head": -35}),
    (_H + r"(up|high|raised|back|lifted)\b|\b(lift\w*|rais\w*)\W+(?:\w+\W+)?head\b|\bneck\W+(?:\w+\W+)?(up|stretched)|"
     r"\bstretch\w*\W+(?:\w+\W+)?neck|\bchin up\b|\blook\w* up\b", {"head": 25}),
    (r"\blean\w* (forward|in|toward\w*)\b", {"head": -12, "energy": 1.2}),
    (r"\b(calm\w*|slow\w*|relax\w*|sleepy|tired|drowsy|peaceful|gentl\w*|soft\w*)\b", {"energy": 0.5}),
    (r"\b(livel\w*|excit\w*|eager\w*|happ\w*|playful\w*|energetic|fast|quick\w*)\b", {"energy": 1.5}),
    (r"\b(still|freez\w*|frozen|motionless|stiff)\b", {"energy": 0.12, "shuffle": 0.0}),
    (r"\b(nervous\w*|jitter\w*|jumpy|restless|fidget\w*|anxious\w*|dart\w*|scared|afraid|fright\w*)\b",
     {"energy": 1.3, "nervous": True}),
    (r"\b(trembl\w*|shak\w*|shiver\w*|quiver\w*|wobbl\w*)\b", {"tremble": True}),
    (r"\b(bob\w*|hop\w*|bounc\w*|danc\w*|nod\w*)\b", {"bob": True}),
    (r"\b(tail|wag\w*|wiggl\w*|waggl\w*|sway\w*|shimm\w*)\b", {"wiggle": True}),
    (r"\b(tilt\w*|curious\w*|puzzled|cock\w* (my|his) head)\b", {"tilt": 0.8}),
    (r"\b(look\w* away|avoid\w*|turn\w* away|hid\w*|not looking|tuck\w*)\b", {"gaze": "away"}),
    (r"\b(look\w* (up )?at|watch\w*|eyes on|toward\w*|facing|staring at)\b", {"gaze": "toward"}),
]


def read_style(text):
    st = {}
    t = (text or "").lower()
    for pattern, change in STYLE_WORDS:
        if re.search(pattern, t):
            st.update(change)
    if st.get("z", 0) < 0 and "head" not in st:      # crouched low: the head comes down with the body
        st["head"] = -15
    return st


def _rot(p, yaw_deg, up_deg=0.0):
    """Turn a look-at point around his trunk: yaw_deg to the left, up_deg up."""
    x, y, z = p
    d = math.hypot(x, y)
    a = math.atan2(y, x) + math.radians(yaw_deg)
    e = math.atan2(z, d) + math.radians(up_deg)
    r = math.hypot(d, z)
    return (r * math.cos(e) * math.cos(a), r * math.cos(e) * math.sin(a), r * math.sin(e))


def _smooth(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def _zero():
    return {k: 0.0 for k in AXES}


class Living:
    def __init__(self, sb, on=True):
        self.sb, self.link, self.on = sb, sb.link, on
        self.acting = False                 # True while the brain's move is going on
        self.home = None                    # the head angles the brain's last move left him with (None = head free)
        self.moved_head = False             # did the loop move his head since the brain's last move?
        # the look: offsets from the brain's head position (radians) + the beak
        self.cur, self.anchor, self.frm = _zero(), _zero(), _zero()
        self.mouth_cur, self.mouth_to, self.mouth_from = 0.0, 0.0, 0.0
        self.look_start, self.travel, self.look_until = 0.0, 0.3, 0.0
        self.mode = "idle"                  # idle / follow / voice
        self.nudge, self.nudge_cur, self.next_nudge = _zero(), _zero(), 0.0
        self.last_move_end = time.time()
        self.next_idle = time.time() + random.uniform(1.0, 3.0)
        self.phases = [random.uniform(0, math.tau) for _ in range(5)]
        # the body pose
        self.pose_cur, self.pose_to, self.pose_on = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], False
        self.heard_t, self.voice_done = 0.0, 0.0
        self.legs, self.leg_until, self.side_since, self.look_name = [], 0.0, None, None
        self.turned, self.turns, self.turned_to, self.next_turn = 0.0, 0, [], 0.0
        self.track = {n: deque(maxlen=20) for n in SMALL}
        self.following, self.follow_seen = None, 0.0
        self._solved = {}
        # the still meter
        self.win = deque(maxlen=int(0.5 / TICK) + 1)
        self.t = {"between": 0.0, "between_still": 0.0, "acting": 0.0, "acting_still": 0.0}
        self.still_run, self.longest_still, self.stretches = 0.0, 0.0, []
        self.glances = Counter()
        self.style, self.style_t, self.style_text, self.styles = {}, 0.0, "", []
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    # ---------------------------------------------------------------- called by SimBody
    def before_move(self):
        """The brain chose a move: head, beak and body back to normal (the depth sensor and the camera must look where
        his legs are going, and walking needs the body pose switched off)."""
        self.mode, self.following = "idle", None
        if self.legs or time.time() < self.leg_until:
            self.legs, self.leg_until = [], 0.0
            try:
                self.link.stop()
            except Exception:
                pass
            time.sleep(0.3)
        if not (self.moved_head or self.pose_on):
            return
        try:
            if self.home is not None:
                self.link.head(**self.home)
            else:
                self.link.head_free()
            self.link.mouth(0.0)
            if self.pose_on:
                self.link.pose(0.0, 0.0, 0.0, True)      # glide back to the normal stance first ...
        except Exception:
            pass
        time.sleep(0.4)
        self._pose_off()                                 # ... then hand the body back to walking
        self.moved_head = False
        self.cur, self.anchor, self.nudge, self.nudge_cur = _zero(), _zero(), _zero(), _zero()
        self.mouth_cur = self.mouth_to = 0.0

    def after_move(self):
        h = getattr(self.link, "_head", None)
        self.home = dict(h) if isinstance(h, dict) else None
        self.cur, self.anchor, self.frm = _zero(), _zero(), _zero()
        self.last_move_end = time.time()
        self.look_until = 0.0
        self.next_idle = time.time() + random.uniform(1.0, 3.0)

    def heard(self):
        self.heard_t = time.time()

    def set_style(self, text):
        """The brain's BODY line: how his feelings show in his body right now (his own words)."""
        text = (text or "").strip()
        if not text or re.fullmatch(r"(?i)\W*(none|no|nothing|normal|n/?a)\W*", text):
            return
        self.style, self.style_t, self.style_text = read_style(text), time.time(), text[:100]
        self.styles.append(text[:60])
        self.look_until = 0.0                    # show it at once
        self.next_nudge = 0.0

    def _style(self, key, default):
        """A style value, fading back to normal (default) over the last third of STYLE_LASTS."""
        if key not in self.style:
            return default
        age = time.time() - self.style_t
        if age > STYLE_LASTS:
            return default
        v = self.style[key]
        if (isinstance(v, (int, float)) and not isinstance(v, bool) and isinstance(default, (int, float))
                and age > STYLE_LASTS * 2 / 3):
            k = (age - STYLE_LASTS * 2 / 3) / (STYLE_LASTS / 3)
            return v + (default - v) * k
        return v

    def take_news(self):
        """What his body did by itself that he should know (his feet turned him) - told once, in his next body text."""
        if not self.turns:
            return []
        net, n, to = math.degrees(self.turned), self.turns, [t for t in dict.fromkeys(self.turned_to) if t]
        self.turned, self.turns, self.turned_to = 0.0, 0, []
        words = (f"your body turned by itself {'once' if n == 1 else f'{n} times'} towards where you were looking"
                 + (f" ({', '.join(to[-3:])})" if to else ""))
        words += (f" – in all about {abs(net):.0f}° to the {'left' if net > 0 else 'right'}" if abs(net) >= 5
                  else " – in all you face about the same way as before")
        return [words]

    def _pose_off(self):
        if self.pose_on:
            try:
                self.link.pose(0.0, 0.0, 0.0, False)
            except Exception:
                pass
        self.pose_on, self.pose_cur, self.pose_to = False, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    # ---------------------------------------------------------------- the loop
    def _loop(self):
        last = time.time()
        while not self._stop.is_set():
            time.sleep(TICK)
            now = time.time()
            dt, last = min(0.5, now - last), now
            try:
                self._measure(dt)
            except Exception:
                pass
            if not self.on or self.acting:
                continue
            try:
                down = self.sb._halted or self.link.fallen()       # (sitting: the head still lives, body and feet rest)
            except Exception:
                down = True
            if down:
                self._pose_off()
                continue
            if not self.sb.lock.acquire(blocking=False):     # a move of his own is starting - keep out of its way
                continue
            try:
                if not self.acting:
                    self._step(now)
            except Exception as e:
                key = "(problem: " + type(e).__name__ + ")"
                if not self.glances[key]:          # 2026-11-12: say once WHERE it went wrong, so it can be fixed
                    import traceback
                    where = traceback.extract_tb(e.__traceback__)[-1]
                    print(f"[living body] problem: {type(e).__name__}: {e} – in {os.path.basename(where.filename)} "
                          f"line {where.lineno} ({where.line})", flush=True)
                self.glances[key] += 1
            finally:
                self.sb.lock.release()

    def _energy(self, now):
        """1 right after his own move, calming towards DOZE_FLOOR the longer he stands - times his BODY style."""
        base = DOZE_FLOOR + (1 - DOZE_FLOOR) * math.exp(-(now - self.last_move_end) / DOZE_TAU)
        e = self._style("energy", base)     # (fades back to his own energy - with None it crashed: TypeError, 2026-11-12)
        return max(0.08, min(1.6, e))

    def _base(self):
        return dict(self.home) if self.home else self._solve(self._home_point())

    def _home_point(self):
        return getattr(self.sb, "_home_point", (1.5, 0.0, 0.13))

    def _offsets_to(self, point):
        """Head offsets (from the brain's head position) that point his eyes at a point in the trunk frame."""
        a, b = self._solve(point), self._base()
        return {k: a.get(k, 0.0) - b.get(k, 0.0) for k in AXES}

    def _set_look(self, now, offs, hold, mouth=None):
        self.frm, self.anchor = dict(self.cur), offs
        dist = abs(offs["head_yaw"] - self.cur["head_yaw"]) + abs(offs["head_pitch"] - self.cur["head_pitch"])
        self.travel = MOVE_MIN_S + MOVE_S_PER_RAD * dist
        self.look_start, self.look_until = now, now + self.travel + hold
        self.mouth_from = self.mouth_cur
        self.mouth_to = self.mouth_cur if mouth is None else mouth

    def _step(self, now):
        e = self._energy(now)
        # 1. something moving in front of him: his eyes follow it
        mover = self._mover(now)
        if mover:
            name, point = mover
            if self.following != name:
                self.glances[f"followed the moving {name}"] += 1
            self.following, self.follow_seen, self.mode = name, now, "follow"
            self.anchor = self._offsets_to(point)
            self.look_until = now + 0.6
        elif self.following and now - self.follow_seen > 1.5:
            self.following, self.mode = None, "idle"
            self.look_until = 0.0
        # 2. a voice: lift the head towards where faces are, beak a little open
        if not self.following and self.heard_t > self.voice_done and now - self.heard_t < 1.5:
            self.voice_done, self.mode = now, "voice"
            self.glances["looked up at a voice"] += 1
            self._set_look(now, self._offsets_to(FACE_POINT), random.uniform(2.5, 4.0), mouth=0.25)
        # 3. looking about
        if not self.following and now >= self.look_until:
            self.mode = "idle"
            self._new_look(now, e)
        # the head: travel to the look (ease-in-ease-out), or chase a moving thing
        if self.mode == "follow":
            for k in AXES:
                self.cur[k] += (self.anchor[k] - self.cur[k]) * FOLLOW_EASE
        else:
            u = _smooth((now - self.look_start) / max(self.travel, 1e-3))
            for k in AXES:
                self.cur[k] = self.frm[k] + (self.anchor[k] - self.frm[k]) * u
            self.mouth_cur = self.mouth_from + (self.mouth_to - self.mouth_from) * u
        # the small re-fixations while a look is held (not while following)
        if now >= self.next_nudge:
            self.next_nudge = now + random.uniform(*NUDGE_EVERY) / max(e, 0.3)
            self.nudge = _zero()
            if self.mode == "idle" and now > self.look_start + self.travel:
                k = 1.6 if self._style("nervous", False) else 1.0      # BODY nervous: quick, darting re-fixations
                self.nudge["head_yaw"] = random.uniform(-NUDGE_YAW, NUDGE_YAW) * e * k
                self.nudge["head_pitch"] = random.uniform(-NUDGE_PITCH, NUDGE_PITCH) * e * k
                if self._style("nervous", False):
                    self.next_nudge = now + random.uniform(0.2, 0.5)
        for k in AXES:
            self.nudge_cur[k] += (self.nudge[k] - self.nudge_cur[k]) * 0.3
        self._feet(now, e)
        # the never-quite-still drift
        drift = {k: DRIFT * e * math.sin(math.tau * DRIFT_HZ[i] * now + self.phases[i]) for i, k in enumerate(AXES)}
        base = self._base()
        out = {k: base.get(k, 0.0) + self.cur[k] + self.nudge_cur[k] + drift[k] for k in AXES}
        if self._style("bob", False):                 # BODY: bobbing - the head nods (joint angles: symmetric, sign-free)
            out["head_pitch"] += 0.14 * math.sin(math.tau * 1.6 * now)
            out["neck_pitch"] += 0.06 * math.sin(math.tau * 1.6 * now)
        if self._style("tremble", False):             # BODY: trembling - a fast small shiver of the head
            out["head_roll"] += 0.05 * math.sin(math.tau * 8.0 * now) + random.uniform(-0.015, 0.015)
            out["head_yaw"] += 0.02 * math.sin(math.tau * 6.5 * now)
        if self._style("wiggle", False):              # BODY: wiggling / wagging - a side-to-side sway
            out["head_roll"] += 0.12 * math.sin(math.tau * 2.2 * now)
        tilt_head = self._style("head", 0.0)          # BODY: head held down / up
        if abs(tilt_head) > 1:
            down = self._offsets_to(_rot(self._home_point(), 0, tilt_head))
            out = {k: out[k] + down[k] for k in AXES}
        self.link.head(**out)
        self.moved_head = True
        try:
            self.link.mouth(self.mouth_cur)
        except Exception:
            pass
        self._body(now, e)

    def _new_look(self, now, e):
        gaze = self._style("gaze", None)
        if gaze == "toward" and random.random() < 0.6:           # BODY: eyes on the person (where a face would be)
            self.glances["looked at the person (body style)"] += 1
            self._set_look(now, self._offsets_to(FACE_POINT), random.uniform(1.5, 3.5) / max(e, 0.3), mouth=0.0)
            return
        if gaze == "away" and random.random() < 0.7:             # BODY: looking away, a bit down
            offs = _zero()
            offs["head_yaw"] = random.choice((-1, 1)) * random.uniform(0.6, 1.0) * GLANCE_YAW
            offs = {k: offs[k] + v for k, v in self._offsets_to(_rot(self._home_point(), 0, -12)).items()}
            self.glances["looked away (body style)"] += 1
            self._set_look(now, offs, random.uniform(1.5, 3.0), mouth=0.0)
            return
        r = random.random()
        hold = random.uniform(*HOLD) / max(e, 0.3)
        mouth = random.uniform(0.5, 1.0) * GLANCE_MOUTH * e if random.random() < 0.6 else 0.0
        near = self._remembered_near() if r < CENTRE_CHANCE + REMEMBERED_CHANCE else None
        if r < CENTRE_CHANCE:
            self.glances["looked back to the middle"] += 1
            offs = _zero()
        elif near:
            name, p = near
            self.glances[f"glanced at the {name}"] += 1
            offs = self._offsets_to(p)
        elif r < CENTRE_CHANCE + REMEMBERED_CHANCE + FLOOR_CHANCE:
            self.glances["peeked at the floor"] += 1
            offs = self._offsets_to((0.45, random.uniform(-0.15, 0.15), -0.10))
            hold = min(hold, random.uniform(0.8, 1.6))
        else:
            for _ in range(6):     # somewhere new - never right where he already looks
                offs = _zero()
                offs["head_yaw"] = random.choice((-1, 1)) * random.uniform(0.25, 1.0) * GLANCE_YAW * e
                offs["head_pitch"] = random.uniform(-GLANCE_PITCH, GLANCE_PITCH) * e
                if (abs(offs["head_yaw"] - self.cur["head_yaw"]) + abs(offs["head_pitch"] - self.cur["head_pitch"])) >= GLANCE_MIN:
                    break
            self.glances["glanced about"] += 1
        offs["neck_pitch"] += random.uniform(-GLANCE_NECK, GLANCE_NECK) * e      # the whole head shifts, not only the face
        if random.random() < self._style("tilt", TILT_CHANCE):
            offs["head_roll"] += random.choice((-1, 1)) * random.uniform(0.5, 1.0) * TILT_ROLL
            self.glances["tilted his head"] += 1
        if mouth >= 0.15:
            self.glances["opened his beak a little"] += 1
        self._set_look(now, offs, hold, mouth=mouth)
        self.look_name = near[0] if near else None
        if LEGS and random.random() < SHUFFLE_CHANCE * e * self._style("shuffle", 1.0) and not self.legs and now >= self.leg_until:
            a = random.choice((-1, 1)) * SHUFFLE
            self.legs = [a, -a]
            self.glances["shuffled his feet"] += 1
        if random.random() < SHIFT_CHANCE:
            self.pose_to = [random.uniform(-SHIFT_CROUCH, 0.0), random.uniform(-SHIFT_ROLL, SHIFT_ROLL) * e,
                            random.uniform(-SHIFT_PITCH, SHIFT_PITCH) * e]
            self.glances["shifted his weight"] += 1

    def _feet(self, now, e):
        """Small leg reflexes: turn the body a little towards where he has been looking to the side; a shuffle now
        and then. The step counting sees every bit of it, so his map stays right."""
        if not LEGS or now < self.leg_until or self.sb.sitting:
            return
        if self.legs:                                   # the next part of a shuffle
            a = self.legs.pop(0)
            self._leg_turn(now, a)
            return
        side = self._base().get("head_yaw", 0.0) + self.cur["head_yaw"]
        if self.mode in ("idle", "follow") and abs(side) >= TURN_TO_GAZE_MIN and now >= self.next_turn:
            self.side_since = self.side_since or now
            if now - self.side_since >= TURN_TO_GAZE_AFTER:
                a = math.copysign(min(TURN_TO_GAZE_MAX, abs(side) * 0.7), side)
                self._leg_turn(now, a)
                for d in (self.cur, self.anchor, self.frm):     # his eyes stay on the spot while the body turns under them
                    d["head_yaw"] -= a
                what = f"the {self.following}" if self.following else (f"the {self.look_name}" if self.look_name else "")
                self.turned, self.turns = self.turned + a, self.turns + 1
                self.turned_to.append(what)
                self.next_turn = now + random.uniform(*TURN_REST)
                self.glances["turned his body towards where he looked"] += 1
                self.side_since = None
        else:
            self.side_since = None

    def _leg_turn(self, now, a):
        self._pose_off()
        secs = abs(a) / LEG_TURN_RATE
        self.link.move(0.0, 0.0, math.copysign(LEG_TURN_RATE, a), seconds=secs)
        self.leg_until = now + secs + 0.35          # (the legs keep going a moment after the command stops)

    def _body(self, now, e):
        """Breathing + the slow weight shifts, through robot.pose (only while he stands still - walking turns it off)."""
        if self.legs or now < self.leg_until or self.sb.sitting:   # the body pose zeroes walking (and sitting is its own pose) - off while the feet move
            if self.pose_on:
                self._pose_off()
            return
        for i in range(3):
            self.pose_cur[i] += (self.pose_to[i] - self.pose_cur[i]) * POSE_EASE
        breath = BREATH_M * e * math.sin(math.tau * now / BREATH_PERIOD + self.phases[4])
        if self._style("bob", False):                            # BODY: a happy bob
            breath = 0.009 * math.sin(math.tau * now / 0.6)
        z = max(-0.025, min(0.010, self.pose_cur[0] + self._style("z", 0.0) + breath))
        roll = self.pose_cur[1]
        if self._style("tremble", False):                        # BODY: trembling
            roll += 0.045 * math.sin(math.tau * 7.0 * now) + random.uniform(-0.012, 0.012)
        if self._style("wiggle", False):                         # BODY: a side-to-side wiggle of the whole body
            roll += 0.08 * math.sin(math.tau * 2.2 * now)
        roll = max(-0.2, min(0.2, roll))
        try:
            self.link.pose(z, roll, self.pose_cur[2], True)
            self.pose_on = True
        except Exception:
            pass

    def _remembered_near(self):
        m = self.sb.mind
        try:
            if m.pose() is None:
                return None
            options = []
            for name in list(m.things):
                p = m.pos_of(name)
                if p is None:
                    continue
                d, rel = m.from_me(p)
                if 0.3 < d < 4.0 and abs(rel) < 80:
                    z = EYE_Z if name in SMALL else 0.15
                    options.append((name, (d * math.cos(math.radians(rel)), d * math.sin(math.radians(rel)), z)))
            return random.choice(options) if options else None
        except Exception:
            return None

    def _mover(self, now):
        """Something small that moved about a quarter of a metre in a second - seen with his eyes, put into his own
        step-counting coordinates so his own head turning doesn't look like the thing moving."""
        o = self.link.odom()
        if o is None:
            return None
        for name in SMALL:
            v = self.sb._visible_name(name, max_age=0.35)
            if not v:
                continue
            rel = math.radians(v["b"] + v.get("head", 0.0))
            d = v["d"]
            wx, wy = o[0] + d * math.cos(o[2] + rel), o[1] + d * math.sin(o[2] + rel)
            tr = self.track[name]
            if not tr or tr[-1][0] < v["t"]:
                tr.append((v["t"], wx, wy))
            old = [p for p in tr if 0.7 < now - p[0] < 1.6]
            new = [p for p in tr if now - p[0] < 0.4]
            if len(old) >= 2 and len(new) >= 2:
                ox = sorted(p[1] for p in old)[len(old) // 2]; oy = sorted(p[2] for p in old)[len(old) // 2]
                nx = sorted(p[1] for p in new)[len(new) // 2]; ny = sorted(p[2] for p in new)[len(new) // 2]
                if self.following == name or math.hypot(nx - ox, ny - oy) > MOVING * (1 + 0.15 * d):
                    return name, (d * math.cos(rel), d * math.sin(rel), EYE_Z)
        return None


    def _solve(self, p):
        key = (round(p[0], 1), round(p[1], 1), round(p[2], 1))
        if key in self._solved:
            return dict(self._solved[key])
        res = None
        try:
            res = self.link.call("robot.look", {"x": key[0], "y": key[1], "z": key[2], "neck_pitch": 0.0})
        except Exception:
            pass
        if isinstance(res, dict) and isinstance(res.get("head"), dict):
            h = res["head"]
            ang = {k: float(h.get(k, 0.0)) for k in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")}
        else:      # (no answer from the robot: a simple guess - used by the fake test duck)
            ang = {"neck_pitch": 0.0, "head_pitch": math.atan2(key[2] - 0.08, math.hypot(key[0], key[1])),
                   "head_yaw": math.atan2(key[1], key[0]), "head_roll": 0.0}
        if len(self._solved) > 400:
            self._solved.clear()
        self._solved[key] = ang
        return dict(ang)

    # ---------------------------------------------------------------- the still meter
    def _measure(self, dt):
        if getattr(self.sb, "_halted", False):      # asleep: not part of the day
            self.win.clear()
            return
        o = self.link.odom()
        yaw, pitch = self.sb._gaze_angles()
        self.win.append((o, yaw, pitch))
        if len(self.win) < 3 or o is None:
            return
        os_ = [w[0] for w in self.win if w[0] is not None]
        span_xy = max(math.hypot(a[0] - os_[0][0], a[1] - os_[0][1]) for a in os_)
        span_turn = max(abs(math.remainder(a[2] - os_[0][2], math.tau)) for a in os_)
        span_gaze = max(max(w[1] for w in self.win) - min(w[1] for w in self.win),
                        max(w[2] for w in self.win) - min(w[2] for w in self.win))
        moving_legs = getattr(self.link, "_move", None) is not None
        still = (span_xy < STILL_MOVE and span_turn < math.radians(0.5) and span_gaze < STILL_GAZE and not moving_legs)
        part = "acting" if self.acting else "between"
        self.t[part] += dt
        if still:
            self.t[part + "_still"] += dt
        if still and not self.acting:
            self.still_run += dt
        else:
            if self.still_run >= 3.0:
                self.stretches.append(self.still_run)
            self.longest_still = max(self.longest_still, self.still_run)
            self.still_run = 0.0

    def numbers(self):
        b, bs = self.t["between"], self.t["between_still"]
        return {"body_styles": len(self.styles),"still_between": round(bs / b, 2) if b else None,
                "longest_still_s": round(max(self.longest_still, self.still_run), 1),
                "body_turns": self.glances.get("turned his body towards where he looked", 0),
                "self_moves": sum(self.glances.values())}

    def report(self):
        b, bs = self.t["between"], self.t["between_still"]
        a, as_ = self.t["acting"], self.t["acting_still"]
        longest = max(self.longest_still, self.still_run)
        lines = [f"LIVING BODY ({'on' if self.on else 'OFF (--no-living)'}): between his moves he stood completely still "
                 f"{bs:.0f} of {b:.0f} s ({100 * bs / b if b else 0:.0f} %); longest still stretch {longest:.0f} s; "
                 f"{len(self.stretches)} still stretches of 3 s or more"
                 + (f" (median {sorted(self.stretches)[len(self.stretches) // 2]:.0f} s)" if self.stretches else ""),
                 f"   (during his own moves: still {as_:.0f} of {a:.0f} s)"]
        if self.styles:
            lines.append(f"   how he said his feelings showed in his body (BODY, {len(self.styles)}×): " + " | ".join(self.styles[-8:]))
        if self.glances.get("opened his beak a little"):
            lines.append("   (the beak: the simulated duck has no beak motor, so the beak moves only on the real duck)")
        if self.glances:
            lines.append("   what his body did by itself: " + ", ".join(f"{k} ×{v}" for k, v in self.glances.most_common()))
        return lines
