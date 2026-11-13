"""Anka's body. For now a virtual one: a small 2D room he can walk around in (python run_anka.py --simulate --body).

The brain decides what to do (a line like "ACT: walk to the ball"); the body does it and says what happened.
It only offers what the Microduck can do (walk, turn, sit, kick, pick up with the beak), so when the real duck
arrives, a Microduck body with the same do() / describe() can take this one's place."""
import math
import re
import threading
import time
from collections import deque

import numpy as np

import random

from settings import BODY, BODY_SENSES, HEAR_DIRECTION, MUJOCO, NAME
CAM_H_ = 0.20   # (his eyes' height – the same as in body_view)
from anka.shared import S, cv2, state

ROOM_W, ROOM_H = 5.0, 4.0     # metres. x goes east (towards the door), y goes north (towards the window)
DUCK_RADIUS = 0.12            # the Microduck is about 25 cm tall and ~20 cm wide
SEE_ANGLE = 35                # degrees to each side he can see (the camera looks straight ahead)
SEE_FAR = 5.5                 # metres
MAX_STEP = 2.0                # metres per walk action at most
STOP_BEFORE = 0.35            # 'walk to' stops this far before the target
WALK_SPEED = 0.35             # m/s – only how fast it is SHOWN in the room window (the brain gets the result at once)
TURN_SPEED = 150              # degrees/s (shown)
DOOR_IN = (4.75, 1.3)         # where people come in and go out

FURNITURE = {                 # name: (x1, y1, x2, y2, height in cm) – he can't walk through or climb these
    "sofa": (0.3, 0.0, 2.3, 0.9, 45),
    "table": (2.6, 1.8, 3.6, 2.6, 75),
    "bookshelf": (4.6, 2.4, 5.0, 4.0, 180),
}
WALL_THINGS = {"window": (2.0, ROOM_H), "door": (ROOM_W, 1.3)}   # in the walls – he can look at them and go to them
SPOTS = {                     # where people can be (for the test script): next to the furniture, not inside it
    "sofa": (1.3, 1.15), "table": (3.1, 1.5), "window": (2.0, 3.6), "door": (4.65, 1.3),
    "bookshelf": (4.3, 3.2), "middle": (2.5, 2.0),
}
ON_TOP = {"sofa": 0.45, "table": 0.75, "bookshelf": 1.0}   # how high a thing put "on" this furniture is (m)
PUT_SPOTS = {"sofa": (1.7, 0.84), "bookshelf": (4.8, 3.2)}   # where the test script puts things up on furniture
START = {                     # the small things in the room
    "ball": {"pos": (3.8, 0.7), "small": True, "kick": True},
    "sock": {"pos": (1.0, 2.9), "small": True, "kick": False},
    "guitar": {"pos": (4.3, 3.7), "small": False, "kick": False},
    "cup": {"pos": (3.1, 2.2), "small": True, "kick": False, "on": "table"},
}


def _angle_to(dx, dy):
    return math.degrees(math.atan2(dy, dx))


def _wrap(a):
    return (a + 180) % 360 - 180


def _side(rel):
    """Angle relative to where he faces -> words."""
    a = abs(rel)
    side = "left" if rel > 0 else "right"
    if a <= 12:
        return "straight ahead"
    if a <= 35:
        return f"ahead, a bit to the {side}"
    if a <= 80:
        return f"to your {side}"
    if a <= 140:
        return f"behind you on the {side}"
    return "behind you"


def _metres(d):
    return f"{d * 100:.0f} cm" if d < 1 else f"{d:.1f} m"


def _compass(h):
    return ["east (towards the door)", "north (towards the window)", "west", "south (towards the sofa)"][
        int(((h % 360) + 45) // 90) % 4]


class VirtualBody:
    """A duck in a 2D room. All positions in metres; heading in degrees (0 = east, 90 = north)."""

    def __init__(self):
        self.x, self.y, self.heading = 2.5, 1.4, 90.0
        self.sitting = False
        self.holding = None
        self.things = {k: dict(v) for k, v in START.items()}
        self.people = {}          # name -> (x, y)
        self.trail = [(self.x, self.y)]
        self.actions = []         # (what he chose, what happened)
        self.walked = 0.0
        # what the room window shows: it plays his moves one after the other, at walking speed
        self.keys = deque()
        self.show = self._pose()
        self.show_things = {k: list(v["pos"]) for k, v in self.things.items()}
        self.show_people = {}
        self.show_trail = [(self.x, self.y)]
        self.last_frame = time.time()

    def _pose(self):
        return {"x": self.x, "y": self.y, "h": self.heading, "sit": self.sitting, "hold": self.holding,
                "things": {k: tuple(v["pos"]) for k, v in self.things.items()}}

    def _key(self):
        """Remembers how he stands right now, so the window can play it."""
        pose = self._pose()
        if not self.keys or self.keys[-1] != pose:
            self.keys.append(pose)

    # ---------- the world ----------
    def place_person(self, name, spot):
        if spot is None:
            self.people.pop(name, None)
        else:
            self.people[name] = SPOTS.get(spot, spot) if isinstance(spot, str) else spot

    def _positions(self):
        """Everything he could look at or go to: name -> (x, y, kind)."""
        out = {}
        for name, (x1, y1, x2, y2, _) in FURNITURE.items():
            out[name] = ((x1 + x2) / 2, (y1 + y2) / 2, "furniture")
        for name, (x, y) in WALL_THINGS.items():
            out[name] = (x, y, "wall")
        for name, t in self.things.items():
            if name != self.holding:
                out[name] = (*t["pos"], "thing")
        for name, (x, y) in self.people.items():
            out[name] = (x, y, "person")
        return out

    def _nearest_point(self, name):
        """For furniture: the closest point of it (you walk to the sofa's edge, not its middle)."""
        if name in FURNITURE:
            x1, y1, x2, y2, _ = FURNITURE[name]
            return min(max(self.x, x1), x2), min(max(self.y, y1), y2)
        x, y, _ = self._positions()[name]
        return x, y

    def _blocked(self, x, y):
        """What would he bump into at (x, y)? None if free."""
        r = DUCK_RADIUS
        if x < r or y < r or x > ROOM_W - r or y > ROOM_H - r:
            return "the wall"
        for name, (x1, y1, x2, y2, _) in FURNITURE.items():
            if x1 - r < x < x2 + r and y1 - r < y < y2 + r:
                return f"the {name}"
        for name, (px, py) in self.people.items():
            if math.hypot(px - x, py - y) < 0.3:
                return name
        for name, t in self.things.items():
            if name != self.holding and not t.get("on") and not t["small"] and math.hypot(t["pos"][0] - x, t["pos"][1] - y) < 0.25:
                return f"the {name}"
        return None

    def _clear(self, a, b):
        """Can he walk in a straight line from a to b? Returns what is in the way, or None."""
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.03))
        for k in range(1, n + 1):
            hit = self._blocked(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
            if hit:
                return hit
        return None

    def _path(self, px, py, stop):
        """His legs find a way around furniture (like a walking controller does) – the brain only says where.
        Returns the points to walk through, or None if there is no way."""
        cell = 0.05
        W, H = int(ROOM_W / cell), int(ROOM_H / cell)
        centre = lambda c: ((c[0] + 0.5) * cell, (c[1] + 0.5) * cell)
        start = (int(self.x / cell), int(self.y / cell))
        prev, queue, end = {start: None}, deque([start]), None
        while queue:
            c = queue.popleft()
            x, y = centre(c)
            if math.hypot(x - px, y - py) <= stop:
                end = c
                break
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                n = (c[0] + di, c[1] + dj)
                if n not in prev and 0 <= n[0] < W and 0 <= n[1] < H and self._blocked(*centre(n)) is None:
                    prev[n] = c
                    queue.append(n)
        if end is None:
            return None
        cells = []
        while end is not None:
            cells.append(end)
            end = prev[end]
        points = [centre(c) for c in reversed(cells)][1:] or [(self.x, self.y)]
        route, k = [], -1
        here = (self.x, self.y)
        while k < len(points) - 1:   # straight lines where possible
            far = len(points) - 1
            while far > k + 1 and self._clear(here, points[far]):
                far -= 1
            route.append(points[far])
            here, k = points[far], far
        return route

    def _go_to(self, name):
        """Walks to something, around furniture if needed. Returns (what happened, bumped into)."""
        px, py = self._nearest_point(name)
        t = self.things.get(name)
        stop = STOP_BEFORE + (0.45 if t and t.get("on") else 0)
        who = self._person(name) if name in self.people else "the " + name
        if math.hypot(px - self.x, py - self.y) <= stop + 0.02:
            self._face(name)
            return f"turned to {who} – already right there", None
        in_way = self._clear((self.x, self.y), (px, py))
        route = self._path(px, py, stop)
        if route is None:
            self._face(name)
            moved, bumped = self._walk(math.hypot(px - self.x, py - self.y) - stop, f"walked towards {who}:", limit=8)
            return moved, bumped
        self.sitting = False
        walked = 0.0
        for wx, wy in route:
            self.heading = _angle_to(wx - self.x, wy - self.y) % 360
            self._key()
            walked += math.hypot(wx - self.x, wy - self.y)
            self.x, self.y = wx, wy
            self.trail.append((self.x, self.y))
            self._key()
        self.walked += walked
        self._face(name)
        around = in_way if in_way and in_way not in (name, f"the {name}") else None
        return f"walked to {who}: {_metres(walked)}" + (f", around {around}" if around else ""), None

    def _reach(self, name):
        """Kicking or picking something up that is out of reach: his legs walk there first (around furniture).
        Returns (what the walk did or None, problem or None)."""
        t = self.things[name]
        d = math.hypot(t["pos"][0] - self.x, t["pos"][1] - self.y)
        rel = _wrap(_angle_to(t["pos"][0] - self.x, t["pos"][1] - self.y) - self.heading)
        if d <= 0.45 and abs(rel) <= 35:
            return None, None
        walked, bumped = self._go_to(name)
        d = math.hypot(t["pos"][0] - self.x, t["pos"][1] - self.y)
        if bumped or d > 0.45:
            return walked, walked
        return walked, None

    def _find(self, words):
        """'the ball' / 'Alex' / 'to the sofa' -> a name in the room, or None."""
        words = re.sub(r"(?i)\b(the|to|towards|at|my|a|an|over|there|please)\b", " ", words).strip().lower()
        names = self._positions()
        for name in sorted(names, key=len, reverse=True):
            if re.search(rf"\b{re.escape(name.lower())}\b", words) or (len(words) >= 4 and words in name.lower()):
                return name
        return None

    def _not_here(self, a, act):
        """Why he can't find what he wants to go to."""
        if self.holding and self.holding in a:
            return f"you are already holding the {self.holding}"
        gone = [n for n in (S.faces_known.names if S.faces_known else []) if n.lower() in a and n not in self.people]
        if gone:
            return f"{gone[0]} isn't in the room"
        return f"didn't know where '{act}' is"

    # ---------- what he senses ----------
    def visible(self):
        """What the camera sees right now: [(name, kind, distance, angle relative to his heading)]."""
        out = []
        for name, (x, y, kind) in self._positions().items():
            px, py = self._nearest_point(name) if kind == "furniture" else (x, y)
            d = math.hypot(px - self.x, py - self.y)
            rel = _wrap(_angle_to(px - self.x, py - self.y) - self.heading)
            if d <= SEE_FAR and abs(rel) <= SEE_ANGLE:
                out.append((name, kind, d, rel))
        return sorted(out, key=lambda v: v[2])

    def _person(self, name):
        known = S.faces_known.names if S.faces_known else []
        return name if name in known else "someone you don't recognise"

    def _describe_thing(self, name, kind, d, rel):
        extra = ""
        t = self.things.get(name)
        if t and t.get("on"):
            extra = f" (on the {t['on']} – too high for you)"
        elif name == "table":
            cups = [k for k, v in self.things.items() if v.get("on") == "table" and k != self.holding]
            if cups:
                extra = f" ({', '.join('a ' + c for c in cups)} on it)"
        who = self._person(name) if kind == "person" else f"the {name}"
        return f"{who} {_metres(d)} {_side(rel)}{extra}"

    def where_am_i(self):
        near = min(SPOTS, key=lambda s: math.hypot(SPOTS[s][0] - self.x, SPOTS[s][1] - self.y))
        d = math.hypot(SPOTS[near][0] - self.x, SPOTS[near][1] - self.y)
        place = f"near the {near}" if d < 1.0 else "in the middle of the room" if near == "middle" else f"in the room, {_metres(d)} from the {near}"
        return place

    def describe(self, last=None):
        """The text the brain gets about his body (short – it comes in every thought)."""
        seen = self.visible()
        lines = [f"[Your body: you are {'sitting' if self.sitting else 'standing'} {self.where_am_i()}, "
                 f"facing {_compass(self.heading)}."
                 + (f" You are holding the {self.holding} in your beak." if self.holding else " Your beak is empty.") + "]",
                 "[In front of you: " + ("; ".join(self._describe_thing(*v) for v in seen) if seen
                                         else "nothing – a bare wall or floor") + ".]"]
        unseen = [p for p in self.people if p not in [v[0] for v in seen]]
        if unseen:
            lines.append("[In the room but not in front of you (you can hear them): "
                         + "; ".join(f"{self._person(p)} – {self.voice_from(p)}" for p in unseen) + ".]")
        if last:
            lines.append(f"[Your last move: {last}]")
        return "\n".join(lines)

    def voice_from(self, name):
        """Where a voice comes from (he can hear people he can't see)."""
        if name not in self.people:
            return "from another room"
        px, py = self.people[name]
        d = math.hypot(px - self.x, py - self.y)
        rel = _wrap(_angle_to(px - self.x, py - self.y) - self.heading)
        return f"{_side(rel).replace('straight ahead', 'in front of you')}, {'close' if d < 1.2 else 'across the room' if d > 2.5 else 'a few steps away'}"

    # ---------- what he does ----------
    def _walk(self, dist, label, limit=MAX_STEP):
        self.sitting = False
        self._key()
        step = 0.05 * (1 if dist >= 0 else -1)
        moved, bumped = 0.0, None
        rad = math.radians(self.heading)
        while abs(moved) + 1e-9 < min(abs(dist), limit):
            nx, ny = self.x + step * math.cos(rad), self.y + step * math.sin(rad)
            bumped = self._blocked(nx, ny)
            if bumped:
                break
            self.x, self.y = nx, ny
            moved += step
        self.walked += abs(moved)
        self.trail.append((self.x, self.y))
        self._key()
        done = f"{label} {_metres(abs(moved))}" if abs(moved) > 0.02 else "couldn't move"
        if bumped in self.people:
            return done + f" – you are right next to {self._person(bumped)} already", bumped
        return done + (f" – bumped into {bumped} (to get past it: walk to <where you want to go> – your legs "
                       "find a way around)" if bumped else ""), bumped

    def _turn(self, degrees):
        self.heading = (self.heading + degrees) % 360
        self._key()
        return f"turned {'left' if degrees > 0 else 'right'} {abs(degrees):.0f}° – now facing {_compass(self.heading)}"

    def _face(self, name):
        px, py = self._nearest_point(name)
        target = _angle_to(px - self.x, py - self.y)
        rel = _wrap(target - self.heading)
        self.heading = target % 360
        self._key()
        return rel

    def do(self, act):
        """Carries out one action the brain chose. Returns (what happened, did it go well?)."""
        a = act.strip().strip(".").lower()
        num = re.search(r"(-?\d+(?:\.\d+)?)", a)
        n = float(num.group(1)) if num else None
        result, bumped = None, None
        self.bumped = None
        if not a or re.fullmatch(r"(none|nothing|no|wait|stay|stay here|stay still|stay where i am|rest)\b.*", a):
            return None, True
        elif re.match(r"(walk|go|come|move|waddle|run|head)( over| back)? (to|towards|up to)\b|approach\b", a):
            name = self._find(re.sub(r"^\w+( over| back)? (to|towards|up to)?", "", a))
            if not name:
                result = self._not_here(a, act)
            else:
                result, bumped = self._go_to(name)
        elif re.match(r"(walk|go|move|waddle) around\b", a) and not re.search(r"\baround (the room|a bit)\b", a):
            result = "your legs need a place to go – say walk to <thing or person>, and they find a way around"
        elif re.search(r"\b(walk|go|move|waddle|step)\b", a):
            back = bool(re.search(r"\b(back|backward|backwards)\b", a))
            if n is not None and re.search(r"\bsteps?\b", a):
                dist = n * 0.15                      # a duck step is about 15 cm
            else:
                dist = (n if n is not None else 50) / (100 if (n is None or n > 5 or "cm" in a) else 1)
            if back and dist > 0.3:
                dist = 0.3   # a duck only takes a few small steps backwards
            result, bumped = self._walk(-dist if back else dist, "walked backwards" if back else "walked forward")
            if back:
                result += " (ducks only take small steps backwards – to go somewhere behind you, walk to it)"
        elif re.search(r"\b(turn|rotate|spin)\b", a):
            if "around" in a:
                result = self._turn(180)
            elif re.search(r"\bto\b|\btowards\b|\bface\b", a) and self._find(a.split("to", 1)[-1]):
                name = self._find(a.split("to", 1)[-1])
                rel = self._face(name)
                result = f"already facing {self._person(name) if name in self.people else 'the ' + name}" if abs(rel) < 5 else \
                    f"turned {'left' if rel > 0 else 'right'} to face {self._person(name) if name in self.people else 'the ' + name}"
            else:
                deg = n if n is not None else 90
                result = self._turn(-deg if "right" in a else deg)
        elif re.search(r"\blook\b", a) and not re.search(r"\baround\b|\bbehind\b", a) and self._find(a.replace("look", "")):
            name = self._find(a.replace("look", ""))
            rel = self._face(name)
            result = (f"turned {'left' if rel > 0 else 'right'} to look at "
                      f"{self._person(name) if name in self.people else 'the ' + name}" if abs(rel) > 3 else
                      f"looked at {self._person(name) if name in self.people else 'the ' + name}")
        elif re.search(r"\blook around\b|\blook behind\b|\bscan\b", a):
            seen = []
            for name, (x, y, kind) in self._positions().items():
                px, py = self._nearest_point(name) if kind == "furniture" else (x, y)
                d = math.hypot(px - self.x, py - self.y)
                if d <= SEE_FAR:
                    rel = _wrap(_angle_to(px - self.x, py - self.y) - self.heading)
                    seen.append(self._describe_thing(name, kind, d, rel))
            result = "looked all around: " + "; ".join(seen)
        elif re.search(r"\bsit\b", a):
            self.sitting, result = True, "sat down"
        elif re.search(r"\b(stand|get up)\b", a):
            self.sitting, result = False, "stood up"
        elif "kick" in a:
            name = self._find(re.sub(r"\b(kick|to|at)\b.*?(?=\b(ball|sock|cup|guitar)\b|$)", " ", a)) or "ball"
            t = self.things.get(name)
            if not t or not t.get("kick"):
                result = f"can't kick {('the ' + name) if name else 'that'}"
            else:
                walked, problem = self._reach(name)
                if problem:
                    result = f"couldn't get to the {name}: {problem}"
                else:
                    rad = math.radians(self.heading)
                    bx, by = t["pos"]
                    stopped_by = None
                    for _ in range(30):  # it rolls up to 1.5 m, stops at walls and furniture
                        nx, ny = bx + 0.05 * math.cos(rad), by + 0.05 * math.sin(rad)
                        hit = [f for f, (x1, y1, x2, y2, _) in FURNITURE.items() if x1 < nx < x2 and y1 < ny < y2]
                        if not (0.1 < nx < ROOM_W - 0.1 and 0.1 < ny < ROOM_H - 0.1) or hit:
                            stopped_by = f"the {hit[0]}" if hit else "the wall"
                            break
                        bx, by = nx, ny
                    rolled = math.hypot(bx - t["pos"][0], by - t["pos"][1])
                    t["pos"] = (bx, by)
                    near = [p for p, (px, py) in self.people.items() if math.hypot(px - bx, py - by) < 0.45]
                    result = (f"{walked}, then " if walked else "") + (f"kicked the {name}, but it can't roll that way – {stopped_by} is right in front of it"
                              if rolled < 0.05 else
                              f"kicked the {name} – it rolled {_metres(rolled)} straight ahead"
                              + (f" until it hit {stopped_by}" if stopped_by else "")
                              + (f" and stopped right at {self._person(near[0])}" if near else
                                 f" and is now {_metres(math.hypot(bx - self.x, by - self.y))} from you"))
        elif re.search(r"\b(pick up|grab|take|bite)\b", a):
            name = self._find(re.sub(r"\b(pick up|grab|take|bite)\b", "", a))
            t = self.things.get(name) if name else None
            if self.holding:
                result = f"can't – already holding the {self.holding}"
            elif not t:
                result = "there's nothing like that to pick up"
            elif t.get("on"):
                result = f"the {name} is on the {t['on']} – too high for a small duck"
            elif not t["small"]:
                result = f"the {name} is too big for your beak"
            else:
                walked, problem = self._reach(name)
                if problem:
                    result = f"couldn't get to the {name}: {problem}"
                else:
                    self.holding = name
                    result = (f"{walked}, then " if walked else "") + f"picked up the {name} with your beak"
        elif re.search(r"\b(put down|drop|let go|give)\b", a):
            if not self.holding:
                result = "you aren't holding anything"
            else:
                rad = math.radians(self.heading)
                spot = (self.x + 0.2 * math.cos(rad), self.y + 0.2 * math.sin(rad))
                if any(x1 < spot[0] < x2 and y1 < spot[1] < y2 for x1, y1, x2, y2, _ in FURNITURE.values()) or not (
                        0.05 < spot[0] < ROOM_W - 0.05 and 0.05 < spot[1] < ROOM_H - 0.05):
                    spot = (self.x, self.y)   # no room in front of him – it lands at his feet
                near = [p for p, (x, y) in self.people.items() if math.hypot(x - spot[0], y - spot[1]) < 0.6]
                self.things[self.holding]["pos"] = spot
                result = f"put down the {self.holding} on the floor" + (f", right by {self._person(near[0])}" if near else "")
                self.holding = None
        else:
            result = f"your body doesn't know how to '{act}'"
        self.actions.append((act, result))
        self._key()
        self.bumped = bumped
        ok = not bumped and not re.match(r"(couldn't|can't|the \w+ is (too|on)|there's nothing|you aren't|you are already holding|\w+ isn't in the room|your body doesn't|didn't know)", result)
        return result, ok

    # ---------- the room window: his moves played at walking speed ----------
    def animate(self):
        now = time.time()
        dt, self.last_frame = min(0.2, now - self.last_frame), now
        hurry = 1 + len(self.keys) / 12          # far behind? play a bit faster
        budget_turn, budget_walk = TURN_SPEED * dt * hurry, WALK_SPEED * dt * hurry
        while self.keys:
            k, sh = self.keys[0], self.show
            diff = _wrap(k["h"] - sh["h"])
            if abs(diff) > 0.5:
                turn = math.copysign(min(abs(diff), budget_turn), diff)
                sh["h"] = (sh["h"] + turn) % 360
                budget_turn -= abs(turn)
                if budget_turn <= 0:
                    break
                continue
            d = math.hypot(k["x"] - sh["x"], k["y"] - sh["y"])
            if d > 0.005:
                step = min(d, budget_walk)
                sh["x"] += (k["x"] - sh["x"]) * step / d
                sh["y"] += (k["y"] - sh["y"]) * step / d
                budget_walk -= step
                tx, ty = self.show_trail[-1]
                if math.hypot(sh["x"] - tx, sh["y"] - ty) > 0.03:
                    self.show_trail.append((sh["x"], sh["y"]))
                if budget_walk <= 0:
                    break
                continue
            sh.update({key: v for key, v in k.items() if key not in ("x", "y", "h")})
            self.keys.popleft()
        for name, (tx, ty) in self.show["things"].items():   # a kicked ball rolls
            p = self.show_things.setdefault(name, [tx, ty])
            d = math.hypot(tx - p[0], ty - p[1])
            if d > 0.005:
                step = min(d, 1.2 * dt)
                p[0] += (tx - p[0]) * step / d
                p[1] += (ty - p[1]) * step / d
        for name in set(self.people) | set(self.show_people):   # people walk in and out through the door
            target = self.people.get(name, DOOR_IN)
            p = self.show_people.setdefault(name, list(DOOR_IN))
            d = math.hypot(target[0] - p[0], target[1] - p[1])
            if d > 0.01:
                step = min(d, 0.8 * dt)
                p[0] += (target[0] - p[0]) * step / d
                p[1] += (target[1] - p[1]) * step / d
            elif name not in self.people:
                del self.show_people[name]

    def draw(self, scale=100, caption=(), bubbles=(), live=True):
        """live=True: what the window shows right now (his moves played at walking speed).
        live=False: how things really are (for room_map.png at the end). bubbles: [(who, text)]."""
        if live:
            self.animate()
            sh = self.show
            dx, dy, dh, sitting, holding = sh["x"], sh["y"], sh["h"], sh["sit"], sh["hold"]
            things = {k: tuple(v) for k, v in self.show_things.items()}
            people = {k: tuple(v) for k, v in self.show_people.items()}
            trail = self.show_trail
        else:
            dx, dy, dh, sitting, holding = self.x, self.y, self.heading, self.sitting, self.holding
            things = {k: v["pos"] for k, v in self.things.items()}
            people, trail = dict(self.people), self.trail
        caption = [caption] if isinstance(caption, str) else list(caption)
        W, H, pad = int(ROOM_W * scale), int(ROOM_H * scale), 30
        bar = 22 * max(1, len(caption)) + 8
        img = np.full((H + 2 * pad + bar, W + 2 * pad, 3), (236, 240, 244), np.uint8)
        P = lambda x, y: (int(pad + x * scale), int(pad + (ROOM_H - y) * scale))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(img, P(0, ROOM_H), P(ROOM_W, 0), (90, 90, 90), 3)
        cv2.line(img, P(1.2, ROOM_H), P(2.8, ROOM_H), (235, 190, 120), 6)             # window
        cv2.line(img, P(ROOM_W, 0.8), P(ROOM_W, 1.8), (60, 120, 170), 6)              # door
        for name, (x1, y1, x2, y2, _) in FURNITURE.items():
            cv2.rectangle(img, P(x1, y2), P(x2, y1), (190, 205, 215), -1)
            cv2.putText(img, name, P(x1 + 0.05, (y1 + y2) / 2), font, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
        if len(trail) > 1:
            cv2.polylines(img, [np.array([P(x, y) for x, y in trail], np.int32)], False, (120, 200, 240), 2, cv2.LINE_AA)
        for name, pos in things.items():
            if name == holding:
                continue
            cv2.circle(img, P(*pos), 6, (80, 80, 200), -1, cv2.LINE_AA)
            cv2.putText(img, name, (P(*pos)[0] + 8, P(*pos)[1] + 4), font, 0.4, (80, 80, 200), 1, cv2.LINE_AA)
        for name, (x, y) in people.items():
            cv2.circle(img, P(x, y), 13, (150, 110, 60), -1, cv2.LINE_AA)
            cv2.putText(img, name, (P(x, y)[0] - 18, P(x, y)[1] - 18), font, 0.5, (150, 110, 60), 1, cv2.LINE_AA)
        # what he can see
        cone = [P(dx, dy)] + [P(dx + 1.2 * math.cos(math.radians(dh + a)), dy + 1.2 * math.sin(math.radians(dh + a)))
                              for a in range(-SEE_ANGLE, SEE_ANGLE + 1, 5)]
        overlay = img.copy()
        cv2.fillPoly(overlay, [np.array(cone, np.int32)], (150, 230, 255))
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        r = int(DUCK_RADIUS * scale * (0.8 if sitting else 1))
        cv2.circle(img, P(dx, dy), r, (40, 200, 250), -1, cv2.LINE_AA)
        cv2.circle(img, P(dx, dy), r, (40, 70, 110), 2, cv2.LINE_AA)
        hx, hy = dx + 0.2 * math.cos(math.radians(dh)), dy + 0.2 * math.sin(math.radians(dh))
        cv2.line(img, P(dx, dy), P(hx, hy), (30, 120, 240), 4, cv2.LINE_AA)            # his beak points this way
        if holding:
            cv2.circle(img, P(hx, hy), 6, (80, 80, 200), -1, cv2.LINE_AA)               # in his beak
        cv2.putText(img, NAME + (" (sitting)" if sitting else "") + (f" + {holding}" if holding else ""),
                    (P(dx, dy)[0] - 20, P(dx, dy)[1] + 28), font, 0.5, (40, 70, 110), 1, cv2.LINE_AA)
        # speech bubbles
        for who, text in bubbles:
            if who == NAME:
                bx, by = P(dx, dy)
            elif who in people:
                bx, by = P(*people[who])
            else:
                continue
            words, lines = text.split(), [""]
            for w in words:
                if len(lines[-1]) + len(w) > 28 and lines[-1]:
                    lines.append("")
                lines[-1] = (lines[-1] + " " + w).strip()
            lines = lines[:3]
            wbox = max(cv2.getTextSize(line, font, 0.45, 1)[0][0] for line in lines) + 12
            hbox = 18 * len(lines) + 8
            x0 = min(max(5, bx - wbox // 2), img.shape[1] - wbox - 5)
            y0 = max(5, by - 30 - hbox)
            cv2.rectangle(img, (x0, y0), (x0 + wbox, y0 + hbox), (255, 255, 255), -1)
            cv2.rectangle(img, (x0, y0), (x0 + wbox, y0 + hbox), (120, 120, 120), 1)
            for i, line in enumerate(lines):
                cv2.putText(img, line, (x0 + 6, y0 + 18 + 18 * i), font, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
        for i, line in enumerate(caption):
            cv2.putText(img, line[:100], (10, H + 2 * pad + 18 + 22 * i), font, 0.45,
                        (60, 60, 60) if i == 0 else (120, 100, 80), 1, cv2.LINE_AA)
        return img



# ======================================================================================================
# The same body, but with the senses the real Microduck has (the default for --body; --easy = the one above).
# He gets no map and no list of where things are: only a camera picture from his own eyes, the 8×8 depth
# sensor, bumps and falls (IMU), legs that aren't perfect, a head he can turn, and a battery.
# ======================================================================================================
TABLE_TOP, LEG = 0.70, 0.06
PERSON_COLOURS = {"Alex": (165, 95, 40), "Robin": (70, 150, 60)}
VOX = 0.05                      # his 3D picture of the room: 5 cm cubes …
VOX_SIZE = (160, 140, 44)       # … 8 × 7 m like the floor plan, and from 20 cm below the floor to 2 m up
VOX_ORIGIN = (-1.0, -1.0, -0.2)      # (the same corner as his floor plan)
VOX_SHOW = 4                    # a cube counts as "something there" after this many points landed in it
FREE_NO_HIT = 2.0               # a ray that hits nothing only shows free floor this far (further it may miss a wall)
CAM_HIT = 0.3                   # one camera ray hitting something (a depth-sensor hit counts 0.9)
FREE_FADE, FREE_MIN = 4.0, 0.25    # "free" counts less further away: × (1 − distance / FREE_FADE), at least FREE_MIN
HARD_WALL = 2.5                 # this much "something there" (≈ 3 depth-sensor hits) makes a square hard to erase …
HARD_ERASE = 0.3                # … then "free" looks through it count only this much
SEGMENT_M, SEGMENT_DEG = 0.3, 15.0   # his looks are grouped by stretches of this much walking / turning; when he
                                     # finds out later where he really was, the whole stretch is moved and redrawn
KEEP_LOOKS = 1500               # looks kept for redrawing (older ones stay in the map as they are)
MAP_EVERY_M, MAP_EVERY_DEG, MAP_EVERY_S = 0.03, 3.0, 1.0   # draw into the floor plan after 3 cm / 3° – or each second
FLOOR_WHEN = 12                 # (add the floor when the sensor has fewer points than this; with none, the floor alone)
FLOOR_WEIGHT = 0.3              # (all the floor points together count this much compared with the sensor's points)
DEPTH_AI_FLOOR = True           # the depth AI's scale from the depth sensor AND the floor (camera height + tilt known)
GYRO = True                     # the Microduck's IMU gyro measures how far he really turned (±0.5° + 3 %)
EKF = True                      # where he is: a Kalman filter with the landmarks (False = the older simple nudge)
WALL_HEADING = False            # (walls are mostly square: use them to correct his sense of direction – off: with
                                # the gyro it isn't needed, and a slightly wrong room direction would turn the whole map)
WALL_SD = 3.0                   # (how exact one metre of straight wall tells the direction, degrees)
FURNITURE_NEAR = 0.4            # (an edge this close to furniture he has seen is the furniture, not a wall)
ROOM_RUNS = 15                  # (straight runs before he has a first idea of the room's main direction …)
ROOM_RUNS_FIX = 30              # (… and before the first wall is fixed – then the direction is kept)
WALL_MERGE = 0.35               # (two fixed walls side by side closer than this are one wall)
WALL_MIN_LEN = 0.8              # a straight run at least this long can become a wall …
WALL_SUPPORT = 3                # … once it has been found this many times (each check a few seconds apart);
WALL_EVERY = 3.0                # (seconds between checks) – then it is a fixed line in his floor plan for good
REDRAW = True                   # redraw the map when he finds out where he really was
SCAN_MATCH = False              # OFF since test 22: fitting each look onto his floor plan sometimes pulled him 1–3 m / 20–60° off
ROUTE_ROOM = 0.12      # 2026-11-08: planned routes keep this much room to spare beside things he knows (0.03 = the old, tight way)
ROUTE_DODGES = 2      # max sidesteps per stretch of a planned route (the old way: 8)
ROUTE_DODGE = False   # 2026-11-08: on a PLANNED route (walk to something remembered/heard, explore) the legs no longer
                      # sidestep by themselves when the reflex fires – the route already goes around what he knows, so
                      # sidestepping out and steering back made a sawtooth along walls and the sofa (up to 20 sidesteps
                      # in one walk). Instead he plans again (the new obstacle is on his map now). Walking to something
                      # he SEES still sidesteps as before. True = the old way.
                                # (offline: 3–6 of 16 runs lost themselves with it, 0–2 without, same floor-plan quality)
MATCH_SHIFT, MATCH_TURN = 0.10, 6.0   # it tries up to ±10 cm and ±6° …
MATCH_TRUST = 0.5                     # … and moves his guess half of the way to the best fit
MATCH_LAG = 1.0                       # (fit onto the map as it was this much walking ago – a 90° turn counts as 1 m)
MATCH_CAM = 0.4                       # (the camera's rays count less – they are less exact)
MATCH_PRIOR, MATCH_GAIN = 0.1, 0.08   # (prefer small corrections; only act on a clearly better fit)
SCAN_ONLY_WHEN_LOST = True            # scan matching only helps when the landmark filter is unsure where he is: when it
SCAN_LOST_POS, SCAN_LOST_HEAD = 0.15, 4.0   # knows his place to better than 15 cm and 4° it only pulls him off (test 22)
FADE_PEOPLE = True              # people in his floor plan fade instead of staying as walls
MAP_SURE = 1.5        # his floor plan is sure a square is taken (≈ the depth sensor saw it twice, or he bumped it)
MAP_AHEAD, MAP_SIDE = 0.06, 0.12   # how far ahead / to each side his floor plan is checked (his map has 5 cm squares)
MAP_VETO_AFTER = 2      # stopped by his floor plan this many times at the same spot (within 12 cm) …
MAP_VETO_CLEAR = 0.4     # … while the depth sensor sees open floor this far ahead (m) → the squares are ignored …
MAP_VETO_FOR = 120       # … for this many seconds (the floor plan is memory, the depth sensor is live)
REFLEX = 0.12          # the depth sensor stops his legs when something is this close in front of him
FALL_BUMP = 0.15       # chance to fall over when walking into something (backwards: more, he can't see there)
BATTERY_PER_MIN = 1.6  # % per minute (the real one lasts about an hour)
# Landmarks: things that don't move. When he sees one he already knows, the difference between where he sees it and
# where he expected it corrects his own idea of where he is (so his mental map doesn't drift away).
LANDMARKS = {"sofa": (1.3, 0.9), "table": (3.1, 2.2), "bookshelf": (4.6, 3.2), "door": (ROOM_W, 1.3), "window": (2.0, ROOM_H)}
MAP_CELL = 0.05        # his own floor plan: 5 cm squares
MAP_ORIGIN = (-1.0, -1.0)
MAP_SIZE = (160, 140)  # 8 × 7 m (more than the room – he doesn't know how big it is)
FIX_STRENGTH = 0.1     # how much one sighting of a landmark corrects him (it happens several times a second)
HEADING_FIX = 1.5      # ...and his sense of direction (relative to FIX_STRENGTH)


def _about(d):
    d = abs(d)
    return f"about {max(10, round(d * 10) * 10):.0f} cm" if d < 0.95 else f"about {round(d * 2) / 2:g} m"


# ---------- 2D poses for his map (heading in degrees, x, y): "pose B seen from pose A" etc. ----------
def _tf(h, x, y):
    return (h, x, y)


def _tf_mul(a, b):
    c, s = math.cos(math.radians(a[0])), math.sin(math.radians(a[0]))
    return (a[0] + b[0], a[1] + c * b[1] - s * b[2], a[2] + s * b[1] + c * b[2])


def _tf_inv(a):
    c, s = math.cos(math.radians(a[0])), math.sin(math.radians(a[0]))
    return (-a[0], -(c * a[1] + s * a[2]), -(-s * a[1] + c * a[2]))



def put_thing_on(body_, name, furniture):
    """Test script: someone puts a small thing UP on furniture (2026-11-09 transfer test: the sock on the sofa)."""
    t = body_.things.get(name)
    if t is None or furniture not in PUT_SPOTS or getattr(body_, "holding", None) == name:
        return False
    t["pos"], t["on"] = PUT_SPOTS[furniture], furniture
    return True


class RealSensesBody(VirtualBody):
    real = True

    def __init__(self):
        super().__init__()
        self.head_yaw, self.pitch, self.fallen = 0.0, 0.0, False
        self.panorama_next, self.falls, self.started = False, 0, time.time()
        self._cache = {}
        self.show = self._pose()
        # what he THINKS: his own position from counting steps and turns (it drifts away from the truth)
        self.est_x, self.est_y, self.drift = self.x, self.y, 0.0
        self.mind_map = {}      # name -> {"pos": (x, y) in his own estimate, "t": when he last saw it, "kind"}
        self.place_notes = []   # [(pos, text, time)] – notes he chose to write about places
        self.stopped_in_row = 0
        self.landmarks = {}     # name -> {"pos": where he thinks it is, "n": how many sightings it is based on}
        self.fixes = 0          # how many times a landmark has corrected him (shown after the test)
        # his own floor plan, built from the depth sensor like a robot vacuum's map: for every square, how sure he
        # is that something is there (> 0) or that it's free floor (< 0); 0 = he doesn't know (never looked there)
        self.floor = np.zeros(MAP_SIZE, np.float32)
        self.people_floor = np.zeros(MAP_SIZE, np.float32)   # people he saw in the way – fades (people move)
        self._match_map, self._match_next, self._match_age, self.scan_fixes, self._turn_total = None, None, 0.0, 0, 0.0
        self._last_drawn = None     # where he was the last time he drew into his floor plan
        # the floor plan is made from the evidence of all his looks: how often something was there (hit) and how
        # often he looked through a square (free). Every look is kept with where he was – from his step counting –
        # so when a landmark or the map itself shows he had drifted, the looks are moved and the map redrawn.
        self.map_hit = np.zeros(MAP_SIZE, np.float32)
        self.map_free = np.zeros(MAP_SIZE, np.float32)
        self.base_hit = np.zeros(MAP_SIZE, np.float32)     # (looks too old to redraw)
        self.base_free = np.zeros(MAP_SIZE, np.float32)
        self.looks = []             # [(stretch, pose from step counting, head yaw, rays, footprint?)]
        self.bumps = []             # [(stretch, point from step counting)]
        self.corr = (0.0, 0.0, 0.0)           # his corrections so far: where he thinks he is = corr ∘ step counting
        self.stretch, self.stretch_start = 0, (0.0, self.x, self.y)
        self.stretch_corr = {0: self.corr}    # the best correction known for each stretch (its last one)
        self.redraws = 0
        self._look_truth = []
        self.ekf_mu = None
        self.wall_ref, self.wall_fixes = None, 0
        self.walls = []             # walls he is sure of: fixed straight lines that never change (see _find_walls)
        self._wall_candidates, self._walls_checked, self._wall_mask = [], 0.0, None
        self.replans = 0
        self.unsure = 0.0       # how much the last landmark sighting had to correct him (metres)
        self.lock = threading.RLock()   # his eyes (5 times a second) and his legs take turns
        self._da = {"busy": False, "last": 0.0, "result": None}   # the depth AI works in the background
        self.depth_ai_errors = []       # (in the simulation) how far its guesses were from the truth
        self.depth_ai_errors_people = []   # (the same, for pictures where a person stands close – measured roughly there:
                                           #  the 'truth' is one pixel column, which can look between the legs)
        # his 3D picture of the room: how many depth points landed in each 5 cm cube, and their colour (summed)
        self.vox_n = np.zeros(VOX_SIZE, np.float32)
        self.vox_rgb = np.zeros(VOX_SIZE + (3,), np.float32)

    # ---------- what he believes about where he is (odometry) ----------
    def est_heading(self):
        return (self.heading + self.drift) % 360

    def _turned(self, degrees):
        """The gyro measures turns well, but not perfectly."""
        self._turn_total = getattr(self, "_turn_total", 0.0) + abs(degrees)
        self.drift += random.gauss(0, 0.5 + 0.03 * abs(degrees))

    def _counted_step(self, believed, back):
        rad = math.radians(self.est_heading()) + (math.pi if back else 0)
        self.est_x += believed * math.cos(rad)
        self.est_y += believed * math.sin(rad)

    def detect(self):
        """A pretend object detector (like YOLO on the real duck) + the depth sensor: what is in the picture, at which
        angle (accurate, ±2°) and how far away (±8 % within the depth sensor's 4 m, ±20 % further away).
        Returns [(name, kind, angle relative to where his head points, distance)]."""
        yaw = self.heading + self.head_yaw
        out = []
        for name, v in self.seen_names(30).items():
            pos = self.people.get(name) if v["kind"] == "person" else (self.things[name]["pos"] if name in self.things else None)
            if pos:
                out.append((name, v["kind"], pos))
        for name, pos in LANDMARKS.items():
            d = math.hypot(pos[0] - self.x, pos[1] - self.y)
            if abs(_wrap(_angle_to(pos[0] - self.x, pos[1] - self.y) - yaw)) <= 33 and 0.3 < d < 5.0:
                out.append((name, "landmark", pos))
        seen = []
        for name, kind, (px, py) in out:
            d = math.hypot(px - self.x, py - self.y)
            d *= random.gauss(1, 0.08 if d < 4 else 0.2)
            seen.append((name, kind, _wrap(_angle_to(px - self.x, py - self.y) - yaw) + random.gauss(0, 2), d))
        return seen

    def _landmarks_only(self):
        """A quick look while walking: only the landmarks, to keep his sense of where he is right."""
        self._localise([x for x in self.detect() if x[1] == "landmark"])

    def _localise(self, sightings):
        """Where am I? Landmarks he sees (name, kind, angle, distance) correct his step counting. With EKF = True a
        small Kalman filter (EKF-SLAM) estimates where he is AND where the landmarks are, together, weighing each
        measurement by how exact it is; otherwise the older simple nudge (_fix_position)."""
        if not EKF:
            for name, kind, bearing, d in sightings:
                self._fix_position(name, bearing, d)
            return
        self._ekf_predict()
        before = self._est_pose()
        for name, kind, bearing, d in sightings:
            self._ekf_see(name, bearing, d)
        self._ekf_apply(before)

    def _ekf_apply(self, before):
        """His estimate becomes what the filter says (and the map remembers the correction)."""
        x, y, th = self.ekf_mu[:3]
        self.est_x, self.est_y = float(x), float(y)
        self.drift = _wrap(math.degrees(th) - self.heading)       # (est_heading = heading + drift)
        moved = math.hypot(self.est_x - before[1], self.est_y - before[2])
        if moved > 1e-9 or abs(_wrap(self.est_heading() - before[0])) > 1e-9:
            self._corrected(before)
            self.fixes += 1
        self.unsure = max(self.unsure, moved)
        for name, j in self.ekf_index.items():
            lm = self.landmarks.setdefault(name, {"pos": (0.0, 0.0), "n": 0})
            lm["pos"] = (float(self.ekf_mu[j]), float(self.ekf_mu[j + 1]))

    def _wall_heading(self, rays):
        """Walls and big furniture usually stand at right angles to each other. Straight runs of depth points
        (a wall, a sofa side) seen now are compared with the room's main direction: if they all look a few degrees
        tilted, it is his own sense of direction that is off – a measurement for the filter."""
        # (the depth sensor's 8 columns – exact; the camera's distances wobble too much to see a wall's angle)
        pts = sorted(((self.head_yaw + ang, reach) for ang, reach, hit, src in rays if hit and reach < 3.5
                      and src == "tof" and abs(((ang * 8 / 45) % 1) - 0.5) < 0.01), key=lambda p: p[0])
        if len(pts) < 4:
            return
        xy = [(r * math.cos(math.radians(a)), r * math.sin(math.radians(a))) for a, r in pts]
        runs, run = [], [xy[0]]
        for p_, q in zip(xy, xy[1:]):
            if math.hypot(q[0] - p_[0], q[1] - p_[1]) < 0.12 + 0.1 * math.hypot(*q):
                run.append(q)
            else:
                runs.append(run)
                run = [q]
        runs.append(run)
        angles, weights = [], []
        for r in runs:
            if len(r) < 4:
                continue
            P = np.array(r)
            span = math.hypot(*(P[-1] - P[0]))
            if span < 0.4:
                continue
            c = P - P.mean(0)
            w_, v = np.linalg.eigh(c.T @ c)
            if w_[0] > 0.02 * w_[1]:          # (not straight enough – a corner or a person)
                continue
            angles.append(math.degrees(math.atan2(v[1, 1], v[0, 1])))
            weights.append(span)
        if not angles:
            return
        world = [self.est_heading() + a for a in angles]
        self._wall_dbg = (world, weights)
        if not self.walls:                          # the room's main direction: from all straight runs so far,
            runs = self.__dict__.setdefault("_wall_first", [])   # refined until the first wall is fixed
            runs += [(w % 90, sp) for w, sp in zip(world, weights)]
            del runs[:-400]
            if len(runs) >= ROOM_RUNS and sum(sp for _, sp in runs) >= ROOM_RUNS * 0.8:
                self.wall_ref = self._room_direction([a for a, _ in runs], [sp for _, sp in runs])
        if not WALL_HEADING or self.wall_ref is None:
            return
        dev = np.array([((w - self.wall_ref + 45) % 90) - 45 for w in world])
        good = np.abs(dev) < 12
        if good.sum() == 0:
            return
        delta = float(np.average(dev[good], weights=np.array(weights)[good]))
        sd = math.radians(WALL_SD / math.sqrt(float(np.sum(np.array(weights)[good]))))
        self._ekf_predict()
        before = self._est_pose()
        mu, P = self.ekf_mu, self.ekf_cov
        n = len(mu)
        H = np.zeros(n)
        H[2] = 1.0
        innov = -math.radians(delta)               # (he should turn his idea of his heading back by delta)
        S = P[2, 2] + sd ** 2
        if innov ** 2 / S > 9:
            return
        K = P[:, 2] / S
        self.ekf_mu = mu + K * innov
        self.ekf_cov = P - np.outer(K, P[2, :])
        self.ekf_cov = (self.ekf_cov + self.ekf_cov.T) / 2
        self._ekf_apply(before)
        self.wall_fixes += 1

    @staticmethod
    def _room_direction(angles, weights):
        """The room's main direction (0–90°) from many straight runs: the most common direction, then the
        weighted average of the runs that agree with it within 5° (odd ones – a slanted thing, a mistake – are
        left out)."""
        a = np.array(angles) % 90
        w = np.array(weights, float)
        best, best_w = 0.0, -1.0
        for c in np.arange(0, 90, 1.0):              # the direction most runs agree with
            dev = ((a - c + 45) % 90) - 45
            sw = w[np.abs(dev) < 5].sum()
            if sw > best_w:
                best, best_w = c, sw
        dev = ((a - best + 45) % 90) - 45
        keep = np.abs(dev) < 5
        return float((best + np.average(dev[keep], weights=w[keep])) % 90)

    def _ekf_sync(self):
        """(when something else moved his estimate – scan matching – the filter follows)"""
        if EKF and getattr(self, "ekf_mu", None) is not None:
            self.ekf_mu[:3] = (self.est_x, self.est_y, math.radians(self.est_heading()))
            self.ekf_odom = _tf_mul(_tf_inv(self.corr), self._est_pose())

    def _ekf_predict(self):
        """Move the estimate by what his step counting says happened since last time, and grow the uncertainty:
        legs ±10 % of the distance, sideways ±5 %, wobble ~0.5° per step, gyro 0.5° + 3 % per turn."""
        odom = _tf_mul(_tf_inv(self.corr), self._est_pose())
        if getattr(self, "ekf_mu", None) is None:
            self.ekf_mu = np.array([self.est_x, self.est_y, math.radians(self.est_heading())])
            self.ekf_cov = np.eye(3) * 1e-6
            self.ekf_index = {}
            self.ekf_odom = odom
            return
        dh, dx, dy = _tf_mul(_tf_inv(self.ekf_odom), odom)
        self.ekf_odom = odom
        d = math.hypot(dx, dy)
        if d < 1e-6 and abs(dh) < 1e-6:
            return
        mu, P = self.ekf_mu, self.ekf_cov
        th = mu[2]
        c, s_ = math.cos(th), math.sin(th)
        mu[0] += c * dx - s_ * dy
        mu[1] += s_ * dx + c * dy
        mu[2] += math.radians(dh)
        G = np.eye(3)
        G[0, 2] = -s_ * dx - c * dy
        G[1, 2] = c * dx - s_ * dy
        sd_along, sd_side = 0.10 * d + 0.005, 0.05 * d + 0.005
        sd_th = math.radians((0.05 if GYRO else 0.5) * math.sqrt(d / 0.05) + (0.5 + 0.03 * abs(dh) if abs(dh) > 1 else 0.0))
        R = np.array([[c, -s_], [s_, c]])
        Q = np.zeros((3, 3))
        Q[:2, :2] = R @ np.diag([sd_along ** 2, sd_side ** 2]) @ R.T
        Q[2, 2] = sd_th ** 2
        P[:3, :] = G @ P[:3, :]
        P[:, :3] = P[:, :3] @ G.T
        P[:3, :3] += Q

    def _ekf_see(self, name, bearing, d):
        """One landmark sighting: angle ±2.5°, distance ±8 % (±20 % beyond 4 m)."""
        mu, P = self.ekf_mu, self.ekf_cov
        head = math.radians(self.head_yaw)
        b = math.radians(bearing)
        sr = (0.08 if d < 4 else 0.2) * d + 0.02
        Rm = np.diag([sr ** 2, math.radians(2.5) ** 2])
        if name not in self.ekf_index:                     # first sighting: put it on the map
            phi = mu[2] + head + b
            lx, ly = mu[0] + d * math.cos(phi), mu[1] + d * math.sin(phi)
            n = len(mu)
            Gp = np.array([[1, 0, -d * math.sin(phi)], [0, 1, d * math.cos(phi)]])
            Gz = np.array([[math.cos(phi), -d * math.sin(phi)], [math.sin(phi), d * math.cos(phi)]])
            P2 = np.zeros((n + 2, n + 2))
            P2[:n, :n] = P
            P2[n:, n:] = Gp @ P[:3, :3] @ Gp.T + Gz @ Rm @ Gz.T
            P2[n:, :n] = Gp @ P[:3, :]
            P2[:n, n:] = P2[n:, :n].T
            self.ekf_mu = np.concatenate([mu, [lx, ly]])
            self.ekf_cov = P2
            self.ekf_index[name] = n
            return
        j = self.ekf_index[name]
        dx, dy = mu[j] - mu[0], mu[j + 1] - mu[1]
        q = dx * dx + dy * dy
        r = math.sqrt(q)
        if r < 0.2:
            return
        z_hat = np.array([r, math.atan2(dy, dx) - mu[2] - head])
        innov = np.array([d, b]) - z_hat
        innov[1] = math.atan2(math.sin(innov[1]), math.cos(innov[1]))
        n = len(mu)
        H = np.zeros((2, n))
        H[0, [0, 1, j, j + 1]] = [-dx / r, -dy / r, dx / r, dy / r]
        H[1, [0, 1, 2, j, j + 1]] = [dy / q, -dx / q, -1, -dy / q, dx / q]
        S = H @ P @ H.T + Rm
        if innov @ np.linalg.solve(S, innov) > 13.8:       # (a sighting that doesn't fit at all: ignore it)
            return
        K = P @ H.T @ np.linalg.inv(S)
        self.ekf_mu = mu + K @ innov
        self.ekf_cov = (np.eye(n) - K @ H) @ P
        self.ekf_cov = (self.ekf_cov + self.ekf_cov.T) / 2

    def _est_pose(self):
        return (self.est_heading(), self.est_x, self.est_y)

    def _corrected(self, before):
        """His idea of where he is was just corrected (landmark or scan matching) – remember the correction."""
        delta = _tf_mul(self._est_pose(), _tf_inv(before))
        self.corr = _tf_mul(delta, self.corr)
        self.stretch_corr[self.stretch] = self.corr

    def _fix_position(self, name, bearing, d):
        """He sees a landmark he knows: where it SHOULD be (his map) vs where he SEES it tells him how far his own
        idea of where he is has drifted – and he corrects it a little (each sighting a bit)."""
        yaw_est = self.est_heading() + self.head_yaw
        a = math.radians(yaw_est + bearing)
        seen_at = (self.est_x + d * math.cos(a), self.est_y + d * math.sin(a))
        lm = self.landmarks.get(name)
        if lm is None or lm["n"] < 4:   # the first few sightings: learn where the landmark is (an average)
            if lm is None:
                self.landmarks[name] = {"pos": seen_at, "n": 1}
            else:
                n = lm["n"]
                lm["pos"] = ((lm["pos"][0] * n + seen_at[0]) / (n + 1), (lm["pos"][1] * n + seen_at[1]) / (n + 1))
                lm["n"] = n + 1
            return
        L = lm["pos"]
        before = self._est_pose()
        expected = _wrap(_angle_to(L[0] - self.est_x, L[1] - self.est_y) - yaw_est)
        if math.hypot(L[0] - self.est_x, L[1] - self.est_y) > 0.5:
            self.drift += FIX_STRENGTH * HEADING_FIX * _wrap(expected - bearing)   # his sense of direction
        a = math.radians(self.est_heading() + self.head_yaw + bearing)
        seen_at = (self.est_x + d * math.cos(a), self.est_y + d * math.sin(a))
        self.est_x -= FIX_STRENGTH * (seen_at[0] - L[0])                     # and of where he stands
        self.est_y -= FIX_STRENGTH * (seen_at[1] - L[1])
        self.fixes += 1
        self._corrected(before)
        self.unsure = max(self.unsure, math.hypot(seen_at[0] - L[0], seen_at[1] - L[1]))

    def _update_places(self):
        """Where things are, in words ("by the bookshelf"), kept in his long-term memory between days. A thing seen
        somewhere new simply gets its new place."""
        if len(self.landmarks) < 2:
            return   # (too early – he doesn't know the room well enough yet to say where things are)
        votes = self.__dict__.setdefault("_place_votes", {})
        changed = False
        for name, m in self.mind_map.items():
            if m["kind"] not in ("thing", "flat") or "pos" not in m or time.time() - m["t"] > 2:
                continue
            near, dist = min(((lm, math.hypot(m["pos"][0] - v["pos"][0], m["pos"][1] - v["pos"][1]))
                              for lm, v in self.landmarks.items()), key=lambda x: x[1])
            where = (f"by the {near}" if dist < 0.9 else f"near the {near}" if dist < 1.8 else "in the middle of the room")
            old = places.get(name)
            new_near = near if dist < 1.8 else None
            if old and old.get("near") == new_near:
                where = old.get("where")   # same landmark – just "by" vs "near" wobbling: not a move
            if old and old.get("where") != where:   # seen somewhere else: only believed after ~2 s of seeing it there
                w, n = votes.get(name, (None, 0))
                votes[name] = (where, n + 1 if w == where else 1)
                if votes[name][1] < 10:
                    continue
            votes.pop(name, None)
            if not old or old.get("where") != where:
                with memory_lock:
                    places[name] = {"where": where, "near": near if dist < 1.8 else None, "date": date.today().isoformat()}
                changed = True
                if old and old.get("where") != where:
                    print(f"🗺️  ({NAME} noticed the {name} has moved: {old.get('where')} → {where})")
            elif old.get("date") != date.today().isoformat():
                with memory_lock:
                    old["date"] = date.today().isoformat()
                changed = True
        if changed:
            save_memory()

    def _put_down_here(self, name):
        """He put something down himself – so he knows where it is now, without having to notice it later: his
        mental map and his long-term places get the new spot at once."""
        rad = math.radians(self.est_heading())
        pos = (self.est_x + 0.2 * math.cos(rad), self.est_y + 0.2 * math.sin(rad))
        kind = self.mind_map.get(name, {}).get("kind", "thing")
        self.mind_map[name] = {"pos": pos, "t": time.time(), "kind": kind if kind in ("thing", "flat") else "thing"}
        self.__dict__.setdefault("_place_votes", {}).pop(name, None)
        if len(self.landmarks) < 2:
            return
        near, dist = min(((lm, math.hypot(pos[0] - v["pos"][0], pos[1] - v["pos"][1]))
                          for lm, v in self.landmarks.items()), key=lambda x: x[1])
        where = (f"by the {near}" if dist < 0.9 else f"near the {near}" if dist < 1.8 else "in the middle of the room")
        with memory_lock:
            places[name] = {"where": where, "near": near if dist < 1.8 else None, "date": date.today().isoformat()}
        save_memory()

    # ---------- his own floor plan (occupancy grid) ----------
    def _cell(self, x, y):
        return int((x - MAP_ORIGIN[0]) / MAP_CELL), int((y - MAP_ORIGIN[1]) / MAP_CELL)

    def _map_update(self):
        """Draws what the depth sensor and the camera see into his floor plan – from where he THINKS he is.
        Each of the depth sensor's 8 columns and 32 camera columns is a ray: free floor up to the first thing at his
        height, and something there. First the new look is fitted onto the map he already has (scan matching, like a
        robot vacuum): if it fits better a few cm / degrees off, his idea of where he stands is nudged that way.
        Things in the way that are people go into a layer that fades – people move."""
        from anka.body_view import TOF_RANGE, depth_sensor
        # only when he has moved or turned (head too) since the last time – or once a second when standing still.
        # (Drawing the same view in again and again lets the small wobble in his position wear the walls away.)
        pose = (self.est_x, self.est_y, self.est_heading() + self.head_yaw, self.pitch)
        last = self._last_drawn
        moved = last is None or math.hypot(pose[0] - last[0], pose[1] - last[1]) > MAP_EVERY_M \
            or abs(_wrap(pose[2] - last[2])) > MAP_EVERY_DEG or abs(pose[3] - last[3]) > MAP_EVERY_DEG
        if not moved and time.time() - last[4] < MAP_EVERY_S:
            return
        self._last_drawn = pose + (time.time(),)
        dist, floor = depth_sensor(self, self.x, self.y, self.heading + self.head_yaw, self.pitch)
        low = np.where(floor[3:, :], np.inf, dist[3:, :])          # the lower rows: what's in the way of a small duck
        rays = []      # (angle from where his head points, how far it's free, something there?, source)
        for j in range(8):
            d = float(low[:, j].min())
            hit = np.isfinite(d) and d < TOF_RANGE
            reach = min(d, TOF_RANGE) if hit else TOF_RANGE
            for sub in (-1 / 3, 0, 1 / 3):
                rays.append(((3.5 - j + sub) * 45 / 8, reach, hit, "tof"))
        # the camera too (the floor trick – wider and further than the depth sensor, but less exact: ±12 %)
        if not self.fallen:
            from anka.body_view import FOV, W as CW, render
            key = ("camdepth",) + self._key_now()
            if key not in self._cache:
                self._cache[key] = render(self, self.x, self.y, self.heading + self.head_yaw, self.pitch,
                                          with_depth=True)[2]
            cols = self._cache[key]
            f = (CW / 2) / math.tan(math.radians(FOV / 2))
            for c in range(4, CW, 10):              # 32 rays across the picture
                d = float(cols[c])
                hit = math.isfinite(d) and d < 5.0
                rays.append((math.degrees(math.atan((CW / 2 - c) / f)), (d * random.gauss(1, 0.12)) if hit else 5.0,
                             hit, "cam"))
        if EKF and not self.fallen and moved:
            self._wall_heading(rays)
            self._last_drawn = (self.est_x, self.est_y, self.est_heading() + self.head_yaw, self.pitch, time.time())
        if SCAN_MATCH and not self.fallen and moved:      # (standing still: nothing new to fit)
            self._scan_match(rays)
            self._last_drawn = (self.est_x, self.est_y, self.est_heading() + self.head_yaw, self.pitch, time.time())
        # which hits are people (they go into the fading layer, not into the map)
        yaw = self.est_heading() + self.head_yaw
        people = getattr(self, "_people_seen", [])
        kept, person_cells = [], []
        for ang, reach, hit, src in rays:
            if hit and FADE_PEOPLE and people:
                a = math.radians(yaw + ang)
                hx, hy = self.est_x + reach * math.cos(a), self.est_y + reach * math.sin(a)
                if any(math.hypot(hx - px, hy - py) < 0.35 for px, py in people):
                    person_cells.append(self._cell(hx, hy))
                    hit = False
                    reach = max(0.0, reach - 0.15)
            kept.append((ang, reach, 1.0 if hit else 0.0, 0.0 if src == "tof" else 1.0))
        rays_np = np.array(kept, np.float32).reshape(-1, 4)
        # keep the look, with where he was by step counting (so it can be moved later), and draw it in now
        odom = _tf_mul(_tf_inv(self.corr), self._est_pose())
        self._next_stretch(odom)
        self.looks.append((self.stretch, odom, self.head_yaw, rays_np))
        self._look_truth.append(((self.heading, self.x, self.y), self._est_pose()))   # (simulation only: to measure)
        self.stretch_corr[self.stretch] = self.corr
        look_hit, look_free = np.zeros(MAP_SIZE, np.float32), np.zeros(MAP_SIZE, np.float32)
        self._draw_look(look_hit, look_free, self._est_pose(), self.head_yaw, rays_np)
        self.map_hit += look_hit
        self.map_free += look_free
        W, H = MAP_SIZE
        # the depth AI's last guess (worked out in the background), and maybe start a new one
        ai_free, ai_hit = self._depth_ai_cells()
        for (i, k), step, layer in [(c, -0.15, self.map_free) for c in set(ai_free)] + [(c, 0.4, self.map_hit) for c in set(ai_hit)]:
            if 0 <= i < W and 0 <= k < H:
                layer[i, k] += step
        if not self.fallen and not self._da["busy"] and time.time() - self._da["last"] > 0.5:
            from anka.body_view import _depth_ai
            from settings import DEPTH_AI
            if DEPTH_AI and not _depth_ai["failed"]:
                self._da["busy"] = True
                self._da["near_person"] = any(v["pixels"] > 2000 for k, v in self.look()[1].items() if k in self.people)
                snapshot = (self.look()[0], dist, floor, self.pitch, (self.est_x, self.est_y, yaw),
                            self._cache.get(("camdepth",) + self._key_now()))
                threading.Thread(target=self._depth_ai_work, args=snapshot, daemon=True).start()
        # people fade (half gone after ~15 looks); seeing free floor there clears them at once
        self.people_floor *= 0.95
        for i, k in set(person_cells):
            if 0 <= i < W and 0 <= k < H:
                self.people_floor[i, k] = min(4.0, self.people_floor[i, k] + 0.9)
        self.people_floor[look_free < -0.1] = 0.0          # (he sees free floor where a person was)
        if getattr(self, "_redraw_due", False) and time.time() - getattr(self, "_redrawn_at", 0.0) > 2.0:
            self._redraw_due, self._redrawn_at = False, time.time()   # (at most every 2 s – it takes a moment)
            self._redraw()
        else:
            self._combine()

    def _draw_look(self, hit_map, free_map, pose, head, rays):
        """One look into the evidence maps: each ray shows free floor (less sure further away; a ray that hits
        nothing only up to FREE_NO_HIT) and, if it hit something, something there. Where he stands is free."""
        if not len(rays):
            return
        W, H = MAP_SIZE
        ang, reach, hit, cam = rays[:, 0], rays[:, 1], rays[:, 2] > 0, rays[:, 3] > 0
        a = np.radians(pose[0] + head + ang)
        ca, sa = np.cos(a), np.sin(a)
        far = np.where(hit, reach - np.where(cam, 0.1, 0.06), np.minimum(reach, FREE_NO_HIT))
        steps = np.arange(1, int(5.0 / MAP_CELL) + 1, dtype=np.float32) * MAP_CELL
        d = steps[None, :]
        ok = d <= far[:, None]
        X = pose[1] + d * ca[:, None]
        Y = pose[2] + d * sa[:, None]
        w = np.where(cam, -0.12, -0.35)[:, None] * np.maximum(FREE_MIN, 1.0 - d / FREE_FADE)
        I = ((X - MAP_ORIGIN[0]) / MAP_CELL).astype(np.int32)
        K = ((Y - MAP_ORIGIN[1]) / MAP_CELL).astype(np.int32)
        ok &= (I >= 0) & (I < W) & (K >= 0) & (K < H)
        np.add.at(free_map, (I[ok], K[ok]), np.broadcast_to(w, ok.shape)[ok])
        if hit.any():
            hx = pose[1] + reach[hit] * ca[hit]
            hy = pose[2] + reach[hit] * sa[hit]
            hi = ((hx - MAP_ORIGIN[0]) / MAP_CELL).astype(np.int32)
            hk = ((hy - MAP_ORIGIN[1]) / MAP_CELL).astype(np.int32)
            inside = (hi >= 0) & (hi < W) & (hk >= 0) & (hk < H)
            np.add.at(hit_map, (hi[inside], hk[inside]), np.where(cam[hit], CAM_HIT, 0.9)[inside])
        i, k = int((pose[1] - MAP_ORIGIN[0]) / MAP_CELL), int((pose[2] - MAP_ORIGIN[1]) / MAP_CELL)
        free_map[max(0, i - 2):i + 3, max(0, k - 2):k + 3] -= 1.0        # where he stands is free

    def _combine(self):
        """The floor plan from the evidence: something there minus the free looks – a square seen as something
        many times is hard to erase (HARD_WALL), so a few slightly-off looks can't wear a wall away."""
        hit = np.minimum(self.base_hit + self.map_hit, 6.0)
        free = -(self.base_free + self.map_free)
        self.floor = np.clip(hit - free * np.where(hit >= HARD_WALL, HARD_ERASE, 1.0), -4.0, 4.0).astype(np.float32)
        if self._wall_mask is not None:
            self.floor[self._wall_mask] = 4.0          # fixed walls are always there
        if time.time() - self._walls_checked > WALL_EVERY:
            self._walls_checked = time.time()
            self._find_walls()

    def _find_walls(self):
        """Walls never move. Long straight runs in his floor plan are candidates; one found again and again (each
        time a few seconds apart) becomes a fixed wall: one line, kept for good – later looks can make it longer,
        but never move or remove it. Lines close to the room's main direction are squared up to it."""
        occ = (self.floor > 0.6).astype(np.uint8)
        if self._wall_mask is not None:
            occ[cv2.dilate(self._wall_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 0
        closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        found = cv2.HoughLinesP(closed * 255, 1, np.pi / 180, threshold=12,
                                minLineLength=int(WALL_MIN_LEN / MAP_CELL), maxLineGap=4)
        if found is None:
            return
        for k_a, i_a, k_b, i_b in found.reshape(-1, 4):      # (the picture's x is the map's 2nd index)
            a = (MAP_ORIGIN[0] + (i_a + 0.5) * MAP_CELL, MAP_ORIGIN[1] + (k_a + 0.5) * MAP_CELL)
            b = (MAP_ORIGIN[0] + (i_b + 0.5) * MAP_CELL, MAP_ORIGIN[1] + (k_b + 0.5) * MAP_CELL)
            self._wall_seen(a, b)
        self._wall_mask_update()

    @staticmethod
    def _line_of(a, b, angle=None):
        """(angle 0–180°, distance of the line from the origin, where along the line a and b are)"""
        ang = (math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180) if angle is None else angle
        ux, uy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        rho = -uy * (a[0] + b[0]) / 2 + ux * (a[1] + b[1]) / 2
        ta, tb = ux * a[0] + uy * a[1], ux * b[0] + uy * b[1]
        return ang, rho, min(ta, tb), max(ta, tb)

    def _on_furniture(self, a, b):
        """Is this straight edge part of something he has recognised as furniture (sofa, table, bookshelf …)?
        Furniture can be moved, so it never becomes a fixed wall – it stays ordinary 'something there'.
        (He knows from his eyes – on the real duck: YOLO – where he saw the furniture's edges.)"""
        pts = [p for name, m in self.mind_map.items() if m.get("kind") == "furniture" and name not in WALL_THINGS
               for p in m.get("points", [])]
        pts += [lm["pos"] for name, lm in self.landmarks.items() if name in FURNITURE]
        if not pts:
            return False
        P = np.array(pts)
        near = 0
        for f_ in np.linspace(0, 1, 9):
            x, y = a[0] + f_ * (b[0] - a[0]), a[1] + f_ * (b[1] - a[1])
            near += np.min(np.hypot(P[:, 0] - x, P[:, 1] - y)) < FURNITURE_NEAR
        return near >= 3                               # (a third of it lies on furniture he saw)

    def _free_behind(self, a, b):
        """A real wall has free floor on ONE side only (the room) – he can never see floor behind it. A straight
        line with free floor seen on both sides is something standing in the room, not a wall."""
        ux, uy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(ux, uy)
        if n < 1e-6:
            return False
        nx, ny = -uy / n, ux / n
        W, H = MAP_SIZE
        both = 0
        for f_ in np.linspace(0.1, 0.9, 9):
            px, py = a[0] + f_ * ux, a[1] + f_ * uy
            sides = []
            for sign in (1, -1):
                free = False
                for d in (0.15, 0.3, 0.45, 0.6):
                    i, k = self._cell(px + sign * d * nx, py + sign * d * ny)
                    if 0 <= i < W and 0 <= k < H and self.floor[i, k] < -0.5:
                        free = True
                        break
                sides.append(free)
            both += sides[0] and sides[1]
        return both >= 3

    def _wall_seen(self, a, b):
        if self._on_furniture(a, b) or self._free_behind(a, b):
            return
        ang, rho, t0, t1 = self._line_of(a, b)
        if self.wall_ref is None or len(getattr(self, "_wall_first", [])) < ROOM_RUNS_FIX:
            return                                    # (he doesn't know the room's main direction well enough yet)
        off = ((ang - self.wall_ref + 45) % 90) - 45
        if abs(off) >= 6:                             # only walls square to the room are fixed – a slanted line is
            return                                    # more likely a mistake (it would stay wrong for good)
        ang, rho, t0, t1 = self._line_of(a, b, (ang - off) % 180)
        def same(w):
            dang = abs(((w["ang"] - ang + 90) % 180) - 90)
            if dang > 5:
                return False
            flip = 1 if abs(((w["ang"] - ang + 180) % 360) - 180) < 90 else -1
            rho_w = w["rho"] * flip
            lo, hi = (w["t0"], w["t1"]) if flip == 1 else (-w["t1"], -w["t0"])
            return abs(rho_w - rho) < 0.15 and t0 < hi + 0.3 and t1 > lo - 0.3
        for w in self.walls:                          # a fixed wall: it may only grow longer
            if same(w):
                if abs(((w["ang"] - ang + 180) % 360) - 180) < 90:
                    w["t0"], w["t1"] = min(w["t0"], t0), max(w["t1"], t1)
                return
        for c in self._wall_candidates:
            if same(c):
                if abs(((c["ang"] - ang + 180) % 360) - 180) < 90:
                    n = c["n"]
                    c["rho"] = (c["rho"] * n + rho) / (n + 1)
                    c["t0"], c["t1"] = min(c["t0"], t0), max(c["t1"], t1)
                c["n"] += 1
                if c["n"] >= WALL_SUPPORT and c["t1"] - c["t0"] >= WALL_MIN_LEN:
                    self._wall_candidates.remove(c)
                    self._wall_fix(c)
                return
        self._wall_candidates.append({"ang": ang, "rho": rho, "t0": t0, "t1": t1, "n": 1})

    def _wall_fix(self, c):
        """A candidate becomes a fixed wall (unless he has seen free floor behind it by now). If it lies in line with a fixed wall already (another part of the same
        wall – maybe with a door in between), it takes exactly that wall's line, so the room stays straight."""
        if self._free_behind(*self.wall_ends(c)):
            return
        if self.wall_ref is not None:                 # square it up (it may have been found before he knew the room)
            off = ((c["ang"] - self.wall_ref + 45) % 90) - 45
            if abs(off) < 6:
                a, b = self.wall_ends(c)
                ang, rho, t0, t1 = self._line_of(a, b, (c["ang"] - off) % 180)
                c = dict(c, ang=ang, rho=rho, t0=t0, t1=t1)
        for w in self.walls:
            same_dir = abs(((w["ang"] - c["ang"] + 180) % 360) - 180) < 3
            flipped = abs(((w["ang"] - c["ang"]) % 360) - 180) < 3
            if not (same_dir or flipped):
                continue
            rho, t0, t1 = (c["rho"], c["t0"], c["t1"]) if same_dir else (-c["rho"], -c["t1"], -c["t0"])
            if abs(rho - w["rho"]) < 0.25:
                if t0 < w["t1"] + 0.5 and t1 > w["t0"] - 0.5:          # touching: the same piece of wall, longer
                    w["t0"], w["t1"] = min(w["t0"], t0), max(w["t1"], t1)
                    return
                c = {"ang": w["ang"], "rho": w["rho"], "t0": t0, "t1": t1, "n": c["n"]}
                break
        self.walls.append(c)
        self._merge_walls()

    def _merge_walls(self):
        """Two fixed walls side by side (parallel, less than WALL_MERGE apart, overlapping) are the same wall seen
        twice: they become one line (in between, weighted by length). Done once, when a wall is fixed."""
        merged = True
        while merged:
            merged = False
            for i, w in enumerate(self.walls):
                for v in self.walls[i + 1:]:
                    same_dir = abs(((w["ang"] - v["ang"] + 180) % 360) - 180) < 2
                    flipped = abs(((w["ang"] - v["ang"]) % 360) - 180) < 2
                    if not (same_dir or flipped):
                        continue
                    rho, t0, t1 = (v["rho"], v["t0"], v["t1"]) if same_dir else (-v["rho"], -v["t1"], -v["t0"])
                    if abs(rho - w["rho"]) < WALL_MERGE and t0 < w["t1"] + 0.3 and t1 > w["t0"] - 0.3:
                        lw, lv = w["t1"] - w["t0"], t1 - t0
                        w["rho"] = (w["rho"] * lw + rho * lv) / (lw + lv)
                        w["t0"], w["t1"] = min(w["t0"], t0), max(w["t1"], t1)
                        self.walls.remove(v)
                        merged = True
                        break
                if merged:
                    break

    def wall_ends(self, w):
        ux, uy = math.cos(math.radians(w["ang"])), math.sin(math.radians(w["ang"]))
        return ((w["t0"] * ux - w["rho"] * uy, w["t0"] * uy + w["rho"] * ux),
                (w["t1"] * ux - w["rho"] * uy, w["t1"] * uy + w["rho"] * ux))

    def _wall_mask_update(self):
        if not self.walls:
            return
        mask = np.zeros((MAP_SIZE[1], MAP_SIZE[0]), np.uint8)       # (drawn as a picture: rows = 2nd index)
        for w in self.walls:
            a, b = self.wall_ends(w)
            ia, ka = self._cell(*a)
            ib, kb = self._cell(*b)
            cv2.line(mask, (ia, ka), (ib, kb), 1, 2)
        self._wall_mask = mask.T > 0

    def _next_stretch(self, odom):
        """A new stretch starts after SEGMENT_M of walking or SEGMENT_DEG of turning (by step counting). When a
        stretch ends, everything is redrawn with what he now knows about where he was during each stretch."""
        s0 = self.stretch_start
        if math.hypot(odom[1] - s0[1], odom[2] - s0[2]) < SEGMENT_M and abs(_wrap(odom[0] - s0[0])) < SEGMENT_DEG:
            return
        self.stretch_corr[self.stretch] = self.corr
        self.stretch += 1
        self.stretch_start = odom
        self.stretch_corr[self.stretch] = self.corr
        self._redraw_due = REDRAW

    def _redraw(self):
        """Draws the whole floor plan again: every look from where he now believes he was then."""
        if len(self.looks) > KEEP_LOOKS:            # the oldest looks are baked in for good
            old, self.looks = self.looks[:-KEEP_LOOKS], self.looks[-KEEP_LOOKS:]
            for st, odom, head, rays in old:
                self._draw_look(self.base_hit, self.base_free, _tf_mul(self.stretch_corr[st], odom), head, rays)
        hit = np.zeros(MAP_SIZE, np.float32)
        free = np.zeros(MAP_SIZE, np.float32)
        for st, odom, head, rays in self.looks:
            self._draw_look(hit, free, _tf_mul(self.stretch_corr.get(st, self.corr), odom), head, rays)
        for st, point in self.bumps:
            p = _tf_mul(self.stretch_corr.get(st, self.corr), (0.0,) + point)
            i, k = self._cell(p[1], p[2])
            if 0 <= i < MAP_SIZE[0] and 0 <= k < MAP_SIZE[1]:
                hit[i, k] += 4.0
        self.map_hit, self.map_free = hit, free
        self.redraws += 1
        self._combine()

    def _scan_match(self, rays):
        """Scan matching: try his position a few cm and degrees off and keep the one where the things he sees now
        land best on the walls and furniture already on his map. Only when the map around him is sure enough."""
        hits = [(ang, reach, 1.0 if src == "tof" else MATCH_CAM) for ang, reach, hit, src in rays if hit and reach < 3.5]
        hits = [h for h in hits if h[2] > 0]
        if len(hits) < 8:
            return
        if SCAN_ONLY_WHEN_LOST and EKF and getattr(self, "ekf_mu", None) is not None:
            self._ekf_predict()                      # (bring the filter up to date, then ask how sure it is)
            P = self.ekf_cov
            sd_pos = math.sqrt(max(P[0, 0] + P[1, 1], 0.0))
            sd_head = math.degrees(math.sqrt(max(P[2, 2], 0.0)))
            self._scan_sd = (sd_pos, sd_head)
            if sd_pos < SCAN_LOST_POS and sd_head < SCAN_LOST_HEAD:
                return
        # he fits the new look onto his map as it was a while ago (1–2 m of walking or a few turns back) – the
        # newest part of the map was drawn with the same error he has now, so it would always "fit"
        moved = self.walked + self._turn_total / 90.0
        if moved - self._match_age >= MATCH_LAG:
            self._match_map = None if self._match_next is None else cv2.GaussianBlur(self._match_next, (5, 5), 1.0)
            self._match_next = np.clip(self.floor, -1.5, 4.0)
            self._match_age = moved
        M = self._match_map
        if M is None or (M > 1.0).sum() < 40:      # (not enough on that map yet to fit anything onto)
            return
        ang = np.radians(np.array([h[0] for h in hits]))
        r = np.array([h[1] for h in hits])
        w = np.array([h[2] for h in hits])
        yaw0 = math.radians(self.est_heading() + self.head_yaw)
        dth = np.radians(np.arange(-MATCH_TURN, MATCH_TURN + 1e-6, 1.0))
        dxy = np.arange(-MATCH_SHIFT, MATCH_SHIFT + 1e-6, MAP_CELL / 2)
        DT, DX, DY = np.meshgrid(dth, dxy, dxy, indexing="ij")
        DT, DX, DY = DT.ravel(), DX.ravel(), DY.ravel()
        A = yaw0 + DT[:, None] + ang[None, :]
        X = self.est_x + DX[:, None] + r[None, :] * np.cos(A)
        Y = self.est_y + DY[:, None] + r[None, :] * np.sin(A)
        I = ((X - MAP_ORIGIN[0]) / MAP_CELL).astype(int)
        K = ((Y - MAP_ORIGIN[1]) / MAP_CELL).astype(int)
        inside = (I >= 0) & (I < MAP_SIZE[0]) & (K >= 0) & (K < MAP_SIZE[1])
        vals = np.where(inside, M[np.clip(I, 0, MAP_SIZE[0] - 1), np.clip(K, 0, MAP_SIZE[1] - 1)], 0.0)
        score = (vals * w[None, :]).sum(1) / w.sum()
        score -= MATCH_PRIOR * ((DX ** 2 + DY ** 2) / MATCH_SHIFT ** 2 + (DT / math.radians(MATCH_TURN)) ** 2)   # small moves first
        best = int(np.argmax(score))
        here = int(np.argmin(np.abs(DT) + np.abs(DX) + np.abs(DY)))
        self._last_match = (float(score[best] - score[here]), float(DX[best]), float(DY[best]), math.degrees(DT[best]))
        if score[best] - score[here] < MATCH_GAIN:
            return
        before = self._est_pose()
        self.est_x += MATCH_TRUST * DX[best]
        self.est_y += MATCH_TRUST * DY[best]
        self.drift += MATCH_TRUST * math.degrees(DT[best])
        self.scan_fixes += 1
        self._corrected(before)
        self._ekf_sync()

    def _depth_ai_work(self, picture, tof, tof_floor, pitch, pose, truth):
        """(background) The depth AI looks at the picture; its guess is scaled to metres with the depth sensor's
        64 exact distances, and turned into rays like the camera's: how far it is free, and what's in the way."""
        from anka.body_view import FOV, H as CH, W as CW, depth_ai
        try:
            disp = depth_ai(picture)
            if disp is None:
                return
            f = (CW / 2) / math.tan(math.radians(FOV / 2))
            yh = CH / 2 + f * math.tan(math.radians(pitch))
            xs, inv = [], []
            for i in range(8):
                for j in range(8):
                    d = tof[i, j]
                    if not np.isfinite(d):
                        continue
                    yo, po = math.radians((3.5 - j) * 45 / 8), math.radians(pitch + (3.5 - i) * 45 / 8)
                    px, py = CW / 2 - f * math.tan(yo), yh - f * math.tan(po)
                    half = max(1, int(f * math.tan(math.radians(45 / 16))))   # the zone's size in the picture
                    if half <= px < CW - half and half <= py < CH - half:
                        # the sensor reports the CLOSEST thing in its zone – so compare with the AI's closest there
                        xs.append(float(np.percentile(disp[int(py) - half:int(py) + half, int(px) - half:int(px) + half], 95)))
                        inv.append(1.0 / max(0.05, d * math.cos(po) * math.cos(yo)))
            if len(xs) < 6 or np.std(xs) < 1e-6:
                # the depth sensor sees (almost) nothing close – open floor ahead. Then the floor alone gives the
                # scale: the bottom of the picture, right in front of him, is floor (camera height + tilt known)
                rows_f = np.arange(int(max(yh + 8, CH * 0.8)), CH - 1, 2)
                cols_f = np.arange(int(CW * 0.25), int(CW * 0.75), 4)
                if not (DEPTH_AI_FLOOR and len(rows_f) >= 5):
                    return
                V, U = np.meshgrid(rows_f, cols_f, indexing="ij")
                xs = disp[V, U].ravel()
                inv = (V.ravel() - yh) / (f * CAM_H_)
            xs, inv = np.array(xs), np.array(inv)
            if np.std(xs) < 1e-6:
                return
            a, b = np.polyfit(xs, inv, 1)       # 1 / depth ≈ a · (the AI's number) + b
            res = np.abs(inv - (a * xs + b))
            keep = res <= max(2 * np.median(res), 1e-6) * 1.5
            if keep.sum() >= 6:
                a, b = np.polyfit(xs[keep], inv[keep], 1)   # once more without the ones that don't fit
            if a <= 0:
                return
            if DEPTH_AI_FLOOR and len(xs) < FLOOR_WHEN:   # the floor too: the camera's height and tilt (IMU) are known, so where the picture
                # shows floor, its true distance is known as well – that pins the AI's scale down beyond the sensor
                rows_f = np.arange(int(max(yh + 8, CH * 0.6)), CH - 1, 3)
                cols_f = np.arange(8, CW - 8, 6)
                if len(rows_f) and len(cols_f):
                    V, U = np.meshgrid(rows_f, cols_f, indexing="ij")
                    dv = disp[V, U].ravel()
                    zv = 1.0 / np.maximum(a * dv + b, 0.05)
                    hv = CAM_H_ + zv * (yh - V.ravel()) / f
                    fl = np.abs(hv) < 0.05                         # (looks like floor by the first fit)
                    if fl.sum() >= 20:
                        inv_floor = (V.ravel()[fl] - yh) / (f * CAM_H_)
                        X = np.concatenate([xs, dv[fl]])
                        Y = np.concatenate([inv, inv_floor])
                        Wt = np.concatenate([np.ones(len(xs)), np.ones(int(fl.sum())) * FLOOR_WEIGHT * len(xs) / max(1, fl.sum())])
                        a2, b2 = np.polyfit(X, Y, 1, w=np.sqrt(Wt))
                        if a2 > 0:
                            a, b = a2, b2
            # the heat picture for the window: near = bright, far = dark (metres, 0.1–4 m)
            zz = np.clip(1.0 / np.maximum(a * disp + b, 0.05), 0.1, 4.0)
            self.depth_ai_view = cv2.applyColorMap((255 * (1 - (zz - 0.1) / 3.9)).astype(np.uint8), cv2.COLORMAP_INFERNO)
            from settings import ROOM_3D
            if ROOM_3D and self.unsure < 0.25:  # (just corrected a lot? then he doesn't trust where he is)
                self._add_3d(picture, disp, a, b, tof, pitch, pose)
            rays = []
            rows = np.arange(CH, dtype=np.float32)
            for c in range(4, CW, 10):
                z = 1.0 / np.maximum(a * disp[:, c] + b, 0.05)          # metres, straight ahead
                height = CAM_H_ + z * (yh - rows) / f                    # how high each pixel's point is
                ok = (z > 0.05) & (z < 6)
                obstacle = ok & (height > 0.06) & (height < 0.4)      # (the AI is noisy: the floor wobbles a few cm)
                floor_pts = ok & (height <= 0.06)
                ang = math.atan((CW / 2 - c) / f)
                zs = np.sort(z[obstacle])
                hit = float(zs[min(5, len(zs) - 1)]) / math.cos(ang) if len(zs) >= 6 else None   # several pixels, not one
                reach = min(hit if hit else 5.0, float(z[floor_pts].max()) / math.cos(ang) if floor_pts.any() else 0.0, 5.0)
                if hit is not None and hit < 5.0:
                    reach = hit
                rays.append((math.degrees(ang), hit if hit is not None and hit < 5.0 else None, reach))
                if truth is not None and hit is not None and np.isfinite(truth[c]) and truth[c] < 5:
                    (self.depth_ai_errors_people if self._da.get("near_person") else self.depth_ai_errors).append(
                        abs(hit - truth[c]) / truth[c])
            self._da["result"] = (pose, rays)
            if len(self.depth_ai_errors) and not getattr(self, "_da_said", False):
                self._da_said = True
                print(f"🔭 (depth AI: first guesses are {np.median(self.depth_ai_errors) * 100:.0f} % off on average "
                      "compared to the real distances)")
        except Exception as error:
            print(f"🔭 (depth AI problem: {str(error)[:100]})")
        finally:
            self._da["last"] = time.time()
            self._da["busy"] = False

    def _add_3d(self, picture, disp, a, b, tof, pitch, pose):
        """Every depth guess becomes points in the room (from where he thinks he stands), dropped into 5 cm cubes
        with the colour of the pixel – slowly a 3D picture of the room grows. The depth sensor's 64 exact points
        count double."""
        from anka.body_view import FOV, H as CH, W as CW
        ex, ey, yaw = pose
        f = (CW / 2) / math.tan(math.radians(FOV / 2))
        yh = CH / 2 + f * math.tan(math.radians(pitch))
        rows, cols = np.mgrid[2:CH:4, 2:CW:4]
        z = 1.0 / np.maximum(a * disp[rows, cols] + b, 0.05)        # metres straight ahead (level)
        left = z * (CW / 2 - cols) / f
        up = CAM_H_ + z * (yh - rows) / f
        ok = (z > 0.08) & (z < 4.5)
        ya = math.radians(yaw)
        X = ex + z * math.cos(ya) - left * math.sin(ya)
        Y = ey + z * math.sin(ya) + left * math.cos(ya)
        colours = picture[rows, cols][..., ::-1].astype(np.float32)   # (BGR → RGB)
        self._vox_put(X[ok], Y[ok], up[ok], colours[ok], 1.0)
        # the depth sensor: exact, so each of its points counts double (no colour of its own: take the picture's)
        tx, ty, tz, tc = [], [], [], []
        for i in range(8):
            for j in range(8):
                d = tof[i, j]
                if not np.isfinite(d):
                    continue
                yo, po = math.radians((3.5 - j) * 45 / 8), math.radians(pitch + (3.5 - i) * 45 / 8)
                fwd, lf = d * math.cos(po) * math.cos(yo), d * math.cos(po) * math.sin(yo)
                tx.append(ex + fwd * math.cos(ya) - lf * math.sin(ya))
                ty.append(ey + fwd * math.sin(ya) + lf * math.cos(ya))
                tz.append(CAM_H_ + d * math.sin(po))
                px = int(min(CW - 1, max(0, CW / 2 - f * math.tan(yo))))
                py = int(min(CH - 1, max(0, yh - f * math.tan(po))))
                tc.append(picture[py, px][::-1])
        if tx:
            self._vox_put(np.array(tx), np.array(ty), np.array(tz), np.array(tc, np.float32), 2.0)

    def _vox_put(self, X, Y, Z, colours, weight):
        i = ((X - VOX_ORIGIN[0]) / VOX).astype(int)
        k = ((Y - VOX_ORIGIN[1]) / VOX).astype(int)
        h = ((Z - VOX_ORIGIN[2]) / VOX).astype(int)
        inside = (i >= 0) & (i < VOX_SIZE[0]) & (k >= 0) & (k < VOX_SIZE[1]) & (h >= 0) & (h < VOX_SIZE[2])
        i, k, h, colours = i[inside], k[inside], h[inside], colours[inside]
        np.add.at(self.vox_n, (i, k, h), weight)
        np.add.at(self.vox_rgb, (i, k, h), colours * weight)

    def cubes_3d(self, at_least=VOX_SHOW):
        """The cubes he is fairly sure about: [(x, y, height, r, g, b), ...] in metres (his own coordinates)."""
        idx = np.argwhere(self.vox_n >= at_least)
        if not len(idx):
            return np.zeros((0, 6), np.float32)
        n = self.vox_n[idx[:, 0], idx[:, 1], idx[:, 2]][:, None]
        rgb = self.vox_rgb[idx[:, 0], idx[:, 1], idx[:, 2]] / n
        xyz = np.array(VOX_ORIGIN) + (idx + 0.5) * VOX
        return np.hstack([xyz, rgb]).astype(np.float32)

    def _depth_ai_cells(self):
        res, self._da["result"] = self._da["result"], None
        free, hit = [], []
        if not res:
            return free, hit
        (ex, ey, yaw), rays = res
        for ang, d, reach in rays:
            a = math.radians(yaw + ang)
            ca, sa = math.cos(a), math.sin(a)
            for k in range(int(max(0.0, reach - 0.1) / MAP_CELL)):
                free.append(self._cell(ex + (k + 1) * MAP_CELL * ca, ey + (k + 1) * MAP_CELL * sa))
            if d is not None:
                hit.append(self._cell(ex + d * ca, ey + d * sa))
        return free, hit

    def _mark_bump(self, back=False, person=False, turn=0.0):
        """A bump means something is right there, even if the depth sensor never saw it. A PERSON he bumped into
        goes in the layer that fades (people walk away) – in test 22 Robin's spot stayed a 'wall' after she left and
        he was stopped by it 6 times in a row."""
        a = math.radians(self.est_heading() + turn + (180 if back else 0))   # (turn = ±90 for a bump while stepping sideways)
        x, y = self.est_x + 0.17 * math.cos(a), self.est_y + 0.17 * math.sin(a)
        i, k = self._cell(x, y)
        if person:
            if 0 <= i < MAP_SIZE[0] and 0 <= k < MAP_SIZE[1]:
                self.people_floor[i, k] = min(4.0, self.people_floor[i, k] + 2.0)
            return
        if 0 <= i < MAP_SIZE[0] and 0 <= k < MAP_SIZE[1]:
            self.map_hit[i, k] += 4.0
            self._combine()
        p = _tf_mul(_tf_inv(self.corr), (0.0, x, y))      # (kept by step counting too, so it moves with the rest)
        self.bumps.append((self.stretch, (p[1], p[2])))

    def _blocked_grid(self, margin=0.03):
        """Squares he believes are taken – grown by his own size, so a route through free squares fits him."""
        blocked = ((self.floor > 0.6) | (self.people_floor > 0.6)).astype(np.uint8)
        r = int(math.ceil((DUCK_RADIUS + margin) / MAP_CELL))
        return cv2.dilate(blocked, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0

    def _plan(self, goal, stop=0.3):
        """A route on HIS OWN floor plan (unknown squares count as free – he hasn't seen anything there).
        Returns way points in his own coordinates, or None. First tries a route that keeps some room to spare
        (ROUTE_ROOM), so the legs' reflex doesn't stop him all the time beside walls; if there is none, a tight one."""
        if ROUTE_ROOM > 0.03:
            wide = self._plan_with(goal, stop, ROUTE_ROOM)
            if wide:
                return wide
        return self._plan_with(goal, stop, 0.03)

    def _plan_with(self, goal, stop, margin):
        blocked = self._blocked_grid(margin)
        W, H = MAP_SIZE
        start, g = self._cell(self.est_x, self.est_y), self._cell(*goal)
        if not (0 <= g[0] < W and 0 <= g[1] < H):
            return None
        rstop = stop / MAP_CELL
        prev, queue, end = {start: None}, deque([start]), None
        while queue:
            c = queue.popleft()
            if (c[0] - g[0]) ** 2 + (c[1] - g[1]) ** 2 <= rstop ** 2:
                end = c
                break
            for di, dk in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nb = (c[0] + di, c[1] + dk)
                if nb not in prev and 0 <= nb[0] < W and 0 <= nb[1] < H and (not blocked[nb] or nb == start):
                    prev[nb] = c
                    queue.append(nb)
        if end is None:
            return None
        cells = []
        while end is not None:
            cells.append(end)
            end = prev[end]
        cells.reverse()
        centre = lambda c: (MAP_ORIGIN[0] + (c[0] + 0.5) * MAP_CELL, MAP_ORIGIN[1] + (c[1] + 0.5) * MAP_CELL)
        clear = lambda a, b: all(not blocked[(int(a[0] + (b[0] - a[0]) * t / 20), int(a[1] + (b[1] - a[1]) * t / 20))]
                                 for t in range(21))
        route, k = [], 0
        while k < len(cells) - 1:     # straight lines where his map allows
            far = len(cells) - 1
            while far > k + 1 and not clear(cells[k], cells[far]):
                far -= 1
            route.append(centre(cells[far]))
            k = far
        return route or [goal]

    def _walk_route(self, goal, watch=None, stop=0.3):
        """Walks to a spot on his own map: plans a route around what he knows is in the way, follows it by his
        own sense of where he is, and plans again if something new blocks him. Returns (metres, why it stopped)."""
        total, dodged, why = 0.0, 0, None
        for attempt in range(3):
            route = self._plan(goal, stop) or [goal]
            for wx, wy in route:
                d, rel = self._from_me((wx, wy))
                true_turn = rel + random.gauss(0, 3)
                self.heading = (self.heading + true_turn) % 360
                if not GYRO:
                    self.drift += rel - true_turn   # (without a gyro he believes he turned as much as he asked)
                self._turned(rel)
                self.head_yaw = 0.0
                self._key()
                last = (wx, wy) == route[-1]
                self.no_dodge = not ROUTE_DODGE
                try:
                    moved, why = self._step_walk(d + 0.3, aim_est=(wx, wy), watch=watch, arrive=0.35 if last else 0.1)
                finally:
                    self.no_dodge = False
                total += moved
                dodged += self.dodged
                self._map_update()
                if why == "arrived":
                    why = None
                    continue
                break
            if why in (None, "seen"):
                break
            if why == "reflex" or why:      # something new in the way: it's on his map now – plan again
                if why != "reflex":
                    self._mark_bump(person=why in self.people)
                self.replans += 1
        self.dodged = dodged
        return total, why or "arrived"

    def frontier(self):
        """The nearest edge of what he has explored: free floor next to squares he has never seen.
        Returns (distance, angle) from where he thinks he is, or None."""
        known_free = self.floor < -0.5
        unknown = np.abs(self.floor) < 0.05
        edge = known_free & (np.roll(unknown, 1, 0) | np.roll(unknown, -1, 0) | np.roll(unknown, 1, 1) | np.roll(unknown, -1, 1))
        edge[:3, :] = edge[-3:, :] = edge[:, :3] = edge[:, -3:] = False
        idx = np.argwhere(edge)
        if not len(idx):
            return None
        xs = MAP_ORIGIN[0] + (idx[:, 0] + 0.5) * MAP_CELL
        ys = MAP_ORIGIN[1] + (idx[:, 1] + 0.5) * MAP_CELL
        dd = np.hypot(xs - self.est_x, ys - self.est_y)
        far_enough = dd > 1.0
        if not far_enough.any():
            return None
        i = int(np.argmin(np.where(far_enough, dd, np.inf)))
        return (float(xs[i]), float(ys[i])), float(dd[i]), _wrap(_angle_to(xs[i] - self.est_x, ys[i] - self.est_y) - self.est_heading())

    def floor_words(self):
        explored = float((self.floor < -0.5).sum()) * MAP_CELL ** 2
        f = self.frontier()
        return (f"you have explored about {explored:.0f} m² of floor"
                + (f"; the nearest place you haven't explored is {_about(f[1])} {_side(f[2])}" if f
                   else "; you don't know of any unexplored places nearby"))

    def remember_view(self, things=True):
        with self.lock:
            self._remember_view(things)

    def _remember_view(self, things=True):
        """What the camera sees goes into his mental map, at a rough distance (one camera can't measure distance
        well) and seen from where he THINKS he is. Called while he looks around (like a hippocampus, automatic).
        Landmarks he knows correct where he thinks he is first."""
        now = time.time()
        self.unsure = 0.0
        yaw_true = self.heading + self.head_yaw
        seen = self.detect()
        self._localise([x for x in seen if x[1] == "landmark"])
        yaw_est = self.est_heading() + self.head_yaw
        self._people_seen = [(self.est_x + d * math.cos(math.radians(yaw_est + bearing)),
                              self.est_y + d * math.sin(math.radians(yaw_est + bearing)))
                             for name, kind, bearing, d in seen if kind == "person"]
        found = {}
        steady = self.unsure < 0.25     # (just corrected a lot? then don't trust where he thinks he is yet)
        for name, kind, bearing, d in seen:
            if kind != "landmark" and things and (steady or kind == "person"):
                found[name] = ((bearing, d), kind)
        for name in WALL_THINGS:
            px, py = self._nearest_point(name)
            rel = _wrap(_angle_to(px - self.x, py - self.y) - yaw_true)
            if abs(rel) <= 30 and math.hypot(px - self.x, py - self.y) < 4.5:
                found[name] = ((rel + random.gauss(0, 2), math.hypot(px - self.x, py - self.y) * random.gauss(1, 0.08)),
                               "furniture")
        # big furniture: the parts of its edge he can see (a sofa is 2 m long – one point isn't enough)
        for name, (x1, y1, x2, y2, _) in FURNITURE.items():
            near_d = math.hypot(*(a - b for a, b in zip(self._nearest_point(name), (self.x, self.y))))
            edge = []
            for k in range(int((x2 - x1) / 0.5) + 1):
                edge += [(x1 + k * 0.5, y1), (x1 + k * 0.5, y2)]
            for k in range(int((y2 - y1) / 0.5) + 1):
                edge += [(x1, y1 + k * 0.5), (x2, y1 + k * 0.5)]
            for px, py in edge:
                d = math.hypot(px - self.x, py - self.y)
                rel = _wrap(_angle_to(px - self.x, py - self.y) - yaw_true)
                if abs(rel) <= 30 and d < 4.5 and d <= near_d + 0.7:   # (the near side, not the hidden far side)
                    dd = d * random.uniform(0.85, 1.15)
                    a = math.radians(yaw_est + rel + random.gauss(0, 3))
                    guess = (self.est_x + dd * math.cos(a), self.est_y + dd * math.sin(a))
                    m = self.mind_map.setdefault(name, {"points": [], "t": now, "kind": "furniture"})
                    m.setdefault("points", [])
                    m["t"] = now
                    close = [i for i, q in enumerate(m["points"]) if math.hypot(q[0] - guess[0], q[1] - guess[1]) < 0.4]
                    if close:
                        q = m["points"][close[0]]
                        m["points"][close[0]] = (q[0] * 0.7 + guess[0] * 0.3, q[1] * 0.7 + guess[1] * 0.3)
                    else:
                        m["points"] = (m["points"] + [guess])[-12:]
        for name, ((rel, d), kind) in found.items():
            a = math.radians(yaw_est + rel)
            guess = (self.est_x + d * math.cos(a), self.est_y + d * math.sin(a))
            old = self.mind_map.get(name)
            if old and now - old["t"] < 120 and kind != "person" and "pos" in old:
                guess = (old["pos"][0] * 0.7 + guess[0] * 0.3, old["pos"][1] * 0.7 + guess[1] * 0.3)
            self.mind_map[name] = {"pos": guess, "t": now, "kind": kind}
        # people he remembers at a spot he now looks at – not in the picture, and nothing in front hiding them: not there any more
        # (a fact for his map – in test 24 it still said \"Robin about 2 m away\" 5 min after she had left)
        if steady and not self.fallen:
            dist, floor = self.depth()
            low = np.where(floor[3:, :], np.inf, dist[3:, :]).min(axis=0)
            for name, m in self.mind_map.items():
                if m.get("kind") != "person" or name in found or m.get("gone_at") or now - m["t"] < 5:
                    continue
                d, rel = self._from_me(m["pos"])
                rel -= self.head_yaw
                if 0.5 < d < 4.0 and abs(rel) <= 20:
                    # (the camera would have seen them there; the sensor isn't blocked before that spot, so nothing
                    #  hides them – people usually stand by furniture, so what's just behind them doesn't matter)
                    j = min(7, max(0, int(round(3.5 - rel * 8 / 45))))
                    if low[j] > d - 0.3:
                        m["gone_at"] = now
        self._update_places()
        self._map_update()

    def hear(self, name):
        """He heard someone talk: his map gets a rough guess of where they are (direction ±15°, distance from how
        loud – very rough). Only if he can hear directions."""
        if not HEAR_DIRECTION or name not in self.people:
            return
        px, py = self.people[name]
        d = math.hypot(px - self.x, py - self.y) * random.uniform(0.7, 1.3)
        rel = _wrap(_angle_to(px - self.x, py - self.y) - self.heading) + random.gauss(0, 15)
        a = math.radians(self.est_heading() + rel)
        self.mind_map[name] = {"pos": (self.est_x + d * math.cos(a), self.est_y + d * math.sin(a)),
                               "t": time.time(), "kind": "person", "heard": True}

    def _mm_aim(self, name):
        """Which way something is, for turning towards it: for big furniture its nearest remembered part – but when he
        is right next to it (or under the table) its MIDDLE, else the direction swings wildly (2026-11-09: under the
        table "turn to the table" pointed him 144° wrong, and the table never showed as "ahead")."""
        m = self.mind_map[name]
        pts = m.get("points")
        if pts:
            near = min(pts, key=lambda q: math.hypot(q[0] - self.est_x, q[1] - self.est_y))
            if math.hypot(near[0] - self.est_x, near[1] - self.est_y) < 0.6:
                return (sum(q[0] for q in pts) / len(pts), sum(q[1] for q in pts) / len(pts))
            return near
        return m["pos"]

    def under_what(self):
        """Is he standing UNDER furniture (the table)? A fact for his body text."""
        for k, v in FURNITURE.items():
            x1, y1, x2, y2 = v[:4]
            if k == "table" and x1 < self.x < x2 and y1 < self.y < y2:
                return k
        return None

    def _mm_pos(self, name):
        """Where he thinks something is – for big furniture, its closest remembered part."""
        m = self.mind_map[name]
        if m.get("points"):
            return min(m["points"], key=lambda q: math.hypot(q[0] - self.est_x, q[1] - self.est_y))
        return m["pos"]

    def note_place(self, text):
        text = text.strip().strip('"').strip()
        if text and not re.fullmatch(r"(?i)(none|no|nothing)\W*", text):
            here = (self.est_x, self.est_y)
            # a note close to an earlier one replaces it (one note per spot – not the same note again and again)
            near = [i for i, (pos, _, _) in enumerate(self.place_notes) if math.hypot(pos[0] - here[0], pos[1] - here[1]) < 0.5]
            if near:
                old = self.place_notes[near[-1]][1]
                self.place_notes[near[-1]] = (here, text[:80], time.time())
                if old.lower() != text[:80].lower():
                    print(f"📍 ({NAME} changed his note about this place: {text[:80]})")
                return
            self.place_notes = (self.place_notes + [(here, text[:80], time.time())])[-12:]
            print(f"📍 ({NAME} noted about this place: {text[:80]})")

    def _from_me(self, pos):
        """Where a remembered spot is, seen from where he thinks he is now: (distance, angle relative to his body)."""
        dx, dy = pos[0] - self.est_x, pos[1] - self.est_y
        return math.hypot(dx, dy), _wrap(_angle_to(dx, dy) - self.est_heading())

    def _memory_ahead(self):
        """After a turn: which things does HIS OWN MAP say are now about straight ahead (within 25°)? A fact only – in
        test 22 he turned around 4 times in a row because he couldn't tell whether he now faced the guitar."""
        out = []
        for name, m in self.mind_map.items():
            d, rel = self._from_me(self._mm_aim(name))
            if (d > 0.4 or (m.get("points") and d > 0.1)) and abs(rel) <= 25:
                label = name if m["kind"] == "person" else "the " + name
                when = ""
                if m["kind"] == "person":   # people move – say how old the sighting is (test 24: \"Robin about 2 m away\" 5 min after she left)
                    s_ = time.time() - m["t"]
                    when = "" if s_ < 20 else (f", {'heard' if m.get('heard') else 'seen'} there "
                                               + (f"{int(s_ // 60)} min ago" if s_ >= 60 else f"{int(s_)} s ago"))
                    if m.get("gone_at"):
                        when += ", not there when you looked that way again"
                out.append((abs(rel), f"{label} ({_about(d)} away" + (f", {abs(rel):.0f}° to the {'left' if rel > 0 else 'right'}"
                                                                      if abs(rel) > 5 else "") + when + ")"))
        out.sort()
        if not out:
            return " – nothing on your mental map is roughly ahead of you now"
        return " – roughly ahead of you now, by your mental map: " + ", ".join(t for _, t in out[:3])

    def map_words(self):
        now = time.time()
        items = sorted(self.mind_map.items(), key=lambda kv: -kv[1]["t"])[:10]
        def ago_(t):
            s = now - t
            return "just now" if s < 20 else f"{int(s // 60)} min ago" if s >= 60 else f"{int(s)} s ago"
        parts = []
        for name, m in items:
            d, rel = self._from_me(self._mm_pos(name))
            label = name if m["kind"] == "person" else "the " + name
            if m.get("points") and d < 0.6:      # right next to big furniture: say which way its middle is
                dm, rel = self._from_me(self._mm_aim(name))
                where = "right next to you – its middle is " + ("about where you stand" if dm < 0.2 else
                                                                 f"{_about(dm)} {_side(rel)}")
            else:
                where = None
            where = where or ("right here" if d < 0.3 else f"{_about(d)} {_side(rel)}") + (
                f" [{abs(rel):.0f}° to your {'left' if rel > 0 else 'right'}]" if abs(rel) > 12 else "")
            parts.append(f"{label} – {where} ({'heard' if m.get('heard') else 'seen'} {ago_(m['t'])}"
                         + (", not there when you looked that way again" if m.get("gone_at") else "") + ")")
        notes = []
        for pos, text, t in self.place_notes[-4:]:
            d, rel = self._from_me(pos)
            notes.append(f'"{text}" – {"right here" if d < 0.3 else _about(d) + " " + _side(rel)}')   # (notes: words only)
        return parts, notes

    # ---------- the room, as the senses see it ----------
    def room_size(self):
        return ROOM_W, ROOM_H

    def solid_boxes(self):
        x1, y1, x2, y2 = FURNITURE["table"][:4]
        wood = (80, 125, 175)
        legs = [(x, y, x + LEG, y + LEG, 0.0, TABLE_TOP, wood, "table")
                for x in (x1, x2 - LEG) for y in (y1, y2 - LEG)]
        return [(0.3, 0.0, 2.3, 0.9, 0.0, 0.45, (70, 80, 160), "sofa"),
                (0.3, 0.0, 2.3, 0.2, 0.0, 0.85, (60, 70, 145), "sofa"),
                (x1, y1, x2, y2, TABLE_TOP, TABLE_TOP + 0.05, wood, "table"),
                *legs,
                (4.6, 2.4, 5.0, 4.0, 0.0, 1.8, (70, 100, 140), "bookshelf")]

    def sprites(self):
        look = {"ball": ("ball", 0.22, 0.0, 0.22, (40, 40, 215), "thing"),
                "sock": ("box", 0.18, 0.0, 0.05, (175, 135, 110), "flat"),
                "guitar": ("guitar", 0.36, 0.0, 1.0, (40, 95, 165), "thing"),
                "cup": ("box", 0.08, TABLE_TOP + 0.05, TABLE_TOP + 0.15, (235, 235, 235), "thing")}
        out = []
        for name, t in self.things.items():
            if name == self.holding or name not in look:
                continue
            shape, w, z0, z1, col, kind = look[name]
            lift = ON_TOP.get(t.get("on"), 0.0) if t.get("on") != "table" else 0.0
            if lift:                      # put up on furniture (e.g. the sock on the sofa seat) – drawn up there
                z0, z1, kind = z0 + lift, max(z1, 0.07) + lift, "thing"
            out.append({"name": name, "x": t["pos"][0], "y": t["pos"][1], "shape": shape, "w": w, "z0": z0,
                        "z1": z1, "color": col, "kind": kind})
        for name, (x, y) in self.people.items():
            out.append({"name": name, "x": x, "y": y, "shape": "person", "w": 0.42, "z0": 0.0, "z1": 1.75,
                        "color": PERSON_COLOURS.get(name, (140, 60, 140)), "kind": "person",
                        "hair": (20, 30, 40) if name == "Robin" else (40, 60, 90),
                        "trousers": (40, 40, 45) if name == "Robin" else (140, 95, 55)})
        return out

    def _blocked(self, x, y):
        r = DUCK_RADIUS
        if x < r or y < r or x > ROOM_W - r or y > ROOM_H - r:
            return "the wall"
        for (x1, y1, x2, y2, z0, _, _, name) in self.solid_boxes():
            if z0 < 0.25 and x1 - r < x < x2 + r and y1 - r < y < y2 + r:   # (he fits under the table top)
                return f"the {name}" + (" leg" if name == "table" else "")
        for name, (px, py) in self.people.items():
            if math.hypot(px - x, py - y) < 0.3:
                return name
        for name, t in self.things.items():
            if name != self.holding and not t.get("on") and not t["small"] and math.hypot(t["pos"][0] - x, t["pos"][1] - y) < 0.25:
                return f"the {name}"
        return None

    # ---------- his senses ----------
    def _key_now(self, x=None, y=None, yaw=None):
        x = self.x if x is None else x
        y = self.y if y is None else y
        yaw = self.heading + self.head_yaw if yaw is None else yaw
        world = (tuple((k, round(v["pos"][0], 2), round(v["pos"][1], 2)) for k, v in self.things.items()),
                 tuple(sorted(self.people.items())), self.holding)
        return round(x, 3), round(y, 3), round(yaw, 1), round(self.pitch, 1), self.fallen, world

    def look(self):
        """(camera picture, what's in it) from where he really is – cached, it's asked for often."""
        key = ("look",) + self._key_now()
        if key not in self._cache:
            from anka.body_view import render
            if len(self._cache) > 40:
                self._cache.clear()
            self._cache[key] = render(self, self.x, self.y, self.heading + self.head_yaw, self.pitch, self.fallen)
        return self._cache[key]

    def depth(self):
        key = ("depth",) + self._key_now()
        if key not in self._cache:
            from anka.body_view import depth_sensor
            self._cache[key] = depth_sensor(self, self.x, self.y, self.heading + self.head_yaw, self.pitch)
        return self._cache[key]

    def camera_picture(self):
        with self.lock:
            return self._camera_picture()

    def _camera_picture(self):
        """The picture the brain gets: normally what the camera sees; after 'look around' three views."""
        if self.panorama_next:
            from anka.body_view import panorama
            self.panorama_next = False
            pic, seen = panorama(self, self.x, self.y, self.heading + self.head_yaw, self.pitch)
            self.pano_seen = {k: v for k, v in seen.items() if v["pixels"] >= 15}
            keep = self.head_yaw
            for turn in (60, -60):   # the depth sensor sits in his head – it saw left and right too
                self.head_yaw = keep + turn
                self._map_update()
            self.head_yaw = keep
            return pic
        return self.look()[0]

    def seen_names(self, min_pixels=30):
        return {k: v for k, v in self.look()[1].items() if v["pixels"] >= min_pixels}

    def battery(self):
        return max(1, int(100 - (time.time() - self.started) / 60 * BATTERY_PER_MIN - self.walked * 0.6))

    def describe(self, last=None):
        from anka.body_view import depth_words
        if self.fallen:
            posture = "lying on your side – you fell over"
        else:
            posture = "sitting" if self.sitting else "standing"
        head = []
        if abs(self.head_yaw) > 3:
            head.append(f"turned {abs(self.head_yaw):.0f}° to the {'left' if self.head_yaw > 0 else 'right'}")
        if abs(self.pitch) > 3:
            head.append("tilted up" if self.pitch > 0 else "tilted down")
        lines = [f"[Your body: you are {posture}"
                 + (f", your head is {' and '.join(head)}" if head else "") + "."
                 + (f" You are holding the {self.holding} in your beak." if self.holding else " Your beak is empty.")
                 + f" Battery {self.battery()}%.]",
                 "[The attached picture: your camera right now (a simple drawing)"
                 + (" – sideways, because you are lying down" if self.fallen else "") + ".]",
                 f"[Depth sensor (closest thing): {depth_words(*self.depth())}]"]
        under = self.under_what()
        if under:   # 2026-11-09: he didn't know he stood under the table, and expected to SEE it when he turned
            lines.append(f"[You are standing UNDER the {under}: its top is right above your head, and its legs are "
                         f"around you. Whichever way you turn you see the room between the legs – to see the {under} "
                         f"itself, walk out from under it and turn back.]")
        parts, notes = self.map_words()
        if parts:
            lines.append("[Your mental map (rough, from your own step counting): " + "; ".join(parts) + ".]")
        if notes:
            lines.append("[Your own notes about places: " + "; ".join(notes) + ".]")
        usual = [f"the {k} – {v['where']}" + ("" if v.get("date") == date.today().isoformat() else f" (seen {v.get('date')})")
                 for k, v in places.items() if k not in self.mind_map and k != self.holding]
        if usual:
            lines.append("[Where things were when you last saw them (they may have been moved): " + "; ".join(usual) + ".]")
        lines.append(f"[Your floor plan: {self.floor_words()}.]")
        if self.stopped_in_row >= 2:
            # Only facts: how many tries, that he has hardly moved, and what is ahead and how far (same words as
            # the stop message). What to do about it is his own choice.
            d_, what_ = self._what_is_ahead()
            ahead_ = "" if d_ is None else f" {what_} is about {d_ * 100:.0f} cm in front of you."
            lines.append(f"[Note: your last {self.stopped_in_row} tries to walk were stopped almost at once and you have "
                         f"hardly moved from this spot.{ahead_}]")
        if last:
            lines.append(f"[Your last move: {last}]")
        return "\n".join(lines)

    def voice_from(self, name):
        if name not in self.people:
            return "from far away – maybe another room"
        if not HEAR_DIRECTION:
            return "nearby"
        px, py = self.people[name]
        d = math.hypot(px - self.x, py - self.y)
        rel = _wrap(_angle_to(px - self.x, py - self.y) - self.heading + random.gauss(0, 15))
        where = ("in front of you" if abs(rel) < 30 else "from behind you" if abs(rel) > 140
                 else f"from your {'left' if rel > 0 else 'right'}")
        return f"{where}, {'close' if d < 1.2 else 'far away' if d > 2.8 else 'not far'}"

    # ---------- his body ----------
    def _pose(self):
        p = super()._pose()
        p.update(hy=getattr(self, "head_yaw", 0.0), pitch=getattr(self, "pitch", 0.0), fallen=getattr(self, "fallen", False))
        return p

    def _reflex(self, backwards):
        """True if the depth sensor says something is right in front (it can't see behind him) – or, for things
        beside him that the sensor's 45° can't see any more, if his own floor plan has something there."""
        self.stop_by_map = False
        if backwards:
            return False
        from anka.body_view import depth_sensor
        dist, floor = depth_sensor(self, self.x, self.y, self.heading, min(self.pitch, 0))
        self._live_depth = (dist, floor)          # (the floor-plan check below uses it: the live sensor beats memory)
        near = None
        for j in range(8):   # is anything in his way: close ahead AND within his body's width?
            ang = math.radians((3.5 - j) * 45 / 8)
            d = np.where(floor[:, j], np.inf, dist[:, j]).min()
            if np.isfinite(d) and d * math.cos(ang) < REFLEX + DUCK_RADIUS and abs(d * math.sin(ang)) < DUCK_RADIUS + 0.04:
                near = d * math.cos(ang) if near is None else min(near, d * math.cos(ang))
        if near is not None:
            self._ahead_d, self._ahead_off = near, 0.0
            return True
        if self._map_in_the_way():
            self.stop_by_map = True
            # stopped by his MEMORY again and again at the same spot, while the live depth sensor sees open floor far
            # ahead? Then the memory is out of date (a person left, a thing was moved, the map is a bit off): the
            # sensor wins. Those squares are ignored for a while (test 22: stuck at the door 6 times in a row).
            now_ = time.time()
            log = self._map_stops = [x for x in getattr(self, "_map_stops", []) if now_ - x[0] < 30] + [(now_, self.est_x, self.est_y)]
            same = [x for x in log if math.hypot(x[1] - self.est_x, x[2] - self.est_y) < 0.12]
            low = np.where(floor[3:, :], np.inf, dist[3:, :]).min(0)       # per column: what is in a small duck's way
            if len(same) >= MAP_VETO_AFTER and low[2:6].min() > MAP_VETO_CLEAR:
                veto = self.__dict__.setdefault("_map_veto", {})
                for c_ in getattr(self, "_ahead_cells", []):
                    veto[c_] = now_ + MAP_VETO_FOR
                self.map_vetoes = getattr(self, "map_vetoes", 0) + 1
                self.stop_by_map = False
                return False
            return True
        return False

    def _map_in_the_way(self, turn=0.0):
        """His floor plan: is a square he is SURE is taken where his body is about to go (the next ~MAP_AHEAD m),
        even beside him where the depth sensor's 45° can't see? (Like a robot vacuum remembering a chair leg.)
        Not when he isn't sure where he is."""
        if self.unsure >= 0.25:
            return False
        self._ahead_off = turn          # (0 = straight ahead; ±90 = to his left / right, for side steps)
        a = math.radians(self.est_heading() + turn)
        ca, sa = math.cos(a), math.sin(a)
        i0, k0 = self._cell(self.est_x, self.est_y)
        n = int((DUCK_RADIUS + MAP_AHEAD + MAP_SIDE) / MAP_CELL) + 1
        W, H = MAP_SIZE
        i1, i2, k1, k2 = max(0, i0 - n), min(W, i0 + n + 1), max(0, k0 - n), min(H, k0 + n + 1)
        sure = np.argwhere(self.floor[i1:i2, k1:k2] > MAP_SURE)
        if not len(sure):
            return False
        cx = MAP_ORIGIN[0] + (sure[:, 0] + i1 + 0.5) * MAP_CELL - self.est_x
        cy = MAP_ORIGIN[1] + (sure[:, 1] + k1 + 0.5) * MAP_CELL - self.est_y
        fwd, side = cx * ca + cy * sa, -cx * sa + cy * ca
        # the strip his body sweeps through: in front of his middle, up to MAP_AHEAD past his nose, as wide as him
        ahead = (fwd > 0.03) & (fwd < DUCK_RADIUS + MAP_AHEAD) & (np.abs(side) < MAP_SIDE)
        veto = getattr(self, "_map_veto", None)
        if ahead.any() and turn == 0.0 and veto:
            now_ = time.time()
            for n_ in np.nonzero(ahead)[0]:
                if veto.get((int(sure[n_, 0]) + i1, int(sure[n_, 1]) + k1), 0.0) > now_:
                    ahead[n_] = False
        live = getattr(self, "_live_depth", None)
        if ahead.any() and turn == 0.0 and live is not None:
            # a remembered obstacle that the depth sensor looks straight through right now isn't there any more
            # (a person who left, a thing that was moved). Beside him, outside the sensor's 45°, memory still counts.
            dist_, floor_ = live
            low = np.where(floor_[3:, :], np.inf, dist_[3:, :]).min(0)     # per column: what is in a small duck's way
            for n_ in np.nonzero(ahead)[0]:
                r_ = math.hypot(fwd[n_], side[n_])
                j_ = 3.5 - math.degrees(math.atan2(side[n_], fwd[n_])) / (45 / 8)   # (column 0 = left)
                if -0.5 <= j_ < 7.5 and low[int(math.floor(j_ + 0.5))] > r_ + 0.06:
                    ahead[n_] = False
        if ahead.any():
            self._ahead_d = float(fwd[ahead].min())
            self._ahead_cells = [(int(sure[n_, 0]) + i1, int(sure[n_, 1]) + k1) for n_ in np.nonzero(ahead)[0]]
            return True
        return False

    def _stopped_words(self):
        return ("your legs stopped: your floor plan says something you saw earlier is right in front of you"
                if getattr(self, "stop_by_map", False) else "your depth sensor stopped you")

    def _what_is_ahead(self):
        """What stopped him, as far as HE knows: how far ahead it is (from his beak) and what he thinks it is –
        a wall he has fixed in his floor plan, or furniture / a thing / a person he has seen near that spot.
        Only a fact for the brain (test 22: he walked at the door wall 6 times because the message just said
        'something is right in front of you'); what to do about it is his own choice."""
        d = getattr(self, "_ahead_d", None)
        if d is None:
            return None, "something"
        a = math.radians(self.est_heading() + getattr(self, "_ahead_off", 0.0))
        x, y = self.est_x + d * math.cos(a), self.est_y + d * math.sin(a)
        cands = []                                         # (distance from that spot, name)
        for w in self.walls:
            p, q = self.wall_ends(w)
            vx, vy = q[0] - p[0], q[1] - p[1]
            L2 = vx * vx + vy * vy or 1e-9
            t = max(0.0, min(1.0, ((x - p[0]) * vx + (y - p[1]) * vy) / L2))
            cands.append((math.hypot(x - (p[0] + t * vx), y - (p[1] + t * vy)) - 0.15, "a wall"))
        for name, m in self.mind_map.items():
            pts = m.get("points") or ([m["pos"]] if "pos" in m else [])
            limit = 0.5 if m.get("kind") == "furniture" else 0.4
            for px, py in pts:
                dd = math.hypot(x - px, y - py)
                if dd < limit:
                    cands.append((dd - limit * 0.5, name if name in self.people else "the " + name))
        near = [c for c in cands if c[0] < 0.25]
        what = min(near)[1] if near else "something you haven't identified"
        return max(0.0, d - DUCK_RADIUS), what

    def _stopped_text(self, on_the_way=False):
        """The whole 'he was stopped' sentence, with what is ahead and how far."""
        d, what = self._what_is_ahead()
        cm = "" if d is None else f" about {d * 100:.0f} cm"
        if getattr(self, "stop_by_map", False):
            return (f"your legs stopped{' on the way' if on_the_way else ''}: your floor plan (from what you saw earlier) "
                    f"says {what} is{cm} in front of you")
        return (f"your depth sensor stopped you{' on the way' if on_the_way else ''}: {what} is{cm} in front of you")

    def _step_walk(self, dist, aim=None, stop_at=None, aim_est=None, watch=None, arrive=0.35):
        self.arrive = arrive
        self.pano_seen = {}
        self.dodged = 0
        self.watch = watch   # a name: stop as soon as the camera sees it (then the eyes guide the last bit)
        return self._step_walk2(dist, aim, stop_at, aim_est)

    def _side_step(self, left, dist):
        """Steps sideways (the Microduck can walk to the side). Returns metres moved, and what it bumped into."""
        rad = math.radians(self.heading + (90 if left else -90))
        moved = 0.0
        self._side_stop = False
        while moved < dist - 1e-6:
            if self._map_in_the_way(90 if left else -90):     # his floor plan: something he saw earlier is right there
                self._side_stop = True
                break
            nx, ny = self.x + 0.05 * math.cos(rad), self.y + 0.05 * math.sin(rad)
            hit = self._blocked(nx, ny)
            if hit:
                return moved, hit
            self.x, self.y = nx, ny
            moved += 0.05
            erad = math.radians(self.est_heading() + (90 if left else -90))
            self.est_x += 0.05 * math.cos(erad)
            self.est_y += 0.05 * math.sin(erad)
        self.trail.append((self.x, self.y))
        self._key()
        return moved, None

    def _dodge(self):
        """Something in the way while walking to a target: his legs step around it by themselves (towards the
        side where the depth sensor sees more room), like a walking controller with a depth sensor does."""
        from anka.body_view import depth_sensor
        dist, floor = depth_sensor(self, self.x, self.y, self.heading, min(self.pitch, 0))
        free = lambda cols: np.where(floor[:, cols], np.inf, dist[:, cols]).min()
        left_first = free(slice(0, 3)) >= free(slice(5, 8))
        for left in (left_first, not left_first):
            moved, _ = self._side_step(left, 0.25)
            if moved >= 0.1:
                self.dodged += 1
                return True
        return False

    def _step_walk2(self, dist, aim=None, stop_at=None, aim_est=None):
        """Walks with real-ish legs: a bit shorter than asked and a bit crooked; stops for the depth sensor.
        aim = a point to keep steering at (walking to something he sees). Returns (metres, why it stopped)."""
        self.sitting, self.head_yaw = False, 0.0
        back = dist < 0
        goal = abs(dist) * random.uniform(0.85, 1.0)
        moved, why, n, tries = 0.0, None, 0, 0
        steering = aim is not None or aim_est is not None
        self._key()
        while moved < goal - 1e-6:
            if aim is not None and n % 5 == 0:
                self.heading = (_angle_to(aim[0] - self.x, aim[1] - self.y) + random.gauss(0, 3)) % 360
                self._key()
            elif aim_est is not None and n % 5 == 0:   # steering by where he THINKS he is
                self.heading = (_angle_to(aim_est[0] - self.est_x, aim_est[1] - self.est_y) - self.drift) % 360
                self._key()
            if stop_at is not None and aim is not None and math.hypot(aim[0] - self.x, aim[1] - self.y) <= stop_at:
                why = "arrived"
                break
            if aim_est is not None and math.hypot(aim_est[0] - self.est_x, aim_est[1] - self.est_y) <= getattr(self, "arrive", 0.35):
                why = "arrived"
                break
            if getattr(self, "watch", None) and n % 4 == 0 and self._visible(self.watch)[0]:
                why = "seen"
                break
            if n % 2 == 0 and self._reflex(back):
                if steering and tries < (ROUTE_DODGES if getattr(self, "no_dodge", False) else 8) and not (aim is not None and stop_at is not None
                                                   and math.hypot(aim[0] - self.x, aim[1] - self.y) < stop_at + 0.35):
                    tries += 1
                    if self._dodge():
                        n = 1   # a few steps straight on before steering at the target again
                        continue
                why = "reflex"
                break
            wobble = random.gauss(0, 0.5)
            self.heading = (self.heading + wobble) % 360
            self.drift -= wobble * (0.1 if GYRO else 1.0)   # (the gyro notices most of his legs going crooked)
            rad = math.radians(self.heading) + (math.pi if back else 0)
            nx, ny = self.x + 0.05 * math.cos(rad), self.y + 0.05 * math.sin(rad)
            hit = self._blocked(nx, ny)
            near_goal = (aim is not None and stop_at is not None
                         and math.hypot(aim[0] - self.x, aim[1] - self.y) < stop_at + 0.35)
            if hit and steering and tries < (ROUTE_DODGES if getattr(self, "no_dodge", False) else 8) and not (hit in self.people and near_goal):   # (a person in the way too)
                tries += 1
                if self._dodge():
                    n = 1
                    continue
            if hit:
                why = hit
                break
            self.x, self.y = nx, ny
            moved += 0.05
            self._counted_step(0.05 * abs(dist) / goal, back)   # he thinks his steps were as long as he asked
            if n % 5 == 4:
                self._landmarks_only()                           # his eyes keep working while he walks
            n += 1
            if n % 4 == 0:
                self.trail.append((self.x, self.y))
                self._key()
        self.trail.append((self.x, self.y))
        self.walked += moved
        self.stopped_in_row = self.stopped_in_row + 1 if (why and why != "arrived" and moved < 0.15) else 0
        self._key()
        return moved, why

    def _bumped(self, what, back, turn=0.0):
        """A bump the body feels (IMU). Sometimes he falls over."""
        self.bumped = what
        self._mark_bump(back, person=what in self.people, turn=turn)
        if random.random() < (0.35 if back else FALL_BUMP):
            self.fallen, self.falls = True, self.falls + 1
            self._key()
            return f" – BUMP! you walked into {what} and fell over"
        return f" – bump! you walked into {what}"

    def _visible(self, words):
        """The thing or person in 'words' that his camera can see right now (or None, and its name)."""
        name = self._find(words)
        if not name:
            return None, None
        seen = self.seen_names(15)
        if name in seen or name in getattr(self, "pano_seen", {}):   # (also what he saw when he looked around)
            return name, name
        if name in FURNITURE or name in WALL_THINGS:   # big things: in view if they are inside the camera's angle
            px, py = self._nearest_point(name)
            rel = _wrap(_angle_to(px - self.x, py - self.y) - self.heading - self.head_yaw)
            if abs(rel) <= 35:
                return name, name
        return None, name

    def _people_words(self):
        """Facts about the people he has seen or heard in the last 20 s, seen from where he is: who is NOT right next to
        him, how far, which way, and if something is in the way. Said when he arrives somewhere or sits down – in test 23
        he walked to the sofa's corner when Alex asked him to come, said \"I'm sitting right by your legs\" and sat
        1.3 m from Alex. What to do about it is his own choice."""
        out, now = [], time.time()
        for name, m in self.mind_map.items():
            if m.get("kind") != "person" or name not in self.people or now - m["t"] > 20:
                continue
            d, rel = self._from_me(self._mm_pos(name))
            if d < 0.6 or d > 4.5:
                continue
            bit = f"{name} is NOT right next to you: {_about(d)} {_side(rel)}"
            px, py = self.people[name]
            dd = math.hypot(px - self.x, py - self.y)
            if dd > 0.5:
                ux, uy = (px - self.x) / dd, (py - self.y) / dd
                if self._clear((self.x, self.y), (px - 0.4 * ux, py - 0.4 * uy)):
                    bit += " (something is in the way – to get there you have to go around it: walking to " + name + " does that)"
            out.append(bit)
        return out

    def _walk_to(self, name):
        px, py = self._nearest_point(name)
        turn = _wrap(_angle_to(px - self.x, py - self.y) - self.heading)
        self.heading = _angle_to(px - self.x, py - self.y) % 360
        self._turned(turn)
        self.head_yaw = 0.0
        self._key()
        stop = 0.33 if name in self.people else 0.28 if name in self.things else 0.3
        moved, why = self._step_walk(6.0, aim=(px, py), stop_at=stop)
        who = name if name in self.people else "the " + name
        text = f"walked {_about(moved)} towards {who}" if moved > 0.04 else f"turned to {who}"
        if self.dodged:
            text += f" (your legs stepped around something {self.dodged} time{'s' if self.dodged > 1 else ''} on the way)"
        if why == "arrived" or (why == "reflex" and math.hypot(px - self.x, py - self.y) < 0.6):
            extra = ""
            if name not in self.people:
                self.remember_view()
                w = self._people_words()
                extra = (" – " + "; ".join(w)) if w else ""
            return text + f" – {who} is right in front of you now" + extra, True
        if why == "reflex":
            return text + f" – {self._stopped_text()}", False
        if why:
            return text + self._bumped(why, False), False
        return text, True

    def _walk_to_remembered(self, name):
        """Walks to where he REMEMBERS seeing something – going by his own (drifting) idea of where he is."""
        goal = self._mm_pos(name)
        heard = self.mind_map[name].get("heard")
        d, rel = self._from_me(goal)
        true_turn = rel + random.gauss(0, 3)
        self.heading = (self.heading + true_turn) % 360
        if not GYRO:
            self.drift += rel - true_turn
        self._turned(rel)
        self.head_yaw = 0.0
        self._key()
        label = name if name in self.people else "the " + name
        moved, why = self._walk_route(goal, watch=name)
        text = (f"turned {'left' if rel > 0 else 'right'} about {abs(rel):.0f}° and walked {_about(moved)} to where you "
                f"{'heard' if heard else 'remember'} {label}"
                + (f" (stepping around something {self.dodged}×)" if self.dodged else ""))
        if why == "seen":
            text = (f"turned {'left' if rel > 0 else 'right'} about {abs(rel):.0f}° towards where you "
                    f"{'heard' if heard else 'remember'} {label}" + (f", walked {_about(moved)}" if moved > 0.15 else ""))
        if why == "reflex":
            text += f" – {self._stopped_text(True)}"
        elif why and why not in ("arrived", "seen"):
            text += self._bumped(why, False)
        self.remember_view()
        if not self._visible(name)[0] and self._look_for(name):
            text += f" – looked left and right and spotted {label}"
        close = math.hypot(goal[0] - self.est_x, goal[1] - self.est_y)
        if not self._visible(name)[0] and close < 0.6 and self._glance_down(name):
            text += (f" – {label} was right in front of your feet: with your head level your camera can't see the floor closer "
                     f"than about 40 cm, so you tilted your head down and spotted it")
        if self._visible(name)[0]:   # there it is – the legs go the last bit by sight
            last, ok = self._walk_to(name)
            return text + f" – you can see {label} now; then you {last}", ok
        up_on = (self.things.get(name) or {}).get("on")   # he saw it UP on furniture (e.g. the cup on the table)
        if close < 2.0 and up_on and math.hypot(self.x - self.things[name]["pos"][0], self.y - self.things[name]["pos"][1]) < 1.2:
            # 2026-11-09: the old text said "it is not here … probably moved" – but it was on the table above him
            return (text + f" – you are by the {up_on}, where you saw {label} up on top. From down here your camera can't "
                    f"see the top of the {up_on}, so you can't check whether {label} is still there", False)
        if close < 0.6:
            return (text + f" – you are at the spot where you remember {label}, and it is not here: you tilted your head down "
                    f"and looked left and right too. It was probably moved, or your map is a little off. "
                    "Look around before you do anything else here", False)
        return (text + f" – but you are NOT at {label}: you can't see {label} from here (your map was off). "
                "Look around before you do anything else here", False)

    def _glance_down(self, name):
        """His camera sits 20 cm above the floor: with the head level it can't see the floor closer than ~40 cm in
        front of him (a 5 cm sock closer than ~30 cm is gone from the picture) – and a thing he puts down lands 20 cm
        in front of him, right in that blind spot (test 25: \"you are NOT at the sock – your map was off\", he turned
        around 6 times; the map was right). Like a duck bending its neck, the head tips down to look. Stays tipped
        if it finds the thing (so the next move can use it); otherwise the head goes back."""
        if self.fallen or self._visible(name)[0]:
            return False
        keep = self.pitch
        for tilt in (-20.0, -45.0):
            self.pitch = tilt
            self._key()
            self.remember_view()
            if name in self.seen_names(15):
                return True
        self.pitch = keep
        self._key()
        return False

    def _look_for(self, name):
        """Not in sight: he turns his head left and right to look for it; if he finds it, he turns towards it."""
        from anka.body_view import panorama
        _, seen = panorama(self, self.x, self.y, self.heading, self.pitch)
        found = name in seen and seen[name]["pixels"] >= 15
        if not found and (name in FURNITURE or name in WALL_THINGS):
            px, py = self._nearest_point(name)
            found = abs(_wrap(_angle_to(px - self.x, py - self.y) - self.heading)) <= 95
        if found:
            px, py = self._nearest_point(name)
            turn = _wrap(_angle_to(px - self.x, py - self.y) - self.heading)
            self.heading = (self.heading + turn) % 360
            self._turned(turn)
            self._key()
            self.remember_view()
        return found

    def _walk_to_usual(self, name):
        """He hasn't seen it today, but remembers where it was: he goes to that landmark and looks there."""
        near = places[name]["near"]
        if self._visible(near)[0]:
            text, ok = self._walk_to(near)
        elif near in self.mind_map or near in self.landmarks:
            if near not in self.mind_map:
                self.mind_map[near] = {"points": [self.landmarks[near]["pos"]], "t": time.time(), "kind": "furniture"}
            text, ok = self._walk_to_remembered(near)
        else:
            return (f"the {name} was {places[name]['where']} last time, but you don't know where the {near} is from "
                    f"here – look around to find the {near}", False)
        text = f"went to look {places[name]['where']}, where the {name} was last time: " + text
        if not self._visible(name)[0] and self._look_for(name):
            text += f" – looked left and right and spotted the {name}"
        if self._visible(name)[0]:
            last, ok = self._walk_to(name)
            return text + f" – and there is the {name}! you {last}", ok
        return text + f" – but you can't see the {name} there now; maybe someone moved it", False

    def do(self, act):
        with self.lock:
            return self._do(act)

    def _do(self, act):
        a = act.strip().strip(".").lower()
        num = re.search(r"(-?\d+(?:\.\d+)?)", a)
        n = float(num.group(1)) if num else None
        self.bumped = None
        if not a or re.fullmatch(r"(none|nothing|no|wait|stay|stay here|stay still|stay where i am|rest)\b.*", a):
            return None, True
        ok = True
        if self.fallen and not re.search(r"\b(get up|stand up|stand|roll over)\b", a):
            result, ok = "you are lying on your side – you have to get up first", False
        elif re.search(r"\b(get up|stand up|stand|get back up)\b", a):
            if self.fallen:
                if random.random() < 0.8:
                    self.fallen, result = False, "got back up on your feet"
                else:
                    result, ok = "tried to get up, but tipped over again", False
            else:
                self.sitting, result = False, "stood up"
        elif re.match(r"(walk|go|come|move|waddle|run|head)( over| back)? (to|towards|up to)\b|approach\b", a):
            target = re.sub(r"^\w+( over| back)? (to|towards|up to)?", "", a)
            name, wanted = self._visible(target)
            v = state.get("voice_from")
            heard_now = (wanted in self.people and v and v[1] == wanted and time.time() - v[0] < 60 and HEAR_DIRECTION)
            if name:
                result, ok = self._walk_to(name)
            elif heard_now and wanted in self.mind_map:   # he can't see them, but he just heard where they are
                result, ok = self._walk_to_remembered(wanted)
            elif wanted in self.mind_map:
                result, ok = self._walk_to_remembered(wanted)
            elif wanted in places and places[wanted].get("near"):
                result, ok = self._walk_to_usual(wanted)
            else:
                ok = False
                result = (f"you can't see {'the ' + wanted if wanted and wanted not in self.people else wanted or 'that'} "
                          "in your camera right now – turn or look around to find it first")
        elif re.search(r"\b(step|walk|move|shuffle|go)\b.*\b(left|right)\b|\bsidestep\b|\bsideways\b", a) \
                and not re.search(r"\bturn\b", a):
            left = "left" in a
            dist = min(0.5, (n if n is not None else 20) / (100 if (n is None or n > 5 or "cm" in a) else 1))
            moved, hit = self._side_step(left, dist)
            result = f"stepped {_about(moved)} to the {'left' if left else 'right'}" if moved > 0.04 else "couldn't step sideways"
            if hit:
                result, ok = result + self._bumped(hit, False, 90 if left else -90), False
            elif getattr(self, "_side_stop", False):
                d_, what_ = self._what_is_ahead()
                result, ok = (result + f" – your legs stopped: your floor plan (from what you saw earlier) says {what_} is"
                              + ("" if d_ is None else f" about {d_ * 100:.0f} cm") + f" to your {'left' if left else 'right'}"), False
        elif re.match(r"(walk|go|move|waddle) around\b", a):
            result, ok = ("your legs need a direction – turn left or right, then walk forward (or walk to something you "
                          "can see)"), False
        elif re.search(r"\b(walk|go|move|waddle|step)\b", a):
            back = bool(re.search(r"\b(back|backward|backwards)\b", a))
            if n is not None and re.search(r"\bsteps?\b", a):
                dist = n * 0.15
            else:
                dist = (n if n is not None else 50) / (100 if (n is None or n > 5 or "cm" in a) else 1)
            dist = min(dist, 0.3 if back else MAX_STEP)
            moved, why = self._step_walk(-dist if back else dist)
            result = f"walked {'backwards ' if back else 'forward '}{_about(moved)}" if moved > 0.04 else "didn't move"
            if why == "reflex":
                result, ok = result + f" – {self._stopped_text()}", False
            elif why:
                result, ok = result + self._bumped(why, back), False
            if back:
                result += " (only small steps backwards – you can't see behind you)"
        elif re.search(r"\b(turn|rotate|spin)\b", a):
            if "around" in a:
                deg = 180
                self.heading = (self.heading + deg * random.uniform(0.92, 1.08)) % 360
                self._turned(deg)
                self.head_yaw = 0.0
                result = "turned around" + self._memory_ahead()
            elif re.search(r"\bto\b|\btowards\b|\bface\b", a):
                target = re.split(r"\bto\b|\btowards\b|\bface\b", a, 1)[-1]
                name, wanted = self._visible(target)
                v = state.get("voice_from")
                if name:
                    px, py = self._nearest_point(name)
                    if name in FURNITURE and math.hypot(px - self.x, py - self.y) < 0.6:   # right by it: face its middle
                        fx1, fy1, fx2, fy2 = FURNITURE[name][:4]
                        px, py = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                    self.heading, self.head_yaw = _angle_to(px - self.x, py - self.y) % 360, 0.0
                    result = f"turned to face {name if name in self.people else 'the ' + name}"
                elif wanted in self.mind_map and not (wanted in self.people and v and v[1] == wanted
                                                      and time.time() - v[0] < 60):
                    _, rel = self._from_me(self._mm_aim(wanted))
                    self.heading = (self.heading + rel + random.gauss(0, 3)) % 360
                    self._turned(rel)
                    self.head_yaw = 0.0
                    result = f"turned towards where you remember {wanted if wanted in self.people else 'the ' + wanted}"
                elif wanted in self.people and v and v[1] == wanted and time.time() - v[0] < 60 and HEAR_DIRECTION:
                    px, py = self.people[wanted]
                    self.heading = (_angle_to(px - self.x, py - self.y) + random.gauss(0, 15)) % 360
                    self.head_yaw = 0.0
                    result = f"turned towards where {wanted}'s voice came from"
                else:
                    result, ok = f"you don't know where {wanted or 'that'} is – you can't see or hear it", False
            else:
                deg = n if n is not None else 90
                deg = -deg if "right" in a else deg
                true_turn = deg * random.uniform(0.9, 1.1) + random.gauss(0, 3)
                self.heading = (self.heading + true_turn) % 360
                if not GYRO:
                    self.drift += deg - true_turn      # he thinks he turned exactly as much as he asked
                self._turned(deg)
                self.head_yaw = 0.0
                self.pano_seen = {}
                result = f"turned {'left' if deg > 0 else 'right'} about {abs(deg):.0f}°" + self._memory_ahead()
            self._key()
        elif re.search(r"\bexplor|\bunexplored\b", a):
            f = self.frontier()
            if not f:
                result, ok = "you don't know of any unexplored place nearby – look around or walk somewhere new", False
            else:
                moved, why = self._walk_route(f[0], stop=0.25)
                result = (f"walked {_about(moved)} to the edge of what you have explored"
                          + (f" (stepping around something {self.dodged}×)" if self.dodged else ""))
                if why == "reflex":
                    result += f" – {self._stopped_text(True)}"
                elif why not in ("arrived", "seen"):
                    result, ok = result + self._bumped(why, False), False
                self.panorama_next = True
                result += " – you look around there (the next picture shows left, ahead and right)"
        elif re.search(r"\blook (around|all around|behind)\b|\bscan\b", a):
            self.panorama_next = True
            result = "turned your head left and right – the next picture shows three views: left, ahead and right"
        elif re.search(r"\blook\b|\bhead\b|\btilt\b", a):
            name, wanted = self._visible(re.sub(r"\b(look|at)\b", "", a))
            if re.search(r"\bup\b", a):
                self.pitch, result = 25.0, "tilted your head up"
            elif re.search(r"\bdown\b", a):
                self.pitch, result = -20.0, "tilted your head down"
            elif re.search(r"\b(straight|forward|ahead|center|centre)\b", a):
                self.head_yaw, self.pitch, result = 0.0, 0.0, "looked straight ahead again"
            elif re.search(r"\b(left|right)\b", a):
                deg = min(70, n if n is not None else 45)
                self.head_yaw = deg if "left" in a else -deg
                result = f"turned your head {deg:.0f}° to the {'left' if self.head_yaw > 0 else 'right'}"
            elif name:
                px, py = self._nearest_point(name)
                self.head_yaw = max(-70, min(70, _wrap(_angle_to(px - self.x, py - self.y) - self.heading)))
                result = f"looked at {name if name in self.people else 'the ' + name}"
            else:
                result, ok = f"you can't see {wanted or 'that'} right now", False
            self._key()
        elif re.search(r"\bsit\b", a):
            self.sitting, result = True, "sat down"
            self.remember_view()
            w = self._people_words()
            if w:
                result += " – " + "; ".join(w)
        elif "kick" in a or re.search(r"\b(pick up|grab|take|bite)\b", a):
            kick = "kick" in a
            word = re.sub(r"\b(kick|pick up|grab|take|bite|to|at)\b.*?(?=\b(ball|sock|cup|guitar)\b|$)", " ", a)
            name = self._find(word) or ("ball" if kick else None)
            t = self.things.get(name) if name else None
            if not kick and self.holding:
                result, ok = f"can't – already holding the {self.holding}", False
            elif not t or (kick and not t.get("kick")):
                result, ok = f"can't {'kick' if kick else 'pick up'} {('the ' + name) if name else 'that'}", False
            elif t.get("on"):
                result, ok = f"the {name} is up on the {t['on']} – too high for a small duck", False
            elif not kick and not t["small"]:
                result, ok = f"the {name} is too big for your beak", False
            else:
                d = math.hypot(t["pos"][0] - self.x, t["pos"][1] - self.y)
                rel = _wrap(_angle_to(t["pos"][0] - self.x, t["pos"][1] - self.y) - self.heading)
                walked = ""
                if name not in self.seen_names(15) and name in self.mind_map:    # his own map says it is at his feet
                    if math.hypot(self._mm_pos(name)[0] - self.est_x, self._mm_pos(name)[1] - self.est_y) < 0.6:
                        self._glance_down(name)
                if d > 0.45 or abs(rel) > 40:
                    rem = ""
                    if name not in self.seen_names(15) and name in self.mind_map:
                        rem = self._walk_to_remembered(name)[0]
                    if name not in self.seen_names(15):
                        result, ok = (rem or f"you can't see the {name} right now – find it first"), False
                        walked = None
                    else:
                        walked, ok = self._walk_to(name)
                        d = math.hypot(t["pos"][0] - self.x, t["pos"][1] - self.y)
                        rel = _wrap(_angle_to(t["pos"][0] - self.x, t["pos"][1] - self.y) - self.heading)
                        if d > 0.45 or abs(rel) > 40:
                            result, ok = walked + f" – but the {name} isn't within reach yet", False
                            walked = None
                if walked is not None:
                    pre = walked + ", then " if walked else ""
                    if kick:
                        rad = math.radians(self.heading + random.gauss(0, 8))
                        bx, by = t["pos"]
                        rolled, stopped_by = 0.0, None
                        for _ in range(int(random.uniform(18, 32))):
                            nx, ny = bx + 0.05 * math.cos(rad), by + 0.05 * math.sin(rad)
                            hit = [b[7] for b in self.solid_boxes() if b[4] < 0.2 and b[0] < nx < b[2] and b[1] < ny < b[3]]
                            if not (0.12 < nx < ROOM_W - 0.12 and 0.12 < ny < ROOM_H - 0.12) or hit:
                                stopped_by = f"the {hit[0]}" if hit else "the wall"
                                break
                            bx, by, rolled = nx, ny, rolled + 0.05
                        t["pos"] = (bx, by)
                        if rolled < 0.06:
                            ok = False        # (the kick didn't move it – that counts as a failed try)
                        result = pre + (f"kicked the {name}, but it hardly moved – {stopped_by} is right behind it"
                                        if rolled < 0.06 else f"kicked the {name} – it rolled away {_about(rolled)}"
                                        + (f" and hit {stopped_by}" if stopped_by else ""))
                        if random.random() < 0.08:
                            self.fallen, self.falls = True, self.falls + 1
                            result += " – you kicked so hard you fell over"
                    else:
                        self.holding = name
                        result = pre + f"picked up the {name} with your beak"
        elif re.search(r"\b(put down|drop|let go|give)\b", a):
            if not self.holding:
                result, ok = "you aren't holding anything", False
            else:
                rad = math.radians(self.heading)
                spot = (self.x + 0.2 * math.cos(rad), self.y + 0.2 * math.sin(rad))
                if self._blocked(*spot) and self._blocked(*spot) not in self.people:
                    spot = (self.x, self.y)
                self.things[self.holding]["pos"] = spot
                self._put_down_here(self.holding)
                # who is next to it? (people he sees right there, or where he thinks they are) – a fact he keeps
                # forgetting otherwise: in test 22 he put the sock down by Alex and then picked it up again to "bring" it
                seen = self.seen_names(15)
                near = [p_ for p_, v in seen.items() if v.get("kind") == "person" and p_ in self.people
                        and math.hypot(self.people[p_][0] - spot[0], self.people[p_][1] - spot[1]) < 0.8]
                est_spot = (self.est_x + 0.2 * math.cos(math.radians(self.est_heading())),
                            self.est_y + 0.2 * math.sin(math.radians(self.est_heading())))
                near += [n_ for n_, m in self.mind_map.items() if m.get("kind") == "person" and n_ in self.people
                         and n_ not in near and "pos" in m and math.hypot(m["pos"][0] - est_spot[0], m["pos"][1] - est_spot[1]) < 0.8]
                result, self.holding = (f"put down the {self.holding} on the floor in front of you"
                                        + (f", right by {near[0]}" if near else "")
                                        + " – your beak is empty now"), None
        else:
            result, ok = (f"your body doesn't know how to '{act}' – what your body CAN do: walk, step left/right, turn, "
                          "look (left, right, up, down, around), sit, stand, kick a ball, pick up one small thing with your "
                          "beak, put it down, explore"), False   # (2026-11-09: facts about his body)
        self.actions.append((act, result))
        self._key()
        self.remember_view()
        return result, ok

    # ---------- the window: the room from above + what he sees + the depth sensor ----------
    def draw(self, scale=100, caption=(), bubbles=(), live=True):
        top = super().draw(scale, caption, bubbles, live)
        if not live:      # the picture saved after the test: the room + his own floor plan side by side
            return np.hstack([self._draw_beliefs(top, scale), self.floor_picture(top.shape[0])])
        top = self._draw_beliefs(top, scale)
        from anka.body_view import H as CH, W as CW, depth_picture, depth_sensor, render
        return self._draw_live(top, CH, CW, depth_picture, depth_sensor, render)

    def _draw_beliefs(self, top, scale):
        pad = 30
        P = lambda x, y: (int(pad + x * scale), int(pad + (ROOM_H - y) * scale))
        for name, m in self.mind_map.items():   # where he THINKS things are (purple rings)
            if m["kind"] == "furniture":
                for q in m.get("points", []):
                    cv2.circle(top, P(*q), 3, (170, 60, 150), -1, cv2.LINE_AA)
                continue
            cv2.circle(top, P(*m["pos"]), 9, (170, 60, 150), 2, cv2.LINE_AA)
            cv2.putText(top, name + "?", (P(*m["pos"])[0] + 10, P(*m["pos"])[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (170, 60, 150), 1, cv2.LINE_AA)
        for name, lm in self.landmarks.items():   # landmarks as he remembers them
            x, y = P(*lm["pos"])
            cv2.rectangle(top, (x - 5, y - 5), (x + 5, y + 5), (170, 60, 150), 2)
        for pos, text, _ in self.place_notes:
            cv2.drawMarker(top, P(*pos), (170, 60, 150), cv2.MARKER_STAR, 14, 2)
        cv2.circle(top, P(self.est_x, self.est_y), int(DUCK_RADIUS * scale), (170, 60, 150), 1, cv2.LINE_AA)
        cv2.putText(top, "purple = where he thinks he and things are", (pad, pad + int(ROOM_H * scale) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 60, 150), 1, cv2.LINE_AA)
        return top

    def _draw_live(self, top, CH, CW, depth_picture, depth_sensor, render):
        sh = self.show
        key = ("show", round(sh["x"], 2), round(sh["y"], 2), round(sh["h"] + sh.get("hy", 0), 0), sh.get("pitch", 0),
               sh.get("fallen", False), tuple(sorted((k, round(v[0], 2), round(v[1], 2)) for k, v in self.show_things.items())),
               tuple(sorted((k, round(v[0], 1), round(v[1], 1)) for k, v in self.show_people.items())), sh.get("hold"))
        if getattr(self, "_panel_key", None) != key:
            world = _ShownWorld(self)
            yaw = sh["h"] + sh.get("hy", 0)
            cam, _ = render(world, sh["x"], sh["y"], yaw, sh.get("pitch", 0), sh.get("fallen", False))
            tof = depth_picture(*depth_sensor(world, sh["x"], sh["y"], yaw, sh.get("pitch", 0)))
            self._panel = (cam, tof)
            self._panel_key = key
        cam, tof = self._panel
        panel = np.full((top.shape[0], CW + 20, 3), (236, 240, 244), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, f"what {NAME} sees", (10, 22), font, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
        panel[30:30 + CH, 10:10 + CW] = cam
        y0 = 30 + CH + 28
        cv2.putText(panel, "depth sensor (8x8)", (10, y0 - 8), font, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
        if y0 + tof.shape[0] <= panel.shape[0]:
            panel[y0:y0 + tof.shape[0], 10:10 + tof.shape[1]] = tof
        heat = getattr(self, "depth_ai_view", None)     # the depth AI's heat picture (near = bright)
        hx, hw = tof.shape[1] + 20, CW - tof.shape[1] - 10
        if heat is not None and hw > 40:
            hh = int(hw * heat.shape[0] / heat.shape[1])
            cv2.putText(panel, "depth AI", (hx, y0 - 8), font, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
            if y0 + hh <= panel.shape[0]:
                panel[y0:y0 + hh, hx:hx + hw] = cv2.resize(heat, (hw, hh))
        yb = min(panel.shape[0] - 8, y0 + tof.shape[0] + 22)
        cv2.putText(panel, f"battery {self.battery()}%" + (f"   falls: {self.falls}" if self.falls else "")
                    + ("   FALLEN OVER" if sh.get("fallen") else ""), (10, yb), font, 0.45,
                    (40, 40, 200) if sh.get("fallen") else (60, 60, 60), 1, cv2.LINE_AA)
        plan = self.floor_picture(top.shape[0])
        return np.hstack([top, panel, plan])

    def floor_picture(self, height):
        """His own floor plan as he believes it (grey = never seen, white = free floor, black = something there)."""
        px = 60   # pixels per metre
        x0, y0, x1, y1 = -0.5, -0.5, ROOM_W + 0.5, ROOM_H + 0.5     # the part of his map to show
        f = self.floor
        i0, k0 = self._cell(x0, y0)
        i1, k1 = self._cell(x1, y1)
        sub = f[max(0, i0):i1, max(0, k0):k1].T[::-1]             # rows = y (north up)
        img = np.full(sub.shape, 150, np.uint8)
        img[sub < -0.5] = 245
        occupied = (sub > 0.6).astype(np.uint8)
        people = self.people_floor[max(0, i0):i1, max(0, k0):k1].T[::-1] > 0.6
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img[people] = (200, 120, 40)            # people (fading) in blue
        size = (int((x1 - x0) * px), int((y1 - y0) * px))
        P = lambda x, y: (int((x - x0) * px), int((y1 - y) * px))
        img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
        # fixed walls: one solid line each. Everything else that is in the way (a table leg, a ball, a bit of wall
        # not confirmed yet) is drawn as a dark blob.
        near_wall = np.zeros_like(occupied)
        if self._wall_mask is not None:
            wm = self._wall_mask[max(0, i0):i1, max(0, k0):k1].T[::-1].astype(np.uint8)
            near_wall = cv2.dilate(wm, np.ones((5, 5), np.uint8))
        blobs = cv2.resize(occupied & (1 - near_wall), size, interpolation=cv2.INTER_NEAREST) > 0
        img[blobs] = (70, 70, 70)
        for w in self.walls:
            a, b = self.wall_ends(w)
            cv2.line(img, P(*a), P(*b), (20, 20, 20), 2, cv2.LINE_AA)
        for name, lm in self.landmarks.items():
            cv2.rectangle(img, (P(*lm["pos"])[0] - 4, P(*lm["pos"])[1] - 4), (P(*lm["pos"])[0] + 4, P(*lm["pos"])[1] + 4), (170, 60, 150), 2)
        f_ = self.frontier()
        if f_:
            cv2.drawMarker(img, P(*f_[0]), (40, 160, 40), cv2.MARKER_CROSS, 12, 2)
        e = P(self.est_x, self.est_y)
        h = math.radians(self.est_heading())
        cv2.circle(img, e, 6, (40, 200, 250), -1, cv2.LINE_AA)
        cv2.line(img, e, (int(e[0] + 14 * math.cos(h)), int(e[1] - 14 * math.sin(h))), (30, 120, 240), 3, cv2.LINE_AA)
        out = np.full((height, img.shape[1] + 20, 3), (236, 240, 244), np.uint8)
        cv2.putText(out, f"{NAME}'s own floor plan", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
        hh = min(img.shape[0], height - 70)
        out[30:30 + hh, 10:10 + img.shape[1]] = img[:hh]
        cv2.putText(out, "grey = never seen  white = free  lines = walls  dark = in the way  blue = people", (10, 30 + hh + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(out, "green + = nearest unexplored place", (10, 30 + hh + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (40, 140, 40), 1, cv2.LINE_AA)
        return out


class _ShownWorld:
    """The room as the window shows it right now (people and things where they are in the animation)."""
    def __init__(self, body):
        self.b = body

    def room_size(self):
        return ROOM_W, ROOM_H

    def solid_boxes(self):
        return self.b.solid_boxes()

    def sprites(self):
        out = []
        for s in self.b.sprites():
            if s["kind"] == "person" and s["name"] in self.b.show_people:
                s["x"], s["y"] = self.b.show_people[s["name"]]
            elif s["name"] in self.b.show_things and s["kind"] != "person":
                s["x"], s["y"] = self.b.show_things[s["name"]]
            out.append(s)
        hold = self.b.show.get("hold")
        return [s for s in out if s["name"] != hold]


# ---------- the body Anka has right now (None = no body) ----------
def _make_body():
    if MUJOCO:   # the Microduck in Pollen's MuJoCo simulator (see the ankasim folder)
        from ankasim.simbody import SimBody
        return SimBody()
    return RealSensesBody() if BODY_SENSES == "real" else VirtualBody()


body = _make_body() if BODY else None


def body_text():
    """What the brain is told about his body and what's in front of him (instead of the camera list)."""
    t, last = state.get("last_move", (0, None))
    text = body.describe(last if time.time() - t < 90 else None)
    tried = tried_words(goals)   # B2: what he tried lately and how it went (facts only)
    return text + ("\n" + tried if tried else "")


def run_act(text):
    """Carries out what the brain chose (one action, or up to three: 'walk to the sock then pick up the sock').
    Stops early if something goes wrong (bumped into something, too far away ...)."""
    if body is None or not text or re.fullmatch(r"(?i)\W*(none|nothing|no)\W*", text):
        return None
    voice = state.get("voice_from")
    if voice and time.time() - voice[0] < 60 and voice[1] in body.people:   # "walk to you" = to whoever is talking
        text = re.sub(r"(?i)\b(you|me|him|her|the human)\b", voice[1], text)
    steps = [s.strip(" .") for s in re.split(r"(?i)\s*(?:,?\s*\bthen\b|,?\s*\band then\b|;|\n)\s*", text) if s.strip(" .")][:3]
    done, log = [], []
    for step in steps:
        result, ok = body.do(step)
        if result is None:
            continue
        done.append(result)
        log.append({"step": step, "result": result, "ok": bool(ok)})
        print(f"🦶 ({NAME}: {step} → {result})")
        if not ok:
            break
    record_tries(log)   # B2: remembered for "what you tried lately"
    if done:
        state["last_move"] = (time.time(), "; ".join(done))
        # (for his thinking pace: only REAL moving counts as "on the move" – looking, sitting and standing don't.
        #  Before 2026-11-09 a "look up" also put him in the fast 4-6 s pace, so he was almost always in it.)
        if any(re.search(r"(?i)\b(walk|step|turn|explore|kick|pick up|put down|drop|get up|go|come|move|waddle|run)\b", x["step"])
               and not re.match(r"(?i)\s*(look|turn your head|tilt)", x["step"]) for x in log):
            state["last_travel"] = time.time()
        state["caption"] = done[-1]
        if body.bumped:
            state["events"].append(f"You bumped into {body.bumped} while walking.")
    return log   # (what happened, step by step - for the thought log)

from datetime import date
from anka.memory import goals, memory_lock, places, save_memory
from anka.tried import record as record_tries, tried_words
