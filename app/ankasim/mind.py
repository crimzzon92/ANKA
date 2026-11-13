"""mind: what Anka KNOWS about where he is and where things are - built only from his own senses. No truth in here.

Ported from the 2D simulator (anka/body.py), where it was tried in ~100 test runs, but fed with the real Microduck's
odometry instead of an invented drift:

  * WHERE AM I      a small Kalman filter (EKF-SLAM) keeps his position + heading and the landmarks' places together.
                    Step counting moves the estimate (and makes it less sure); every time he sees a landmark (sofa,
                    table, bookshelf, door, window) the filter corrects it.
  * WHERE ARE THINGS  a mental map of what he has seen (the ball, the sock ...), kept in his own coordinates (x forward and
                    y left, from where he woke up), with when he saw it.
  * HIS OWN FLOOR PLAN  a grid of 5 cm squares drawn from the depth sensor: free floor / something there / never seen.
                    Used to plan routes around what he knows is in the way, and to find the nearest unexplored place.
  * PLACES          where things are, in words ("by the bookshelf"), so he can remember them between days.

Nothing here moves the body: SimBody asks the mind (where is it? route to it? what is unexplored?) and the brain decides.
"""
import math
import time
from collections import deque

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

CELL = 0.05                       # his floor plan: 5 cm squares
ORIGIN = (-5.0, -5.0)             # 10 x 10 m around where he woke up (he doesn't know how big the room is)
SIZE = (200, 200)
DUCK_RADIUS = 0.12                # the Microduck is about 25 cm tall and ~20 cm wide
FREE_STEP, HIT_STEP = -0.35, 0.9  # log-odds a look adds: free floor / something there (clipped to +-5)
FREE_FADE, FREE_MIN = 4.0, 0.25   # free floor far away counts less
FREE_NO_HIT = 2.0                 # a ray that hits nothing only shows free floor this far
BLOCKED = 0.6                     # a square counts as taken above this
DRAW_EVERY_M, DRAW_EVERY_DEG, DRAW_EVERY_S = 0.03, 3.0, 1.0    # draw into the floor plan only after he moved / turned / a while
STEADY = 0.25                     # just corrected by more than this (m)? then don't trust where he thinks he is yet
WALL_HEADING = False              # use the straight walls to correct his sense of direction? OFF as in the 2D simulator: a room direction that
                                  # is a few degrees wrong at the time it is first decided would turn the whole map. While it is off he
                                  # still MEASURES what the walls would say (wall_log), so a test report can show whether it would help.
WALL_SD = 3.0                     # how exactly one metre of straight wall tells the direction (degrees)
ROOM_RUNS = 15                    # straight runs needed before he has a first idea of the room's main direction
VOTES_TO_MOVE = 10                # a thing seen somewhere else this many times in a row has really moved


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def metres(d):
    return f"{d * 100:.0f} cm" if d < 1 else f"{d:.1f} m"


def about(d):
    return f"about {d * 100:.0f} cm" if d < 1 else f"about {d:.1f} m"


def side(rel_deg):
    a = abs(rel_deg)
    s = "left" if rel_deg > 0 else "right"
    if a <= 12:
        return "straight ahead"
    if a <= 35:
        return f"ahead, a bit to the {s}"
    if a <= 80:
        return f"to your {s}"
    if a <= 140:
        return f"behind you on the {s}"
    return "behind you"


def _dilate(mask, r):
    """Grows True squares by r squares (a disc), so a route through free squares fits his body."""
    if cv2 is not None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        return cv2.dilate(mask.astype(np.uint8), k) > 0
    out = np.zeros_like(mask)
    for di in range(-r, r + 1):
        for dk in range(-r, r + 1):
            if di * di + dk * dk <= r * r:
                out |= np.roll(np.roll(mask, di, 0), dk, 1)
    return out


class Mind:
    def __init__(self, places=None):
        self._origin = None            # the odometry's first reading: his own coordinates start where he woke up
        self._prev = None              # his own pose from the odometry the last time (x, y, yaw)
        self.mu = None                 # EKF: [x, y, yaw, landmark1 x, y, landmark2 x, y ...]
        self.cov = None
        self.index = {}                # landmark name -> position in mu
        self.fixes = 0                 # how many times a landmark corrected him (shown after a test)
        self.unsure = 0.0              # how much the last landmark sighting had to correct him (m)
        self.things = {}               # name -> {"pos": (x, y) (None for landmarks: the filter has them), "t", "kind", "n"}
        self.floor = np.zeros(SIZE, np.float32)    # > 0: something there, < 0: free floor, 0: never looked there
        self._drawn = None             # where he was the last time he drew into the floor plan
        self.places = dict(places or {})           # name -> {"where": "by the bookshelf", "near": "bookshelf", "date": ...}
        self.places_changed = False
        self._votes = {}
        self.wall_ref = None           # the room's main direction (degrees, 0-90, in his own coordinates), once he has seen enough walls
        self._wall_runs = []           # (direction mod 90, length) of the straight runs seen so far
        self.wall_log = []             # (time, how far the walls say his heading is off in degrees, weight) - for the test report
        self.wall_fixes = 0
        self.moved = None             # (name, old place, new place) when a thing was found somewhere else

    # ------------------------------------------------------------ where am I
    def _own(self, odom):
        """The odometry (x, y, yaw in the duck's own odometry frame) in his own coordinates (0, 0, 0 = where he woke up)."""
        if self._origin is None:
            self._origin = odom
        ox, oy, oyaw = self._origin
        dx, dy = odom[0] - ox, odom[1] - oy
        c, s = math.cos(oyaw), math.sin(oyaw)
        return (c * dx + s * dy, -s * dx + c * dy, wrap(odom[2] - oyaw))

    def predict(self, odom):
        """Step counting: move the estimate by what the odometry says happened since last time, and grow the
        uncertainty (legs ~6 % of the distance along, ~4 % sideways; the gyro measures turns to 0.5 deg + 3 %)."""
        own = self._own(odom)
        if self.mu is None:
            self.mu = np.array(own, float)
            self.cov = np.eye(3) * 1e-6
            self._prev = own
            return
        px, py, pth = self._prev
        self._prev = own
        ddx, ddy = own[0] - px, own[1] - py
        dxl, dyl = math.cos(pth) * ddx + math.sin(pth) * ddy, -math.sin(pth) * ddx + math.cos(pth) * ddy
        dth = wrap(own[2] - pth)
        d = math.hypot(dxl, dyl)
        if d < 1e-6 and abs(dth) < 1e-6:
            return
        mu, P = self.mu, self.cov
        th = mu[2]
        c, s = math.cos(th), math.sin(th)
        mu[0] += c * dxl - s * dyl
        mu[1] += s * dxl + c * dyl
        mu[2] = wrap(mu[2] + dth)
        G = np.eye(3)
        G[0, 2] = -s * dxl - c * dyl
        G[1, 2] = c * dxl - s * dyl
        sd_along, sd_side = 0.06 * d + 0.005, 0.04 * d + 0.005
        sd_th = math.radians(0.05 * math.sqrt(d / 0.05) + (0.5 + 0.03 * abs(math.degrees(dth)) if abs(dth) > math.radians(1) else 0.0))
        Rm = np.array([[c, -s], [s, c]])
        Q = np.zeros((3, 3))
        Q[:2, :2] = Rm @ np.diag([sd_along ** 2, sd_side ** 2]) @ Rm.T
        Q[2, 2] = sd_th ** 2
        P[:3, :] = G @ P[:3, :]
        P[:, :3] = P[:, :3] @ G.T
        P[:3, :3] += Q

    def pose(self):
        """His own idea of where he is: (x, y, yaw in radians) - the filter's estimate, or None before the first reading."""
        return None if self.mu is None else (float(self.mu[0]), float(self.mu[1]), float(self.mu[2]))

    def pos_error(self):
        """How sure the filter is of his position (one standard deviation, in m)."""
        if self.cov is None:
            return 0.0
        return float(math.sqrt(max(self.cov[0, 0] + self.cov[1, 1], 0.0)))

    def _see_landmark(self, name, bearing, d, head):
        """One landmark sighting: angle +-2.5 deg, distance +-8 % (+-20 % beyond 4 m)."""
        mu, P = self.mu, self.cov
        b = bearing
        sr = (0.08 if d < 4 else 0.2) * d + 0.02
        Rm = np.diag([sr ** 2, math.radians(2.5) ** 2])
        if name not in self.index:                      # first sighting: put it on the map
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
            self.mu = np.concatenate([mu, [lx, ly]])
            self.cov = P2
            self.index[name] = n
            return
        j = self.index[name]
        dx, dy = mu[j] - mu[0], mu[j + 1] - mu[1]
        q = dx * dx + dy * dy
        r = math.sqrt(q)
        if r < 0.2:
            return
        z_hat = np.array([r, math.atan2(dy, dx) - mu[2] - head])
        innov = np.array([d, b]) - z_hat
        innov[1] = wrap(innov[1])
        n = len(mu)
        H = np.zeros((2, n))
        H[0, [0, 1, j, j + 1]] = [-dx / r, -dy / r, dx / r, dy / r]
        H[1, [0, 1, 2, j, j + 1]] = [dy / q, -dx / q, -1, -dy / q, dx / q]
        S = H @ P @ H.T + Rm
        if innov @ np.linalg.solve(S, innov) > 13.8:     # a sighting that doesn't fit at all: ignore it
            return
        K = P @ H.T @ np.linalg.inv(S)
        self.mu = mu + K @ innov
        self.mu[2] = wrap(self.mu[2])
        P2 = (np.eye(n) - K @ H) @ P
        self.cov = (P2 + P2.T) / 2
        self.fixes += 1

    # ------------------------------------------------------------ walls tell which way is which
    def wall_heading(self, rays, now=None):
        """Walls and big furniture usually stand at right angles to each other. Straight runs of depth points (a wall, a
        sofa side) are compared with the room's main direction: if they all look a few degrees tilted, his own sense of
        direction is off. Ported from the 2D simulator (body._wall_heading). rays = what add_scan gets (angle relative to
        his body + = left, free distance, distance to something standing there or None). Returns how many degrees the walls
        say his heading is off (+ = he thinks he faces too far left), or None. Only moves the filter if WALL_HEADING is on."""
        if self.mu is None or not rays:
            return None
        now = time.time() if now is None else now
        pts = sorted(((a, h) for a, _f, h in rays if h is not None and h < 3.5), key=lambda p: p[0])
        if len(pts) < 4:
            return None
        xy = [(h * math.cos(a), h * math.sin(a)) for a, h in pts]
        runs, run = [], [xy[0]]
        for p_, q in zip(xy, xy[1:]):
            if math.hypot(q[0] - p_[0], q[1] - p_[1]) < 0.12 + 0.1 * math.hypot(*q):
                run.append(q)
            else:
                runs.append(run)
                run = [q]
        runs.append(run)
        yaw_deg = math.degrees(self.mu[2])
        world, weights = [], []
        for r in runs:
            if len(r) < 4:
                continue
            Pm = np.array(r)
            span = math.hypot(*(Pm[-1] - Pm[0]))
            if span < 0.4:
                continue
            c = Pm - Pm.mean(0)
            w_, v = np.linalg.eigh(c.T @ c)
            if w_[0] > 0.02 * w_[1]:                  # not straight enough (a corner, a table leg, a person)
                continue
            world.append(yaw_deg + math.degrees(math.atan2(v[1, 1], v[0, 1])))
            weights.append(span)
        if not world:
            return None
        if self.wall_ref is None:                     # the room's main direction: from all straight runs so far
            self._wall_runs += [(w % 90, sp) for w, sp in zip(world, weights)]
            del self._wall_runs[:-400]
            if len(self._wall_runs) >= ROOM_RUNS and sum(sp for _, sp in self._wall_runs) >= ROOM_RUNS * 0.8:
                self.wall_ref = self._room_direction([a for a, _ in self._wall_runs], [sp for _, sp in self._wall_runs])
            return None
        dev = np.array([((w - self.wall_ref + 45) % 90) - 45 for w in world])
        wts = np.array(weights)
        good = np.abs(dev) < 12
        if not good.any():
            return None
        delta = float(np.average(dev[good], weights=wts[good]))
        weight = float(wts[good].sum())
        self.wall_log.append((now, delta, weight))
        del self.wall_log[:-300]
        if WALL_HEADING:
            sd = math.radians(WALL_SD / math.sqrt(weight))
            P = self.cov
            innov = -math.radians(delta)
            S = P[2, 2] + sd ** 2
            if innov ** 2 / S <= 9:
                K = P[:, 2] / S
                self.mu = self.mu + K * innov
                self.mu[2] = wrap(self.mu[2])
                P2 = P - np.outer(K, P[2, :])
                self.cov = (P2 + P2.T) / 2
                self.wall_fixes += 1
        return delta

    @staticmethod
    def _room_direction(angles, weights):
        """The room's main direction (0-90 degrees) from many straight runs: the most common direction, then the weighted
        average of the runs that agree with it within 5 degrees (a slanted thing or a mistake is left out)."""
        a = np.array(angles) % 90
        w = np.array(weights, float)
        best, best_w = 0.0, -1.0
        for c in np.arange(0, 90, 1.0):
            dev = ((a - c + 45) % 90) - 45
            sw = w[np.abs(dev) < 5].sum()
            if sw > best_w:
                best, best_w = c, sw
        dev = ((a - best + 45) % 90) - 45
        keep = np.abs(dev) < 5
        return float((best + np.average(dev[keep], weights=w[keep])) % 90)

    # ------------------------------------------------------------ where are things
    def see(self, sightings, head, now=None):
        """What the detector reports: [(name, kind, bearing_deg (relative to where the camera points, + = left), distance)].
        `head` = how far his head is turned (radians, + = left). Landmarks correct his position first; then everything
        goes into the mental map from where he now believes he is."""
        if self.mu is None:
            return
        now = time.time() if now is None else now
        self.unsure = 0.0
        before = self.pose()
        for name, kind, bearing, d in sightings:
            if kind == "landmark":
                self._see_landmark(name, math.radians(bearing), d, head)
        after = self.pose()
        self.unsure = math.hypot(after[0] - before[0], after[1] - before[1])
        x, y, yaw = after
        steady = self.unsure < STEADY
        for name, kind, bearing, d in sightings:
            if kind == "landmark":
                self.things[name] = {"pos": None, "t": now, "kind": "landmark", "n": self.things.get(name, {}).get("n", 0) + 1}
                continue
            if not steady:
                continue
            a = yaw + head + math.radians(bearing)
            guess = (x + d * math.cos(a), y + d * math.sin(a))
            old = self.things.get(name)
            if old and now - old["t"] < 120 and old.get("pos") and not old.get("kicked"):
                guess = (old["pos"][0] * 0.7 + guess[0] * 0.3, old["pos"][1] * 0.7 + guess[1] * 0.3)
            self.things[name] = {"pos": guess, "t": now, "kind": "thing", "n": (old or {}).get("n", 0) + 1}
        self._update_places(now)

    def pos_of(self, name):
        """Where he believes the thing is, in his own coordinates, or None."""
        if name in self.index:
            j = self.index[name]
            return float(self.mu[j]), float(self.mu[j + 1])
        t = self.things.get(name)
        return t["pos"] if t and t.get("pos") else None

    def forget(self, name):
        self.things.pop(name, None)

    def note_kicked(self, name, now=None):
        """He kicked it: where he last saw it stays in his memory, but it has probably rolled away since."""
        t = self.things.get(name)
        if t and t.get("pos"):
            t["kicked"] = time.time() if now is None else now

    def from_me(self, pos):
        """(distance in m, angle in degrees: + = to his left) from where he believes he is."""
        x, y, yaw = self.pose()
        dx, dy = pos[0] - x, pos[1] - y
        return math.hypot(dx, dy), math.degrees(wrap(math.atan2(dy, dx) - yaw))

    def landmarks(self):
        return {n: self.pos_of(n) for n in self.index}

    def things_words(self, now=None, limit=10, head_pitch=None):
        """The mental map in words: what he remembers seeing, where (from where he thinks he is) and how long ago."""
        if self.mu is None:
            return []
        now = time.time() if now is None else now
        out = []
        for name, t in sorted(self.things.items(), key=lambda kv: -kv[1]["t"])[:limit]:
            pos = self.pos_of(name)
            if pos is None:
                continue
            d, rel = self.from_me(pos)
            s = now - t["t"]
            ago = "just now" if s < 20 else f"{int(s // 60)} min ago" if s >= 60 else f"{int(s)} s ago"
            where = "right here" if d < 0.3 else f"{about(d)} {side(rel)}" + (f" [{abs(rel):.0f}° to your {'left' if rel > 0 else 'right'}]" if abs(rel) > 12 else "")
            too_close = ""
            if head_pitch is not None and t["kind"] == "thing" and d < 0.45 and abs(rel) < 40 and head_pitch > math.radians(-20):
                too_close = " – so close to your feet that your camera can't see it with your head level (look down to see it)"
            kicked = " – you KICKED it after that, so it has probably rolled away from there (look around for it)" if t.get("kicked") else ""
            out.append(f"the {name} – {where} (seen {ago}){too_close}{kicked}")
        return out

    # ------------------------------------------------------------ places in words, kept between days
    def _words_for(self, pos):
        near = None
        for lm, p in self.landmarks().items():
            dd = math.hypot(pos[0] - p[0], pos[1] - p[1])
            if near is None or dd < near[1]:
                near = (lm, dd)
        if near is None:
            return None, None
        lm, dd = near
        where = f"by the {lm}" if dd < 0.9 else f"near the {lm}" if dd < 1.8 else "in the middle of the room"
        return where, (lm if dd < 1.8 else None)

    def _update_places(self, now):
        """A thing he sees gets a place in words. Seen somewhere else, it only counts as moved after he has seen it there
        several times in a row (so a wobble in his map doesn't make things move)."""
        if len(self.index) < 2:
            return                                   # too early: he doesn't know the room well enough to say where things are
        today = time.strftime("%Y-%m-%d")
        for name, t in self.things.items():
            if t["kind"] != "thing" or not t.get("pos") or now - t["t"] > 2:
                continue
            where, near = self._words_for(t["pos"])
            if where is None:
                continue
            old = self.places.get(name)
            if old and old.get("near") == near:
                where = old["where"]                 # same landmark: "by" vs "near" wobbling is not a move
            if old and old.get("where") != where:
                w, n, t0 = self._votes.get(name, (None, 0, now))
                if w != where:
                    n, t0 = 0, now
                if self.from_me(t["pos"])[0] < 3.0:          # (far away his distance guess is too rough to say it moved)
                    n += 1
                self._votes[name] = (where, n, t0)
                if n < VOTES_TO_MOVE or now - t0 < 4.0:
                    continue
            self._votes.pop(name, None)
            if not old or old.get("where") != where:
                self.places[name] = {"where": where, "near": near, "date": today}
                self.places_changed = True
                self.moved = (name, old.get("where") if old else None, where) if old else None
            elif old.get("date") != today:
                old["date"] = today
                self.places_changed = True

    # ------------------------------------------------------------ his own floor plan
    def cell(self, x, y):
        return int((x - ORIGIN[0]) / CELL), int((y - ORIGIN[1]) / CELL)

    def centre(self, c):
        return ORIGIN[0] + (c[0] + 0.5) * CELL, ORIGIN[1] + (c[1] + 0.5) * CELL

    def add_scan(self, rays, now=None, force=False):
        """Draws one look into the floor plan from where he believes he is.
        rays = [(angle in radians relative to where his body faces (+ = left), free distance in m, distance to something
        standing there in m or None)]. Only after he has moved 3 cm / turned 3 deg or a second has passed (looking at the
        same thing again and again would only wear the plan out)."""
        if self.mu is None or not rays:
            return False
        now = time.time() if now is None else now
        x, y, yaw = self.pose()
        if self._drawn is not None and not force:
            px, py, pyaw, pt = self._drawn
            if (math.hypot(x - px, y - py) < DRAW_EVERY_M and abs(math.degrees(wrap(yaw - pyaw))) < DRAW_EVERY_DEG
                    and now - pt < DRAW_EVERY_S):
                return False
        self._drawn = (x, y, yaw, now)
        W, H = SIZE
        for ang, free, hit in rays:
            a = yaw + ang
            ca, sa = math.cos(a), math.sin(a)
            n = int(max(free, 0) / CELL)
            if n > 0:
                ds = (np.arange(1, n + 1) * CELL).astype(np.float32)
                I = ((x + ds * ca - ORIGIN[0]) / CELL).astype(np.int32)
                K = ((y + ds * sa - ORIGIN[1]) / CELL).astype(np.int32)
                ok = (I >= 0) & (I < W) & (K >= 0) & (K < H)
                w = np.maximum(FREE_MIN, 1.0 - ds[ok] / FREE_FADE)
                self.floor[I[ok], K[ok]] = np.maximum(self.floor[I[ok], K[ok]] + FREE_STEP * w, -5.0)
            if hit is not None and hit < 4.0:
                i, k = self.cell(x + hit * ca, y + hit * sa)
                if 0 <= i < W and 0 <= k < H:
                    self.floor[i, k] = min(self.floor[i, k] + HIT_STEP, 5.0)
        return True

    def blocked_grid(self, extra=0.03):
        """Squares he believes are taken, grown by his own size so a route through free squares fits him."""
        blocked = self.floor > BLOCKED
        return _dilate(blocked, int(math.ceil((DUCK_RADIUS + extra) / CELL)))

    def plan(self, goal, stop=0.3):
        """A route on his own floor plan (unknown squares count as free - he hasn't seen anything there) from where he
        believes he is to within `stop` m of `goal`. Returns way points in his own coordinates, or None."""
        blocked = self.blocked_grid()
        W, H = SIZE
        x, y, _ = self.pose()
        start, g = self.cell(x, y), self.cell(*goal)
        if not (0 <= start[0] < W and 0 <= start[1] < H and 0 <= g[0] < W and 0 <= g[1] < H):
            return None
        rstop = stop / CELL
        prev, queue, end = {start: None}, deque([start]), None
        while queue:
            c = queue.popleft()
            if (c[0] - g[0]) ** 2 + (c[1] - g[1]) ** 2 <= rstop ** 2:
                end = c
                break
            for di, dk in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nb = (c[0] + di, c[1] + dk)
                if nb not in prev and 0 <= nb[0] < W and 0 <= nb[1] < H and (not blocked[nb] or (nb[0] - start[0]) ** 2 + (nb[1] - start[1]) ** 2 <= 16):
                    prev[nb] = c
                    queue.append(nb)
        if end is None:
            return None
        cells = []
        while end is not None:
            cells.append(end)
            end = prev[end]
        cells.reverse()

        def clear(a, b):
            return all(not blocked[int(a[0] + (b[0] - a[0]) * t / 20), int(a[1] + (b[1] - a[1]) * t / 20)] for t in range(21))

        route, k = [], 0
        while k < len(cells) - 1:                        # straight lines where his map allows
            far = len(cells) - 1
            while far > k + 1 and not clear(cells[k], cells[far]):
                far -= 1
            route.append(self.centre(cells[far]))
            k = far
        return route or [goal]

    def frontier(self, min_dist=1.0):
        """The nearest edge of what he has explored: free floor next to squares he has never looked at.
        Returns ((x, y), distance, angle in degrees + = left) from where he believes he is, or None."""
        known_free = self.floor < -0.5
        unknown = np.abs(self.floor) < 0.05
        edge = known_free & (np.roll(unknown, 1, 0) | np.roll(unknown, -1, 0) | np.roll(unknown, 1, 1) | np.roll(unknown, -1, 1))
        edge[:3, :] = edge[-3:, :] = edge[:, :3] = edge[:, -3:] = False
        edge &= ~self.blocked_grid(extra=0.0)
        idx = np.argwhere(edge)
        if not len(idx) or self.mu is None:
            return None
        xs = ORIGIN[0] + (idx[:, 0] + 0.5) * CELL
        ys = ORIGIN[1] + (idx[:, 1] + 0.5) * CELL
        x, y, yaw = self.pose()
        dd = np.hypot(xs - x, ys - y)
        far_enough = dd > min_dist
        if not far_enough.any():
            return None
        i = int(np.argmin(np.where(far_enough, dd, np.inf)))
        return (float(xs[i]), float(ys[i])), float(dd[i]), math.degrees(wrap(math.atan2(ys[i] - y, xs[i] - x) - yaw))

    def floor_words(self):
        explored = float((self.floor < -0.5).sum()) * CELL ** 2
        f = self.frontier()
        return (f"you have explored about {explored:.0f} m² of floor"
                + (f"; the nearest place you haven't explored is {about(f[1])} {side(f[2])}" if f
                   else "; you don't know of any unexplored places nearby"))

    def explored(self):
        return float((self.floor < -0.5).sum()) * CELL ** 2
