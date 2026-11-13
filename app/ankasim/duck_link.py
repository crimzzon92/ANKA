"""duck_link: Anka's connection to Pollen's simulated Microduck (and later, to the real one - it speaks the same way).

What it gives you
    link = DuckLink()                 # connects in the background; keeps trying if the simulator is not up yet
    link.frame()                      # the camera picture, upright (BGR, like OpenCV), or None
    link.tof()                        # the 8x8 depth sensor: {"m": 8x8 metres (nan = nothing seen), "status": ..., ...}
    link.state()                      # robotd's state: odometry, IMU, head, joints, fallen ...
    link.move(0.15, 0, 0)             # walk forward at 0.15 m/s (keeps sending, as robotd wants, until you stop it)
    link.look_at(0.5, 0, 0.0)         # point the head/camera at a spot (metres, in the trunk's frame)
    link.do("sit_toggle")             # a skill

Where the data comes from (the same sockets a real duck has, so nothing here is special for the simulator):
    robotd   ~/.cache/duck-sim/duck-a.sock       JSON-RPC: commands in, robot.state out (odometry, IMU, joints)
    tofd     ~/.cache/duck-sim/duck-a-tof.sock   JSON-RPC: tof.frame (8x8 depth)
    camera   tcp 127.0.0.1:7901                  raw UYVY frames, 4 bytes length first (only in the simulator)
    "truth"  tcp 127.0.0.1:7801                  the simulator's own body port: where the duck REALLY is. A real duck can
                                                 not know this - it is only here to check how well his own senses do.

Everything runs in threads, so the rest of Anka never waits for the simulator.
"""
import json
import math
import os
import socket
import struct
import threading
import time

import numpy as np

try:
    import cv2
except ImportError:          # the link works without it (no camera picture then)
    cv2 = None

try:
    from . import room_layout as R
except ImportError:
    import room_layout as R

STATE_DIR = os.environ.get("DUCK_SIM_STATE", os.path.expanduser("~/.cache/duck-sim"))
TOF_VALID = (5, 9)           # ST's two "valid" status codes (255 = nothing in range)
CAMERA_TURN = "cw"           # the simulator's frames come out on their side: turning them clockwise puts them upright
SIZES = ((640, 360), (320, 180), (160, 90), (1280, 720), (960, 540), (480, 270), (1920, 1080))


def _yaw_of(quat):
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def decode_uyvy(buf):
    """Raw UYVY bytes -> BGR picture as the camera made it (still on its side), or None."""
    if cv2 is None:
        return None
    for w, h in SIZES:
        if w * h * 2 == len(buf):
            arr = np.frombuffer(buf, np.uint8).reshape(h, w, 2)
            return cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_UYVY)
    return None


def upright(bgr):
    """Turns the camera's sideways picture so that up is up."""
    if bgr is None or cv2 is None:
        return bgr
    if CAMERA_TURN == "cw":
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if CAMERA_TURN == "ccw":
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr


class DuckLink:
    def __init__(self, state_dir=None, duck="duck-a", host="127.0.0.1", body_port=7801, camera_port=7901,
                 camera=True, truth=True, state_hz=20):
        self.state_dir, self.duck, self.host = state_dir or STATE_DIR, duck, host
        self.body_port, self.camera_port = body_port, camera_port
        self.want = {"camera": camera, "truth": truth}
        self.state_hz = state_hz
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._wlock = threading.Lock()                  # one writer at a time on the robot socket
        self._state, self._state_t = None, 0.0
        self._gyro_yaw, self._gyro_t, self._gyro_n = 0.0, None, 0     # his heading by adding up the IMU gyro (a real duck has one)
        self._tof, self._tof_t = None, 0.0
        self._raw, self._frame, self._frame_t, self._frames = None, None, 0.0, 0
        self._truth, self._truth_t, self._realtime = None, 0.0, None
        self._ack = {}                                  # what robot.subscribe told us (which skills there are ...)
        self._pending, self._ids = {}, 0
        self._rsock = None
        self._move, self._move_until = None, 0.0         # (vx, vy, vyaw) we keep sending, and until when
        self._head, self._head_until = None, 0.0
        self._errors = {"robot": "not tried yet", "tof": "not tried yet", "camera": "not tried yet", "truth": "not tried yet"}
        self._threads = []

    # ---------------- life ----------------
    def start(self):
        if self._threads:
            return self
        loops = [self._robot_loop, self._intent_loop, self._tof_loop]
        if self.want["camera"]:
            loops.append(self._camera_loop)
        if self.want["truth"]:
            loops.append(self._truth_loop)
        for fn in loops:
            t = threading.Thread(target=fn, daemon=True, name=f"duck_link.{fn.__name__}")
            t.start()
            self._threads.append(t)
        return self

    def close(self):
        self._stop.set()
        try:
            self.stop()
        except Exception:
            pass

    def wait_ready(self, timeout=5.0):
        """Waits until what is going to connect has connected. Returns {"robot": bool, "tof": bool, ...}."""
        end = time.time() + timeout
        while time.time() < end:
            s = self.connected()
            if s["robot"] and s["tof"] and (s["camera"] or not self.want["camera"]):
                break
            time.sleep(0.05)
        return self.connected()

    def connected(self, fresh=3.0):
        now = time.time()
        return {"robot": now - self._state_t < fresh, "tof": now - self._tof_t < fresh,
                "camera": now - self._frame_t < fresh, "truth": now - self._truth_t < fresh}

    def errors(self):
        return {k: v for k, v in self._errors.items() if v}

    # ---------------- what the duck senses ----------------
    def state(self):
        """robotd's latest robot.state as a dict (or None): odom, imu, head, joints, safety, policy ..."""
        with self._lock:
            return self._state

    def _add_gyro(self, st, now):
        """Adds up the IMU's turn rate around 'up' (rad/s, in the world's own seconds: the simulator may run slower than the
        clock) into a heading. Used by the test report to show how much the odometry's heading can be trusted; None if there is no gyro."""
        try:
            imu = st.get("imu") or {}
            gyro, g = imu.get("gyro"), (st.get("safety") or {}).get("gravity") or imu.get("gravity")
            if not gyro or not g or len(gyro) != 3 or len(g) != 3:
                self._gyro_t = None
                return
            n = math.sqrt(sum(c * c for c in g)) or 1.0
            rate = -sum(a * b for a, b in zip(gyro, g)) / n          # turn rate around up (+ = left)
            if self._gyro_t is not None:
                dt = min(now - self._gyro_t, 0.2) * (self._realtime or 1.0)
                self._gyro_yaw += rate * dt
                self._gyro_n += 1
            self._gyro_t = now
        except Exception:
            self._gyro_t = None

    def gyro_yaw(self):
        """His heading by adding up the IMU gyro since the start (radians), or None if the robot sends no gyro."""
        return self._gyro_yaw if self._gyro_n > 10 else None

    def tof(self):
        """{"m": 8x8 array of metres (nan where nothing was seen), "mm", "status", "age": seconds} or None."""
        with self._lock:
            t = self._tof
        if t is None:
            return None
        return dict(t, age=time.time() - self._tof_t)

    def frame(self):
        """The camera picture, upright, BGR - or None (no camera, or not here yet)."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def frame_info(self):
        return {"age": time.time() - self._frame_t if self._frame_t else None, "count": self._frames}

    def truth(self):
        """Where the duck REALLY is in the simulator (a real duck cannot know this - for checking only):
        {"x", "y", "heading"} in Anka's room coordinates, plus the raw MuJoCo numbers."""
        with self._lock:
            t = self._truth
        return t

    def realtime(self):
        """How fast the simulated world runs compared to the clock (1.0 = real time; below ~0.95 the duck cannot balance)."""
        return self._realtime

    def odom(self):
        """(x, y, yaw_radians) in the duck's own odometry (0, 0, 0 = where it was switched on), or None."""
        s = self.state()
        if not s or "odom" not in s:
            return None
        o = s["odom"]
        return o["position"][0], o["position"][1], o["yaw"]

    def odom_anka(self):
        """The odometry in Anka's room coordinates (x east, y north, heading in degrees) - assumes he started at
        the room's start spot, which is where the simulator puts him."""
        o = self.odom()
        if o is None:
            return None
        ax, ay = R.from_mj(o[0], o[1])
        return ax, ay, R.heading_from_mj_yaw(o[2])

    def fallen(self):
        s = self.state()
        return bool(s and s.get("safety", {}).get("fallen"))

    def skills(self):
        return list(self._ack.get("skills", []))

    # ---------------- commands ----------------
    def move(self, vx=0.0, vy=0.0, vyaw=0.0, seconds=None):
        """Walk: forward m/s, left m/s, yaw rad/s (positive = left). Keeps sending until stop() or `seconds` have passed."""
        self._move = (float(vx), float(vy), float(vyaw))
        self._move_until = time.time() + seconds if seconds else float("inf")

    def stop(self):
        self._move, self._move_until = None, 0.0
        self._notify("robot.move", {"vx": 0.0, "vy": 0.0, "vyaw": 0.0})

    def head(self, neck_pitch=0.0, head_pitch=0.0, head_yaw=0.0, head_roll=0.0, seconds=None):
        """Head joints in radians; held (re-sent) until head(None) or `seconds`."""
        self._head = dict(neck_pitch=neck_pitch, head_pitch=head_pitch, head_yaw=head_yaw, head_roll=head_roll)
        self._head_until = time.time() + seconds if seconds else float("inf")

    def head_free(self):
        self._head = None

    def pose(self, z=0.0, roll=0.0, pitch=0.0, active=True):
        """The standing body's pose (2026-11-10): height offset in metres (negative = crouch), roll and pitch in radians.
        While active the robot stands (walking is zeroed) - so switch it off (active=False) before any walking or skill."""
        self._notify("robot.pose", {"z": float(z), "roll": float(roll), "pitch": float(pitch), "active": bool(active)})

    def mouth(self, open_=0.0):
        """The beak, 0 closed to 1 open (not part of any policy - only this moves it)."""
        self._notify("robot.mouth", {"open": max(0.0, min(1.0, float(open_)))})

    def look_at(self, x, y, z, neck_pitch=0.0):
        """Point the camera at (x forward, y left, z up) metres in the trunk frame; the floor is about 0.12 m below.
        The robot solves the head angles; we keep holding them. Returns the answer ({"head": {...}, "clamped": bool})."""
        res = self.call("robot.look", {"x": x, "y": y, "z": z, "neck_pitch": neck_pitch})
        if isinstance(res, dict) and isinstance(res.get("head"), dict):
            h = res["head"]
            self.head(h.get("neck_pitch", 0.0), h.get("head_pitch", 0.0), h.get("head_yaw", 0.0), h.get("head_roll", 0.0))
        return res

    def do(self, skill):
        """Run a skill by name: sit_toggle, ground_pick, ball_kick_left, ball_kick_right ... (see skills())."""
        return self.call("robot.do", {"skill": skill})

    def enable(self, on=True):
        return self.call("robot.enable", {"on": bool(on)})

    # ---------------- robotd: JSON-RPC over a unix socket ----------------
    def _next_id(self):
        self._ids += 1
        return self._ids

    def _send(self, obj):
        s = self._rsock
        if s is None:
            raise ConnectionError("robotd is not connected")
        data = (json.dumps(obj) + "\n").encode()
        with self._wlock:
            s.sendall(data)

    def _notify(self, method, params=None):
        try:
            self._send({"jsonrpc": "2.0", "method": method, **({"params": params} if params is not None else {})})
        except (OSError, ConnectionError):
            pass

    def call(self, method, params=None, timeout=3.0):
        """A request that wants an answer. Returns the result, or {"error": ...}."""
        rid = self._next_id()
        ev = threading.Event()
        self._pending[rid] = [ev, None]
        try:
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, **({"params": params} if params is not None else {})})
        except (OSError, ConnectionError) as e:
            self._pending.pop(rid, None)
            return {"error": str(e)}
        if not ev.wait(timeout):
            self._pending.pop(rid, None)
            return {"error": f"no answer to {method} within {timeout} s"}
        return self._pending.pop(rid)[1]

    def _robot_loop(self):
        path = os.path.join(self.state_dir, f"{self.duck}.sock")
        while not self._stop.is_set():
            try:
                s = socket.socket(socket.AF_UNIX)
                s.settimeout(2.0)
                s.connect(path)
                s.settimeout(None)
            except OSError as e:
                self._errors["robot"] = f"cannot reach {path}: {e}"
                time.sleep(1.0)
                continue
            self._rsock = s
            self._errors["robot"] = None
            try:
                # subscribe from a helper thread: this thread must be free to read the answer
                threading.Thread(target=self._subscribe, daemon=True).start()
                for line in s.makefile("rb"):
                    try:
                        msg = json.loads(line)
                    except ValueError:
                        continue
                    if "method" in msg:
                        if msg["method"] == "robot.state" and isinstance(msg.get("params"), dict):
                            with self._lock:
                                self._state, self._state_t = msg["params"], time.time()
                                self._add_gyro(msg["params"], self._state_t)
                        continue
                    rid = msg.get("id")
                    if rid in self._pending:
                        self._pending[rid][1] = msg.get("result") if "result" in msg else {"error": msg.get("error")}
                        self._pending[rid][0].set()
            except OSError:
                pass
            finally:
                self._rsock = None
                try:
                    s.close()
                except OSError:
                    pass
            self._errors["robot"] = "connection to robotd lost"
            time.sleep(0.5)

    def _subscribe(self):
        res = self.call("robot.subscribe", {"hz": self.state_hz})
        if isinstance(res, dict) and "error" not in res:
            self._ack = res

    def _intent_loop(self):
        """robotd drops a movement command 500 ms after the last one arrives, so keep sending (20 Hz)."""
        while not self._stop.is_set():
            now = time.time()
            if self._move is not None:
                if now < self._move_until:
                    vx, vy, vyaw = self._move
                    self._notify("robot.move", {"vx": vx, "vy": vy, "vyaw": vyaw})
                else:
                    self._move = None
                    self._notify("robot.move", {"vx": 0.0, "vy": 0.0, "vyaw": 0.0})
            if self._head is not None:
                if now < self._head_until:
                    self._notify("robot.head", self._head)
                else:
                    self._head = None
            time.sleep(0.05)

    # ---------------- tofd ----------------
    def _tof_loop(self):
        path = os.path.join(self.state_dir, f"{self.duck}-tof.sock")
        while not self._stop.is_set():
            try:
                s = socket.socket(socket.AF_UNIX)
                s.settimeout(2.0)
                s.connect(path)
                s.settimeout(None)
                s.sendall((json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tof.stream"}) + "\n").encode())
            except OSError as e:
                self._errors["tof"] = f"cannot reach {path}: {e}"
                time.sleep(1.0)
                continue
            self._errors["tof"] = None
            try:
                for line in s.makefile("rb"):
                    try:
                        msg = json.loads(line)
                    except ValueError:
                        continue
                    p = msg.get("params")
                    if msg.get("method") == "tof.frame" and isinstance(p, dict):
                        rows, cols = int(p.get("rows", 8)), int(p.get("cols", 8))
                        mm = np.array(p.get("distance_mm", []), dtype=np.float32)
                        st = np.array(p.get("status", []), dtype=np.int32)
                        if mm.size != rows * cols or st.size != rows * cols:
                            continue
                        m = np.where(np.isin(st, TOF_VALID), mm / 1000.0, np.nan).reshape(rows, cols)
                        with self._lock:
                            self._tof = {"m": m, "mm": mm.reshape(rows, cols), "status": st.reshape(rows, cols),
                                         "seq": p.get("seq", 0), "rows": rows, "cols": cols}
                            self._tof_t = time.time()
                    elif "error" in msg:
                        self._errors["tof"] = f"tofd says: {msg['error']}"
            except OSError:
                pass
            finally:
                try:
                    s.close()
                except OSError:
                    pass
            self._errors["tof"] = self._errors["tof"] or "connection to tofd lost"
            time.sleep(0.5)

    # ---------------- the camera ----------------
    def _camera_loop(self):
        while not self._stop.is_set():
            try:
                s = socket.create_connection((self.host, self.camera_port), timeout=2.0)
                s.settimeout(5.0)
            except OSError as e:
                self._errors["camera"] = (f"no camera on port {self.camera_port} ({e}) - was the simulator started "
                                          "with a camera (DUCK_SIM_CAMERAS=a)?")
                time.sleep(1.5)
                continue
            self._errors["camera"] = None
            try:
                while not self._stop.is_set():
                    head = self._read_exact(s, 4)
                    (n,) = struct.unpack("<I", head)
                    if n > 64 * 1024 * 1024:
                        raise OSError("absurd frame size")
                    buf = self._read_exact(s, n)
                    raw = decode_uyvy(buf)
                    if raw is None:
                        self._errors["camera"] = f"a frame of {n} bytes is not a size I know"
                        continue
                    with self._lock:
                        self._raw, self._frame = raw, upright(raw)
                        self._frame_t, self._frames = time.time(), self._frames + 1
            except (OSError, struct.error) as e:
                self._errors["camera"] = f"camera connection lost: {e}"
            finally:
                try:
                    s.close()
                except OSError:
                    pass
            time.sleep(0.5)

    @staticmethod
    def _read_exact(s, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                raise OSError("closed")
            buf += chunk
        return bytes(buf)

    # ---------------- the simulator's own body port (truth, for checking) ----------------
    def _truth_loop(self):
        while not self._stop.is_set():
            try:
                s = socket.create_connection((self.host, self.body_port), timeout=2.0)
                s.settimeout(3.0)
                f = s.makefile("rw")
            except OSError as e:
                self._errors["truth"] = f"cannot reach the simulator on port {self.body_port}: {e}"
                time.sleep(1.5)
                continue
            self._errors["truth"] = None
            last = None
            try:
                while not self._stop.is_set():
                    f.write('{"op":"read"}\n')
                    f.flush()
                    msg = json.loads(f.readline())
                    if "trunk" in msg and "imu" in msg:
                        x, y = msg["trunk"][0], msg["trunk"][1]
                        ax, ay = R.from_mj(x, y)
                        yaw = _yaw_of(msg["imu"]["quat"])
                        with self._lock:
                            self._truth = {"x": ax, "y": ay, "heading": R.heading_from_mj_yaw(yaw),
                                           "mj": (x, y, msg["trunk"][2]), "yaw": yaw, "z": msg.get("trunk_z"),
                                           "sim_time": msg.get("sim_time")}
                            self._truth_t = time.time()
                        now = time.time()
                        if last is None or now - last[0] > 3.0:
                            if last is not None and msg.get("sim_time") is not None:
                                self._realtime = (msg["sim_time"] - last[1]) / (now - last[0])
                            last = (now, msg.get("sim_time"))
                    time.sleep(0.2)
            except (OSError, ValueError) as e:
                self._errors["truth"] = f"lost the simulator's body port: {e}"
            finally:
                try:
                    s.close()
                except OSError:
                    pass
            time.sleep(0.5)
