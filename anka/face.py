"""His face: the little duckling drawn next to the camera picture."""
import time

import numpy as np

from settings import FLIP_GAZE, SHOW_FACE, STRONG
from anka.shared import answering, cv2, listening, state



# What each expression looks like: eye size, how open the eyes are, eyebrows (raised, sad tilt), rosy cheeks
EXPRESSIONS = {
    # eyes: dot (round eyes), big (big shiny eyes), happy (^ ^), squint (> <), sleepy (closed, relaxed), half (bored)
    # mouth: how open the beak is when Anka is quiet. blush: how rosy the cheeks are. brows: raised (+) / sad tilt
    "neutral":   dict(eyes="dot", mouth=0.12, blush=0.6, brows=None, bounce=1.0),
    "happy":     dict(eyes="happy", mouth=0.45, blush=1.0, brows=None, bounce=1.4),
    "curious":   dict(eyes="big", mouth=0.1, blush=0.6, brows="up", bounce=1.0),
    "surprised": dict(eyes="big", mouth=0.75, blush=0.5, brows="high", bounce=1.0),
    "excited":   dict(eyes="squint", mouth=0.6, blush=1.0, brows=None, bounce=2.2),
    "sleepy":    dict(eyes="sleepy", mouth=0.05, blush=0.5, brows=None, bounce=0.5),
    "bored":     dict(eyes="half", mouth=0.0, blush=0.3, brows="flat", bounce=0.4),
    "sad":       dict(eyes="dot", mouth=0.0, blush=0.4, brows="sad", bounce=0.4),
    "thinking":  dict(eyes="dot", mouth=0.05, blush=0.5, brows="up", bounce=0.6),
    "love":      dict(eyes="heart", mouth=0.4, blush=1.0, brows=None, bounce=1.3),
    "shy":       dict(eyes="dot", mouth=0.05, blush=1.0, brows="up", bounce=0.6),     # looks away and blushes
    "scared":    dict(eyes="big", mouth=0.35, blush=0.2, brows="high", bounce=0.3),   # big eyes, trembles
    "annoyed":   dict(eyes="half", mouth=0.0, blush=0.3, brows="angry", bounce=0.5),
    "playful":   dict(eyes="wink", mouth=0.5, blush=0.9, brows=None, bounce=1.6),
}
HEART = (90, 70, 235)  # (blue, green, red) – for the heart eyes
mouth_track = {"start": 0.0, "levels": None}   # how loud Anka's voice is, 30 times a second, while it speaks
_face = {"blink_at": time.time() + 3, "drawn": 0.0, "camera": None, "tilt": 0.0, "expression": None,
         "changed_at": 0.0, "mouth": 0.0, "energy": 0.0, "level": 0.0, "nod_at": 0.0, "tuft": 0.0, "bob": 0.0,
         "stretch": 0.0}

# Colours (OpenCV uses blue, green, red)
FACE_BG = (240, 228, 205)
YELLOW, YELLOW_LIGHT, OUTLINE = (70, 206, 252), (118, 222, 254), (40, 60, 105)
BEAK, BEAK_DARK, MOUTH, TONGUE = (40, 140, 248), (30, 105, 215), (45, 40, 150), (95, 100, 238)
BLUSH, BLUSH_LINES, EYE = (175, 160, 250), (120, 105, 232), (40, 35, 45)


def track_mouth(sound, sr, delay=0.1, sentence=""):
    """Called when a sentence starts playing: remembers how loud it is moment by moment, so the beak moves,
    and what kind of sentence it is (a question gets a tilt at the end, an exclamation gets more bounce)."""
    sound = np.asarray(sound, dtype="float32")
    step = max(1, int(sr / 30))  # 30 values per second
    levels = np.array([np.sqrt(np.mean(sound[i:i + step] ** 2)) for i in range(0, len(sound), step)] or [0.0])
    mouth_track["levels"] = levels / (levels.max() + 1e-9)
    mouth_track["start"] = time.time() + delay   # afplay needs a moment to start
    mouth_track["per_second"] = sr / step
    mouth_track["length"] = len(levels) / mouth_track["per_second"]
    ending = sentence.strip()[-1:] if sentence.strip() else "."
    mouth_track["kind"] = ending if ending in "?!" else "."
    mouth_track["sway"] = float(np.random.choice([-1, 1]) * np.random.uniform(0.6, 1.0))  # which way to sway
    _face["blink_at"] = mouth_track["start"] + mouth_track["length"] + 0.05  # blink when the sentence ends


def mouth_open():
    levels = mouth_track["levels"]
    if levels is None or not speaking():
        return 0.0
    i = int((time.time() - mouth_track["start"]) * mouth_track.get("per_second", 30))
    return float(levels[i]) if 0 <= i < len(levels) else 0.0


def sentence_progress():
    """How far into the sentence Anka is right now: 0 = the start, 1 = the end (None when quiet)."""
    if mouth_track["levels"] is None or not speaking():
        return None
    p = (time.time() - mouth_track["start"]) / max(0.1, mouth_track.get("length", 1))
    return p if 0 <= p <= 1.2 else None


def current_expression():
    """The brain's own choice (FACE in its thoughts) for a while – otherwise Anka's strongest feeling."""
    chosen, since = state.get("face", (None, 0))
    if chosen in EXPRESSIONS and time.time() - since < 40:
        return chosen
    with feelings_lock:
        f = dict(feelings)
    if f.get("nervousness", 0) >= STRONG:
        return "scared"
    if f.get("affection", 0) >= 0.8 and f.get("happiness", 0) >= STRONG and human_is_here():
        return "love"
    if f.get("playfulness", 0) >= STRONG and human_is_here():
        return "playful"
    if f.get("happiness", 0) >= 0.8:
        return "happy"
    if f.get("boredom", 0) >= STRONG:
        return "bored"
    if f.get("loneliness", 0) >= STRONG:
        return "sad"
    if f.get("curiosity", 0) >= STRONG:
        return "curious"
    return "happy" if human_is_here() else "neutral"


def _ease(value, target, speed=0.25):
    return value + (target - value) * speed


def draw_face(size):
    """Draws Anka's face as a picture (size × size pixels): a round little duckling that bobs, breathes,
    blinks, looks at you and moves its beak with the voice. Drawn twice as big and scaled down for smooth lines."""
    now = time.time()
    S = size * 2
    img = np.full((S, S, 3), FACE_BG, dtype=np.uint8)
    aa = cv2.LINE_AA
    expression = current_expression()
    icon, look_x, look_y = None, 0.0, 0.0
    if state.get("gaze") is not None:
        look_x = (0.5 - state["gaze"]) * 2 * (-1 if FLIP_GAZE else 1)  # look at you
    else:
        look_x = 0.5 * np.sin(now / 3)  # nobody there – look around slowly
    tilt = 0.0
    if listening.is_set():
        expression, icon, look_x, tilt = "curious", "?", 0.0, 10.0
    elif (answering.is_set() or state.get("thinking")) and not speaking():
        expression, icon, look_x, look_y = "thinking", "...", 0.7, -0.8
    elif state.get("napping"):
        expression, icon, look_x, look_y = "sleepy", "z", 0.0, 0.3
    elif state.get("rest", 0) >= 2 and not human_is_here():
        expression, icon = "sleepy", "z"
    elif expression == "curious":
        tilt = 6.0
    look = EXPRESSIONS[expression]
    if expression == "shy":
        look_x, look_y, tilt = 2.2, 1.6, tilt - 9  # looks away, down to the side, and tilts the head
    if expression != _face["expression"]:
        _face["expression"], _face["changed_at"] = expression, now  # a little "pop" when the face changes

    # Talking: a nod on every stressed syllable, a gentle sway, a tilt at the end of a question,
    # more bounce and happy eyes on an exclamation, eyebrows that lift with the voice
    talk = mouth_open()
    progress = sentence_progress()
    talking = progress is not None
    kind = mouth_track.get("kind", ".")
    sway = mouth_track.get("sway", 1.0)
    _face["energy"] = _ease(_face["energy"], talk if talking else 0.0, 0.12)  # how lively the sentence is
    energy = _face["energy"]
    if talk - _face["level"] > 0.3 and now - _face["nod_at"] > 0.25:
        _face["nod_at"] = now  # a new loud syllable – nod
    _face["level"] = talk
    if talking:
        tilt += sway * 4 * np.sin(now * 1.6)
        if kind == "?" and progress > 0.6:
            tilt += sway * 9 * min(1.0, (progress - 0.6) / 0.3)
    _face["tilt"] = _ease(_face["tilt"], tilt, 0.15)

    # Blinking (only with open eyes)
    if now > _face["blink_at"]:
        _face["blink_at"] = now + np.random.uniform(2.5, 6.0)
        _face["blink_until"] = now + 0.14
    eyes = look["eyes"]
    if now < _face.get("blink_until", 0) and eyes in ("dot", "big", "half"):
        eyes = "blink"
    if talking and kind == "!" and energy > 0.45 and eyes in ("dot", "big"):
        eyes = "happy"  # a big happy squint on the loud part of an exclamation

    # Body movement: a gentle bob, breathing, and a bounce when talking or excited
    _face["mouth"] = _ease(_face["mouth"], max(look["mouth"], talk), 0.6)
    m = _face["mouth"]
    R = S * 0.33
    nod = R * 0.06 * np.exp(-(now - _face["nod_at"]) / 0.12) if talking else 0.0
    bounce = look["bounce"] * (1.6 if talking and kind == "!" else 1.0)
    bob = np.sin(now * 2 * np.pi * 0.45 * bounce) * R * 0.025 * bounce + nod - energy * R * 0.03
    breathe = 0.012 * np.sin(now * 2 * np.pi * 0.25)
    _face["stretch"] = _ease(_face["stretch"], talk, 0.5)
    stretch = 0.04 * _face["stretch"]  # the head stretches a little when the beak opens wide
    cx, cy = S / 2, S * 0.56 + bob
    if expression == "scared":
        cx += R * 0.012 * np.sin(now * 45)  # a little tremble
    # the tuft lags behind the head's movement, like real feathers
    _face["tuft"] = _ease(_face["tuft"], float(np.clip(-(bob - _face["bob"]) * 2.5, -25, 25)), 0.3)
    _face["bob"] = bob

    def P(x, y):
        return int(round(x)), int(round(y))

    line = max(2, S // 110)

    # Tuft of three feathers
    for i, (dx, dy, angle, a, b) in enumerate(((-0.15, -0.9, -38, 0.085, 0.2), (0.0, -1.0, 0, 0.095, 0.24),
                                                (0.14, -0.89, 38, 0.08, 0.18))):
        angle += _face["tuft"] * (0.8 + 0.3 * i)
        centre = P(cx + dx * R, cy + dy * R * (1 + stretch))
        cv2.ellipse(img, centre, P(a * R, b * R), angle, 0, 360, YELLOW, -1, aa)
        cv2.ellipse(img, centre, P(a * R, b * R), angle, 0, 360, OUTLINE, line, aa)

    # Head: rounder and wider at the bottom, like a duckling
    angles = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    sin, cos = np.sin(angles), np.cos(angles)
    width = R * (1.0 + 0.1 * sin) * (1 + breathe - stretch * 0.5)
    height = np.where(sin > 0, R * 0.8, R * 0.92) * (1 - breathe + stretch)
    head = np.stack([cx + width * cos, cy + height * sin], axis=1).astype(np.int32)
    cv2.fillPoly(img, [head], YELLOW, aa)
    cv2.ellipse(img, P(cx - R * 0.3, cy - R * 0.5), P(R * 0.32, R * 0.17), -20, 0, 360, YELLOW_LIGHT, -1, aa)
    cv2.polylines(img, [head], True, OUTLINE, line, aa)

    # Rosy cheeks with little lines
    eye_y = cy + R * 0.05
    k = min(1.0, 0.8 * look["blush"] * (1 + 0.4 * energy))  # cheeks glow a little more when Anka talks eagerly
    blush = tuple(int(y + (b - y) * k) for y, b in zip(YELLOW, BLUSH))
    for side in (-1, 1):
        bx, by = cx + side * R * 0.64, eye_y + R * 0.27
        cv2.ellipse(img, P(bx, by), P(R * 0.17, R * 0.1), 0, 0, 360, blush, -1, aa)
    if look["blush"] >= 0.6:
        for side in (-1, 1):
            bx, by = cx + side * R * 0.64, eye_y + R * 0.27
            for k in (-1, 0, 1):
                x = bx + k * R * 0.06
                cv2.line(img, P(x - R * 0.02, by + R * 0.035), P(x + R * 0.02, by - R * 0.035), BLUSH_LINES, max(1, line // 2), aa)

    # Eyes
    for side in (-1, 1):
        ex, ey = cx + side * R * 0.4, eye_y
        e = eyes if eyes != "wink" else ("dot" if side < 0 else "happy")  # a wink: one eye open, one ^
        if e == "heart":
            hx, hy, hr = ex + look_x * R * 0.03, ey, R * 0.12 * (1 + 0.08 * np.sin(now * 6))  # a beating heart
            cv2.circle(img, P(hx - hr * 0.55, hy - hr * 0.3), int(hr * 0.62), HEART, -1, aa)
            cv2.circle(img, P(hx + hr * 0.55, hy - hr * 0.3), int(hr * 0.62), HEART, -1, aa)
            cv2.fillPoly(img, [np.array([P(hx - hr * 1.15, hy - hr * 0.1), P(hx + hr * 1.15, hy - hr * 0.1),
                                         P(hx, hy + hr * 1.1)])], HEART, aa)
            cv2.circle(img, P(hx - hr * 0.6, hy - hr * 0.5), max(2, int(hr * 0.2)), (255, 255, 255), -1, aa)
        elif e in ("dot", "big", "half"):
            big = 1.3 if e == "big" else 1.0
            rx, ry = R * 0.1 * big, R * 0.125 * big
            px, py = ex + look_x * R * 0.05, ey + look_y * R * 0.05
            cv2.ellipse(img, P(px, py), P(rx, ry), 0, 0, 360, EYE, -1, aa)
            cv2.circle(img, P(px - rx * 0.3, py - ry * 0.4), max(2, int(rx * 0.38)), (255, 255, 255), -1, aa)
            cv2.circle(img, P(px + rx * 0.35, py + ry * 0.35), max(1, int(rx * 0.16)), (255, 255, 255), -1, aa)
            if e == "big":  # two little lashes on the outside
                for k in (0, 1):
                    x0, y0 = px + side * rx * (0.55 + 0.25 * k), py - ry * (0.85 - 0.3 * k)
                    cv2.line(img, P(x0, y0), P(x0 + side * R * 0.05, y0 - R * 0.04), EYE, line, aa)
            if e == "half":  # a heavy eyelid
                cv2.rectangle(img, P(px - rx - 3, py - ry - 3), P(px + rx + 3, py - ry * 0.1), YELLOW, -1)
                cv2.line(img, P(px - rx * 1.1, py - ry * 0.1), P(px + rx * 1.1, py - ry * 0.1), OUTLINE, line, aa)
        elif e == "happy":  # ^ ^
            cv2.ellipse(img, P(ex, ey + R * 0.04), P(R * 0.1, R * 0.08), 0, 200, 340, EYE, line * 2, aa)
        elif e == "squint":  # > <
            w, h = R * 0.08, R * 0.07
            tip = ex - side * w * 0.2
            cv2.polylines(img, [np.array([P(tip - side * w, ey - h), P(tip + side * w * 0.6, ey), P(tip - side * w, ey + h)])],
                          False, EYE, line * 2, aa)
        elif e in ("sleepy", "blink"):
            depth = 0.06 if e == "sleepy" else 0.02
            cv2.ellipse(img, P(ex, ey - R * 0.01), P(R * 0.09, R * depth), 0, 20, 160, EYE, line * 2, aa)
        # Eyebrows
        brows = look["brows"] or ("up" if talking and energy > 0.4 else None)
        if talking and kind == "?" and progress > 0.6 and brows != "sad":
            brows = "high"  # eyebrows go up at the end of a question
        if brows:
            lift = {"up": 0.06, "high": 0.11, "flat": 0.0, "sad": 0.04, "angry": 0.02}[brows] \
                + (energy * 0.04 if talking else 0)
            by = ey - R * (0.2 + lift)
            inner, outer = ex - side * R * 0.08, ex + side * R * 0.1
            sad = R * 0.05 if brows == "sad" else -R * 0.05 if brows == "angry" else 0.0  # angry: inner ends down
            cv2.line(img, P(outer, by + sad * 0.4), P(inner, by - sad), OUTLINE, line * (2 if brows == "angry" else 1), aa)

    # Beak: a wide, smiling bill. The mouth under it opens with the voice
    W, top, bottom = R * 0.48, R * 0.15, R * 0.07
    by = cy + R * 0.3
    smile = {"sad": -R * 0.06, "annoyed": -R * 0.03, "scared": -R * 0.02}.get(expression, R * 0.05)  # up = smile
    open_h = R * (0.03 + 0.22 * m)
    mouth_c = P(cx, by + bottom * 0.4)
    cv2.ellipse(img, mouth_c, P(W * 0.8, open_h + R * 0.07), 0, 0, 180, BEAK_DARK, -1, aa)          # lower bill
    cv2.ellipse(img, mouth_c, P(W * 0.8, open_h + R * 0.07), 0, 0, 180, OUTLINE, line, aa)
    cv2.ellipse(img, mouth_c, P(W * 0.66, open_h), 0, 0, 180, MOUTH, -1, aa)                       # open mouth
    if m > 0.15:
        cv2.ellipse(img, P(cx, by + bottom * 0.4 + open_h * 0.75), P(W * 0.36, open_h * 0.3), 0, 180, 360, TONGUE, -1, aa)
    angles = np.linspace(0, np.pi, 40)
    upper = [P(cx + W * np.cos(a), by - top * np.sin(a) - smile * np.cos(a) ** 2 * 0.3) for a in angles]
    upper += [P(cx + W * np.cos(a), by + bottom * np.sin(a) - smile * np.cos(a) ** 2) for a in angles[::-1]]
    upper = np.array(upper, np.int32)
    cv2.fillPoly(img, [upper], BEAK, aa)
    cv2.polylines(img, [upper], True, OUTLINE, line, aa)
    for side in (-1, 1):  # nostrils
        cv2.ellipse(img, P(cx + side * R * 0.11, by - top * 0.55), P(R * 0.028, R * 0.013), 0, 0, 360, BEAK_DARK, -1, aa)

    # Little sound waves beside the beak while Anka talks
    if talking and talk > 0.2:
        for side in (-1, 1):
            for k in (1, 2):
                if talk > 0.2 * k:
                    cv2.ellipse(img, P(cx + side * R * (1.06 + 0.09 * k), by + R * 0.02), P(R * 0.06 * k, R * 0.1 * k),
                                0, -50 if side > 0 else 130, 50 if side > 0 else 230, (150, 140, 135), max(2, line // 1), aa)

    # Little signs of what Anka is doing
    if icon == "...":
        for i in range(3):
            on = int(now * 3) % 3 >= i
            cv2.circle(img, P(S * 0.76 + i * S * 0.06, S * 0.12), max(3, S // 55), (90, 90, 90) if on else (205, 195, 180), -1, aa)
    elif icon == "?":
        y = S * 0.2 + np.sin(now * 4) * S * 0.01
        cv2.putText(img, "?", P(S * 0.8, y), cv2.FONT_HERSHEY_SIMPLEX, S / 230, (90, 90, 90), max(2, S // 80), aa)
    elif icon == "z":
        for i in range(2):
            phase = (now / 2.5 + i * 0.5) % 1  # each z floats up and fades
            colour = tuple(int(c + (b - c) * phase) for c, b in zip((110, 100, 90), FACE_BG))
            cv2.putText(img, "z", P(S * (0.72 + phase * 0.12), S * (0.3 - phase * 0.2)), cv2.FONT_HERSHEY_SIMPLEX,
                        S / 420 * (0.7 + phase), colour, max(2, S // 120), aa)
    if expression == "excited":  # sparkles
        for k, (sx, sy) in enumerate(((0.14, 0.22), (0.86, 0.3), (0.2, 0.78))):
            glow_size = (0.5 + 0.5 * np.sin(now * 6 + k * 2)) * S * 0.03
            x, y = S * sx, S * sy
            cv2.line(img, P(x - glow_size, y), P(x + glow_size, y), (60, 200, 255), max(2, S // 150), aa)
            cv2.line(img, P(x, y - glow_size), P(x, y + glow_size), (60, 200, 255), max(2, S // 150), aa)

    # Head tilt and the little "pop" when the face changes
    pop = 1 + 0.06 * max(0.0, 1 - (now - _face["changed_at"]) / 0.3)
    if abs(_face["tilt"]) > 0.3 or pop > 1.001:
        M = cv2.getRotationMatrix2D((S / 2, S * 0.6), _face["tilt"], pop)
        img = cv2.warpAffine(img, M, (S, S), borderValue=FACE_BG)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def show_window(window, camera_picture=None):
    """Shows the camera picture (when there is a new one) with Anka's face next to it, ~20 times a second."""
    if camera_picture is not None:
        _face["camera"] = camera_picture
    now = time.time()
    if _face["camera"] is None or (camera_picture is None and now - _face["drawn"] < 0.05):
        return
    _face["drawn"] = now
    shown = _face["camera"]
    if SHOW_FACE:
        shown = np.hstack([shown, draw_face(shown.shape[0])])
    cv2.imshow(window, shown)


# ---- other parts of Anka (imported last, so the parts can use each other) ----
from anka.feelings import feelings, feelings_lock
from anka.eyes import human_is_here
from anka.voice import speaking
