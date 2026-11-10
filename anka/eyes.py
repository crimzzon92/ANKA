"""Eyes: the camera, the face sensor, recognising and learning faces, and describing what he sees."""
import base64
import os
import re
import time
import urllib.error
import urllib.request

import numpy as np

from settings import (AWAY_AFTER, CAMERA, CAMERA_WINDOW_SIZE, CLOUD_PICTURE_WIDTH, EYE_CHECKS_PER_SECOND, FACE_MATCH, FACE_SURE, FAMILIAR_AFTER, FOLDER, KNOWN_FACES_FILE, LEARN_FACES_MYSELF, LOOK_ALWAYS_WITH_CLOUD, LOOK_EVERY, LOOK_IN_THOUGHTS, MAX_FACE_PICTURES, NAME, NEW_THING_AFTER, PICTURE_WIDTH, RECOGNISE_FACES, SFACE_FILE, SFACE_URL, SHORT_OUTPUT, SHOW_CAMERA, SIM, YUNET_FILE, YUNET_URL)
from anka.shared import S, cv2, state, wake



def heard_voice():
    """With a body: whose voice he just heard and where it came from – (name, direction), or None."""
    v = state.get("voice_from")
    return v[1:] if v and time.time() - v[0] < 30 else None


def speaker_note():
    """If the face sensor sees just ONE person and it isn't the main human, it's probably them talking."""
    known, strangers, main = list(dict.fromkeys(state.get("who", []))), state.get("strangers", 0), human_name()
    voice = heard_voice()
    if voice:
        name, direction = voice
        return (f"(The voice came {direction} – it sounded like {name}"
                + (f", not {main}" if main and name != main else "") + ".)\n")
    if len(known) == 1 and not strangers and main and known[0] != main:
        return f"(Only {known[0]} is in front of you right now – so it's most likely {known[0]} talking to you, not {main}.)\n"
    return ""


def current_speaker():
    known, strangers, main = list(dict.fromkeys(state.get("who", []))), state.get("strangers", 0), human_name()
    voice = heard_voice()
    if voice:
        return voice[0] if main and voice[0] != main else None
    return known[0] if len(known) == 1 and not strangers and main and known[0] != main else None


def brighten(picture):
    """Makes a dark picture brighter (like eyes getting used to the dark). Does nothing to a normal picture."""
    gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    if mean >= 90 or mean < 15:
        return picture  # bright enough – or practically black (covered camera): brightening only makes noise
    lab = cv2.cvtColor(picture, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gain = min(3.0, 110 / max(mean, 10))
    return cv2.convertScaleAbs(out, alpha=gain, beta=0)


def picture_to_base64(picture):
    """Turns a camera picture into a small JPEG (base64) that Ollama can send to the brain."""
    h, w = picture.shape[:2]
    width = min(w, CLOUD_PICTURE_WIDTH) if cloud_is_on() else PICTURE_WIDTH
    small = brighten(cv2.resize(picture, (width, int(h * width / w))))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80 if cloud_is_on() else 70])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


def place(x_center, size):
    """Where something is: e.g. 'left, close'."""
    side = "left" if x_center < 0.33 else "right" if x_center > 0.66 else "straight ahead"
    return f"{side}, {'close' if size > 0.15 else 'far away'}"


def scene_text(things):
    """'2 persons (left, close; right, close), 1 cup (right, far away)' – with counts.
    (Used in the simulation, where there is no real picture.)"""
    if not things:
        return "[RIGHT NOW your camera sees nothing in particular.]"
    groups = {}
    for name, where in things:
        groups.setdefault(name, []).append(where)
    parts = []
    for name, places in groups.items():
        count = len(places)
        word = name if count == 1 else (name + "s" if not name.endswith("s") else name)
        parts.append(f"{count} {word} ({'; '.join(places)})")
    text = "[RIGHT NOW your camera sees: " + ", ".join(parts) + "."
    if "person" in groups and who_text():
        text += f" Your face sensor{who_text()}."
    elif "person" in groups:
        text += (" One person is most likely the human you talk with." if len(groups["person"]) > 1
                 else " The person is most likely the human.")
        text += " (You can see THAT people are there – not what they are doing or holding.)"
    return text + "]"


def brief_scene():
    """What Anka sees, without positions (for thoughts and normal talk – the details only distract)."""
    if body is not None:
        return body_text()  # with a body, where things are matters: he can walk there
    seen_now = state.get("sees_short", "nothing")
    name = human_name()
    known, strangers = list(dict.fromkeys(state.get("who", []))), state.get("strangers", 0)
    if known or strangers:  # the face sensor says who it is
        rest = re.sub(r"\d+ (person|people)(, )?", "", seen_now).strip(", ")
        people = known + ([f"{strangers} {'person' if strangers == 1 else 'people'} you don't recognise"]
                          if strangers else [])
        seen_now = ", ".join(x for x in [" and ".join(people), rest] if x)
    elif name and state.get("faces"):
        others = state["faces"] - 1
        rest = re.sub(r"\d+ (person|people)(, )?", "", seen_now).strip(", ")
        seen_now = ", ".join(x for x in [name + (f" and {others} other {'person' if others == 1 else 'people'}"
                                                  if others else ""), rest] if x)
    return f"[Right now you see: {seen_now}.]"


def who_text():
    """What the face recogniser says about who is in view, e.g. 'it recognises Robin, and 1 face it doesn't know'.
    Empty if it knows nobody yet (then nothing is claimed)."""
    if not (S.faces_known and S.faces_known.names):
        return ""
    known, strangers = list(dict.fromkeys(state.get("who", []))), state.get("strangers", 0)
    parts = ([f"it recognises {' and '.join(known)}"] if known else []) \
        + ([f"{strangers} face{'s' if strangers != 1 else ''} it doesn't recognise"] if strangers else [])
    return (" – " + ", and ".join(parts)) if parts else ""


def picture_scene(faces):
    """The scene text when the brain gets the camera picture."""
    if state.get("dark"):
        return ("[The attached picture is what your own eyes see right now: it is almost completely black. "
                "Something may be covering your camera, or the light is off. Your face sensor finds nothing.]")
    return ("[The attached picture is what your own eyes see right now (it is not a photo). "
            + (f"Your face sensor found {faces} face{'s' if faces != 1 else ''}"
               + (who_text() or (f" – one is probably {human_name()}" if human_name() else "")) + ".]" if faces
               else "Your face sensor found no face right now.]"))


def asks_about_sight(text):
    """True if you ask what Anka sees right now."""
    # "what's that?" only counts when it is the whole question – not "what's that, a dream you had?"
    return bool(re.search(r"\b(see|seeing|look|looking|camera|how many|who is (here|there)|"
                          r"in front of you|describe)\b", text, re.I)
                or re.search(r"\bwhat('s| is) (this|that|here)\s*[?.!]*\s*$", text.strip(), re.I))


face_rows = []   # YuNet's latest finds, with eyes/nose/mouth points (needed to recognise who it is)


class FaceMemory:
    """Remembers what known people's faces look like (SFace, from OpenCV) and recognises them."""
    def __init__(self):
        self.model, self.names, self.vectors = None, [], np.zeros((0, 128), "float32")
        self.load()

    def load(self):
        try:
            with np.load(KNOWN_FACES_FILE) as data:
                self.names, self.vectors = [str(n) for n in data["names"]], data["vectors"].astype("float32")
        except FileNotFoundError:
            pass
        except Exception as error:
            print(f"👤 (couldn't read {os.path.basename(KNOWN_FACES_FILE)}: {error})")

    def save(self):
        save_json(KNOWN_FACES_FILE, lambda f: np.savez(f, names=np.array(self.names, dtype=str), vectors=self.vectors))

    def ready(self):
        """Loads the recognition model (downloads it once). False if it can't be used."""
        if self.model is None:
            path = os.path.join(FOLDER, SFACE_FILE)
            if not os.path.exists(path):
                try:
                    print("👤 Downloading the face recogniser (37 MB, only once)...")
                    urllib.request.urlretrieve(SFACE_URL, path)
                except Exception as error:
                    print(f"👤 (couldn't download it: {error})")
                    self.model = False
                    return False
            try:
                self.model = cv2.FaceRecognizerSF.create(path, "")
            except Exception as error:
                print(f"👤 (the face recogniser didn't work: {error})")
                self.model = False
        return bool(self.model)

    def vector(self, picture, row):
        """A face -> 128 numbers that describe it (row = YuNet's find, in this picture's size)."""
        face = self.model.alignCrop(picture, np.asarray(row, dtype="float32"))
        v = self.model.feature(face).flatten().astype("float32")
        return v / (np.linalg.norm(v) + 1e-9)

    def who(self, picture, rows):
        """Names for each face: the best match, or None for someone he doesn't know."""
        return [name for name, _, _ in self.who_full(picture, rows)]

    def who_full(self, picture, rows):
        """For each face: (name or None, how sure 0–1, the face's numbers)."""
        if not self.ready():
            return [(None, 0.0, None)] * len(rows)
        out = []
        for row in rows:
            try:
                v = self.vector(picture, row)
            except Exception:
                out.append((None, 0.0, None))
                continue
            if not self.names:
                out.append((None, 0.0, v))
                continue
            score = self.vectors @ v
            best = int(np.argmax(score))
            out.append((self.names[best] if score[best] >= FACE_MATCH else None, float(score[best]), v))
        return out

    def add(self, name, vectors):
        """Remembers more looks of a person – and lets go of the most alike ones if there are too many."""
        self.names += [name] * len(vectors)
        self.vectors = np.vstack([self.vectors, np.array(vectors, dtype="float32")])
        mine = [i for i, n in enumerate(self.names) if n == name]
        while len(mine) > MAX_FACE_PICTURES:
            V = self.vectors[mine]
            alike = (V @ V.T).mean(axis=1)  # the look most like all the others adds the least
            drop = mine[int(np.argmax(alike))]
            del self.names[drop]
            self.vectors = np.delete(self.vectors, drop, axis=0)
            mine = [i for i, n in enumerate(self.names) if n == name]
        self.save()

    def learn_from(self, results, now):
        """Called about once a second with what the face sensor sees: finishes learning a new face
        (when the brain asked for it), and now and then keeps a new look of someone he knows well."""
        task = state.get("learning")
        if task:
            # only while there is exactly one face that isn't someone else he knows
            mine = [r for r in results if r[0] in (None, task["name"])]
            if len(mine) == 1 and mine[0][2] is not None:
                task["vecs"].append(mine[0][2])
            if len(task["vecs"]) >= 8 or now > task["until"]:
                state.pop("learning", None)
                name = task["name"]
                if len(task["vecs"]) >= 3:
                    self.add(name, task["vecs"])
                    print(f"👤 {NAME} now knows what {name} looks like")
                    with memory_lock:
                        write_diary(f"I learned to recognise {name}'s face.")
                        save_memory()
                    state["events"].append(f"You just learned to recognise {name}'s face.")
                else:
                    print(f"👤 (couldn't get a good look at {name} – try again face to face, in good light)")
        if not LEARN_FACES_MYSELF:
            return
        self.last_added = getattr(self, "last_added", {})
        for name, score, v in results:
            # sure who it is, but this look is a bit new (other light, glasses...) – keep it, at most every 10 minutes
            if name and v is not None and FACE_SURE <= score < 0.85 and now - self.last_added.get(name, 0) > 600:
                self.last_added[name] = now
                self.add(name, [v])
                if not SHORT_OUTPUT:
                    print(f"👤 (learned a new look of {name})")




def full_size_rows(rows, scale):
    """YuNet's finds on the small picture -> the same places on the full camera picture."""
    out = []
    for r in rows:
        r = np.array(r, dtype="float32")
        r[:14] *= scale  # box and the five points (the last value is the score)
        out.append(r)
    return out


def learn_face(name):
    """python run_anka.py --learn-face Robin – looks at one person for a few seconds and remembers their face."""
    memory = FaceMemory()
    if not memory.ready():
        return
    find = make_face_sensor()
    camera = cv2.VideoCapture(find_camera())
    print(f"👤 Look at the camera, {name} – only you in the picture. Turn your head a little, smile, look serious...")
    taken, last, started = [], 0.0, time.time()
    while len(taken) < 15 and time.time() - started < 60:
        ok, picture = camera.read()
        if not ok:
            continue
        h, w = picture.shape[:2]
        scale = w / 320
        small = brighten(cv2.resize(picture, (320, int(h / scale))))
        found = find(small)
        shown = picture.copy()
        if len(found) == 1 and face_rows and face_rows[0][-1] > 0.9 and time.time() - last > 0.3:
            last = time.time()
            taken.append(memory.vector(brighten(picture), full_size_rows(face_rows, scale)[0]))
            print(f"   📸 {len(taken)}/15")
        for (x, y, fw, fh) in found:
            cv2.rectangle(shown, (int(x * scale), int(y * scale)), (int((x + fw) * scale), int((y + fh) * scale)),
                          (0, 200, 0) if len(found) == 1 else (0, 0, 255), 2)
        cv2.putText(shown, f"{name}: {len(taken)}/15" + ("" if len(found) == 1 else "  (one face only!)"),
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
        cv2.imshow("Learning a face", shown)
        cv2.waitKey(1)
    camera.release()
    cv2.destroyAllWindows()
    if len(taken) < 5:
        print("👤 Didn't get enough good pictures – try again with more light and only one face in view.")
        return
    memory.names += [name] * len(taken)
    memory.vectors = np.vstack([memory.vectors, np.array(taken)])
    memory.save()
    print(f"✅ {NAME} now knows what {name} looks like ({len(taken)} pictures). "
          f"Run it again another day, in other light, and he gets even better at it.")


def make_face_sensor():
    """Returns a function: small picture -> list of faces (x, y, w, h).
    Tries in order: YuNet (best, downloaded once, 0.2 MB), Haar cascade, HOG (whole bodies)."""
    path = os.path.join(FOLDER, YUNET_FILE)
    if not os.path.exists(path):
        try:
            print("👀 Downloading a tiny face detector (0.2 MB, only once)...")
            urllib.request.urlretrieve(YUNET_URL, path)
        except Exception as error:
            print(f"👀 (couldn't download it: {error})")
    try:
        if os.path.exists(path) and os.path.getsize(path) > 100_000:
            yunet = cv2.FaceDetectorYN.create(path, "", (320, 320), 0.7)

            def find(picture):
                h, w = picture.shape[:2]
                yunet.setInputSize((w, h))
                _, found = yunet.detect(picture)
                face_rows[:] = [] if found is None else list(found)  # (eyes, nose and mouth too – for recognising)
                return [] if found is None else [tuple(int(v) for v in f[:4]) for f in found]
            print("👀 Face sensor: YuNet")
            return find
    except Exception as error:
        print(f"👀 (YuNet didn't work: {error})")
    cascade = os.path.join(getattr(getattr(cv2, "data", None), "haarcascades", ""),
                           "haarcascade_frontalface_default.xml")
    if os.path.exists(cascade):
        haar = cv2.CascadeClassifier(cascade)
        print("👀 Face sensor: Haar cascade")
        return lambda picture: list(haar.detectMultiScale(cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY),
                                                          scaleFactor=1.2, minNeighbors=5, minSize=(24, 24)))
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    print("👀 Face sensor: HOG (finds whole bodies, not faces)")

    def find_bodies(picture):
        boxes, _ = hog.detectMultiScale(picture, winStride=(8, 8))
        # a body -> roughly where the head is, so the rest of the code works the same
        return [(int(x + w / 3), int(y), int(w / 3), int(h / 6)) for (x, y, w, h) in boxes]
    return find_bodies


def camera_frames(camera):
    """Reads the camera, finds faces and movement with OpenCV (almost no memory), and keeps the
    latest picture so the brain can look at it. Yields a list of (name, place) for each picture."""
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    pass  # (shared values live on S)
    find_faces = make_face_sensor()
    S.faces_known = FaceMemory() if RECOGNISE_FACES else None
    if S.faces_known and S.faces_known.names:
        print(f"👤 Knows the faces of: {', '.join(dict.fromkeys(S.faces_known.names))}")
    elif RECOGNISE_FACES:
        print("👤 Knows no faces yet – teach one with:  python run_anka.py --learn-face Alex")
    previous_gray, last, missed, last_read, last_motion = None, 0.0, 0, time.time(), 0.0
    last_recognised, who = 0.0, []
    window = f"{NAME} sees"
    try:
        while True:
            if time.time() - last_read > 0.5:
                for _ in range(5):
                    cap.grab()  # throw away old pictures after a pause
            ok, picture = cap.read()
            last_read = time.time()
            if not ok:
                missed += 1
                if missed > 200:
                    print("📷 The camera stopped sending pictures.")
                    return
                time.sleep(0.05)
                continue
            missed = 0
            now = time.time()
            if now - last < 1.0 / EYE_CHECKS_PER_SECOND:
                if SHOW_CAMERA:
                    show_window(window)  # the face moves smoothly between the camera checks
                    camera_key()
                continue
            last = now
            h, w = picture.shape[:2]
            scale = w / 320
            small = brighten(cv2.resize(picture, (320, int(h / scale))))  # faces are found in a dim room too
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            sh, sw = small.shape[:2]
            things, boxes = [], []
            gaze, biggest = None, 0
            found = find_faces(small)
            if S.faces_known and (S.faces_known.names or LEARN_FACES_MYSELF) and found \
                    and len(face_rows) == len(found) \
                    and (len(found) != len(who) or now - last_recognised > (0.4 if state.get("learning") else 1.0)):
                last_recognised = now
                results = S.faces_known.who_full(brighten(picture), full_size_rows(face_rows, scale))
                who = [r[0] for r in results]
                S.faces_known.learn_from(results, now)
            elif not found:
                who = []
                if S.faces_known and state.get("learning"):
                    S.faces_known.learn_from([], now)  # (the person walked away – the learning times out)
            state["who"] = [n for n in who if n] if len(who) == len(found) else []
            state["strangers"] = sum(1 for n in who if not n) if len(who) == len(found) else 0
            for (x, y, fw, fh) in found:
                if fw * fh > biggest:
                    biggest, gaze = fw * fh, (x + fw / 2) / sw  # where the nearest face is, for Anka's eyes
                # a face -> a rough "person" box (head + body) for distance and position
                x1, y1 = max(0, x - fw), y
                x2, y2 = min(sw, x + 2 * fw), min(sh, y + 5 * fh)
                things.append(("person", place((x1 + x2) / 2 / sw, (x2 - x1) * (y2 - y1) / (sw * sh))))
                boxes.append((int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)))
            # Movement: how much the picture changed since last time
            blurred = cv2.GaussianBlur(gray, (9, 9), 0)
            motion = float(np.mean(cv2.absdiff(blurred, previous_gray))) / 255 if previous_gray is not None else 0.0
            previous_gray = blurred
            state["picture"] = picture
            state["gaze"] = gaze
            state["dark"] = float(gray.mean()) < 15  # (gray is already brightened if it was only dim)
            if motion > 0.04 and not things and now - last_motion > 30:
                last_motion = now
                state["events"].append("Something moved in front of your camera.")
                wake.set()
            if SHOW_CAMERA:
                shown = picture.copy()
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    if i < len(who) and len(who) == len(boxes):
                        cv2.putText(shown, who[i] or "?", (x1 + 4, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (0, 200, 0), 2)
                if CAMERA_WINDOW_SIZE != 1:
                    shown = cv2.resize(shown, None, fx=CAMERA_WINDOW_SIZE, fy=CAMERA_WINDOW_SIZE,
                                       interpolation=cv2.INTER_AREA)
                show_window(window, shown)
                camera_key()
            yield things
    finally:
        cap.release()


def find_camera():
    """Tries CAMERA first, then 0, 1, 2 – takes the first one that works."""
    for nr in [CAMERA] + [n for n in (0, 1, 2) if n != CAMERA]:
        camera = cv2.VideoCapture(nr)
        ok = camera.isOpened() and camera.read()[0]
        camera.release()
        if ok:
            print(f"📷 Using camera {nr}")
            return nr
    raise SystemExit("No camera found. Check camera permission for Terminal.")


def human_is_here():
    """Are you here? Yes if the eyes saw you in the last few seconds."""
    return bool(state["person_last"]) and time.time() - state["person_last"] < 8


def scene_without_picture():
    faces = state["faces"]
    return ("[You are not looking closely right now"
            + (f" (you last did {ago(time.time() - state['looked_at'])})" if state.get("looked_at") else "")
            + ". " + (f"Your face sensor finds {faces} face{'s' if faces != 1 else ''}{who_text()}.]" if faces
                      else "Your face sensor finds no face.]"))


def _look():
    state["looked_at"], state["faces_looked"] = time.time(), state["faces"]
    return state["picture"]


def picture_for_reply(human_said):
    """Should Anka look closely at the picture when it answers you? Looking takes several seconds.
    Yes if you ask about what it sees, if something changed (someone came or left), or if it's been a while."""
    if body is not None and getattr(body, "real", False):
        return body.camera_picture()   # the virtual body's camera: he always looks (there's no list to go by)
    if SIM or state.get("picture") is None or not S.can_see_pictures:
        return None
    if LOOK_ALWAYS_WITH_CLOUD and cloud_is_on():
        return _look()
    if (asks_about_sight(human_said) or state["faces"] != state.get("faces_looked")
            or time.time() - state.get("looked_at", 0) > LOOK_EVERY):
        return _look()
    return None


def picture_for_thought(something_happened):
    """Should Anka look at the camera picture in this thought? Looking takes several seconds on an M1."""
    if body is not None and getattr(body, "real", False):
        return body.camera_picture()
    if SIM or state.get("picture") is None or not S.can_see_pictures:
        return None
    if LOOK_ALWAYS_WITH_CLOUD and cloud_is_on():
        return _look()
    if not LOOK_IN_THOUGHTS:
        return None
    if LOOK_IN_THOUGHTS == "sometimes" and not something_happened \
            and time.time() - state.get("looked_at", 0) < LOOK_EVERY:
        return None
    return _look()


def update_sight(things, now):
    """Takes in what the eyes see right now. Returns True if Anka should think at once."""
    think_now = False
    names = tuple(sorted({name for name, _ in things}))
    new_kinds = [n for n in names if n not in seen]
    if new_kinds:
        with memory_lock:
            for n in new_kinds:
                seen.append(n)
                write_diary(f"I saw a {n} for the first time.")
                state["events"].append(f"You saw a {n} for the first time ever.")
                think_now = True
            save_memory()
    faces = sum(1 for name, _ in things if name == "person")
    counts = {}
    for name, _ in things:
        counts[name] = counts.get(name, 0) + 1
    state["sees_short"] = ", ".join(f"{n} {name if n == 1 else 'people' if name == 'person' else name + 's'}"
                                    for name, n in counts.items()) or "nothing"
    state["scene"] = make_scene(things, faces)
    state["faces"] = faces

    # Are you here?
    person_now = faces > 0 or bool(state.get("people_nearby"))  # (with a body: people behind him are still here)
    if not person_now and state["person_last"] and not state.get("missing_since"):
        state["missing_since"] = now  # the first picture where you can't be seen
    if not person_now and state.get("missing_since") and not state.get("left_announced") \
            and now - state["missing_since"] > AWAY_AFTER:
        state["left_announced"] = True
        heard_go = state.get("heard_leave_at", 0) >= state.get("missing_since", now) - 120
        state["events"].append((f"The human seems to be gone – you heard someone go out, and you haven't seen or heard them "
                                f"for {int(now - state['missing_since'])} seconds.") if heard_go else
                               f"You haven't seen or heard the human for {int(now - state['missing_since'])} seconds.")
        print(f"\n🚪 ({NAME} noticed you left)")
        state["rest"] = 0
        wake.set()
    if person_now:
        state["left_announced"] = False
        missing = state.pop("missing_since", None)
        away = (now - missing) if missing else 0
        known_now = list(dict.fromkeys(state.get("who", [])))
        names_here = " and ".join(known_now)
        if state["person_last"] == 0 or away > AWAY_AFTER:
            if state["person_last"] == 0:
                state["events"].append(f"You just woke up and {names_here} {'are' if len(known_now) > 1 else 'is'} here."
                                       if known_now else "You just woke up and the human is here.")
            else:
                minutes = round(away / 60)
                gone = "less than a minute" if away < 45 else "about a minute" if minutes <= 1 else \
                    f"about {minutes} minutes" if minutes < 60 else f"about {round(away / 3600)} hours"
                heard_go = state.get("heard_leave_at", 0) >= (missing or now) - 120
                state["events"].append(f"{names_here or 'The human'} just came back after being away for {gone}." if heard_go
                                       or not getattr(body, "real", False) else
                                       f"{names_here or 'The human'} is in view again – you hadn't seen them for {gone} "
                                       "(they may have been here all along, out of view).")
                state["came_back"] = (now, gone)  # also mentioned when you talk right after
                print(f"\n👋 ({NAME} sees {names_here + ' is' if names_here else 'you are'} back after {gone})")
            think_now = True
        else:
            # Someone he knows joins (e.g. Robin comes in while Alex is already there)
            for who in known_now:
                if now - state.setdefault("who_seen_at", {}).get(who, 0) > AWAY_AFTER:
                    state["events"].append(f"{who} just came {'into the room' if body is not None and not getattr(body, 'real', False) else 'into view'}."
                                           + (" (You can see them again – that doesn't mean they just arrived.)"
                                              if getattr(body, "real", False) else ""))
                    print(f"\n👤 ({NAME} {'hears' if body is not None and not getattr(body, 'real', False) else 'sees'} {who})")
                    think_now = True
        for who in known_now:
            state.setdefault("who_seen_at", {})[who] = now
        state["person_last"] = now
    for n in names:
        if now - state["seen_at"].get(n, 0) > NEW_THING_AFTER:
            state["seen_since"][n] = now  # a new "round" – it has been gone for a while
            if n not in state["appeared"]:
                state["appeared"].append(n)
        state["seen_at"][n] = now
    state["familiar"] = [n for n in names if n != "person"
                         and now - state["seen_since"].get(n, now) > FAMILIAR_AFTER]
    return think_now


def make_scene(things, faces):
    """The scene text: the real picture is described by the brain itself, the simulation gets a list."""
    if body is not None:
        return body_text()
    return scene_text(things) if SIM or not S.can_see_pictures else picture_scene(faces)


# ---- other parts of Anka (imported last, so the parts can use each other) ----
from anka.memory import (ago, human_name, memory_lock, save_json, save_memory, seen, write_diary)
from anka.brain import cloud_is_on
from anka.face import show_window
from anka.ears import camera_key
from anka.body import body, body_text
