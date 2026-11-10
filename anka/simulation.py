"""The test days: a made-up past, the fake camera and the script that plays 'you'."""
import os
import re
import sys
import time
from datetime import date

from settings import SIM_SOUND, EYE_CHECKS_PER_SECOND, NAME, SHOW_ROOM, SIM_SCRIPT
from anka.shared import S, heard, state, thought_times, thoughts, wake



def sim_past():
    """A made-up week of history for the simulation, so it tests a duck with a life – like the real Anka.
    (python run_anka.py --simulate --newborn = start as a newly hatched duck with an empty memory)"""
    def day(n, hour=18):
        d = date.today().fromordinal(date.today().toordinal() - n)
        return d.isoformat(), f"{d.isoformat()}T{hour:02d}:15:00"
    diary_lines = [(6, "I woke up for the first time and saw Alex."), (6, "I saw a person for the first time."),
                   (6, "I saw a cup for the first time."), (5, "Alex told me about his partner Robin."),
                   (4, "Alex came home late and was tired, so we only talked a little."),
                   (3, "Robin laughed when I said her name wrong."), (3, "I saw a cat for the first time."),
                   (2, "Alex played me a song from his phone and I liked the drums."),
                   (1, "Alex said he is learning to play the guitar."),
                   (1, "I dreamt I bounced on a giant drum while Sixten tried to catch me.")]
    talks = [(5, "My partner is called Robin. She lives here too.", "Hi Robin! I hope I get to meet her soon."),
             (4, "I'm so tired today, long day at work.", "Then rest, Alex. I'll keep quiet for a bit."),
             (3, "That was Robin's cat, Sixten. He's not really ours.", "Sixten looked at me like I was a snack."),
             (2, "Do you like music?", "Yes! The drums made me want to bounce."),
             (1, "I'm learning guitar. I'm really bad at it.", "Everyone starts bad. Play me something someday!")]
    facts = ["The human is named Alex.", "The human's partner is called Robin.", "The human works long days.",
             "The human likes music.", "The human is learning to play the guitar.",
             "Robin has a cat called Sixten."]
    return {
        "self": ["I love it when Alex comes home.", "Music makes me want to bounce."],
        "facts": {"unsorted": facts},
        "diary": [{"date": day(n)[0], "time": day(n)[1], "text": t} for n, t in diary_lines],
        "seen": ["person", "cup", "cat"],
        "conversations": [{"date": day(n)[0], "time": day(n, 19)[1], "human": h, "duck": d} for n, h, d in talks],
        "last_dream": day(1, 23)[1],
        # where things were on earlier days (in words – the body test uses it; the guitar is still there, the sock not)
        "places": {"guitar": {"where": "by the bookshelf", "near": "bookshelf", "date": day(1)[0]},
                   "sock": {"where": "near the sofa", "near": "sofa", "date": day(2)[0]}},
    }


def sim_frames():
    """Follows the script: gives fake camera pictures and 'says' your lines at the right time."""
    start = time.time()
    sees, step = [], 0
    print(f"🎬 SIMULATION: {len(SIM_SCRIPT)} events over about {SIM_SCRIPT[-1][0] // 60 + 1} minutes")
    while True:
        t = time.time() - start
        while step < len(SIM_SCRIPT) and SIM_SCRIPT[step][0] <= t:
            _, event = SIM_SCRIPT[step]
            step += 1
            kind, _, content = event.partition(":")
            if kind.strip() == "room" and body is not None:   # where people are: "Alex=sofa, Robin=gone"
                for part in content.split(","):
                    who, _, spot = part.partition("=")
                    who, gone_now = who.strip(), spot.strip() in ("", "gone")
                    was_here = who in body.people
                    body.place_person(who, None if gone_now else spot.strip())
                    # 2026-11-09: he HEARS people come and go (footsteps, the door) – so "out of view" is no longer
                    # taken for "gone". (A real duck has a microphone too.)
                    if t > 1 and was_here and gone_now:
                        state["heard_leave_at"] = time.time()
                        state["events"].append("You heard footsteps going away and the door – someone went out of the room.")
                        wake.set()
                    elif t > 1 and not was_here and not gone_now:
                        state["heard_enter_at"] = time.time()
                        state["events"].append("You heard the door and footsteps – someone came into the room.")
                        wake.set()
                print(f"\n🎬 [{int(t)}s] in the room: " + (", ".join(f"{p} by the {spot_name(p)}" for p in body.people)
                                                          or "nobody"))
            elif kind.strip() == "see":
                sees = [x.strip() for x in content.split(",") if x.strip()]
                print(f"\n🎬 [{int(t)}s] camera now sees: {', '.join(sees) or 'nothing'}")
            elif kind.strip().startswith("say"):
                tone = re.search(r"\((.+)\)", kind)
                speaker = re.search(r"\[(\w+)\]", kind)
                if speaker and body is not None:   # he hears where the voice comes from
                    state["voice_from"] = (time.time(), speaker.group(1), body.voice_from(speaker.group(1)))
                    state["person_said"] = (time.time(), speaker.group(1), content.strip())
                    if hasattr(body, "hear"):
                        body.hear(speaker.group(1))   # (his map gets a rough idea of where the voice came from)
                print(f"\n🎬 [{int(t)}s] you say{f' ({tone.group(1)})' if tone else ''}: {content.strip()}")
                heard.put((content.strip(), tone.group(1)) if tone else content.strip())
            elif kind.strip() == "put" and body is not None:    # a thing is put UP on furniture: "sock=sofa"
                from anka.body import put_thing_on
                name, _, furn = content.partition("=")
                if put_thing_on(body, name.strip(), furn.strip()):
                    print(f"\n🎬 [{int(t)}s] someone put the {name.strip()} up on the {furn.strip()}")
            elif kind.strip() == "move" and body is not None:   # a thing is moved (by someone else): "ball=2.0,3.3"
                name, _, xy = content.partition("=")
                name = name.strip()
                if name in getattr(body, "things", {}) and getattr(body, "holding", None) != name:
                    x_, y_ = (float(v) for v in xy.split(","))
                    body.things[name]["pos"] = (x_, y_)
                    print(f"\n🎬 [{int(t)}s] someone moved the {name} (to {x_:.1f}, {y_:.1f})")
            elif kind.strip() == "event":
                print(f"\n🎬 [{int(t)}s] {content.strip()}")
                state["events"].append(content.strip())
                state["rest"] = 0
                wake.set()
        if body is not None:
            show_room(t, 1.0 / EYE_CHECKS_PER_SECOND)   # (the window is redrawn many times while the eyes wait)
            yield body_frame(t)
            continue
        time.sleep(1.0 / EYE_CHECKS_PER_SECOND)
        # person(Robin) = a face the sensor recognises – if he knows Robin's face (otherwise she's a stranger to him)
        tagged = [re.match(r"person\((.*)\)$", x) for x in sees]
        if any(tagged):
            known = S.faces_known.names if S.faces_known else []
            who = [m.group(1) for m in tagged if m and m.group(1) in known]
            state["who"], state["strangers"] = who, sum(1 for m in tagged if m) - len(who)
        else:
            state["who"], state["strangers"] = [], 0
        things = ["person" if m else x for x, m in zip(sees, tagged)]
        # every thing side by side, from left to right, close to the camera
        yield [(name, place((225 + 300 * i) / 1280, 0.21)) for i, name in enumerate(things)]


def spot_name(person):
    from anka.body import SPOTS
    x, y = body.people[person]
    return min(SPOTS, key=lambda k: (SPOTS[k][0] - x) ** 2 + (SPOTS[k][1] - y) ** 2)


_room_window = [SHOW_ROOM and (sys.platform == "darwin" or bool(os.environ.get("DISPLAY")))]  # (only where there is a screen)


def body_frame(t):
    """With the virtual body: what his camera sees comes from where he stands and looks. Also draws the room."""
    if getattr(body, "real", False):
        return real_frame()
    known = S.faces_known.names if S.faces_known else []
    in_room = list(body.people)
    state["who"] = [p for p in in_room if p in known]          # (he hears and knows who is in the room)
    state["strangers"] = len(in_room) - len(state["who"])
    state["people_nearby"] = bool(in_room)
    things = []
    for name, kind, d, rel in body.visible():
        if kind in ("person", "thing"):
            x = min(0.95, max(0.05, 0.5 - rel / (2 * SEE_ANGLE)))  # left in the picture = left of his beak
            things.append(("person" if kind == "person" else name, place(x, min(0.5, 0.25 / max(d, 0.2)))))
    return things


def real_frame():
    """With real senses: the face sensor recognises only faces it can SEE in the camera picture (he must look up to
    see a face up close), the object list is what the camera shows, and he knows people are around only if he has
    seen or heard them lately."""
    known = S.faces_known.names if S.faces_known else []
    body.remember_view()
    seen = body.seen_names(15)
    faces = [k for k, v in seen.items() if v["kind"] == "person" and v["face"] >= 25]
    state["who"] = [p for p in faces if p in known]
    state["strangers"] = len(faces) - len(state["who"])
    now = time.time()
    if any(v["kind"] == "person" for v in seen.values()):
        state["person_seen_at"] = now
    voice = state.get("voice_from")
    # Someone is still around if he saw or heard them lately – or if he saw/heard someone in the last 10 minutes and has
    # NOT heard anyone go out since (out of view is not gone: they may be behind him or beside him). 2026-11-09
    last_sign = max(state.get("person_seen_at", 0), voice[0] if voice else 0, state.get("heard_enter_at", 0))
    state["people_nearby"] = (now - state.get("person_seen_at", 0) < 30
                              or bool(voice and now - voice[0] < 60 and voice[1] in body.people)
                              or (bool(last_sign) and now - last_sign < 600 and state.get("heard_leave_at", 0) < last_sign))
    return [("person" if v["kind"] == "person" else k, "in view") for k, v in seen.items()]


def show_room(t, wait):
    """Draws the room about 30 times a second for 'wait' seconds (or just waits, if there's no screen)."""
    end = time.time() + wait
    while _room_window[0] and time.time() < end:
        now = time.time()
        mins, secs = divmod(int(t), 60)
        lines = [f"{mins}:{secs:02d}   {state.get('caption', '')}"]
        if thoughts and thought_times and now - thought_times[-1] < 15:
            lines.append(f"thinks: {thoughts[-1]}")
        bubbles = []
        if state["said"] and now - state["said"][-1][0] < 7:
            bubbles.append((NAME, state["said"][-1][1]))
        ps = state.get("person_said")
        if ps and now - ps[0] < 6:
            bubbles.append((ps[1], ps[2]))
        try:
            img = body.draw(caption=lines, bubbles=bubbles)
            img = _who_moves(img)
            if SIM_SOUND:
                img = _sound_button(img)
            cv2.imshow(f"{NAME}'s room", img)
            if SIM_SOUND and not _button["hooked"]:
                cv2.setMouseCallback(f"{NAME}'s room", _click)
                _button["hooked"] = True
            key = cv2.waitKey(1) & 0xFF
            if SIM_SOUND and key in (ord("m"), ord("M")):
                set_sound(not S.sound_on)
        except Exception as error:   # no screen (e.g. a test on a server) – the test goes on without the window
            print(f"🗺️  (can't show the room window: {error})")
            _room_window[0] = False
        time.sleep(max(0.0, 1 / 30 - (time.time() - now)))
    rest = end - time.time()
    if rest > 0:
        time.sleep(rest)


_button = {"hooked": False, "rect": (0, 0, 0, 0)}


def _who_moves(img):
    """Bottom left of the room window: what drives his body right now - one of Pollen's animations (orange) or the
    living body following his own BODY words (blue). 2026-11-10"""
    try:
        text, colour = None, None
        sh = getattr(body, "showing", None)
        if sh and time.time() < sh[1]:
            text, colour = sh[0], (0, 120, 240)
        else:
            lv = getattr(body, "living", None)
            if lv is not None and lv.style_text and time.time() - lv.style_t < 90:
                text, colour = "HIS BODY (living body code): " + lv.style_text[:60], (200, 110, 30)
        if text:
            img = img.copy()
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y2 = img.shape[0] - 8
            cv2.rectangle(img, (8, y2 - th - 14), (min(img.shape[1] - 8, 22 + tw), y2), colour, -1)
            cv2.putText(img, text, (15, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    except Exception:
        pass
    return img


def _sound_button(img):
    """The mute button, top right of the room window: click it (or press M) to turn his voice and duck sounds off/on."""
    img = img.copy()
    on = S.sound_on
    text = "SOUND ON  (click / M)" if on else "MUTED  (click / M)"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    x1, y1 = img.shape[1] - tw - 30, 8
    x2, y2 = img.shape[1] - 8, y1 + th + 16
    cv2.rectangle(img, (x1, y1), (x2, y2), (60, 160, 60) if on else (60, 60, 200), -1)
    cv2.putText(img, text, (x1 + 11, y2 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    _button["rect"] = (x1, y1, x2, y2)
    return img


def _click(event, x, y, flags, param):
    x1, y1, x2, y2 = _button["rect"]
    if event == cv2.EVENT_LBUTTONDOWN and x1 <= x <= x2 and y1 <= y <= y2:
        set_sound(not S.sound_on)


class Tee:
    """Writes to the screen and a file at the same time (for the simulation log)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, x):
        for s in self.streams:
            s.write(x)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# ---- other parts of Anka (imported last, so the parts can use each other) ----
from anka.eyes import place
from anka.shared import cv2
from anka.body import SEE_ANGLE, body
from anka.voice import set_sound
