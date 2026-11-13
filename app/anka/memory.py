"""His memory: facts, diary, conversations, saving to disk, searching by meaning (memory_vectors.npz)
and how memories felt (feeling_vectors.npz)."""
import difflib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import date, datetime

import numpy as np

from settings import (PICTURE_MEMORY, PICTURE_DIR, MAX_PICTURES, PICTURE_KEEP_WIDTH, HABITUATION, HABIT_FADE, HABIT_STRENGTH, MEMORY_EARLIER_SLOT, MEMORY_RECENT, MEMORY_REPEAT_PENALTY, POPUPS_FOLLOW_THOUGHT, FEELING_BONUS, FEELING_VECTORS, FEELING_VECTOR_FILE, FOLDER, MAX_DIARY, MAX_FACTS, MEMORIES_TO_MIND, MEMORY_AI, MEMORY_FILE, MOOD_BONUS, NAME, OLD_VECTOR_FILE, ROOMS, ROOM_BONUS, SAVE_EVERY, SEARCH_CONVERSATIONS, SEARCH_DIARY, SHORT_OUTPUT, SIM, VECTOR_FILE)
from anka.shared import S, state, unsure_words



def move_old_memory_files():
    """Very early versions (anka.py) saved the memory under Swedish file names – move it to the new name once."""
    if SIM:
        return
    for old, new in (("minne.json", MEMORY_FILE), ("minne_vektorer.json", OLD_VECTOR_FILE)):
        old = os.path.join(FOLDER, old)
        if os.path.exists(old) and not os.path.exists(new):
            os.replace(old, new)
            if os.path.exists(old + ".bak"):
                os.replace(old + ".bak", new + ".bak")
            print(f"💾 Moved {os.path.basename(old)} → {os.path.basename(new)}")


facts = []          # facts about you (saved sorted into rooms in memory.json)
room_of = {}        # fact -> the room it is in
learned_on = {}     # fact -> the date Anka learned it
old_facts = []      # facts that were true but aren't any more: {"text", "room", "from", "until"}
goals = []          # Anka's own goals: {"text", "since", "t"}
cleanup_log = []    # everything the tidy-up changed, so it can be undone: {"date", "what", "before", "after"}
diary = []          # things Anka itself did and experienced: {"date", "time", "text"}
seen = []           # kinds of things Anka has ever seen
conversations = []  # all conversations: {"date", "time", "human", "duck"}
self_image = []     # who Anka has become
lessons = []        # what Anka learned about how it behaves, from looking back on its days
ideas = []          # 2026-11-09: things he thought up at night that he could TRY – untested ({"text", "date"})
own_said = []       # what Anka said by himself, not as an answer: {"time", "text"} (for looking back at night)
places = {}         # where things usually are, in words, kept between days: name -> {"where", "near", "date"}
memory_lock = threading.RLock()


def save_json(path, data, **json_options):
    """Saves safely: writes to a temporary file first and then swaps.
    If the disk is full or something goes wrong, the old file stays untouched.
    'data' can also be a function that writes the file itself (for binary files)."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb" if callable(data) else "w", **({} if callable(data) else {"encoding": "utf-8"})) as f:
            if callable(data):
                data(f)
            else:
                f.write(data if isinstance(data, str) else json.dumps(data, **json_options))
            f.flush()
            os.fsync(f.fileno())
        if os.path.exists(path):
            try:
                os.replace(path, path + ".bak")  # the previous version is kept as a backup
            except OSError:
                pass
        os.replace(tmp, path)
        return True
    except OSError as error:
        try:
            os.remove(tmp)
        except OSError:
            pass
        if error.errno == 28:
            print("💾 Disk is full – couldn't save memory! Free up space on your Mac. "
                  f"({NAME} keeps going, and the old memory file is untouched.)")
        else:
            print(f"💾 Couldn't save {os.path.basename(path)}: {error}")
        return False


def load_json(path, default):
    """Reads a saved file. If it is damaged, the backup (.bak) is tried instead."""
    for candidate in (path, path + ".bak"):
        try:
            with open(candidate, encoding="utf-8") as f:
                data = json.load(f)
            if candidate != path:
                print(f"💾 {os.path.basename(path)} was damaged – loaded the backup instead")
            return data
        except FileNotFoundError:
            continue
        except ValueError:
            damaged = candidate + f".damaged-{int(time.time())}"
            try:
                os.replace(candidate, damaged)  # keep the damaged file, never overwrite it
                print(f"💾 {os.path.basename(candidate)} is damaged – kept it as {os.path.basename(damaged)}")
            except OSError:
                pass
    return default


def load_memory():
    # (the lists and dicts are filled IN PLACE – every part of Anka holds the very same ones)
    pass  # (shared values live on S)
    data = load_json(MEMORY_FILE, {})
    saved = data.get("facts", [])
    room_of.clear()
    facts.clear()
    if isinstance(saved, dict):          # {"room": [fact, ...]}
        for room, items in saved.items():
            for f in items:
                facts.append(f)
                if room != "unsorted":
                    room_of[f] = room
    else:                                # an old plain list – sorted into rooms at start
        facts[:] = list(saved)
    learned_on.clear()
    learned_on.update(data.get("fact_dates", {}))
    old_facts[:] = data.get("old_facts", [])
    diary[:] = data.get("diary", [])
    seen[:] = data.get("seen", [])
    conversations[:] = data.get("conversations", [])
    self_image[:] = data.get("self", [])
    lessons[:] = data.get("lessons", [])
    ideas[:] = data.get("ideas", [])
    own_said[:] = data.get("own_said", [])
    places.clear()
    places.update(data.get("places", {}))
    S.last_dream = data.get("last_dream")
    feelings.clear()
    feelings.update({**START_FEELINGS, **data.get("feelings", {})})
    goals[:] = data.get("goals", [])
    cleanup_log[:] = data.get("cleanup_log", [])
    prune_pictures()
    print(f"🧠 Remembers {len(facts)} facts, {len(diary)} diary entries "
          f"and {len(conversations)} old conversations")
    if old_facts:
        print(f"🕰️  Knows {len(old_facts)} things that used to be true")
    if self_image:
        print("🪞 Who I am: " + " ".join(self_image))


memory_changed = threading.Event()
vectors_changed = threading.Event()


def save_memory():
    """Marks the memory as changed. It is written to disk in the background (see save_now)."""
    memory_changed.set()


def save_now():
    """Writes what has changed to disk. Runs every SAVE_EVERY seconds, and when Anka falls asleep."""
    if memory_changed.is_set():
        memory_changed.clear()
        with memory_lock:  # turn it into text while nothing changes – the slow disk part comes after
            palace = {}
            for f in facts:
                palace.setdefault(room_of.get(f, "unsorted"), []).append(f)
            text = json.dumps({
                "self": self_image, "lessons": lessons, "ideas": ideas, "own_said": own_said, "places": places, "facts": palace, "old_facts": old_facts, "diary": diary,
                "seen": seen, "conversations": conversations, "last_dream": S.last_dream,
                "feelings": {k: round(v, 2) for k, v in feelings.items()}, "goals": goals,
                "cleanup_log": cleanup_log[-300:],
                "fact_dates": {k: v for k, v in learned_on.items() if k in facts}},
                indent=2, ensure_ascii=False)
        save_json(MEMORY_FILE, text)
    if vectors_changed.is_set():
        vectors_changed.clear()
        with vector_lock:
            texts = list(vectors)
            matrix = np.array([vectors[t] for t in texts], dtype="float16") if texts else np.zeros((0, 0), "float16")
        save_json(VECTOR_FILE, lambda f: np.savez(f, texts=np.array(texts, dtype=str), vectors=matrix))
    if feeling_vecs_changed.is_set():
        feeling_vecs_changed.clear()
        with feeling_lock:
            ftexts = list(feeling_vecs)
            fmatrix = (np.array([feeling_vecs[t] for t in ftexts], dtype="float16") if ftexts
                       else np.zeros((0, 0), "float16"))
        save_json(FEELING_VECTOR_FILE, lambda f: np.savez(f, texts=np.array(ftexts, dtype=str), vectors=fmatrix))


def saver_loop():
    while True:
        time.sleep(SAVE_EVERY)
        try:
            save_now()
        except Exception as error:
            print(f"💾 Couldn't save ({type(error).__name__}: {error})")


def write_diary(text, feel=None, feeling=None):
    """Adds an entry to Anka's diary (call with memory_lock held). feel = how much it meant (else: right now)."""
    if feel is None and text.startswith(("I decided:", "I gave up on:", "I finished my goal:", "I took a nap")):
        feel = 0.1  # (bookkeeping, not a moment that moved him)
    diary.append({"date": date.today().isoformat(), "time": datetime.now().isoformat(timespec="seconds"),
                  "text": text, "feel": moment_strength() if feel is None else feel})
    if feeling is None and not text.startswith(("I decided:", "I gave up on:", "I finished my goal:", "I took a nap")):
        feeling = feeling_now(120)  # (how he feels right now, in his own words)
    if feeling:
        diary[-1]["feeling"] = feeling
    del diary[:-MAX_DIARY]
    if not (SHORT_OUTPUT and text.startswith("I thought: ")):  # the thought already shows in the 💭 line
        print(f"📔 Diary: {text}")


def keep_picture(picture, why, feel=None):
    """A picture memory: a small JPEG kept with a diary line in his own words (call WITHOUT memory_lock)."""
    import cv2
    if picture is None or not PICTURE_MEMORY:
        return None
    try:
        os.makedirs(PICTURE_DIR, exist_ok=True)
        h, w = picture.shape[:2]
        small = cv2.resize(picture, (PICTURE_KEEP_WIDTH, int(h * PICTURE_KEEP_WIDTH / w))) if w > PICTURE_KEEP_WIDTH else picture
        name = datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".jpg"
        if not cv2.imwrite(os.path.join(PICTURE_DIR, name), small, [cv2.IMWRITE_JPEG_QUALITY, 80]):
            return None
    except Exception as error:
        print("🖼️  couldn't keep the picture:", error)
        return None
    with memory_lock:
        write_diary(f"I kept a picture in my memory: {why}", feel=feel)
        diary[-1]["picture"] = name
        with_pictures = [d for d in diary if d.get("picture")]
        for d in with_pictures[:-MAX_PICTURES]:         # the oldest pictures are let go - the words stay
            try:
                os.remove(os.path.join(PICTURE_DIR, d.pop("picture")))
            except Exception:
                pass
    save_memory()
    print(f"🖼️  (a picture kept in his memory: {name})")
    return name


def picture_b64(name):
    """An old picture memory as base64 JPEG, or None if it is gone."""
    import base64
    path = os.path.join(PICTURE_DIR, name or "")
    if not name or not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def prune_pictures():
    """Picture files that no diary line points to any more (the diary is limited) are deleted."""
    if not os.path.isdir(PICTURE_DIR):
        return
    with memory_lock:
        used = {d.get("picture") for d in diary if d.get("picture")}
    for f in os.listdir(PICTURE_DIR):
        if f.endswith(".jpg") and f not in used:
            try:
                os.remove(os.path.join(PICTURE_DIR, f))
            except Exception:
                pass


def when(day):
    days = (date.today() - date.fromisoformat(day)).days
    return "today" if days == 0 else "yesterday" if days == 1 else f"{days} days ago"


def forget_everything():
    pass  # (shared values live on S)
    with memory_lock:
        for x in (facts, diary, seen, conversations, self_image, lessons, ideas, own_said, old_facts, goals, cleanup_log):
            x.clear()
        S.last_dream = None
        room_of.clear()
        learned_on.clear()
        feelings.clear()
        feelings.update(START_FEELINGS)
        save_memory()
    with vector_lock:
        vectors.clear()
        save_vectors()
    with feeling_lock:
        feeling_vecs.clear()
        feeling_vecs_changed.set()
    save_now()
    print("🧠 Memory erased")


vectors = {}            # text -> list of numbers
vector_lock = threading.Lock()


def load_vectors():
    """Loads the memory AI's numbers. The old text format (memory_vectors.json) is converted once."""
    vectors.clear()
    for path in (VECTOR_FILE, VECTOR_FILE + ".bak"):
        try:
            with np.load(path) as data:
                vectors.update({str(t): v.astype("float32") for t, v in zip(data["texts"], data["vectors"])})
            break
        except FileNotFoundError:
            continue
        except Exception as error:
            print(f"💾 {os.path.basename(path)} couldn't be read ({error}) – {NAME} works it out again")
    if not SIM and os.path.exists(OLD_VECTOR_FILE):
        old = load_json(OLD_VECTOR_FILE, {})
        for t, v in old.items():
            vectors.setdefault(t, np.asarray(v, dtype="float32"))
        size = os.path.getsize(OLD_VECTOR_FILE) if os.path.exists(OLD_VECTOR_FILE) else 0
        save_vectors()
        save_now()
        for path in (OLD_VECTOR_FILE, OLD_VECTOR_FILE + ".bak"):
            if os.path.exists(path):
                os.remove(path)
        print(f"💾 Moved the memory AI's numbers to a compact file "
              f"({size / 1e6:.1f} MB → {os.path.getsize(VECTOR_FILE) / 1e6:.1f} MB)")


def save_vectors():
    vectors_changed.set()


# ---------- The feeling space: how memories FELT, as numbers (separate from what they MEAN) ----------
feeling_vecs = {}                       # feeling in words -> numbers
feeling_lock = threading.Lock()
feeling_vecs_changed = threading.Event()


def load_feeling_vectors():
    feeling_vecs.clear()
    for path in (FEELING_VECTOR_FILE, FEELING_VECTOR_FILE + ".bak"):
        try:
            with np.load(path) as data:
                feeling_vecs.update({str(t): v.astype("float32") for t, v in zip(data["texts"], data["vectors"])})
            break
        except FileNotFoundError:
            continue
        except Exception as error:
            print(f"💾 {os.path.basename(path)} couldn't be read ({error}) – {NAME} works it out again")


def feeling_vectors(texts):
    """Numbers for feelings in words (worked out once, then kept). None if the memory AI is away.
    Uses the memory AI's 'clustering' mode: made for 'which of these are alike', not for searching."""
    if not S.memory_ai_ok or not FEELING_VECTORS:
        return None
    with feeling_lock:
        missing = [t for t in dict.fromkeys(texts) if t not in feeling_vecs]
    try:
        for i in range(0, len(missing), 64):
            new = embed(missing[i:i + 64], "clustering")
            with feeling_lock:
                feeling_vecs.update((t, normalize(np.asarray(v, dtype="float32"))) for t, v in zip(missing[i:i + 64], new))
        if missing:
            feeling_vecs_changed.set()
    except Exception:
        return None
    with feeling_lock:
        return [feeling_vecs[t] for t in texts]


def feeling_now(max_age=180):
    """How he feels right now in his own words (from his latest thought or answer), if it's fresh."""
    t, text = state.get("feeling_text", (0, None))[:2]
    return text if text and time.time() - t < max_age else None


def mood_match(texts):
    """How alike each feeling is to how he feels right now: 0 = not at all, 1 = the same. None = can't tell."""
    mood = feeling_now()
    if not mood or not texts:
        return None
    vecs = feeling_vectors([mood] + list(texts))
    if vecs is None:
        return None
    sims = np.array(vecs[1:]) @ vecs[0]
    # How alike is "unrelated" in this space? Measured against all the feelings he has had, not guessed:
    # only feelings clearly closer than his usual ones count
    with feeling_lock:
        others = [v for t, v in feeling_vecs.items() if t != mood]
    usual = float(np.mean(np.array(others) @ vecs[0])) if len(others) >= 8 else 0.74  # (measured: typical median)
    return np.clip((sims - usual) / max(0.05, (1 - usual) / 2), 0, 1)


def embed(texts, kind):
    """Turns texts into numbers with the memory AI. kind = 'search_query' or 'search_document'."""
    data = json.dumps({"model": MEMORY_AI, "input": [f"{kind}: {t}" for t in texts],
                       "keep_alive": "30m"}).encode()  # stay loaded, so searching doesn't wait for it
    req = urllib.request.Request("http://localhost:11434/api/embed", data=data,
                                 headers={"Content-Type": "application/json"})
    with heavy(), urllib.request.urlopen(req, timeout=60) as answer:
        return json.loads(answer.read())["embeddings"]


def document_vectors(texts):
    """Gets saved numbers, and only computes the missing ones."""
    with vector_lock:
        missing = [t for t in dict.fromkeys(texts) if t not in vectors]
    if missing:
        new = []
        for i in range(0, len(missing), 64):  # in batches, so the first start isn't too heavy
            new += embed(missing[i:i + 64], "search_document")
        with vector_lock:
            vectors.update((t, np.asarray(v, dtype="float32")) for t, v in zip(missing, new))
            save_vectors()
    with vector_lock:
        return [vectors[t] for t in texts]


def all_memories():
    """Everything Anka remembers, as (text to search in, text to show the brain, kind, how strongly it was felt)."""
    with memory_lock:
        out = [(f, f, "fact", 0.0) for f in facts]
        out += [(o["text"], f"until {when(o['until'])}: {o['text']}", "old", 0.0) for o in old_facts]
        out += [(d["text"], f"{when(d['date'])}: {d['text']}" + (f" (it felt: {d['feeling']})" if d.get("feeling") else "")
                 + (" [a picture memory]" if d.get("picture") else ""),
                 "diary", d.get("feel", 0.0), d.get("feeling"), d["date"], d.get("picture")) for d in diary[-SEARCH_DIARY:]]
        out += [(f'The human said "{c["human"]}" and I answered "{c["duck"]}"',
                 f'{when(c["date"])}: {c.get("who") or "the human"} said "{c["human"]}", you answered "{c["duck"]}"'
                 + (f" (it felt: {c['feeling']})" if c.get("feeling") else f" (you felt {c['felt']})" if c.get("felt") else ""),
                 "talk", c.get("feel", 0.0), c.get("feeling"), c.get("date")) for c in conversations[-SEARCH_CONVERSATIONS:]]
    return out


def normalize(M):
    return M / (np.linalg.norm(M, axis=-1, keepdims=True) + 1e-9)


_rooms = {}
_questions = {}  # the memory AI's numbers for recent questions


def room_vectors():
    """The memory AI's numbers for every room description (computed once)."""
    if not _rooms:
        names = list(ROOMS)
        _rooms["v"] = (names, np.array(document_vectors([f"{n}: {ROOMS[n]}" for n in names])))
    return _rooms["v"]


def question_vector(question):
    if question not in _questions:
        if len(_questions) > 200:
            _questions.pop(next(iter(_questions)))
        _questions[question] = np.array(embed([question], "search_query")[0])
    return _questions[question]


_recently_shown = []   # B5: the memories shown in the latest questions (to vary them)
GOAL_LINES = ("I decided:", "I gave up on:", "I finished my goal:")   # his goal bookkeeping in the diary
_used_to = {}          # habituation: memory text -> how "used to it" he is right now (grows when shown, fades per question)


def search_memory(question, candidates, count):
    """Returns the candidates closest in meaning to 'question'. None if the memory AI is missing."""
    pass  # (shared values live on S)
    if not S.memory_ai_ok:
        return None
    if not candidates:
        return []
    try:
        q = question_vector(question)
        M = np.array(document_vectors([c[0] for c in candidates]))
    except Exception as error:
        S.memory_ai_ok = False
        print(f"🧠 Memory AI not available ({error}). Run: ollama pull {MEMORY_AI}")
        return None
    similarity = normalize(M) @ normalize(q)
    hit = similarity > 0.45
    # The memory palace: memories in the rooms the question is about get a little extra weight
    try:
        names, R = room_vectors()
        R = normalize(R)
        question_rooms = set(np.argsort(-(R @ normalize(q)))[:2])
        candidate_rooms = (normalize(M) @ R.T).argmax(axis=1)
        similarity = similarity + ROOM_BONUS * np.isin(candidate_rooms, list(question_rooms))
    except Exception:
        pass
    # Moments that moved him strongly come back more easily
    similarity = similarity + FEELING_BONUS * np.array([c[3] if len(c) > 3 else 0.0 for c in candidates])
    # ...and memories that FELT like he feels right now come back more easily (like a sad day brings sad memories)
    felt = [i for i, c in enumerate(candidates) if len(c) > 4 and c[4]]
    if MOOD_BONUS and felt:
        match = mood_match([candidates[i][4] for i in felt])
        if match is not None:
            similarity[felt] += MOOD_BONUS * match
    # Habituation: a memory that just came up pulls weaker for a while (grows each time it comes up, fades each question)
    if HABITUATION:
        similarity = similarity - HABIT_STRENGTH * np.array([_used_to.get(c[1], 0.0) for c in candidates])
    # (old way) what he was shown in his last few questions counts a bit less
    elif MEMORY_REPEAT_PENALTY:
        recent = set(x for shown in _recently_shown for x in shown)
        similarity = similarity - MEMORY_REPEAT_PENALTY * np.array([c[1] in recent for c in candidates])
    order = [i for i in np.argsort(-similarity) if hit[i]]
    best, surprises, bookkeeping = [], 0, 0
    for i in order:
        if not HABITUATION and candidates[i][0].startswith("I was surprised"):
            if surprises:
                continue                          # at most one surprise line per question
            surprises += 1
        if not HABITUATION and candidates[i][0].startswith(GOAL_LINES):
            if bookkeeping:
                continue                          # ...and at most one goal line ("I decided:" …) – 2026-11-09
            bookkeeping += 1
        best.append(i)
        if len(best) == count:
            break
    # B5: one slot for the best-fitting memory from an EARLIER day (else yesterday never reaches his thoughts)
    today = date.today().isoformat()
    earlier = lambda i: len(candidates[i]) > 5 and candidates[i][5] and candidates[i][5] < today
    if MEMORY_EARLIER_SLOT and best and not any(earlier(i) for i in best):
        old = [i for i in order if earlier(i) and i not in best]
        if old:
            best = best[:count - 1] + [old[0]]
    _recently_shown.append([candidates[i][1] for i in best])
    del _recently_shown[:-MEMORY_RECENT]
    for k in list(_used_to):                      # the "used to it" feeling fades a little with every question...
        _used_to[k] *= HABIT_FADE
        if _used_to[k] < 0.05:
            del _used_to[k]
    for i in best:                                # ...and grows for what came up now
        _used_to[candidates[i][1]] = _used_to.get(candidates[i][1], 0.0) + 1.0
    return [candidates[i] for i in best]


def which_room(texts):
    """Sorts texts into rooms with the memory AI. None if it isn't available."""
    if not texts or not S.memory_ai_ok:
        return None
    try:
        names, R = room_vectors()
        M = np.array(document_vectors(texts))
    except Exception:
        return None
    best = (normalize(M) @ normalize(R).T).argmax(axis=1)
    return [names[i] for i in best]


def sort_into_rooms(texts):
    """Puts facts in the right room and saves. Call without memory_lock held."""
    rooms = which_room(texts)
    if rooms is None:
        return {}
    with memory_lock:
        for t, r in zip(texts, rooms):
            room_of[t] = r
        save_memory()
    return dict(zip(texts, rooms))


def add_fact(text):
    """Adds a fact (call with memory_lock held)."""
    facts.append(text)
    learned_on[text] = date.today().isoformat()
    while len(facts) > MAX_FACTS:
        oldest = facts.pop(0)
        room_of.pop(oldest, None)
        learned_on.pop(oldest, None)


def mark_outdated(text):
    """Moves a fact to the timeline of things that used to be true (call with memory_lock held)."""
    if text in facts:
        facts.remove(text)
    old_facts.append({"text": text, "room": room_of.pop(text, "unsorted"),
                      "from": learned_on.pop(text, None), "until": date.today().isoformat()})
    del old_facts[:-MAX_FACTS]


def delete_fact(text):
    """Removes a fact completely (wrong from the start, or you asked Anka to forget it)."""
    if text in facts:
        facts.remove(text)
    room_of.pop(text, None)
    learned_on.pop(text, None)


def core(fact):
    """Removes 'The human' at the start, so 'has a cat' and 'has a dog' are compared fairly."""
    return re.sub(r"^(the human'?s?|the human is)\s*", "", fact.lower()).strip(" .")


def same_fact(a, b):
    return difflib.SequenceMatcher(None, core(a), core(b)).ratio() > 0.85


def remember(human_said, duck_said, speaker=None, feel=None, felt=None, feeling=None, stop=None):
    """Runs when the conversation pauses: adds, corrects and removes memories."""
    with memory_lock:
        everything = list(facts)
    shown = everything
    if len(everything) > 15:
        hits = search_memory(human_said, [(f, f, "fact") for f in everything], 12)
        if hits is not None:
            shown = [h[0] for h in hits]
    instructions = (
        "You manage the long-term memory of a robot duck. The memory holds facts about the HUMAN "
        "it talks to. Read what the human just said and update the memory.\n\n"
        "Write one command per line, using ONLY these forms:\n"
        "ADD: <new fact, a short sentence starting with 'The human'>\n"
        "OUTDATED: <number of an existing fact that WAS true but has changed, e.g. they moved>\n"
        "REMOVE: <number of an existing fact that was never true, or that the human wants forgotten>\n"
        "DIARY: <one short sentence in first person from the duck's view about something notable "
        "that happened, that the duck did or learned, e.g. 'I told Alex a joke about bread.'>\n"
        "NONE  (if nothing should change)\n\n"
        "Rules:\n"
        "- If the human corrects a mistake (\"no, my cat is called Sixten, not Max\"), "
        "REMOVE the wrong fact and ADD the correct one.\n"
        "- If something changed in the human's life (\"I moved to Cork\", \"I got a new job\"), "
        "mark the old fact OUTDATED and ADD the new one.\n"
        "- If the human asks to forget something, REMOVE it.\n"
        "- ADD only facts that will still be true weeks from now: name, age, family, friends, pets, "
        "home, job, hobbies, likes, dislikes, bigger plans, or things they ask the duck to remember.\n"
        "- Do NOT add what they want to do right now (like playing a game) or questions they ask.\n"
        f"- Only facts about the HUMAN. Things about the duck ({NAME}) are not saved.\n"
        "- Never ADD something that is already in the memory.\n"
        "- ADD facts only from what the HUMAN said, never from what the duck said.\n"
        "- 'our', 'we' and 'us' mean the human and the people they live with – never the duck. "
        "Write who someone is to the HUMAN, e.g. 'The human's partner is called Robin.'\n"
        "- Write at most one DIARY line, and only if something worth remembering happened "
        "(not for small talk like 'how are you')."
    )
    numbered = "\n".join(f"{i}. {m}" for i, m in enumerate(shown, 1)) or "(empty)"
    try:
        text = ollama([
            {"role": "system", "content": instructions},
            {"role": "user", "content": f'Current memory:\n{numbered}\n\n'
                                        + (f'{speaker} (a visitor, NOT the main human – write facts about '
                                           f'{speaker} with the name, e.g. "{speaker} has a cat.") just said: '
                                           if speaker else "The human just said: ") + f'"{human_said}"\n'
                                        f'The duck ({NAME}) answered: "{duck_said}"'
                                        + ("\n(The ears were NOT sure about what the human said – it may be "
                                           "misheard. Don't ADD facts from it.)" if human_said in unsure_words else "")},
        ], temperature=0.1, stop=stop)
    except Interrupted:
        raise  # you started talking – it's done again later
    except Exception as error:
        print("🧠 Could not update memory:", error)
        return

    remove, outdated, new, event = set(), set(), [], None
    for line in text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        m = re.match(r"(?i)diary\s*:\s*(.+)", line)
        if m and event is None and 3 <= len(m.group(1).split()) <= 30:
            event = m.group(1).strip()
            continue
        m = re.match(r"(?i)(remove|outdated)\s*:?\s*#?(\d+)", line)
        if m:
            nr = int(m.group(2))
            if 1 <= nr <= len(shown):
                (outdated if m.group(1).lower() == "outdated" else remove).add(nr - 1)
            continue
        m = re.match(r"(?i)add\s*:\s*(.+)", line)
        if m:
            fact = re.sub(r"^\d+[.)]\s*", "", m.group(1).strip())
            # Guard against made-up facts: numbers and names must come from what you said (or be known)
            known = human_said + " " + " ".join(everything)
            new_numbers = [t for t in re.findall(r"\d+", fact) if t not in known]
            new_names = [n for n in re.findall(r"\b[A-Z][a-z]+\b", fact)
                         if n not in ("The", "Human", "He", "She", "They") and n.lower() not in known.lower()]
            new_number_words = [w for w in re.findall(r"\b(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|"
                                                      r"ninety|one|two|three|four|five|six|seven|eight|nine|ten)\b",
                                                      fact.lower())
                                if w not in known.lower()]
            if new_numbers or new_names or new_number_words:
                print(f"🧠 (didn't save – the human never said that: {fact})")
                continue
            gone = {shown[i] for i in remove | outdated}
            if 3 <= len(fact.split()) <= 25 and not any(
                    same_fact(fact, x) for x in everything if x not in gone) \
                    and not any(same_fact(fact, x) for x in new):
                new.append(fact)

    outdated -= remove
    if not remove and not outdated and not new and not event:
        if not SHORT_OUTPUT:
            print("🧠 (nothing new to remember)")
        return
    with memory_lock:
        if event:
            own_feeling = feeling  # (how it felt in his own words)
            # the diary says how it felt in HIS words; only if he gave none, the guess from his feeling numbers
            feeling = own_feeling or (felt if feel is not None else strong_feeling())
            write_diary(event.rstrip(".") + f" (I felt {feeling})." if feeling else event, feel=feel,
                        feeling=own_feeling)
        removed = [shown[i] for i in sorted(remove)]
        for f in removed:
            delete_fact(f)
        changed = [shown[i] for i in sorted(outdated)]
        for f in changed:
            mark_outdated(f)
        for f in new:
            add_fact(f)
        save_memory()
    for f in removed:
        print(f"🧠 Forgot: {f}")
    for f in changed:
        print(f"🕰️  No longer true: {f}")
    rooms = sort_into_rooms(new)
    for f in new:
        print(f"🧠 Remembered ({rooms.get(f, 'unsorted')}): {f}")


def since_last_dream(entry):
    """True if a diary entry or a conversation happened after the last dream."""
    if not S.last_dream:
        return True
    if "time" in entry:
        return entry["time"] > S.last_dream
    return entry["date"] >= S.last_dream[:10]  # older entries only have a date


def prune_vectors():
    """Forgets the memory AI's numbers for things that are no longer in the memory (old diary lines,
    removed facts, mistakes forgotten at night) – otherwise memory_vectors.npz only grows, forever."""
    keep = {c[0] for c in all_memories()} | {f"{n}: {ROOMS[n]}" for n in ROOMS}
    with vector_lock:
        old = [t for t in vectors if t not in keep]
        for t in old:
            del vectors[t]
        if old:
            save_vectors()
    if old:
        print(f"🧹 Cleared {len(old)} unused memory numbers")
    with memory_lock:
        felt = {x.get("feeling") for x in diary + conversations} | {x.get("first_feeling") for x in diary + conversations}
    felt |= {feeling_now(10 ** 9)}
    with feeling_lock:
        gone = [t for t in feeling_vecs if t not in felt]
        for t in gone:
            del feeling_vecs[t]
        if gone:
            feeling_vecs_changed.set()


def human_name():
    """The human's name, if Anka has learned it (e.g. from "The human's name is Alex.")."""
    with memory_lock:
        for f in facts:
            m = re.search(r"(?i)\b(?:name is|named|called)\s+([A-Z][a-z]+)", f)
            if m and f.lower().startswith("the human") and "friend" not in f.lower():
                return m.group(1)
    return None


_came_to_mind = []   # memories that popped up lately (so the same ones don't come back every thought)


def comes_to_mind(name, how_many=MEMORIES_TO_MIND, about=None):
    """A few random things from the WHOLE memory – like memories that just pop up in a real mind.
    Older days, facts about the human, things said long ago, who Anka is. What to do with them is up to the brain."""
    if how_many <= 0:
        return []
    today = date.today().isoformat()
    who = name or "The human"
    with memory_lock:
        felt = lambda x: f" (it felt: {x['feeling']})" if x.get("feeling") else ""
        pool = [("fact", f, 0.0, None) for f in facts
                if not re.match(r"(?i)the human('s name)? is (named|called)|the human's name", f)]
        pool += [("diary", f'({when(d["date"])}) {d["text"]}{felt(d)}', d.get("feel", 0.0), d.get("feeling")) for d in diary
                 if d["date"] != today and not d["text"].startswith(("I thought:", "I decided:", "I finished my goal:", "I gave up on:"))]
        pool += [("talk", f'({when(c["date"])}) {c.get("who") or who} said: "{c["human"]}" – you answered: "{c["duck"]}"{felt(c)}',
                  c.get("feel", 0.0), c.get("feeling")) for c in conversations[-SEARCH_CONVERSATIONS:] if c["date"] != today]
        pool += [("self", f"About yourself: {x}", 0.0, None) for x in self_image]
    pool = [p for p in pool if p[1] not in _came_to_mind]
    # memories that felt like he feels now pop up more easily (up to 3 times as often)
    with_feeling = [i for i, p in enumerate(pool) if p[3]]
    match = mood_match([pool[i][3] for i in with_feeling]) if MOOD_BONUS and with_feeling else None
    mood = {i: float(m) for i, m in zip(with_feeling, match)} if match is not None else {}
    pool = [(k, t, w * (1 + 2 * mood.get(i, 0.0)) + 0.5 * mood.get(i, 0.0)) for i, (k, t, w, _) in enumerate(pool)]
    # B5: memories closer to what he was just thinking about pop up more easily (still a random pick – a mind wanders)
    near = {}
    if POPUPS_FOLLOW_THOUGHT and about and pool and S.memory_ai_ok:
        try:
            q = normalize(question_vector(about))
            sims = normalize(np.array(document_vectors([t for _, t, _ in pool]))) @ q
            lo, hi = float(sims.min()), float(sims.max())
            near = {i: (float(v) - lo) / (hi - lo + 1e-9) for i, v in enumerate(sims)}
        except Exception:
            near = {}
    rel = {t: near[i] for i, (k, t, w) in enumerate(pool)} if near else {}
    if not pool:
        return []
    picked = []
    kinds = list({k for k, _, _ in pool})
    random.shuffle(kinds)
    for kind in kinds[:how_many]:  # a mix of kinds, not three facts in a row
        options = [(t, w) for k, t, w in pool if k == kind]
        # moments that moved him strongly pop up more often (a strong one up to 4 times as often)
        picked.append(random.choices([t for t, _ in options],
                                     weights=[(1 + 3 * w * (FEELING_BONUS > 0)) * ((0.3 + 2.7 * rel.get(t_, 0.5)) if rel else 1)
                                              for t_, w in options])[0])
    _came_to_mind.extend(picked)
    del _came_to_mind[:-12]
    return [t.replace("The human", who) if name else t for t in picked]


def how_long(seconds):
    """'about 5 minutes' – for how long something lasted."""
    minutes = round(seconds / 60)
    return "less than a minute" if seconds < 45 else "about a minute" if minutes <= 1 else \
        f"about {minutes} minutes" if minutes < 60 else f"about {round(seconds / 3600)} hours"


def ago(seconds):
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} minutes ago"
    return f"{int(seconds // 3600)} hours ago"


# ---- other parts of Anka (imported last, so the parts can use each other) ----
from anka.resources import heavy
from anka.feelings import START_FEELINGS, feelings, moment_strength, strong_feeling
from anka.brain import Interrupted, ollama
