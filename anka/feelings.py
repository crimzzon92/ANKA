"""His feelings: the seven numbers, how strongly a moment moved him, and reading FEEL / MOMENT / FEELING."""
import re
import threading
import time

from settings import (DRIFT, ECHO_HALF_LIFE, ECHO_MIN_JUMP, ECHO_SHOW, FEELING_ECHO, FEELING_SPEED, FROZEN_FEELINGS,
                      RESTING, SET_FEELINGS, STRONG)
from anka.shared import state

START_FEELINGS = {"curiosity": 0.5, "boredom": 0.0, "loneliness": 0.3,
                  "happiness": 0.5, "affection": 0.4, "nervousness": 0.1, "playfulness": 0.4}
FEELING_NAMES = "|".join(START_FEELINGS)
feelings = dict(START_FEELINGS)  # Anka's drives right now
feelings_lock = threading.RLock()  # (RLock: the diary reads his feelings while the goals code holds them)


def hold_frozen_feelings():
    """Experiments only (--frozen-feelings / --feelings k=v): puts the numbers at their resting level (or the set
    values) and keeps them there. Called after his memory is loaded, because the memory also stores feelings."""
    if FROZEN_FEELINGS:
        with feelings_lock:
            feelings.clear()
            feelings.update({**START_FEELINGS, **SET_FEELINGS})
        print("🧊 EXPERIMENT – his feelings are held still: " + ", ".join(f"{k}={v:.2f}" for k, v in feelings.items()))


def level(v):
    return "low" if v < 0.35 else "medium" if v < STRONG else "HIGH"


FEELING_MEANING = {
    "curiosity": "you really want to discover or learn something new",
    "boredom": "nothing is happening and you are bored",
    "loneliness": "you long for company and attention",
    "happiness": "you are really glad and joyful",
    "affection": "you feel warm love for the people around you",
    "nervousness": "you feel uneasy or a bit scared",
    "playfulness": "you want to joke, tease and play",
}


def inner_state(with_goals=True):
    """Text for the brain about how Anka feels and what it wants."""
    with feelings_lock:
        lines = []
        for k, v in feelings.items():
            meaning = FEELING_MEANING.get(k, "")
            lines.append(f"- {k}: {level(v)}" + (f" ({meaning})" if v >= STRONG and meaning else ""))
        text = "How you feel right now:\n" + "\n".join(lines)
        after = echo_text()
        if after:
            text += "\n" + after
        if not with_goals:
            pass
        elif goals:
            text += "\nYour own goals right now (drop a goal with DONE if your senses can't reach it):\n" + "\n".join(
                f"{i}. {g['text']}" + (f"  (set {ago(time.time() - g['t'])}, not done yet)" if "t" in g else "")
                for i, g in enumerate(goals, 1))
        else:
            text += "\nYou have no goals right now."
    return text


FEELING_WORDS = {"happiness": "happy", "affection": "full of love", "nervousness": "nervous",
                 "playfulness": "playful", "loneliness": "lonely", "boredom": "bored", "curiosity": "curious"}


JOLT_WORDS = {("happiness", 1): "very happy", ("happiness", -1): "hurt", ("affection", 1): "full of love",
              ("affection", -1): "hurt", ("nervousness", 1): "scared", ("nervousness", -1): "relieved",
              ("playfulness", 1): "playful", ("loneliness", 1): "lonely"}
recent_moments = []   # (when, how strong 0–1, a feeling word or None) – how much things just moved him


def note_jolt(before, after, source="thought"):
    """Feelings just changed (the brain set them): remembers how big the jump was – a shout that drops his
    happiness from 0.9 to 0.3 is a strong moment even if he was already happy all evening."""
    best, which = 0.0, None
    for k in list(RESTING) + ["loneliness"]:
        d = after.get(k, 0) - before.get(k, 0)
        if abs(d) > abs(best):
            best, which = d, k
    size = min(1.0, abs(best) * 2)
    if size >= 0.1 or (source == "reply" and best):  # (a reply's small change still says WHICH way it moved him)
        recent_moments.append((time.time(), size, JOLT_WORDS.get((which, 1 if best > 0 else -1)), source))
        del recent_moments[:-20]


# ---- feelings that outlast the event ----
echoes = {}   # feeling -> {"peak", "rest", "t", "cause", "word"}: a big jump that is still fading


def _echo_bound(e, now):
    """How far from normal the after-feeling still holds the feeling (fades half the way every half-life)."""
    return e["rest"] + (e["peak"] - e["rest"]) * 0.5 ** ((now - e["t"]) / e["half"])


def settle_echoes(before, cause=None, now=None):
    """Call with feelings_lock held, right after the brain (or time) changed the feelings.
    (1) A big jump away from normal starts an after-feeling (or renews it if it is stronger).
    (2) No feeling with an after-feeling may be closer to normal than the after-feeling still is – raising it
        more, or feeling something else as well, is always possible. Like a body: fear fades, it doesn't switch off."""
    if not FEELING_ECHO or FROZEN_FEELINGS:
        return
    now = now or time.time()
    for k, half in ECHO_HALF_LIFE.items():
        rest = RESTING.get(k, (0.5, 0))[0]
        b, a = before.get(k, rest), feelings.get(k, rest)
        old = echoes.get(k)
        same_way = old and (old["peak"] > rest) == (a > rest)
        if abs(a - b) >= ECHO_MIN_JUMP - 1e-9 and (a - b > 0) == (a - rest > 0) and abs(a - rest) >= 0.2 \
                and not (same_way and abs(a - rest) < abs(_echo_bound(old, now) - rest)):
            echoes[k] = {"peak": a, "rest": rest, "t": now, "since": now, "half": half,      # a new big moment
                         "cause": (cause or "something happened")[:160],
                         "word": JOLT_WORDS.get((k, 1 if a > rest else -1)) or FEELING_WORDS.get(k, k)}
        elif same_way and abs(a - rest) > abs(_echo_bound(old, now) - rest):
            old["peak"], old["t"] = a, now        # it grew stronger (still the same moment, e.g. 0.8 -> 0.98)
    for k, e in list(echoes.items()):
        bound = _echo_bound(e, now)
        if now - e["t"] > ECHO_SHOW and abs(bound - e["rest"]) < 0.05:
            del echoes[k]
            continue
        if abs(bound - e["rest"]) < 0.05:
            continue                                   # (faded – only remembered now)
        cur = feelings.get(k, e["rest"])
        feelings[k] = max(cur, bound) if e["peak"] > e["rest"] else min(cur, bound)


def echo_text(now=None):
    """Facts for the brain: big feelings from earlier that are still in him, where they came from, and how much."""
    if not FEELING_ECHO or FROZEN_FEELINGS or not echoes:
        return ""
    now = now or time.time()
    lines = []
    for k, e in sorted(echoes.items(), key=lambda x: x[1]["t"]):
        left = abs(_echo_bound(e, now) - e["rest"]) / max(0.01, abs(e["peak"] - e["rest"]))
        how = ("still strong in you" if left > 0.6 else "still there, slowly fading" if left > 0.25 else
               "a little of it is still there" if left > 0.08 else "it has faded now, but you remember how it felt")
        lines.append(f"- you felt {e['word']} {ago(now - e.get('since', e['t']))}, when {e['cause']} – {how}")
    return ("Big feelings from earlier (a feeling like this doesn't switch off when things change – it fades "
            "slowly, like in an animal's body):\n" + "\n".join(lines))


def clean_feeling(text):
    """'happy but a bit shy.' -> 'happy but a bit shy' (None if it says nothing)."""
    text = re.sub(r"[\"“”<>]", "", (text or "").splitlines()[0] if text else "").strip(" .,-")
    return text if 1 <= len(text.split()) <= 15 and not re.fullmatch(r"(?i)none|nothing|n/?a", text) else None


def take_feeling(answer, source="reply"):
    """Takes the FEELING line (how he feels, in his own words) off an answer and remembers it."""
    m = re.search(r"(?i)\n?[ \t]*FEELING\s*:\s*([^\n]*)", answer or "")
    if not m:
        return answer
    text = clean_feeling(m.group(1))
    if text and not FROZEN_FEELINGS:   # (experiment: his own feeling words don't carry over either)
        state["feeling_text"] = (time.time(), text, source)
    return (answer[:m.start()] + answer[m.end():]).strip()


def take_moment(answer):
    """Takes the brain's MOMENT line (how much this moment means to him, 0–1) off an answer and remembers it."""
    m = re.search(r"(?i)\n?[ \t]*MOMENT\s*:\s*([01](?:\.\d+)?)[^\n]*", answer or "")
    if not m:
        return answer
    recent_moments.append((time.time(), min(1.0, float(m.group(1))), "moment", "reply"))
    del recent_moments[:-20]
    return (answer[:m.start()] + answer[m.end():]).strip()


def strong_feeling(limit=0.75, since=None):
    """How he felt about what JUST happened, as words for the diary ('hurt', 'very happy') – only if it moved
    him. Otherwise None. First choice: what his feelings jumped to; else, for a moment he rated as big,
    his strongest emotion right now."""
    now = time.time()
    fresh = [(t, s, w) for t, s, w, src in recent_moments
             if (t >= since and src == "reply" if since is not None else now - t < 120)]
    jolts = [(s, w) for t, s, w in fresh if w not in (None, "moment") and s >= 0.5]
    if jolts:
        return max(jolts)[1]
    if not any(w == "moment" and s >= 0.6 for t, s, w in fresh):
        return None
    moved = [(s, w) for t, s, w in fresh if w in ("very happy", "full of love", "scared", "playful", "lonely")]
    if moved:  # a big moment, but his feelings only moved a little (he was already happy) – name that way
        return max(moved)[1]
    with feelings_lock:
        candidates = {k: v for k, v in feelings.items() if k in RESTING or k == "loneliness"}
    normal = {k: rest for k, (rest, _) in RESTING.items()} | {"loneliness": 0.3}
    strong = {k: v for k, v in candidates.items() if v >= limit}
    if not strong:
        return None
    k = max(strong, key=lambda k: strong[k] - normal.get(k, 0.5))  # the one furthest from normal
    word = FEELING_WORDS[k]
    return word if word.startswith("full") else f"very {word}"


def moment_strength(since=None):
    """How much what just happened meant to Anka, 0–1: his own rating (MOMENT) or how much his feelings jumped,
    fading over two minutes. Saved with each memory, so moments that moved him come back more easily later."""
    now = time.time()
    if since is not None:  # only this answer: his own rating, else how much his feelings jumped from it
        mine = [(w, s) for t, s, w, src in recent_moments if t >= since and src == "reply"]
        rated = [s for w, s in mine if w == "moment"]
        return round(rated[-1] if rated else max([s for _, s in mine] or [0.0]), 2)
    return round(max([s * max(0.0, 1 - (now - t) / 120) for t, s, _, _ in recent_moments] or [0.0]), 2)


def feel_now(answer, cause=None):
    """Takes the FEEL line off an answer and updates the feelings at once – so a shout or a compliment
    shows on his face and in his voice straight away, not only at his next thought."""
    if not answer:
        return answer
    m = re.search(r"(?is)\n?\s*FEEL\s*:\s*(.*)$", answer)
    if not m:
        return answer
    if FROZEN_FEELINGS:              # experiment: the numbers are held still - the brain's FEEL line is ignored
        return answer[:m.start()].strip()
    with feelings_lock:
        before = dict(feelings)
        for k, v in re.findall(rf"({FEELING_NAMES})\s*[=:]\s*([01](?:\.\d+)?)", m.group(1), re.I):
            feelings[k.lower()] = min(1.0, max(0.0, float(v)))
        note_jolt(before, feelings, "reply")
        heard = state.get("feel_cause")
        settle_echoes(before, cause or (heard[1] if heard and time.time() - heard[0] < 90 else None))
    return answer[:m.start()].strip()


FEEL_LABELS = {"curiosity": "curious", "boredom": "bored", "loneliness": "lonely", "happiness": "happy",
               "affection": "love", "nervousness": "nervous", "playfulness": "playful"}


def feelings_short():
    with feelings_lock:
        return " · ".join(f"{FEEL_LABELS.get(k, k)} {v:.2f}" for k, v in feelings.items())


def drift_feelings(previous, now):
    """Time passes: the needs grow slowly by themselves, like hunger in an animal.
    Everything else – how Anka feels when you talk, when something happens – the brain decides itself."""
    if FROZEN_FEELINGS:              # experiment: nothing drifts
        return
    minutes = (now - previous) / 60
    with feelings_lock:
        for k, (rest, speed) in RESTING.items():  # emotions slowly settle back to normal
            current = feelings.get(k, rest)
            feelings[k] = current + (rest - current) * min(1.0, speed * minutes * FEELING_SPEED)
        settle_echoes(dict(feelings))
    if now - state["last_human"] < 60:
        return  # you're talking – no needs grow (the brain says in its next thought how the talk felt)
    with feelings_lock:
        # Grows fast when the feeling is low and slower and slower towards 1
        for k, v in DRIFT.items():
            current = feelings.get(k, 0)
            feelings[k] = min(1.0, current + v * minutes * FEELING_SPEED * (1 - current))


# ---- other parts of Anka (imported last, so the parts can use each other) ----
from anka.memory import ago, goals
