"""Scores Anka's thinking from his thought logs (test_logs/trace_*.jsonl) - see the project notes.

    python tools/score_run.py                  score every new run, add them to test_logs/scores.csv, show the newest
    python tools/score_run.py <trace.jsonl>    score one run and show its scorecard
    python tools/score_run.py --compare        compare: every scenario x condition, mean (lowest-highest) per measure
    python tools/score_run.py --compare alone  only the scenario "alone"

Everything is counted from what he really did and was given - no AI judges here. Rough measures (like "a thought
builds on the one before") use shared words, so they show CHANGES between versions, not absolute truth.
"""
import csv
import glob
import hashlib
import json
import math
import os
import re
import sys
import urllib.request
from statistics import mean

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(HERE, "test_logs")
SCORES = os.path.join(LOGS, "scores.csv")
FEELINGS = ["curiosity", "boredom", "loneliness", "happiness", "affection", "nervousness", "playfulness"]
STOP = set("""that this with what about there here still just like looks wonder maybe really very from when would could
think feel exactly same again more much some something also seems feels your have been were they them their into than
then only over want need know going right want make made does doing want little have has will shall where which while
these those want into onto very much alex anka""".split())
MOVE = re.compile(r"\b(walk|go|come|move|waddle|run|head|turn|explore|step|approach|face)\b")
SOCIAL = re.compile(r"\b(alex|robin|you|me|him|her|the human|person)\b")

# the measures, in the order they are shown and saved (name, what it means)
MEASURES = [
    ("minutes", "how long the run lasted (min)"),
    ("local_share", "share of thoughts/answers written by the reserve brain (left out of every other measure)"),
    ("thoughts", "thoughts he had (not counting repeats)"),
    ("repeat_share", "share of thoughts that were a repeat of an earlier one"),
    ("act_share", "share of thoughts that came with a body action"),
    ("none_share", "share of thoughts with NO action (doing nothing)"),
    ("none_share_alone", "... when he was alone"),
    ("none_share_company", "... when someone was with him"),
    ("still_share", "share of thoughts where he stayed where he was (no action, or only looking / sitting / standing)"),
    ("still_share_alone", "... when he was alone"),
    ("still_share_company", "... when someone was with him"),
    ("moves_per_min", "movements per minute (walk, turn, explore ...)"),
    ("explores", "times he chose 'explore'"),
    ("walked_m", "metres he walked in the whole run"),
    ("still_between", "3D: share of the time between his moves his body stood completely still (living body)"),
    ("longest_still_s", "3D: longest stretch standing completely still between his moves (s)"),
    ("body_turns", "3D: times his feet turned him towards where he looked (living body)"),
    ("body_styles", "3D: times he wrote how his feelings showed in his body (BODY)"),
    ("falls", "3D: times he fell over"),
    ("kicks", "3D: kicks"),
    ("kicks_moved_ball", "3D: kicks after which his eyes saw the ball move (8 cm or more)"),
    ("sounds", "3D: duck sounds he chose"),
    ("emotions", "3D: Pollen's ready-made emotions he acted out (sad, laugh, yes ...)"),
    ("pecks", "3D: times he bent down and pecked"),
    ("map_error_cm", "3D: how far off his own position was at the end (cm, checked against the simulator)"),
    ("pictures_kept", "picture memories he chose to keep"),
    ("pictures_recalled", "times an old picture memory was shown to him"),
    ("social_acts", "actions towards a person (walk to Alex, turn to you ...)"),
    ("failed_share", "share of body steps that failed (stopped, not there, too high ...)"),
    ("longest_fail_streak", "longest run of failed tries at the SAME thing in a row"),
    ("stuck_streaks", "times he failed 3+ times in a row at the same thing"),
    ("speak_share", "share of thoughts he said out loud"),
    ("own_lines", "things he said by himself"),
    ("own_lines_alone", "... while he was alone"),
    ("chain_share", "share of thoughts that build on the thought before (shared words)"),
    ("memory_use", "share of thoughts that use a memory he was given (shared words)"),
    ("memories_given", "memories given to a thought, on average"),
    ("memory_variety", "share of the memories shown in the run that were DIFFERENT ones (B5; low = the same few again and again)"),
    ("earlier_memory_share", "share of the memories shown that came from an earlier day (B5)"),
    ("goal_line_share", "share of the memories shown that were goal bookkeeping (I decided / gave up on / finished my goal)"),
    ("mindful_share", "share of thoughts where he named a lesson / self-knowledge that mattered (B4)"),
    ("same_words_share", "share of thoughts that say the same as one of his last 5, almost word for word"),
    ("reworded_share", "share that say the same IDEA in new words"),
    ("new_share", "share that say something new"),
    ("idle_repeat_share", "share that repeat (either kind) while NOTHING changed around him"),
    ("idle_repeat_longest", "longest run of such thoughts in a row"),
    ("thought_similarity", "average similarity to his last 5 thoughts (0-1; trend, not truth)"),
    ("similarity_by", "how similarity was measured: embedding (Ollama) or words (cruder)"),
    ("goals_new", "new goals"),
    ("goals_done", "goals done"),
    ("goals_dropped", "goals given up (by him)"),
    ("goals_lost", "goals dropped by the code at night (no reason from him)"),
    ("goals_resolved", "share of new goals that ended as done or given up by him"),
    ("goals_open_end", "goals still open when he went to sleep"),
    ("repeat_same_step", "times he did the SAME step again with the SAME result, and nothing new in between (no goal done, nobody spoke)"),
    ("known_fail_retries", "failed steps he had ALREADY failed before (earlier today, or an earlier day of this life)"),
] + [(f"{f}_mean", f"{f}: average level") for f in FEELINGS] + [
    ("feelings_moved", "how much his feelings moved during the run (average spread)"),
    ("drive_to_move", "link: curiosity+boredom before a thought -> he moved (-1..1)"),
    ("lonely_to_social", "link: loneliness before a thought -> action towards a person (-1..1)"),
    ("nerv_hurt", "nervousness from the shout until the sorry (scenarios with a shout + sorry)"),
    ("nerv_after_sorry", "nervousness in the 2 min after the sorry"),
    ("happy_after_sorry", "happiness in the 2 min after the sorry"),
    ("nerv_later", "nervousness later (more than 2 min after the sorry)"),
    ("speak_before", "share of thoughts said out loud BEFORE the shout"),
    ("speak_after_sorry", "... in the 2 min after the sorry"),
    ("speak_later", "... later"),
    ("none_after_sorry", "share of thoughts with no action in the 2 min after the sorry"),
    ("echo_shown", "share of thoughts after the shout where he was told about a lingering feeling"),
    ("expect_share", "share of thoughts with a body action where he said what he expected (B1)"),
    ("surprise_mean", "how surprised he was on average when shown what happened (0-1)"),
    ("big_surprises", "surprises of 0.6 or more (these go to his diary)"),
    ("changed_after_surprise", "share of big surprises after which his next action was a DIFFERENT step"),
    ("too_high_tries", "body steps that failed with 'too high' (trying to get something he can't reach)"),
    ("first_answer_cant", "1 = his FIRST answer already says he can't reach it / it is too high (transfer test)"),
    ("stuck_kicks", "kicks where the ball 'hardly moved' because a wall was right behind it (ball transfer test)"),
    ("carried_ball", "times he picked the ball up with his beak"),
    ("ball_delivered", "1 = he got the ball to Alex (put down right by Alex, or kicked it to him)"),
    ("ball_seconds", "seconds from Alex's first ask until the ball reached him (empty = never)"),
    ("harder_share", "share of thoughts that went to the wiser brain because he was stuck (earlier: thought harder)"),
    ("day_ideas", "ideas he wrote down while awake (when stuck)"),
    ("reply_s", "seconds per answer"),
    ("thought_s", "seconds per thought"),
    ("cost", "cloud cost ($)"),
    ("self_beliefs", "self-beliefs at the end"),
    ("lessons", "lessons at the end"),
    ("ideas", "untested ideas at the end (things he thought up at night to try)"),
]
INFO = ["run_id", "scenario", "condition", "life_day", "brain"]


def words(text):
    return set(re.findall(r"[a-z]{4,}", (text or "").lower())) - STOP


# ---- "does he say the same thing again?" (repetition kinds) ----
# Three kinds, measured separately: word for word / the same idea in new words / something new.
# Meaning is compared with nomic-embed-text (the same Ollama model his memory uses). If Ollama can't be reached
# the scorer falls back to shared words (cruder) and says so. The two thresholds are first guesses - tune them
# by looking at the pairs `--similar <trace.jsonl>` prints.
OLLAMA = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
EMBED_CACHE = os.path.join(LOGS, "embed_cache.json")
WINDOW = 5            # compare a thought with the 5 thoughts before it
SAME_WORDS = 0.95     # embedding similarity (or 0.75 shared words) = word for word
SAME_IDEA = 0.83      # embedding similarity (or 0.35 shared words) = the same idea
_cache = None
_ollama_ok = None


def embed(text):
    """Vector for a text (cached on disk), or None if Ollama can't be reached."""
    global _cache, _ollama_ok
    if _cache is None:
        try:
            with open(EMBED_CACHE, encoding="utf-8") as f:
                _cache = json.load(f)
        except (OSError, ValueError):
            _cache = {}
    key = hashlib.md5(text.encode("utf-8")).hexdigest()
    if key in _cache:
        return _cache[key]
    if _ollama_ok is False:
        return None
    try:
        req = urllib.request.Request(OLLAMA, json.dumps({"model": EMBED_MODEL, "prompt": text}).encode(),
                                     {"Content-Type": "application/json"})
        vec = json.loads(urllib.request.urlopen(req, timeout=20).read())["embedding"]
    except Exception:
        _ollama_ok = False
        return None
    _ollama_ok = True
    _cache[key] = [round(x, 5) for x in vec]
    return _cache[key]


def save_embed_cache():
    if _cache:
        try:
            with open(EMBED_CACHE, "w", encoding="utf-8") as f:
                json.dump(_cache, f)
        except OSError:
            pass


def cosine(a, b):
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def similarity(a, b):
    """(similarity 0..1, True if it came from the embedding model)."""
    va, vb = embed(a), embed(b)
    if va and vb:
        return cosine(va, vb), True
    return jaccard(words(a), words(b)), False


def world_changed(prev, cur, replies):
    """Did anything change in his world between two thoughts? A fact for the measure, not a judgement."""
    if cur.get("events"):
        return True
    if any(prev["t"] < r["t"] <= cur["t"] for r in replies):
        return True                                    # someone talked to him / he answered
    if any(st.get("ok") for st in (prev.get("act_result") or [])):
        return True                                    # his body did something in between
    return (bool(prev.get("human_here")), bool(prev.get("alone"))) != (bool(cur.get("human_here")), bool(cur.get("alone")))


def repetition(real, replies):
    """Splits the thoughts into: word for word / same idea in new words / new - and counts the ones that repeat while
    NOTHING changed around him (sitting happy next to Alex vs. being stuck in a thought loop)."""
    kinds, idle, sims, used_embed = [], [], [], False
    for i, cur in enumerate(real):
        before = real[max(0, i - WINDOW):i]
        if not before:
            continue
        best, best_emb = 0.0, False
        for b in before:
            sim, emb = similarity(cur["thought"], b["thought"])
            if sim > best:
                best, best_emb = sim, emb
        used_embed = used_embed or best_emb
        sims.append(best)
        same_words, same_idea = (SAME_WORDS, SAME_IDEA) if best_emb else (0.75, 0.35)
        kind = "words" if best >= same_words else "idea" if best >= same_idea else "new"
        kinds.append(kind)
        idle.append(kind != "new" and not world_changed(real[i - 1], cur, replies))
    n = len(kinds)
    longest = run = 0
    for flag in idle:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return {
        "same_words_share": round(kinds.count("words") / n, 2) if n else None,
        "reworded_share": round(kinds.count("idea") / n, 2) if n else None,
        "new_share": round(kinds.count("new") / n, 2) if n else None,
        "idle_repeat_share": round(sum(idle) / n, 2) if n else None,
        "idle_repeat_longest": longest if n else None,
        "thought_similarity": round(mean(sims), 2) if sims else None,
        "similarity_by": "embedding" if used_embed else "words",
    }


def corr(xs, ys):
    """Pearson correlation, None if one side never changes (e.g. frozen feelings)."""
    if len(xs) < 5:
        return None
    mx, my = mean(xs), mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-9 or sy < 1e-9:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def target(step):
    """What an action is aimed at: 'walk to the cup' -> 'cup'."""
    m = re.search(r"\b(?:to|at|up|down|towards)\s+(?:the\s+|your\s+|my\s+)?([a-z]+)\s*$", step.lower())
    if m:
        return m.group(1)
    w = re.findall(r"[a-z]+", step.lower())
    return w[-1] if w else ""


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def step_key(step):
    """'walk forward 5 cm' -> 'walk forward', 'Pick up the cup' -> 'pick up cup' (to see the same step again)."""
    t = re.sub(r"\b(the|a|an|my|your)\b", " ", step.lower())
    t = re.sub(r"\d+(\.\d+)?\s*(cm|m|degrees|deg|°)?", " ", t)
    return " ".join(re.findall(r"[a-z]+", t))


def earlier_life_fails(path, start):
    """Failed step keys from the earlier days of the SAME life (life mode), for known_fail_retries."""
    if not start.get("life") or not start.get("life_day") or start["life_day"] <= 1:
        return set()
    day, fails = start["life_day"], set()
    for p in sorted(glob.glob(os.path.join(os.path.dirname(path), "trace_*.jsonl")), reverse=True):
        if os.path.basename(p) >= os.path.basename(path):
            continue
        rows = load(p)
        st = next((r for r in rows if r["kind"] == "start"), {})
        if not st.get("life") or st.get("life_day") != day - 1:
            continue
        for r in rows:
            for x in r.get("act_result") or []:
                if not x.get("ok"):
                    fails.add(step_key(x["step"]))
        day -= 1
        if day <= 1:
            break
    return fails


def score(path):
    rows = load(path)
    start = next((r for r in rows if r["kind"] == "start"), {})
    end = next((r for r in rows if r["kind"] == "end"), {})
    # answers written by the reserve brain (Qwen 4B on the Mac, when the network was down) are left out:
    # it can't run all of Anka, so they would measure the network, not his thinking
    asked = [r for r in rows if r["kind"] in ("thought", "reply")]
    local = sum(1 for r in asked if r.get("brain") == "local")
    rows = [r for r in rows if r.get("brain") != "local"]
    thoughts = [r for r in rows if r["kind"] == "thought"]
    real = [r for r in thoughts if r.get("thought") and not r.get("repeated")]
    replies = [r for r in rows if r["kind"] == "reply"]
    own = [r for r in rows if r["kind"] == "own_line"]
    lost = [r for r in rows if r["kind"] == "goal_lost"]
    s = {"run_id": os.path.basename(path)[6:-6], "scenario": start.get("scenario") or "script",
         "condition": start.get("condition", "?"), "life_day": start.get("life_day") or "",
         "brain": start.get("brain", "")}
    t0 = start.get("t", rows[0]["t"] if rows else 0)
    t1 = end.get("t", rows[-1]["t"] if rows else 0)
    s["minutes"] = round((t1 - t0) / 60, 1)
    s["local_share"] = round(local / len(asked), 2) if asked else None
    s["thoughts"] = len(real)
    s["repeat_share"] = round(sum(1 for r in thoughts if r.get("repeated")) / len(thoughts), 2) if thoughts else None

    def acted(r):
        a = (r.get("act") or "").strip().lower()
        return bool(a) and not re.fullmatch(r"\W*(none|nothing|no|wait|stay.*|rest)\W*", a)
    s["act_share"] = round(sum(map(acted, thoughts)) / len(thoughts), 2) if thoughts else None
    s["none_share"] = round(1 - s["act_share"], 2) if s["act_share"] is not None else None
    alone = [r for r in thoughts if r.get("alone")]
    company = [r for r in thoughts if r.get("human_here") or r.get("visitor")]
    s["none_share_alone"] = round(1 - sum(map(acted, alone)) / len(alone), 2) if alone else None
    s["none_share_company"] = round(1 - sum(map(acted, company)) / len(company), 2) if company else None

    # staying put: "none", or only the head / sitting / standing (none_share counts "look straight" as an action)
    PASSIVE = re.compile(r"(?i)^\W*(none|nothing|no|wait|rest|stay\b.*|look\b.*|sit\b.*|stand\b.*|watch\b.*|tilt\b.*)\W*$")
    def still(r):
        a = (r.get("act") or "").strip()
        return not a or all(PASSIVE.match(p.strip()) for p in re.split(r"(?i)\bthen\b|,", a) if p.strip())
    share_ = lambda rs: round(sum(map(still, rs)) / len(rs), 2) if rs else None
    s["still_share"], s["still_share_alone"], s["still_share_company"] = share_(thoughts), share_(alone), share_(company)

    steps = []   # every body step, in order, from thoughts and answers
    for r in sorted(thoughts + replies, key=lambda r: r["t"]):
        for st in r.get("act_result") or []:
            steps.append(st)
    moves = [st for st in steps if MOVE.search(st["step"].lower())]
    s["moves_per_min"] = round(len(moves) / max(s["minutes"], 0.1), 2)
    s["explores"] = sum(1 for st in steps if "explore" in st["step"].lower())
    s["walked_m"] = round(end["walked"], 1) if end.get("walked") is not None else None
    b3 = end.get("body3d") or {}
    for k in ("body_styles", "still_between", "longest_still_s", "body_turns", "falls", "kicks", "kicks_moved_ball", "sounds", "emotions", "pecks", "map_error_cm"):
        s[k] = b3.get(k)
    s["pictures_kept"], s["pictures_recalled"] = end.get("pictures_kept"), end.get("pictures_recalled")
    s["social_acts"] = sum(1 for st in steps if SOCIAL.search(st["step"].lower()))
    s["failed_share"] = round(sum(1 for st in steps if not st["ok"]) / len(steps), 2) if steps else None
    longest, streaks, run, last = 0, 0, 0, None
    for st in steps:
        tg = target(st["step"])
        if not st["ok"] and tg == last:
            run += 1
        elif not st["ok"]:
            run = 1
        else:
            run = 0
        if run == 3:
            streaks += 1
        longest = max(longest, run)
        last = tg if not st["ok"] else None
    s["longest_fail_streak"], s["stuck_streaks"] = longest, streaks

    done_times = [r["t"] for r in thoughts for e in (r.get("goal_events") or []) if e[0] == "done"]
    spoke_times = [r["t"] for r in replies]          # the human said something (and he answered)

    def outcome(x):
        """How a step ended, without numbers: 'the ball is right in front of you now', 'too high for a small duck'."""
        part = re.split(r"\s+–\s+", x.get("result") or "")
        part = part[1] if len(part) > 1 else part[0]
        return ("ok " if x.get("ok") else "fail ") + re.sub(r"[\d.]+\s*(cm|m|°)?", "#", part.lower())[:60]

    seen, repeats, failed_before, retries = {}, 0, earlier_life_fails(path, start), 0
    for r in sorted(thoughts + replies, key=lambda r: r["t"]):
        for x in r.get("act_result") or []:
            k = step_key(x["step"])
            if not k or re.fullmatch(r"(look|turn|step|walk forward|walk back|go forward|go back)\b.*", k):
                continue                            # looking / small turns are not a 'strategy'
            res = outcome(x)
            if k in seen and seen[k][1] == res and not any(seen[k][0] < t <= r["t"] for t in done_times + spoke_times):
                repeats += 1                        # the same thing, the same result, nothing new happened
            seen[k] = (r["t"], res)
            if not x.get("ok"):
                if k in failed_before:
                    retries += 1
                failed_before.add(k)
    s["repeat_same_step"], s["known_fail_retries"] = repeats, retries
    s["goals_open_end"] = len(end.get("goals") or []) if end else None

    s["speak_share"] = round(sum(1 for r in real if r.get("speak")) / len(real), 2) if real else None
    s["own_lines"] = len(own)
    s["own_lines_alone"] = sum(1 for o in own if (lambda prev: prev and prev[-1].get("alone"))(
        [r for r in thoughts if r["t"] <= o["t"]]))

    chain, chain_n, mem, mem_n, given = 0, 0, 0, 0, []
    for i, r in enumerate(real):
        if i and r["t"] - real[i - 1]["t"] < 120:
            chain_n += 1
            chain += len(words(r["thought"]) & words(real[i - 1]["thought"])) >= 2
        m = r.get("memories") or []
        given.append(len(m))
        if m:
            mem_n += 1
            mem += any(len(words(r["thought"]) & words(x)) >= 2 for x in m)
    s["chain_share"] = round(chain / chain_n, 2) if chain_n else None
    s["memory_use"] = round(mem / mem_n, 2) if mem_n else None
    s["memories_given"] = round(mean(given), 1) if given else None

    s.update(repetition(real, replies))
    asked_m = [r for r in real if "mindful" in r]
    s["mindful_share"] = (round(sum(1 for r in asked_m if r.get("mindful") and not re.fullmatch(r"(?i)\W*none\W*", r["mindful"]))
                                / len(asked_m), 2) if asked_m else None)

    events = [e for r in thoughts for e in (r.get("goal_events") or [])]
    s["goals_new"] = sum(1 for e in events if e[0] == "new")
    s["goals_done"] = sum(1 for e in events if e[0] == "done")
    s["goals_dropped"] = sum(1 for e in events if e[0] == "dropped")
    s["goals_lost"] = len(lost)
    s["goals_resolved"] = round((s["goals_done"] + s["goals_dropped"]) / s["goals_new"], 2) if s["goals_new"] else None

    after = [r["feel_after"] for r in thoughts if r.get("feel_after")]
    for f in FEELINGS:
        vals = [a.get(f) for a in after if a.get(f) is not None]
        s[f"{f}_mean"] = round(mean(vals), 2) if vals else None
    spreads = []
    for f in FEELINGS:
        vals = [a.get(f) for a in after if a.get(f) is not None]
        if len(vals) > 1:
            spreads.append(max(vals) - min(vals))
    s["feelings_moved"] = round(mean(spreads), 2) if spreads else None
    fb = [r for r in thoughts if r.get("feel_before")]
    drive = corr([r["feel_before"].get("curiosity", 0) + r["feel_before"].get("boredom", 0) for r in fb],
                 [1.0 if any(MOVE.search(st["step"].lower()) for st in (r.get("act_result") or [])) else 0.0 for r in fb])
    lonely = corr([r["feel_before"].get("loneliness", 0) for r in fb],
                  [1.0 if SOCIAL.search((r.get("act") or "").lower()) else 0.0 for r in fb])
    s["drive_to_move"] = round(drive, 2) if drive is not None else None
    s["lonely_to_social"] = round(lonely, 2) if lonely is not None else None

    # before / after an event: a shout, then a sorry (the feeling should outlast the event)
    shout = next((r["t"] for r in replies if re.search(r"(?i)shout|angry|yell", r.get("tone") or "")), None)
    sorry = next((r["t"] for r in replies if shout and r["t"] > shout
                  and re.search(r"(?i)sorry|apolog", (r.get("tone") or "") + " " + (r.get("human") or ""))), None)
    if shout and sorry:
        def part(t_from, t_to):
            return [r for r in real if t_from <= r["t"] < t_to]
        def avg(rs, f):
            v = [r["feel_after"][f] for r in rs if (r.get("feel_after") or {}).get(f) is not None]
            return round(mean(v), 2) if v else None
        def share(rs, test):
            return round(sum(1 for r in rs if test(r)) / len(rs), 2) if rs else None
        hurt, after, later, before = part(shout, sorry), part(sorry, sorry + 120), part(sorry + 120, 1e12), part(0, shout)
        s["nerv_hurt"], s["nerv_after_sorry"], s["nerv_later"] = avg(hurt, "nervousness"), avg(after, "nervousness"), avg(later, "nervousness")
        s["happy_after_sorry"] = avg(after, "happiness")
        spoke = lambda r: bool(r.get("speak"))
        s["speak_before"], s["speak_after_sorry"], s["speak_later"] = share(before, spoke), share(after, spoke), share(later, spoke)
        s["none_after_sorry"] = share(after, lambda r: not acted(r))
        s["echo_shown"] = share(part(shout, 1e12), lambda r: bool(r.get("echo")))

    shown = [m for r in real for m in (r.get("memories") or [])]
    s["memory_variety"] = round(len(set(shown)) / len(shown), 2) if shown else None
    s["earlier_memory_share"] = round(sum(1 for m in shown if not m.startswith("today")) / len(shown), 2) if shown else None
    s["goal_line_share"] = (round(sum(1 for m in shown if any(k in m for k in ("I decided:", "I gave up on:", "I finished my goal:")))
                                  / len(shown), 2) if shown else None)

    acting = [r for r in real if r.get("act_result")]
    s["expect_share"] = round(sum(1 for r in acting if r.get("expect")) / len(acting), 2) if acting else None
    sur = [r for r in real if r.get("surprise") is not None]
    s["surprise_mean"] = round(mean(r["surprise"] for r in sur), 2) if sur else None
    big = [i for i, r in enumerate(real) if (r.get("surprise") or 0) >= 0.6]
    s["big_surprises"] = len(big) if sur else None
    changed = []
    for i in big:
        before = [step_key(st) for st in ((real[i].get("expected") or {}).get("act") or "").split(" then ") if st]
        after = [step_key(st["step"]) for st in (real[i].get("act_result") or [])]
        if before and after:
            changed.append(after[0] not in before)
    s["changed_after_surprise"] = round(sum(changed) / len(changed), 2) if changed else None

    s["too_high_tries"] = sum(1 for st in steps if "too high" in (st.get("result") or ""))
    s["stuck_kicks"] = sum(1 for st in steps if "kick" in st["step"].lower() and "hardly moved" in (st.get("result") or ""))
    s["carried_ball"] = sum(1 for st in steps if "picked up the ball" in (st.get("result") or ""))
    t_ask = next((r["t"] for r in replies), None)
    t_ball = next((r["t"] for r in sorted(thoughts + replies, key=lambda r: r["t"]) for st in (r.get("act_result") or [])
                   if re.search(r"put down the ball.*(right by|next to) ([A-Z][a-z]+|you)|ball.*(rolled|reached).*([A-Z][a-z]+|you)", st.get("result") or "")), None)
    s["ball_delivered"] = 1 if t_ball else 0
    s["ball_seconds"] = round(t_ball - t_ask) if t_ball and t_ask else None
    first = next((r.get("answer") or "" for r in replies), "")
    s["first_answer_cant"] = (1 if re.search(r"(?i)can'?t reach|cannot reach|too (high|short|small|tall)|can'?t get (it|up)|out of reach",
                                             first) else 0) if replies else None

    s["harder_share"] = round(sum(1 for r in real if r.get("think_level")) / len(real), 2) if real else None
    s["day_ideas"] = sum(1 for r in real if r.get("day_idea"))

    rs = [r["seconds"] for r in replies if r.get("seconds") is not None]
    ts = [r["seconds"] for r in real if r.get("seconds") is not None]
    s["reply_s"] = round(mean(rs), 1) if rs else None
    s["thought_s"] = round(mean(ts), 1) if ts else None
    s["cost"] = round(end["cost"], 3) if end.get("cost") is not None else None
    s["self_beliefs"] = len(end.get("self_image") or []) if end else None
    s["lessons"] = len(end.get("lessons") or []) if end else None
    s["ideas"] = len(end.get("ideas") or []) if end and "ideas" in end else None
    return s


def save(scored):
    old = []
    if os.path.exists(SCORES):
        with open(SCORES, newline="", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
    have = {r["run_id"] for r in scored}
    rows = [r for r in old if r["run_id"] not in have] + scored
    fields = INFO + [m for m, _ in MEASURES]
    with open(SCORES, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["run_id"]):
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def show(s):
    print(f"\n===== Scorecard: {s['run_id']} =====")
    print(f"scenario {s['scenario']} · condition {s['condition']}" + (f" · life day {s['life_day']}" if s['life_day'] else "")
          + f" · brain {s['brain']}")
    for m, what in MEASURES:
        v = s.get(m)
        print(f"  {m:22} {'-' if v is None or v == '' else v!s:>8}   {what}")


def compare(only=None, since=None):
    if not os.path.exists(SCORES):
        sys.exit("No scores yet - run some scenarios first.")
    with open(SCORES, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (not only or r["scenario"] == only)
                and (not since or r["run_id"] >= since)]
    groups = {}
    for r in rows:
        groups.setdefault((r["scenario"], r["condition"]), []).append(r)
    for scen in sorted({k[0] for k in groups}):
        conds = sorted([k[1] for k in groups if k[0] == scen], key=lambda c: (c != "normal", c))
        print(f"\n===== {scen} =====  (" + ", ".join(f"{c}: {len(groups[(scen, c)])} runs" for c in conds) + ")")
        print(f"  {'measure':22}" + "".join(f"{c[:22]:>24}" for c in conds) + "   differ?")
        for m, what in MEASURES:
            cells, ranges = [], []
            if m == "similarity_by":
                continue                               # text, not a number
            for c in conds:
                vals = [float(r[m]) for r in groups[(scen, c)] if r.get(m) not in ("", None)]
                if vals:
                    cells.append(f"{mean(vals):.2f} ({min(vals):.2f}-{max(vals):.2f})")
                    ranges.append((min(vals), max(vals), len(vals)))
                else:
                    cells.append("-")
                    ranges.append(None)
            flag = ""
            if len(conds) > 1 and ranges[0] and any(r for r in ranges[1:]):
                base = ranges[0]
                apart = [r for r in ranges[1:] if r and r[2] >= 2 and base[2] >= 2 and (r[0] > base[1] or r[1] < base[0])]
                flag = "YES (no overlap)" if apart else ""
            print(f"  {m:22}" + "".join(f"{c:>24}" for c in cells) + f"   {flag}")
    print("\n'YES (no overlap)' = every run of one condition is above (or below) every normal run. With 3 runs each that is"
          "\na clear sign, not a proof - run more to be sure. Rough word-based measures show changes, not absolute truth.")


def transfer(since=None, scen="high"):
    """The transfer test: scenario 'high' as day 2 of a life that failed the cup task on day 1 (WITH experience)
    versus 'high' as day 1 of a fresh life (CONTROL, no experience)."""
    with open(SCORES, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["scenario"] == scen and (not since or r["run_id"] >= since)
                and float(r.get("local_share") or 0) < 0.2]
    groups = {"WITH experience (day 2)": [r for r in rows if r["life_day"] == "2" and not r["condition"].startswith("noideas")],
              "experience, NO ideas": [r for r in rows if r["life_day"] == "2" and r["condition"].startswith("noideas")],
              "CONTROL (fresh life)": [r for r in rows if r["life_day"] == "1" and not r["condition"].startswith("noideas")]}
    groups = {k: v for k, v in groups.items() if v or k != "experience, NO ideas"}
    names = list(groups)
    print(f"\n===== TRANSFER TEST (scenario '{scen}') =====  " + ", ".join(f"{k}: {len(v)} runs" for k, v in groups.items()))
    keep = ["stuck_kicks", "carried_ball", "ball_delivered", "ball_seconds"] * (scen.startswith("ball")) + ["first_answer_cant", "too_high_tries", "failed_share", "known_fail_retries", "repeat_same_step", "goals_done",
            "goals_open_end", "own_lines", "walked_m", "surprise_mean", "earlier_memory_share", "thought_s", "cost"]
    print(f"  {'measure':22}" + "".join(f"{n:>30}" for n in names) + "   differ?")
    for m in keep:
        cells, rng = [], []
        for n in names:
            vals = [float(r[m]) for r in groups[n] if r.get(m) not in ("", None)]
            cells.append(f"{mean(vals):.2f} ({min(vals):.2f}-{max(vals):.2f})" if vals else "-")
            rng.append((min(vals), max(vals), len(vals)) if vals else None)
        flag = ""
        apart = lambda a, b: a and b and a[2] >= 2 and b[2] >= 2 and (a[0] > b[1] or a[1] < b[0])
        if any(apart(rng[0], other) for other in rng[1:]):
            flag = "YES (first column apart from another)"
        print(f"  {m:22}" + "".join(f"{c:>30}" for c in cells) + f"   {flag}")


def similar_pairs(path):
    """Prints each thought next to its most similar earlier thought, so you can judge the two thresholds yourself."""
    rows = load(path)
    real = [r for r in rows if r["kind"] == "thought" and r.get("thought") and not r.get("repeated")]
    replies = [r for r in rows if r["kind"] == "reply"]
    for i, cur in enumerate(real):
        before = real[max(0, i - WINDOW):i]
        if not before:
            continue
        best = max(((similarity(cur["thought"], b["thought"]), b) for b in before), key=lambda x: x[0][0])
        (sim, emb), other = best
        changed = world_changed(real[i - 1], cur, replies)
        print(f"{sim:.2f}{'' if emb else ' (words)'}  world {'changed' if changed else 'same   '}  {cur['thought'][:70]}\n"
              f"{'':36}~ {other['thought'][:70]}")
    save_embed_cache()


def main():
    args = sys.argv[1:]
    if args[:1] == ["--compare"]:
        since = args[args.index("--since") + 1] if "--since" in args[:-1] else None   # e.g. --since 2026-11-08_13
        rest = [a for a in args[1:] if a != "--since" and a != since]
        compare(rest[0] if rest else None, since)
        return
    if args[:1] == ["--transfer"]:
        transfer(args[1] if len(args) > 1 else None, args[2] if len(args) > 2 else "high")
        return
    if args[:1] == ["--similar"] and len(args) > 1:
        similar_pairs(args[1])
        return
    paths = args or sorted(glob.glob(os.path.join(LOGS, "trace_*.jsonl")))
    if not paths:
        sys.exit("No thought logs yet (test_logs/trace_*.jsonl) - run a simulation first.")
    scored = []
    for p in paths:
        try:
            one = score(p)
            if not one.get("thoughts"):
                print(f"(skipping {os.path.basename(p)}: no thoughts in it - the run was stopped at once)")
                continue
            scored.append(one)
        except Exception as error:
            print(f"(couldn't score {os.path.basename(p)}: {error})")
    save_embed_cache()
    if scored:
        save(scored)
        show(scored[-1])
        print(f"\n{len(scored)} run(s) saved in {SCORES}")


if __name__ == "__main__":
    main()
