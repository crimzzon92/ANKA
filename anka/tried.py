"""B2 – "what I tried": a short history of what his body did lately and what came of it, grouped by what it was
aimed at (the sofa, the cup ...). Only FACTS from his body – Python never says "this doesn't work" or "try
something else". What the history means (keep trying, a new plan, give up the goal) is his own choice.
See claude/anka-thinking-plan.md (B2). Written 2026-11-06 after day 3 of the cup test: he walked to the sofa twice
"to climb up to the cup" and never noticed it brought him no closer."""
import re
import threading
import time

WINDOW = 15 * 60      # remember tries from the last 15 minutes
MAX_TARGETS = 5       # at most this many things in the list (the most recent ones)
MAX_PER_TARGET = 4    # at most this many tries shown per thing
_lock = threading.Lock()
_tries = []           # {"t", "step", "target", "result", "ok"}
_day = []             # the same, for the whole day (since he woke up) – for the night's look back

_NO_TARGET = re.compile(r"(?i)^\s*(look|turn|step|walk forward|walk back|go forward|go back|move forward|"
                        r"move back|stop|wait|stay|sit|stand|none|nothing)\b")


def _target(step, result):
    """What a step was aimed at: 'walk to the sofa' -> 'the sofa'. Steps with no object (walk forward 10 cm)
    only count when they failed, and then the thing that stopped him is the target."""
    m = re.search(r"(?i)\b(?:to|at|up|down|towards|toward|on|into|in|kick|push|nudge|grab|pick up|touch|explore)"
                  r"\s+(?:the\s+|your\s+|my\s+|a\s+)?([a-z][a-z' ]{1,30}?)\s*$", step.strip(" ."))
    if m and not re.match(r"(?i)\s*(walk|go|move|step) (forward|back)", step):
        thing = m.group(1).strip()
        return thing if thing[:1].isupper() else "the " + thing     # 'Alex', not 'the Alex'
    hit = re.search(r"(?i)\b(?:the|a)\s+([a-z]+)\s+is about \d+\s*cm in front of you|walked into (?:the\s+)?([a-z]+)",
                    result or "")
    if hit:
        return "the " + (hit.group(1) or hit.group(2))
    return None


def _short(result, ok, target=""):
    """The part of the body's answer that says how it went (the part about the target, if there is one)."""
    parts = [p.strip() for p in re.split(r"\s+–\s+", result or "") if p.strip()]
    if not parts:
        return "?"
    noun = target.replace("the ", "").split()[-1] if target else ""
    about = [p for p in parts[1:] if noun and noun in p.lower()]
    if not ok:   # keep why it failed; skip a first part that only says how far he walked or turned
        moved = re.match(r"(?i)(walked|turned|stepped|went)\b", parts[0])
        text = " – ".join(parts[1:] if moved and len(parts) > 1 else parts)
    else:
        text = about[0] if about else parts[1] if len(parts) > 1 else parts[0]
    text = re.sub(r"\s*\(.*?\)", "", text)
    return text if len(text) <= 80 else text[:77].rsplit(" ", 1)[0] + "…"


def record(log):
    """Called after every action (from thoughts AND answers) with run_act's step list."""
    now = time.time()
    with _lock:
        for x in log or []:
            target = _target(x["step"], x["result"])
            if target is None or (x["ok"] and _NO_TARGET.match(x["step"])):
                continue
            entry = {"t": now, "step": x["step"].strip(" ."), "target": target,
                     "result": _short(x["result"], x["ok"], target), "ok": bool(x["ok"])}
            _tries.append(entry)
            _day.append(entry)
        _tries[:] = [t for t in _tries if now - t["t"] < WINDOW]


def _ago(seconds):
    m = int(seconds // 60)
    return "just now" if m < 1 else f"{m} min ago"


def tried_words(goals=()):
    """The history as text for his prompt, or '' if he hasn't tried anything lately."""
    now = time.time()
    with _lock:
        tries = [t for t in _tries if now - t["t"] < WINDOW]
    if not tries:
        return ""
    order = []
    for t in reversed(tries):                     # the most recently tried things first
        if t["target"] not in order:
            order.append(t["target"])
    lines = []
    for target in order[:MAX_TARGETS]:
        mine = [t for t in tries if t["target"] == target]
        shown = mine[-MAX_PER_TARGET:]
        lines.append(f"- {target}: {len(mine)} {'try' if len(mine) == 1 else 'tries'} – "
                     + "; ".join(f"'{t['step']}' ({_ago(now - t['t'])}): {t['result']}" for t in shown))
    goal_lines = []
    for g in goals:
        if "t" in g:
            since = [t for t in tries if t["t"] >= g["t"]]
            goal_lines.append(f"- \"{g['text']}\": set {_ago(now - g['t'])}, still open; since then your body "
                              f"tried {len(since)} thing{'s' if len(since) != 1 else ''}")
    return ("[What you tried lately (facts from your body, the last 15 minutes):\n" + "\n".join(lines)
            + ("\nYour open goals:\n" + "\n".join(goal_lines) if goal_lines else "") + "]")


def day_words():
    """The whole day's tries, grouped by what they were aimed at, with how often each result came – facts for the
    night, so his self-beliefs and lessons come from what his body REALLY managed, not from what he hoped
    (2026-11-08: the night wrote "I look for things to climb when I want to reach high places" after a day where
    climbing never worked, and the next day he spent all day trying to climb)."""
    with _lock:
        day = list(_day)
    if not day:
        return ""
    lines = []
    for target in dict.fromkeys(t["target"] for t in day):
        mine = [t for t in day if t["target"] == target]
        steps, results = {}, {}
        for t in mine:
            steps[t["step"]] = steps.get(t["step"], 0) + 1
            key = ("" if t["ok"] else "FAILED: ") + t["result"]
            results[key] = results.get(key, 0) + 1
        lines.append(f"- {target}: " + ", ".join(f"'{k}'" + (f" ×{v}" if v > 1 else "") for k, v in steps.items())
                     + " → " + "; ".join(k + (f" (×{v})" if v > 1 else "") for k, v in results.items()))
    return "\n".join(lines[:15])


def clear_day():
    with _lock:
        _day.clear()

