"""Anka's thought log ("trace"): one JSON line per thought, answer, move, goal change and night, written while a
simulation runs - so his thinking can be MEASURED afterwards (tools/score_run.py).
Only facts are written here. Nothing in this file changes what he thinks, feels or does, and he never reads it."""
import json
import os
import threading
import time

from settings import TRACE, TRACE_FILE

_lock = threading.Lock()
_broken = [False]


def trace(kind, **data):
    """Adds one line to the thought log: {"t": seconds, "kind": kind, ...data}. Never raises."""
    if not TRACE or _broken[0]:
        return
    row = {"t": round(time.time(), 2), "kind": kind}
    if kind in ("thought", "reply", "own_line", "thought_dropped"):
        try:   # which brain wrote it – the reserve brain (Qwen 4B) can't run all of Anka, so the scorer leaves its answers out
            from anka.brain import answered
            row["brain"] = getattr(answered, "by", None)
        except Exception:
            pass
    row.update(data)
    try:
        with _lock:
            os.makedirs(os.path.dirname(TRACE_FILE), exist_ok=True)
            with open(TRACE_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except Exception as error:   # the log must never stop Anka
        _broken[0] = True
        print(f"📈 (couldn't write the thought log: {error})")
