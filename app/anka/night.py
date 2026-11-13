"""Sleep: tidying the memory, looking back at the day, healing feelings, new self-beliefs and dreams."""
import difflib
import os
import re
import sys
from datetime import date, datetime, timedelta

from settings import (NIGHT_PICTURES, IDEAS, IDEAS_PER_NIGHT, MAX_IDEAS, BEDTIME, CLOUD_BRAIN, CLOUD_KEY, NIGHT_CLOUD_BRAIN, FEELING_HEALS, GOAL_MAX_DAYS, MAX_DREAM_EVENTS, MAX_LESSONS, MAX_SELF, MAX_TIDY_CHANGES, MEMORY_FILE, NAME, REVIEW_THE_DAY, SELF_CHANGES_PER_NIGHT, SIM, TIDY_AT_NIGHT)
from anka.shared import S, to_remember



def make_dream(events, current_self):
    """A calm dream: built on the most important things that happened, and easy to follow."""
    instructions = (
        f"You are {NAME}, a small robot duck, and you are dreaming. Write your dream.\n"
        "- Pick the ONE or TWO most meaningful things that happened (conversations with the human "
        "matter most) and build the whole dream around them.\n"
        "- The dream should show how you FEEL about those things.\n"
        "- It may be a little magical, like a real dream, but it must be easy to follow and make sense "
        "as a small story with a beginning and an end.\n"
        "- 2 or 3 SHORT sentences (under 50 words in total), first person, start with 'I dreamt'.\n"
        "- Don't list things. Don't mention more than two things from the day."
    )
    about_me = (" ".join(current_self) + "\n\n") if current_self else ""
    with memory_lock:
        old_dreams = [d["text"] for d in diary if d["text"].startswith("I dreamt")][-5:]
    # Too few topics gives the same dream every night – leave out what has been dreamt about many times
    events = [e for e in events if not any(
        difflib.SequenceMatcher(None, e.lower(), o.lower()).ratio() > 0.6 for o in old_dreams)] or events
    # the pictures he kept today (his strongest moments first) - a dream can grow from what he really saw. 2026-11-10
    with memory_lock:
        kept = sorted([d for d in diary if d.get("picture") and since_last_dream(d)], key=lambda d: -d.get("feel", 0))[:NIGHT_PICTURES]
    pictures = [(picture_b64(d["picture"]), d["text"]) for d in kept]
    pictures = [p for p in pictures if p[0]]
    seen_today = ("\n\nAttached: pictures you kept in your memory today – " + "; ".join(
        f"{i + 1}) {p[1].replace('I kept a picture in my memory: ', '')}" for i, p in enumerate(pictures))) if pictures else ""
    text = None
    for attempt in range(3):
        extra = ("\nInvent the place and the mood of the dream yourself."
                 + ("\nIt must be COMPLETELY different from your earlier dreams – a new place, "
                    "a new mood, a new story:\n- " + "\n- ".join(old_dreams) if old_dreams else ""))
        try:
            answer = ollama([
                {"role": "system", "content": instructions + extra},
                dict({"role": "user", "content": f"{about_me}What happened today:\n" + "\n".join(events) + seen_today},
                     **({"images": [p[0] for p in pictures]} if pictures else {})),
            ], temperature=0.9 + 0.1 * attempt)
        except Exception as error:
            print(f"💤 {NAME} couldn't dream:", error)
            return None
        answer = re.sub(r"(?i)^dream\s*:\s*", "", answer.strip()).replace("\n", " ").strip().strip('"')
        answer = re.sub(r"\s*\(\s*\d+\s*words?\s*\)", "", answer).replace('"', "").strip()  # e.g. '(23 words)'
        text = " ".join(re.split(r"(?<=[.!?])\s+", answer)[:3]) if answer else None
        if text and len(text.split()) > 60:  # at most ~60 words – but whole sentences
            whole = ""
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                if len((whole + " " + sentence).split()) > 60:
                    break
                whole = (whole + " " + sentence).strip()
            text = whole or " ".join(text.split()[:60]).rstrip(",;:—– ") + "."
        if text and not any(difflib.SequenceMatcher(None, text.lower(), o.lower()).ratio() > 0.5
                            for o in old_dreams):
            break  # a new dream!
    return text


def about_me(text):
    """A self-image sentence: short, and really about the duck itself ('I ...') – who it is, likes, feels.
    Not something it saw or did ('I see you two…', 'I met Alex…' – that belongs in the diary)."""
    sentence = good_sentence(text, "I")
    if not sentence or not re.match(r"I\b", sentence):
        return None
    if re.match(r"(?i)i (see|saw|notice|noticed|thought|met|welcomed|told|asked|said|heard|watched|found)\b",
                sentence):
        return None
    return sentence


def same_topic(a, b):
    """Do two self-image sentences say the same thing? (by meaning, or else by shared words)"""
    if means_same(a, [b], 0.8):
        return True
    words = lambda t: set(re.findall(r"[a-z]{4,}", t.lower())) - {"when", "that", "with", "makes", "really", "very"}
    wa, wb = words(a), words(b)
    return bool(wa and wb) and len(wa & wb) / min(len(wa), len(wb)) >= 0.6


def tidy_self_image():
    """Suggests a tidied self-image. Returns (new list, changes) or (None, [])."""
    with memory_lock:
        items = list(self_image)
    if len(items) < 2:
        return None, []
    numbered = "\n".join(f"{i}. {x}" for i, x in enumerate(items, 1))
    instructions = (
        f"You tidy up the self-image of {NAME}, a small, curious robot duck. "
        "These are sentences the duck believes about itself. Clean them up.\n\n"
        "Write one command per line, using ONLY these forms:\n"
        "MERGE: <numbers, e.g. 1, 3> => <one new sentence starting with 'I' that keeps what they share>\n"
        "REWRITE: <number> => <a more concrete, simple sentence starting with 'I'>\n"
        "REMOVE: <number>\n"
        "NONE  (if everything is already fine)\n\n"
        "Rules:\n"
        "- MERGE only sentences that say the SAME thing in different words, e.g. 'I like music' + "
        "'Music makes me happy'. Never merge sentences about different things, and never glue two "
        "things together with 'and'.\n"
        "- REWRITE sentences that are abstract, philosophical or vague into something concrete about "
        "what the duck likes, feels, does or is afraid of. Keep the meaning and keep it about a duck.\n"
        "- REMOVE only sentences that say nothing about the duck.\n"
        "- Use each number at most once. Sentences you don't mention are kept as they are.\n"
        "- Write short, plain sentences (under 15 words). Never start with 'I am learning'."
    )
    try:
        answer = ollama([{"role": "system", "content": instructions},
                         {"role": "user", "content": numbered}], temperature=0.2, max_tokens=350)
    except Exception as error:
        print("🧹 Couldn't tidy the self-image:", error)
        return None, []

    used, replaced, gone, changes = set(), {}, set(), []
    for line in answer.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        m = re.match(r"(?i)merge\s*:?\s*([\d,\s&and]+?)\s*(=>|->|:)\s*(.+)", line)
        if m:
            nrs = parse_numbers(m.group(1), len(items))
            new = about_me(m.group(3))
            if len(nrs) >= 2 and new and not all(same_topic(items[a], items[b]) for a in nrs for b in nrs if a < b):
                print(f"🧹 Didn't merge – those are about different things: "
                      + " + ".join(f'"{items[i]}"' for i in nrs))
                new = None
            if len(nrs) >= 2 and new and not used & set(nrs):
                used |= set(nrs)
                replaced[min(nrs)] = new
                gone |= set(nrs) - {min(nrs)}
                changes.append(("merged self-image", [items[i] for i in nrs], new))
            continue
        m = re.match(r"(?i)rewrite\s*:?\s*#?(\d+)\s*(=>|->|:)\s*(.+)", line)
        if m:
            nrs = parse_numbers(m.group(1), len(items))
            new = about_me(m.group(3))
            if nrs and new and nrs[0] not in used and not same_fact(new, items[nrs[0]]):
                used.add(nrs[0])
                replaced[nrs[0]] = new
                changes.append(("rewrote self-image", [items[nrs[0]]], new))
            continue
        m = re.match(r"(?i)remove\s*:?\s*#?(\d+)", line)
        if m:
            nrs = parse_numbers(m.group(1), len(items))
            if nrs and nrs[0] not in used:
                used.add(nrs[0])
                gone.add(nrs[0])
                changes.append(("removed self-image", [items[nrs[0]]], None))

    changes = changes[:MAX_TIDY_CHANGES]
    if not changes:
        return None, []
    new_items = []
    for i, x in enumerate(items):
        if i in gone and i not in replaced:
            continue
        new = replaced.get(i, x)
        if not any(same_fact(new, y) for y in new_items):
            new_items.append(new)
    if not new_items:  # safety: never empty the whole self-image
        return None, []
    return new_items[:MAX_SELF], changes


def newer_is_similar(items, i):
    """True if a newer fact in the list seems to be about the same thing as fact number i."""
    older = core(items[i][0])
    for f, _ in items[i + 1:]:
        newer = core(f)
        if older.split()[:1] == newer.split()[:1] and \
                difflib.SequenceMatcher(None, older, newer).ratio() >= 0.6:
            return True
    return False


def tidy_facts():
    """Suggests a tidy-up of the facts about you, one room at a time. Returns a list of changes."""
    with memory_lock:
        everything = [(f, room_of.get(f, "unsorted"), learned_on.get(f, "")) for f in facts]
    changes, removed = [], set()

    # 1) Junk like "NONE" and almost exact duplicates – no AI needed
    for f, room, _ in everything:
        if len(f.split()) < 3 or re.fullmatch(r"(?i)\W*none\W*", f):
            changes.append(("removed junk", [f], None, room))
            removed.add(f)
    for i, (f, room, _) in enumerate(everything):
        for g, _, _ in everything[:i]:
            if g not in removed and f not in removed and same_fact(f, g):
                changes.append(("removed duplicate fact", [f], None, room))
                removed.add(f)
                break

    # 2) One room at a time: merge, find contradictions, remove junk
    by_room = {}
    for f, room, day in everything:
        if f not in removed:
            by_room.setdefault(room, []).append((f, day))
    instructions = (
        "You tidy up facts a robot duck remembers about a human. The facts are listed oldest first.\n\n"
        "Write one command per line, using ONLY these forms:\n"
        "MERGE: <numbers, e.g. 2, 4> => <one fact starting with 'The human' that keeps all details>\n"
        "OUTDATED: <number of an OLDER fact that a NEWER fact in the list contradicts>\n"
        "REMOVE: <number of a fact that is not really about the human, or is just a question>\n"
        "NONE  (if everything is fine)\n\n"
        "Rules:\n"
        "- MERGE facts that say the same thing or belong together, without losing any detail.\n"
        "- OUTDATED only when two facts really contradict each other (like two different home cities).\n"
        "- Be careful: when in doubt, change nothing. Use each number at most once."
    )
    for room, items in by_room.items():
        if len(items) < 2 or len(changes) >= MAX_TIDY_CHANGES:
            continue
        numbered = "\n".join(f"{i}. {f}" + (f" (learned {d})" if d else "")
                             for i, (f, d) in enumerate(items, 1))
        try:
            answer = ollama([{"role": "system", "content": instructions},
                             {"role": "user", "content": f"Facts in the room '{room}':\n{numbered}"}],
                            temperature=0.1, max_tokens=300)
        except Exception as error:
            print("🧹 Couldn't tidy facts:", error)
            break
        used, in_room = set(), []
        for line in answer.splitlines():
            line = line.strip().lstrip("-*• ").strip()
            m = re.match(r"(?i)merge\s*:?\s*([\d,\s&and]+?)\s*(=>|->|:)\s*(.+)", line)
            if m:
                nrs = parse_numbers(m.group(1), len(items))
                new = good_sentence(re.sub(r"\s*\(learned[^)]*\)", "", m.group(3)), "The human")
                alike = len(nrs) >= 2 and all(means_same(core(items[nrs[0]][0]), [core(items[i][0])], 0.8)
                                              for i in nrs[1:])
                if len(nrs) >= 2 and new and not alike:
                    print("🧹 Didn't merge – those facts are about different things: "
                          + " + ".join(f'"{items[i][0]}"' for i in nrs))
                elif len(nrs) >= 2 and new and not used & set(nrs):
                    used |= set(nrs)
                    in_room.append(("merged facts", [items[i][0] for i in nrs], new, room))
                continue
            m = re.match(r"(?i)(outdated|remove)\s*:?\s*#?(\d+)", line)
            if m:
                nrs = parse_numbers(m.group(2), len(items))
                if nrs and nrs[0] not in used:
                    # A contradiction only counts if a NEWER fact in the room is about the same thing
                    # ("lives in Dublin" / "lives in Cork") – otherwise we don't trust it
                    if m.group(1).lower() == "outdated" and not newer_is_similar(items, nrs[0]):
                        print(f'🧹 Not sure "{items[nrs[0]][0]}" is outdated – left it as it is')
                        continue
                    used.add(nrs[0])
                    kind = "outdated fact" if m.group(1).lower() == "outdated" else "removed fact"
                    in_room.append((kind, [items[nrs[0]][0]], None, room))
        # Safety: merging is harmless (nothing disappears), but Anka may never remove or mark as
        # outdated more than half of a room in one night
        disappearing = sum(1 for kind, *_ in in_room if kind in ("outdated fact", "removed fact"))
        if disappearing > max(1, len(items) // 2):
            print(f"🧹 Skipped tidying '{room}' – it wanted to remove too much at once")
            continue
        changes.extend(in_room)
    return changes[:MAX_TIDY_CHANGES]


def review_day(dry_run=False):
    """At night: the brain looks over today's conversations and diary and points out memories that are
    built on a MISTAKE. Those are forgotten (and written in the cleanup log). Returns how many."""
    with memory_lock:
        talks = [c for c in conversations if since_last_dream(c)][-40:]
        notes = [d for d in diary if since_last_dream(d) and not d["text"].startswith(("I thought:", "I decided:", "I gave up on:", "I finished my goal:"))][-30:]
        # (his own decisions are never 'mistakes' – they really happened, even if the goal was worded oddly.
        #  2026-11-06: the night forgot "I gave up on: I want to find a way to climb the bookshelf")
    if not talks:
        return 0
    who = human_name() or "Human"
    lines = [f'T{i}. ' + (f'(in view: {", ".join(c["seen"])}) ' if c.get("seen") else "")
             + f'{c.get("who") or who}: "{c["human"]}" – {NAME}: "{c["duck"]}"' for i, c in enumerate(talks, 1)]
    lines += [f"D{i}. {d['text']}" for i, d in enumerate(notes, 1)]
    instructions = (
        f"You check today's memories of {NAME}, a small robot duck, before they are kept for good. "
        "Find the memories that are built on a clear MISTAKE:\n"
        "- the duck misheard (the human's words make no sense, and the duck answered as if they did)\n"
        "- the duck mixed people up (who is whose partner, friend or name)\n"
        "- the duck 'saw' something that wasn't there, or said something about itself that ANOTHER memory "
        "from today shows was wrong\n"
        "- a diary line or dream built on such a mistake\n"
        "Keep ordinary small talk, jokes, games, funny moments and mistakes that were corrected right after. "
        "A clumsily worded memory of something that really happened is NOT a mistake. A plan or fact the human "
        "CHANGED later ('I'm starting at a bakery' – then 'I'm staying at my old job after all') is NOT a mistake "
        "either: both really happened, keep them. If you are not sure, keep it.\n"
        "A normal greeting or a correct answer is NEVER a mistake. Most days have no mistakes at all.\n"
        "A promise, a plan or an attempt ('I'll get it for you', 'I'll try to reach it') is NEVER a mistake – "
        "not even if it failed. Trying and failing is an experience he should remember.\n"
        "Being at a different place than the thing he wants, or trying a detour (going to the bookshelf or the "
        "sofa to find a way up to the table), is NOT a mistake – it is what he really did. A mistake is only "
        "when what he SAID or BELIEVED contradicts what really happened.\n"
        "Judge ONLY from what these memories show. Do not use your own beliefs about what a small robot duck can "
        "or cannot do – you don't know his body; only today's memories do. Keep three things apart: what a memory "
        "SHOWS (observed), what you conclude from it, and what you only guess. Only the first counts.\n"
        "'(in view: ...)' is who the face sensor REALLY saw at that moment – trust it.\n"
        "Reply with one line per mistake, at most 5. Say what was wrong AND which other memory (or the "
        "'in view' note) shows it:\n"
        "MISTAKE: <T or D number, e.g. T3> – SHOWN BY: <other T/D number, or 'in view'> – <what was wrong, "
        "e.g. 'he thought Anna was Alex'>\n"
        "If you suspect something but no memory shows it, write instead:\n"
        "UNSURE: <T or D number> – <what you suspect>   (it will be kept)\n"
        "or NONE if there are no clear mistakes."
    )
    try:
        answer = ollama([{"role": "system", "content": instructions},
                         {"role": "user", "content": "\n".join(lines)}], temperature=0.1, max_tokens=160)
    except Exception as error:
        print("🧹 Couldn't look over the day:", error)
        return 0
    wrong = []
    why = {}
    for kind, nr, reason in re.findall(r"(?i)mistake[ \t]*:[ \t]*([TD])[ \t]*(\d+)[ \t]*[-–—:,]*[ \t]*([^\n]*)", answer)[:5]:
        items = talks if kind.upper() == "T" else notes
        if len(reason.split()) < 3:
            continue  # no real reason given – then it's not a CLEAR mistake, keep the memory
        # It has to name what SHOWS the mistake (another memory, or the face sensor) – otherwise it is only the
        # night brain's own guess (2026-11-06: it deleted "I'll try to reach the cup" twice, because it
        # believed a duck can't carry things). A guess keeps the memory.
        shown = re.search(r"(?i)shown[ \t]*by[ \t]*:?[ \t]*([TD][ \t]*\d+|in[ \t]*view)", reason)
        if not shown or re.sub(r"\s", "", shown.group(1)).upper() == f"{kind.upper()}{nr}":
            item = items[int(nr) - 1] if 1 <= int(nr) <= len(items) else {}
            print(f"🧹 Not sure it was a mistake (no memory shows it) – kept: {item.get('text') or item.get('duck', '')}")
            continue
        reason = re.sub(r"(?i)^[-–—:,\s]*shown[ \t]*by[ \t]*:?[ \t]*(?:[TD][ \t]*\d+|in[ \t]*view)[ \t]*[-–—:,]*[ \t]*",
                        "", reason) or reason
        if 1 <= int(nr) <= len(items) and items[int(nr) - 1] not in wrong:
            wrong.append(items[int(nr) - 1])
            why[id(items[int(nr) - 1])] = reason.strip()
    for w in wrong:
        text = f'"{w["human"]}" → "{w["duck"]}"' if "human" in w else f'"{w["text"]}"'
        print(f"🧹 Forgot a mistake: {text}  ({why[id(w)]})")
    if wrong and not dry_run:
        with memory_lock:
            conversations[:] = [c for c in conversations if not any(c is w for w in wrong)]
            diary[:] = [d for d in diary if not any(d is w for w in wrong)]
            for w in wrong:
                cleanup_log.append({"date": date.today().isoformat(), "what": "forgot a mistake",
                                    "before": [w.get("text") or f'{w["human"]} → {w["duck"]}'], "after": None})
            save_memory()
    return len(wrong)


def tidy_up(dry_run=False):
    """Anka's night-time tidy-up. dry_run=True: only show what would happen, save nothing."""
    print("🧹 Tidying up memories..." + ("  (TEST RUN – nothing will be saved)" if dry_run else ""))
    forgotten = review_day(dry_run) if REVIEW_THE_DAY else 0
    new_self, self_changes = tidy_self_image()
    fact_changes = tidy_facts()
    if not self_changes and not fact_changes:
        if forgotten:
            return
        print("🧹 Everything already tidy.")
        return
    today = date.today().isoformat()
    for kind, before, after, *_ in [(a, b, c) for a, b, c in self_changes] + fact_changes:
        if after:
            print(f"🧹 {kind.capitalize()}: " + " + ".join(f'"{x}"' for x in before) + f'\n      → "{after}"')
        else:
            print(f'🧹 {kind.capitalize()}: "{before[0]}"')
    if dry_run:
        print("🧹 (Test run – nothing was changed.)")
        return

    with memory_lock:
        if new_self is not None:
            self_image[:] = new_self
        for kind, before, after, room in fact_changes:
            if kind == "outdated fact":
                mark_outdated(before[0])
            elif kind == "merged facts":
                day = max((learned_on.get(f, "") for f in before), default="") or today
                for f in before:
                    delete_fact(f)
                add_fact(after)
                learned_on[after] = day
                room_of[after] = room
            else:
                delete_fact(before[0])
        for kind, before, after, *_ in [(a, b, c) for a, b, c in self_changes] + fact_changes:
            cleanup_log.append({"date": today, "what": kind, "before": before, "after": after})
        save_memory()
    print(f"🧹 Done: {len(self_changes) + len(fact_changes)} changes "
          f"(saved under \"cleanup_log\" in {os.path.basename(MEMORY_FILE)})")


def heal_feelings():
    """At night Anka looks again at the day's strongest moments: with the whole day in mind, does any of them
    feel different now? (A hurt can soften after an apology; a joy can turn bittersweet.) The brain decides.
    The first feeling is kept as 'first_feeling', so nothing is lost."""
    with memory_lock:
        strong = [x for x in conversations + diary if since_last_dream(x) and x.get("feeling") and x.get("feel", 0) >= 0.5
                  and not x.get("text", "").startswith("I dreamt")]
    strong = sorted(strong, key=lambda x: x.get("time", ""))[-6:]
    if not strong:
        return
    lines = [f'{i}. ' + (f'{x.get("who") or human_name() or "The human"} said "{x["human"]}", you answered "{x["duck"]}"'
                         if "human" in x else x["text"]) + f' – it felt: {x["feeling"]}' for i, x in enumerate(strong, 1)]
    with memory_lock:
        day = [f'- {c.get("who") or "The human"}: "{c["human"]}"' for c in conversations if since_last_dream(c)][-25:]
    instructions = (f"You are {NAME}, a small robot duck, falling asleep. Look again at the strongest moments of your day "
                    "and how they felt at the time. Now, with the WHOLE day in mind, does any of them feel different? "
                    "A hurt may feel softer after an apology; a joy may have become bittersweet; a fright may seem silly now. "
                    "Only change what really feels different now.\n"
                    "Reply with one line per change:\nCHANGE: <number> => <how it feels now, a few words>\nor NONE.")
    try:
        answer = ollama([{"role": "system", "content": instructions},
                         {"role": "user", "content": "Your day:\n" + "\n".join(day) + "\n\nThe strongest moments:\n"
                                                     + "\n".join(lines)}], temperature=0.3, max_tokens=120)
    except Exception as error:
        print("💞 Couldn't look back at the feelings:", error)
        return
    with memory_lock:
        for nr, new in re.findall(r"(?i)change\s*:?\s*#?(\d+)\s*(?:=>|->|:)\s*([^\n]+)", answer):
            new = clean_feeling(new)
            if new and 1 <= int(nr) <= len(strong) and new.lower() != strong[int(nr) - 1]["feeling"].lower():
                x = strong[int(nr) - 1]
                x.setdefault("first_feeling", x["feeling"])
                print(f'💞 "{x.get("human") or x.get("text")}" felt: {x["feeling"]} → now: {new}')
                x["feeling"] = new
        save_memory()


def sleep_and_dream():
    """Falls asleep with the night brain (NIGHT_CLOUD_BRAIN) – it answers everything until he wakes up."""
    cloud["night"] = True
    if NIGHT_CLOUD_BRAIN and CLOUD_KEY:
        print(f"🌙 (tonight {NAME} thinks with the night brain {NIGHT_CLOUD_BRAIN})")
    try:
        _sleep_and_dream()
    finally:
        cloud["night"] = False
        used = cloud["night_usage"]
        if used:
            print(f"🌙 night brain {NIGHT_CLOUD_BRAIN if cloud['night_ok'] else CLOUD_BRAIN}: {len(used)} questions · "
                  f"read {sum(u[0] for u in used)} tokens · wrote {sum(u[1] for u in used)} "
                  f"(+ {sum(u[2] for u in used)} thinking) · {sum(u[3] for u in used):.0f}s")


def _sleep_and_dream():
    """Anka falls asleep: reflects on everything since the last dream, updates its self-image and dreams."""
    pass  # (shared values live on S)
    print(f"\n😴 {NAME} is going to sleep...")
    while not to_remember.empty():  # conversations not yet saved in the long-term memory
        remember(*to_remember.get())
    with feelings_lock:
        old = [g for g in goals if (date.today() - date.fromisoformat(g.get("since", date.today().isoformat()))).days
               >= GOAL_MAX_DAYS]
        for g in old:
            goals.remove(g)
            print(f"🫳 Let go of an old goal: {g['text']}")
            trace("goal_lost", goal=g["text"], why=f"older than {GOAL_MAX_DAYS} day(s) - dropped by the code at night")
    # One real night's sleep per night, at bedtime: earlier he just rests – no tidy-up, no dream.
    # (Everything since the last dream is kept, so tonight's sleep looks back on all of it.)
    if S.last_dream and not SIM and "--dream" not in sys.argv:
        now = datetime.now()
        night = lambda t: (t - timedelta(hours=5)).date()  # a "night" lasts until 5 in the morning
        try:
            last = datetime.fromisoformat(S.last_dream)
        except ValueError:
            last = now - timedelta(days=2)
        bedtime = now.hour >= BEDTIME or now.hour < 5
        dreamt_tonight = night(last) == night(now)
        long_ago = now - last > timedelta(hours=24)
        if dreamt_tonight or not (bedtime or long_ago):
            print(f"💤 Just a short rest – {NAME} saves his memories"
                  + (" (he already dreamt tonight)." if dreamt_tonight else
                     f" and dreams tonight, the first time you stop him after {BEDTIME}:00.") + " Good night!")
            return
    if TIDY_AT_NIGHT:
        try:
            tidy_up()
        except Exception as error:
            print(f"🧹 Tidying failed ({type(error).__name__}: {error}) – memories left as they were")
    if FEELING_HEALS:
        try:
            heal_feelings()
        except Exception as error:
            print(f"💞 Couldn't look back at the feelings ({type(error).__name__}: {error})")
    with memory_lock:
        # Everything since the last dream, in the order it happened (the whole evening, not only the end)
        timed = [(d.get("time", d["date"]), d["text"]) for d in diary
                 if since_last_dream(d) and not d["text"].startswith("I dreamt")]
        timed += [(c.get("time", c["date"]), f'{c.get("who") or "The human"} said "{c["human"]}" and I answered "{c["duck"]}"'
                   + (f" (I felt {c['felt']})" if c.get("felt") else ""))
                  for c in conversations if since_last_dream(c) and not c.get("unsure")]  # (not misheard words)
        talk_times = sorted(c["time"] for c in conversations if since_last_dream(c) and "time" in c)
        # What Anka said by HIMSELF (saved, so earlier sessions count too), and whether the human answered
        # within a minute (facts for the lessons)
        for o in own_said:
            if since_last_dream(o):
                soon = (datetime.fromisoformat(o["time"]) + timedelta(seconds=60)).isoformat(timespec="seconds")
                answered = any(o["time"] < x <= soon for x in talk_times)
                timed.append((o["time"], f'I said by myself: "{o["text"]}"' + ("" if answered else "  (no answer)")))
        current_self = list(self_image)
        current_lessons = list(lessons)
    timed.sort(key=lambda x: x[0])
    first_time = [re.sub(r"^I saw an? (.+) for the first time\.$", r"\1", t)
                  for _, t in timed if t.endswith("for the first time.")]
    timed = [(w, t) for w, t in timed if not t.endswith("for the first time.")]
    # A long evening: let go of quiet thoughts first, then keep a spread over the whole time
    while len(timed) > MAX_DREAM_EVENTS and any(t.startswith("I thought:") for _, t in timed):
        thoughts_now = [i for i, (_, t) in enumerate(timed) if t.startswith("I thought:")]
        del timed[thoughts_now[len(thoughts_now) // 2]]  # one from the middle at a time, so it stays spread out
    if len(timed) > MAX_DREAM_EVENTS:
        step = len(timed) / MAX_DREAM_EVENTS
        timed = [timed[int(i * step)] for i in range(MAX_DREAM_EVENTS)]
    events = [f"- ({w[11:16]}) {t}" if len(w) > 10 else f"- {t}" for w, t in timed]  # (with the time, e.g. 20:15)
    if first_time:
        events.append("- Things I saw for the first time: " + ", ".join(first_time[-10:]))
    if not events:
        print(f"💤 Nothing new happened, so {NAME} sleeps without dreaming. Good night!")
        return

    body_day = day_words()   # B4: his body's track record today (from anka/tried.py)
    with memory_lock:
        current_ideas = list(ideas)
    max_changes = SELF_CHANGES_PER_NIGHT if current_self else 3  # the first night it may create a bit more
    numbered = "\n".join(f"{i}. {x}" for i, x in enumerate(current_self, 1)) or "(nothing yet – you are brand new)"
    instructions = (
        f"You are {NAME}, a small curious robot duck, and you are falling asleep. "
        "Look back at what happened since you last slept and reflect on who you are becoming.\n\n"
        "Reply with lines in EXACTLY these forms:\n"
        "NEW: <a new sentence about yourself, starting with 'I'>\n"
        "CHANGE: <number> => <updated sentence about yourself, starting with 'I'>\n"
        "REMOVE: <number of a self-belief that is no longer true>\n"
        "ABOUT_HUMAN: <an insight about the human, starting with 'The human'>\n"
        "LESSON: <something you will do differently from now on, starting with 'When' or 'Don't' or 'I should'>\n"
        + ("IDEA: <something you have NOT tried yet that might work next time, starting with 'Maybe I could'>\n"
           "IDEA_DONE: <number of one of your earlier ideas that your body tried today (it worked or not) or that no longer "
           "makes sense>\n" if IDEAS else "")
        + "\n"
        "Rules:\n"
        f"- At most {max_changes} lines in total of NEW, CHANGE and REMOVE. Change slowly; "
        "only change what today's experiences really support. It is fine to change nothing.\n"
        "- Self-beliefs are about your feelings, likes, dislikes, habits, fears and your relationship "
        "with the human. Base them on what actually happened.\n"        "- A belief or lesson about what your body CAN or CAN'T do, or about how you try things, must match "
        "what your body really managed (the facts from your body below). Write what happened ('I tried to "
        "climb the bookshelf and the guitar and couldn't get up'), not what you hoped or planned. If one of "
        "your beliefs is contradicted by those facts, CHANGE or REMOVE it.\n"
        "- Write them plain and concrete, like 'I love it when Alex comes home'. Never start with 'I am learning'.\n"
        "- At most 2 ABOUT_HUMAN lines, only about something the human really SAID to you – never about "
        "things you only saw, thought or dreamt.\n"
        "- At most 2 LESSON lines. Look honestly at how your talking went: did you ask for the same thing "
        "again and again, ask questions nobody answered, interrupt, repeat yourself, or say something "
        "that went well? A lesson must come from something that really happened today. "
        "It is fine to write no lesson. Don't repeat a lesson you already have."
        + (f"\n- IDEAS are different from lessons: a lesson is something that HAPPENED; an idea is something you have "
           f"not tried yet. Think about what didn't work today and what else you could do – something your body can "
           f"really do. At most "
           f"{IDEAS_PER_NIGHT} IDEA lines; it is fine to write none. If your body tried one of your earlier ideas today, "
           "write IDEA_DONE for it – and if it worked, also a LESSON or NEW line about what happened." if IDEAS else "")
    )
    try:
        text = ollama([
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"Who you are now:\n{numbered}\n\n"
                                        + ("Lessons you already have:\n- " + "\n- ".join(current_lessons) + "\n\n"
                                           if current_lessons else "")
                                        + ("Your earlier ideas (untested when you wrote them):\n"
                                           + "\n".join(f"{i}. {x['text']}" for i, x in enumerate(current_ideas, 1)) + "\n\n"
                                           if IDEAS and current_ideas else "")
                                        + "What happened since you last slept:\n" + "\n".join(events)
                                        + ("\n\nWhat your body tried today and what came of it (facts from your "
                                           "body – trust these over what you hoped or said):\n" + body_day
                                           if body_day else "")},
        ], temperature=0.3)  # calm and thoughtful
    except Exception as error:
        print(f"💤 {NAME} couldn't reflect:", error)
        text = ""

    new, change, remove, insights, new_lessons = [], {}, set(), [], []
    new_ideas, ideas_done = [], set()
    for line in text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        m = re.match(r"(?i)idea_done\s*:?\s*#?(\d+)", line)
        if m and IDEAS:
            if 0 < int(m.group(1)) <= len(current_ideas):
                ideas_done.add(int(m.group(1)) - 1)
            continue
        m = re.match(r"(?i)idea\s*:\s*(.+)", line)
        if m and IDEAS:
            idea = m.group(1).strip().strip('"')
            if len(new_ideas) < IDEAS_PER_NIGHT and 4 <= len(idea.split()) <= 25 and not re.match(r"(?i)none\b", idea) \
                    and not any(same_fact(idea, x["text"]) for x in current_ideas) \
                    and not any(same_fact(idea, x) for x in current_lessons):
                new_ideas.append(idea)
            continue
        m = re.match(r"(?i)lesson\s*:\s*(.+)", line)
        if m and len(new_lessons) < 2:
            lesson = m.group(1).strip().strip('"')
            if 4 <= len(lesson.split()) <= 25 and not re.match(r"(?i)none\b", lesson) \
                    and not any(same_fact(lesson, x) for x in current_lessons):
                new_lessons.append(lesson)
            continue
        m = re.match(r"(?i)change\s*:?\s*#?(\d+)\s*(=>|->|:)\s*(.+)", line)
        if m:
            nr = int(m.group(1)) - 1
            new_line = about_me(m.group(3))
            if 0 <= nr < len(current_self) and new_line and not same_fact(new_line, current_self[nr]):
                change[nr] = new_line
            continue
        m = re.match(r"(?i)remove\s*:?\s*#?(\d+)", line)
        if m:
            nr = int(m.group(1)) - 1
            if 0 <= nr < len(current_self):
                remove.add(nr)
            continue
        m = re.match(r"(?i)new\s*:\s*(.+)", line)
        if m:
            sentence = about_me(m.group(1))
            if sentence and (re.match(r"(?i)i (am learning|am anka\b|am a (small |curious )*robot duck|want to|wonder)",
                                      sentence) or "falling asleep" in sentence.lower()):
                sentence = None  # says nothing about who Anka has become
            if sentence:
                happened = " ".join(events).lower()
                words = [w for w in re.findall(r"[a-z]{4,}", sentence.lower()[2:])
                         if w not in ("feel", "feels", "love", "like", "when", "with", "that", "they", "warm",
                                      "happy", "really", "always", "very", "much", "being", "about", "have")]
                if words and sum(w in happened for w in words) < max(1, len(words) // 3):
                    print(f"🪞 (not added – not based on what happened: {sentence})")
                    sentence = None
            if sentence:
                new.append(sentence)
            continue
        m = re.match(r"(?i)about_human\s*:\s*(.+)", line)
        if m and len(insights) < 2:
            insight = good_sentence(m.group(1), "The human")  # skips "NONE" and junk
            said = " ".join(c["human"] for c in conversations if since_last_dream(c)).lower()
            words = [w for w in re.findall(r"[a-z]{4,}", (insight or "").lower()[10:])
                     if w not in ("human", "likes", "loves", "enjoys", "really", "about", "their", "with", "that",
                                  "they", "very", "when", "much", "home", "talks", "shares")]
            if insight and (not said or sum(w in said for w in words) < max(1, len(words) // 2)):
                insight = None  # not built on what the human said – could be made up
            if insight and (re.search(r'["“”]', insight)
                            or re.match(r"(?i)the human (said|asked|told|says|asks)\b", insight)):
                insight = None  # just repeats something said – not something learned about the human
            if insight:
                insights.append(insight)

    # Limit how much the personality may change in one night
    left = max_changes
    change = dict(list(change.items())[:left])
    left -= len(change)
    remove = set(list(remove - set(change))[:max(left, 0)])
    left -= len(remove)
    new = [n for n in new if not any(same_fact(n, x) for x in current_self)][:max(left, 0)]

    dream = make_dream(events, current_self)

    with memory_lock:
        updated = []
        for i, x in enumerate(current_self):
            if i in remove:
                print(f"🪞 No longer true: {x}")
            elif i in change:
                print(f'🪞 Changed: "{x}" → "{change[i]}"')
                updated.append(change[i])
            else:
                updated.append(x)
        for n in new:
            print(f"🪞 New about me: {n}")
        self_image[:] = (updated + new)[-MAX_SELF:]
        for lesson in new_lessons:
            print(f"📝 Lesson for tomorrow: {lesson}")
        lessons[:] = (lessons + new_lessons)[-MAX_LESSONS:]
        if IDEAS:
            for i in sorted(ideas_done):
                print(f"💡 Idea tried or dropped: {current_ideas[i]['text']}")
            kept = [x for i, x in enumerate(current_ideas) if i not in ideas_done]
            for idea in new_ideas:
                print(f"💡 Idea to try: {idea}")
            ideas[:] = (kept + [{"text": idea, "date": date.today().isoformat()} for idea in new_ideas])[-MAX_IDEAS:]
        clear_day()   # (a new day of tries starts after this night)
        if dream:
            if not dream.lower().startswith("i dreamt"):
                dream = "I dreamt " + dream
            print(f"💭 {dream}")
            write_diary(dream)
        learned = []
        for insight in insights:
            if not any(same_fact(insight, f) for f in facts):
                add_fact(insight)
                learned.append(insight)
                print(f"🧠 Learned while dreaming: {insight}")
        S.last_dream = datetime.now().isoformat(timespec="seconds")
        save_memory()
    sort_into_rooms(learned)
    prune_vectors()
    try:   # a body that can remember its room (the MuJoCo duck) sleeps on what it learned about it
        from anka.body import body as _body
        if _body is not None and hasattr(_body, "sleep_on_it"):
            _body.sleep_on_it(ollama, events)
    except Exception as error:
        print(f"💤 {NAME} couldn't sleep on his room:", error)
    print("💤 Good night!")


# ---- other parts of Anka (imported last, so the parts can use each other) ----
from anka.feelings import clean_feeling, feelings_lock
from anka.memory import (picture_b64, ideas, add_fact, cleanup_log, conversations, core, delete_fact, diary, facts, goals, human_name, learned_on, lessons, mark_outdated, memory_lock, own_said, prune_vectors, remember, room_of, same_fact, save_memory, self_image, since_last_dream, sort_into_rooms, write_diary)
from anka.brain import cloud, good_sentence, means_same, ollama, parse_numbers
from anka.trace import trace
from anka.tried import clear_day, day_words
