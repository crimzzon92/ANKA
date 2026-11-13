"""The thinking loop: a thought, a feeling, "do I want to say something?", and goals."""
import random
import re
import time
from datetime import date, datetime

from settings import MAX_PICTURES, PICTURE_MEMORY, EXPECT, FROZEN_FEELINGS, IDEAS, MAX_IDEAS, SURPRISE_DIARY, STUCK_AFTER, WISE_WHEN_STUCK
from settings import (AWAY_AFTER, BODY_SENSES, CHAIN_OF_THOUGHTS, HAS_BODY, CONVERSATION_PAUSE, FEELING_STEP, GOALS_EVERY, MAX_GOALS, MIN_PAUSE, MOVING_FOR, NAME, NAP_AFTER, NAP_WHEN_ALONE, ONE_STEP_THOUGHTS, REST_MAX, SHORT_OUTPUT, SHOW_FEELINGS, SIM, TALK_FADES, TALK_IN_THOUGHTS, TALK_ON_ITS_OWN, TEMP_REPLY, TEMP_THOUGHT, THINK_EVERY, THINK_EVERY_ALONE, THINK_EVERY_MOVING, THOUGHTS_FADE, THOUGHTS_IN_MIND, THOUGHT_MEMORY)
from anka.shared import (S, answering, awake, heard, listening, state, thought_times, thoughts, timings, to_remember, voice_waiting, wake, wants_to_say)



def think_pace():
    """Seconds between quiet thoughts: slower when nobody is there. A (low, high) pair = a random time in between."""
    pace = THINK_EVERY if human_is_here() else THINK_EVERY_ALONE
    if HAS_BODY and time.time() - state.get("last_travel", 0) < MOVING_FOR:
        pace = THINK_EVERY_MOVING   # on the move (exploring, on an errand): think more often
    return random.uniform(*pace) if isinstance(pace, (tuple, list)) else pace


# 2026-11-08: speaking is an optional line, not a yes/no question every thought (asked every 4–6 s, a direct
# "do you want to speak?" pulled him towards yes: 14–18 lines in 5 min alone). Still entirely his choice.
SAY_FORMAT = ("Most of your thoughts can simply stay in your head. Only if you have something you really want to say "
              "out loud right now (NEW, not said before{heard_or_not}), fill in SAY – otherwise write none.\n"
              "SAY: <what you say out loud, or none>\n"
              "WHY: <only if you say something: the real reason, in a few words – otherwise none>")


def is_alone(now=None):
    """Alone = nobody seen or heard for AWAY_AFTER seconds – not just 'out of the camera's view for a moment'.
    (A fact he is told; what he does about it is his own choice.)"""
    now = now or time.time()
    if human_is_here() or current_speaker():
        return False
    last = state["person_last"]
    if last:
        return now - last > AWAY_AFTER
    return now - state.get("woke_up", now) > AWAY_AFTER      # nobody at all since he woke up


def presence_facts(now):
    """Who is with him, as facts. When he is alone he is told so plainly – with how long, and the last thing the human
    said before it went quiet (test 26: all 28 things he said by himself were said with Alex in the room; alone he
    was almost silent – he should know when he really is alone, and that talking to himself then is fine)."""
    visitor = current_speaker()
    if visitor:
        return f"{visitor} is here with you right now – {human_name()} is not in view.\n"
    if human_is_here():
        seen = state.get("person_seen_at", 0)
        if HAS_BODY and state.get("faces", 0) == 0 and seen and now - seen > 5:   # around, but not in front of him
            return (f"{human_name() or 'The human'} is in the room with you but not in view right now (you last saw them "
                    f"{ago(now - seen)}, and you haven't heard anyone go out) – maybe behind you or beside you.\n")
        return "The human is here with you right now.\n"
    if not is_alone(now):
        return ("The human is not in front of your camera right now"
                + (f" (out of view for {how_long(now - state['person_last'])})" if state["person_last"] else "")
                + ". They might be close by without being in view, or in another room – if you call out, "
                  "maybe they hear you, maybe not.\n")
    who = human_name() or "the human"
    last = state["person_last"]
    out = ("YOU ARE ALONE: "
           + (f"you haven't seen or heard anyone for {how_long(now - last)}" if last else
              f"you haven't seen or heard anyone since you woke up ({how_long(now - state.get('woke_up', now))} ago)")
           + (" and you heard someone go out" if state.get("heard_leave_at") else "")
           + ", so as far as you can tell nobody is in the room with you.\n")
    with memory_lock:
        last_talk = conversations[-1].get("human") if conversations else None
    if last_talk and state["last_human"] and now - state["last_human"] < 1800:
        out += f'The last thing {who} said to you ({ago(now - state["last_human"])}): "{last_talk[:160]}"\n'
    return out + "Nobody can hear you right now: anything you say out loud, only you will hear.\n"


def reflection_material(name):
    """When Anka is alone, it can also look inwards. Gives it a few things from its own life to think about –
    what it thinks about is still up to the brain."""
    if human_is_here():
        return ""
    with memory_lock:
        today = date.today().isoformat()
        events = [d["text"] for d in diary if d["date"] == today
                  and not d["text"].startswith(("I thought:", "I dreamt", "I decided:", "I finished my goal:", "I gave up on:"))
                  and "for the first time" not in d["text"]][-3:]
        talks = [f'{name or "The human"} said: "{c["human"]}"' for c in conversations if c["date"] == today][-2:]
        dreams = [d["text"] for d in diary if d["text"].startswith("I dreamt")][-1:]
    material = []
    for m in events[-2:] + talks[-1:] + dreams + comes_to_mind(name, about=(thoughts[-1] if thoughts else None)):
        if not any(re.sub(r"^\(.*?\) ", "", m) in x or x in m for x in material):
            material.append(m)  # (the same memory only once)
    return (("\nYou are alone right now." if is_alone() else "\nNobody is in view right now.")
            + " Besides what is around you, you could also think about yourself: "
            "who you are becoming, what you like or fear, something that happened, a dream, or how you feel "
            f"about {name or 'the human'}.\n"
            + ("Things from your own life that come to mind (use one, or none):\n- " + "\n- ".join(material) + "\n"
               if material else ""))


def should_yield():
    """Should the thought be dropped? Yes if you talk, if Anka is answering you, or if the voice needs the turn."""
    return listening.is_set() or not heard.empty() or answering.is_set() or voice_waiting.is_set()


def think_loop():
    """Runs the thoughts – and restarts them if something goes wrong, instead of Anka stopping thinking."""
    while awake.is_set():
        try:
            _think_loop()
        except Interrupted:
            print("💭 (stopped my thought – you're talking)")
        except Exception as error:
            print(f"💭 Thinking crashed ({type(error).__name__}: {error}) – starting again")
            time.sleep(5)


def _think_loop():
    """Anka feels, thinks, sets its own goals and decides whether to say something."""
    next_thought = time.time() + 10  # the first thought comes soon after Anka wakes up
    previous = time.time()
    while awake.is_set():
        if wake.wait(timeout=1):  # something happened – think now!
            wake.clear()
            next_thought = min(next_thought, time.time())
        now = time.time()
        # Napping? Nobody has been around for a long time – no thoughts until someone comes.
        if NAP_WHEN_ALONE and not state.get("napping") and not human_is_here() and not speaking():
            alone_for = now - max(state["person_last"], state["last_human"], state["last_pip"],
                                  state.get("woke_up", now))
            if alone_for > NAP_AFTER:
                state["napping"] = now
                print(f"\n😴 ({NAME} takes a nap – nobody has been around for {how_long(alone_for)}. "
                      "No thinking until someone comes)")
                with memory_lock:
                    write_diary("I took a nap because nobody was around.")
                    save_memory()
        if state.get("napping"):
            if human_is_here() or not heard.empty() or listening.is_set() or state["last_human"] > state["napping"]:
                slept = now - state.pop("napping")
                print(f"🌅 ({NAME} wakes up from the nap)")
                state["events"] = [e for e in state["events"] if "human left" not in e]  # (old news after a nap)
                state["events"].append(f"You just woke up from a nap (you slept for {how_long(slept)}).")
                state["rest"] = 0
                next_thought = now
            else:
                previous = now  # (feelings rest too while he sleeps)
                continue
        if now - previous >= FEELING_STEP:
            drift_feelings(previous, now)
            previous = now
            if SHOW_FEELINGS:
                print(f"❤️  {feelings_short()}")
        if now < next_thought:
            continue
        # Don't disturb while you talk, while Anka listens, answers or already speaks
        if should_yield() or speaking() or now - max(state["last_human"], state["last_pip"]) < CONVERSATION_PAUSE:
            next_thought = now + 5
            continue
        # Resting? Then Anka thinks less often until something happens (saves the computer, breaks loops)
        next_thought = now + min(REST_MAX, think_pace() * (2 ** state["rest"]))

        # Did you talk? Go through it for the long-term memory now, while it's calm.
        while not to_remember.empty():
            item = to_remember.get()
            try:
                remember(*item, stop=should_yield)  # stops at once if you start talking again
            except Interrupted:
                to_remember.put(item)  # try again at the next calm moment
                raise

        # What has turned up since the last thought?
        appeared = state["appeared"]
        state["appeared"] = []
        events = state["events"]
        state["events"] = []

        quiet_for = now - max(state["last_human"], state["last_pip"], state.get("woke_up", now))
        may_talk = TALK_ON_ITS_OWN and now - state["last_pip"] > MIN_PAUSE  # only a safety limit
        # working memory: only his latest thoughts, and only fresh ones (old ones make Anka go in circles)
        recent = [t for t, at in zip(thoughts, thought_times) if now - at < THOUGHTS_FADE][-THOUGHTS_IN_MIND:]
        talk_in_mind = (state["recent_talk"][-TALK_IN_THOUGHTS:]
                        if TALK_IN_THOUGHTS and now - state["last_human"] < TALK_FADES else [])
        # Which words turn up in almost all recent thoughts? (only information for the brain)
        stop_words = {"wonder", "maybe", "think", "still", "there", "that", "this", "what", "when", "with",
                      "about", "from", "just", "like", "notice", "something", "would", "could", "might", "again"}
        latest = state["all_thoughts"][-6:]
        counts = {}
        for t in latest:
            for w in set(re.findall(r"[a-z]{4,}", t.lower())) - stop_words:
                counts[w] = counts.get(w, 0) + 1
        nagging = sorted([w for w, n in counts.items() if n >= 4], key=lambda w: -counts[w])[:3]
        openings, opening_text = {}, {}
        for t in latest:
            start = " ".join(re.findall(r"[A-Za-z']+", t)[:2])
            if start:
                openings[start.lower()] = openings.get(start.lower(), 0) + 1
                opening_text.setdefault(start.lower(), start)
        same_start = [o for o, n in openings.items() if n >= 3]
        with feelings_lock:
            goal_count = len(goals)
            feelings_now = ", ".join(f"{k}={v:.2f}" for k, v in feelings.items())
            feel_before = {k: round(v, 2) for k, v in feelings.items()}   # (for the thought log)

        # Facts about the situation – no decisions. The brain decides itself what is interesting.
        stuck = len(recent) >= 3 and all(means_same(recent[-1], [x], 0.8) for x in recent[-3:-1])
        familiar = [f"{n} (has been there for {ago(now - state['seen_since'].get(n, now)).replace(' ago', '')})"
                    for n in state["familiar"]]
        visitor = current_speaker()
        presence = presence_facts(now)
        here_now, alone_now = human_is_here(), is_alone(now)   # (for the thought log)
        picture = picture_for_thought(events or appeared)
        name = human_name()
        happened = " ".join(events).replace("The human", name) if name else " ".join(events)
        scene = brief_scene() if SIM else state["scene"] if picture is not None else scene_without_picture()
        situation = (
            f"{scene}\n"
            + (f"Just appeared: {', '.join(appeared)}.\n" if appeared else "")
            # (left out when something happened or Anka is alone – then the same old things only pull him back)
            + (f"Things that haven't changed for a while: {', '.join(familiar)}.\n"
               if familiar and not events and human_is_here() else "")
            + presence
            + (f"The human last talked to you {ago(now - state['last_human'])}.\n"
               if state["last_human"] else "The human hasn't talked to you yet since you woke up.\n")
            + (f"You last said something out loud {ago(now - state['last_pip'])}.\n"
               if state["last_pip"] else "You haven't said anything yet since you woke up.\n")
            + (f"It has been quiet for {ago(quiet_for).replace(' ago', '')}.\n" if quiet_for > 60 else "")
        )

        # B4: what he has learned about himself, right where he decides what to do and say (not only far up
        # in the long system text, where the narration lessons were ignored for 3 days). Information, not
        # orders: he names which one matters now (MINDFUL), or none – and he may disagree with it.
        with memory_lock:
            learned = list(lessons) + [s_ for s_ in self_image[-6:] if s_ not in lessons]
            untested = [x["text"] for x in ideas] if IDEAS else []
        know_myself = ("\nWhat you have learned about yourself, from looking back on your own days (it can be "
                       "wrong – you may disagree – but think about it before you act or speak):\n- "
                       + "\n- ".join(learned) + "\n" if learned else "")
        # 2026-11-09: ideas he thought up at night – NOT tested yet; trying one is his own choice
        know_myself += ("Ideas you thought up but have NOT tried yet (they may or may not work – only trying shows):\n- "
                        + "\n- ".join(untested) + "\n" if untested else "")

        # B1: what he expected from his last action, and what his body reported – he rates the surprise himself
        expected = state.get("expectation") if EXPECT and HAS_BODY else None
        if expected and now - expected["t"] > 180:
            expected = state.pop("expectation", None) and None     # (too long ago to compare)
        expect_check = (f"\nLast time your body did something: '{expected['act']}'. You expected: \"{expected['expect']}\". "
                        f"What your body reported: {expected['result']}\n" if expected else "")

        # Stuck (several failed tries in a row, or a big surprise): THIS thought goes to the wiser brain (the night's model).
        # Only a better brain for the moment – nothing about what to think. (Replaces "think harder" + day ideas, 2026-11-09.)
        # One principle (2026-11-10): STUCK = things went wrong STUCK_AFTER thoughts in a row - a failed try OR a big
        # surprise counts as a miss. (One surprise alone used to be enough; in the 3D room he is surprised all the time
        # while searching, so the wiser brain came without him being stuck.)
        streak = state.get("fail_streak", 0)
        misses = state.get("miss_streak", 0)
        level = None
        if WISE_WHEN_STUCK and misses >= STUCK_AFTER:
            level = "wise"
            state["miss_streak"] = 0      # the wiser brain has had its turn - it takes two NEW misses to ask it again
            print(f"🦉 (stuck – {misses} misses in a row (a failed try or a big surprise): this thought goes to his wiser brain)")

        # ===== Step 1: one thought and one feeling – nothing else =====
        step1 = (
            "[PRIVATE THINKING TIME. Nobody can hear you right now.]\n"
            + situation
            + ((("Your thoughts from BEFORE this just happened (they may be out of date now):\n- " if events
                 else "Your recent thoughts (oldest first):\n- ") + "\n- ".join(recent) + "\n") if recent else "")
            + (f"NOTE: {counts[nagging[0]]} of your latest {len(latest)} thoughts were about: {', '.join(nagging)}. "
               "You keep circling back to it – let it go for now and think about something else: the human, "
               "how you feel, something from your memories, or what just happened.\n" if nagging else "")
            + "".join(f'NOTE: {openings[o]} of your latest {len(latest)} thoughts started with "{opening_text[o]}". '
                      "Start this one another way.\n" for o in same_start)
            + (f"NOTE: you just thought \"{state.pop('repeated_thought')}\" again – think about something "
               "different this time.\n" if state.get("repeated_thought") else "")
            + (f"NOTE: your last {len(recent[-3:])} thoughts were all about the same thing.\n" if stuck else "")
            + f"Your feelings right now (as numbers): {feelings_now}\n"
            + reflection_material(name)
            # what just happened comes last – small brains pay most attention to the end
            + (f"\nIMPORTANT – this just happened: {happened}\n"
               "This is the most important thing right now – let your thought be about it.\n" if events else "")
            + "\nThink ONE new thought, like a small, curious, living duck. Then say how you feel now.\n"
            + know_myself
            + expect_check
            + "Reply in EXACTLY this form:\n"
            "THOUGHT: <one short private thought, first person, at most 20 words>\n"
            "FEEL: curiosity=<0-1>, boredom=<0-1>, loneliness=<0-1>, happiness=<0-1>, affection=<0-1>, "
            "nervousness=<0-1>, playfulness=<0-1>\n"
            "FEELING: <how you feel, in your own words – a few words, mixed feelings are fine>\n"
            "FACE: <your face right now, one word: happy, curious, surprised, excited, sleepy, bored, sad, "
            "thinking, love, shy, scared, annoyed or playful>\n"
            "REMEMBER: <yes if this moment is important enough for your diary, otherwise no>"
            + ("\nMINDFUL: <which of the things you learned about yourself (or an idea you want to try) matters for what "
               "you do or say now – in a few words – or none>" if know_myself else "")
            + ("\nSURPRISE: <0-1: how different was what happened from what you expected (0 = just as you thought, "
               "1 = completely different) – not how glad or sad you are about it> – <why, in a few words>" if expect_check else "")
            + ("\nACT: <what your body does now – or none. One action, or up to three joined with 'then'>"
               if HAS_BODY else "")
            + ("\nEXPECT: <if your body does something: what you think will happen, in a few words – otherwise none>"
               if HAS_BODY and EXPECT else "")
            + ("\nBODY: <how your feelings show in your body right now, in a few words – or none>"
               if HAS_BODY and getattr(body, "living", None) is not None else "")

            + ("\nPLACE_NOTE: <only if you want to remember something about this spot – otherwise none>"
               if HAS_BODY and BODY_SENSES == "real" else "")
            + ("\nKEEP_PICTURE: <only if you want to keep what you see right now as a picture in your memory: why, in a "
               "few words – otherwise none>" + _pictures_note() if PICTURE_MEMORY and picture is not None else "")
        )
        with memory_lock:
            answers = {c["duck"] for c in conversations[-20:]}
            talk_times = [datetime.fromisoformat(c["time"]).timestamp() for c in conversations[-20:] if "time" in c]
        by_myself = [t for t, x in state["said"] if now - t < 120 and x not in answers]  # not answers to the human
        got_answer = sum(any(t < h <= t + 60 for h in talk_times) for t in by_myself)
        chatty = (f"In the last 2 minutes you said {len(by_myself)} things by yourself, and the human answered "
                  f"{got_answer} of them.\n" if len(by_myself) >= 2 else "")
        asked = [(t, x) for t, x in state["said"] if t > state["last_human"] and x.rstrip().endswith("?")]
        unanswered = ("".join(f'You asked "{x}" {ago(now - t)} – the human has not answered yet.\n'
                              for t, x in asked[-3:]) if asked else "")
        said_before = chatty + unanswered + ("Things you already said OUT LOUD recently – don't say any of them again, not even with "
                       "other words:\n- " + "\n- ".join(f"({ago(now - t)}) {x}" for t, x in state["said"][-8:]) + "\n"
                       if state["said"] else "")
        heard_or_not = "" if human_is_here() else \
            (" – nobody is here to hear you, only you would hear it"
             if is_alone(now) else " – the human isn't in view; if you call out, they might or might not hear you")
        # one trip or two? (decided once per thought, so the answer format matches what was asked)
        one_step = ONE_STEP_THOUGHTS is True or (ONE_STEP_THOUGHTS == "cloud" and cloud_is_on())
        if one_step:
            # the same questions as step 2 below, asked in the same trip to the brain
            step1 += ("\n" + (said_before + SAY_FORMAT.format(heard_or_not=heard_or_not)
                               if may_talk else "SAY: none  (you spoke a moment ago)"))
        # B5: what he searches his memory with – what just happened, his last thought, his first goal, how he feels,
        # and what he sees (before: only the last thought + the scene)
        ft = state.get("feeling_text")
        search = " ".join(x for x in (happened if events else "", recent[-1] if recent else "",
                                      goals[0]["text"] if goals else "",
                                      ft[1] if ft and time.time() - ft[0] < 300 else "", state["scene"]) if x)
        try:
            think_level.level = level
            answer, sent = ask_brain(talk_in_mind + [{"role": "user", "content": step1}],
                                     temperature=TEMP_THOUGHT + (0.2 if state["retry"] else 0), search=search,
                                     max_tokens=(180 if one_step else 120) + (45 if HAS_BODY else 0) + (25 if know_myself else 0)
                                     + (35 if HAS_BODY and EXPECT else 0) + (15 if PICTURE_MEMORY else 0) + (15 if HAS_BODY else 0),
                                     with_goals=False,
                                     stop=should_yield, picture=picture, return_sent=True)
        except Interrupted:
            think_level.level = None
            state["events"] = events + state["events"]  # think about what happened next time
            raise
        except Exception as error:
            think_level.level = None
            print("💭 Couldn't think:", error)
            continue
        think_level.level = None
        if not awake.is_set():
            break
        mem_shown = memories_shown()   # (for the thought log)
        if not heard.empty() or listening.is_set() or time.time() - state["last_human"] < CONVERSATION_PAUSE:
            trace("thought_dropped", why="you started talking", thought=field("THOUGHT", answer))
            continue  # you started talking while Anka was thinking – drop the thought

        thought = field("THOUGHT", answer)
        feeling_line = field("FEEL", answer)
        own_words = clean_feeling(field("FEELING", answer))
        if own_words and not FROZEN_FEELINGS:   # (experiment: held feelings – his own words don't carry over)
            state["feeling_text"] = (time.time(), own_words, "thought")
        worth_remembering = (field("REMEMBER", answer) or "").lower().startswith("yes")
        mindful_raw = field("MINDFUL", answer) if know_myself else None   # (B4 – read now, before step 2 replaces 'answer')
        face = re.findall(r"[a-z]+", (field("FACE", answer) or "").lower())
        if face and face[0] in EXPRESSIONS:
            state["face"] = (face[0], time.time())  # the brain chose its own face
            timings["faces"][face[0]] = timings["faces"].get(face[0], 0) + 1
        style = field("BODY", answer) if HAS_BODY and getattr(body, "living", None) is not None else None
        if style:                                        # how his feelings show in his body (the living body keeps it)
            body.living.set_style(style.splitlines()[0])
            if not re.fullmatch(r"(?i)\W*(none|no|nothing)\W*", style.splitlines()[0]):
                print(f"🫀 (body: {style.splitlines()[0][:80]})")
        keep = field("KEEP_PICTURE", answer) if PICTURE_MEMORY and picture is not None else None
        if keep and not re.fullmatch(r"(?i)\W*(none|no|nothing|n/?a)\W*", keep.splitlines()[0]):
            keep_picture(picture, keep.splitlines()[0].strip(" .\"")[:120])     # his choice: keep what he sees now
        act_text, act_result = "", None
        failed_now = False
        if HAS_BODY:
            note = field("PLACE_NOTE", answer)
            if note and hasattr(body, "note_place"):
                body.note_place(note.splitlines()[0])
            act_text = (field("ACT", answer) or "").splitlines()[0].strip() if field("ACT", answer) else ""
            act_result = run_act(act_text)  # his body does what he decided (even if the thought was an old one)
            if act_result:   # how many thoughts in a row his body failed at something (for thinking harder)
                failed_now = any(not x.get("ok", True) for x in act_result)
                state["fail_streak"] = state.get("fail_streak", 0) + 1 if failed_now else 0
        day_idea = None
        surprise, surprise_why, expect_now = None, "", None
        state["last_surprise"] = 0.0
        if expected:                                   # B1: how surprised was he by what happened last time?
            state.pop("expectation", None)
            m = re.match(r"\s*([01](?:\.\d+)?)\s*[–—:,\-]?\s*(.*)", (field("SURPRISE", answer) or "").splitlines()[0]
                         if field("SURPRISE", answer) else "")
            if m:
                surprise, surprise_why = min(1.0, float(m.group(1))), m.group(2).strip(" .")[:120]
                state["last_surprise"] = surprise
                if re.fullmatch(r"(?i)none|nothing|n/?a", surprise_why):
                    surprise_why = ""
                if surprise >= SURPRISE_DIARY:
                    print(f"😮 (surprised {surprise:.1f}: expected \"{expected['expect']}\"" + (f" – {surprise_why}" if surprise_why else "") + ")")
                    with memory_lock:
                        # (short: the long body report filled his memory search with walking words – B5, 2026-11-09)
                        write_diary(f"I was surprised: I expected {expected['expect']}, but "
                                    + (surprise_why or expected['result'].split(' → ', 1)[-1][:90]), feel=surprise)
                        save_memory()
        if act_result or surprise is not None:      # a miss = his body failed at something, or a big surprise
            missed = failed_now or (surprise or 0) >= 0.6
            state["miss_streak"] = state.get("miss_streak", 0) + 1 if missed else 0
        if HAS_BODY and EXPECT and act_text and act_result:   # B1: what he expects from what his body just did
            raw = (field("EXPECT", answer) or "").splitlines()[0].strip(" .\"") if field("EXPECT", answer) else ""
            if raw and not re.fullmatch(r"(?i)none|nothing|n/?a|-", raw):
                expect_now = raw[:120]
                state["expectation"] = {"t": time.time(), "act": act_text, "expect": expect_now,
                                        "result": "; ".join(f"'{x['step']}' → {x['result']}" for x in act_result)[:300]}
        common_log = dict(feel_before=feel_before, feeling_words=own_words, act=act_text, act_result=act_result,
                          expect=expect_now, expected=expected, surprise=surprise, surprise_why=surprise_why or None,
                          think_level=level, fail_streak=streak, day_idea=day_idea,
                          human_here=here_now, alone=alone_now, visitor=visitor, events=events, appeared=appeared,
                          notes={"nagging": nagging, "same_start": same_start, "stuck": stuck},
                          memories=mem_shown, recent=recent[-2:], picture=picture is not None)
        if not thought:
            trace("thought", thought=None, repeated=False, feel_after=dict(feelings), **common_log)
            continue
        thought = thought.splitlines()[0].strip()
        thought = re.sub("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]", "", thought)  # no emojis
        thought = " ".join(" ".join(re.split(r"(?<=[.!?])\s+", thought)[:2]).split())  # short: at most two sentences
        repeated = means_same(thought, thoughts[-THOUGHT_MEMORY:], 0.9)
        # ...unless it takes the previous thought further (new words, not just the same in other words)
        common = {"that", "this", "with", "what", "about", "there", "here", "still", "just", "like", "looks",
                  "wonder", "maybe", "really", "very", "from", "when", "would", "could", "think", "feel",
                  "exactly", "same", "again", "more", "much", "some", "something", "also", "seems", "feels"}
        words = lambda t: set(re.findall(r"[a-z]{4,}", t.lower())) - common
        before = set().union(*(words(t) for t in thoughts[-3:])) if thoughts else set()
        if repeated and len(words(thought) - before) >= 2 and state.get("chain", 0) < CHAIN_OF_THOUGHTS:
            repeated = False
            state["chain"] = state.get("chain", 0) + 1  # a chain of thoughts – allowed for a while
            print(f"💭 (thinking further on the same thing, {state['chain']}/{CHAIN_OF_THOUGHTS})")
        elif not means_same(thought, thoughts[-1:], 0.8):
            state["chain"] = 0  # a new subject
        state["all_thoughts"] = (state["all_thoughts"] + [thought])[-20:]
        if repeated and not events:
            if state["retry"] >= 1:
                state["retry"] = 0  # already thought again once – wait for the next thought
                state["rest"] = min(state["rest"] + 1, 4)
                rest = min(REST_MAX, think_pace() * (2 ** state["rest"]))
                next_thought = time.time() + rest
                print(f"💭 (same thought again: {thought})  😪 nothing new – resting {rest}s")
            else:
                state["repeated_thought"] = thought
                state["retry"] += 1
                next_thought = time.time() + 2
                print(f"💭 (same thought again: {thought} – thinking again)")
            trace("thought", thought=thought, repeated=True, feel_after=dict(feelings), **common_log)
            continue
        # (If something happened – e.g. you came back – Anka goes on to step 2 even if the thought was old)
        state["retry"] = 0
        state["rest"] = 0  # a new thought, or something happened – awake again
        if not repeated:
            thoughts.append(thought)
            thought_times.append(time.time())
            del thought_times[:-50]
            del thoughts[:-50]
        if feeling_line and not FROZEN_FEELINGS:   # (experiment: held feelings – the FEEL line is ignored)
            with feelings_lock:
                before = dict(feelings)
                for k, v in re.findall(rf"({FEELING_NAMES})\s*[=:]\s*([01](?:\.\d+)?)",
                                       feeling_line, re.I):
                    feelings[k.lower()] = min(1.0, max(0.0, float(v)))
                note_jolt(before, feelings)
                settle_echoes(before, (f"this happened: {happened}" if events else f"you thought: {thought}"))
                pending = state.pop("unrated", None)
                if pending and time.time() - pending[0] < 90 and recent_moments \
                        and recent_moments[-1][3] == "thought" and recent_moments[-1][0] >= pending[0]:
                    _, size, word, _ = recent_moments[-1]
                    with memory_lock:  # how the answer felt = how his feelings moved right after it
                        talk = pending[1]
                        talk["feel"] = max(talk.get("feel", 0.0), size)
                        if not talk.get("feeling") and own_words:
                            talk["feeling"] = own_words
                        if word and size >= 0.5 and not talk.get("felt"):
                            talk["felt"] = word
                        for d in diary:
                            if d.get("time", "") >= talk["time"] and not d.get("feel") and not d["text"].startswith(
                                    ("I thought:", "I decided:", "I gave up on:", "I finished my goal:", "I took a nap")):
                                d["feel"] = size
                        save_memory()
                timings["feelings"].append(dict(feelings))  # (for the summary after a simulation)

        # ===== Step 2: do I want to say something? =====
        text, reason = "", ""
        if may_talk and one_step:
            answer2 = answer  # already answered in step 1
        elif may_talk:
            with feelings_lock:
                feelings_after = ", ".join(f"{k}={v:.2f}" for k, v in feelings.items())
            step2 = (
                "[Now you are deciding whether to say something out loud. The situation is the same as above.]\n"
                + (f"IMPORTANT – this just happened: {happened}\n" if events else "")
                + ("The human is here with you right now.\n" if human_is_here() else
                   "You are alone right now – nobody is here to hear you.\n" if is_alone(now) else
                   "The human is not in front of your camera right now.\n")
                + f"Your thought just now: {thought}\n"
                f"Your feelings: {feelings_after}\n"
                + said_before
                + "\nReply in EXACTLY this form:\n" + SAY_FORMAT.format(heard_or_not=heard_or_not)
            )
            try:
                # (the same conversation as step 1 – the brain has already read the situation)
                answer2 = follow_up(sent, answer, step2, temperature=TEMP_REPLY, max_tokens=70, stop=should_yield)
                sent, answer = sent + [{"role": "assistant", "content": answer},
                                       {"role": "user", "content": step2}], answer2
            except Interrupted:
                raise
            except Exception:
                answer2 = ""
        if may_talk:
            said = field("SAY", answer2) or ""
            if said and not re.search(r"(?i)speak\s*:?\s*no\b", answer2):
                if HAS_BODY and re.match(r"(?i)\s*ACT\s*:", said):   # (a move, not words – never said out loud)
                    run_act(re.sub(r"(?i)^\s*ACT\s*:\s*", "", said).splitlines()[0])
                    said = ""
                text = clean(said)
                if re.fullmatch(r"(?i)\W*(nothing|none|no)\W*", text):
                    text = ""
                reason = (field("WHY", answer2) or "").splitlines()[0] if field("WHY", answer2) else ""
                if re.fullmatch(r"(?i)\W*(nothing|none|no)\W*", reason):
                    reason = ""

        # His words were written BEFORE his body acted (thought, ACT and SAY come in one trip to the brain). If the
        # body did something, he hears what it reported before he speaks, and decides himself if his words still
        # fit (2026-11-06: "I've got it! I'm coming to you now!" while the body said "the cup is not here").
        said_check = None
        if text and act_result:
            holding = getattr(body, "holding", None)
            report = "\n".join(f"- '{x['step']}' → {x['result']}" for x in act_result)
            step_check = (
                "[Your body has now done what you chose. What it reported (facts):\n" + report + "\n"
                + (f"You are holding the {holding} in your beak." if holding else "Your beak is empty.") + "]\n"
                + f'You were about to say out loud: "{text}"\n'
                "Does that still fit what really happened? Say it, change it, or say nothing. Only say you did, have "
                "or carry something if the report above says so – walking up to a thing is not having it.\n"
                "Reply in EXACTLY this form:\n"
                "SAY: <what you say out loud now, or NONE>"
            )
            try:
                checked = field("SAY", follow_up(sent, answer, step_check, temperature=TEMP_REPLY, max_tokens=50,
                                                 stop=should_yield))
            except Interrupted:
                raise
            except Exception:
                checked = None
            if checked is not None:
                checked = clean(checked.splitlines()[0]) if checked.strip() else ""
                if re.fullmatch(r"(?i)\W*(nothing|none|no)\W*", checked):
                    checked = ""
                if checked != text:
                    said_check = {"before": text, "after": checked}
                    print(f"🔎 (heard what his body did – changed what he says: \"{text}\" → "
                          f"{chr(34) + checked + chr(34) if checked else 'nothing'})")
                    text = checked
                    if not text:
                        reason = ""

        if SHORT_OUTPUT and reason and len(reason.split()) > 8:
            reason = " ".join(reason.split()[:8]) + "…"
        status = (f"🗣️  wants to say something{' – ' + reason if reason else ''}" if text else
                  "🤐 decided to stay quiet" if may_talk else f"🤐 spoke {max(0, int(now - state['last_pip']))}s ago")
        mindful_now = mindful_raw.splitlines()[0].strip() if mindful_raw else ""
        if mindful_now and not re.fullmatch(r"(?i)\W*none\W*", mindful_now):
            print(f"🧭 (mindful of: {mindful_now})")
        print(f"\n💭 (thinking) {thought}   " + ("" if SHORT_OUTPUT else f"[{feelings_short()}]   ")
              + f"{status}   ⏱️ {time.time() - now:.0f}s")
        timings["thought"].append(time.time() - now)

        # ===== Step 3: goals – only every GOALS_EVERY:th thought =====
        state["thought_nr"] += 1
        new_goal = done = drop = None
        goal_events = []   # (for the thought log)
        if state["thought_nr"] % GOALS_EVERY == 0:
            with memory_lock:
                heard_lately = [f'({ago(now - datetime.fromisoformat(c["time"]).timestamp())}) "{c["human"]}"'
                                for c in conversations[-3:] if "time" in c
                                and now - datetime.fromisoformat(c["time"]).timestamp() < 900]
            with feelings_lock:
                goal_list = "\n".join(
                    f"{i}. {g['text']}" + (f"  (set {ago(time.time() - g['t'])})" if "t" in g else "")
                    for i, g in enumerate(goals, 1)) or "(none)"
                goal_count = len(goals)
            tried_now = tried_words(goals)   # B2: facts about what he tried – he decides what they mean for his goals
            step3 = (
                "[Now you are thinking about your goals.]\n"
                + "Your latest thoughts:\n- " + "\n- ".join(thoughts[-3:]) + "\n"
                + ("What the human said lately:\n- " + "\n- ".join(heard_lately) + "\n" if heard_lately
                   else "The human hasn't said anything lately.\n")
                + f"Your goals right now:\n{goal_list}\n"
                + (tried_now + "\n" if tried_now else "")
                + "\nA goal must be something you can really do with your senses: notice which objects are "
                "there, talk, ask the human, " + ("move around and explore with your body, " if HAS_BODY else "")
                + "or wait. Drop goals your senses can't reach.\n"
                "Reply in EXACTLY this form:\n"
                + ("GOAL: <a new small goal starting with 'I want to', or NONE>\n" if goal_count < MAX_GOALS
                   else "GOAL: NONE\n")
                + "DONE: <number of a goal that REALLY happened – you saw or heard it happen, "
                  "not just hoped or asked for it – or NONE>\n"
                + "DROP: <number of a goal you no longer want, or NONE>\n"
            )
            try:
                answer3 = follow_up(sent, answer, step3, temperature=TEMP_REPLY, max_tokens=70, stop=should_yield)
            except Interrupted:
                raise
            except Exception:
                answer3 = ""
            new_goal, done, drop = field("GOAL", answer3), field("DONE", answer3), field("DROP", answer3)
            done = re.match(r"#?(\d+)", done) if done else None
            drop = re.match(r"#?(\d+)", drop) if drop and not done else None

        with memory_lock:
            earlier_thoughts = [d["text"][11:] for d in diary[-30:] if d["text"].startswith("I thought: ")]
            if worth_remembering and not means_same(thought, earlier_thoughts[-10:], 0.85):
                feeling = own_words or strong_feeling()  # (in his own words, if he gave them)
                write_diary(f"I thought: {thought}" + (f" (I felt {feeling})" if feeling else ""))
            with feelings_lock:
                if done:
                    nr = int(done.group(1)) - 1
                    if 0 <= nr < len(goals):
                        finished = goals.pop(nr)["text"]
                        goal_events.append(["done", finished])
                        state["finished_goals"] = (state.get("finished_goals", []) + [finished])[-5:]
                        print(f"🏁 Goal done: {finished}")
                        write_diary(f"I finished my goal: {finished}")
                elif drop:
                    nr = int(drop.group(1)) - 1
                    if 0 <= nr < len(goals):
                        dropped = goals.pop(nr)["text"]
                        goal_events.append(["dropped", dropped])
                        state["finished_goals"] = (state.get("finished_goals", []) + [dropped])[-5:]
                        print(f"🫳 Gave up a goal: {dropped}")
                        write_diary(f"I gave up on: {dropped}")
                if new_goal and len(goals) < MAX_GOALS:
                    goal = new_goal.splitlines()[0].strip()
                    if not re.fullmatch(r"(?i)none.*", goal) and len(goal.split()) >= 3 \
                            and not any(same_fact(goal, g["text"]) for g in goals) \
                            and not means_same(goal, state.get("finished_goals", []), 0.85):
                        goals.append({"text": goal, "since": date.today().isoformat(), "t": time.time()})
                        goal_events.append(["new", goal])
                        print(f"🎯 New goal: {goal}")
                        write_diary(f"I decided: {goal}")
            save_memory()

        said_lately = [x for _, x in state["said"][-8:]]
        greeting = any("came back" in e or "woke up" in e for e in events)
        if text and not greeting and means_same(text, said_lately):  # a greeting may sound like an old one
            # The brain chooses itself: say something else, or stay quiet
            print(f"🔁 ({NAME} was about to repeat itself: {text})")
            try:
                other = ask_brain(talk_in_mind + [{"role": "user", "content":
                    f"{scene}\n" + (f"IMPORTANT – this just happened: {happened}\n" if events else "")
                    + f"Your thought: {thought}\n" + (f"Why you wanted to speak: {reason}\n" if reason else "")
                    + f"You were about to say out loud: \"{text}\" – but you already said "
                    "something that means the same:\n- " + "\n- ".join(said_lately[-4:])
                    + "\nSay something genuinely different instead, or reply NOTHING if you'd rather stay quiet."}],
                    temperature=0.9, max_tokens=50, no_memories=True, with_goals=False, stop=should_yield)
                move = re.search(r"(?i)\bACT\s*:\s*([^\n]*)", other or "") if HAS_BODY else None
                if move:
                    run_act(move.group(1))  # he chose to move instead of talking
                other = clean(other)
            except Interrupted:
                raise
            except Exception:
                other = ""
            text = "" if (not other or re.fullmatch(r"(?i)\W*nothing\W*", other)
                          or means_same(other, said_lately)) else other
            if not text:
                print(f"🤐 ({NAME} chose to stay quiet instead)")
        if text:
            if S.voice is not None:
                S.voice.preload()  # get the voice ready before Anka speaks
            # (text, a greeting?, when the thought started, when it was done)
            wants_to_say.put((text, any("came back" in e or "woke up" in e for e in events), now, time.time()))
        with feelings_lock:
            goals_now = [g["text"] for g in goals]
        trace("thought", thought=thought, repeated=False, feel_after={k: round(v, 2) for k, v in feelings.items()},
              speak=text or None, speak_reason=reason or None, said_check=said_check, may_talk=may_talk, echo=echo_text() or None,
              mindful=mindful_raw,
              goal_events=goal_events,
              goals=goals_now, goals_asked=state["thought_nr"] % GOALS_EVERY == 0,
              seconds=round(time.time() - now, 1), **common_log)


def _pictures_note():
    """A fact for KEEP_PICTURE (2026-11-10): how many pictures he kept today and how many his memory holds."""
    today = date.today().isoformat()
    with memory_lock:
        n = sum(1 for d in diary if d.get("picture") and d.get("date") == today)
        total = sum(1 for d in diary if d.get("picture"))
    return (f" (you have kept {n} picture{'s' if n != 1 else ''} today, {total} in all; your memory holds {MAX_PICTURES} "
            "pictures – when it is full, keeping a new one lets the oldest go)")


# ---- other parts of Anka (imported last, so the parts can use each other) ----
from anka.feelings import (FEELING_NAMES, clean_feeling, drift_feelings, feelings, feelings_lock, feelings_short, note_jolt, recent_moments, settle_echoes, strong_feeling, echo_text)
from anka.memory import (keep_picture, ideas, ago, comes_to_mind, conversations, diary, goals, how_long, human_name, lessons, memory_lock, remember, same_fact, save_memory, self_image, write_diary)
from anka.brain import (think_level, Interrupted, ask_brain, clean, cloud_is_on, field, follow_up, means_same, memories_shown)
from anka.trace import trace
from anka.eyes import brief_scene, current_speaker, human_is_here, picture_for_thought, scene_without_picture
from anka.face import EXPRESSIONS
from anka.voice import speaking
from anka.body import body, run_act
from anka.tried import tried_words
