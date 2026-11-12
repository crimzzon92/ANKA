"""Anka's day: wakes up, sees, listens, answers – and falls asleep."""
import difflib
import queue
import re
import threading
import time
import types
from datetime import date, datetime

from settings import (SIM_SOUND, BRAIN, CLOUD_BRAIN, CLOUD_KEY, CONVERSATION_MEMORY, HANDS_FREE, HAS_BODY, HEAR_WITH_CLOUD, MAX_CONVERSATIONS, NAME, OWN_LINES_MEMORY, SHORT_OUTPUT, SIM)
from anka.shared import (S, answering, awake, cv2, heard, listening, state, timings, to_remember, unsure_words, wake, wants_to_say)



def main():
    print("Loading models, this can take a minute the first time...")
    move_old_memory_files()
    load_memory()
    load_vectors()
    load_feeling_vectors()
    hold_frozen_feelings()   # (only in the feeling experiments: --frozen-feelings / --feelings k=v)
    with memory_lock:
        unsorted = [f for f in facts if f not in room_of]
    if unsorted:
        print(f"🏛️  Sorting {len(unsorted)} memories into rooms...")
        sort_into_rooms(unsorted)
    if SIM:
        print(f"🎬 Simulation: fake camera, fake voice, a script instead of you – but {NAME}'s REAL brain")
        pass  # (shared values live on S)
        S.faces_known = types.SimpleNamespace(names=["Alex", "Robin"])  # (pretend he was taught these two faces)
        if SIM_SOUND:      # (his real voice in the MuJoCo sim - the room window has a mute button)
            S.sound_on = False   # 2026-11-12: starts MUTED - click the button (or press M) to hear him
            load_voice()
    else:
        print("👀 Eyes: the brain looks at the camera picture itself + a light face/motion sensor")
        test_microphone()
        if HEAR_WITH_CLOUD and CLOUD_KEY:
            # the cloud brain listens itself – Whisper is only loaded if the cloud is away (saves ~0.5 GB and startup time)
            ears = LazyModel("ears", load_ears)
            print("👂 Ears: the cloud brain listens (Whisper waits as the reserve)")
        else:
            ears = LazyModel("ears", load_ears, load_ears())
        load_voice()
        threading.Thread(target=listen_loop, args=(ears,), daemon=True).start()
        if HANDS_FREE:
            threading.Thread(target=hands_free_loop, args=(ears,), daemon=True).start()
        threading.Thread(target=unload_loop, args=([ears, S.voice],), daemon=True).start()

    threading.Thread(target=saver_loop, daemon=True).start()
    conversation = []  # only the conversation with you
    own_lines = []     # what Anka said of its own accord

    if CLOUD_KEY:
        print(f"🧠 Brain: {CLOUD_BRAIN} in the cloud  ·  reserve on this Mac: {BRAIN}")
    else:
        print(f"🧠 Brain: {BRAIN} on this Mac" + ("  (cloud turned off)" if not CLOUD_BRAIN else "  (no cloud key)"))
    print("The duck is awake! Press ENTER to talk (in the terminal or the camera window). Ctrl + C = go to sleep.")
    awake.set()
    state["woke_up"] = time.time()
    thinker = threading.Thread(target=think_loop, daemon=True)
    thinker.start()

    try:
        camera = None if SIM else find_camera()
        for things in (sim_frames() if SIM else camera_frames(camera)):
            now = time.time()
            if update_sight(things, now):
                state["rest"] = 0
                wake.set()  # only now, when everything about the picture is done
            scene = state["scene"]

            # 1) Did you say something? Then Anka answers at once.
            answering.clear()  # the previous answer is done
            try:
                human_said = heard.get_nowait()
                answering.set()  # Anka answers now – no new thoughts, and a running thought is dropped
            except queue.Empty:
                human_said = None

            ready_answer, tone = None, None
            if isinstance(human_said, tuple):  # (words, how they were said) – from the simulation
                human_said, tone = human_said
            if isinstance(human_said, HeardSound):
                sound = human_said.sound
                result = hear_and_answer(sound, conversation)
                if result is None:  # the cloud couldn't – Whisper takes over
                    transcribe = ears.get()
                    with heavy():  # (one heavy AI at a time, like everywhere else)
                        human_said = transcribe(sound) if transcribe else ""
                    if not human_said:
                        print("🎤 (couldn't make out any words)")
                        continue
                else:
                    human_said, tone, ready_answer, asked_at_cloud = result
                    if not human_said:
                        print("🎤 (the cloud heard no words)")
                        continue

            earlier = own_lines + [m["content"] for m in conversation if m["role"] == "assistant"]
            # The duck goes quiet before the mic listens, so its own voice can only sneak in if it was talking
            # when you pressed ENTER – and then the words are almost exactly the same. Repeating it yourself is fine.
            if human_said and state.pop("voice_was_on", False) and any(
                    difflib.SequenceMatcher(None, human_said.lower(), old.lower()).ratio() > 0.85
                    for old in earlier[-2:]):
                print(f"\n🔇 (ignored – that was the duck's own voice: {human_said})")
                continue

            if human_said and re.search(r"\b(go to sleep|good ?night)\b", human_said, re.IGNORECASE):
                print(f"\n🧑 You: {human_said}")
                state["last_human"] = time.time()  # no new thoughts now
                night = reply_to_event("You are going to sleep now. Say good night in your own way.",
                                       human_said, "Good night!")
                print(f"🦆 {night}")
                say(night)
                break

            if human_said and re.search(r"forget (everything|all)", human_said, re.IGNORECASE):
                print(f"\n🧑 You: {human_said}")
                forget_everything()
                conversation = []
                answer = reply_to_event("Your whole memory was just erased. You don't know the human anymore.",
                                        human_said, "Okay, I forgot everything.")
                print(f"🦆 {answer}")
                say(answer)
                continue

            if human_said:
                state["last_human"] = now
                state["rest"] = 0  # you're talking – Anka is awake again
                wake.set()

            # 2) Has Anka thought of something it wants to say? Say it if it fits right now.
            if not human_said:
                if not wants_to_say.empty() and not speaking() and not listening.is_set():
                    text, greeting, started, finished = wants_to_say.get()
                    # Only drop it if you talked while Anka was thinking, or if it waited in the queue too long.
                    # (Thinking itself can take long on an M1 when the brain also looks at a picture.)
                    if started < state["last_human"]:
                        print(f"🔇 (skipped – you spoke while I was thinking that: {text})")
                    elif now - finished > 20:
                        print(f"🔇 (skipped – waited too long to say it: {text})")
                    elif not greeting and means_same(text, [x for _, x in state["said"][-8:]]):
                        print(f"🔇 (skipped – means the same as something I already said: {text})")
                    else:
                        print(f"🦆 {text}")
                        timings["own"] += 1
                        trace("own_line", text=text, waited=round(now - finished, 1))
                        say(text)
                        state["said"] = (state["said"] + [(now, text)])[-20:]
                        own_lines = (own_lines + [text])[-OWN_LINES_MEMORY:]
                        # (not added to the conversation – a small brain starts copying its own remarks and drifts;
                        #  the latest one is mentioned when you answer, see below)
                        state["last_own"] = (now, text)
                        with memory_lock:  # kept until tonight's sleep, so he can look back on how it went
                            own_said.append({"time": datetime.now().isoformat(timespec="seconds"), "text": text})
                            del own_said[:-100]
                            save_memory()
                        state["last_pip"] = now
                continue

            # 3) Answer you
            print(f"\n🧑 You: {human_said}" + (f"   (sounded: {tone})" if tone else ""))
            asked_at = asked_at_cloud if ready_answer else time.time()
            picture = picture_for_reply(human_said)
            # (what caused a big feeling, if this answer brings one – told to him while it fades)
            state["feel_cause"] = (time.time(), f'{current_speaker() or human_name() or "the human"} said to you'
                                   + (f' ({tone})' if tone else '') + f': "{human_said[:100]}"')
            if picture is None and not SIM:
                scene = scene_without_picture()
            if SHORT_OUTPUT and SIM:
                print(f"👀 {state.get('sees_short', 'nothing')}")
            elif SHORT_OUTPUT:
                faces = state["faces"]
                print(f"👀 {faces} face{'s' if faces != 1 else ''}"
                      + (" · looking at the picture" if picture is not None else ""))
            else:
                print(f"👀 Right now: {scene[1:-1]}")
            if ready_answer:
                pass  # the cloud already heard you and answered
            elif asks_about_sight(human_said):
                # Small models trust what comes last – so the camera comes after the question
                question = (f'The human says to you: "{human_said}"\n{scene}\n'
                            "Answer ONLY from this camera view. It is the truth right now – even if "
                            "earlier in the conversation there were other people or things.")
                messages = [{"role": "user", "content": question}]  # only the camera – the talk before can mislead
                search = None  # don't look in old memories – they may be out of date
            else:
                own_t, own_text = state.get("last_own", (0, ""))
                # only if you answer right after it – otherwise Anka "answers" its own old question
                before = speaker_note() + (f'(Just before this, you had said to the human: "{own_text}")\n'
                          if own_text and now - own_t < 25 else "")
                back_t, back_gone = state.get("came_back", (0, ""))
                if back_gone and now - back_t < 120:
                    who = human_name() or "The human"
                    before = f"({who} was away for {back_gone} and came back {ago(now - back_t)}.)\n" + before
                if state.pop("unsure_hearing", False):
                    before += ("(Your ears weren't sure about these words – you may have misheard. If what you "
                               "heard doesn't make sense, ask instead of guessing.)\n")
                if re.search(r"(?i)\b(say (that |it )?again|repeat|what did you (just )?say|pardon|come again)\b",
                             human_said) and state["said"]:
                    before += f'(The last thing you said out loud was: "{state["said"][-1][1]}")\n'
                # words Anka has used in many of its latest lines – a small brain otherwise gets stuck on them
                counts = {}
                for _, line in state["said"][-6:]:
                    for w in set(re.findall(r"[a-z]{4,}", line.lower())) - {"that", "this", "with", "your", "have",
                                                                             "what", "just", "here", "there"}:
                        counts[w] = counts.get(w, 0) + 1
                worn = [w for w, n in counts.items() if n >= 3]
                if worn:
                    before += f"(You have said these words a lot lately – use other words: {', '.join(worn)})\n"
                question = (f'{brief_scene() if SIM else scene}\n{before}The human says to you'
                            + (f' ({tone})' if tone else '') + f': "{human_said}"')
                messages = conversation[-CONVERSATION_MEMORY:] + [{"role": "user", "content": question}]
                search = human_said
            state.pop("act", None)
            words = 90 + (25 if HAS_BODY else 0)  # (room for an ACT line)
            try:
                if ready_answer:
                    answer = ready_answer
                    raise StopIteration
                answer = clean(feel_now(ask_brain(messages, search=search, max_tokens=words, reply_to_human=True,
                                                  picture=picture)))
                # Too much like something Anka already said? Try again, a bit wilder.
                attempts = 0
                while too_similar(answer, earlier) and attempts < 2:
                    attempts += 1
                    answer = clean(feel_now(ask_brain(messages, temperature=1.0, avoid=earlier[-6:], search=search,
                                                      max_tokens=words, reply_to_human=True, picture=picture)))
            except StopIteration:
                pass
            except Exception as error:
                print("Could not reach Ollama:", error)
                continue

            # Anka speaks first – everything else (saving, remembering) waits until the voice is going
            timings["reply"].append(time.time() - asked_at)
            print(f"🦆 {answer}" + (f"   ⏱️ {timings['reply'][-1]:.1f}s" if not SHORT_OUTPUT or SIM else ""))
            if answer and not listening.is_set():
                say(answer)
                state["said"] = (state["said"] + [(now, answer)])[-20:]
            reply_act, reply_act_result = state.get("act") or "", None
            if state.get("act"):
                reply_act_result = run_act(state.pop("act"))  # he answers first, then his body does what he decided
            with feelings_lock:
                feel_after = {k: round(v, 2) for k, v in feelings.items()}
            trace("reply", human=human_said, tone=tone, answer=answer, act=reply_act, act_result=reply_act_result,
                  memories=memories_shown(), feel_after=feel_after, seconds=round(timings["reply"][-1], 1))
            conversation += [{"role": "user", "content": f'The human says to you'
                                                         + (f' ({tone})' if tone else '') + f': "{human_said}"'},
                             {"role": "assistant", "content": answer}]
            conversation = conversation[-CONVERSATION_MEMORY:]
            with memory_lock:
                speaker, feel, felt = current_speaker(), moment_strength(since=asked_at), strong_feeling(since=asked_at)
                ft = state.get("feeling_text", (0, None, None))
                own_feeling = ft[1] if ft[0] >= asked_at and ft[2] == "reply" else None
                seen = list(dict.fromkeys(state.get("who", []))) + ["someone he doesn't know"] * state.get("strangers", 0)
                entry = {"date": date.today().isoformat(), "time": datetime.now().isoformat(timespec="seconds"),
                         "human": human_said, "duck": answer, "feel": feel,
                         **({"felt": felt} if felt else {}),
                         **({"feeling": own_feeling} if own_feeling else {}),
                         **({"who": speaker} if speaker else {}),
                         **({"seen": seen} if seen else {}),
                         **({"unsure": True} if human_said in unsure_words else {})}
                conversations.append(entry)
                if not any(t >= asked_at and src == "reply" for t, _, _, src in recent_moments):
                    state["unrated"] = (time.time(), entry)  # the brain forgot to say how it felt – his next thought tells
                del conversations[:-MAX_CONVERSATIONS]
                save_memory()
            # gone through when you pause – so answers are faster
            to_remember.put((human_said, answer, speaker, feel, felt, own_feeling))
            state["recent_talk"] = conversation
            state["last_pip"] = now
    except KeyboardInterrupt:
        pass

    awake.clear()  # stop the thinking loop
    try:  # going to sleep: the body stops at once, it must not go on walking while he dreams
        import anka.body as _body_module
        if getattr(_body_module, "body", None) is not None and hasattr(_body_module.body, "halt"):
            _body_module.body.halt()
    except Exception:
        pass
    thinker.join(timeout=120)  # wait until a running thought is done before Anka falls asleep
    if not SIM:
        cv2.destroyAllWindows()
    try:
        sleep_and_dream()
    except KeyboardInterrupt:
        print("\n⏰ Woke up before the dream was finished.")
    finally:
        save_now()  # everything that changed is on the disk before Anka shuts down
    if S.speech is not None:
        S.speech.wait()  # let "Good night" finish


# ---- other parts of Anka (imported last, so the parts can use each other) ----
from anka.resources import LazyModel, heavy, unload_loop
from anka.feelings import feel_now, feelings, feelings_lock, hold_frozen_feelings, moment_strength, recent_moments, strong_feeling
from anka.trace import trace
from anka.memory import (ago, conversations, facts, forget_everything, human_name, load_feeling_vectors, load_memory, load_vectors, memory_lock, move_old_memory_files, own_said, room_of, save_memory, save_now, saver_loop, sort_into_rooms)
from anka.brain import ask_brain, clean, means_same, memories_shown, reply_to_event, too_similar
from anka.night import sleep_and_dream
from anka.eyes import (asks_about_sight, brief_scene, camera_frames, current_speaker, find_camera, picture_for_reply, scene_without_picture, speaker_note, update_sight)
from anka.thinking import think_loop
from anka.voice import load_voice, say, speaking
from anka.ears import HeardSound, hands_free_loop, hear_and_answer, listen_loop, load_ears, test_microphone
from anka.simulation import sim_frames
from anka.body import run_act
