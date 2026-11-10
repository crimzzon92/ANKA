# run_anka.py – starts the brain of Anka, the robot duck (a Microduck)
#
# Camera + Qwen see -> you talk (Whisper) -> Qwen thinks -> Kokoro speaks.
# Press ENTER (in the terminal or the camera window) to talk to Anka.
# Anka also thinks quietly by itself all the time and sometimes says something of its own accord.
# Principle: the code gives Anka senses and facts – the brain (the LLM) makes the decisions.
# Ctrl + C (or say "go to sleep") = Anka falls asleep, tidies its memory, dreams and shuts down.
#
#   python run_anka.py                 run Anka
#   python run_anka.py --simulate      simulation: fake camera, mic and voice, a script instead of you
#                                 (real brain, separate memory, log saved to simulation_log.txt)
#   python run_anka.py --simulate --body   test day with a virtual body in a 2D room (a window shows the room from above)
#   python run_anka.py --tidy          test the night-time memory tidy-up without saving anything
#   python run_anka.py --learn-face Robin   teach Anka a face (--faces = who he knows, --forget-face Robin = forget it)
#   python run_anka.py --dream         a full night's sleep with a dream at the end, even if he dreamt a moment ago
#   python run_anka.py --microphones   list the microphones
#   python run_anka.py --voice-test    hear Anka's voice (or a new mix: --voice-test "af_bella*0.5 + am_puck*0.5" 1.2)
#   python run_anka.py --record-voice  record a short voice sample for VOICE_ENGINE = "clone"


# The code is split into parts: settings.py (what you may want to change) and the anka/ folder.

import json
import math
import os
import sys
from datetime import date, datetime

import numpy as np

from settings import (BRAIN, BRAIN_NAME, CLOUD_KEY, FEELING_VECTOR_FILE, KOKORO_VOICE, MEMORY_FILE, NAME, ROOM_MAP_FILE, SIM, SIM_LOG_FILE, VECTOR_FILE, VOICE_PITCH)
from settings import (BODY, FROZEN_FEELINGS, LIFE, MUJOCO, NIGHT_CLOUD_BRAIN, RUN_CONDITION, RUN_NAME, SCENARIO, SET_FEELINGS, SIM_SCRIPT)
from anka.trace import trace
from anka.shared import S, cv2, timings
from anka.feelings import FEEL_LABELS
from anka.memory import (conversations, diary, facts, feeling_lock, feeling_vecs, lessons, load_feeling_vectors, load_memory, load_vectors, memory_lock, move_old_memory_files, old_facts, save_json, self_image)
from anka.brain import brain_speed
from anka.night import tidy_up
from anka.eyes import (FaceMemory, learn_face)
from anka.voice import voice_test
from anka.ears import (mic_info, microphone, record_voice)
from anka.simulation import (Tee, sim_past)
from anka.main import main
from anka.body import body



def keep_test_log(started):
    """Keeps a copy of this test's log (and the map of his walk) in test_logs/, named by when the test started –
    the normal files are overwritten by the next test."""
    import shutil
    folder = os.path.join(os.path.dirname(SIM_LOG_FILE), "test_logs")
    try:
        os.makedirs(folder, exist_ok=True)
        stamp = started.strftime("%Y-%m-%d_%H%M")
        name = os.path.splitext(os.path.basename(SIM_LOG_FILE))[0]
        shutil.copy2(SIM_LOG_FILE, os.path.join(folder, f"{name}_{stamp}.txt"))
        if os.path.exists(ROOM_MAP_FILE) and os.path.getmtime(ROOM_MAP_FILE) >= started.timestamp():
            shutil.copy2(ROOM_MAP_FILE, os.path.join(folder, f"room_map_{stamp}.png"))
        print(f"📁 A copy is kept in {folder}")
    except OSError as error:
        print("📁 Couldn't keep a copy of the log:", error)

if __name__ == "__main__":
    if "--microphones" in sys.argv:
        print("Microphones:")
        try:
            microphone.ask("TEST", 10, show_list=True)
            print(f"\nPip uses: {mic_info['name']}  (change with MICROPHONE at the top of run_anka.py)")
        except Exception as error:
            print("🎤", error)
    elif "--voice-test" in sys.argv:
        args = sys.argv[sys.argv.index("--voice-test") + 1:]
        voice_test(args[0] if args else KOKORO_VOICE, float(args[1]) if len(args) > 1 else VOICE_PITCH,
                   float(args[2]) if len(args) > 2 else None, float(args[3]) if len(args) > 3 else None,
                   float(args[4]) if len(args) > 4 else None)
    elif "--record-voice" in sys.argv:
        record_voice()
    elif "--learn-face" in sys.argv[:-1]:
        learn_face(sys.argv[sys.argv.index("--learn-face") + 1].strip().capitalize())
    elif "--forget-face" in sys.argv[:-1]:
        name = sys.argv[sys.argv.index("--forget-face") + 1].strip().capitalize()
        memory = FaceMemory()
        keep = [i for i, n in enumerate(memory.names) if n != name]
        if len(keep) == len(memory.names):
            print(f"👤 {NAME} doesn't know a face called {name}.")
        else:
            memory.names, memory.vectors = [memory.names[i] for i in keep], memory.vectors[keep]
            memory.save()
            print(f"👤 Forgot {name}'s face.")
    elif "--faces" in sys.argv:
        names = FaceMemory().names
        print("👤 Faces " + NAME + " knows: " + (", ".join(f"{n} ({names.count(n)} pictures)" for n in dict.fromkeys(names))
                                               if names else "none yet – teach one with: python run_anka.py --learn-face Alex"))
    elif "--tidy" in sys.argv:
        # Test the tidy-up on your real memory – without saving anything
        move_old_memory_files()
        load_memory()
        load_vectors()
        load_feeling_vectors()
        tidy_up(dry_run=True)
    elif SIM:
        def _shift_life_dates(path, days_lived):
            """Life mode runs several of his 'days' on the same real day. Without this, yesterday's memories are dated
            TODAY on day 2 (and the alone-memories skip everything from today). So: if his last day was lived on the same
            real day, every date in his memory moves one day back – his last day becomes 'yesterday'. (2026-11-09)"""
            import re as _re
            from datetime import date as _date, timedelta as _td
            if not days_lived:
                return
            gap = (_date.today() - _date.fromisoformat(days_lived[-1][:10])).days
            if gap >= 1:
                return
            shift = _td(days=-(1 - gap))
            iso = _re.compile(r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?")
            def move(x, key=None):
                if isinstance(x, dict):
                    return {k: move(v, k) for k, v in x.items()}
                if isinstance(x, list):
                    return [move(v, key) for v in x]
                if isinstance(x, str) and iso.fullmatch(x):
                    if len(x) == 10:
                        return (_date.fromisoformat(x) + shift).isoformat()
                    return (datetime.fromisoformat(x) + shift).isoformat(timespec="seconds")
                if key == "t" and isinstance(x, (int, float)) and 1.6e9 < x < 2.2e9:
                    return x + shift.total_seconds()
                return x
            with open(path, encoding="utf-8") as f:
                mem = json.load(f)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(move(mem), f, indent=2, ensure_ascii=False)
            print(f"🌱 (his last day was lived earlier today – his memories are moved one day back, so it was 'yesterday')")

        life_days_file = MEMORY_FILE.replace(".json", "_days.json")   # life mode: which runs (days) he has lived
        if LIFE and os.path.exists(MEMORY_FILE) and "--newlife" not in sys.argv:
            life_days = json.load(open(life_days_file)) if os.path.exists(life_days_file) else []
            print(f"🌱 LIFE MODE: {NAME} goes on with his own life (day {len(life_days) + 1}) – his memory is kept")
            _shift_life_dates(MEMORY_FILE, life_days)
        else:
            for path in (MEMORY_FILE, VECTOR_FILE, MEMORY_FILE + ".bak", VECTOR_FILE + ".bak",
                         FEELING_VECTOR_FILE, FEELING_VECTOR_FILE + ".bak", life_days_file):
                if os.path.exists(path):
                    os.remove(path)  # every simulation starts fresh (life mode: only the first day / --newlife)...
            if "--newborn" not in sys.argv:
                save_json(MEMORY_FILE, json.dumps(sim_past(), indent=2, ensure_ascii=False))  # ...but with a past
            life_days = []
            if LIFE:
                print(f"🌱 LIFE MODE: a new life starts (day 1) – from now on his memory is kept between runs")
        if LIFE:
            with open(life_days_file, "w") as f:
                json.dump(life_days + [datetime.now().isoformat(timespec="seconds")], f)
        from anka.memory import ideas as _ideas
        trace("start", run=RUN_NAME, scenario=SCENARIO or None, condition=RUN_CONDITION, life=LIFE,
              life_day=len(life_days) + 1 if LIFE else None, frozen_feelings=FROZEN_FEELINGS,
              set_feelings=SET_FEELINGS, brain=BRAIN_NAME, night_brain=NIGHT_CLOUD_BRAIN,
              body="mujoco" if MUJOCO else "2d" if BODY else None, script_seconds=SIM_SCRIPT[-1][0],
              args=sys.argv[1:])
        run_cost = None
        with open(SIM_LOG_FILE, "w", encoding="utf-8") as log:
            sys.stdout = Tee(sys.__stdout__, log)
            started = datetime.now()
            print(f"🎬 Simulation started {datetime.now():%Y-%m-%d %H:%M}  (brain: {BRAIN_NAME}" + (f", reserve: {BRAIN})" if CLOUD_KEY else ")"))
            try:
                main()
                t, a = timings["thought"], timings["reply"]
                print(f"\n===== {BRAIN_NAME} =====\n"
                      f"answers: {len(a)}, average {np.mean(a) if a else 0:.1f}s, slowest {max(a) if a else 0:.1f}s\n"
                      f"thoughts: {len(t)}, average {np.mean(t) if t else 0:.1f}s, slowest {max(t) if t else 0:.1f}s\n"
                      f"{NAME} spoke by itself {timings['own']} times")
                if timings["faces"]:
                    print("faces he chose: " + ", ".join(f"{k} {v}×" for k, v in
                                                         sorted(timings["faces"].items(), key=lambda kv: -kv[1])))
                if timings["feelings"]:
                    print("feelings during the day (lowest – highest):  " + "  ·  ".join(
                        f"{FEEL_LABELS.get(k, k)} {min(f[k] for f in timings['feelings']):.2f}–"
                        f"{max(f[k] for f in timings['feelings']):.2f}" for k in timings["feelings"][0]))
                if "--full" in sys.argv:
                    with memory_lock:
                        print("\n===== What he has after the day =====")
                        print("faces he knows: " + ", ".join(dict.fromkeys(S.faces_known.names)))
                        print("facts:\n  " + "\n  ".join(facts))
                        print("no longer true:\n  " + ("\n  ".join(o["text"] for o in old_facts) or "(none)"))
                        today = date.today().isoformat()
                        print("today's diary (not quiet thoughts):\n  " + "\n  ".join(
                            d["text"] + (f"  [felt {d['feel']}]" if d.get("feel") else "") for d in diary
                            if d["date"] == today and not d["text"].startswith("I thought:")))
                        print("strongest moments in conversation:\n  " + "\n  ".join(
                            f'{c.get("feel", 0):.2f}  "{c["human"]}" → "{c["duck"]}"'
                            + (f'\n        it felt: {c["feeling"]}' if c.get("feeling") else "")
                            + (f'  (at first: {c["first_feeling"]})' if c.get("first_feeling") else "")
                            for c in sorted(conversations, key=lambda c: -c.get("feel", 0))[:4] if c["date"] == today))
                        healed = [x for x in conversations + diary if x.get("first_feeling")]
                        print("feelings that changed overnight:\n  " + ("\n  ".join(
                            f'{x.get("human") or x.get("text")}: {x["first_feeling"]} → {x["feeling"]}' for x in healed)
                            or "(none)"))
                        with feeling_lock:
                            fv = np.array(list(feeling_vecs.values())) if len(feeling_vecs) > 2 else None
                        if fv is not None:
                            sims = (fv @ fv.T)[~np.eye(len(fv), dtype=bool)]
                            print(f"feeling space: {len(fv)} feelings, how alike two of them are: "
                                  f"min {sims.min():.2f} · median {np.median(sims):.2f} · max {sims.max():.2f}")
                        with_words = [c for c in conversations if c["date"] == today and c.get("feeling")]
                        print(f"conversations with a feeling in words: {len(with_words)} of "
                              f"{sum(1 for c in conversations if c['date'] == today)} · feelings in the feeling space: "
                              f"{len(feeling_vecs)}")
                        print("who he is:\n  " + "\n  ".join(self_image))
                        print("lessons:\n  " + ("\n  ".join(lessons) or "(none)"))
                if body is not None:
                    print(f"\n===== His body =====\nhe walked {body.walked:.1f} m and did {len(body.actions)} things"
                          + (f" (fell over {body.falls} times)" if getattr(body, "falls", 0) else "") + ":")
                    if hasattr(body, "landmarks"):
                        err = math.hypot(body.est_x - body.x, body.y - body.est_y)
                        print(f"where he thought he was at the end: {err * 100:.0f} cm off (landmarks corrected him "
                              f"{body.fixes} times; he knows {len(body.landmarks)} landmarks)")
                        if hasattr(body, "walls"):
                            facing = ((body.est_heading() - body.heading + 180) % 360) - 180
                            ref = getattr(body, "wall_ref", None)
                            print(f"which way he thought he faced: {facing:+.1f}° off · the room's main direction he decided: "
                                  + (f"{((ref + 45) % 90) - 45:+.1f}° (0 = right)" if ref is not None else "not yet")
                                  + f" · fixed walls: {len(body.walls)}")
                        from anka.memory import places
                        if getattr(body, "depth_ai_errors", None):
                            e = np.array(body.depth_ai_errors)
                            print(f"depth AI: {len(e)} guesses, typically {np.median(e) * 100:.0f} % off "
                                  f"(90 % of them within {np.percentile(e, 90) * 100:.0f} %) compared to the real distances")
                        if getattr(body, "depth_ai_errors_people", None):
                            e = np.array(body.depth_ai_errors_people)
                            print(f"   + {len(e)} guesses with a person standing close in view: typically {np.median(e) * 100:.0f} % off "
                                  "(rough measure there – the 'true' distance is one pixel column, which can look between the legs)")
                        print("where things are, kept for tomorrow: " + "; ".join(f"{k} {v['where']}" for k, v in places.items()))
                    if hasattr(body, "report") and not hasattr(body, "landmarks"):
                        try:
                            for line in body.report():
                                print(line)
                        except Exception as error:
                            print("(could not make the mind report:", error, ")")
                    for chose, happened in body.actions:
                        print(f"  {chose}  →  {happened}")
                    try:
                        cv2.imwrite(ROOM_MAP_FILE, body.draw(live=False, caption=f"{NAME} walked {body.walked:.1f} m today "
                                                                     f"(the yellow line is his path)"))
                        print(f"🗺️  The map of his walk: {ROOM_MAP_FILE}")
                    except Exception as error:
                        print("🗺️  Couldn't save the map:", error)
                    try:
                        from settings import ROOM_3D
                        if ROOM_3D and hasattr(body, "cubes_3d"):
                            from anka.room3d import save_3d
                            room_3d = os.path.join(os.path.dirname(ROOM_MAP_FILE), "room_3d.html")
                            words, _ = save_3d(room_3d, body)
                            print(f"🧊 His 3D picture of the room: {room_3d} – {words}")
                    except Exception as error:
                        print("🧊 Couldn't save the 3D picture:", error)
                    try:
                        cv2.destroyAllWindows()
                    except Exception:
                        pass
                try:
                    from anka.brain import cloud_cost_words
                    lines, total = cloud_cost_words()
                    run_cost = total
                    if lines:
                        print("💰 cloud brains this run: " + " | ".join(lines) + f" — in all ≈ ${total:.2f}")
                except Exception as error:
                    print("💰 (couldn't count the cost:", error, ")")
                try:
                    from anka.memory import goals
                    with memory_lock:
                        trace("end", cost=run_cost, answers=len(a), thoughts=len(t), own_lines=timings["own"],
                              walked=round(body.walked, 1) if body is not None else None,
                              actions=len(body.actions) if body is not None else None,
                              self_image=list(self_image), lessons=list(lessons), ideas=[x["text"] for x in _ideas],
                              goals=[g["text"] for g in goals], facts=len(facts), diary=len(diary),
                              conversations=len(conversations),
                              body3d=getattr(body, "summary_numbers", None),
                              pictures_kept=sum(1 for d in diary if d.get("picture") and d.get("date") == date.today().isoformat()),
                              pictures_recalled=len(getattr(__import__("anka.brain", fromlist=["x"]), "_picture_events", [])))
                except Exception as error:
                    print("📈 (couldn't write the end of the thought log:", error, ")")
                if brain_speed:
                    read_n, read_s, write_n, write_s, load_s = (np.array(x) for x in zip(*brain_speed))
                    print(f"brain: {len(brain_speed)} requests, each read {read_n.mean():.0f} tokens on average "
                          f"({read_n.sum() / max(read_s.sum(), 1e-9):.0f} tokens/s) and wrote {write_n.mean():.0f} "
                          f"({write_n.sum() / max(write_s.sum(), 1e-9):.1f} tokens/s)\n"
                          f"time spent: reading {read_s.sum():.0f}s · writing {write_s.sum():.0f}s · "
                          f"loading the brain {load_s.sum():.0f}s")
            finally:
                sys.stdout = sys.__stdout__
        print(f"\n🎬 Done! Everything is saved in {SIM_LOG_FILE}")
        keep_test_log(started)
    else:
        main()
