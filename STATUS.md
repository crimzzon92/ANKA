# Status – what works, how well, and what's next

*Last updated: 10 November 2026.*

Everything below was measured in simulation: the 2D room and Pollen's MuJoCo simulator. **Nothing has run on a real
Microduck yet**, so treat every number as "works in the simulator". Most comparisons are 3 runs against 3 runs, so a
difference is a strong hint, not proof. The language model answers differently every time.

The tool that does the measuring is `tools/score_run.py`. It reads the thought traces (`test_logs/trace_*.jsonl`) and
counts what he actually did: actions, failures, repeats, goals, feelings, memory use, cost. No AI judges anything.

This page is the current state. The history, including the experiments that failed and what they taught us, is in
[docs/JOURNAL.md](docs/JOURNAL.md).

---

## ✅ Done and working

### Thinking and feeling

| Feature | How well (measured) |
|---|---|
| **Thinks on his own** (thought → feeling → act → speak, every 4–16 s) | Stable through all runs. About $0.15–0.25 per 10 simulated minutes with `gemini-3-flash-preview` (pictures included). |
| **Speaking is optional** (most thoughts stay in his head) | Alone, he says about 60 % fewer things out loud (2–9 lines in 5 min instead of 14–17). With people around he is as chatty as before. |
| **Feelings that fade slowly** ("after-feelings") | 2 minutes after an apology his nervousness was 0.70, against 0.09 without the feature (no overlap between runs). His thoughts stay careful ("Alex is nice again, but I'm still shaky"). His actions changed only a little: he was a bit quieter and kept more distance. |
| **Do the feelings change what he does?** | When he is alone and nothing happens, **yes**. With loneliness held high he walked 11–16 m and explored once. With it held low he walked 19–27 m and explored 3 times (no overlap). When something happens (a shout), the event itself decides his first reaction, not the feeling numbers. |
| **Expectation and surprise** | He states an expectation on 93–98 % of his actions. Big surprises go into his diary, and after one, his next action is different 78–100 % of the time. Over three days in one life his surprise went from 0.28 to 0.11 to 0.04: he learned to predict better. |
| **Wiser brain when stuck** (2 misses in a row → `gemini-4-argon` for that one thought) | He solved "the ball is stuck in the corner" the same day (he carried it instead of kicking it) in 3 of 3 runs, against 2 of 8 with only the day brain. It costs a little more per day: the bigger model is used only for those thoughts. |
| **Knowing when he's alone** | He no longer thinks someone has left just because they are out of view. Only a departure he heard counts (footsteps, the door). |

### Memory

| Feature | How well |
|---|---|
| **Memory by meaning** (facts, diary, conversations, with the feeling of each) | Works. Strong moments come back more easily. |
| **Habituation** (a memory that just came up pulls weaker for a while) | The most-repeated memory went from 28 of 30 thoughts to 7–12 of 22–35. Variety of memories went up from 0.19 to 0.27–0.40. |
| **Memories from earlier days** (life mode: several days in a row) | 46–84 % of the memories shown come from earlier days, instead of 0–19 %. He says things like "I remember my red ball was near the sofa yesterday." |
| **Picture memories** (he chooses to keep what he sees; an old picture comes back next to the current one) | Works. He keeps about 10 pictures in a 17-minute day once he is told how many he has kept. |

### Nights

| Feature | How well |
|---|---|
| **Night review** | Mistakes are judged only from what the day showed. Wrongly deleted memories went from 3 to 0 over three nights. A promise or a plan is never treated as a mistake. |
| **Self-image from his body's track record** | His beliefs about himself now match what his body really did. He stopped believing "I can climb the bookshelf" after a day of failing to. |
| **Lessons and ideas** | Untested ideas ("Maybe I could carry the ball…") are kept apart from lessons. Every time a night wrote an idea, he tried it the next day (2 of 2). When it was the right idea, the ball was delivered in 18 s. **But this is not yet a proven gain in performance:** in the ball transfer test the group *without* experience was faster (13 s against 17–133 s), and with ideas the results overlap with the control. He only learns from what he actually tried, and a success the night doesn't write down is lost. Details in the [journal](docs/JOURNAL.md#-ball-transfer-test-the-control-group-was-faster). |
| **Dreams** | Written by the night model from the day's strongest moments and the pictures he kept. |

### Body: 2D simulator
- He walks, turns, looks, picks up and carries, kicks, sits, explores, and runs errands ("bring me the sock").
- **His own map:** step counting, an EKF with 5 landmarks, a floor plan from his depth sensor, and fixed walls. At the end of a test his position is typically 3–12 cm off.

### Body: MuJoCo simulator (Pollen's real duck software)
- **Walking, turning, looking, sitting, exploring:** works. His position at the end of a 17-minute day is 5–13 cm off.
- **The living body between thoughts:** before it, he stood completely still 97 % of the time between his own moves. Now it is 20–25 %, and never more than a few seconds in a row. It does glances, head tilts, weight shifts, breathing, small turns of the body towards where he looks, and his eyes follow a ball that rolls.
- **Feelings in the body (the `BODY` line):** he writes it on almost every thought and reply. When shouted at he wrote "Crouched low, head tucked down, trembling", and the body did that.
- **Pollen's 13 emotions** (sad, laugh, yes, no, excited …), a peck (bending down), and 7 duck sounds: all of them are his own choice, as actions.
- **His real voice in the simulator,** with a mute button in the room window.

---

## 🔧 Should be fixed

- **The kick in MuJoCo has not hit the ball yet.** Pollen's kick is blind and needs the ball 9 cm in front of the kicking foot. The step-up to the ball is still being tuned: very short steps don't make the walking policy move at all, and longer ones push the ball away.
- **No beak motor in the simulated duck,** so in 3D he can peck but not carry anything. The real duck has one.
- **He talks a lot when people are around:** about 16–19 lines of his own in 5–6 minutes. A lesson about it exists, he is reminded of it, and it barely changes his behaviour.
- **He chooses a lot of duck sounds** (30–40 in 17 minutes). Telling him the count worked for picture memories; it may be needed here too.
- **`BODY` words are read with word lists.** The vocabulary is limited ("tail wagging" means nothing to a body without a tail). When he sits, only the head can show his feelings.
- **The reserve brain** (Qwen3.5-4B, 3-bit, local) is too weak to run all of Anka. When the network drops, his behaviour gets odd, so those stretches are left out of the scores.
- **`gemini-3-flash-preview` is deprecated.** It should move to a newer Flash model, which costs about 50 % more per token.
- **The code has grown fast.** `anka/body.py` (the 2D body) is 170 KB. The 2D and 3D bodies share ideas but not code. There are many small test scripts, and `settings.py` is long.
- **Small samples:** most experiments are 3 runs against 3 runs.
- **Only tested on macOS.** Sound now has a Linux and a Windows path (`anka/sound.py`), and the rest is plain Python, but nobody has run Anka on Linux or Windows yet.

## 📋 Next

The work goes on in the simulator until the real duck arrives, and this repository is updated along the way.

**In the simulator (now):**
1. **Make the kick work in MuJoCo:** a reliable step-up to the ball, so Pollen's kick actually hits it.
2. **Self-paced thinking:** he chooses when his next thought comes (soon / in a while / when something happens), instead of a fixed timer.
3. **Goals with reasons:** why he wants something, and checking that later.
4. **Richer body language:** a better reading of his `BODY` words, more use of Pollen's emotions, and fewer but better-chosen sounds.
5. **A better transfer test:** does experience beat common sense in a new situation?
6. **Tidy the code:** split the big files, and share more between the 2D and 3D bodies.
7. **Flowing conversation** with a live voice model, where he can be interrupted.

**When the real duck arrives** (expected around March 2027):
- Run `ankasim/simbody.py` against the real robot: the same JSON-RPC calls, a real camera, a real depth sensor.
- Test what only real hardware can show:
  - the camera floor test,
  - depth from the camera (Depth Anything V2),
  - YOLO for things in the room,
  - hearing where a voice comes from,
  - hands-free listening,
  - the beak.

**Later:** a stronger local brain (more RAM, or a brain on board the duck), and emotion steering vectors for the local model.
