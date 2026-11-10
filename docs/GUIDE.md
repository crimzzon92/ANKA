# The Anka guide – how everything works

This guide explains how to set Anka up, how to run him in each of his modes, how his mind works inside, and how to
measure and tune him. For what works and what doesn't yet, see [STATUS.md](../STATUS.md).

---

## 1. What you need

| | |
|---|---|
| **Computer** | A Mac with Apple Silicon. Developed on an M1 with 8 GB. Linux should mostly work too, but the `.command` launchers are for macOS. |
| **Python** | 3.11 or newer, in a virtual environment at `~/anka-env`. The launchers use `~/anka-env/bin/python`. |
| **Cloud brain** | A Google Gemini API key, used for the day brain and the night brain. |
| **Local brain** | [Ollama](https://ollama.com), for memory search and the reserve brain. |
| **Camera + microphone** | Only for the real mode. The simulators don't need them. |
| **3D simulator (optional)** | Pollen Robotics' [microduck](https://github.com/pollen-robotics/microduck) and `microduck_rl` repositories, plus Rust (`cargo`) and GStreamer. |

## 2. Installation

```bash
git clone <this repository> Duck && cd Duck

# Python environment (the launchers expect it in ~/anka-env)
python3 -m venv ~/anka-env
~/anka-env/bin/pip install -r requirements.txt

# Ollama: memory search + the reserve brain
ollama pull nomic-embed-text
ollama pull hf.co/unsloth/Qwen3.5-4B-GGUF:UD-Q3_K_XL

# The voice model (too big for GitHub)
./download_models.sh

# Your Gemini key: one line, never committed
echo "YOUR-KEY" > cloud_key.txt
```

These download themselves the first time they are needed: the face sensor (0.2 MB), the face recogniser (37 MB) and
the depth AI (about 100 MB).

**For the 3D simulator:**
1. Clone `pollen-robotics/microduck` and `pollen-robotics/microduck_rl` into `~/Pollen/`.
2. Install GStreamer: `brew install gstreamer libnice-gstreamer`.
3. Run `tools/Update Pollen (0.18.9).command` once. It moves Pollen's code to the version Anka is tested with, which includes the 13 emotions.

## 3. Anka starts with an empty memory

A fresh copy of this repository holds **no memory at all**. The first time you run him for real, he wakes up like a
newly hatched duck. He has no facts, no diary, no self-image and no known faces. Everything he knows, he learns from you.

His memory lives in two folders. Both are created automatically, and both are in `.gitignore`:

| Folder | What is in it |
|---|---|
| `memory/real/` | **The real Anka:** `memory.json` (facts, diary, conversations, self-image, lessons, ideas, goals), `memory_vectors.npz` (memory search), `feeling_vectors.npz` (his feeling words), `known_faces.npz`, picture memories, and `.bak` copies. |
| `memory/simulation/` | **The test days.** This is rebuilt at every simulation start, except in life mode (see below). It never touches the real memory. |

- **Start over:** delete (or move away) `memory/real/`.
- **Simulations start with a made-up past.** It contains a few facts about "Alex" and "Robin", a cat and some old diary lines (`sim_past()` in `anka/simulation.py`), so the test scripts have something to work with. `--newborn` starts a simulation with an empty memory instead.

## 4. The ways to run him

| Double-click | Same as | What it does |
|---|---|---|
| `Start Anka.command` | `python run_anka.py` | **The real Anka:** your Mac's camera, microphone and speakers. |
| `Learn a face.command` | `python run_anka.py --learn-face <name>` | Teaches him a face. He must be asleep first, because he uses the camera. |
| `Start Anka sim.command` | (Pollen's `duck-sim`) | Starts the **MuJoCo simulator** with Anka's room. Leave it running. |
| `Anka in sim.command` | `python run_anka.py --simulate --mujoco` | Anka's brain with the simulated duck as his body (the default test day). |
| `tools/Anka in sim (everything).command` | `… --mujoco --scenario all` | A **17-minute day that tests everything** (see section 8). |
| `Start everything.command` | | Starts the simulator, waits for it, and starts Anka in it. |
| `Camera floor test.command` | `python camera_floor_test.py` | Tests depth from the camera with your real webcam on the floor. |

**Other command-line options for `run_anka.py`:**

| Option | What it does |
|---|---|
| `--simulate` | Fake camera, mic and voice, a script instead of you. Uses his real brain and a separate memory. |
| `--simulate --body` | The 2D virtual body in a room, with a window that shows the room from above. |
| `--scenario <name>` | A fixed test day: `alone`, `cup`, `sock`, `shout`, `surprise`, `high`, `ballcorner`, `ballwall`, `all` (see `SCENARIOS` in `settings.py`). |
| `--life` / `--newlife` | Life mode: the simulation keeps its memory between runs, so you can test several days in a row. `--newlife` starts a new life. |
| `--local` | Use only the local brain. |
| `--cloud <model>` | Use another cloud model. |
| `--dream` | A full night with a dream now, even if he dreamt recently. |
| `--tidy` | A dry run of the night's memory tidy-up. |
| `--faces` | List the faces he knows. |
| `--forget-face <name>` | Forget a face. |
| `--voice-test ["af_bella*0.5 + am_puck*0.5" 1.2]` | Hear his voice, or try a new voice mix. |
| `--microphones` | List the microphones. |
| `--record-voice` | Record a voice sample for the cloned-voice option. |

**Switching features off for one run** (to compare with and without):

| Option | Switches off |
|---|---|
| `--no-echo` | Feelings that fade slowly |
| `--no-expect` | Expectation and surprise |
| `--no-wise` | The wiser brain when stuck |
| `--no-ideas` | Night ideas |
| `--no-pictures` | Picture memories |
| `--no-living` | The living body (3D) |
| `--quiet` | Sound in the 3D simulator (`--sound` turns it on in any simulation) |
| `--frozen-feelings` | All feelings held at their resting level |
| `--feelings loneliness=0.9` | A feeling set to this value and held there |

## 5. A day with Anka (real mode)

- **A camera window** shows what he sees, and a drawn duckling face shows his expression.
- **Talk to him:** press **ENTER** (in the terminal or the camera window) and speak. With the cloud brain, Gemini hears your voice itself, words and tone. Whisper is the fallback. Hands-free listening is built in but off: `HANDS_FREE = "name"` makes him react when he hears "Anka".
- **He thinks by himself:** every 8–16 s when someone is there, every 25 s when he is alone, and every 4–6 s while his body is moving. Sometimes he says something out loud; most thoughts stay in his head.
- **He knows who he sees:**
  - faces he learned (`Learn a face.command`),
  - people introduced to him while exactly one unknown face is in view, whom he can learn by himself.
- **Going to sleep:** press **Ctrl + C**, or say "go to sleep". The first sleep after `BEDTIME` (22:00) is a full night: a review, tidying, re-feeling, self-image, lessons, ideas and a dream. Other shutdowns are a short rest that only saves his memory.

## 6. How his mind works

### One thought

Each thought is one call to the language model. The call contains:

- **Who he is** (fixed text, first), plus his newest self-beliefs.
- **What he senses now:** the camera picture, the time and day, and who is around. With a body it also has the depth sensor, what his last move did, his mental map and his floor plan.
- **His inner state:** seven feelings, plus any "big feelings from earlier" that are still fading.
- **Memories that fit the moment:**
  - found by meaning,
  - boosted when they felt strong or feel like now,
  - weakened when they came up very recently,
  - with one slot kept for an earlier day.

  At most one old picture memory is included.
- **What he has learned about himself** (lessons, self-beliefs, untested ideas). He writes which of them matters now (`MINDFUL`).
- **What he tried lately** and what came of it: facts from his body.

He answers in fields:

```
THOUGHT:  his thought                     FEEL:     feeling numbers that changed
FEELING:  how he feels, in his own words  FACE:     his face expression
SURPRISE: how different the last result was from what he expected (0–1) – why
ACT:      what his body does (or none)    EXPECT:   what he thinks will happen
BODY:     how his feelings show in his body right now
SAY:      something to say out loud – or none (most thoughts stay inside)
KEEP_PICTURE: why he wants to keep what he sees – or none
PLACE_NOTE:   a note about the spot he's at
```

Every few thoughts he also sets, finishes or drops goals.

**Answers to you** use the same parts: feelings, moment strength, act, body. The first reaction to something you say
comes with the answer, not with the next thought.

### Rules the code follows

- **The code gives facts, never behaviour rules.** For example, "the cup is too high for a small duck" is a fact the body reports. The code never says "give up after three tries".
- **One general principle beats special cases.** For example, habituation (a memory that just came up pulls weaker) replaced several memory-filter rules.
- **His ideas are his own.** The prompts never contain example solutions to the tests.

### Feelings

- Seven numbers between 0 and 1. The model changes them in every thought and answer, and they drift back towards rest over time.
- **After-feelings:** a big jump (0.3 or more) starts a slow fade, with half-lives of 60–120 s. While it lasts, the feeling can't drop back to normal faster than that. He is told about it: "you felt scared 1 min ago, when Alex shouted … still there, slowly fading".
- His own feeling words become vectors, so memories that *felt* similar can be found again.

### Memory

| Kind | What it is |
|---|---|
| Facts | About the human, sorted into "rooms" (a memory palace). Outdated facts are kept as "used to be true". |
| Diary | What he did, saw and felt, with how strongly it moved him. |
| Conversations | With who said what, and what he felt. |
| Picture memories | Small JPEGs he chose to keep (at most 40). |
| Places | Where things are in the room, in words ("the guitar – by the bookshelf"), kept between days. |

### The night (`anka/night.py`, model: `NIGHT_CLOUD_BRAIN`)

1. **Review:** mistakes are judged only from what the day showed. A memory is forgotten only if another memory proves it wrong.
2. **Tidying:** facts are merged, corrected and sorted.
3. **Re-feeling:** the strongest moments are felt again, so a hurt can soften overnight.
4. **Self-image and lessons:** written from his body's track record (what he really managed, not what he hoped). Also up to 2 **untested ideas** for the next day.
5. **Dream:** built from the day's strongest moments and the pictures he kept.

### The body

**2D** (`anka/body.py`, `--simulate --body`):
- A 5 × 4 m room with a sofa, a table, a bookshelf, a guitar, a ball, a sock and people.
- His senses are like the Microduck's: a camera 20 cm above the floor, an 8×8 depth sensor, an IMU, imperfect legs and a battery.
- He keeps his own map: odometry, an EKF with landmarks, a floor plan, fixed walls, and places he remembers.

**3D** (`ankasim/`, `--simulate --mujoco`):
- He talks to the duck's real software (`robotd`) through `duck_link.py`, the same JSON-RPC the real duck uses.
- The room is built by `make_scene.py`.
- Skills: walk, turn, look, sit, kick (with a step-up to the ball), peck, Pollen's emotions, and duck sounds.

**The living body** (`ankasim/living.py`, 3D) is a 10 Hz loop that runs *between* the brain's moves. It never decides
anything; it only moves the head, the body pose, the beak and small foot turns. It does:
- glances, re-fixations, slow drift, breathing and weight shifts,
- eyes that follow a moving ball,
- a head lift towards a voice,
- turning the body towards where he looks,
- the `BODY` style, which reads his words: crouched or tall, head down or up, still, calm, lively or nervous, trembling, bobbing, tilting, looking away or towards.

## 7. The 3D simulator in practice

1. `Start Anka sim.command` builds the room and starts Pollen's simulator. The first time after a Pollen update it rebuilds the duck's programs, which takes a few minutes. Leave the window open.
2. Start Anka: `Anka in sim.command`, or one of the tools in `tools/`.
3. Windows:
   - **"Anka's room"**, the room from above. Top right is a **mute button** for his voice and duck sounds (click it, or press **M**). Bottom left says what moves his body right now: orange for one of Pollen's animations, blue for the living body following his `BODY` words.
   - **"Anka eyes"** (`tools/Anka eyes.command`): his camera, the depth grid, the IMU, and keys to drive him by hand.
4. Afterwards the log, map and thought trace are in `test_logs/`. The end of the log has a report on his body: how still he stood, his map error, kicks, sounds, emotions.

Small test programs, to run with the simulator on and Anka off:
- `tools/Anka walktest.command`
- `tools/Anka turntest.command`
- `tools/Anka headtest.command`
- `tools/Anka kick test.command`

## 8. Measuring him

Every simulation writes a **thought trace** (`test_logs/trace_<time>_<scenario>_<condition>.jsonl`), one JSON line per
thought, answer, own line or goal. Each line records:
- what he was shown (memories, notes, feelings before),
- what he answered (thought, act, expect, surprise, body),
- what his body did, and how long it took.

`tools/score_run.py` turns the traces into numbers in `test_logs/scores.csv`. No AI judges anything; it counts. Some of the
measures:
- action and stillness shares, failed tries, repeated steps,
- how much he speaks, chains of thought, memory use and variety,
- goals done and dropped, feeling averages,
- reactions to a shout and an apology, surprise,
- in 3D: still time, kicks, sounds, map error.

```bash
python tools/score_run.py                 # score new runs, show the newest
python tools/score_run.py --compare       # every scenario × condition: mean (low–high)
python tools/score_run.py --compare shout --since 2026-11-08
```

**Suites** run a scenario several times, with and without a feature, and compare. They are all in `tools/`:
`Anka suite.command <name>` and its shortcuts:

| Suite | Compares |
|---|---|
| `quick` | One run of each scenario |
| `feelings` | Normal vs frozen feelings |
| `lonely` | Loneliness held high vs low |
| `echo` | With vs without slowly fading feelings |
| `expect` | With vs without expectation and surprise |
| `twodays` / `nextday` | Several days in one life |
| `transfer` / `balltransfer` | Experience vs a fresh life |
| `wise` | With vs without the wiser brain |

Each run costs $0.07–0.25.

## 9. Settings worth knowing (`settings.py`)

| Setting | What it does |
|---|---|
| `CLOUD_BRAIN`, `NIGHT_CLOUD_BRAIN`, `BRAIN` | The day, night and local brains. |
| `THINK_EVERY`, `THINK_EVERY_ALONE`, `THINK_EVERY_MOVING` | How often he thinks. |
| `BEDTIME` | When the full night happens. |
| `HANDS_FREE` | `False`, `"name"` or `"always"`. |
| `KOKORO_VOICE`, `VOICE_PITCH`, `DUCK_*` | His voice and its duck effect. |
| `CAMERA` | Which camera to try first. |
| `PICTURE_MEMORY`, `MAX_PICTURES` | Picture memories. |
| `HABIT_STRENGTH`, `HABIT_FADE` | How quickly a repeated memory loses its pull. |
| `ECHO_HALF_LIFE` | How long big feelings linger. |
| `WISE_WHEN_STUCK`, `STUCK_AFTER` | The wiser brain. |
| `SCENARIOS` | The scripted test days. Write your own here. |

## 10. What it costs

- **Normal use** with the cloud brain: roughly $0.25 an hour.
- **A 17-minute 3D test day:** about $0.25 (the camera picture is sent with every thought).
- **A night:** about $0.02.

The log prints a 💰 line after each simulation with the exact cost.

## 11. Troubleshooting

| Problem | What to do |
|---|---|
| "no cloud key found" | Put your key in `cloud_key.txt`, next to `run_anka.py`. |
| He thinks slowly or oddly | Check the log for "using the reserve brain". The network dropped and the small local model took over. |
| Memory search unavailable | Start Ollama and run `ollama pull nomic-embed-text`. |
| The 3D simulator has no camera | Install GStreamer (`brew install gstreamer libnice-gstreamer`). |
| The duck can't walk in 3D | Start with `./"Start Anka sim.command" --robot groundcontact`. |
| No sound in 3D | Check the mute button. The Mac plays his voice and duck sounds itself (Pollen's simulator plays sound with `aplay`, which macOS doesn't have). |
| A `.command` file won't open | Run `chmod +x "<file>.command"` and try again. |
