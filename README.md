# Anka – an AI brain for the Microduck robot duck

*An LLM-driven mind for [Pollen Robotics' Microduck](https://github.com/pollen-robotics/microduck): he thinks, feels, remembers, dreams and explores – in the Microduck MuJoCo simulator today, on the real robot duck next.*

Anka is a hobby project: a mind for **Microduck**, Pollen Robotics' 25 cm walking robot duck. Until the real duck
arrives, Anka lives in two simulators. One is a simple 2D room. The other is Pollen's own **MuJoCo** simulator, which runs
the real duck software.

The idea behind him: **the language model makes the decisions.** It decides what Anka thinks, feels, says and does.
The Python code gives him *facts*: what his camera sees, what his body did, what he remembers, how much time has passed.
It does not give him behaviour rules. His personality is meant to grow from what happens to him, and his nights are spent
looking back on the day.

![Anka's walk through the room in the example day](examples/mujoco-day/room_map.png)

👉 **[docs/GUIDE.md](docs/GUIDE.md)** – how to install and run him, and how everything works inside.
👉 **[STATUS.md](STATUS.md)** – what is done and working, how well (with numbers), what should be fixed, and what's next.

## About this project

I'm a hobbyist building this in my spare time. **I don't have a Microduck yet**, so I honestly don't know how much of this
holds up in the real world. Everything here has only run in simulation. As soon as my duck arrives (planned for around
March 2027), I'll test it on the real thing and report what works and what doesn't.

This repository is updated often. Expect things to change, break and get better. Ideas, questions and issues are welcome.

Much of the code was written together with an AI assistant (Claude).

## What he can do

**Thinking and feeling**
- He thinks on his own every few seconds: a thought, a feeling, sometimes a goal.
- He decides by himself whether to say something out loud, or whether to keep it in his head.
- He has seven feelings (curiosity, boredom, loneliness, happiness, affection, nervousness, playfulness). He describes them in his own words.
- Big feelings **fade slowly**, like adrenaline: being shouted at stays with him for minutes, not seconds.
- He has **expectations and surprise**: before acting he says what he expects. Next time he rates how surprised he was, and big surprises go into his diary.
- When he is stuck (two misses in a row) that one thought goes to a **wiser brain** (a bigger model).

**Memory**
- His memory holds facts, a diary, and conversations, each stored with how they felt.
- Memories are found by meaning (`nomic-embed-text`). Moments that felt strongly come back more easily.
- A memory that has just come up pulls weaker for a while (habituation), so he does not keep repeating himself.
- **Picture memories**: he can choose to keep what he sees. An old picture comes back next to what he sees now.

**Nights**
- He looks back on the day and corrects his mistakes, based only on what the day showed.
- He re-feels the strongest moments.
- He updates his self-image from his body's track record: what he actually managed, not what he planned.
- He writes lessons and *untested ideas*, which he can try the next day.
- He dreams.

**Body (MuJoCo / the real duck)**
- He can walk, turn, look about, sit, kick a ball, peck at the floor, and make duck sounds.
- He can act out one of Pollen's 13 ready-made emotions: sad, laugh, yes, no, excited …
- **A living body** keeps him moving *between* thoughts, like an animal: glances, a head tilt, a weight shift, breathing, eyes following a rolling ball, the body turning towards what he looks at.
- **His feelings show in his body**: each thought and reply can say how he feels in his body, in his own words ("crouched low, head tucked down, trembling"). The body takes that on as its style.
- He builds his own map and floor plan from step counting, landmarks and his depth sensor, like a robot vacuum.

## An example day

[`examples/mujoco-day/`](examples/mujoco-day/) holds one test day in the MuJoCo simulator. It is the scripted scenario
"all", about 17 minutes plus the night. It contains:
- the full log
- the thought trace, one JSON line per thought or reply
- the map of his walk
- a few camera frames

The human in the script is called Alex.

In the day he looks around, walks to the sock and tries to kick the ball. Then Alex goes out, and Anka explores alone.
When Alex comes back, Anka is asked to sit, and then he is shouted at:

```
🧑 You: STOP IT! YOU'RE SO NOISY, YOU SILLY DUCK!   (sounded: shouting, angry)
🫀 (body: head down and trembling slightly)
🦆 I'm sorry, I'll be very quiet.
😮 (surprised 0.8: … I didn't expect Alex to be so loud and angry with me)
💭 (thinking) Alex is very angry and I am scared. I should stay very still and not make noise.
❤️  curious 0.20 · … · happy 0.10 · love 0.80 · nervous 0.90
🫀 (body: Crouched low, head tucked in, trembling slightly.)
…
🧑 You: Sorry Anka, I didn't mean to shout. You're a good duck.
🫀 (body: sitting very still, slowly lifting my head to look toward you)
🦆 It's okay, Alex, I'm glad you're not mad anymore.
```

That night he wrote:

```
🪞 New about me: I get scared and sad when Alex shouts at me, and I feel safe again when he says sorry.
📝 Lesson for tomorrow: When Alex asks me to sit down for a little while, I should stay quiet and rest instead of making noise.
💡 Idea to try: Maybe I could walk closer to the ball before I try to kick it.
💭 I dreamt a cold storm shook the tall bookshelf, and I was too scared to make a sound. Then Alex came home and wrapped
   me in a giant, warm red sock. We sat together in the quiet glow, and I finally felt safe again.
```

The day cost about $0.25 in cloud calls.

## How it is built

| Part | What it does |
|---|---|
| `run_anka.py` | Start-up and command-line modes (`--simulate`, `--mujoco`, `--scenario`, `--life` …) |
| `settings.py` | **All settings** in one place: the brains, timings, memory, the body, the test scenarios |
| `anka/brain.py` | Talks to the cloud brain (Gemini, through Google's API or any OpenAI-style one) and to the local reserve brain (Ollama) |
| `anka/thinking.py` | His thoughts: thought → feeling → (act, expect, body, speak) → goals |
| `anka/main.py` | His day: answering, listening, the face window |
| `anka/memory.py` | Facts, diary, conversations, picture memories, search by meaning, habituation |
| `anka/feelings.py` | The seven feelings, feeling words, and the fading "after-feelings" |
| `anka/night.py` | Sleep: review, re-feeling, self-image, lessons, ideas, dream |
| `anka/body.py` | The 2D virtual body (room, senses, mental map, floor plan, EKF) |
| `ankasim/simbody.py` | The body in Pollen's MuJoCo simulator (the same `do()` / `describe()` as the real duck will have) |
| `ankasim/living.py` | The living body between thoughts, and the `BODY` styles |
| `ankasim/mind.py` | His map in 3D: landmarks, floor plan, walls |
| `ankasim/duck_link.py` | The link to the duck's own software (`robotd` JSON-RPC, depth sensor, camera) |
| `anka/simulation.py` | Scripted test days and the room window (with a mute button) |
| `tools/` | Test suites, the scorer (`score_run.py`), replays, small test programs |

**Brains used:**
- **Day:** `gemini-3-flash-preview`.
- **Night and "wise" thoughts:** `gemini-4-argon`.
- **Reserve:** Qwen3.5-4B (3-bit) in Ollama, for when the network is gone. It fits in the 8 GB of RAM that the dev machine (an M1 MacBook) has.
- **Memory search:** `nomic-embed-text`.

## Running it

Developed on macOS (Apple Silicon). The `.command` files are double-click launchers for macOS.

1. **Python packages:** run `pip install -r requirements.txt`. Then install [Ollama](https://ollama.com) and run `ollama pull nomic-embed-text`.
2. **The voice model:** run `./download_models.sh`. The face models and the depth AI download themselves the first time they are needed.
3. **A Gemini API key:** put it in `cloud_key.txt`, one line. This file is in `.gitignore`.
4. Then:
   - **2D test day:** `python run_anka.py --simulate --body` (or a scenario: `--scenario shout`, `sock`, `cup`, `alone` …)
   - **MuJoCo:** this needs Pollen's [microduck](https://github.com/pollen-robotics/microduck) and `microduck_rl` in `~/Pollen/`, Rust, and GStreamer. Run `tools/Update Pollen (0.18.9).command` once, then `Start Anka sim.command`, then `tools/Anka in sim (everything).command`.
   - **With a real camera and microphone:** `Start Anka.command` (`python run_anka.py`).

**Anka starts with an empty memory**: this repository contains none. He learns everything from you. His memory is kept in `memory/real/` and `memory/simulation/`, and it is never part of the repository. The [guide](docs/GUIDE.md) has all the details.

## License

Apache License 2.0 – see [LICENSE](LICENSE), the same license as Pollen's Microduck software.

## Credits

The 3D body builds on Pollen Robotics' open-source [Microduck](https://github.com/pollen-robotics/microduck) software,
its MuJoCo models and its emotion set (Apache-2.0). The voice is [Kokoro](https://github.com/thewh1teagle/kokoro-onnx).
The face sensor and recogniser come from [OpenCV Zoo](https://github.com/opencv/opencv_zoo). Depth from the camera uses
Depth Anything V2.
