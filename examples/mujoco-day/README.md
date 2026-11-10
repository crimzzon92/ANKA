# Example: one day in the MuJoCo simulator

This folder holds a complete recorded test day: Anka in Pollen Robotics' Microduck simulator, running the scripted
scenario **"all"** (about 17 minutes, then a night). The human in the script is called **Alex**. Everything Anka thinks,
says and does was decided by the language model at the time. Nothing here was edited, except that the human's name was
changed.

You can run the same day yourself with `tools/Anka in sim (everything).command`. It will not turn out the same, because
he decides differently every time.

## The files

| File | What it is |
|---|---|
| [`simulation_log.txt`](simulation_log.txt) | The whole day as it scrolled past in the terminal: what Alex said, every thought, feeling, move and answer, the night, and a report on his body at the end. |
| [`thought_trace.jsonl`](thought_trace.jsonl) | The same day as data: one JSON line per thought or answer, with what he was shown (memories, notes, feelings before) and what he answered. This is what `tools/score_run.py` measures. |
| [`room_map.png`](room_map.png) | **Left:** the room from above. The yellow line is the path he really walked, and the purple marks show where *he believed* things and he were. **Right:** the floor plan he built himself from his depth sensor and step counting (white = floor he has seen, black = something in the way, green = landmarks, orange = things he remembers). |
| [`camera/`](camera/) | A few frames from his own camera, about 20 cm above the floor, with the boxes his colour finder drew around the ball and the sock. |

## How the day goes (the script)

| Time | What happens | What it tests |
|---|---|---|
| 0:05 | "Have a good look around the room." | Senses, looking around |
| 0:55 | "What can you see right now?" | Describing what he sees |
| 1:45 | "Can you go over to the sock?" | Walking to a thing |
| 3:20 | "Now walk up to the ball and kick it!" | Walking to the ball and kicking |
| 5:00 | "Where did the ball go?" | Looking for something |
| 5:50 | Alex goes out; Anka is alone for about 3 minutes | Exploring alone, the living body |
| 9:20 | "I'm back! What did you find while I was away?" | Remembering what he did |
| 10:25 | "Where is the guitar? Can you go there?" | His map memory |
| 11:55 | "What is on top of the bookshelf?" | Looking up, the limits of his neck |
| 13:00 | "Could you sit down for a little while?" | Sitting |
| 13:50 | Alex shouts at him | A big feeling |
| 14:40 | Alex says sorry | A feeling that lingers |
| 15:50 | "What was the best part of your day?" | Memory and feelings |
| 16:40 | "Go to sleep" | The night: review, lessons, ideas, dream |

## Reading the log

Each line starts with a symbol:

| Symbol | Meaning |
|---|---|
| 🎬 | The script: something Alex says or does |
| 🧑 | What Alex said (with the tone of voice) |
| 💭 | A thought, inside his head |
| 🦆 | Something Anka said out loud |
| ❤️ | His seven feelings (0–1) |
| 🫀 | How his feelings show in his body (`BODY`) |
| 🦶 | A move his body made, and what his body reported back |
| 😮 | A surprise: what he expected vs what happened |
| 🧭 | A lesson or self-belief he said matters right now |
| 🦉 | Stuck (two misses in a row): this thought went to the bigger model |
| 🔎 | He checked what his body really did before speaking, and changed his words |
| 🎯 / 🏁 / 🫳 | A new goal / a goal done / a goal given up |
| 📔 | A line written in his diary |
| 🖼️ | A picture he chose to keep in his memory |
| 🗺️ | His map noticed something (a thing that moved) |
| 🌙 🪞 📝 💡 | The night: the night brain, new beliefs about himself, lessons, ideas |
| 💰 | What the run cost in cloud calls |

## Good places to look

- **The shout and the apology:** search the log for `STOP IT`.
- **The night:** search for `going to sleep`, then read on to the dream.
- **The report on his body** is at the very end: how still he stood between moves, how far off his own map was, his kicks, sounds and emotions.
