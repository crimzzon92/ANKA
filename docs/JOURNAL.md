# Journal – what we tried, what worked, what didn't

This is the honest history of Anka: the experiments, the numbers, and the lessons. It includes the ideas that failed,
because those are often the most useful part for anyone building something similar.

How to read it:
- ✅ **worked**, kept
- ❌ **didn't work**, changed or dropped
- 🤔 **mixed or unclear**: works, but not the way we hoped, or not proven

Almost every comparison here is 3 runs against 3 runs in a simulator, so a difference is a strong hint, not proof.
"No overlap" means every run in one group was better than every run in the other. The numbers come from
`tools/score_run.py`, which counts what he actually did. No AI judges anything.

The human in the test scripts is called Alex. Robin and Anna are other people in some of the test days.

---

## Contents

1. [The brain: thinking, feelings, memory](#1-the-brain-thinking-feelings-memory)
2. [Learning from experience: nights, lessons, self-image](#2-learning-from-experience-nights-lessons-self-image)
3. [The 2D body: seeing and mapping](#3-the-2d-body-seeing-and-mapping)
4. [The MuJoCo body: Pollen's real duck software](#4-the-mujoco-body-pollens-real-duck-software)
5. [Cost and brains](#5-cost-and-brains)
6. [Lessons that apply to everything](#6-lessons-that-apply-to-everything)

---

## 1. The brain: thinking, feelings, memory

### ❌ → ✅ Do his feelings change what he does?
Anka has seven feelings (numbers from 0 to 1) that the language model sees and updates.

- ❌ **Experiment E1: freeze every feeling at neutral.** Nothing in his behaviour changed. When he was shouted at with
  his nervousness held at 0.10, he *still* apologised and hid by the bookshelf "so I don't make him more angry". The
  feelings showed up in his thoughts, but not in his choices. The model reacted to the *situation* (the words, the
  scene) directly.
- ✅ **Experiment E2: alone, with loneliness held high (0.9) or low (0.1).** This was the first real difference caused
  by a feeling, with no overlap. Lonely Anka walked 11–16 m and explored once per run. Not-lonely Anka walked 19–27 m
  and explored three times. With no event to react to, the feeling was the only thing that differed, and it mattered.
- **Lesson:** an event decides the first reaction. Feelings matter in the quiet time afterwards and when he is alone.
  That led to the next idea.

### ✅ Feelings that outlast the event ("after-feelings")
Before this, a shout and an apology 20 seconds later meant he was back to "I love you!" at once. Now a strong feeling
fades slowly, like adrenaline, and he is told how he still feels.

- Two minutes after the apology his nervousness was **0.70, against 0.09** without the feature (no overlap).
- His thoughts changed clearly: *"Alex is nice again, but I'm still shaky"*, *"I'll watch from here, so I don't get yelled
  at again"*. In one run he picked up his sock as a comfort and came closer step by step.
- 🤔 His *actions* only changed a little: slightly quieter, slightly more distance. Not proven with 3 runs.

### ✅ Speaking is optional
At first he talked out loud on almost every thought, even alone: 14–17 lines in 5 minutes with nobody there.
We changed only the wording: the prompt now makes it clear that speaking out loud is optional, and it describes being
alone in neutral words. No rule about when to talk.

- Alone: **about 60 % fewer** lines out loud (2–9 instead of 14–17). Talk now comes in bursts when something catches
  his interest.
- ❌ With a person in the room: **no change** (16–19 lines in 5–6 minutes). A lesson about it exists, he is reminded of
  it, and it barely helps. This is still open.

### ✅ Knowing when he's alone
He used to think someone had left as soon as they walked out of view, and he said things like "Oh wait, you are the
human!" to a visitor. Now only a departure he *heard* counts (footsteps, the door), and the thoughts know who is there.

### 🤔 Expectation and surprise
Before every action he says what he expects. Afterwards he rates how surprised he was.

- ✅ As a mechanism it works: an expectation on 93–98 % of actions, and after a big surprise his next action is
  different 78–100 % of the time.
- ❌ It did **not** help him finish a task within one day (the sock was handed over in 1 of 3 runs both with and without
  it).
- ❌ At first he rated *gladness* as surprise ("I finally spotted the ball right where I hoped" → 0.8). Fixed by adding
  "– not how glad or sad you are about it" to the question.
- ✅ Where it pays off is **across days**: over three days in one life his average surprise fell from 0.28 to 0.11 to
  0.04. He learned to predict his world.

### ❌ → ✅ The same memory every time
Memory search by meaning works, but one strong memory (a dream about "a forest of table legs") came up in **28 of 30
thoughts** in one day. We first tried special rules (caps on goal lines, extra filters). They helped a bit and made
the code messier.

- ✅ **Habituation** replaced all of them: a memory that just came up pulls weaker for a while, like a smell you stop
  noticing. The most-repeated memory dropped to 7–12 of 22–35 thoughts, and the variety of memories went from 0.19 to
  0.27–0.40. Behaviour stayed the same or got better.
- **Lesson:** one simple, general mechanism beat several special rules.

### ✅ Memories from earlier days
In "life mode" he lives several days in a row with the same memory. At first almost everything recalled was from today
(0–19 %). After memory was integrated across days, **46–84 %** of recalled memories came from earlier days, and he says
things like *"The sofa is right here! I remember my red ball was near it yesterday."*

### ✅ Picture memories
He can choose to keep what he sees, and an old picture comes back next to the current one when it fits.

- ❌ At first he hardly kept any pictures.
- ✅ Telling him how many he had kept so far was enough: he now keeps **about 10 in a 17-minute day**. No rule, just a
  fact.

### ✅ The wiser brain, but only when stuck
When a fast thought fails, a bigger model (`gemini-4-argon`) takes the next one.

- ✅ **The ball stuck in a corner:** kicking got him nowhere. With the bigger model he thought *"Maybe I can pick the
  ball up with my beak!"* and carried it out. He solved it the same day in **3 of 3 runs**, against **2 of 8** with
  only the fast model.
- ❌ In the first version he used the bigger model far too often, even when he wasn't stuck and hadn't tried anything.
  Fixed: it only switches after **two misses in a row**, and the count starts over after each wise thought.

---

## 2. Learning from experience: nights, lessons, self-image

### ❌ → ✅ The night review deleted true memories
Each night he goes over the day and removes mistakes. The first versions were too eager:

- ❌ It deleted a correct *"You are Anna!"* as a mistake, because it didn't know who had said what.
- ❌ It treated a **changed plan** ("I'm staying at my old job after all") as a mistake.
- ✅ Fixed: the review is told who said what and must give a reason for every mistake, and a promise or a plan is never a
  mistake. Wrongly deleted memories went from **3 to 0** over three nights.

### ❌ → ✅ His self-image didn't match his body
- ❌ He believed *"I can climb the bookshelf"* after a whole day of failing to.
- ✅ Fixed: the night writes beliefs about himself from his **body's own track record** (what he tried and what really
  happened). Now he writes things like *"I tried to pick up the cup from the table, but I am too short to reach things
  up there"*, and the next morning he doesn't even try.

### 🤔 Lessons and ideas
- ✅ Untested ideas ("Maybe I could carry the ball…") are now kept apart from real lessons, so a guess doesn't turn into
  a fact overnight.
- 🤔 When a night wrote the right idea, the next day he used it in a new place: the ball was delivered in 18 s. That has
  happened in **1 of 3** runs so far.

### 🤔 Transfer test: does experience carry over to a *similar* situation?
Day 1: he can't reach a cup on the table. Day 2: a sock is on the sofa. That's a different thing, different furniture
and different words. A control group got day 2 with a fresh memory.

- ✅ In his thoughts he **generalised**: *"I am too small to reach things high up"*, *"… on the table or sofa"*.
- ❌ In his actions there was **no clear difference**. The control group also gave up after a single "too high" from
  the body. The task was too easy to show a difference, and when the human said "Give it one more go, please" the
  experienced Anka tried 5 more times, to please him.
- **Lesson:** the test needs a harder task. Common sense alone already solves this one.

### ❌ Ball transfer test: the control group was *faster*
A harder version. Day 1: "kick the ball over to me", with the ball stuck in a corner, so kicking fails and carrying it
in his beak works. Day 2: the ball is against another wall. Three lives with day 1 behind them against three fresh
lives (control).

- ❌ **The control won, with no overlap.** The ball reached the human after **13 / 13 / 13 s** in the control group and
  **17 / 81 / 133 s** with experience.
- Why: the day-2 words were "could you *get* the ball over to me?", which makes carrying the common-sense choice. The
  control never kicked at all. And in a fresh life he walked over to the "new" ball out of curiosity before anyone spoke,
  so he was already standing next to it.
- Why experience was slower: it depended entirely on **what the night wrote**.
  - One life never tried carrying on day 1. Its lesson was "stop kicking and ask the human for help", and on day 2 he
    asked for help (133 s).
  - One life learned something irrelevant, about side steps (81 s).
  - ✅ One life tried carrying at the end of day 1. Its lesson was *"When the wall blocks me from kicking the ball, I
    should pick the ball up with my beak instead"*. On day 2: no stuck kicks, and the ball was delivered in 17 s. That is
    real transfer: the right lesson used in a new place.
- **Lessons:** (1) he only learns from what he actually **tried**, like an animal. (2) When the night writes the right
  lesson, he uses it. (3) The test let common sense win again.

### 🤔 Ideas to try: thinking up something new at night
So that he can learn from more than what he happened to try, each night may now write an untested **idea** ("Maybe I
could…"), kept apart from lessons. We removed a hint from the night's instructions that listed "carry" among his
skills, so the ideas have to be his own. The test was the same as above, but day 2 used the same verb ("kick it to
me"), so the naive choice fails for everyone.

| Day 2 | With ideas | Experience, no ideas | Control (fresh life) |
|---|---|---|---|
| Kicks into the wall | 0, 1, 8 | 13, 17, 3 | 7, 2, 10 |
| Ball delivered | 1 of 3 (18 s) | 1 of 3 (250 s) | 2 of 3 (187, 219 s) |

- 🤔 There is overlap everywhere with 3 runs, so this is a hint, not proof. **It is not a demonstrated improvement in
  performance.**
- ✅ The chain *idea → memory → trying it* worked **every time a night wrote an idea** (2 of 3 nights). One night
  invented "carry it away from the wall" (delivered in 18 s). Another invented its own sensible idea, "ask the human to
  move the ball out of the corner", which the script didn't respond to.
- ❌ One day-1 **succeeded** by carrying, but the night wrote no lesson about it, so day 2 went back to kicking. **A
  success that isn't written down is lost.**
- ❌ The ideal case never happened in these runs: never carrying on day 1, inventing carrying at night, then doing it.

---

## 3. The 2D body: seeing and mapping

Before Pollen's simulator, Anka had a simple 2D room. It is fast and cheap, so most of the brain work was done there.

### ❌ → ✅ Easy senses vs real senses
- With an easy list of what was around him, he walked 18–19 m per test and did everything.
- ❌ With **realistic senses** (a drawn picture from 20 cm above the floor and an 8×8 depth sensor, with no list) he
  walked only 7.8 m. He called the sofa "a huge red wall" and a person's legs "a dark pillar", and he got stuck in a
  corner for two minutes.
- ✅ After a clearer drawing, a **mental map** from step counting, and a smarter walk that steps around obstacles, he
  walked 20–24 m with real senses. He found the ball by searching himself (*"That's not a ball, that's a sock!"*) and
  did the full "bring me the sock" errand.

### ✅ A map like a robot vacuum's
- Step counting alone drifted by **1.6 m** in an offline test.
- Landmarks (sofa, table, bookshelf, door, window) and then a proper **EKF** brought the error at the end of a test down
  to typically **3–12 cm**. The best run was 3 cm.

### ❌ Things that made the map worse
- ❌ **Scan matching** wiped the walls off the floor plan. The eyes drew the same view about 5 times a second while he
  stood still, and the tiny wobble in his position wore the walls away. He ended 57 cm off.
- ❌ **The whole map came out turned 8°.** The room's main direction was decided too early, and a "square the walls"
  correction then pulled his sense of direction towards that wrong angle. Removing the correction fixed it, because
  the gyro already did the job.
- ❌ **"Fixed walls" were found on furniture.** Fixed: a straight edge near furniture he has recognised, or with free
  floor on both sides, is not a wall.

### ❌ The depth AI added almost nothing (in the simulator)
**Depth Anything V2** guesses depth from the camera picture. In the simulator its guesses were typically 13 % off, and
it added very little to the floor plan, while costing a lot of CPU on an 8 GB laptop. It is off in the simulator for
now. It will be tested again with a real camera, where it may matter more.

### ❌ A 3D picture of the room
We built a 3D model of the room from depth guesses: coloured 5 cm cubes. It looked nice, but nothing used it, so it was
switched off.

### ✅ The blind zone found by accident
He kept failing to pick up a sock he had just put down. The cause: his eyes are 20 cm above the floor, so with his head
level he **cannot see the floor closer than about 38 cm**, and "put down" places things 20 cm in front of him. The fix
was a fact, not a rule: the body tells him when something is too close to see, and to look down.

---

## 4. The MuJoCo body: Pollen's real duck software

Pollen's MuJoCo simulator runs the real Microduck software: the real walking policy, falls, a camera and the 8×8 depth
sensor. That makes it much harder than the 2D room.

### ✅ Connecting at all
On the first day he walked, turned, looked around, slept and dreamt in the simulator. His room description was vague:
the window was *"a blue block floating in the air"*, and the sock was called a ball.

### ❌ YOLO saw nothing
We tried **YOLO** (a learned object detector, which the real duck will need) to find the ball, sock, cup and guitar. It
found **nothing**: the simulator's objects are simple shapes, and YOLO is trained on photos. We switched to a simple
**colour finder** as an honest stand-in, and YOLO waits for the real camera.

### ❌ → ✅ Turns overshot
Small turns came out far too big: asked for 45° he turned 68–72°, and asked for 30° he turned 71°. His map was 75 cm and
30° off at the end of one run. Fixed with shorter turn pulses for small angles and by checking the heading against the
real gyro. The map is now **5–13 cm off** after a 17-minute day.

### ❌ → ✅ A false ball
The colour finder saw a "ball" on the guitar and on door handles. Fixed: in the saved camera pictures the green boxes now
sit only on the real ball and sock.

### ❌ → ✅ He stood completely still
Between his own thoughts the body did nothing, so he looked like a statue.

- First measurement: he stood **completely still 97 %** of the time between his own moves.
- ❌ The first "living body" made almost no visible difference.
- ✅ The second version uses Pollen's own idle motions: glances, head tilts, weight shifts, breathing, small turns
  towards where he looks, and eyes that follow a rolling ball. Still time is down to **20–25 %**, and never more than a
  few seconds in a row.

### ❌ → 🤔 Feelings in the body were too weak
He writes how he feels in his body (`BODY: crouched low, head tucked in, trembling`), and the body acts it out.

- ❌ At first the movements were so small they often didn't show.
- 🤔 After Pollen's 0.18.9 update and bigger movements it looks much better. But the words are read with word lists,
  which is a limited vocabulary ("tail wagging" means nothing to a duck without a tail). That is on the to-do list.

### ❌ The kick (not solved yet)
This one taught us the most.

- ❌ For many runs **the kick did nothing**, while Anka *believed* he had kicked the ball away. He was only told what his
  body tried, not what his eyes saw afterwards. Fixed first: the kick's result now comes from what his camera sees.
- A separate kick test (no brain, just the body) showed that Pollen's kick **does** work, and found four reasons it
  failed inside Anka:
  1. Pollen's kick is **blind**. It is trained with the ball 9 cm in front of and 4 cm beside the kicking foot. Anka
     kicked from much further away.
  2. **Looking down kills the kick.** The policy needs the head straight. Anka was looking down at the ball.
  3. The colour finder **dropped anything closer than 12.5 cm**, so he lost the ball just as he got close.
  4. **Very short steps are ignored** by the walking policy, so he couldn't take the last small step up to the ball.
     Longer steps pushed the ball away.
- 🤔 All four are fixed in the code: walk closer, straighten the head before kicking, see close up, and use step pulses
  that adapt. It has not hit the ball in a full run yet.
- **Lesson:** a learned skill has hidden assumptions. Test the skill alone first, then inside the brain.

### ✅ His voice and sounds in the simulator
Duck sounds, Pollen's 13 emotions and his real voice all play in the simulator, with a mute button in the room window.
🤔 He chooses a lot of duck sounds (30–40 in 17 minutes). Telling him the count may help, like it did for pictures.

---

## 5. Cost and brains

- ✅ **One request per thought instead of two** roughly halved the number of calls.
- ✅ **Google's own Gemini API** instead of the OpenAI-style one, with a lower picture resolution: **19 % fewer tokens**
  per question.
- ❌ Caching didn't happen: Google reused nothing for this model, so caching saved 0 %.
- ✅ A day in the simulator costs about **$0.15–0.25**, and the night about $0.015.
- ❌ **The local reserve brain** (Qwen3.5-4B, 3-bit, which fits in 8 GB of RAM) is too weak to *be* Anka. Replies took
  8.6 s and thoughts 21 s, and it missed who it was talking to, didn't feel hurt and ignored corrections. It once
  crashed for lack of memory. It stays as a reserve for when the network is gone, and those stretches are left out of
  the scores.
- ❌ One test day was lost because the **cloud credit ran out** halfway. Now runs with many reserve-brain answers are
  marked and skipped.

---

## 6. Lessons that apply to everything

1. **Give facts, not rules.** Almost every fix that lasted was a new *fact* for the model ("you can't see closer than
   38 cm", "you have kept 3 pictures", "the ball didn't move"), not an "if X then do Y".
2. **Measure before and after.** Many changes that *felt* better did nothing in the numbers (expectation and surprise
   within one day), and some that felt small did a lot (a change of wording in the prompt cut solo talk by 60 %).
3. **One general mechanism beats many special cases.** Habituation replaced a pile of memory rules and worked better.
4. **The model believes what you tell it.** If the body says "kicked", he thinks the ball flew, even when he is looking
   at it. Tell him what actually happened.
5. **Test the body alone first.** The kick test without the brain found in minutes what many full runs couldn't.
6. **A simulator is not the world.** YOLO, the depth AI and the camera all need the real duck before we know anything.
   That is the next big chapter.

➡️ Where things stand right now is in [STATUS.md](../STATUS.md).
