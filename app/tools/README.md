# tools/ – what everything here is for

You don't need any of these to run Anka. The main launchers are one folder up, in `app` (`Start Anka.command`,
`Anka in sim.command`, `Start everything.command`). These are for the 3D simulator, for measuring him, for
repeating the experiments in the [journal](../../docs/JOURNAL.md), and for finding faults.

The `.command` files are double-click launchers for macOS. On Linux, run them with `bash "<file>.command"`.
They all use the Python in `~/anka-env`.

## Running him in the 3D simulator

Start Pollen's simulator first with `Start Anka sim.command` (in the top folder).

| File | What it does |
|---|---|
| `Anka in sim (everything).command` | The 17-minute test day that tests everything (`--scenario all`). The [example day](../../examples/mujoco-day/) is one of these. |
| `Anka in sim (still body).command` | The same as `Anka in sim.command`, but with the living body switched off (`--no-living`), to compare. |
| `Anka eyes.command` | A window with what the simulated duck senses: camera, depth grid, IMU. You can drive him by hand with the keys. |
| `Update Pollen (0.18.9).command` | Moves your copy of Pollen's `microduck` to the release Anka is tested with, and remembers where you were. |
| `Pollen go back.command` | Undoes that: back to the Pollen version you had before. |

## Measuring him

| File | What it does |
|---|---|
| `score_run.py` | The scorer. Reads the thought traces in `test_logs/` and counts what he actually did (actions, failures, repeats, goals, feelings, memory use, cost) into `test_logs/scores.csv`. `--compare` makes a table per scenario and condition. No AI judges anything. |
| `Anka score.command` | Runs the scorer and shows the comparison. |
| `replay_test.py` | Replays the moves from a 2D test log offline (no cloud brain, no cost) with many random seeds, to measure his map. |
| `trail_plot.py` | Draws the path of a replayed log, old settings against new. |

## Repeating the experiments

Each suite runs a set of test days in the 2D simulator with the real cloud brain, then scores them. They cost money:
roughly $0.10 per run. The numbers they produced are in [STATUS](../../STATUS.md) and the [journal](../../docs/JOURNAL.md).

| File | Experiment |
|---|---|
| `Anka suite.command` | The baseline: alone, cup, sock and shout, 3 runs each. It also runs all the others when given a name. |
| `Anka suite - quick.command` | One run of each scenario, to check that everything works. |
| `Anka suite - feelings.command` | E1: feelings frozen at neutral, against normal. |
| `Anka suite - lonely.command` | E2: alone with loneliness held high, against low. |
| `Anka suite - echo.command` | Feelings that fade slowly ("after-feelings"), with against without. |
| `Anka suite - talk.command` | How much he talks, alone and with company. |
| `Anka suite - expect.command` | Expectation and surprise, with against without. |
| `Anka suite - twodays.command` | Learning: the same task on day 1 and day 2 of one life. |
| `Anka suite - nextday.command` | Day 3 of the same life (run after twodays). |
| `Anka suite - transfer.command` | Transfer test: can't reach a cup on day 1, then a sock on the sofa on day 2, against a fresh life. |
| `Anka suite - balltransfer.command` | Ball transfer test: with night ideas, without them, and a fresh life. |
| `Anka suite - wise.command` | The wiser brain when stuck: the ball in the corner, with against without. |

## Finding faults in the 3D body

For developers. They test one part of the body alone, without the brain.

| File | What it tests |
|---|---|
| `Anka kick test.command` | Pollen's kick with the ball placed exactly where the kick expects it. This is how we found why his kicks missed. |
| `Anka walktest.command` | How far he really walks compared with what he asked for. |
| `Anka turntest.command` | How far he really turns compared with what he asked for. |
| `Anka headtest.command` | The head joints and the camera angle, standing and walking. |
| `Anka fake test.command` | The body and his map against a fake simulator (no MuJoCo needed, runs fast). |
| `Anka yolotest.command` | What YOLO recognises in the simulator's camera picture (it found nothing; see the journal). Installs `ultralytics` the first time. |
