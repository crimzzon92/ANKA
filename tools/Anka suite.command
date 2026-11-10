#!/bin/bash
# Anka suite – double-click (or run with a suite name). Runs the measuring scenarios one after another in the 2D body
# simulator, then scores them (test_logs/scores.csv). Uses the real cloud brain: each run costs about $0.10 and takes
# ~7 minutes (5-6 min day + the night). Leave the Mac alone while it runs; Ctrl+C stops after the current run.
#
#   "Anka suite.command"            = baseline   (alone, cup, sock, shout – 3 runs each, ≈ 12 runs, ~1.5 h, ≈ $1.30)
#   "Anka suite.command" feelings   = experiment 1: the same with his feelings held still (alone + shout, 6 runs)
#   "Anka suite.command" lonely     = experiment 2: alone with loneliness held HIGH (3 runs) and LOW (3 runs)
#   "Anka suite.command" twodays    = learning: the cup task on day 1, a night, the cup task again on day 2 (life mode)
#   "Anka suite.command" nextday    = one more day of the same life (cup task, life mode, memory kept) – run after twodays
#   "Anka suite.command" transfer   = TRANSFER TEST: cup (day 1) → night → the sock up on the sofa (day 2), vs the sofa task
#                                     in a fresh life (control) – ×3 (9 runs, ~65 min, ≈ $1.20)
#   "Anka suite.command" wise       = ASK THE WISER BRAIN WHEN STUCK: the ball in the corner ×3 (~21 min, ≈ $0.50) – compare by hand
#                                     with the 2026-11-09 19:11–20:07 runs ("thinkhalf" = the same runs)
#   "Anka suite.command" think      = THINK HARDER WHEN STUCK: the ball stuck in the corner, ×3 with / ×3 without (one day each)
#   "Anka suite.command" balltransfer = TRANSFER TEST 2: ball stuck in a corner (day 1) → ball against another wall (day 2):
#                                     WITH night ideas / WITHOUT ideas (--no-ideas) / fresh life (control) – ×3 (15 runs, ~1 h 50, ≈ $2)
#   "Anka suite.command" expect     = B1: the surprise scenario ×3 with expectations + ×3 without (--no-expect)
#   "Anka suite.command" echo       = shout ×3 with feelings that outlast the event + shout ×3 without (--no-echo)
#   "Anka suite.command" talk       = alone ×3 + cup ×3 (~42 min, ≈ $0.60) – how much he talks alone AND with Alex
#   "Anka suite.command" quick      = one run of each scenario (4 runs) – to check that everything works
cd "$(dirname "$0")/.." || exit 1
PY="$HOME/anka-env/bin/python"
SUITE="${1:-baseline}"
case "$SUITE" in
    baseline) RUNS=("alone" "cup" "sock" "shout" "alone" "cup" "sock" "shout" "alone" "cup" "sock" "shout") ;;
    feelings) RUNS=("alone --frozen-feelings" "shout --frozen-feelings" "alone --frozen-feelings"
                    "shout --frozen-feelings" "alone --frozen-feelings" "shout --frozen-feelings") ;;
    lonely)   RUNS=("alone --feelings loneliness=0.9" "alone --feelings loneliness=0.1"
                    "alone --feelings loneliness=0.9" "alone --feelings loneliness=0.1"
                    "alone --feelings loneliness=0.9" "alone --feelings loneliness=0.1") ;;
    twodays)  RUNS=("cup --life --newlife" "cup --life") ;;
    nextday)  RUNS=("cup --life") ;;
    quick)    RUNS=("alone" "cup" "sock" "shout") ;;
    talk)     RUNS=("alone" "cup" "alone" "cup" "alone" "cup") ;;
    transfer) RUNS=("cup --life --newlife" "high --life" "high --life --newlife"
                    "cup --life --newlife" "high --life" "high --life --newlife"
                    "cup --life --newlife" "high --life" "high --life --newlife") ;;
    thinkhalf|wise) RUNS=("ballcorner --life --newlife" "ballcorner --life --newlife" "ballcorner --life --newlife") ;;
    think)    RUNS=("ballcorner --life --newlife" "ballcorner --life --newlife --no-think-harder"
                    "ballcorner --life --newlife" "ballcorner --life --newlife --no-think-harder"
                    "ballcorner --life --newlife" "ballcorner --life --newlife --no-think-harder") ;;
    balltransfer) RUNS=("ballcorner --life --newlife" "ballwall --life" "ballcorner --life --newlife --no-ideas" "ballwall --life --no-ideas" "ballwall --life --newlife"
                    "ballcorner --life --newlife" "ballwall --life" "ballcorner --life --newlife --no-ideas" "ballwall --life --no-ideas" "ballwall --life --newlife"
                    "ballcorner --life --newlife" "ballwall --life" "ballcorner --life --newlife --no-ideas" "ballwall --life --no-ideas" "ballwall --life --newlife") ;;
    expect)   RUNS=("surprise" "surprise --no-expect" "surprise" "surprise --no-expect" "surprise" "surprise --no-expect") ;;
    echo)     RUNS=("shout" "shout --no-echo" "shout" "shout --no-echo" "shout" "shout --no-echo") ;;
    *) echo "Unknown suite '$SUITE' (baseline, feelings, lonely, twodays, nextday, quick, talk, echo, expect, transfer, balltransfer, think, thinkhalf, wise)"; read -r -p "Press Enter..."; exit 1 ;;
esac
if [ ! -x "$PY" ]; then echo "❌ Can't find $PY"; read -r -p "Press Enter..."; exit 1; fi
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then open -a Ollama; sleep 8; fi   # (memory search needs it)

STARTED=$(date +%Y-%m-%d_%H%M)
echo "📈 Suite '$SUITE': ${#RUNS[@]} runs, about $(( ${#RUNS[@]} * 7 )) minutes. Started $(date +%H:%M)."
i=0
for RUN in "${RUNS[@]}"; do
    i=$((i + 1))
    set -- $RUN
    SCEN="$1"; shift
    echo; echo "================ run $i of ${#RUNS[@]}: $SCEN $* ($(date +%H:%M)) ================"
    "$PY" run_anka.py --simulate --body --scenario "$SCEN" "$@" || echo "⚠️  run $i ended with an error – going on"
    sleep 3
done

echo; echo "📈 Scoring..."
"$PY" tools/score_run.py >/dev/null
SINCE_ARGS=""; [ "$SUITE" = "echo" ] && SINCE_ARGS="shout --since $STARTED"
[ "$SUITE" = "expect" ] && SINCE_ARGS="surprise --since $STARTED"
[ "$SUITE" = "think" ] && SINCE_ARGS="ballcorner --since $STARTED"
[ "$SUITE" = "thinkhalf" ] && SINCE_ARGS="ballcorner --since 2026-11-09_19"
[ "$SUITE" = "wise" ] && SINCE_ARGS="ballcorner --since 2026-11-09_19"
if [ "$SUITE" = "balltransfer" ]; then
    "$PY" tools/score_run.py --transfer "$STARTED" ballwall | tee "test_logs/compare_$(date +%Y-%m-%d_%H%M)_$SUITE.txt"
elif [ "$SUITE" = "transfer" ]; then
    "$PY" tools/score_run.py --transfer "$STARTED" | tee "test_logs/compare_$(date +%Y-%m-%d_%H%M)_$SUITE.txt"
else
    "$PY" tools/score_run.py --compare $SINCE_ARGS | tee "test_logs/compare_$(date +%Y-%m-%d_%H%M)_$SUITE.txt"
fi
echo; echo "✅ Suite '$SUITE' done $(date +%H:%M). Scores: test_logs/scores.csv"
read -r -p "Press Enter to close this window..."
