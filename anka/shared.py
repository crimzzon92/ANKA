"""Anka's nervous system: things every part of him shares while he runs (the camera library, what is
happening right now, signals between his parts, and the values in S that change while he runs)."""
import os
import queue
import threading
import types


# Values that change while Anka runs and that several parts of him must see the same way
# (when he last dreamt, his voice, what he is saying right now ...)
S = types.SimpleNamespace()
S.last_dream = None   # when Anka last dreamt (ISO time)
S.memory_ai_ok = True
S.can_turn_off_thinking = True   # False if the brain doesn't understand Ollama's 'think' setting
S.can_see_pictures = True        # False if the brain can't look at pictures
S.faces_known = None   # (FaceMemory – made when the eyes start)
S.speech = None   # what Anka is saying right now
S.sound_on = True  # the mute button (room window / M key) - only used in the simulation
S.voice = None    # the Kokoro voice (LazyModel), if it could be loaded

# OpenCV and Whisper's sound library (PyAV) both bring their own copy of the same Mac video library, so macOS
# prints a long, harmless "Class AVF... is implemented in both" warning. Hide it while loading them.
# (A real error while importing still shows – it is printed after the screen output is switched back on.)
_stderr, _devnull = os.dup(2), os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 2)
try:
    import cv2
    from faster_whisper import WhisperModel
finally:
    os.dup2(_stderr, 2)
    os.close(_devnull)
    os.close(_stderr)

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # hide harmless OpenCV warnings
except Exception:
    pass
heard = queue.Queue()             # what you said ends up here
listening = threading.Event()     # set while Anka listens to you


state = {"scene": "[Your camera sees nothing in particular.]", "last_human": 0.0, "last_pip": 0.0,
         "said": [], "all_thoughts": [], "retry": 0, "thought_nr": 0, "rest": 0,
         "seen_at": {}, "seen_since": {}, "familiar": [], "appeared": [], "events": [],
         "person_last": 0.0, "recent_talk": [], "faces": 0}
thoughts = []                 # Anka's latest quiet thoughts
thought_times = []            # when each of them was thought
timings = {"thought": [], "reply": [], "own": 0, "faces": {}, "feelings": []}  # how long thoughts and answers took (shown after a simulation)
wants_to_say = queue.Queue()  # things Anka has decided to say out loud
wake = threading.Event()      # set when something happens, so Anka thinks at once instead of waiting
to_remember = queue.Queue()   # conversations to go through for the long-term memory when Anka has time
awake = threading.Event()     # set as long as Anka is awake
answering = threading.Event()     # Anka is answering you
voice_waiting = threading.Event()  # the voice is waiting to make sound


unsure_words = set()   # things the ears heard but weren't sure about
