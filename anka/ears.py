"""Ears: the microphone, Whisper, the cloud listening itself, and hands-free listening."""
import base64
import gc
import io
import json
import os
import re
import subprocess
import sys
import termios
import threading
import time
import wave

import numpy as np

from settings import (CLONE_VOICE, CONVERSATION_MEMORY, FOLDER, FOLLOW_UP_SECONDS, HANDS_FREE, HEAR_WITH_CLOUD, LISTEN_MAX, MICROPHONE, NAME, NAME_SOUNDS, SAMPLE_RATE, SHORT_OUTPUT, SILENCE_END, SIM, UNSURE_HEARING, WAIT_FOR_SPEECH, WHISPER_BEAM, WHISPER_MODEL)
from anka.shared import (S, WhisperModel, answering, cv2, heard, listening, state, unsure_words)

class HeardSound:
    """What the microphone recorded – for the cloud brain to listen to itself."""
    def __init__(self, sound):
        self.sound = sound


def sound_to_wav_base64(sound):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(sound, -1, 1) * 32767).astype("<i2").tobytes())
    return base64.b64encode(buffer.getvalue()).decode()


def record_voice():
    """python run_anka.py --record-voice – records a short sample of your voice for VOICE_ENGINE = "clone"."""
    global LISTEN_MAX, WAIT_FOR_SPEECH, SILENCE_END
    LISTEN_MAX, WAIT_FOR_SPEECH, SILENCE_END = 15, 8, 2.0
    print("Read this out loud in your normal voice, in a quiet room:\n\n"
          "   \"Hi! I'm a small robot duck, and I love to watch what happens around me.\n"
          "    Today I saw a cup, a guitar and my best friend. What are you building?\"\n")
    input("Press ENTER and start reading...")
    sound, spoke = record()
    if not spoke or len(sound) < SAMPLE_RATE * 4:
        print("🎤 That was too short or too quiet – try again (5–10 seconds is best).")
        return
    path = os.path.join(FOLDER, CLONE_VOICE)
    data = (np.clip(sound / (np.max(np.abs(sound)) + 1e-9) * 0.9, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(data.tobytes())
    print(f"✅ Saved {len(sound) / SAMPLE_RATE:.1f} seconds as {CLONE_VOICE}.\n"
          "   Set VOICE_ENGINE = \"clone\" at the top of run_anka.py and test it with: python run_anka.py --voice-test")


def whisper_hint():
    """Names and words Anka knows – helps Whisper hear them right."""
    with memory_lock:
        words = re.findall(r"\b[A-Z][a-z]{2,}\b", " ".join(facts))
        heard_words = re.findall(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b",
                                 " ".join(c["human"] for c in conversations[-200:]))
    often = [w for w in dict.fromkeys(heard_words) if heard_words.count(w) >= 2]  # names said again and again
    names = [w for w in dict.fromkeys(words + often) if w not in ("The", "Human", "I'm", "What", "How")][:20]
    return f"{NAME} the robot duck talks with the human" + (f" about {', '.join(names)}." if names else ".")


def load_ears():
    """Loads Whisper. Returns a function: sound -> text."""
    gc.collect()
    try:  # already downloaded? Then it doesn't need to ask the internet (that could take very long)
        whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", local_files_only=True)
    except Exception:
        whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    def transcribe(sound):
        segments, _ = whisper.transcribe(sound, language="en", vad_filter=True,
                                         beam_size=WHISPER_BEAM, initial_prompt=whisper_hint())
        segments = list(segments)
        # How sure were the ears? (a low score = it may have misheard – the brain is told)
        sure = np.mean([s.avg_logprob for s in segments]) if segments else 0.0
        state["unsure_hearing"] = bool(segments) and sure < UNSURE_HEARING
        text = " ".join(s.text for s in segments).strip()
        if state["unsure_hearing"]:
            unsure_words.add(text)  # (so a mishearing doesn't end up in memories and dreams)
        return text

    print(f"👂 Ears: Whisper {WHISPER_MODEL}")
    return transcribe


# The microphone runs in its own small program that opens the microphone ONCE and then waits for
# "REC". On a MacBook the microphone and the speakers share a clock – opening the microphone again and
# again (while Anka speaks or has just spoken) gave "Internal PortAudio error -9986" and broken sound.
# A separate program also keeps PortAudio out of Anka itself: if it breaks, Anka just starts a new one.
MIC_PROGRAM = r"""
import sys, json, time, queue, threading, collections
sys.excepthook = lambda kind, error, _: print(f"🎤 ({kind.__name__}: {error})", file=sys.stderr)
import numpy as np
import sounddevice as sd
a = json.loads(sys.argv[1])
out = sys.stdout.buffer
def send(info, sound=None):
    data = b"" if sound is None else sound.astype("float32").tobytes()
    info["bytes"] = len(data)
    out.write((json.dumps(info) + "\n").encode() + data)
    out.flush()
devices = [(i, d) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
if not devices:
    send({"error": "no microphone found"}); sys.exit(1)
if a["list"]:
    for _, d in devices:
        print("   " + d["name"], file=sys.stderr)
try:
    default = sd.query_devices(kind="input")["name"]
except Exception:
    default = devices[0][1]["name"]
chosen = []
if not a["default"]:
    if a["mic"]:
        chosen = [x for x in devices if a["mic"].lower() in x[1]["name"].lower()]
    elif "iphone" in default.lower():  # an iPhone used as camera often becomes the microphone too
        chosen = [x for x in devices if "macbook" in x[1]["name"].lower()]
chosen = chosen or [x for x in devices if x[1]["name"] == default] or devices[:1]
i, d = chosen[0]
sr = int(d["default_samplerate"]) or 16000  # the microphone's own sound quality
block = int(sr * 0.05)
history = collections.deque(maxlen=40)  # the last 2 seconds (to measure how quiet the room is)
blocks = queue.Queue()
recording = threading.Event()
def callback(data, frames, when, status):
    b = data[:, 0].copy()
    history.append(b)
    if recording.is_set():
        blocks.put(b)
def rms(b):
    return float(np.sqrt(np.mean(b ** 2)))
try:
    stream = sd.InputStream(device=i, samplerate=sr, channels=1, dtype="float32", blocksize=block,
                            callback=callback)
    stream.start()
except Exception as error:
    send({"error": f"{type(error).__name__}: {error}"}); sys.exit(1)
time.sleep(0.6)
send({"name": d["name"], "ready": True})
for command in sys.stdin:
    command = command.strip()
    before = list(history)
    if command == "TEST":
        send({"name": d["name"], "loudest": max([float(np.max(np.abs(b))) for b in before] or [0.0])})
        continue
    if not before or not stream.active:
        send({"error": "the microphone stopped sending sound"}); sys.exit(1)
    # How quiet is the room? The quietest moments of the last 2 seconds (Anka was quiet then)
    noise = float(np.percentile([rms(b) for b in before], 20))
    limit = min(0.04, max(0.006, noise * 3))
    while not blocks.empty():
        blocks.get_nowait()
    recording.set()
    parts = before[-4:]  # 0.2 s from just before – so the first syllable isn't lost
    quiet = command == "LISTEN"  # hands-free: listens a moment for someone talking, without a fuss
    wait = 1.0 if quiet else a["wait"]
    if not quiet:
        print("🎤 Listening... speak now! (I stop when you stop)", file=sys.stderr, flush=True)
    result = {"name": d["name"], "spoke": False, "loudest": 0.0, "limit": limit}
    t, silent_since, broken, voiced = 0.0, None, False, 0
    while True:
        try:
            b = blocks.get(timeout=1.5)
        except queue.Empty:
            broken = True
            break
        parts.append(b)
        t += 0.05
        level = rms(b)
        result["loudest"] = max(result["loudest"], level)
        if level > limit:
            voiced += 1
            if not result["spoke"]:
                print("🎤 (hearing you...)", file=sys.stderr, flush=True)
            result["spoke"], silent_since = True, None
        elif result["spoke"]:
            silent_since = silent_since if silent_since is not None else t
            if t - silent_since > a["silence"]:
                break
        if t > a["max"] or (not result["spoke"] and t > wait):
            break
    recording.clear()
    if broken:
        send({"error": "the microphone stopped sending sound"}); sys.exit(1)
    if quiet and voiced < 6:  # under 0.3 s of sound – a bump or a click, not someone talking
        result["spoke"] = False
    if quiet and not result["spoke"]:
        send(result)  # (nothing to send back)
        continue
    sound = np.concatenate(parts)
    if sr != 16000:  # resample to 16 kHz, which the ears want
        new = np.linspace(0, len(sound) - 1, int(len(sound) * 16000 / sr))
        sound = np.interp(new, np.arange(len(sound)), sound)
    send(result, sound)
"""
mic_info = {}  # the chosen microphone and the latest recording: name, loudest, limit


class Microphone:
    """Keeps the microphone program running. If it has problems, a new one is started."""
    def __init__(self):
        self.p = None
        self.lock = threading.Lock()
        self.on_hearing = None

    def _close(self):
        if self.p is not None and self.p.poll() is None:
            self.p.kill()
        self.p = None

    def _read_answer(self, seconds):
        result = []

        def read():
            line = self.p.stdout.readline()
            if line:
                info = json.loads(line)
                data = self.p.stdout.read(info["bytes"]) if info.get("bytes") else b""
                result.append((info, data))
        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(timeout=seconds)
        if not result:
            ended = self.p is not None and self.p.poll() is not None
            self._close()
            raise RuntimeError("the microphone program stopped" if ended else "the microphone stopped answering")
        info, data = result[0]
        if "error" in info:
            self._close()
            raise RuntimeError(info["error"])
        if info.get("name") and info["name"] != mic_info.get("name"):
            print(f"🎤 Microphone: {info['name']}")
        mic_info.update(info)
        return info, data

    def _start(self, use_default=False, show_list=False):
        self._close()
        arg = json.dumps({"mic": MICROPHONE, "default": use_default, "list": show_list,
                          "silence": SILENCE_END, "max": LISTEN_MAX, "wait": WAIT_FOR_SPEECH})
        self.p = spawn([sys.executable, "-c", MIC_PROGRAM, arg], stdin=subprocess.PIPE,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p = self.p

        def show_status():
            for line in p.stderr:
                line = line.decode(errors="ignore").rstrip()
                if line and "MallocStackLogging" not in line:
                    if "(hearing you" in line and self.on_hearing:
                        self.on_hearing()  # hands-free: someone started talking – Anka holds his tongue
                    print(line, flush=True)
        threading.Thread(target=show_status, daemon=True).start()
        self._read_answer(15)

    def ask(self, command, seconds, use_default=False, show_list=False):
        with self.lock:
            if self.p is None or self.p.poll() is not None:
                self._start(use_default, show_list)
            try:
                self.p.stdin.write((command + "\n").encode())
                self.p.stdin.flush()
            except (BrokenPipeError, OSError):
                self._close()
                raise RuntimeError("the microphone program stopped")
            return self._read_answer(seconds)


microphone = Microphone()


def test_microphone():
    """Listens for half a second at the start. Only zeros = Terminal may not use the microphone."""
    try:
        info, _ = microphone.ask("TEST", 10)
        if info["loudest"] == 0.0:
            print("🎤 ⚠️  The microphone only gives silence. On the Mac: System Settings → Privacy & Security →\n"
                  f"      Microphone → turn on Terminal. Then quit Terminal completely and start {NAME} again.")
    except Exception as error:
        print(f"🎤 ⚠️  Couldn't use the microphone ({error})")


def record():
    """Records until you have finished talking. Returns (sound, whether you said something).
    If the microphone has problems: wait a moment, start it again, and finally try the Mac's default one."""
    try:
        info, data = microphone.ask("REC", LISTEN_MAX + 15)
    except RuntimeError as error:
        print(f"🔈 (the microphone had a hiccup: {error} – starting it again)")
        time.sleep(1.0)
        try:
            info, data = microphone.ask("REC", LISTEN_MAX + 15)
        except RuntimeError:
            info, data = microphone.ask("REC", LISTEN_MAX + 15, use_default=True)
    return np.frombuffer(data, dtype="float32").copy(), info["spoke"]


talk_now = threading.Event()  # ENTER was pressed (in the terminal or in the camera window)


def read_keys():
    while True:
        try:
            input()
        except EOFError:
            return
        talk_now.set()


def camera_key():
    """Updates the camera window. ENTER or space there = talk to Anka."""
    if cv2.waitKey(1) in (13, 10, 32):
        talk_now.set()


def listen_loop(ears):
    """Runs in the background: waits for ENTER, listens until you have finished and turns speech into text."""
    threading.Thread(target=read_keys, daemon=True).start()
    while True:
        talk_now.wait()
        talk_now.clear()
        listening.set()
        cloud_hears = HEAR_WITH_CLOUD and cloud_is_on() and not SIM
        if not cloud_hears:
            ears.preload()   # start loading the ears while you talk
        if S.voice is not None:
            S.voice.preload()  # and the voice, so Anka can answer at once
        try:
            state["voice_was_on"] = speaking()  # (only then can the mic catch the duck's own voice)
            hush()           # Anka stays quiet so the microphone doesn't hear it
            time.sleep(0.4)  # let the speaker go completely quiet
            sound, spoke = record()
            if not spoke:
                if mic_info.get("loudest", 0) == 0:
                    print("🎤 (heard nothing – the microphone gives only silence. Check that Terminal may use "
                          "the microphone in System Settings → Privacy & Security → Microphone)")
                else:
                    print(f"🎤 (heard nothing – loudest sound {mic_info['loudest']:.3f}, "
                          f"needed {mic_info['limit']:.3f}. Mic: {mic_info.get('name')})")
                continue
            print("🎤 Got it, thinking...")
            if cloud_hears:
                heard.put(HeardSound(sound))  # the cloud brain listens to it itself
                continue
            transcribe = ears.get()
            if transcribe is None:
                print("🎤 My ears aren't working right now – try again in a moment.")
                continue
            if busy():
                print("🎤 (waiting for my brain to finish a thought...)")
            with heavy():
                text = transcribe(sound)
            if text:
                heard.put(text)
            else:
                print("🎤 (couldn't make out any words)")
        except Exception as error:
            print("🎤 Microphone problem:", error)
        finally:
            listening.clear()  # always let go, even if something went wrong
            try:
                # Throw away ENTER presses that piled up, so it doesn't start listening again by itself
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
            except Exception:
                pass
            talk_now.clear()


def hands_free_loop(ears):
    """HANDS_FREE: listens all the time – no ENTER. Stays deaf while Anka himself speaks (and a moment after),
    so he doesn't hear his own voice."""
    microphone.on_hearing = listening.set  # someone started talking – Anka holds his tongue until they're done
    quiet_since = time.time()
    print("🎤 Hands-free: " + ("I hear everything said in the room" if HANDS_FREE == "always"
                              else f"say \"{NAME}\" to talk to me") + " (ENTER works too)")
    while True:
        if listening.is_set() or speaking() or answering.is_set():
            quiet_since = time.time()
            time.sleep(0.1)
            continue
        if time.time() - quiet_since < 1.0:  # let his own voice die away first
            time.sleep(0.1)
            continue
        follow_on = time.time() - max(state["last_human"], state["last_pip"]) < FOLLOW_UP_SECONDS
        need_name = HANDS_FREE == "name" and not follow_on
        cloud_hears = HEAR_WITH_CLOUD and cloud_is_on() and not SIM
        if need_name or not cloud_hears:
            ears.preload()  # the ears listen for his name (they stay loaded in this mode)
        voice_before = S.speech
        try:
            info, data = microphone.ask("LISTEN", LISTEN_MAX + 15)
        except Exception as error:
            print(f"🎤 (hands-free listening had a hiccup: {error})")
            time.sleep(2)
            continue
        if not info.get("spoke") or S.speech is not voice_before:
            listening.clear()  # nobody talked – or Anka started speaking meanwhile (then it was his own voice)
            continue
        try:
            sound = np.frombuffer(data, dtype="float32").copy()
            state["voice_was_on"] = True  # (so his own voice is caught if it slipped in)
            if cloud_hears and not need_name:
                print("🎤 Got it, thinking...")
                heard.put(HeardSound(sound))  # the cloud brain listens to it itself
                continue
            transcribe = ears.get()
            if transcribe is None:
                continue
            with heavy():
                text = transcribe(sound)
            if not text:
                continue
            if need_name and not re.search(NAME_SOUNDS, text, re.I):
                if not SHORT_OUTPUT:
                    print(f"🎤 (heard talking, but not to me: {text})")
                continue
            print("🎤 Got it, thinking...")
            heard.put(HeardSound(sound) if cloud_hears else text)
        except Exception as error:
            print("🎤 Microphone problem:", error)
        finally:
            listening.clear()


def hear_and_answer(sound, conversation):
    """The cloud brain listens to your voice itself and answers in the same trip:
    it writes down what you said (for the memory) and how you said it, and what Anka answers.
    Returns (heard words, tone, answer, start time) – or None if the cloud couldn't do it (then Whisper hears)."""
    started = time.time()
    if not cloud_is_on():
        return None
    now = time.time()
    notes = ""
    own_t, own_text = state.get("last_own", (0, ""))
    if own_text and now - own_t < 25:
        notes += f'(Just before this, you had said to the human: "{own_text}")\n'
    back_t, back_gone = state.get("came_back", (0, ""))
    if back_gone and now - back_t < 120:
        notes += f"({human_name() or 'The human'} was away for {back_gone} and came back {ago(now - back_t)}.)\n"
    if state["said"]:
        notes += f'(The last thing you said out loud was: "{state["said"][-1][1]}")\n'
    notes += speaker_note()
    question = (f"{state['scene']}\n{notes}"
                "The human just said something to you – it is in the attached sound recording. Listen to it.\n"
                "Reply in EXACTLY this form:\n"
                "HEARD: <exactly what the human said, word for word, in their language>\n"
                "TONE: <how they said it, 1-4 words, e.g. 'cheerful', 'tired', 'laughing', 'whispering'>\n"
                "SAY: <your answer out loud>\n"
                "FEEL: <only the feelings this changed, e.g. happiness=0.3, nervousness=0.8 – or none>\n"
                "If you can't make out any words, write HEARD: NONE.")
    messages = conversation[-CONVERSATION_MEMORY:] + [{"role": "user", "content": question}]
    last_words = next((m["content"] for m in reversed(conversation) if m["role"] == "user"), None)
    cloud["cloud_only"] = True  # (the reserve brain can't hear – then Whisper takes over instead)
    try:
        answer = ask_brain(messages, search=last_words, max_tokens=150, reply_to_human=True,
                           picture=picture_for_reply(""), audio=sound_to_wav_base64(sound))
    except Exception as error:
        print(f"☁️  (the cloud couldn't listen: {error} – Whisper hears instead)")
        return None
    finally:
        cloud["cloud_only"] = False
    if cloud.get("answered_by") != "cloud":
        return None  # the reserve brain answered – it can't hear, so Whisper has to
    heard_words = (field("HEARD", answer) or "").strip()
    if re.fullmatch(r"(?i)none\W*", heard_words):
        heard_words = ""
    feel_now("\nFEEL: " + (field("FEEL", answer) or ""))
    said = clean(field("SAY", answer) or "")
    if heard_words and not said:
        return None
    return heard_words, (field("TONE", answer) or "").strip(" .").lower() or None, said, started


# ---- other parts of Anka (imported last, so the parts can use each other) ----
from anka.resources import busy, heavy
from anka.feelings import feel_now
from anka.memory import ago, conversations, facts, human_name, memory_lock
from anka.brain import ask_brain, clean, cloud, cloud_is_on, field
from anka.eyes import picture_for_reply, speaker_note
from anka.voice import hush, spawn, speaking
