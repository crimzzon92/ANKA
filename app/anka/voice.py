"""Voice: Kokoro (or a cloned voice, or the Mac's), the duck sound and the mood in his voice."""
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave

import numpy as np

from settings import (SIM_SOUND, CLONE_VOICE, DUCK_BUZZ, DUCK_NOSE, DUCK_VOICE, FOLDER, KOKORO_MODEL, KOKORO_SPEED, KOKORO_VOICE, KOKORO_VOICES, MAC_VOICE, MOOD_VOICE, NAME, SIM, VOICE_ENGINE, VOICE_PITCH)
from anka.shared import S, voice_waiting
from anka.sound import play_file





def spawn(args, **kw):
    """Starts another program (afplay, say, the microphone) with posix_spawn instead of fork.
    With fork, macOS frameworks already loaded (camera, sound) run code in the new process, and it
    prints "MallocStackLogging: can't turn off malloc stack logging..." – posix_spawn skips that."""
    args = list(args)
    args[0] = shutil.which(args[0]) or args[0]  # posix_spawn needs the full path
    kw.setdefault("env", {k: v for k, v in os.environ.items() if not k.startswith("MallocStack")})
    return subprocess.Popen(args, close_fds=False, **kw)


def voice_style(kokoro, recipe):
    """'af_heart' -> that voice. 'af_heart*0.6 + af_nicole*0.4' -> a new voice mixed from both."""
    if "+" not in recipe and "*" not in recipe:
        return recipe.strip()
    parts = []
    for part in recipe.split("+"):
        name, _, weight = part.strip().partition("*")
        parts.append((name.strip(), float(weight.rstrip("% ")) / (100 if "%" in weight else 1) if weight else 1.0))
    total = sum(w for _, w in parts) or 1.0
    return sum(kokoro.get_voice_style(name) * (w / total) for name, w in parts).astype("float32")


def duck_effect(audio, sr, amount, buzz=None, nose=None):
    """Makes a voice sound more like a duck: a thin, nasal sound with a fast buzz and a little rasp.
    amount: 0 = unchanged, 1 = full duck. Cheap – a few milliseconds per sentence."""
    if amount <= 0 or len(audio) < 64:
        return audio
    n = len(audio)
    f = np.fft.rfftfreq(n, 1 / sr)
    spectrum = np.fft.rfft(audio)
    thin = 1 / (1 + (350 / np.maximum(f, 1)) ** 4)                       # less low end = a small body
    nose = nose or DUCK_NOSE
    nasal = 1 + amount * 2.0 * np.exp(-0.5 * ((f - nose) / (nose * 0.28)) ** 2)  # a honking band
    out = np.fft.irfft(spectrum * (1 - amount + amount * thin) * nasal, n)
    t = np.arange(n) / sr
    out *= 1 - amount * 0.35 * (0.5 + 0.5 * np.sin(2 * np.pi * (buzz or DUCK_BUZZ) * t))  # the quack buzz
    drive = 1 + amount * 4
    out = np.tanh(out * drive) / np.tanh(drive)                           # a little rasp
    # keep the same loudness as the plain voice (the rasp would otherwise make it louder)
    out *= (float(np.sqrt(np.mean(audio ** 2))) + 1e-9) / (float(np.sqrt(np.mean(out ** 2))) + 1e-9)
    peak = float(np.max(np.abs(out)))
    return (out * (0.95 / peak if peak > 0.95 else 1.0)).astype("float32")


def mood_voice():
    """How Anka's mood colours his voice right now: (tempo, pitch, shakiness). 1, 1, 0 = his normal voice."""
    if not MOOD_VOICE:
        return 1.0, 1.0, 0.0
    with feelings_lock:
        f = dict(feelings)
    glad = f.get("happiness", 0.5) - 0.5 + 0.5 * (f.get("playfulness", 0.4) - 0.4)   # -0.5 .. +0.8
    down = max(0.0, f.get("loneliness", 0) - 0.6) + max(0.0, 0.3 - f.get("happiness", 0.5))  # sad
    tempo = 1.0 + 0.18 * glad - 0.25 * down
    pitch = 1.0 + 0.06 * glad - 0.08 * down
    shaky = max(0.0, f.get("nervousness", 0.1) - 0.4) / 0.6  # 0 .. 1
    return float(np.clip(tempo, 0.8, 1.15)), float(np.clip(pitch, 0.93, 1.05)), float(np.clip(shaky, 0, 1))


def shake_voice(audio, sr, shaky):
    """A nervous voice: it trembles a little (the loudness wobbles about 7 times a second)."""
    if shaky <= 0.02:
        return audio
    t = np.arange(len(audio)) / sr
    wobble = 1 - 0.25 * shaky * (0.5 + 0.5 * np.sin(2 * np.pi * 7 * t + np.sin(2 * np.pi * 1.3 * t)))
    return (audio * wobble).astype("float32")


def load_kokoro(recipe=None, pitch=None, duck=None, buzz=None, nose=None):
    """Kokoro through ONNX – no PyTorch needed. Returns a function: text -> (sound, sample rate) per sentence."""
    from kokoro_onnx import Kokoro
    recipe = recipe or KOKORO_VOICE
    pitch = pitch or VOICE_PITCH
    duck = DUCK_VOICE if duck is None else duck
    model, voices = os.path.join(FOLDER, KOKORO_MODEL), os.path.join(FOLDER, KOKORO_VOICES)
    for path in (model, voices):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{os.path.basename(path)} is missing next to run_anka.py")
    kokoro = Kokoro(model, voices)
    style = voice_style(kokoro, recipe)
    language = "en-gb" if recipe.strip().startswith("b") else "en-us"
    # Pitch: speak a bit slower and play it back faster (or the other way round) – same speed, other pitch
    speed = min(2.0, max(0.5, KOKORO_SPEED / pitch))

    def sound(text):
        # One sentence at a time, so Anka starts talking straight away instead of waiting for everything
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if sentence.strip():
                tempo, mood_pitch, shaky = mood_voice()  # (checked again for every sentence)
                samples, sr = kokoro.create(sentence, voice=style, lang=language,
                                            speed=min(2.0, max(0.5, speed * tempo / mood_pitch)))
                rate = int(sr * pitch * mood_pitch)
                audio = duck_effect(np.asarray(samples, dtype="float32"), rate, duck, buzz, nose)
                yield shake_voice(audio, rate, shaky), rate, sentence

    for _ in sound("Hi!"):  # warm up
        pass
    return sound


def load_clone():
    """Pocket TTS (Kyutai, ~100M parameters, runs on the CPU): speaks with the voice in CLONE_VOICE."""
    from pocket_tts import TTSModel
    path = os.path.join(FOLDER, CLONE_VOICE)
    if not os.path.exists(path) and CLONE_VOICE.lower().endswith((".wav", ".mp3", ".m4a")):
        raise FileNotFoundError(f"{CLONE_VOICE} is missing next to run_anka.py – record one with: python run_anka.py --record-voice")
    model = TTSModel.load_model()
    state = model.get_state_for_audio_prompt(path if os.path.exists(path) else CLONE_VOICE)

    def sound(text):
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if sentence.strip():
                audio = model.generate_audio(state, sentence)
                yield np.asarray(audio.numpy() if hasattr(audio, "numpy") else audio, dtype="float32"), \
                    int(model.sample_rate), sentence

    for _ in sound("Hi!"):  # warm up
        pass
    return sound


def voice_test(recipe, pitch, duck=None, buzz=None, nose=None):
    """python run_anka.py --voice-test "am_puck" 1.35 0.8 – hear a voice (voice, pitch, duck) without starting Anka."""
    if VOICE_ENGINE == "clone":
        print(f"🔊 Voice: cloned from {CLONE_VOICE}")
        sound = load_clone()
    else:
        duck = DUCK_VOICE if duck is None else duck
        print(f"🔊 Voice: {recipe}   pitch: {pitch}   duck: {duck}")
        sound = load_kokoro(recipe, pitch, duck, buzz, nose)
    test = QuietSpeech()
    for part in sound(f"Hi, I'm {NAME}! I'm a small robot duck. What are you building today?"):
        play(part[0], part[1], test, part[2])


class QuietSpeech:
    """A tiny stand-in for a speech object, so play() can be used on its own."""
    def __init__(self):
        self.stopped = threading.Event()
        self.player = None


def load_voice():
    """Loads the voice. If the cloned voice doesn't work, Kokoro is used; if Kokoro doesn't work, the Mac's voice."""
    pass  # (shared values live on S)
    S.voice = None
    engines = {"clone": [("cloned voice", load_clone), ("Kokoro", load_kokoro)],
               "kokoro": [("Kokoro", load_kokoro)]}.get(VOICE_ENGINE, [])
    for name, load in engines:
        try:
            with heavy():
                sound = load()
            S.voice = LazyModel(f"voice ({name})", load, sound)
            print(f"🔊 Voice: {name} " + (f"'{KOKORO_VOICE}'" if load is load_kokoro else f"from {CLONE_VOICE}"))
            return
        except Exception as error:
            print(f"🔊 Couldn't load {name} ({type(error).__name__}: {error}).")
    print(f"🔊 Voice: macOS '{MAC_VOICE}'")


def play(sound, sr, talk, sentence=""):
    """Plays sound with the computer's own player (afplay on a Mac, see anka/sound.py) instead of
    PortAudio. afplay copes with switching headphones/speakers without 'Audio Hardware Error'."""
    path = os.path.join(tempfile.gettempdir(), f"pip_voice_{os.getpid()}.wav")
    data = (np.clip(np.asarray(sound, dtype="float32"), -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(data.tobytes())
    for attempt in range(2):
        track_mouth(data.astype("float32") / 32767, sr, sentence=sentence)
        talk.player = play_file(path, start=spawn, stderr=subprocess.PIPE)
        while talk.player.poll() is None and not talk.stopped.is_set():
            time.sleep(0.03)
        if talk.player.poll() is None:
            talk.player.terminate()
            return True
        if talk.player.returncode == 0 or talk.stopped.is_set():
            return True
        error = (talk.player.stderr.read() or b"").decode(errors="ignore").strip()
        error = " ".join(line for line in error.splitlines() if "MallocStackLogging" not in line)
        print(f"🔊 (the speaker didn't play: {error or 'error ' + str(talk.player.returncode)}"
              + (" – trying again)" if attempt == 0 else ")"))
        time.sleep(0.3)
    return False


def mac_say(text, talk):
    """Says the text with the Mac's own voice (stops if Anka is interrupted). Only on a Mac:
    elsewhere the text is just printed."""
    if sys.platform != "darwin" or not shutil.which("say"):
        print(f"🔊 (no voice on this computer – Kokoro didn't start): {text}")
        return
    mac = spawn(["say", "-v", MAC_VOICE, text])
    while mac.poll() is None and not talk.stopped.is_set():
        time.sleep(0.05)
    if mac.poll() is None:
        mac.terminate()


class KokoroSpeech:
    """Plays Kokoro speech in the background (poll/terminate/wait, like a running program)."""
    def __init__(self, text):
        self.done = threading.Event()
        self.stopped = threading.Event()
        self.player = None
        threading.Thread(target=self._run, args=(text,), daemon=True).start()

    def _make_sounds(self, sound_maker, text, ready):
        """Makes the sound for each sentence – the next one is made while the one before is playing,
        so there is no pause between sentences."""
        try:
            parts = sound_maker(text)
            while not self.stopped.is_set():
                voice_waiting.set()  # a running thought lets go of the brain, so the voice gets through at once
                try:
                    with heavy():
                        voice_waiting.clear()
                        try:
                            ready.put(next(parts))
                        except StopIteration:
                            break
                finally:
                    voice_waiting.clear()
        except Exception as error:
            print("🔊 Voice problem:", error)
        finally:
            ready.put(None)  # no more sentences

    def _run(self, text):
        try:
            sound_maker = S.voice.get() if S.voice is not None else None
            if sound_maker is None:  # no Kokoro – use the Mac's voice this time
                mac_say(text, self)
                return
            ready = queue.Queue()
            threading.Thread(target=self._make_sounds, args=(sound_maker, text, ready), daemon=True).start()
            while not self.stopped.is_set():
                part = ready.get()
                if part is None or self.stopped.is_set():
                    break
                if not play(part[0], part[1], self, part[2] if len(part) > 2 else ""):
                    print("🔊 (using the Mac's own voice for this sentence)")
                    mac_say(text, self)
                    break
        except Exception as error:
            print("🔊 Voice problem:", error)
        finally:
            self.done.set()

    def poll(self):
        return 0 if self.done.is_set() else None

    def terminate(self):
        self.stopped.set()
        if self.player is not None and self.player.poll() is None:
            self.player.terminate()

    def wait(self):
        self.done.wait(timeout=30)


class SimSpeech:
    """A fake voice for the simulation: 'talks' as long as a real voice would."""
    def __init__(self, text):
        self.end = time.time() + 0.07 * len(text) + 0.5

    def poll(self):
        return 0 if time.time() > self.end else None

    def terminate(self):
        self.end = 0

    def wait(self):
        while self.poll() is None:
            time.sleep(0.05)


def hush():
    """Stops Anka if it is talking."""
    if S.speech is not None and S.speech.poll() is None:
        S.speech.terminate()


def speaking():
    return S.speech is not None and S.speech.poll() is None


def say(text):
    pass  # (shared values live on S)
    hush()
    S.speech = KokoroSpeech(text) if (not SIM or (SIM_SOUND and S.sound_on)) else SimSpeech(text)


def set_sound(on):
    """The mute button: his voice and his duck sounds on or off (simulation)."""
    S.sound_on = bool(on)
    if not S.sound_on:
        hush()
    print(f"🔊 sound {'ON' if S.sound_on else 'OFF (muted)'}")


# ---- other parts of Anka (imported last, so the parts can use each other) ----
from anka.resources import LazyModel, heavy
from anka.feelings import feelings, feelings_lock
from anka.face import track_mouth
