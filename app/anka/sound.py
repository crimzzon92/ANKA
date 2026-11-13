"""Plays a .wav file on any computer.

- macOS: the Mac's own `afplay` (copes with switching headphones/speakers without errors).
- Linux: `paplay` (PulseAudio/PipeWire), `aplay` (ALSA) or `ffplay`, whichever is installed.
- Anything else (Windows) or none of the above: Python's own `sounddevice`, already in requirements.txt.

play_file() always returns something that behaves like a running program: poll(), terminate(), wait(),
returncode and stderr. So the code that plays his voice works the same way on every computer.

Only the macOS way has been tested. The others are written to work but have not been tried yet.
"""
import io
import shutil
import subprocess
import sys
import threading
import time
import wave

import numpy as np


def player_command(path):
    """The program that plays a sound file on this computer, as a command list (None = use Python)."""
    if sys.platform == "darwin" and shutil.which("afplay"):
        return ["afplay", path]
    if sys.platform.startswith("linux"):
        for command in (["paplay", path],
                        ["aplay", "-q", path],
                        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]):
            if shutil.which(command[0]):
                return command
    return None


def player_name():
    """What plays sound here, in words (for start-up messages)."""
    command = player_command("x.wav")
    return command[0] if command else "Python (sounddevice)"


class PythonPlayer:
    """Plays a .wav with sounddevice in its own thread, but looks like a running program."""

    def __init__(self, path):
        self.returncode = None
        self.stderr = io.BytesIO()
        self._stop = threading.Event()
        self._done = threading.Event()
        threading.Thread(target=self._run, args=(path,), daemon=True).start()

    def _run(self, path):
        try:
            import sounddevice as sd
            with wave.open(path, "rb") as w:
                rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
                raw = w.readframes(w.getnframes())
            kind = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
            data = np.frombuffer(raw, dtype=kind).reshape(-1, channels).astype("float32")
            data = (data - 128) / 128 if width == 1 else data / float(np.iinfo(kind).max)
            sd.play(data, rate)
            ends = time.time() + len(data) / rate + 0.1
            while time.time() < ends and not self._stop.is_set():
                time.sleep(0.03)
            sd.stop()
            self.returncode = 0
        except Exception as error:
            self.stderr = io.BytesIO(f"{type(error).__name__}: {error}".encode())
            self.returncode = 1
        finally:
            self._done.set()

    def poll(self):
        return self.returncode

    def terminate(self):
        self._stop.set()

    kill = terminate

    def wait(self, timeout=None):
        self._done.wait(timeout)
        return self.returncode


def play_file(path, start=None, **options):
    """Starts playing a .wav file and returns at once. start: the function that starts a program
    (default subprocess.Popen); options are passed to it (ignored when Python plays the sound)."""
    command = player_command(path)
    if command:
        return (start or subprocess.Popen)(command, **options)
    return PythonPlayer(path)
