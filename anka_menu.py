#!/usr/bin/env python3
"""Anka's start menu: one place to start everything, check that it is installed, and connect to a real Microduck.

    macOS:    double-click "Anka menu (mac).command"
    Linux:    run "Anka menu (linux).sh" (or: python3 anka_menu.py) - the menu can also add Anka to your apps menu
    Windows:  double-click "Anka menu (win).bat"   (or: py anka_menu.py)

It only uses Python's own library, so it works before anything else is installed - and it can install the rest.
It changes nothing in how Anka thinks: it starts the same programs as the other launchers.
"""
import getpass
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")

def find_env():
    """Anka's Python environment: the one this menu runs in (if it is one), else ~/anka-env.
    Returns (folder, python) - a new environment will be made at ~/anka-env if none is found."""
    exe = "Scripts/python.exe" if IS_WIN else "bin/python"
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return Path(sys.prefix), Path(sys.executable)
    for name in ("anka-env",):
        folder = Path.home() / name
        if (folder / exe).exists() or (folder / exe).is_symlink():
            return folder, folder / exe
    return Path.home() / "anka-env", Path.home() / "anka-env" / exe


VENV, PY = find_env()                             # the same Python environment the .command launchers use
POLLEN = Path.home() / "Pollen" / "microduck"
POLLEN_RL = Path.home() / "Pollen" / "microduck_rl"
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
VOICE_FILES = ["kokoro-v1.0.onnx", "voices-v1.0.bin"]
KEY_FILE = HERE / "cloud_key.txt"
DUCK_FILE = HERE / "duck_address.txt"             # where the real duck is on your network (never part of the repo)
OLLAMA = "http://localhost:11434"
MEMORY_MODEL = "nomic-embed-text"
PACKAGES = ["numpy", "cv2", "onnxruntime", "kokoro_onnx", "sounddevice"]   # what must import (faster-whisper is optional)

# ---------------------------------------------------------------------------------------------------- small helpers

def say(text=""):
    print(text, flush=True)


def ask(question, default=""):
    try:
        answer = input(question).strip()
    except EOFError:                                    # the window or the input was closed: stop, don't loop
        say("\nBye! 🦆")
        sys.exit(0)
    except KeyboardInterrupt:
        say()
        return default
    return answer or default


def yes(question):
    return ask(f"{question} [y/n] ").lower().startswith("y")


def pause():
    ask("\nPress Enter to go back to the menu...")


def run(cmd, **kw):
    """Runs a program in this window and waits for it. Returns its exit code (or None if it couldn't start)."""
    try:
        return subprocess.call([str(c) for c in cmd], **kw)
    except FileNotFoundError:
        say(f"❌ Couldn't start {cmd[0]}")
        return None
    except KeyboardInterrupt:
        return None


def http_get(url, timeout=3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def ollama_models():
    """The models Ollama has, or None if Ollama isn't running."""
    raw = http_get(OLLAMA + "/api/tags")
    if raw is None:
        return None
    try:
        return [m.get("name", "") for m in json.loads(raw).get("models", [])]
    except Exception:
        return []


def port_open(host, port, timeout=2.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def download(url, dest):
    say(f"   downloading {dest.name} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=30) as r, open(tmp, "wb") as out:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r   {got / 1e6:6.0f} / {total / 1e6:.0f} MB", end="", flush=True)
        tmp.replace(dest)
        say("   done")
        return True
    except Exception as error:
        say(f"\n   ❌ the download failed: {error}")
        try:
            tmp.unlink()
        except Exception:
            pass
        return False

# ---------------------------------------------------------------------------------------------------- the checks
# Each check gives (ok, what it found) and may have a fix that the menu offers.

def check_python():
    v = sys.version_info
    return v >= (3, 11), f"Python {v.major}.{v.minor}" + ("" if v >= (3, 11) else " – Anka needs 3.11 or newer")


def check_venv():
    return PY.exists(), (f"Anka's Python environment ({VENV}, {sys.version.split()[0] if Path(sys.executable) == PY else 'found'})" if PY.exists()
                         else f"no Python environment for Anka yet ({VENV})")


def fix_venv():
    if not PY.exists():
        say(f"   creating {VENV} ...")
        if run([sys.executable, "-m", "venv", VENV]) != 0:
            return False
    say("   installing Anka's Python packages (a few minutes the first time) ...")
    run([PY, "-m", "pip", "install", "--upgrade", "pip", "-q"])
    return run([PY, "-m", "pip", "install", "-r", "requirements.txt"]) == 0


def check_packages():
    if not PY.exists():
        return False, "packages: no environment yet"
    code = "import importlib,sys\nbad=[m for m in %r if importlib.util.find_spec(m) is None]\nprint(' '.join(bad))" % PACKAGES
    try:
        out = subprocess.run([str(PY), "-c", code], capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as error:
        return False, f"packages: couldn't check ({error})"
    return not out, ("Python packages installed" if not out else f"missing Python packages: {out}")


def check_ollama():
    models = ollama_models()
    if models is not None:
        return True, "Ollama is running"
    installed = shutil.which("ollama") or (IS_MAC and Path("/Applications/Ollama.app").exists())
    return False, ("Ollama is installed but not running" if installed
                   else "Ollama is not installed (get it free at https://ollama.com)")


def fix_ollama():
    if not (shutil.which("ollama") or (IS_MAC and Path("/Applications/Ollama.app").exists())):
        say("   Install Ollama from https://ollama.com first, then choose this again.")
        return False
    say("   starting Ollama ...")
    if IS_MAC and Path("/Applications/Ollama.app").exists():
        subprocess.Popen(["open", "-a", "Ollama"])
    else:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        time.sleep(1)
        if ollama_models() is not None:
            return True
    return False


def check_memory_model():
    models = ollama_models()
    if models is None:
        return False, f"memory search model ({MEMORY_MODEL}): Ollama isn't running"
    ok = any(m.startswith(MEMORY_MODEL) for m in models)
    return ok, (f"memory search model ({MEMORY_MODEL})" if ok else f"the memory search model {MEMORY_MODEL} is missing")


def fix_memory_model():
    if ollama_models() is None and not fix_ollama():
        return False
    return run(["ollama", "pull", MEMORY_MODEL]) == 0


def check_voice():
    missing = [f for f in VOICE_FILES if not (HERE / f).exists()]
    return not missing, ("his voice (Kokoro)" if not missing else f"his voice files are missing: {', '.join(missing)}")


def fix_voice():
    ok = True
    for f in VOICE_FILES:
        if not (HERE / f).exists():
            ok = download(f"{KOKORO_URL}/{f}", HERE / f) and ok
    return ok


def check_key():
    try:
        key = KEY_FILE.read_text().strip()
    except Exception:
        key = ""
    return bool(key), ("Gemini API key (cloud_key.txt)" if key
                       else "no Gemini API key yet – Anka needs one to think (cloud_key.txt)")


def fix_key():
    say("   Anka's brain is Google's Gemini. Get an API key at https://aistudio.google.com/apikey")
    say("   (a paid key is recommended: on the free tier Google uses what you send to improve its products).")
    key = getpass.getpass("   Paste your key here (it isn't shown, and it never goes into the repository): ").strip()
    if not key:
        return False
    KEY_FILE.write_text(key + "\n")
    try:
        os.chmod(KEY_FILE, 0o600)
    except Exception:
        pass
    return True


def check_disk():
    try:
        free = shutil.disk_usage(HERE).free / 1e9
    except Exception:
        return True, "disk space: unknown"
    return free >= 10, f"{free:.0f} GB free on the disk" + ("" if free >= 10 else " – Anka works best with 10 GB or more")


def check_3d():
    """Only for the 3D simulator: Pollen's software, Rust and GStreamer."""
    missing = []
    if IS_WIN:
        return False, "3D simulator: Pollen's software doesn't support Windows (use macOS or Linux)"
    if not (POLLEN / "scripts" / "duck-sim").exists():
        missing.append(f"Pollen's microduck in {POLLEN}")
    if not POLLEN_RL.exists():
        missing.append(f"Pollen's microduck_rl in {POLLEN_RL}")
    if not shutil.which("cargo"):
        missing.append("Rust (https://rustup.rs)")
    if not shutil.which("gst-launch-1.0"):
        missing.append("GStreamer" + (" (brew install gstreamer libnice-gstreamer)" if IS_MAC else ""))
    return not missing, ("3D simulator: everything is there" if not missing
                         else "3D simulator needs: " + "; ".join(missing) + " – see docs/GUIDE.md, section 7")


CHECKS = [
    ("python", check_python, None, True),
    ("venv", check_venv, fix_venv, True),
    ("packages", check_packages, fix_venv, True),
    ("ollama", check_ollama, fix_ollama, True),
    ("memory", check_memory_model, fix_memory_model, True),
    ("voice", check_voice, fix_voice, True),
    ("key", check_key, fix_key, True),
    ("disk", check_disk, None, False),
    ("3d", check_3d, None, False),
]


def run_checks(offer_fixes=True, quiet_if_ok=False):
    """Runs every check, shows the result and offers to fix what is missing. True if all needed parts are OK."""
    problems = []
    for name, check, fix, needed in CHECKS:
        ok, words = check()
        if not quiet_if_ok or (not ok and needed):        # quiet: only what is missing and needed
            say(f"  {'✅' if ok else ('❌' if needed else '⚠️ ')} {words}")
        if not ok and needed:
            problems.append((name, fix))
    if not problems:
        if not quiet_if_ok:
            say("\n  Everything Anka needs is in place.")
        return True
    if offer_fixes:
        checks = {n: c for n, c, _, _ in CHECKS}
        offered = set()
        for name, fix in problems:
            if fix is None or fix in offered:               # (the environment and the packages share one fix)
                continue
            offered.add(fix)
            ok, words = checks[name]()
            if ok:                                        # an earlier fix may have fixed this one too
                continue
            if yes(f"\n  Fix this now? ({words})"):
                say("  ✅ fixed" if fix() else "  ❌ that didn't work – see docs/GUIDE.md")
    return all(c()[0] for _, c, _, needed in CHECKS if needed)


def ready_for_anka():
    """A quick check before starting Anka. Offers fixes; asks whether to go on anyway if something is still missing."""
    if run_checks(offer_fixes=True, quiet_if_ok=True):
        return True
    return yes("\n  Something is still missing. Start anyway?")

# ---------------------------------------------------------------------------------------------------- the actions

def start_real():
    say("\n🏠 Anka for real: your camera, microphone and speakers. Press ENTER in this window to talk; Ctrl+C puts him to sleep.\n")
    if ready_for_anka():
        run([PY, "run_anka.py"])
    pause()


def start_2d():
    say("\n🟦 Anka in the 2D simulator: a test day in a drawn room (about 13 minutes, then a night).\n")
    if ready_for_anka():
        run([PY, "run_anka.py", "--simulate", "--body"])
    pause()


def screen_size():
    """The screen's size in points (macOS), for placing the windows."""
    if IS_MAC:
        try:
            out = subprocess.run(["osascript", "-e", 'tell application "Finder" to get bounds of window of desktop'],
                                 capture_output=True, text=True, timeout=5).stdout
            x0, y0, x1, y1 = [int(v) for v in out.replace(" ", "").split(",")]
            return x1, y1
        except Exception:
            pass
    return 1440, 900


def run_simulation(args):
    """Runs a 3D launcher. On a Mac it gets its own Terminal window (bottom right), separate from this menu, and the
    window and the simulator close by themselves when Anka is asleep. Elsewhere it runs here."""
    args = list(args) + ["--auto-close"]
    w, h = screen_size()
    env = {"ANKA_ROOM_POS": "0,25", "ANKA_EYES_POS": f"{w // 2},30", "ANKA_SCREEN": f"{w},{h - 25}"}
    if not IS_MAC:
        run(["bash"] + args, env={**os.environ, **env})
        pause()
        return
    cmd = (f"cd {shlex.quote(str(HERE))} && " + " ".join(f"{k}={v}" for k, v in env.items())
           + " bash " + " ".join(shlex.quote(a) for a in args) + "; exit")
    cmd = cmd.replace("\\", "\\\\").replace('"', '\\"')
    script = (f'tell application "Terminal"\n do script "{cmd}"\n delay 0.8\n'
              f' set miniaturized of front window to true\nend tell')
    subprocess.run(["osascript", "-e", script], capture_output=True)
    say("\n  ▶️  Started. Everything shows in ONE window, \"Anka\": the room, his eyes and what is happening.")
    say("     (The text log runs in a Terminal window that is minimised in the Dock, if you want to read it.)")
    say("     When Anka is asleep, the window and the simulator close by themselves. This menu stays here.")
    pause()


def start_3d():
    say("\n🦆 Anka in Pollen's 3D simulator (MuJoCo)\n")
    ok, words = check_3d()
    if not ok:
        say(f"  ❌ {words}")
        pause()
        return
    say("  1) A normal test day                      (everything in one window)")
    say("  2) The 17-minute day that tests everything (everything in one window)")
    say("  3) The 17-minute day, with Pollen's 3D window too (slower)")
    say("  4) Only start the simulator")
    say("  5) Start Anka in a simulator that is already running")
    say("  6) Open Anka's eyes (camera and depth sensor window)")
    choice = ask("\n  Choose (Enter = back): ")
    if not choice:
        return
    if choice in ("1", "2", "3", "5") and not ready_for_anka():
        pause()
        return
    if choice == "1":
        run_simulation(["Start everything.command"])
    elif choice == "2":
        run_simulation(["Start everything.command", "--scenario", "all"])
    elif choice == "3":
        run_simulation(["Start everything.command", "--show-3d", "--scenario", "all"])
    elif choice == "4":
        run(["bash", "Start Anka sim.command"])
    elif choice == "5":
        run([PY, "run_anka.py", "--simulate", "--mujoco"])
        pause()
    elif choice == "6":
        if IS_MAC:
            subprocess.Popen(["open", "-a", "Terminal", str(HERE / "tools" / "Anka eyes.command")])
            say("  Opened in a new window.")
        else:
            run([PY, "ankasim/eyes.py"])
        pause()


def teach_face():
    say("\n👤 Teach Anka a face. Anka must be asleep (he uses the camera).\n")
    name = ask("  Whose face should he learn? (Enter = back): ")
    if not name:
        return
    if not PY.exists():
        say("  ❌ Anka isn't installed yet – choose 'Check that everything is installed' first.")
        pause()
        return
    say("\n  Sit in front of the camera, alone in the picture. Turn your head a little, smile, look serious...\n")
    run([PY, "run_anka.py", "--learn-face", name])
    run([PY, "run_anka.py", "--faces"])
    pause()


def linux_shortcut():
    """Linux: puts "Anka" in the apps menu (a .desktop file), so the menu can be started with a click."""
    apps = Path.home() / ".local" / "share" / "applications"
    entry = apps / "anka-menu.desktop"
    if entry.exists() or not yes("\n  Add Anka to your apps menu, so you can start this menu with a click?"):
        return
    apps.mkdir(parents=True, exist_ok=True)
    entry.write_text("[Desktop Entry]\nType=Application\nName=Anka\nComment=Anka's start menu\n"
                     f"Exec=bash \"{HERE / 'Anka menu (linux).sh'}\"\nPath={HERE}\nTerminal=true\n"
                     "Categories=Science;Robotics;\n")
    os.chmod(entry, 0o755)
    say(f"  ✅ added: look for 'Anka' in your apps menu ({entry})")


def check_all():
    say("\n🔎 Checking what Anka needs\n")
    run_checks(offer_fixes=True)
    if IS_LINUX:
        linux_shortcut()
    pause()

# ---------------------------------------------------------------------------------------------------- the real duck

def duck_address():
    try:
        return DUCK_FILE.read_text().strip() or None
    except Exception:
        return None


def find_duck():
    """Asks for the duck's name or address (or finds it with Pollen's duckctl, if installed), and remembers it."""
    say("  A Microduck calls itself 'duck-' plus four letters or numbers (for example duck-c51b), unless it was renamed.")
    say("  On your network it is then usually reachable as duck-c51b.local, or by its address (like 192.168.1.42).")
    if shutil.which("duckctl"):
        say("  Pollen's duckctl is installed – asking it where the duck is ...")
        try:
            out = subprocess.run(["duckctl", "open", "--print"], capture_output=True, text=True, timeout=30).stdout
            if "://" in out:
                host = out.strip().split("://", 1)[1].split("/", 1)[0].split(":", 1)[0]
                say(f"  duckctl found it at {host}")
                if yes(f"  Use {host}?"):
                    DUCK_FILE.write_text(host + "\n")
                    return host
        except Exception as error:
            say(f"  (duckctl couldn't find it: {error})")
    current = duck_address()
    host = ask(f"  The duck's name or address{' [' + current + ']' if current else ''}: ", current or "")
    if not host:
        return None
    if "." not in host and not host.replace(":", "").isdigit():
        host += ".local"
    DUCK_FILE.write_text(host + "\n")
    say(f"  Remembered: {host} (in {DUCK_FILE.name})")
    return host


def test_duck(host):
    """Tests the connection the way Pollen's own console does it: the address, the console port (8080) with its camera
    picture, the signalling port (8443) and SSH (22). Saves the duck's camera picture if it answers."""
    say(f"\n  Testing {host} ...\n")
    try:
        ip = socket.gethostbyname(host)
        say(f"  ✅ found on the network: {host} = {ip}")
    except Exception:
        say(f"  ❌ can't find {host} on the network. Is the duck switched on and on the same wifi as this computer?")
        say("     (Join it to your wifi first with Pollen's app or 'duckctl wifi connect'.)")
        return False
    console = port_open(ip, 8080)
    say(f"  {'✅' if console else '❌'} the duck's console (mediad, port 8080)" + ("" if console else " doesn't answer"))
    if console:
        frame = http_get(f"http://{ip}:8080/frame", timeout=5)
        if frame and frame[:4] == b"\x89PNG":
            out = HERE / "test_logs" / "duck_camera.png"
            out.parent.mkdir(exist_ok=True)
            out.write_bytes(frame)
            say(f"  ✅ his camera works – the picture is saved in {out.relative_to(HERE)}")
            say(f"     (open http://{ip}:8080/ in a browser to see Pollen's console with the live camera)")
        else:
            say("  ⚠️  the console answers, but no camera picture came back")
    sig = port_open(ip, 8443)
    say(f"  {'✅' if sig else '⚠️ '} the console's live connection (port 8443)" + ("" if sig else " doesn't answer – often a firewall"))
    ssh = port_open(ip, 22)
    say(f"  {'✅' if ssh else '⚠️ '} SSH (port 22)" + ("" if ssh else " doesn't answer") + " – for developer access")
    return console


def connect_duck():
    say("\n🤖 Connect to my Microduck\n")
    host = duck_address()
    if host:
        say(f"  Your duck: {host}")
    say("  1) Find my duck (enter or change its name or address)")
    say("  2) Test the connection")
    say("  3) Start Anka with the real body")
    choice = ask("\n  Choose (Enter = back): ")
    if choice == "1":
        host = find_duck()
        if host and yes("\n  Test the connection now?"):
            test_duck(host)
        pause()
    elif choice == "2":
        host = host or find_duck()
        if host:
            test_duck(host)
        pause()
    elif choice == "3":
        say("\n  The real body isn't finished yet: it will be built and tested when the duck arrives.")
        say("  Anka's 3D body (ankasim/simbody.py) already talks to the same programs the real duck runs, so most of")
        say("  it will carry over. What is left: a link from this computer to the duck over wifi, a safety stop, and")
        say("  a careful first start (standing, then walking one metre with you next to him). See STATUS.md.")
        pause()

# ---------------------------------------------------------------------------------------------------- the menu

MENU = [
    ("1", "Talk to Anka for real", "camera, microphone, speakers", start_real),
    ("2", "Watch him in 2D", "fast, no extra software", start_2d),
    ("3", "Watch him in 3D", "Pollen's MuJoCo simulator", start_3d),
    ("4", "Teach him a face", "", teach_face),
    ("5", "Check that everything is installed", "and fix what is missing", check_all),
    ("6", "Connect to my Microduck", "the real robot", connect_duck),
]


def first_start():
    """The first time (no environment yet): say hello and offer to install everything."""
    if PY.exists():
        return
    say("👋 Welcome! It looks like Anka isn't installed on this computer yet.")
    say("   This menu can set up what he needs: his Python packages, his voice, Ollama's memory model and your")
    say("   Gemini key. It takes a few minutes and about 1 GB of downloads.\n")
    if yes("   Set everything up now?"):
        say()
        run_checks(offer_fixes=True)
        pause()


def main():
    first_start()
    while True:
        say("\n" + "=" * 52)
        say("  🦆  ANKA – Microduck AI Brain")
        say("=" * 52 + "\n")
        for key, title, note, _ in MENU:
            say(f"  {key}) {title:<36}{('(' + note + ')') if note else ''}")
        say("  q) Quit\n")
        choice = ask("Choose: ").lower()
        if choice in ("q", "quit", "exit"):
            say("Bye! 🦆")
            return
        for key, _, _, action in MENU:
            if choice == key:
                try:
                    action()
                except KeyboardInterrupt:
                    say("\n(stopped)")
                break
        else:
            if choice:
                say("Choose one of the numbers, or q to quit.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        say("\nBye! 🦆")
