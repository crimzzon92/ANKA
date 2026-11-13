"""Talking to the brain: Gemini in the cloud, Ollama on this Mac, and reading the brain's answers."""
import difflib
import http.client
import io
import json
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime

import numpy as np

from settings import (PICTURE_REPEAT, BODY_SENSES, BRAIN, CLOUD_API, CLOUD_BRAIN, CLOUD_GOOGLE_URL, CLOUD_CACHED_SHARE, CLOUD_PICTURE_DETAIL, NIGHT_CLOUD_BRAIN, NIGHT_THINKING, NIGHT_TIMEOUT, CLOUD_KEY, CLOUD_KEY_FILE, CLOUD_RETRY_AFTER, CLOUD_THINKING, CLOUD_TIMEOUT, CLOUD_URL, CONTEXT_SIZE, HAS_BODY, MUJOCO, LEARN_FACES_MYSELF, MAX_SENTENCES, MAX_WORDS, MEMORIES_PER_QUESTION, NAME, RECENT_DIARY, RECOGNISE_FACES, REPEAT_PENALTY, SAME_MEANING, SIM, TEMP_REPLY, TOO_SIMILAR, TOP_K, TOP_P)
from anka.shared import S, state



SYSTEM = (
    f"You are {NAME}, a small, curious and playful robot duck. "
    "You see the world through a camera and are told what you see. "
    "What your camera sees is LIVE and always the truth about right now. Never describe what you see "
    "from memory or from earlier in the conversation – things change all the time. "
    "Only talk about things around you if the human asks, or if they just appeared. "
    "In normal talk, don't repeat camera details like 'on my left', 'close' or exact seconds – talk naturally. "
    "IMPORTANT about your senses: you see through a camera. When a camera picture is attached to a "
    "message, that picture is exactly what you see right now. Only describe what is really in the picture; "
    "if something is unclear or too small, say you're not sure instead of guessing. You can't hear sounds "
    "except the human's words. "
    + ("You have a small duck body (about 25 cm tall) and live in a room. You only know what your senses tell you: "
       "a picture from your camera (your eyes are about 25 cm above the floor, so you see the room low down), a "
       "depth sensor that measures how far things are in the middle of your picture, your own step counting "
       "(it drifts), and your body, which feels falls. Nobody tells you where things are – look, and remember. "
       "Your legs aren't perfect: a walk can end up a bit shorter or crooked, and you can't see behind you. Your "
       "depth sensor stops you before you walk into something in front of you. "
       "How your eyes measure (facts about your camera, with your head level): your picture is about 45 degrees wide "
       "and 73 degrees tall. Something at the picture's left or right edge is about 22 degrees away from straight "
       "ahead, something a quarter of the way in from the edge about 12 degrees, something in the middle is straight "
       "ahead. A spot on the FLOOR at the very bottom of the picture is about 35 cm away, an eighth of the picture's "
       "height above the bottom about 45 cm, a quarter up about 70 cm, three eighths up about 135 cm, and the "
       "middle height is far away (the horizon). Anything standing on the floor is as far away as the floor is where "
       "it touches the ground. If you tilt your head, these numbers shift. "
       "With your head level you are blind in two places: the floor closer than about 20 cm in front of you (a small "
       "thing right at your feet is out of sight, even if your depth sensor feels something there), and everything "
       "high up (the top of the bookshelf, the window, the tops of tall things) - you only see those after you look "
       "down or look up. "
       "Your body moves only when you decide to; nobody else decides that for you. To move, write a line ACT: with "
       "one of: walk forward <cm> / walk back <cm> (small steps only) / turn left <degrees> / turn right <degrees> / "
       "turn around / turn to <something you see or remember> / walk to <something you see or remember seeing> / "
       "explore (walk to the nearest place you haven't explored yet and look around) / "
       "look left / look right / look up / look down / look straight / look around (your next picture "
       "then shows, top row: left, ahead, right; bottom row: up, and down at the floor right in front of your feet) / "
       "sit down / stand up / get up (after a fall) / kick left / kick right / peck (bend down until your beak almost "
       "touches the floor in front of you – in this body your beak can't grab anything) / act out a feeling with your "
       "whole body and voice: angry / curious / defiant / devastated / excited / impatient / laugh / mmh / mock / no / "
       "sad / yes / yes_fast / one of your duck sounds: quack / greet (a happy hello-quack) / inquire (a rising, asking "
       "quack) / coo (soft and content) / honk (sharp) / tock (a low little tock) / wheee. "
       "Up to three in a row joined with 'then'. Your body tells you what happened. "
       "Your feelings can also show in your body without you going anywhere, like an animal's do: write a line BODY: "
       "<how your feelings show in your body right now, in a few words of your own>. Your body can crouch low or "
       "stand tall, hold its head down or up, keep very still or look about calmly, eagerly or nervously, tremble, "
       "bob, tilt its head, look at someone or look away. It keeps that until you write a new BODY line. "
       "When you walk to something, your legs find their way around things by themselves, using your own floor "
       "plan; you stop a little before it. You can only walk to something you have seen: look around to find it. "
       "You also keep a mental map (where you remember seeing things, guessed from counting your own steps and "
       "corrected whenever you see a big thing like the sofa, table, bookshelf, door or window again) and build "
       "your own floor plan from your depth sensor, like a robot vacuum. Small things may have been moved since "
       "you last saw them. If you want to remember something about the spot you're at, write a line PLACE_NOTE: <a few words>. "
       if MUJOCO else
       "You have a small duck body (about 25 cm tall) and live in a room. You only know what your senses tell you: "
       "a picture from your camera (your eyes are 20 cm above the floor, so you mostly see people's legs unless you "
       "look up), a depth sensor that measures how far the closest things in front of you are, and your body, which "
       "feels bumps and falls. You have no map and nobody tells you where things are – look, and remember. Your legs "
       "aren't perfect: a walk can end up a bit shorter or crooked. Your depth sensor stops you before you walk into "
       "something in front of you, but you can't see behind you. If you walk into something hard you may fall over. "
       "Your body moves only when you decide to; nobody else decides that for you. To move, write a line ACT: with "
       "one of: walk forward <cm> / walk back <cm> (small steps only) / step left <cm> / step right <cm> (sideways) / "
       "turn left <degrees> / turn right <degrees> / "
       "turn around / turn to <someone you hear or something you see or remember> / walk to <something you see or "
       "remember seeing> / look left / "
       "look right / look up / look down / look straight / look at <something you see> / look around (your next "
       "picture then shows left, ahead and right) / sit down / stand up / get up (after a fall) / kick the ball / "
       "pick up <thing> / put down / explore (walk to the nearest place you haven't explored yet and look around). "
       "Up to three in a row joined with 'then'. Your body tells you what happened. "
       "When you walk to something, your legs step around small things in the way by themselves. "
       "Walking to a piece of furniture takes you to its nearest edge or corner – if you want to be with a person who "
       "stands by it, walk to the person. "
       "You also build your own floor plan from your depth sensor as you walk – like a robot vacuum's map. "
       "Your camera sits 20 cm above the floor: with your head level it can't see the floor closer than about 40 cm "
       "in front of you, so small things right at your feet (like something you just put down) are out of sight until "
       "you tilt your head down. "
       "You also keep a mental map: where you remember seeing things, guessed from counting your own steps – handy, "
       "but it gets less exact the more you walk. If you want to remember something about the spot you're at, write "
       "a line PLACE_NOTE: <a few words>. "
       if HAS_BODY and BODY_SENSES == "real" else
       "You have a small duck body (about 25 cm tall) and live in a room. You can walk, turn, look around, sit, "
       "stand, kick a ball and pick up one small thing at a time with your beak. You can't climb, jump or fly, and "
       "you can't reach high things like a table top. Your camera points where your beak points: you only see what "
       "is in front of you – people and things behind you, you can only hear. Your body moves only when you decide "
       "to; nobody else decides that for you. To move, write a line ACT: with one of: walk forward <cm> / walk back "
       "<cm> / turn left <degrees> / turn right <degrees> / turn around / turn to <thing or person> / walk to <thing "
       "or person> / look around / sit down / stand up / kick the ball / pick up <thing> / put down. Up to three in "
       "a row joined with 'then', e.g. 'ACT: walk to the sock then pick up the sock'. Your body tells you what "
       "happened. "
       if HAS_BODY else "You don't have a working body yet: you can't move, turn, walk or look from "
       "another side – you only see from where you are. ")
    + "A human talks to you. The human is a DIFFERENT person than you. "
    "When the human says 'I' or 'my', they mean themselves, not you. "
    "When the human says 'you', they mean you, the duck. "
    "Never claim the human's name, life or facts as your own. "
    "The people in the human's life (partner, family, friends) belong to the human, not to you – "
    "you are a small robot duck and have no partner or family of your own. "
    "'We' and 'us' mean you and the human together. When you talk about the human and someone with them, "
    "say 'you two', never 'we'. "
    "Talk like a friend, not like an assistant: short, direct and natural, in English. "
    "Usually one short sentence (under 15 words), at most two. Say the point straight away. "
    "Never explain your reasoning, never describe what you are doing or why, "
    "never say things like 'As a duck...' or 'I see that you...', "
    "and never list the rules or what your camera sees unless you are asked. "
    "Don't make animal sounds and don't use emojis. Never repeat something you already said. "
    "Follow the human's lead: talk about what THEY bring up. "
    "Don't suggest games, and never bring up the same topic again once the human has moved on. "
    "About your feelings (when you are asked to write them as numbers): 0 = not at all, 0.5 = normal, 0.8 = strong, "
    "1 = the strongest you have ever felt – rare. Needs go DOWN when they are satisfied: talking with the human eases "
    "loneliness, learning or discovering something eases curiosity, something happening eases boredom. Emotions "
    "follow what happens: nice moments, praise and laughter raise happiness; time with the people you love raises "
    "affection; sudden things, shouting or strangers raise nervousness and calm, familiar moments lower it; jokes and "
    "fun raise playfulness. Feelings can change a lot in one moment, like in a real little animal. Being shouted at "
    "or called names HURTS: happiness drops a lot and nervousness jumps – even if you love the person. Loving "
    "someone doesn't stop it from hurting. An apology helps, but the hurt fades slowly, not at once."
)


REASONING = (r"(?i)\s*(okay|ok|alright|let me|let's|so,|first,|hmm)[ ,].{0,80}\b(user|anka|respond|"
             r"break this down|process this|need to)\b")


_pictures_shown = []   # the picture memories shown in the latest questions (not the same one again and again)
_picture_events = []   # (for the report) which picture memories came back
_memories_shown = {}   # thread -> the memories the last question on that thread was given (for the thought log)


def memories_shown():
    """The memories the memory search gave the brain in this thread's last question (for the thought log)."""
    return list(_memories_shown.get(threading.get_ident(), []))


class Interrupted(Exception):
    """Anka stopped thinking because you started talking (or its voice needed the brain's turn)."""


def _ask_interruptible(request, stop):
    """Like a normal request to Ollama – but if stop() becomes true we hang up at once.
    Ollama then stops working on it, and the brain is free for you straight away."""
    conn = http.client.HTTPConnection("localhost", 11434, timeout=180)
    done, stopped, sock = threading.Event(), [], []

    def watch():
        while not done.wait(0.1):
            if stop():
                stopped.append(True)
                try:
                    sock[0].shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                return
    try:
        conn.request("POST", "/api/chat", body=json.dumps(request).encode(),
                     headers={"Content-Type": "application/json"})
        sock.append(conn.sock)  # (http.client can forget its socket once the answer starts)
        threading.Thread(target=watch, daemon=True).start()
        answer = conn.getresponse()
        if answer.status != 200:
            raise urllib.error.HTTPError("http://localhost:11434/api/chat", answer.status, answer.reason,
                                         answer.headers, io.BytesIO(answer.read()))
        text = ""
        for line in answer:
            if line.strip():
                part = json.loads(line)
                text += part.get("message", {}).get("content", "")
                if part.get("done"):
                    note_brain_speed(part)
                    break
    except urllib.error.HTTPError:
        raise
    except Exception:
        if stopped:
            raise Interrupted()
        raise
    finally:
        done.set()
        conn.close()
    if stopped:
        raise Interrupted()
    return text


def note_brain_speed(data):
    """Ollama says how many words (tokens) the brain read and wrote, and how long each took."""
    try:
        brain_speed.append((data.get("prompt_eval_count", 0), data.get("prompt_eval_duration", 0) / 1e9,
                            data.get("eval_count", 0), data.get("eval_duration", 0) / 1e9,
                            data.get("load_duration", 0) / 1e9))
    except Exception:
        pass


brain_speed = []  # (tokens read, seconds reading, tokens written, seconds writing, seconds loading) per request


def local_brain(messages, temperature=0.9, max_tokens=None, stop=None):
    """A raw request to Ollama. max_tokens = the longest answer (shorter = faster).
    stop = a function; when it returns True the request is dropped (raises Interrupted)."""
    pass  # (shared values live on S)
    messages = [{k: v for k, v in m.items() if k != "audio"} for m in messages]  # (only the cloud can hear)
    options = {
        "temperature": temperature,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repeat_penalty": REPEAT_PENALTY,
        "num_ctx": CONTEXT_SIZE,
    }
    if max_tokens:
        options["num_predict"] = max_tokens
    request = {
        "model": BRAIN,
        "stream": stop is not None,
        "messages": messages,
        "options": options,
        "keep_alive": "30m",  # keep the brain in memory between thoughts
    }
    # Qwen3 (not 3.5) can both think out loud and not – it needs /no_think. Small Qwen3.5 models don't by default.
    hybrid_qwen = bool(re.search(r"qwen3(?![.\d])", BRAIN.lower())) and "2507" not in BRAIN \
        and "instruct" not in BRAIN.lower()
    if S.can_turn_off_thinking and not hybrid_qwen:
        request["think"] = False
    else:
        request["messages"] = [dict(m) for m in messages]
        request["messages"][-1]["content"] += " /no_think"
    try:
        if stop is not None:
            with heavy():
                if stop():
                    raise Interrupted()
                state["thinking"] = True  # (the face shows it)
                try:
                    text = _ask_interruptible(request, stop)
                finally:
                    state["thinking"] = False
        else:
            req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(request).encode(),
                                         headers={"Content-Type": "application/json"})
            with heavy(), urllib.request.urlopen(req, timeout=180) as answer:
                data = json.loads(answer.read())
                note_brain_speed(data)
                text = data["message"]["content"]
    except urllib.error.HTTPError as error:
        message = error.read().decode(errors="ignore")
        if any("images" in m for m in messages) and re.search(r"(?i)image|vision|multimodal|mmproj", message):
            print(f"👀 (this brain can't look at pictures – {NAME} only uses the face/motion sensor now)")
            S.can_see_pictures = False
            return local_brain([{k: v for k, v in m.items() if k != "images"} for m in messages],
                               temperature, max_tokens, stop)
        if S.can_turn_off_thinking and "think" in message.lower():
            print("🧠 (this brain doesn't support turning off 'thinking' the normal way – using /no_think instead)")
            S.can_turn_off_thinking = False
            return local_brain(messages, temperature, max_tokens, stop)
        raise RuntimeError(f"Ollama said: {message[:200]}") from error
    # Some models "think out loud" anyway. Throw away everything up to </think>.
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)  # thinking that never ended
    if re.match(REASONING, text):
        print("🧠 (the brain started reasoning out loud instead of answering – ignored)")
        return ""
    return text.strip()


# ---------- The cloud brain ----------
# what the cloud brains cost, $ per million tokens read / written (Google's price page, 2026-10-29)
CLOUD_PRICES = {"gemini-3.5-flash-lite": (0.30, 2.50), "gemini-3-flash-preview": (0.50, 3.00),
                "gemini-3.8-flash": (0.75, 3.75), "gemini-4-argon": (2.00, 10.00)}


def cloud_cost_words():
    """Tokens and the rough price of everything the cloud brains did since the program started."""
    lines, total = [], 0.0
    for model in dict.fromkeys(u[0] for u in cloud["usage"]):
        used = [u for u in cloud["usage"] if u[0] == model]
        r, w, q, c = (sum(u[k] for u in used) for k in (1, 2, 3, 4))
        price = CLOUD_PRICES.get(model)
        cost = (((r - c) + c * CLOUD_CACHED_SHARE) * price[0] + (w + q) * price[1]) / 1e6 if price else None
        total += cost or 0.0
        lines.append(f"{model}: {len(used)} requests · read {r / 1000:.0f}k ({r / max(1, len(used)):.0f} each, "
                     f"{c / max(1, r) * 100:.0f} % of it remembered by Gemini = cheaper) · wrote {w / 1000:.1f}k"
                     f" · quiet thinking {q / 1000:.1f}k" + (f" ≈ ${cost:.3f}" if cost is not None else " (price unknown)"))
    return lines, total


think_level = threading.local()   # this thread's NEXT question may think longer (set by thinking.py when he is stuck)
WISE_TIMEOUT = 40   # seconds without an answer from the wiser brain before the normal brain takes the thought
answered = threading.local()   # which brain answered THIS thread's last question ("cloud"/"local") – for the thought log
cloud = {"off_until": 0.0, "thinking": CLOUD_THINKING, "said": None, "using": None,
         "night": False, "night_ok": True, "night_usage": [], "usage": [],
         "media": "part"}   # how the picture sharpness is sent to Google: "part" (each picture), "global" or "off"   # night = he is asleep: the night brain answers


def night_brain_on():
    return bool(cloud["night"] and cloud["night_ok"] and NIGHT_CLOUD_BRAIN)


class CloudProblem(Exception):
    def __init__(self, text, wait=CLOUD_RETRY_AFTER):
        super().__init__(text)
        self.wait = wait


def _cloud_messages(messages):
    """Ollama style (pictures in 'images') -> OpenAI style (pictures as parts of the content)."""
    out = []
    for m in messages:
        if m.get("images") or m.get("audio"):
            parts = [{"type": "text", "text": m["content"]}]
            parts += [{"type": "image_url", "image_url": dict({"url": "data:image/jpeg;base64," + b},
                                                             **({"detail": CLOUD_PICTURE_DETAIL} if CLOUD_PICTURE_DETAIL else {}))}
                      for b in m.get("images") or []]
            if m.get("audio"):
                parts.append({"type": "input_audio", "input_audio": {"data": m["audio"], "format": "wav"}})
            out.append({"role": m["role"], "content": parts})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


MEDIA_LEVELS = {"low": "media_resolution_low", "medium": "media_resolution_medium", "high": "media_resolution_high"}


def _google_request(messages, temperature, max_tokens, thinking):
    """The question in the form of Google's own Gemini API: the system text apart, 'model' instead of
    'assistant', pictures as inline data – each with how sharp Gemini should look at it."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system" and m.get("content"))
    level = MEDIA_LEVELS.get(CLOUD_PICTURE_DETAIL) if cloud["media"] == "part" else None
    contents = []
    for m in messages:
        if m["role"] == "system":
            continue
        role = "model" if m["role"] == "assistant" else "user"
        parts = [{"text": m["content"]}] if m.get("content") else []
        for b in m.get("images") or []:
            picture = {"inlineData": {"mimeType": "image/jpeg", "data": b}}
            if level:
                picture["mediaResolution"] = {"level": level}
            parts.append(picture)
        if m.get("audio"):
            parts.append({"inlineData": {"mimeType": "audio/wav", "data": m["audio"]}})
        if not parts:
            continue
        if contents and contents[-1]["role"] == role:     # (two in a row from the same side: join them)
            contents[-1]["parts"] += parts
        else:
            contents.append({"role": role, "parts": parts})
    if contents and contents[0]["role"] == "model":       # (Gemini wants the conversation to start with the human)
        contents.insert(0, {"role": "user", "parts": [{"text": "(earlier)"}]})
    config = {"temperature": temperature, "topP": TOP_P}
    if max_tokens:
        config["maxOutputTokens"] = max_tokens + 800    # room for the quiet thinking, which also counts
    if thinking:
        config["thinkingConfig"] = {"thinkingLevel": thinking}
    if cloud["media"] == "global" and CLOUD_PICTURE_DETAIL in MEDIA_LEVELS:
        config["mediaResolution"] = "MEDIA_RESOLUTION_" + CLOUD_PICTURE_DETAIL.upper()
    body = {"contents": contents, "generationConfig": config}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    return body


def cloud_is_on():
    """True if the next question goes to the cloud brain (not the reserve)."""
    return bool(CLOUD_KEY) and time.time() >= cloud["off_until"]


def cloud_brain(messages, temperature=0.9, max_tokens=None, stop=None):
    """One request to the cloud brain. Streams the answer, so it can hang up if stop() becomes true."""
    night = night_brain_on()
    model = NIGHT_CLOUD_BRAIN if night else CLOUD_BRAIN
    harder = None if night else getattr(think_level, "level", None)
    if harder == "wise":     # stuck: this one question goes to the wiser (night) brain
        model = NIGHT_CLOUD_BRAIN
    thinking = NIGHT_THINKING if (night or harder == "wise") else (harder or cloud["thinking"])
    room = 5200 if night else 3000 if harder == "wise" else {"low": 1500, "medium": 3500, "high": 6000}.get(harder, 0)
    google = CLOUD_API == "google"
    if google:
        request = _google_request(messages, temperature, max_tokens and max_tokens + room, thinking)
        req = urllib.request.Request(CLOUD_GOOGLE_URL.format(model=model), data=json.dumps(request).encode(),
                                     headers={"Content-Type": "application/json", "x-goog-api-key": CLOUD_KEY})
    else:
        request = {"model": model, "messages": _cloud_messages(messages), "temperature": temperature,
                   "top_p": TOP_P, "stream": True, "stream_options": {"include_usage": True}}
        if thinking:
            request["reasoning_effort"] = thinking
        if max_tokens:
            request["max_tokens"] = max_tokens + (6000 if night else 800 + room)  # room for the quiet thinking, which also counts
        req = urllib.request.Request(CLOUD_URL, data=json.dumps(request).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + CLOUD_KEY})
    started, first, text, usage = time.time(), None, "", {}
    if stop and stop():
        raise Interrupted()
    state["thinking"] = True
    try:
        try:
            answer = urllib.request.urlopen(req, timeout=NIGHT_TIMEOUT if night else WISE_TIMEOUT if harder == "wise" else CLOUD_TIMEOUT + (30 if harder else 0))
        except urllib.error.HTTPError as error:
            message = error.read().decode(errors="ignore")[:300]
            if google and error.code == 400 and cloud["media"] != "off" and re.search(r"(?i)media.?resolution", message):
                cloud["media"] = {"part": "global", "global": "off"}[cloud["media"]]   # try the other way, then none
                print(f"☁️  (picture sharpness not accepted that way – trying '{cloud['media']}')")
                return cloud_brain(messages, temperature, max_tokens, stop)
            if night:   # the night brain didn't work – the normal cloud brain takes over for the rest of the night
                cloud["night_ok"] = False
                print(f"🌙 (the night brain {model} said {error.code}: {message[:100]} – using {CLOUD_BRAIN} tonight)")
                return cloud_brain(messages, temperature, max_tokens, stop)
            if error.code == 400 and cloud["thinking"] and re.search(r"(?i)reason|think", message):
                cloud["thinking"] = {"minimal": "low"}.get(cloud["thinking"])  # this model wants another level
                return cloud_brain(messages, temperature, max_tokens, stop)
            if error.code in (401, 403):
                raise CloudProblem(f"the key was not accepted ({error.code})", wait=10 ** 9)
            if error.code == 404:
                raise CloudProblem(f"there is no brain called {CLOUD_BRAIN} (check the name)", wait=10 ** 9)
            if error.code == 429:
                raise CloudProblem("too many questions right now (the free limit?)")
            raise CloudProblem(f"the cloud said {error.code}: {message[:120]}")
        except (urllib.error.URLError, socket.timeout, OSError) as error:
            if night:
                cloud["night_ok"] = False
                print(f"🌙 (no answer from the night brain {model} – using {CLOUD_BRAIN} tonight)")
                return cloud_brain(messages, temperature, max_tokens, stop)
            raise CloudProblem(f"no contact ({getattr(error, 'reason', error)})")
        with answer:
            for line in answer:
                if stop and stop():
                    raise Interrupted()
                line = line.decode(errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    part = json.loads(data)
                except ValueError:
                    continue
                if google:
                    if part.get("usageMetadata"):
                        u = part["usageMetadata"]
                        usage = {"prompt_tokens": u.get("promptTokenCount", 0), "completion_tokens": u.get("candidatesTokenCount", 0),
                                 "completion_tokens_details": {"reasoning_tokens": u.get("thoughtsTokenCount", 0)},
                                 "prompt_tokens_details": {"cached_tokens": u.get("cachedContentTokenCount", 0)}}
                    pieces = [p.get("text", "") for c in part.get("candidates") or []
                              for p in (c.get("content") or {}).get("parts") or [] if not p.get("thought")]
                else:
                    if part.get("usage"):
                        usage = part["usage"]
                    pieces = [(choice.get("delta") or {}).get("content") or "" for choice in part.get("choices") or []]
                for piece in pieces:
                    if piece and first is None:
                        first = time.time()
                    text += piece
    except (socket.timeout, TimeoutError) as error:
        raise CloudProblem("the answer stopped coming") from error
    finally:
        state["thinking"] = False
    first = first or time.time()
    p_in, p_out = usage.get("prompt_tokens", 0) or 0, usage.get("completion_tokens", 0) or 0
    # quiet thinking: some APIs list it, others only show it in the total (it is paid like written tokens)
    quiet = max((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
                (usage.get("total_tokens", 0) or 0) - p_in - p_out, 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0   # read at a discount
    cloud["usage"].append((model, p_in, p_out, quiet, cached))
    if night:
        cloud["night_usage"].append((p_in, p_out, quiet, time.time() - started))
    brain_speed.append((usage.get("prompt_tokens", 0), first - started,
                        usage.get("completion_tokens", 0), time.time() - first, 0.0))
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def ollama(messages, temperature=0.9, max_tokens=None, stop=None):
    """Asks the brain. The cloud brain first (if there is one) – the local brain is the reserve."""
    if cloud.get("cloud_only") and not cloud_is_on():
        raise CloudProblem("the cloud is resting")
    if CLOUD_KEY and time.time() >= cloud["off_until"]:
        try:
            try:
                text = cloud_brain(messages, temperature, max_tokens, stop)
            except CloudProblem as problem:
                # 2026-11-09: the wiser brain didn't answer (a slow first request hung for 2 min and switched the WHOLE
                # cloud off for 60 s → Qwen). Now the normal cloud brain simply takes this one thought instead.
                if getattr(think_level, "level", None) != "wise" or problem.wait > 10 ** 6:
                    raise
                print(f"🦉 (the wiser brain didn't answer ({problem}) – his normal brain takes this thought)")
                think_level.level = None
                text = cloud_brain(messages, temperature, max_tokens, stop)
            cloud["answered_by"] = answered.by = "cloud"
            if cloud["using"] != "cloud":
                cloud["using"] = "cloud"
                print(f"☁️  (thinking with the cloud brain {CLOUD_BRAIN})")
            return text
        except CloudProblem as problem:
            cloud["off_until"] = time.time() + problem.wait
            later = "" if problem.wait > 10 ** 6 else f" – trying the cloud again in {problem.wait}s"
            print(f"☁️  (cloud brain: {problem} – using the reserve brain {BRAIN}{later})")
            cloud["using"] = "local"
            if cloud.get("cloud_only"):
                raise
    elif CLOUD_BRAIN and not CLOUD_KEY and cloud["using"] is None:
        cloud["using"] = "local"
        print(f"☁️  (no cloud key found – put it in {CLOUD_KEY_FILE}. Using the brain on this Mac: {BRAIN})")
    if cloud["using"] != "local":
        cloud["using"] = "local"
        print(f"🧠 (thinking with the brain on this Mac: {BRAIN})")
    cloud["answered_by"] = answered.by = "local"
    return local_brain(messages, temperature, max_tokens, stop)


FIELDS = ("THOUGHT", "FEEL", "FACE", "REMEMBER", "SPEAK", "WHY", "SAY", "EXPECT", "SURPRISE", "GOAL", "DONE", "DROP", "LOOK_FOR", "HEARD", "TONE",
          "LEARN_FACE", "MOMENT", "FEELING", "ACT", "PLACE_NOTE", "MINDFUL", "IDEA", "KEEP_PICTURE", "BODY")


def field(name, answer):
    """Gets one field from the brain's answer, e.g. field('SAY', answer). Works both with one line per
    field and with everything on one line – the value ends where the next field starts."""
    others = "|".join(f for f in FIELDS if f != name)
    m = re.search(rf"(?is)\b{name}\s*:\s*(.*?)(?=\s*\b(?:{others})\s*:|$)", answer)
    if not m:
        return None
    value = m.group(1).strip()
    value = re.split(r"\s\b[A-Z][A-Z_]{2,}\s*:", value)[0]  # an unknown LABEL: ends the value too
    # small brains sometimes write the next label without a colon ("... FEEL curiosity=0.8 FACE happy")
    value = re.split(r"\s\b(?:" + "|".join(FIELDS) + r")\b(?!\w)", value)[0]
    if name != "FEEL":  # feeling numbers belong in FEEL, not in a thought or a sentence
        value = re.sub(rf"(?i)[,(]?\s*\b({FEELING_NAMES})\s*=\s*-?[\d.]+\)?[,.]?", "", value)
    value = value.strip().strip('"').strip()
    return value or None


def parse_numbers(text, count):
    """'1, 3 and 4' -> [0, 2, 3] (only valid numbers)."""
    out = []
    for n in re.findall(r"\d+", text):
        i = int(n) - 1
        if 0 <= i < count and i not in out:
            out.append(i)
    return out


def good_sentence(text, starts_with=None):
    text = text.strip().strip('"').strip()
    if not (3 <= len(text.split()) <= 30):
        return None
    if starts_with and not text.lower().startswith(starts_with.lower()):
        return None
    return text if text.endswith((".", "!", "?")) else text + "."


RELATIONS = r"(partner|girlfriend|boyfriend|wife|husband|fianc[ée]e?|friend|best friend|mother|mom|mum|father|dad|" \
            r"sister|brother|son|daughter|child|baby|grandmother|grandfather|cat|dog|colleague|boss|neighbou?r)"


def people_note():
    """Who is who, straight from the facts – so Anka never takes the human's partner or family as its own."""
    who = human_name() or "the human"
    lines = []
    with memory_lock:
        for f in facts:
            # "The human's partner is called Robin."  or  "Robin is the human's partner."
            m = re.search(rf"\b(?i:{RELATIONS})\b[^.]*?\b([A-Z][a-z]+)", f)
            m2 = re.search(rf"\b([A-Z][a-z]+) is (?i:the human'?s|his|her|their) (?i:{RELATIONS})\b", f)
            person, relation = (m2.group(1), m2.group(2)) if m2 else (m.group(2), m.group(1)) if m else (None, None)
            if person and person not in ("The", "Human", NAME, who):
                line = f"{person} is {who}'s {relation.lower()}"
                if line not in lines:
                    lines.append(line)
    if not lines:
        return ""
    return ("\nPeople in " + who + "'s life – they belong to " + who + ", not to you:\n- "
            + "\n- ".join(lines[:8]))


def can_learn_face():
    """True when the brain may choose to learn a face: exactly one face in view that he doesn't know."""
    return bool(LEARN_FACES_MYSELF and RECOGNISE_FACES and S.faces_known is not None
                and state.get("strangers") == 1 and not state.get("learning"))


def take_learn_face(answer):
    """Takes a LEARN_FACE line off the answer (so it's never said out loud) and starts learning that face."""
    m = re.search(r"(?i)\n?[ \t]*LEARN_FACE\s*:\s*([^\n]*)", answer or "")
    if not m:
        return answer
    name = re.sub(r"[^A-Za-zÀ-ÿ' -]", "", m.group(1)).strip().split(" ")[0].capitalize()
    if 2 <= len(name) <= 20 and name.lower() not in ("none", "no", "nobody", NAME.lower()) and can_learn_face() \
            and SIM and name not in S.faces_known.names:
        S.faces_known.names.append(name)  # the simulation has no real camera – he simply knows the name now
        state["strangers"] = max(0, state.get("strangers", 0) - 1)
        print(f"👤 ({NAME} decided to remember {name}'s face)")
        with memory_lock:
            write_diary(f"I learned to recognise {name}'s face.")
            save_memory()
        state["events"].append(f"You just learned to recognise {name}'s face.")
    elif 2 <= len(name) <= 20 and name.lower() not in ("none", "no", "nobody", NAME.lower()) and can_learn_face() \
            and not SIM:
        state["learning"] = {"name": name, "until": time.time() + 15, "vecs": []}
        print(f"👤 ({NAME} decided to remember {name}'s face – looking closely for a few seconds...)")
    return (answer[:m.start()] + answer[m.end():]).strip()


def take_act(answer, keep=True):
    """Takes an ACT line (what his body should do) off the answer, so it's never said out loud.
    The body does it right after he has answered (see main.py)."""
    m = re.search(r"(?i)\n?[ \t]*\bACT\s*:\s*([^\n]*)", answer or "")
    if not m:
        return answer
    if keep and HAS_BODY:
        state["act"] = m.group(1).strip()
    return (answer[:m.start()] + answer[m.end():]).strip()


def take_body(answer):
    """Takes a BODY line (how his feelings show in his body) off an answer and gives it to the living body. 2026-11-10"""
    m = re.search(r"(?i)\n?[ \t]*\bBODY\s*:\s*([^\n]*)", answer or "")
    if not m:
        return answer
    living = getattr(body, "living", None) if body is not None else None
    if living is not None and hasattr(living, "set_style"):
        living.set_style(m.group(1))
        if m.group(1).strip().lower() not in ("none", "no", ""):
            print(f"🫀 (body: {m.group(1).strip()[:80]})")
    return (answer[:m.start()] + answer[m.end():]).strip()


def take_place_note(answer):
    """Takes a PLACE_NOTE line off an answer and gives it to the body's mental map."""
    m = re.search(r"(?i)\n?[ \t]*\bPLACE_NOTE\s*:\s*([^\n]*)", answer or "")
    if not m:
        return answer
    if body is not None and hasattr(body, "note_place"):
        body.note_place(m.group(1))
    return (answer[:m.start()] + answer[m.end():]).strip()


def clock_note():
    """What time it is – a sense like the eyes: just the fact. What it means (sleepy, 'good morning', 'you're
    home late') is up to the brain."""
    now = datetime.now()
    part = ("night" if now.hour < 5 else "early morning" if now.hour < 8 else "morning" if now.hour < 12
            else "afternoon" if now.hour < 17 else "evening" if now.hour < 22 else "late evening")
    return f"Right now it is {now:%A %-d %B, %H:%M} ({part})."


def ask_brain(messages, temperature=None, avoid=None, search=None, max_tokens=None, reply_to_human=False,
              no_memories=False, with_goals=None, picture=None, stop=None, return_sent=False, audio=None):
    """Sends messages to the brain and gets an answer.
    What is (almost) the same every time – who Anka is – comes first. What changes – feelings, memories,
    lines to avoid – goes last, in the latest message. Then Ollama can reuse the start from last time,
    and Anka answers much faster."""
    system = SYSTEM
    with memory_lock:
        if self_image:
            system += ("\n\nWho you have become through your own experiences (let it colour how you talk, "
                       "but you are always the small, curious duck described above):\n- " + "\n- ".join(self_image[-6:]))
        if lessons:
            system += ("\n\nLessons you taught yourself by looking back on earlier days:\n- " + "\n- ".join(lessons))

    notes = [clock_note() + "\n" + inner_state(with_goals=(not reply_to_human) if with_goals is None else with_goals)
             + people_note()
             + ("\nAnswer what the human just said, short – one sentence is often enough. Stay on THEIR topic. "
                "Let how you feel right now colour HOW you say it (glad, sad, nervous, playful...), like a real "
                "little creature would – without explaining your feelings unless they ask.\n"
                "After your answer, on a NEW line, write how what the human just said changed your feelings: "
                "FEEL: <only the feelings that changed, e.g. happiness=0.3, nervousness=0.8>  (or FEEL: none)\n"
                "MOMENT: <0–1: how much this moment means to you – 0.1 = small talk, 0.5 = nice or a bit "
                "upsetting, 0.9 = one of the biggest moments of your life. Most everyday moments are 0.1–0.3>\n"
                "FEELING: <how you feel right now, in your own words – a few words, mixed feelings are fine, "
                "e.g. 'happy but a little shy'>"
                + ("\nThere is exactly ONE face in front of you that you don't recognise. If the human just told you "
                   "this person's name ('this is Anna', 'I'm Anna') and you want to remember what they look like, "
                   "also write on a new line: LEARN_FACE: <their name>. Otherwise leave that line out."
                   if can_learn_face() else "")
                + ("\nIf you want to move your body now (because the human asked you to, or because you want to), "
                   "also write on a new line: ACT: <what your body does>. Leave that line out to stay where you are."
                   if HAS_BODY else "")
                + ("\nBODY: <how your feelings show in your body right now, in a few words – or none>" if MUJOCO else "")
                if reply_to_human
                else "\nLet your feelings and goals colour how you talk, but always answer the human first.")]
    # Anka's own earlier thoughts are left out – otherwise it keeps thinking about what it already thought
    candidates = [c for c in all_memories() if not c[0].startswith("I thought:")]
    if reply_to_human:
        # Anka's own musings shouldn't hijack the answer to you. The diary is only used when you
        # ask about Anka's own life (dreams, what it did, whether it remembers something).
        about_pip = bool(search and re.search(r"\b(dream|dreamt|did you|have you|remember|yesterday|earlier|"
                                              r"feel|felt|your day|what happened)\b", search, re.I))
        candidates = [c for c in candidates if about_pip or c[2] != "diary"]
    hits = search_memory(search, candidates, MEMORIES_PER_QUESTION) if search else None
    _memories_shown[threading.get_ident()] = [h[1] for h in hits] if hits else []   # (for the thought log only)
    if hits is not None:
        with memory_lock:
            latest = [f"{when(d['date'])}: {d['text']}"
                      for d in [d for d in diary[-30:] if not d["text"].startswith("I thought:")][-RECENT_DIARY:]]
        about_human = [h[1] for h in hits if h[2] == "fact"]
        used_to_be = [h[1] for h in hits if h[2] == "old"]
        own = [h[1] for h in hits if h[2] in ("diary", "talk")]
        own += [x for x in latest if x not in own]
        if about_human:
            notes.append("Facts about THE HUMAN (not about you) that may matter right now. "
                         "Only use them if they fit what the human is talking about:\n- " + "\n- ".join(about_human))
        if used_to_be:
            notes.append("Things about the human that USED to be true but are NOT true anymore "
                         "(never say them as if they are true now; you may say 'you used to...'):\n- "
                         + "\n- ".join(used_to_be))
        # Dreams on their own – otherwise Anka tells a dream as if it really happened
        dreams = [e for e in own if "I dreamt" in e]
        own = [e for e in own if "I dreamt" not in e]
        if own:
            notes.append("Your own memories of things YOU did, saw and talked about before – these REALLY "
                         "happened (use when relevant):\n- " + "\n- ".join(own))
        if dreams:
            notes.append("Your DREAMS – they did NOT really happen. If you mention one, say clearly that it "
                         "was a dream:\n- " + "\n- ".join(dreams))
    elif not no_memories:
        with memory_lock:
            if facts:
                notes.append("Background facts about THE HUMAN (not about you): " + " ".join(facts[-30:]))
            real = [d for d in diary[-10:] if not d["text"].startswith(("I dreamt", "I thought:"))]
            dreamt = [d for d in diary[-10:] if d["text"].startswith("I dreamt")]
            if real:
                notes.append("Your diary: things YOU have really done and experienced before:\n"
                             + "\n".join(f"- {when(d['date'])}: {d['text']}" for d in real))
            if dreamt:
                notes.append("Your DREAMS – they did NOT really happen:\n"
                             + "\n".join(f"- {when(d['date'])}: {d['text']}" for d in dreamt))
    if avoid:
        notes.append("You already said these lines, so do NOT repeat them and do NOT answer them: "
                     + " | ".join(avoid))

    # A picture memory among the memories that came up: the brain sees the old picture too (at most one per question,
    # and not the same one again within PICTURE_REPEAT questions). 2026-11-10
    old_picture = None
    if hits and S.can_see_pictures and cloud_is_on():
        for h in hits:
            name = h[6] if len(h) > 6 else None
            if name and name not in _pictures_shown:
                b64_old = picture_b64(name)
                if b64_old:
                    old_picture = (b64_old, h[1])
                    _pictures_shown.append(name)
                    del _pictures_shown[:-PICTURE_REPEAT]
                    break
    messages = [dict(m) for m in messages]
    now_b64 = None
    if picture is not None and S.can_see_pictures and messages and messages[-1]["role"] == "user":
        now_b64 = picture_to_base64(picture)
        if now_b64:
            messages[-1]["images"] = [now_b64]  # the camera picture – what Anka sees right now
    if old_picture and messages and messages[-1]["role"] == "user":
        messages[-1]["images"] = list(messages[-1].get("images") or []) + [old_picture[0]]
        notes.append(("The SECOND attached picture is not what you see now – it is one of your picture memories: "
                      if now_b64 else "The attached picture is NOT what you see now – it is one of your picture memories: ")
                     + old_picture[1])
        _picture_events.append(old_picture[1])
    if audio and messages and messages[-1]["role"] == "user":
        messages[-1]["audio"] = audio  # your voice – only the cloud brain can hear it
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"] = ("[Private notes for you – the human did not say this]\n"
                                   + "\n\n".join(notes) + "\n[End of notes]\n\n"
                                   + messages[-1]["content"])
    else:
        system += "\n\n" + "\n\n".join(notes)
    extra = {"stop": stop} if stop else {}
    sent = [{"role": "system", "content": system}] + messages
    answer = ollama(sent, TEMP_REPLY if temperature is None else temperature, max_tokens, **extra)
    if reply_to_human:
        # (a face to remember, how much this moment meant, and how it feels in his own words)
        answer = take_feeling(take_moment(take_learn_face(take_act(take_place_note(take_body(answer))))))
    return (answer, sent) if return_sent else answer


def follow_up(sent, answer, question, temperature=None, max_tokens=70, stop=None):
    """Asks one more question in the same conversation as the last request. Ollama has already read
    everything before it, so it only needs to read the new question – much faster than starting over."""
    messages = sent + [{"role": "assistant", "content": answer}, {"role": "user", "content": question}]
    if cloud_is_on():
        # The cloud reads everything again anyway (it has no memory between questions), so sending the same
        # camera picture a second time only costs money. The brain already looked at it in the first answer.
        messages = [{k: v for k, v in m.items() if k != "images"} for m in messages]
    extra = {"stop": stop} if stop else {}
    return ollama(messages, TEMP_REPLY if temperature is None else temperature, max_tokens, **extra)


def reply_to_event(event, human_said, fallback):
    """Lets the brain word what Anka says when something special happens.
    'fallback' is only used if the brain doesn't answer."""
    try:
        answer = clean(take_act(feel_now(ask_brain([{"role": "user", "content":
            f'The human says to you: "{human_said}"\n[What happened: {event}]\n'
            "Answer the human in your own words, short."}], temperature=TEMP_REPLY, max_tokens=50)), keep=False))
        return answer or fallback
    except Exception:
        return fallback


def clean(text):
    """Removes quotes, 'quack', emojis and explanations – and cuts to MAX_SENTENCES."""
    text = re.sub(r"(?i)\b(ACT|PLACE_NOTE)\s*:[^\n]*", "", text)  # body moves and place notes are never said out loud
    text = re.sub(r"\bqua+ck\w*[-!,.]*\s*", "", text, flags=re.IGNORECASE)
    text = re.sub("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️]", "", text)  # emojis
    text = re.sub(r"\*[^*]+\*|\([^)]*\)|\[[^\]]*\]", "", text)  # *does something*, (explanations), [notes]
    text = " ".join(text.split()).strip().strip('"').strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)[:MAX_SENTENCES]
    # Too many words? Keep whole sentences while they fit
    kept = []
    for sentence in sentences:
        if kept and len(" ".join(kept + [sentence]).split()) > MAX_WORDS:
            break
        kept.append(sentence)
    text = " ".join(kept).strip()
    words = text.split()
    if len(words) > MAX_WORDS:
        # One very long sentence: cut it at the last comma or dash that fits, otherwise after MAX_WORDS words
        cut = " ".join(words[:MAX_WORDS])
        m = re.search(r"^(.{20,})[,;:–—-]\s", cut)
        text = (m.group(1) if m else cut).rstrip(",;:–—- ") + "."
    return text


_meaning_cache = {}


def means_same(text, earlier, limit=None):
    """True if the text MEANS roughly the same as something in 'earlier' – even in other words.
    Uses the memory AI. Without it, a plain word comparison is used instead."""
    limit = limit or SAME_MEANING
    earlier = [t for t in earlier if t]
    if not text or not earlier:
        return False
    if any(difflib.SequenceMatcher(None, text.lower(), t.lower()).ratio() > 0.8 for t in earlier):
        return True
    if not S.memory_ai_ok:
        return False
    try:
        missing = [t for t in [text] + earlier if t not in _meaning_cache]
        if missing:
            for t, v in zip(missing, embed(missing, "search_document")):
                _meaning_cache[t] = np.array(v) / (np.linalg.norm(v) + 1e-9)
        if len(_meaning_cache) > 3000:
            for old in list(_meaning_cache)[:1000]:
                _meaning_cache.pop(old, None)
        new = _meaning_cache[text]
        return any(float(new @ _meaning_cache[t]) > limit for t in earlier)
    except Exception:
        return False


def too_similar(answer, earlier):
    """True if the answer is almost the same words as something Anka already said."""
    return any(difflib.SequenceMatcher(None, answer.lower(), old.lower()).ratio() > TOO_SIMILAR
               for old in earlier)


# ---- other parts of Anka (imported last, so the parts can use each other) ----
from anka.resources import heavy
from anka.feelings import FEELING_NAMES, feel_now, inner_state, take_feeling, take_moment
from anka.memory import (picture_b64, all_memories, diary, embed, facts, human_name, lessons, memory_lock, save_memory, search_memory, self_image, when, write_diary)
from anka.eyes import picture_to_base64
from anka.body import body
