"""Saving memory on an 8 GB Mac: only one heavy AI at a time, and models that load when needed."""
import contextlib
import gc
import threading
import time

from settings import ONE_AT_A_TIME, UNLOAD_AFTER
from anka.shared import listening



class HeavyLock:
    """Makes sure only one heavy AI runs at a time."""
    def __init__(self):
        self.lock = threading.RLock()
        self.count = 0

    def __enter__(self):
        self.lock.acquire()
        self.count += 1

    def __exit__(self, *a):
        self.count -= 1
        self.lock.release()


heavy_lock = HeavyLock()


def heavy():
    return heavy_lock if ONE_AT_A_TIME else contextlib.nullcontext()


def busy():
    """True if a heavy AI is working right now."""
    return ONE_AT_A_TIME and heavy_lock.count > 0


class LazyModel:
    """An AI model that is loaded when needed and unloaded when it hasn't been used for a while."""
    def __init__(self, name, load, obj=None):
        self.name, self.load, self.obj = name, load, obj
        self.last_used = time.time()
        self.lock = threading.Lock()

    def get(self):
        """Returns the model – loads it again if it was unloaded. None if that didn't work."""
        with self.lock:
            if self.obj is None:
                print(f"⏳ Loading {self.name} again...")
                start = time.time()
                try:
                    self.obj = self.load()  # (not inside the heavy lock – that could deadlock)
                    print(f"✅ {self.name} is back ({time.time() - start:.1f} s)")
                except Exception as error:
                    print(f"❌ Couldn't load {self.name} again ({type(error).__name__}: {error}) – will try next time")
                    return None
            self.last_used = time.time()
            return self.obj

    def preload(self):
        """Starts loading in the background now, so the model is ready when it's needed."""
        self.last_used = time.time()
        if self.obj is None:
            threading.Thread(target=self.get, daemon=True).start()

    def unload_if_old(self):
        with self.lock:
            if self.obj is not None and UNLOAD_AFTER and time.time() - self.last_used > UNLOAD_AFTER:
                self.obj = None
                gc.collect()
                print(f"💤 Unloaded {self.name} to save memory")


def unload_loop(models):
    """Runs in the background: unloads models that haven't been used for a while."""
    while True:
        time.sleep(10)
        if listening.is_set() or speaking():
            continue
        for m in models:
            if m is not None:
                m.unload_if_old()


# ---- other parts of Anka (imported last, so the parts can use each other) ----
from anka.voice import speaking
