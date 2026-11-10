"""Planner runs in a worker process (v8 T1).

One persistent worker process (``spawn``) runs jobs one at a time: the station planner (normal,
quick, rules only) and the tool optimizer. The GUI thread that starts a job blocks in
``Runner.run`` (it is already a background thread), forwards the worker's messages (progress,
streamed solutions, ...) to its handler and returns the job's result. A job is
``"module:function"`` called as ``fn(payload, emit, cancelled)`` in the worker; payloads and
messages must pickle.

Cancel sets a shared event the job polls (``cancelled()``); a job that has not ended 2 s later
is killed with its process, which frees its memory (the next job starts a fresh worker). An
idle worker is ended after ``IDLE_S`` (it keeps the session, maps and meshes warm until then).

``USE_WORKER`` is False by default so tests and scripts run in-process; ``app.main`` turns it on
(Settings -> *Run planners in a separate process*).
"""

from __future__ import annotations

import importlib
import itertools
import multiprocessing as mp
import queue
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any

USE_WORKER = False
CANCEL_GRACE_S = 2.0
IDLE_S = 300.0

_CTX = mp.get_context("spawn")


class JobCancelledError(Exception):
    """The job was killed after Cancel (it did not stop within the grace time)."""


def _resolve(target: str) -> Callable:
    mod, name = target.split(":")
    return getattr(importlib.import_module(mod), name)


def _worker_main(inbox, outbox, cancel) -> None:  # pragma: no cover - runs in the child
    while True:
        msg = inbox.get()
        if msg is None:
            return
        job_id, target, payload = msg

        def emit(kind: str, data: Any = None, _id=job_id) -> None:
            outbox.put((_id, kind, data))

        try:
            out = _resolve(target)(payload, emit, cancel.is_set)
            outbox.put((job_id, "done", out))
        except BaseException as err:  # report, keep the worker alive
            outbox.put((job_id, "error", (type(err).__name__, str(err), traceback.format_exc())))


class Runner:
    """One worker process; jobs queue on a lock (one at a time)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc = None
        self._ids = itertools.count(1)
        self.waiting = 0  # jobs waiting for the running one (shown in the status line)
        self._last = 0.0  # end of the last job (idle timer)
        self._timer: threading.Timer | None = None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None and self._proc.is_alive() else None

    def _ensure(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._inbox, self._outbox, self._cancel = _CTX.Queue(), _CTX.Queue(), _CTX.Event()
        self._proc = _CTX.Process(
            target=_worker_main, args=(self._inbox, self._outbox, self._cancel), daemon=True
        )
        self._proc.start()

    def kill(self) -> None:
        p, self._proc = self._proc, None
        if p is not None and p.is_alive():
            p.kill()
            p.join(5)

    def stop(self) -> None:
        """End the worker (server shutdown)."""
        if self._proc is not None and self._proc.is_alive():
            try:
                self._inbox.put(None)
                self._proc.join(2)
            except Exception:
                pass
        self.kill()

    def run(
        self,
        target: str,
        payload: Any,
        on_msg: Callable[[str, Any], None],
        cancelled: Callable[[], bool] = lambda: False,
    ) -> Any:
        """Run ``target`` in the worker and return its result (blocks the calling thread)."""
        if not self._lock.acquire(blocking=False):
            self.waiting += 1
            on_msg("progress", ("waiting for the running planner job to finish", 0.0))
            try:
                while not self._lock.acquire(timeout=0.2):
                    if cancelled():
                        raise JobCancelledError("cancelled while waiting")
            finally:
                self.waiting -= 1
        try:
            self._ensure()
            self._cancel.clear()
            job_id = next(self._ids)
            self._inbox.put((job_id, target, payload))
            cancel_at = None
            while True:
                if cancel_at is None and cancelled():
                    self._cancel.set()
                    cancel_at = time.monotonic()
                try:
                    jid, kind, data = self._outbox.get(timeout=0.1)
                except queue.Empty:
                    if cancel_at is not None and time.monotonic() - cancel_at > CANCEL_GRACE_S:
                        self.kill()
                        raise JobCancelledError("the worker did not stop in time and was ended") from None
                    if self._proc is None or not self._proc.is_alive():
                        self._proc = None
                        raise RuntimeError("the worker process ended unexpectedly") from None
                    continue
                if jid != job_id:
                    continue  # left over from a killed job
                if kind == "done":
                    return data
                if kind == "error":
                    name, msg, tb = data
                    raise RuntimeError(f"{name}: {msg}\n--- worker traceback ---\n{tb}")
                on_msg(kind, data)
        finally:
            self._last = time.monotonic()
            self._lock.release()
            self._arm_idle()

    def _arm_idle(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(IDLE_S, self._idle_kill)
        self._timer.daemon = True
        self._timer.start()

    def _idle_kill(self) -> None:
        if time.monotonic() - self._last >= IDLE_S - 0.5 and self._lock.acquire(blocking=False):
            try:
                self.stop()
            finally:
                self._lock.release()


_RUNNER: Runner | None = None


def runner() -> Runner:
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = Runner()
    return _RUNNER


def call(target: str, payload: Any, on_msg, cancelled, in_process: bool) -> Any:
    """Run a job in the worker, or here (``in_process``: tests, scripts, worker off)."""
    if in_process or not USE_WORKER:
        return _resolve(target)(payload, on_msg, cancelled)
    return runner().run(target, payload, on_msg, cancelled)
