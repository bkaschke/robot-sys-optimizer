"""Planner runs in a worker process (v8 T1).

One persistent worker process (``spawn``) runs jobs one at a time: the station planner (normal,
quick, rules only) and the tool optimizer. The GUI thread that starts a job blocks in
``Runner.run`` (it is already a background thread), forwards the worker's messages (progress,
streamed solutions, ...) to its handler and returns the job's result. A job is
``"module:function"`` called as ``fn(payload, emit, cancelled)`` in the worker; payloads and
messages must pickle.

Everything crossing the process boundary is pickled explicitly (``pickle.dumps``) before it goes
on a queue, because ``mp.Queue`` pickles in a feeder thread that only prints its errors: a result
that does not pickle would otherwise hang ``run`` and a message would be lost. Now a payload that
does not pickle raises ``TypeError`` here at once, a result that does not pickle comes back as a
``WorkerError``, and a message that does not pickle arrives as a ``"warn"`` message. A job that
raises ``core.feasibility.CancelledError`` re-raises that class here (message kind
``"cancelled"``); any other error becomes ``WorkerError`` (a RuntimeError whose first line is the
summary, with the worker traceback in ``worker_traceback``). A worker that dies is reported with
its exit code and a likely cause (``exit_cause``).

Cancel sets a shared event the job polls (``cancelled()``); a job that has not ended 2 s later
is killed with its process, which frees its memory (the next job starts a fresh worker). An
idle worker is ended after ``IDLE_S`` (it keeps the session, maps and meshes warm until then).

``USE_WORKER`` is False by default so tests and scripts run in-process; ``app.main`` turns it on
(Settings -> *Run planners in a separate process*).
"""

from __future__ import annotations

import importlib
import itertools
import logging
import multiprocessing as mp
import os
import pickle
import queue
import sys
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any

USE_WORKER = False
CHECK_WORKERS = 1  # v8 T2: processes for the planner's exact checks (1 = serial; app.main: default)
CANCEL_GRACE_S = 2.0
IDLE_S = 300.0

_CTX = mp.get_context("spawn")
_LOG = logging.getLogger(__name__)
_HINT = "If this was a large plan, try fewer targets or a coarser grid."


class JobCancelledError(Exception):
    """The job was killed after Cancel (it did not stop within the grace time)."""


class WorkerError(RuntimeError):
    """A job failed in the worker. ``str`` starts with a one-line summary ("Type: message", also
    in ``summary``); the worker's traceback follows it and is kept in ``worker_traceback``."""

    def __init__(self, name: str, msg: str, tb: str = "") -> None:
        self.summary = f"{name}: {msg}".splitlines()[0] if msg else name
        self.worker_traceback = tb
        text = f"{name}: {msg}" if msg else name
        super().__init__(f"{text}\n--- worker traceback ---\n{tb}" if tb else text)


def exit_cause(code: int | None) -> str:
    """A likely cause for a worker exit code ("" when there is no known one)."""
    if code is None:
        return ""
    u = code & 0xFFFFFFFF if code < -255 else code  # Windows NTSTATUS, signed or not
    if u == 0xC0000005:
        return "access violation (native crash)"
    if u == 0xC0000409:
        return "stack buffer overrun (native crash)"
    if u in (0xC0000017, 0xC000012D) or code < 0:  # no memory / commit limit / a signal (-9)
        return "killed (possibly out of memory)"
    return ""


def _cancel_class() -> type | None:
    mod = sys.modules.get("core.feasibility")  # not loaded: the job cannot have raised it
    return getattr(mod, "CancelledError", None)


def _resolve(target: str) -> Callable:
    mod, name = target.split(":")
    return getattr(importlib.import_module(mod), name)


def _worker_main(inbox, outbox, cancel) -> None:  # pragma: no cover - runs in the child
    def watch_parent() -> None:  # end with the app, even when it is killed
        parent = mp.parent_process()
        if parent is not None:
            parent.join()
            os._exit(0)

    threading.Thread(target=watch_parent, daemon=True).start()
    while True:
        msg = inbox.get()
        if msg is None:
            return
        job_id, target, raw = msg

        def send(kind: str, data: Any, _id=job_id) -> None:  # pickle here, so errors raise here
            outbox.put((_id, kind, pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)))

        def emit(kind: str, data: Any = None) -> None:
            try:
                send(kind, data)
            except Exception as err:  # keep the job running, but say what was lost
                name = type(err).__name__
                send(
                    "warn",
                    f"a '{kind}' message could not be sent from the worker and was dropped ({name}: {err})",
                )

        try:
            payload = pickle.loads(raw)
            if isinstance(payload, dict) and "collision" in payload:  # v12 #10: the GUI's settings
                from core import collision_config

                collision_config.use(payload["collision"])
            out = _resolve(target)(payload, emit, cancel.is_set)
            try:
                send("done", out)
            except Exception as err:
                why = f"the job finished but its result could not be sent back from the worker ({err})"
                send("error", (type(err).__name__, why, traceback.format_exc()))
        except BaseException as err:  # report, keep the worker alive
            cls = _cancel_class()
            if cls is not None and isinstance(err, cls):
                send("cancelled", str(err))
            else:
                send("error", (type(err).__name__, str(err), traceback.format_exc()))


class Runner:
    """One worker process; jobs queue on a lock (one at a time)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc = None
        self._ids = itertools.count(1)
        self.waiting = 0  # jobs waiting for the running one (shown in the status line)
        self._last = 0.0  # end of the last job (idle timer)
        self._timer: threading.Timer | None = None
        self._atexit = False

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None and self._proc.is_alive() else None

    def _ensure(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._inbox, self._outbox, self._cancel = _CTX.Queue(), _CTX.Queue(), _CTX.Event()
        self._proc = _CTX.Process(
            target=_worker_main, args=(self._inbox, self._outbox, self._cancel), daemon=False
        )  # not a daemon: it may start a check pool (opt.parallel); ended at exit instead
        self._proc.start()
        if not self._atexit:
            import atexit

            atexit.register(self.kill)
            self._atexit = True

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
        try:
            raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as err:
            name = type(err).__name__
            raise TypeError(
                f"the job payload for {target} cannot be sent to the worker process ({name}: {err})"
            ) from err
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
            self._inbox.put((job_id, target, raw))
            cancel_at = None
            while True:
                if cancel_at is None and cancelled():
                    self._cancel.set()
                    cancel_at = time.monotonic()
                try:
                    jid, kind, blob = self._outbox.get(timeout=0.1)
                except queue.Empty:
                    if cancel_at is not None and time.monotonic() - cancel_at > CANCEL_GRACE_S:
                        self.kill()
                        raise JobCancelledError("the worker did not stop in time and was ended") from None
                    if self._proc is None or not self._proc.is_alive():
                        raise RuntimeError(self._died()) from None
                    continue
                if jid != job_id:
                    continue  # left over from a killed job
                try:
                    data = pickle.loads(blob)
                except Exception as err:  # e.g. a class the GUI process cannot import
                    if kind in ("done", "error"):
                        raise WorkerError(
                            type(err).__name__, f"the job's {kind} reply could not be read ({err})"
                        ) from err
                    on_msg(
                        "warn",
                        f"a '{kind}' message from the worker could not be read and was dropped ({err})",
                    )
                    continue
                if kind == "done":
                    return data
                if kind == "error":
                    raise WorkerError(*data)
                if kind == "cancelled":
                    from core.feasibility import CancelledError

                    raise CancelledError(data)
                on_msg(kind, data)
        finally:
            self._last = time.monotonic()
            self._lock.release()
            self._arm_idle()

    def _died(self) -> str:
        p, self._proc = self._proc, None
        code = None
        if p is not None:
            p.join(1)
            code = p.exitcode
        cause = f": {exit_cause(code)}" if exit_cause(code) else ""
        msg = f"the worker process ended unexpectedly (exit code {code}{cause}). {_HINT}"
        _LOG.error("planner worker: %s", msg)
        return msg

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
    if isinstance(payload, dict) and "collision" not in payload:  # v12 #10: one value, every job
        from core.collision_config import current

        payload = {**payload, "collision": current()}
    return runner().run(target, payload, on_msg, cancelled)
