"""Parallel exact checks (v8 T2).

A ``CheckPool`` is a process pool whose workers each build their own ``FeasibilityChecker``
once from a picklable ``CheckerSpec`` (Pinocchio / Coal objects don't pickle). The planner
code stays serial: before a loop of checks it calls ``prefetch(checker, calls)``, which runs
the calls the plan's ``CheckMemo`` doesn't have yet in the pool and stores them in the memo;
the serial loop then reads them from the memo. Results are identical to the serial path
(same inputs, same deterministic checker).

``calls`` are ``(T_world_base, kind, pid, p, n, group, max_poses, clearance)`` with kind
"point" (``check_point``) or "configs" (``check_point_configs``; clearance ignored), and an
optional 9th entry, the part pose for that call (tolerance samples, ``opt.tolerance``).

Workers are kept between runs (v10 #7): one process pool (``_executor``, per worker count) lives
as long as the process that made it, and a ``CheckPool`` is a light handle for one setup. Its spec
is pickled once to a file; each task carries the setup's key and that file, and a worker whose
current setup has another key re-targets (loads the spec, builds that checker) before it runs the
calls. Compare (one setup per robot) and Gantry placement (one per frame) therefore start the
processes once instead of once per robot / frame (a Windows spawn is ~3 s per worker). Workers end
themselves when the process that made the pool ends (watchdog on the parent).
"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing as mp
import os
import pickle
import tempfile
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
on_warn = None  # optional callable(str): the UI hook that shows the user a pool failure

MIN_CALLS = 24  # fewer calls than this: not worth the round trip
CHUNK = 8  # calls per task


def default_workers() -> int:
    """Default pool size: CPU count - 1, at most 4 (each worker holds a robot + scene)."""
    return max(1, min(4, (os.cpu_count() or 2) - 1))


@dataclass
class CheckerSpec:
    """Everything a worker needs to build the planner's checker."""

    robot_spec: object  # core.robot.RobotSpec
    tool: object  # core.tool.ToolModel
    mesh: object  # trimesh.Trimesh (part)
    T_world_base: np.ndarray
    T_world_part: np.ndarray
    base_bodies: list = field(default_factory=list)
    world_bodies: list = field(default_factory=list)
    obstacles: list = field(default_factory=list)
    collision: object = None  # v12 #10: the parent's core.collision_config.CollisionConfig
    margins: object = None  # core.collision.Margins of the session (None: defaults; review R13)
    settings: object = None  # core.feasibility.CheckSettings of the session (None: defaults)

    def build(self):
        from core.collision import CollisionScene
        from core.feasibility import FeasibilityChecker
        from core.ik import make_ik
        from core.robot import Robot

        robot = Robot(self.robot_spec)
        scene = CollisionScene(
            robot,
            self.tool,
            self.mesh,
            self.T_world_base,
            self.T_world_part,
            self.margins,
            base_bodies=self.base_bodies,
            world_bodies=self.world_bodies,
            obstacles=self.obstacles,
            config=self.collision,
        )
        return FeasibilityChecker(
            robot, make_ik(robot), self.tool, scene, self.T_world_base, self.T_world_part, self.settings
        )

    def key(self) -> str:
        return hashlib.sha1(pickle.dumps(self, protocol=5)).hexdigest()


_CHECKER = None  # in a pool worker: the checker of the current setup
_KEY = None  # its key
_INIT_ERROR = None  # in a pool worker: text of the exception spec.build() raised (review M3)


def _watch_parent() -> None:  # pragma: no cover - runs in pool workers
    parent = mp.parent_process()
    if parent is not None:
        parent.join()
        os._exit(0)


def _init() -> None:  # pragma: no cover - runs in pool workers
    threading.Thread(target=_watch_parent, daemon=True).start()
    import core.collision  # noqa: F401  (warm imports: the first re-target is faster)
    import core.feasibility  # noqa: F401


def _target(key: str, path: str):  # pragma: no cover - runs in pool workers
    """This worker's checker for setup ``key`` (built from the spec file when it changed). A build
    that raises is reported by ``_run`` with its traceback (it would otherwise only break the pool)."""
    global _CHECKER, _KEY, _INIT_ERROR
    if key != _KEY:
        _CHECKER, _KEY, _INIT_ERROR = None, None, None  # the old setup's robot / scene go first (memory)
        try:
            with open(path, "rb") as f:
                spec = pickle.load(f)
            _CHECKER = spec.build()
        except BaseException:
            _INIT_ERROR = traceback.format_exc()
        _KEY = key
    return _CHECKER


def _run(key: str, path: str, settings, calls: list) -> list:  # pragma: no cover - in pool workers
    c = _target(key, path)
    if c is None:
        raise RuntimeError(f"check worker could not build its checker: {_INIT_ERROR}")
    return _run_on(c, settings, calls)


def _run_on(c, settings, calls: list) -> list:
    from opt.tolerance import set_part

    c.settings = settings
    T_part0 = getattr(c, "_T_part0", None)
    if T_part0 is None:
        T_part0 = c._T_part0 = c.T_world_part.copy()
    out = []
    for call in calls:  # an optional 9th entry moves the part (v8 T5 tolerance samples)
        T, kind, pid, p, n, group, max_poses, clearance = call[:8]
        set_part(c, call[8] if len(call) > 8 and call[8] is not None else T_part0)
        c.set_base(T)
        if kind == "point":
            out.append(c._check_point(pid, p, n, group, max_poses, clearance))
        else:
            out.append(c._check_point_configs(pid, p, n, group, max_poses))
    return out


def _noop() -> None:  # pragma: no cover - runs in pool workers
    return None


def warm(workers: int) -> None:
    """Start the pool's processes in the background (a spawn blocks ~3 s per worker): the caller
    goes on with work that needs no pool (set cover, frame scoring) meanwhile."""
    if workers > 1:
        threading.Thread(target=lambda: _executor(workers).submit(_noop), daemon=True).start()


_EXECUTORS: dict[int, ProcessPoolExecutor] = {}
_LOCK = threading.Lock()
SPEC_DIR = Path(tempfile.gettempdir()) / f"rso_check_specs_{os.getpid()}"


def _executor(workers: int) -> ProcessPoolExecutor:
    """The persistent process pool with ``workers`` processes (one at a time: each worker holds a
    robot + scene). A pool that broke (a worker died) is replaced."""
    with _LOCK:
        ex = _EXECUTORS.get(workers)
        if ex is not None and not getattr(ex, "_broken", False):
            return ex
        for old in _EXECUTORS.values():
            old.shutdown(wait=False, cancel_futures=True)
        _EXECUTORS.clear()
        ex = _EXECUTORS[workers] = ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init
        )
        return ex


class CheckPool:
    """The check pool for one setup (a handle on the shared processes, see the module docstring)."""

    def __init__(self, spec: CheckerSpec, workers: int):
        self.workers = int(workers)
        self.key = spec.key()
        SPEC_DIR.mkdir(parents=True, exist_ok=True)
        self.path = str(SPEC_DIR / f"{self.key}.pkl")
        if not os.path.exists(self.path):
            tmp = f"{self.path}.{threading.get_ident()}.tmp"
            with open(tmp, "wb") as f:
                pickle.dump(spec, f, protocol=5)
            os.replace(tmp, self.path)
        self._spec = spec
        self._local = None  # serial fallback checker (built on first failure; review M3)
        self._ex = _executor(self.workers)

    def run(self, settings, calls: list, cancelled=None) -> list:
        """Results of ``calls`` in order (``cancelled`` is polled between chunks)."""
        if getattr(self._ex, "_broken", False):
            self._ex = _executor(self.workers)
        chunks = [calls[i : i + CHUNK] for i in range(0, len(calls), CHUNK)]
        futs = [self._ex.submit(_run, self.key, self.path, settings, ch) for ch in chunks]
        out = []
        for f in futs:
            if cancelled is not None and cancelled():
                for g in futs:
                    g.cancel()
                from core.feasibility import CancelledError

                raise CancelledError()
            try:
                out += f.result()
            except (BrokenProcessPool, RuntimeError) as err:
                for g in futs:
                    g.cancel()
                return self._fallback(settings, calls, err)
        return out

    def _fallback(self, settings, calls: list, err: BaseException) -> list:
        """The pool is broken (a worker died or could not build): evict it so the next
        ``pool_for`` builds a fresh one, tell the user, and do this run's calls serially."""
        self.shutdown()
        if _POOLS.get(self.key) is self:
            del _POOLS[self.key]
        with _LOCK:  # the shared processes too: the next pool starts fresh ones
            ex = _EXECUTORS.pop(self.workers, None)
        if ex is not None:
            ex.shutdown(wait=False, cancel_futures=True)
        cause = f"{type(err).__name__}: {err}".strip()
        msg = f"parallel checks failed ({cause}); continuing with serial checks"
        log.warning(msg)
        if on_warn is not None:
            try:
                on_warn(msg)
            except Exception:  # a UI hook must never break the plan
                log.exception("parallel warn hook failed")
        if self._local is None:
            self._local = self._spec.build()  # raises the real error if the setup itself is bad
        return _run_on(self._local, settings, calls)

    def shutdown(self) -> None:
        """Forget this setup (the processes stay for the next one: ``shutdown_all`` ends them)."""
        try:
            os.remove(self.path)
        except OSError:
            pass


_POOLS: dict[str, CheckPool] = {}


def pool_for(spec: CheckerSpec, workers: int) -> CheckPool | None:
    """The pool for this setup (re-targets the shared workers), or None for ``workers <= 1``."""
    if workers <= 1:
        return None
    key = spec.key()
    p = _POOLS.get(key)
    if p is not None and p.workers == workers and os.path.exists(p.path):
        return p
    for old in _POOLS.values():  # one setup at a time (its spec file)
        old.shutdown()
    _POOLS.clear()
    p = _POOLS[key] = CheckPool(spec, workers)
    warm(workers)  # v10 #7: the processes start while the planner's first steps run
    return p


def shutdown_all() -> None:
    for p in _POOLS.values():
        p.shutdown()
    _POOLS.clear()
    with _LOCK:
        for ex in _EXECUTORS.values():
            ex.shutdown(wait=False, cancel_futures=True)
        _EXECUTORS.clear()


def prefetch(checker, calls: list) -> None:
    """Fill the checker's memo with ``calls`` computed in its pool (no-op without a pool or
    a memo, or for a handful of calls)."""
    pool, memo = getattr(checker, "pool", None), checker.memo
    if pool is None or memo is None or len(calls) < MIN_CALLS:
        return
    todo, keys, seen = [], [], set()
    for c in calls:
        T, kind, pid, _p, _n, _g, max_poses, clearance = c
        call = ("point", pid, max_poses, clearance) if kind == "point" else ("configs", pid, max_poses)
        k = (memo.key(T), *call)
        if k in memo.data or k in seen:
            continue
        seen.add(k)
        todo.append(c)
        keys.append(k)
    if len(todo) < MIN_CALLS:
        return
    for k, r in zip(keys, pool.run(checker.settings, todo, checker.cancelled), strict=True):
        memo.data[k] = r
