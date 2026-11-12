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

Pools are kept per spec (``pool_for``): building one costs a few seconds (imports, robot,
collision scene per worker), so later plans with the same setup reuse it. Workers end
themselves when the process that made the pool ends (watchdog on the parent).
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import pickle
import threading
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

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
    dress: float = 0.0  # v8 T18: core.collision.DRESS_M of the parent
    hose: object = None  # core.collision.HOSE of the parent (core.hose.Hose)
    geom: object = None  # core.collision.geom_state() of the parent (collision geometry)

    def build(self):
        from core import collision
        from core.collision import CollisionScene
        from core.feasibility import FeasibilityChecker
        from core.ik import make_ik
        from core.robot import Robot

        collision.DRESS_M = self.dress
        collision.HOSE = self.hose
        collision.set_geom_state(self.geom)
        robot = Robot(self.robot_spec)
        scene = CollisionScene(
            robot,
            self.tool,
            self.mesh,
            self.T_world_base,
            self.T_world_part,
            base_bodies=self.base_bodies,
            world_bodies=self.world_bodies,
            obstacles=self.obstacles,
        )
        return FeasibilityChecker(
            robot, make_ik(robot), self.tool, scene, self.T_world_base, self.T_world_part
        )

    def key(self) -> str:
        return hashlib.sha1(pickle.dumps(self, protocol=5)).hexdigest()


_CHECKER = None  # in a pool worker


def _watch_parent() -> None:  # pragma: no cover - runs in pool workers
    parent = mp.parent_process()
    if parent is not None:
        parent.join()
        os._exit(0)


def _init(spec: CheckerSpec) -> None:  # pragma: no cover - runs in pool workers
    global _CHECKER
    threading.Thread(target=_watch_parent, daemon=True).start()
    _CHECKER = spec.build()


def _run(settings, calls: list) -> list:  # pragma: no cover - runs in pool workers
    from opt.tolerance import set_part

    c = _CHECKER
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


class CheckPool:
    def __init__(self, spec: CheckerSpec, workers: int):
        self.workers = int(workers)
        self.key = spec.key()
        self._ex = ProcessPoolExecutor(
            max_workers=self.workers, mp_context=mp.get_context("spawn"), initializer=_init, initargs=(spec,)
        )

    def run(self, settings, calls: list, cancelled=None) -> list:
        """Results of ``calls`` in order (``cancelled`` is polled between chunks)."""
        chunks = [calls[i : i + CHUNK] for i in range(0, len(calls), CHUNK)]
        futs = [self._ex.submit(_run, settings, ch) for ch in chunks]
        out = []
        for f in futs:
            if cancelled is not None and cancelled():
                for g in futs:
                    g.cancel()
                from core.feasibility import CancelledError

                raise CancelledError()
            out += f.result()
        return out

    def shutdown(self) -> None:
        self._ex.shutdown(wait=False, cancel_futures=True)


_POOLS: dict[str, CheckPool] = {}


def pool_for(spec: CheckerSpec, workers: int) -> CheckPool | None:
    """The pool for this setup (reused while it is the same), or None for ``workers <= 1``."""
    if workers <= 1:
        return None
    key = spec.key()
    p = _POOLS.get(key)
    if p is not None and p.workers == workers:
        return p
    for old in _POOLS.values():  # one pool at a time: workers hold a robot + scene each
        old.shutdown()
    _POOLS.clear()
    p = _POOLS[key] = CheckPool(spec, workers)
    return p


def shutdown_all() -> None:
    for p in _POOLS.values():
        p.shutdown()
    _POOLS.clear()


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
