"""Tolerance check (v8 T5): does a plan survive fixture and positioning error?

Monte Carlo: each sample moves the part by a random offset (uniform within +-``part_mm`` per
axis) and turn (random axis, up to ``part_deg``, about the part's centre), and each stop by a
random offset (uniform within +-``station_mm`` in x and y). Every planned point is then
checked exactly from its stop, as the plan assigned it. The points move with the part (the
robot program is assumed to follow the part, e.g. via a user frame): the question is whether
they are still *reachable*, not whether a fixed program still hits them.

Per point: the reach rate over the samples, how far the point had moved in the smallest
sample that failed (its "safe offset", None = never failed), and the nominal clearance,
joint-limit margin and manipulability. The samples are seeded (repeatable).

Checks run through ``opt.parallel`` when a pool is given (calls carry the sample's part pose).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from core.feasibility import CancelledError


@dataclass
class ToleranceSettings:
    part_mm: float = 2.0
    part_deg: float = 0.5
    station_mm: float = 0.0
    samples: int = 50
    seed: int = 1


@dataclass
class PointTolerance:
    pid: int
    station: int
    rate: float  # reached in this fraction of the samples
    fail_mm: float | None  # the point's displacement in the smallest failing sample
    clearance_mm: float
    limit_deg: float
    manipulability: float


@dataclass
class ToleranceResult:
    label: str
    settings: ToleranceSettings
    points: dict[int, PointTolerance] = field(default_factory=dict)
    all_reached: float = 0.0  # fraction of samples in which every planned point was reached
    unplanned: list[int] = field(default_factory=list)  # unreached in the plan itself: not tested
    lost: list[int] = field(default_factory=list)  # per sample: planned points not reached (v8 T12)

    def worst(self, n: int = 10) -> list[PointTolerance]:
        return sorted(self.points.values(), key=lambda p: (p.rate, p.fail_mm or np.inf))[:n]


def samples(st: ToleranceSettings, centre: np.ndarray, n_stops: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(part move in the world 4x4, stop offsets (k, 3)) per sample; the first is nominal."""
    rng = np.random.default_rng(st.seed)
    out = [(np.eye(4), np.zeros((n_stops, 3)))]
    for _ in range(max(0, st.samples - 1)):
        d = rng.uniform(-1, 1, 3) * st.part_mm / 1000
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        ang = np.radians(rng.uniform(0, st.part_deg))
        R = Rotation.from_rotvec(axis * ang).as_matrix()
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = centre - R @ centre + d  # turn about the part centre, then shift
        e = np.zeros((n_stops, 3))
        e[:, :2] = rng.uniform(-1, 1, (n_stops, 2)) * st.station_mm / 1000
        out.append((M, e))
    return out


def tolerance_check(
    checker,
    t,
    sol,
    results: dict,
    T_stops: list[np.ndarray],
    st: ToleranceSettings,
    pool=None,
    progress: Callable[[str, float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> ToleranceResult:
    """``sol``: schema.Solution (assignment); ``results``: its nominal PointResults; ``T_stops``:
    the robot base pose at each stop (core.export.station_pose)."""
    T_part0 = checker.T_world_part.copy()
    index = {int(pid): i for i, pid in enumerate(t.ids)}
    planned = {
        int(pid): k
        for pid, k in sol.assignment.items()
        if int(pid) in index and getattr(results.get(int(pid)), "status", "unreachable") != "unreachable"
    }
    out = ToleranceResult(sol.label, st)
    out.unplanned = sorted(int(p) for p in index if p not in planned)
    T_stop = [np.asarray(T, float) for T in T_stops]
    centre = t.P_world.mean(axis=0) if len(t.P_world) else T_part0[:3, 3]
    smp = samples(st, centre, len(T_stop))
    ok = np.zeros((len(smp), len(planned)), bool)
    moved = np.zeros((len(smp), len(planned)))
    pids = list(planned)
    for s_i, (M, e) in enumerate(smp):
        if cancelled is not None and cancelled():
            raise CancelledError()
        T_part = M @ T_part0
        calls = []
        for j, pid in enumerate(pids):
            i, k = index[pid], planned[pid]
            T_base = T_stop[k].copy()
            T_base[:3, 3] += e[k]
            p_w = t.P_world[i]
            moved[s_i, j] = 1000 * np.linalg.norm((M[:3, :3] @ p_w + M[:3, 3]) - p_w) + 1000 * np.linalg.norm(
                e[k]
            )
            calls.append((T_base, "point", pid, t.P_part[i], t.N_part[i], t.groups[i], None, False, T_part))
        res = run_calls(checker, calls, pool, cancelled)
        ok[s_i] = [r.status != "unreachable" for r in res]
        if progress is not None:
            progress(f"tolerance check: sample {s_i + 1} / {len(smp)}", (s_i + 1) / len(smp))
    for j, pid in enumerate(pids):
        r0 = results[pid]
        fails = moved[~ok[:, j], j]
        out.points[pid] = PointTolerance(
            pid,
            planned[pid],
            float(ok[:, j].mean()),
            float(fails.min()) if len(fails) else None,
            float(r0.clearance) * 1000 if np.isfinite(r0.clearance) else float("inf"),
            float(r0.limit_margin_deg),
            float(r0.manipulability),
        )
    out.all_reached = float(ok.all(axis=1).mean()) if pids else 1.0
    out.lost = [int(n) for n in (~ok).sum(axis=1)]
    return out


def run_calls(checker, calls: list, pool=None, cancelled=None) -> list:
    """Exact checks for (T_base, kind, pid, p, n, group, max_poses, clearance[, T_part]) calls,
    in a pool or here; the checker's base and part pose are restored afterwards."""
    if pool is not None:
        return pool.run(checker.settings, calls, cancelled)
    T_base0, T_part0 = checker.T_world_base.copy(), checker.T_world_part.copy()
    out = []
    try:
        for c in calls:
            T, kind, pid, p, n, g, max_poses, clearance = c[:8]
            set_part(checker, c[8] if len(c) > 8 and c[8] is not None else T_part0)
            checker.set_base(T)
            if kind == "point":
                out.append(checker._check_point(pid, p, n, g, max_poses, clearance))
            else:
                out.append(checker._check_point_configs(pid, p, n, g, max_poses))
    finally:
        set_part(checker, T_part0)
        checker.set_base(T_base0)
    return out


def set_part(checker, T_world_part: np.ndarray) -> None:
    """Move the part (checker + collision scene); takes effect with the next ``set_base``."""
    checker.T_world_part = np.asarray(T_world_part, float)
    checker.scene.T_world_part = checker.T_world_part
