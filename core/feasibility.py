"""Per-point feasibility (M5): IK over spin/tilt/standoff samples, joint limits with
margin, then collision with contact filtering.

For each target the candidate TCP poses are tried in order of increasing tilt; the first
pose with a feasible branch wins, and among its branches the one with the best
(manipulability + joint-limit distance) score is kept. Status:
    reachable   feasible with comfortable margins
    marginal    feasible but clearance < margin + 5 mm or a joint within 15 deg of its limit
    unreachable with the first stage that no candidate got past as reason:
                no_ik -> joint_limits -> singular (only with CheckSettings.min_manip) -> collision

Target frame convention (spin reference): target z = outward normal n; target x = part
frame +x projected onto the tangent plane (+y if n is nearly parallel to x). A TCP
reaches the target when tcp.z = -n, rotated by spin about n and tilted inside the cone.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from core.collision import CollisionScene
from core.ik import best_copies
from core.robot import Robot
from core.schema import TargetGroup
from core.tool import ToolModel
from core.transforms import make_T

STAGES = ("no_ik", "joint_limits", "singular", "collision")
MARGINAL_CLEARANCE = 0.005  # m above the clearance margin
SPIN_STEP_DEG = 30.0
TILT_AZIMUTHS = 6


@dataclass
class CheckSettings:
    joint_margin_deg: float = 5.0
    clearance: float = 0.01
    # v7 round 2 (T15, rules-only planner): arm poses with |det J| below this are skipped
    # (stage "singular"); 0 = off, as in cost mode
    min_manip: float = 0.0


@dataclass
class PointResult:
    id: int
    status: str  # reachable | marginal | unreachable
    reason: str = ""
    q: np.ndarray | None = None
    tcp: str | None = None
    spin_deg: float = 0.0
    tilt_deg: float = 0.0
    standoff: float = 0.0
    clearance: float = float("inf")
    manipulability: float = 0.0
    limit_margin_deg: float = 0.0
    notes: list[str] = field(default_factory=list)
    config: str = ""  # joint configuration label (core.configs), when zoning is on
    zone: int | None = None  # zone index within the solution
    seq: int | None = None  # position in the zone's path


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return make_T(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]))


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return make_T(np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]]))


def target_frame(p, n) -> np.ndarray:
    """T_part_target: z = n, x = part +x projected on the tangent plane."""
    z = np.asarray(n, float) / np.linalg.norm(n)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = a - (a @ z) * z
    x /= np.linalg.norm(x)
    return make_T(np.column_stack([x, np.cross(z, x), z]), p)


def _spins(spin) -> list[float]:
    if spin == "free":
        return list(np.arange(0.0, 360.0, SPIN_STEP_DEG))
    lo, hi = (-float(spin), float(spin)) if not isinstance(spin, tuple) else spin
    c, half = (lo + hi) / 2, (hi - lo) / 2
    offs = [*np.arange(0.0, half, SPIN_STEP_DEG).tolist(), half] if half > 0 else [0.0]
    vals: list[float] = []
    for o in offs:  # centre first, then outward on both sides, ends included
        for v in (c - o, c + o):
            if not any(abs(v - w) < 1e-9 for w in vals):
                vals.append(v)
    return vals


def _tilts(cone: float) -> list[tuple[float, float]]:
    out = [(0.0, 0.0)]
    if cone > 0:
        for t in (cone / 2, cone):
            out += [(t, az) for az in np.arange(0.0, 360.0, 360.0 / TILT_AZIMUTHS)]
    return out


def _standoffs(s) -> list[float]:
    if isinstance(s, tuple):
        lo, hi = s
        return [(lo + hi) / 2, lo, hi] if hi > lo else [lo]
    return [float(s)]


def candidate_poses(p, n, group: TargetGroup, tcps: list[str]):
    """Yield (tcp, spin_deg, tilt_deg, standoff, T_part_tcp), most preferred first."""
    T0 = target_frame(p, n) @ _rx(np.pi)  # flip: tcp z = -n, tcp x = target x
    for tilt, az in _tilts(group.tilt_deg):
        for so in _standoffs(group.standoff_m):
            for spin in _spins(group.spin_deg):
                for tcp in tcps:
                    T = make_T(p=np.asarray(n) * so) @ T0 @ _rz(np.radians(az)) @ _rx(np.radians(tilt))
                    T = T @ _rz(np.radians(spin - az))
                    yield tcp, spin, tilt, so, T


class CancelledError(Exception):
    """Raised inside a check when the checker's ``cancelled()`` returns True (user Cancel)."""


CHUNK = 32  # largest number of candidate poses per vectorized IK call


class FeasibilityChecker:
    batch_ik = True  # vectorized IK for UR arms (False: the scalar path, for comparisons)
    cancelled = None  # optional () -> bool, polled once per candidate pose (Cancel within ~ms)
    memo = None  # per-plan cache of exact checks (opt.station_checks.CheckMemo), else None
    pool = None  # v8 T2: opt.parallel.CheckPool that fills the memo ahead of serial loops

    def __init__(
        self,
        robot: Robot,
        ik,
        tool: ToolModel,
        scene: CollisionScene,
        T_world_base: np.ndarray,
        T_world_part: np.ndarray,
        settings: CheckSettings | None = None,
    ):
        self.robot, self.ik, self.tool, self.scene = robot, ik, tool, scene
        self.settings = settings or CheckSettings()
        self.T_world_part = T_world_part
        self.set_base(T_world_base)
        m = np.radians(self.settings.joint_margin_deg)
        self.lo, self.hi = robot.lower + m, robot.upper - m
        self.T_flange_tcp = {t.name: tool.T_flange_tcp(t.name) for t in tool.ee.tcps}
        self._T_tcp_flange = {k: np.linalg.inv(v) for k, v in self.T_flange_tcp.items()}

    def set_base(self, T_world_base: np.ndarray) -> None:
        """Evaluate from another base pose (stations, refinement) without rebuilding."""
        self.T_world_base = np.asarray(T_world_base, float)
        self.T_base_part = np.linalg.inv(T_world_base) @ self.T_world_part
        self.scene.set_base(T_world_base)

    def check_point(
        self, pid: int, p, n, group: TargetGroup, max_poses: int | None = None, clearance: bool = True
    ) -> PointResult:
        """``max_poses`` caps the candidate TCP poses tried (fast screening in optimizers).
        ``clearance=False`` skips the part-clearance distance of the accepted solution (the
        collision check itself always runs): for screening and re-checks that only need
        "reachable or not"; the result's clearance is then inf."""
        if self.memo is not None:
            return self.memo.get(
                self,
                ("point", pid, max_poses, clearance),
                lambda: self._check_point(pid, p, n, group, max_poses, clearance),
            )
        return self._check_point(pid, p, n, group, max_poses, clearance)

    def _check_point(self, pid, p, n, group, max_poses, clearance) -> PointResult:
        stage, hit = [0], []
        for r in self._feasible(pid, p, n, group, max_poses, stage, clearance=clearance, hit=hit):
            return r
        out = PointResult(pid, "unreachable", STAGES[stage[0]])
        if out.reason == "collision" and hit:  # what the best-ranked arm pose runs into
            from core.collision import describe_hits

            note = describe_hits(self.scene.colliding_pairs(*hit[0]))
            out.notes = [note] if note else []
        return out

    def check_point_configs(
        self, pid: int, p, n, group: TargetGroup, max_poses: int | None = 40
    ) -> dict[str, PointResult]:
        """Best feasible solution for EACH joint configuration (core.configs labels), in one
        pass over the candidate poses (most preferred pose first, as check_point). Stops
        early once every configuration was found. Empty dict: unreachable."""
        if self.memo is not None:
            return self.memo.get(
                self,
                ("configs", pid, max_poses),
                lambda: self._check_point_configs(pid, p, n, group, max_poses),
            )
        return self._check_point_configs(pid, p, n, group, max_poses)

    def _check_point_configs(self, pid, p, n, group, max_poses) -> dict[str, PointResult]:
        from core.configs import config_label

        out: dict[str, PointResult] = {}
        n_max = getattr(self.ik, "n_branches", 4)
        # branches of configurations already found are dropped before the collision check, and
        # the clearance is measured only for the kept solutions (v7 round 2, T1 / T2)
        skip = lambda q: config_label(self.ik, q) in out  # noqa: E731
        for r in self._feasible(
            pid, p, n, group, max_poses, [0], all_branches=True, clearance=False, skip=skip
        ):
            lab = config_label(self.ik, r.q)
            if lab not in out:
                r.config = lab
                out[lab] = r
                if len(out) == n_max:
                    break
        for r in out.values():
            r.clearance = self.scene.part_clearance(r.q, r.tcp)
            r.status = self._status(r.clearance, r.limit_margin_deg)
        return out

    def _status(self, clear: float, limit_margin_deg: float) -> str:
        marginal = clear < self.settings.clearance + MARGINAL_CLEARANCE or limit_margin_deg < 15.0
        return "marginal" if marginal else "reachable"

    def _ik_stream(self, poses):
        """(pose, joint solutions in (-pi, pi]) per candidate pose, in order. UR arms use the
        vectorized solver on growing chunks (1, 2, 4, ... CHUNK poses): most points succeed on
        their first pose, hard ones get big cheap batches. Wrist-singular poses and solvers
        without a fast batch (numeric) use the scalar ``ik.solve``."""
        if not (self.batch_ik and getattr(self.ik, "batch", False)):
            for item in poses:
                yield item, self.ik.solve(item[-1])
            return
        chunk: list = []
        size = 1
        for item in itertools.chain(poses, [None]):
            if item is not None:
                chunk.append(item)
                if len(chunk) < size:
                    continue
            size = min(2 * size, CHUNK)
            if not chunk:
                break
            Q, V, sing = self.ik.solve_batch(np.array([c[-1] for c in chunk]), return_singular=True)
            for j, c in enumerate(chunk):
                yield c, (self.ik.solve(c[-1]) if sing[j] else Q[j][V[j]])
            chunk = []

    def _feasible(
        self,
        pid,
        p,
        n,
        group,
        max_poses,
        stage,
        all_branches: bool = False,
        clearance: bool = True,
        skip: Callable[[np.ndarray], bool] | None = None,
        hit: list | None = None,
    ):
        """Yield feasible PointResults: candidate poses in preference order; per pose, the
        branches best first (only the best one unless ``all_branches``). ``stage[0]`` is
        set to the furthest check stage any candidate passed. Branches with ``skip(q)`` true
        are dropped before the collision check. ``hit`` (a list) gets the first colliding
        (q, tcp)."""
        tcps = group.tcps or list(self.T_flange_tcp)
        poses = (
            (tcp, spin, tilt, so, self.T_base_part @ T_part_tcp @ self._T_tcp_flange[tcp])
            for tcp, spin, tilt, so, T_part_tcp in candidate_poses(p, n, group, tcps)
        )
        if max_poses is not None:
            poses = itertools.islice(poses, max_poses)
        for (tcp, spin, tilt, so, _), sols in self._ik_stream(poses):
            if self.cancelled is not None and self.cancelled():
                raise CancelledError
            if len(sols) == 0:
                continue
            stage[0] = max(stage[0], 1)
            qs, inside = best_copies(sols, self.lo, self.hi, self.robot.lower, self.robot.upper)
            cands = list(qs[inside])
            if not cands:
                continue
            stage[0] = max(stage[0], 2)
            # rank the branches first (cheap), then check collision lazily in that order
            offset = self.T_flange_tcp[tcp][:3, 3]
            Q = np.array(cands)
            if hasattr(self.ik, "manipulability"):  # |det J| from the solver (any TCP offset)
                mus = np.asarray(self.ik.manipulability(Q), float)
            else:
                mus = np.array([self.robot.manipulability(q, offset) for q in Q])
            lms = np.min(np.minimum(Q - self.robot.lower, self.robot.upper - Q), axis=1)
            if self.settings.min_manip > 0:  # rule: not close to a singularity
                ok = mus >= self.settings.min_manip
                if not ok.any():
                    continue
                Q, mus, lms = Q[ok], mus[ok], lms[ok]
            stage[0] = max(stage[0], 3)
            # ties (mirror branches at equal quality) go to the lower branch index, not float noise
            for b in sorted(range(len(Q)), key=lambda b: (-round(float(mus[b] + 0.1 * lms[b]), 9), b)):
                q, mu, lm = Q[b], float(mus[b]), float(lms[b])
                if skip is not None and skip(q):
                    continue
                if self.scene.collides(q, tcp):
                    if hit is not None and not hit:
                        hit.append((q.copy(), tcp))
                    continue
                clear = self.scene.part_clearance(q, tcp) if clearance else float("inf")
                yield PointResult(
                    pid,
                    self._status(clear, float(np.degrees(lm))),
                    "",
                    q,
                    tcp,
                    spin,
                    tilt,
                    so,
                    clear,
                    mu,
                    float(np.degrees(lm)),
                )
                if not all_branches:
                    break

    def check_all(
        self,
        targets,
        progress: Callable[[int, int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> list[PointResult]:
        groups = {g.name: g for g in targets.groups}
        out = []
        for k, pt in enumerate(targets.points):
            if cancelled is not None and cancelled():
                break
            g = groups.get(pt.group) or TargetGroup(name=pt.group)
            out.append(self.check_point(pt.id, np.asarray(pt.p), np.asarray(pt.n), g))
            if progress is not None:
                progress(k + 1, len(targets.points))
        return out


def result_to_data(r: PointResult):
    """PointResult -> schema.PointResultData (for the project file)."""
    from core.schema import PointResultData

    finite = r.clearance is not None and np.isfinite(r.clearance)
    return PointResultData(
        id=int(r.id),
        status=r.status,
        reason=r.reason,
        q=None if r.q is None else [float(v) for v in r.q],
        tcp=r.tcp,
        spin_deg=float(r.spin_deg),
        tilt_deg=float(r.tilt_deg),
        standoff=float(r.standoff),
        clearance=float(r.clearance) if finite else None,
        manipulability=float(r.manipulability),
        limit_margin_deg=float(r.limit_margin_deg),
        notes=[str(n) for n in r.notes],
        config=r.config,
        zone=r.zone,
        seq=r.seq,
    )


def result_from_data(d) -> PointResult:
    return PointResult(
        id=d.id,
        status=d.status,
        reason=d.reason,
        q=None if d.q is None else np.array(d.q, float),
        tcp=d.tcp,
        spin_deg=d.spin_deg,
        tilt_deg=d.tilt_deg,
        standoff=d.standoff,
        clearance=float("inf") if d.clearance is None else d.clearance,
        manipulability=d.manipulability,
        limit_margin_deg=d.limit_margin_deg,
        notes=list(d.notes),
        config=d.config,
        zone=d.zone,
        seq=d.seq,
    )
