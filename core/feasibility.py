"""Per-point feasibility (M5): IK over spin/tilt/standoff samples, joint limits with
margin, static torque, then collision with contact filtering.

For each target the candidate TCP poses are tried in order of increasing tilt; the first
pose with a feasible branch wins, and among its branches the one with the best
(manipulability + joint-limit distance) score is kept. Status:
    reachable   feasible with comfortable margins
    marginal    feasible but clearance < margin + 5 mm, torque util > 90% of its limit, or a
                joint within 15 deg of its limit
    unreachable with the first stage that no candidate got past as reason:
                no_ik -> joint_limits -> torque -> collision

Target frame convention (spin reference): target z = outward normal n; target x = part
frame +x projected onto the tangent plane (+y if n is nearly parallel to x). A TCP
reaches the target when tcp.z = -n, rotated by spin about n and tilted inside the cone.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from core.collision import CollisionScene
from core.ik import expand_to_limits
from core.payload import static_check
from core.robot import Robot
from core.schema import TargetGroup
from core.tool import ToolModel
from core.transforms import make_T

STAGES = ("no_ik", "joint_limits", "torque", "collision")
MARGINAL_CLEARANCE = 0.005  # m above the clearance margin
SPIN_STEP_DEG = 30.0
TILT_AZIMUTHS = 6


@dataclass
class CheckSettings:
    joint_margin_deg: float = 5.0
    torque_util: float = 0.8
    clearance: float = 0.01


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
    torque_util: float = 0.0
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


class FeasibilityChecker:
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
        mp = tool.mass_properties()
        self.mass, self.com = mp.mass_kg, mp.com_flange
        self.force = tool.ee.process_force_N
        self.T_flange_tcp = {t.name: tool.T_flange_tcp(t.name) for t in tool.ee.tcps}
        self._T_tcp_flange = {k: np.linalg.inv(v) for k, v in self.T_flange_tcp.items()}

    def set_base(self, T_world_base: np.ndarray) -> None:
        """Evaluate from another base pose (stations, refinement) without rebuilding."""
        self.T_base_part = np.linalg.inv(T_world_base) @ self.T_world_part
        self.robot.set_mount(T_world_base[:3, :3])
        self.scene.set_base(T_world_base)

    def check_point(self, pid: int, p, n, group: TargetGroup, max_poses: int | None = None) -> PointResult:
        """``max_poses`` caps the candidate TCP poses tried (fast screening in optimizers)."""
        stage = [0]
        for r in self._feasible(pid, p, n, group, max_poses, stage):
            return r
        return PointResult(pid, "unreachable", STAGES[stage[0]])

    def check_point_configs(
        self, pid: int, p, n, group: TargetGroup, max_poses: int | None = 40
    ) -> dict[str, PointResult]:
        """Best feasible solution for EACH joint configuration (core.configs labels), in one
        pass over the candidate poses (most preferred pose first, as check_point). Stops
        early once every configuration was found. Empty dict: unreachable."""
        from core.configs import config_label

        out: dict[str, PointResult] = {}
        n_max = 8 if hasattr(self.ik, "a2") else 4
        for r in self._feasible(pid, p, n, group, max_poses, [0], all_branches=True):
            lab = config_label(self.ik, r.q)
            if lab not in out:
                r.config = lab
                out[lab] = r
                if len(out) == n_max:
                    break
        return out

    def _feasible(self, pid, p, n, group, max_poses, stage, all_branches: bool = False):
        """Yield feasible PointResults: candidate poses in preference order; per pose, the
        branches best first (only the best one unless ``all_branches``). ``stage[0]`` is
        set to the furthest check stage any candidate passed."""
        tcps = group.tcps or list(self.T_flange_tcp)
        R_bp = self.T_base_part[:3, :3]
        force_base = -self.force * (R_bp @ np.asarray(n, float))  # tool pushes into the surface
        for k, (tcp, spin, tilt, so, T_part_tcp) in enumerate(candidate_poses(p, n, group, tcps)):
            if max_poses is not None and k >= max_poses:
                break
            T_base_flange = self.T_base_part @ T_part_tcp @ self._T_tcp_flange[tcp]
            sols = self.ik.solve(T_base_flange)
            if len(sols) == 0:
                continue
            stage[0] = max(stage[0], 1)
            cands = []
            for s in sols:
                ok = expand_to_limits(s[None], self.lo, self.hi)
                if len(ok):
                    margin = np.minimum(ok - self.robot.lower, self.robot.upper - ok).min(axis=1)
                    cands.append(ok[np.argmax(margin)])
            if not cands:
                continue
            stage[0] = max(stage[0], 2)
            scored = []
            for q in cands:
                pr = static_check(
                    self.robot,
                    q,
                    self.mass,
                    self.com,
                    force_base,
                    self.T_flange_tcp[tcp][:3, 3],
                    self.settings.torque_util,
                )
                if not pr.ok:
                    continue
                mu = self.robot.manipulability(q, self.T_flange_tcp[tcp][:3, 3])
                lm = float(np.min(np.minimum(q - self.robot.lower, self.robot.upper - q)))
                scored.append((mu + 0.1 * lm, q, pr, mu, lm))
            if not scored:
                continue
            stage[0] = max(stage[0], 3)
            for _, q, pr, mu, lm in sorted(scored, key=lambda t: -t[0]):
                if self.scene.collides(q, tcp):
                    continue
                clear = self.scene.part_clearance(q, tcp)
                marginal = (
                    clear < self.settings.clearance + MARGINAL_CLEARANCE
                    or pr.max_util > 0.9 * self.settings.torque_util
                    or np.degrees(lm) < 15.0
                )
                yield PointResult(
                    pid,
                    "marginal" if marginal else "reachable",
                    "",
                    q,
                    tcp,
                    spin,
                    tilt,
                    so,
                    clear,
                    mu,
                    pr.max_util,
                    float(np.degrees(lm)),
                    pr.notes,
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
        torque_util=float(r.torque_util),
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
        torque_util=d.torque_util,
        limit_margin_deg=d.limit_margin_deg,
        notes=list(d.notes),
        config=d.config,
        zone=d.zone,
        seq=d.seq,
    )
