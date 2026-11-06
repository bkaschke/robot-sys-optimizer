"""Paths between planned points, and the joint motion along them (v2 phase 5).

User decision: a path connects the points of each stop in the planner's order (the zones'
path order when zoning is on, otherwise a serpentine order of the stop's points). Between
two consecutive points the TCP follows the part surface:

* position: the chord between the points, sampled every ``step``, projected onto the part
  mesh, plus the standoff along the normal (normal and standoff interpolated);
* orientation: slerp between the two planned TCP orientations (so every waypoint is exactly
  its planned solution, and tilt / spin change smoothly in between).

Every sample is solved by IK, keeping the start point's joint configuration (core.configs
label) and the 2*pi copy closest to the previous sample, and checked for joint limits and
collision. Problems are flagged per sample:

  unreachable   no IK solution within the joint limits
  config_flip   only another shoulder / elbow / wrist configuration reaches it
  jump          a joint moves more than ``jump_deg`` between neighbouring samples
  collision     the arm or tool is closer than the clearance (contact faces excepted)
  reconfigure   the start of a new zone at the same stop (a planned configuration change)

The motion between stops (carrier travel) and the reconfiguration moves themselves are not
planned. Pure geometry + core.feasibility: no viser.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation, Slerp

from core.configs import config_label, order_path
from core.feasibility import CancelledError, FeasibilityChecker, PointResult

ISSUES = ("unreachable", "config_flip", "jump", "collision", "reconfigure")


@dataclass
class PathSample:
    station: int
    zone: int | None
    a: int  # point id the segment starts at
    b: int  # point id it ends at (a == b: the first point of a sequence)
    s: float  # 0..1 along the segment
    T_part_tcp: np.ndarray
    tcp: str
    q: np.ndarray | None
    issue: str = ""


@dataclass
class StationPath:
    station: int
    T_world_base: np.ndarray
    carrier: dict
    samples: list[PathSample] = field(default_factory=list)


@dataclass
class PathPlan:
    stations: list[StationPath]
    step: float

    @property
    def samples(self) -> list[PathSample]:
        return [p for st in self.stations for p in st.samples]

    def counts(self) -> dict[str, int]:
        out = dict.fromkeys(ISSUES, 0)
        for p in self.samples:
            if p.issue:
                out[p.issue] += 1
        return out

    def length(self) -> float:
        """TCP path length (m), within sequences (no jumps between stops or zones)."""
        total = 0.0
        for st in self.stations:
            for p0, p1 in zip(st.samples[:-1], st.samples[1:], strict=True):
                if p1.issue != "reconfigure" and p1.a != p1.b:
                    total += float(np.linalg.norm(p1.T_part_tcp[:3, 3] - p0.T_part_tcp[:3, 3]))
        return total


def sequences(sol, results: dict[int, PointResult], P_part: dict[int, np.ndarray]) -> dict[int, list]:
    """Per station: [(zone index | None, [point ids in path order])]."""
    out: dict[int, list] = {}

    def usable(p: int) -> bool:  # solved, and still a target (points deleted after planning are skipped)
        return p in P_part and results.get(p) is not None and results[p].q is not None

    if sol.zones:
        for zi, z in enumerate(sol.zones):
            pts = [p for p in z.points if usable(p)]
            if pts:
                out.setdefault(z.station, []).append((zi, pts))
        return out
    by: dict[int, list[int]] = {}
    for pid, k in sol.assignment.items():
        if usable(pid):
            by.setdefault(k, []).append(pid)
    for k, pids in by.items():
        pids = sorted(pids)
        order = order_path(np.array([P_part[p] for p in pids])) if len(pids) > 2 else np.arange(len(pids))
        out[k] = [(None, [pids[i] for i in order])]
    return out


class _Tracer:
    def __init__(self, checker: FeasibilityChecker, mesh: trimesh.Trimesh, step: float, jump_deg: float):
        self.ch, self.mesh, self.step = checker, mesh, step
        self.jump = np.radians(jump_deg)
        self.ur = hasattr(checker.ik, "a2")

    def tcp_pose(self, r: PointResult) -> np.ndarray:
        """Planned TCP pose in the part frame (from the planned joints, exact)."""
        ch = self.ch
        return np.linalg.inv(ch.T_base_part) @ ch.robot.fk(r.q) @ ch.T_flange_tcp[r.tcp]

    def solve(self, T_part_tcp: np.ndarray, tcp: str, prev: np.ndarray, label: str):
        """(q, issue) for one sample: keep ``label``, nearest 2*pi copy to ``prev``."""
        ch = self.ch
        T = ch.T_base_part @ T_part_tcp @ ch._T_tcp_flange[tcp]
        if self.ur:
            from core.ik.ur import solve_batch

            Q, V, sing = solve_batch(ch.ik, T[None], return_singular=True)
            sols = ch.ik.solve(T) if sing[0] else Q[0][V[0]]
        else:
            sols = ch.ik.solve(T)
        best = None
        for s in sols:
            q = prev + (np.mod(s - prev + np.pi, 2 * np.pi) - np.pi)  # nearest copy of each joint
            if np.any(q < ch.lo - 1e-9) or np.any(q > ch.hi + 1e-9):
                continue
            same = config_label(ch.ik, q) == label
            key = (not same, float(np.max(np.abs(q - prev))))
            if best is None or key < best[0]:
                best = (key, q, same)
        if best is None:
            return None, "unreachable"
        (_, dq), q, same = best
        if not same:
            return q, "config_flip"
        if dq > self.jump:
            return q, "jump"
        if ch.scene.collides(q, tcp):
            return q, "collision"
        return q, ""

    def segment(
        self, k, zone, ra: PointResult, rb: PointResult, Pa, Pb, Na, Nb, prev, label
    ) -> list[PathSample]:
        Ta, Tb = self.tcp_pose(ra), self.tcp_pose(rb)
        n = max(1, int(np.ceil(np.linalg.norm(Pb - Pa) / self.step)))
        s = np.arange(1, n + 1) / n
        chord = Pa + s[:, None] * (Pb - Pa)
        on, _, _ = trimesh.proximity.closest_point(self.mesh, chord)
        on[-1] = Pb  # land exactly on the planned point
        normal = Na + s[:, None] * (Nb - Na)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        so = ra.standoff + s * (rb.standoff - ra.standoff)
        rot = Slerp([0.0, 1.0], Rotation.from_matrix([Ta[:3, :3], Tb[:3, :3]]))(s).as_matrix()
        out = []
        for i in range(n):
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = rot[i], on[i] + so[i] * normal[i]
            q, issue = self.solve(T, rb.tcp, prev, label)
            out.append(PathSample(k, zone, ra.id, rb.id, float(s[i]), T, rb.tcp, q, issue))
            if q is not None:
                prev = q
        return out


def trace(
    checker: FeasibilityChecker,
    part_mesh: trimesh.Trimesh,
    sol,
    results: dict[int, PointResult],
    points: dict[int, tuple[np.ndarray, np.ndarray]],
    base_poses: list[np.ndarray],
    step: float = 0.02,
    jump_deg: float = 20.0,
    cancelled: Callable[[], bool] | None = None,
    progress: Callable[[float], None] | None = None,
) -> PathPlan:
    """Paths of solution ``sol``. ``points``: point id -> (p, n) in the part frame;
    ``base_poses``: world base pose per station (core.export.station_pose)."""
    tr = _Tracer(checker, part_mesh, step, jump_deg)
    seqs = sequences(sol, results, {p: v[0] for p, v in points.items()})
    plan = PathPlan([], step)
    n_seq = max(1, sum(len(v) for v in seqs.values()))
    done = 0
    for k in sorted(seqs):
        checker.set_base(base_poses[k])
        sp = StationPath(k, base_poses[k], dict(sol.stations[k].carrier))
        for j, (zone, pids) in enumerate(seqs[k]):
            if cancelled is not None and cancelled():
                raise CancelledError
            r0 = results[pids[0]]
            q = np.asarray(r0.q, float)
            label = config_label(checker.ik, q)
            first = "reconfigure" if j > 0 else ""
            sp.samples.append(PathSample(k, zone, pids[0], pids[0], 0.0, tr.tcp_pose(r0), r0.tcp, q, first))
            for a, b in zip(pids[:-1], pids[1:], strict=True):
                if cancelled is not None and cancelled():
                    raise CancelledError
                rb = results[b]
                lab_b = config_label(checker.ik, np.asarray(rb.q, float))
                if lab_b != label:  # the plan uses another configuration for b: the arm
                    q, label = np.asarray(rb.q, float), lab_b  # reconfigures (not traced)
                    sp.samples.append(
                        PathSample(k, zone, a, b, 1.0, tr.tcp_pose(rb), rb.tcp, q, "reconfigure")
                    )
                    continue
                (Pa, Na), (Pb, Nb) = points[a], points[b]
                seg = tr.segment(k, zone, results[a], results[b], Pa, Pb, Na, Nb, q, label)
                sp.samples += seg
                q = next((p.q for p in reversed(seg) if p.q is not None), q)
            done += 1
            if progress is not None:
                progress(done / n_seq)
        plan.stations.append(sp)
    return plan


CSV_COLUMNS = (
    "station",
    "zone",
    "from_point",
    "to_point",
    "s",
    "tcp",
    "x_mm",
    "y_mm",
    "z_mm",
    "qw",
    "qx",
    "qy",
    "qz",
    "j1_deg",
    "j2_deg",
    "j3_deg",
    "j4_deg",
    "j5_deg",
    "j6_deg",
    "issue",
)


def path_csv(plan: PathPlan, T_world_part: np.ndarray) -> str:
    """One row per sample: TCP pose in the world (mm, quaternion wxyz), joints, issue."""
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for p in plan.samples:
        T = T_world_part @ p.T_part_tcp
        x, y, z, qw = Rotation.from_matrix(T[:3, :3]).as_quat()[[0, 1, 2, 3]]
        joints = [""] * 6 if p.q is None else [f"{v:.4f}" for v in np.degrees(p.q)]
        w.writerow(
            [
                p.station + 1,
                "" if p.zone is None else p.zone + 1,
                p.a,
                p.b,
                f"{p.s:.4f}",
                p.tcp,
                *(f"{v * 1000:.3f}" for v in T[:3, 3]),
                f"{qw:.6f}",
                f"{x:.6f}",
                f"{y:.6f}",
                f"{z:.6f}",
                *joints,
                p.issue,
            ]
        )
    return buf.getvalue()
