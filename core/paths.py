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

v4 plan items 2 and 4: the points of each group are ordered by a path pattern
(core.patterns); moves that would cross an earlier move can lift off the surface (kind
"lift"). With a ``home`` pose every stop starts and ends with a joint move from / to home
(kind "home"), and between stops the carrier moves with the arm at home (kind "carrier",
``carrier_path``), so all points of a stop are done before the carrier moves on:

  carrier_blocked   a carrier pose on the way hits a keep-out, the part standoff or a
                    stay-out zone

Joint moves (to / from home, configuration changes) are planned collision-free
(``_Tracer.joint_move``, core.joint_plan). Pure geometry + core.feasibility: no viser.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation, Slerp

from core.configs import config_label, order_path
from core.feasibility import CancelledError, FeasibilityChecker, PointResult
from core.patterns import CORNERS, PatternSettings, crossing_lifts, order_points

ISSUES = ("unreachable", "config_flip", "jump", "collision", "reconfigure", "carrier_blocked")
HOME_STEP = np.radians(3.0)  # joint step of the moves to / from home
RETREAT = 0.15  # m the tool backs off the surface before a planned joint move


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
    kind: str = ""  # "" on the surface, "lift", "home" (to / from home), "carrier" (between stops)
    carrier: dict | None = None  # carrier values of this sample (None = its stop's)
    T_world_base: np.ndarray | None = None  # base pose when it differs from the stop's


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
        """TCP path length (m) on the surface (no lifts, home or carrier moves)."""
        total = 0.0
        for st in self.stations:
            for p0, p1 in zip(st.samples[:-1], st.samples[1:], strict=True):
                if p1.issue != "reconfigure" and p1.a != p1.b and not p1.kind and not p0.kind:
                    total += float(np.linalg.norm(p1.T_part_tcp[:3, 3] - p0.T_part_tcp[:3, 3]))
        return total


def sequences(
    sol,
    results: dict[int, PointResult],
    P_part: dict[int, np.ndarray],
    label_of: Callable | None = None,
    group_of: dict[int, str] | None = None,
) -> dict[int, list]:
    """Per station: [(zone index | None, [point ids in path order])]. Without zones and with
    ``label_of(result)`` (joint configuration label), a stop's points are split by
    configuration, largest group first: one configuration change per extra configuration,
    not one per switch along the path. With ``group_of`` (point id -> target group), each
    sequence is also split by target group (v6): the tool is not dragged over an edge of the
    part from one face to the next; a planned joint move joins them."""
    out = _sequences(sol, results, P_part, label_of)
    if not group_of:
        return out
    split: dict[int, list] = {}
    for k, seqs in out.items():
        for zone, ids in seqs:
            parts: dict[str, list[int]] = {}
            for p in ids:  # keeps the path order inside each group
                parts.setdefault(group_of.get(p, ""), []).append(p)
            for g in parts.values():
                if len(parts) > 1 and len(g) > 2:  # re-order each face on its own
                    g = [g[i] for i in order_path(np.array([P_part[p] for p in g]))]
                split.setdefault(k, []).append((zone, g))
    return split


def _sequences(sol, results, P_part, label_of) -> dict[int, list]:
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
        split: dict[str, list[int]] = {}
        for p in sorted(pids):
            split.setdefault(label_of(results[p]) if label_of is not None else "", []).append(p)
        out[k] = []
        for ids in sorted(split.values(), key=len, reverse=True):
            order = order_path(np.array([P_part[p] for p in ids])) if len(ids) > 2 else np.arange(len(ids))
            out[k].append((None, [ids[i] for i in order]))
    return out


@dataclass
class Seq:
    zone: int | None
    pids: list[int]
    lifts: list[bool]  # per move pids[k] -> pids[k + 1]: lift off the surface


def ordered_sequences(
    sol,
    results,
    points: dict,
    pattern: PatternSettings | None = None,
    label_of: Callable | None = None,
    group_of: dict[int, str] | None = None,
) -> dict[int, list[Seq]]:
    """Per stop: its groups (zones) as Seq. Without a pattern, the planner's order. With one,
    each group is ordered by the pattern, groups follow each other nearest first, and a later
    group starts at the corner nearest where the previous one ended."""
    P_part = {p: v[0] for p, v in points.items()}
    base = sequences(sol, results, P_part, label_of, group_of)
    if pattern is None:
        return {k: [Seq(z, ids, [False] * (len(ids) - 1)) for z, ids in v] for k, v in base.items()}
    out: dict[int, list[Seq]] = {}
    for k, groups in base.items():
        cents = [np.mean([P_part[i] for i in ids], axis=0) for _, ids in groups]
        left, chain = list(range(1, len(groups))), [0]
        while left:
            here = groups[chain[-1]][0]  # finish this zone (configuration) before the next
            j = min(
                left,
                key=lambda i: (groups[i][0] != here, float(np.linalg.norm(cents[i] - cents[chain[-1]]))),
            )
            left.remove(j)
            chain.append(j)
        seqs, end = [], None
        for gi in chain:
            zone, ids = groups[gi]
            P = np.array([points[i][0] for i in ids])
            N = np.array([points[i][1] for i in ids])
            if end is None:
                o = order_points(P, N, pattern)
            else:  # start near the previous group's end
                o = min(
                    (order_points(P, N, pattern, c) for c in CORNERS),
                    key=lambda r: float(np.linalg.norm(P[r.order[0]] - end)),
                )
            pids = [ids[i] for i in o.order]
            lifts = (
                crossing_lifts(o.uv, o.order, o.passes)
                if pattern.no_crossing
                else np.zeros(len(pids) - 1, bool)
            )
            seqs.append(Seq(zone, pids, [bool(x) for x in lifts]))
            end = P[o.order[-1]]
        out[k] = seqs
    return out


class _Tracer:
    def __init__(self, checker: FeasibilityChecker, mesh: trimesh.Trimesh, step: float, jump_deg: float):
        self.ch, self.mesh, self.step = checker, mesh, step
        self.jump = np.radians(jump_deg)
        self.ur = hasattr(checker.ik, "a2")

    def tcp_pose(self, r: PointResult) -> np.ndarray:
        """Planned TCP pose in the part frame (from the planned joints, exact)."""
        return self.fk_pose(r.q, r.tcp)

    def fk_pose(self, q, tcp: str) -> np.ndarray:
        ch = self.ch
        return np.linalg.inv(ch.T_base_part) @ ch.robot.fk(np.asarray(q, float)) @ ch.T_flange_tcp[tcp]

    def joint_move(
        self, k, zone, qa, qb, tcp: str, a: int, b: int, at_point=(True, True)
    ) -> list[PathSample]:
        """Collision-free joint move qa -> qb (to / from home, configuration changes), point
        ids a / b (0 = home; ``at_point``: which end is on the surface). Straight or one joint
        group at a time when clear; else the tool first backs off the surface along its axis
        (``RETREAT``, same configuration) at a point end and the arm goes around the part,
        floor, stay-out zones and carrier bodies (core.joint_plan). When no way is found:
        the straight move, its collisions flagged."""
        from core.joint_plan import densify, plan_joint_path

        qa, qb = np.asarray(qa, float), np.asarray(qb, float)
        ch = self.ch

        def hit(q) -> bool:
            return ch.scene.collides(q, tcp)

        out_a, in_b, way = [], [], None
        if not hit(qb):
            way = plan_joint_path(hit, qa, qb, ch.lo, ch.hi, HOME_STEP, iters=0)
            if way is None:
                out_a = self.retreat(qa, tcp) if at_point[0] else []
                in_b = self.retreat(qb, tcp) if at_point[1] else []
                qa2, qb2 = (out_a or [qa])[-1], (in_b or [qb])[-1]
                way = plan_joint_path(hit, qa2, qb2, ch.lo, ch.hi, HOME_STEP)
        if way is None:
            out_a, in_b = [], []
        qs = out_a + densify(way if way is not None else [qa, qb], HOME_STEP)
        if in_b:
            qs += in_b[-2::-1] + [qb]
        out = []
        for i, q in enumerate(qs):
            issue = "collision" if way is None and hit(q) else ""
            s = (i + 1) / len(qs)
            out.append(PathSample(k, zone, a, b, float(s), self.fk_pose(q, tcp), tcp, q, issue, kind="home"))
        return out

    def nearest_copy(self, q, ref) -> np.ndarray:
        """The 2*pi copy of q (same arm pose) nearest ``ref`` per joint, inside the limits."""
        q, ref = np.asarray(q, float), np.asarray(ref, float)
        c = q + 2 * np.pi * np.round((ref - q) / (2 * np.pi))
        lo, hi = self.ch.lo, self.ch.hi
        c = np.where(c < lo - 1e-9, c + 2 * np.pi, c)
        c = np.where(c > hi + 1e-9, c - 2 * np.pi, c)
        return np.where((c < lo - 1e-9) | (c > hi + 1e-9), q, c)

    def retreat(self, q, tcp: str, dist: float | None = None) -> list[np.ndarray]:
        """Joints backing the tool off the surface along its axis from q (q excluded), in q's
        configuration, up to ``dist`` (default ``RETREAT``) or the first sample with a problem."""
        dist = RETREAT if dist is None else dist
        q = np.asarray(q, float)
        T0 = self.fk_pose(q, tcp)
        label = config_label(self.ch.ik, q)
        n = max(3, int(np.ceil(dist / self.step)))
        out, prev = [], q
        for s in np.arange(1, n + 1) / n:
            T = T0.copy()
            T[:3, 3] = T0[:3, 3] - s * dist * T0[:3, 2]  # tcp z points into the part
            qs, issue = self.solve(T, tcp, prev, label)
            if issue:
                break
            out.append(qs)
            prev = qs
        return out

    def change_move(self, k, zone, qa, qb, tcp: str, a: int, b: int, home=None) -> list[PathSample]:
        """Joint move between two points of a stop. A configuration change goes through home
        (when there is one) so the arm does not flip next to the part."""
        qa, qb = np.asarray(qa, float), np.asarray(qb, float)
        if home is None or config_label(self.ch.ik, qa) == config_label(self.ch.ik, qb):
            return self.joint_move(k, zone, qa, qb, tcp, a, b)
        out = self.joint_move(k, zone, qa, home, tcp, a, 0, (True, False))
        return out + self.joint_move(k, zone, home, qb, tcp, 0, b, (False, True))

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
        self, k, zone, ra: PointResult, rb: PointResult, Pa, Pb, Na, Nb, prev, label, lift: float = 0.0
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
        if lift > 0:  # off the surface: up, across, down (full height from 15 % to 85 %)
            so = so + lift * np.minimum(1.0, 8 * s * (1 - s))
        rot = Slerp([0.0, 1.0], Rotation.from_matrix([Ta[:3, :3], Tb[:3, :3]]))(s).as_matrix()
        out = []
        for i in range(n):
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = rot[i], on[i] + so[i] * normal[i]
            q, issue = self.solve(T, rb.tcp, prev, label)
            out.append(
                PathSample(
                    k, zone, ra.id, rb.id, float(s[i]), T, rb.tcp, q, issue, kind="lift" if lift > 0 else ""
                )
            )
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
    pattern: PatternSettings | None = None,
    lift: float = 0.03,
    home: np.ndarray | None = None,
    carrier_path: Callable[[int, int], list[tuple[dict, np.ndarray, bool]]] | None = None,
    group_of: dict[int, str] | None = None,
) -> PathPlan:
    """Paths of solution ``sol``. ``points``: point id -> (p, n) in the part frame;
    ``base_poses``: world base pose per station (core.export.station_pose). ``pattern``
    orders each group; ``lift`` (m) is the height of lifted moves; ``home`` (joints) adds the
    moves to / from home; ``carrier_path(k_from, k_to)`` gives the carrier poses between two
    stops as (carrier values, T_world_base, clear); ``group_of`` (point id -> target group)
    keeps each face's points in their own sequences."""
    tr = _Tracer(checker, part_mesh, step, jump_deg)
    seqs = ordered_sequences(  # unzoned stops: one group per joint configuration
        sol, results, points, pattern, lambda r: config_label(checker.ik, np.asarray(r.q, float)), group_of
    )
    plan = PathPlan([], step)
    n_seq = max(1, sum(len(v) for v in seqs.values()))
    done = 0
    stops = sorted(seqs)
    for idx, k in enumerate(stops):
        checker.set_base(base_poses[k])
        sp = StationPath(k, base_poses[k], dict(sol.stations[k].carrier))
        if home is not None and idx > 0 and carrier_path is not None:
            r0 = results[seqs[k][0].pids[0]]
            for f, (dofs, T, ok) in enumerate(carrier_path(stops[idx - 1], k)):
                checker.set_base(T)
                issue = "" if ok else "carrier_blocked"
                if not issue and checker.scene.collides(home, r0.tcp):
                    issue = "collision"
                sp.samples.append(
                    PathSample(
                        k, None, 0, 0, float(f), tr.fk_pose(home, r0.tcp), r0.tcp, np.asarray(home, float),
                        issue, kind="carrier", carrier=dict(dofs), T_world_base=np.asarray(T, float),
                    )
                )  # fmt: skip
            checker.set_base(base_poses[k])
        q = None
        for j, sq in enumerate(seqs[k]):
            zone, pids = sq.zone, sq.pids
            if cancelled is not None and cancelled():
                raise CancelledError
            r0 = results[pids[0]]
            ref = home if (home is not None and j == 0) or q is None else q
            q_prev, q = q, tr.nearest_copy(r0.q, r0.q if ref is None else ref)  # short joint moves
            label_prev = None if q_prev is None else config_label(checker.ik, q_prev)
            label = config_label(checker.ik, q)
            first = "reconfigure" if j > 0 and label != label_prev else ""
            if home is not None and j == 0:  # from home to the stop's first point
                sp.samples += tr.joint_move(k, zone, home, q, r0.tcp, 0, pids[0], (False, True))
            elif q_prev is not None:  # the next zone at this stop: a planned joint move
                last = sp.samples[-1]
                sp.samples += tr.change_move(k, zone, q_prev, q, r0.tcp, last.b, pids[0], home)
            sp.samples.append(PathSample(k, zone, pids[0], pids[0], 0.0, tr.tcp_pose(r0), r0.tcp, q, first))
            for m, (a, b) in enumerate(zip(pids[:-1], pids[1:], strict=True)):
                if cancelled is not None and cancelled():
                    raise CancelledError
                rb = results[b]
                lab_b = config_label(checker.ik, np.asarray(rb.q, float))
                if lab_b != label:  # the plan uses another configuration for b: the arm
                    qb = tr.nearest_copy(rb.q, q)  # reconfigures by a planned joint move
                    sp.samples += tr.change_move(k, zone, q, qb, rb.tcp, a, b, home)
                    q, label = qb, lab_b
                    sp.samples.append(
                        PathSample(k, zone, a, b, 1.0, tr.tcp_pose(rb), rb.tcp, q, "reconfigure")
                    )
                    continue
                (Pa, Na), (Pb, Nb) = points[a], points[b]
                h = lift if sq.lifts[m] else 0.0
                seg = tr.segment(k, zone, results[a], results[b], Pa, Pb, Na, Nb, q, label, h)
                sp.samples += seg
                q = next((p.q for p in reversed(seg) if p.q is not None), q)
            done += 1
            if progress is not None:
                progress(done / n_seq)
        if home is not None and q is not None:  # back home before the carrier moves on
            last = sp.samples[-1]
            sp.samples += tr.joint_move(k, last.zone, q, home, last.tcp, last.b, 0, (True, False))
        plan.stations.append(sp)
    return plan


from core.path_csv import CSV_COLUMNS, path_csv  # noqa: E402,F401 (re-export)
