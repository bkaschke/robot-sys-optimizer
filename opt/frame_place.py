"""Gantry / rail placement (v9, issue #9): where the whole carrier frame goes relative to the part,
before the stops are planned.

Candidate frames:
    rail    the track along the part's long axis, on either side, its line at each **standoff**
            (``d_min`` .. ``d_max`` from the part outline, step ``step``)
    gantry  the frame turned along or across the part's long axis, centred on the part and moved
            by ``shift_step`` up to ``shift`` along both axes
each with the robot mount height (``T_mount`` z, the **z range** ``z_lo`` .. ``z_hi`` in ``z_steps``
values): how high the robot sits on the carriage (rail riser) or how low it hangs (gantry).

A candidate is kept when the carrier's fixed bodies (track, legs, beams) keep the base's part
standoff (``carrier_part_min``, 3D distance) and its stops can exist (the planner's admissible
stations: standoff, keep-outs, travel). It is scored like the planner's first step: the targets the
admissible stops cover (reach maps) and the fewest stops that cover them (greedy set cover); ranked
by (targets not covered, stops, mean manipulability). The best few are then planned in full by the
caller (ui.frame_ui).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from core.schema import Base, Pose
from core.transforms import pose_to_matrix


@dataclass
class FrameSettings:
    d_min: float = 0.3  # m, standoff of the track line from the part outline (rail)
    d_max: float = 1.2
    step: float = 0.15  # m
    shift: float = 0.6  # m, gantry: the frame moved this far off the part's centre, both ways
    shift_step: float = 0.3  # m
    z_lo: float | None = None  # m, robot mount height (T_mount z); None: the current one
    z_hi: float | None = None
    z_steps: int = 3
    candidate_step: float = 0.25  # stops along the travel, for scoring (the full plan uses its own)
    reach_threshold: float = 0.5
    top: int = 3  # frames planned in full by the caller


@dataclass
class FrameScore:
    pose: Pose
    mount_z: float
    uncovered: int
    stops: int
    manip: float
    reason: str = ""
    base: Base | None = field(default=None, repr=False)

    def key(self) -> tuple:
        return (self.uncovered, self.stops, -self.manip)


def _axes(part_fp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centre, long axis u, normal n) of the part footprint's minimum rectangle."""
    rect = part_fp.minimum_rotated_rectangle
    c4 = np.asarray(rect.exterior.coords)[:4]
    edges = np.roll(c4, -1, axis=0) - c4
    u = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    u = u / max(np.linalg.norm(u), 1e-12)
    return np.asarray(rect.centroid.coords[0]), u, np.array([-u[1], u[0]])


def _mount_heights(base: Base, st: FrameSettings) -> list[float]:
    z0 = float(base.T_mount.xyz[2])
    lo = z0 if st.z_lo is None else st.z_lo
    hi = z0 if st.z_hi is None else st.z_hi
    if hi < lo:
        lo, hi = hi, lo
    n = max(1, int(st.z_steps)) if hi > lo + 1e-9 else 1
    return [float(v) for v in np.linspace(lo, hi, n)]


def candidate_frames(base: Base, part_fp, st: FrameSettings) -> list[tuple[Pose, float]]:
    """(frame pose, robot mount height) candidates for the base's carrier (see the docstring)."""
    centre, u, n = _axes(part_fp)
    z = base.pose.xyz[2]
    roll, pitch = base.pose.rpy[0], base.pose.rpy[1]
    pts = np.asarray(part_fp.exterior.coords)
    out: list[Pose] = []
    if base.carrier == "rail":
        yaw = float(np.arctan2(u[1], u[0]))
        for side in (1.0, -1.0):
            half = float(np.max((pts - centre) @ (side * n)))
            for d in np.arange(st.d_min, st.d_max + 1e-9, st.step):
                o = centre + side * (half + d) * n
                out.append(Pose(xyz=(float(o[0]), float(o[1]), z), rpy=(roll, pitch, yaw)))
    elif base.carrier == "gantry":
        offs = np.arange(-st.shift, st.shift + 1e-9, st.shift_step) if st.shift > 0 else np.array([0.0])
        for axis in (u, n):  # the frame's x along, then across the part
            ax = np.array([-axis[1], axis[0]])
            yaw = float(np.arctan2(axis[1], axis[0]))
            for a in offs:
                for b in offs:
                    o = centre + a * axis + b * ax
                    out.append(Pose(xyz=(float(o[0]), float(o[1]), z), rpy=(roll, pitch, yaw)))
    else:
        raise ValueError("frame placement is for a rail or a gantry")
    heights = _mount_heights(base, st)
    return [(p, h) for p in out for h in heights]


def with_frame(base: Base, pose: Pose, mount_z: float) -> Base:
    m = base.T_mount
    return base.model_copy(
        update={
            "pose": pose,
            "align_to_part": False,  # the placement chose the frame
            "T_mount": m.model_copy(update={"xyz": (m.xyz[0], m.xyz[1], float(mount_z))}),
        }
    )


def score_frame(
    base: Base, rmap, t, part_mesh, T_part, st: FrameSettings, resolve=None, obstacles=None
) -> FrameScore:
    """How well a frame's stops can cover the targets (the planner's first step)."""
    from core.carrier import Carrier
    from core.standoff import distance_3d
    from opt.reachability import coverage_matrix
    from opt.set_cover import greedy_cover
    from opt.sites import CarrierSites

    pose, mount_z = base.pose, float(base.T_mount.xyz[2])
    c = Carrier(base, resolve=resolve)
    fixed = c.static_bodies()
    if fixed and distance_3d(fixed, part_mesh, T_part) < base.standoff_m.carrier_part_min - 1e-6:
        return FrameScore(pose, mount_z, len(t.ids), 0, 0.0, "fixed bodies too close to the part", base)
    sites = CarrierSites(c, part_mesh, T_part, obstacles=obstacles)
    cand = sites.candidates(t.P_world, st.candidate_step)
    if not cand:
        return FrameScore(pose, mount_z, len(t.ids), 0, 0.0, "no admissible stop", base)
    Ts = np.array([sites.T(v) for v in cand])
    score = np.zeros((len(cand), len(t.ids)))
    mu = np.zeros_like(score)
    rots = np.round(Ts[:, :3, 2], 6)
    for key in np.unique(rots, axis=0):
        rows = np.flatnonzero(np.all(rots == key, axis=1))
        sc, m = coverage_matrix(rmap, t.P_world, t.N_world, t.cones, Ts[rows, :3, 3], Ts[rows[0], :3, :3])
        score[rows], mu[rows] = sc, m
    covered = score >= st.reach_threshold
    reach = covered.any(axis=0)
    chosen = greedy_cover(covered, mu / max(rmap.max_manip, 1e-12), len(cand))
    best_mu = np.where(covered, mu, 0.0).max(axis=0)
    manip = float(best_mu[reach].mean() / max(rmap.max_manip, 1e-12)) if reach.any() else 0.0
    return FrameScore(pose, mount_z, int((~reach).sum()), len(chosen), manip, "", base)


def rank_frames(
    base: Base,
    rmap,
    t,
    part_mesh,
    T_part,
    part_fp,
    st: FrameSettings,
    resolve=None,
    obstacles=None,
    progress: Callable[[str, float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[FrameScore]:
    """Every candidate frame scored, best first (failed ones last, with their reason)."""
    from core.feasibility import CancelledError

    cands = candidate_frames(base, part_fp, st)
    out = []
    for k, (pose, h) in enumerate(cands):
        if cancelled is not None and cancelled():
            raise CancelledError()
        if progress is not None:
            progress(f"frame {k + 1} / {len(cands)}", 0.6 * k / max(len(cands), 1))
        out.append(score_frame(with_frame(base, pose, h), rmap, t, part_mesh, T_part, st, resolve, obstacles))
    return sorted(out, key=lambda f: (bool(f.reason), f.key()))


def frame_text(f: FrameScore) -> str:
    T = pose_to_matrix(f.pose)
    yaw = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
    x, y, z = T[0, 3] * 1000, T[1, 3] * 1000, f.mount_z * 1000
    return f"x {x:.0f}, y {y:.0f} mm, yaw {yaw:.0f}°, mount z {z:.0f} mm"
