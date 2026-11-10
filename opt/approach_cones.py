"""Approach cones (v8 T15): from which directions can a tool reach each target at all?

For every target the tool body is modelled as a straight capsule leaving the target along a
candidate direction ``u`` (within ``max_deg`` of the surface normal): the centre line and four
lines on its radius are cast as rays; ``u`` is *free* when none of them hits the part within the
tool's length. No robot, no IK: it is about the part's geometry (slots, pockets, walls), so it
is fast and runs before any planning.

The free directions suggest tool shapes: a target whose free cone points away from where the
arm comes from (the base) needs a bent head. ``bend_candidates`` takes, per target, the free
direction closest to the arm's side, the angle between it and the direction to the base (the
bend the head needs so the flange can still point back toward the arm), and clusters those
angles into a few candidate bend angles. The tool optimizer can start its shape search from the
best of them (Tool and stations -> Start shape: From approach cones).
"""

from __future__ import annotations

import numpy as np
import trimesh

N_DIRS = 80
MAX_DEG = 75.0


def cap_directions(n: np.ndarray, max_deg: float = MAX_DEG, count: int = N_DIRS) -> np.ndarray:
    """Unit vectors within ``max_deg`` of ``n`` (Fibonacci spiral on the cap; the first is n)."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    k = np.arange(count)
    cos_max = np.cos(np.radians(max_deg))
    z = 1 - (1 - cos_max) * k / max(count - 1, 1)
    phi = k * np.pi * (3 - np.sqrt(5))
    r = np.sqrt(np.maximum(0, 1 - z**2))
    local = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(n, a)
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    return local @ np.stack([x, y, n])


def free_directions(
    mesh: trimesh.Trimesh,
    P: np.ndarray,
    N: np.ndarray,
    length: float = 0.4,
    radius: float = 0.03,
    max_deg: float = MAX_DEG,
    count: int = N_DIRS,
    lift: float = 0.004,
) -> list[np.ndarray]:
    """Per target (part frame): the free directions (k, 3), out of ``count`` candidates."""
    origins, dirs, owner = [], [], []
    cand = []
    for i, (p, n) in enumerate(zip(P, N, strict=True)):
        U = cap_directions(n, max_deg, count)
        cand.append(U)
        for j, u in enumerate(U):
            a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
            e1 = np.cross(u, a)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(u, e1)
            start = (
                p + n * (lift + radius) + u * radius
            )  # past the contact; the capsule's side clear of the surface
            for off in (np.zeros(3), e1, -e1, e2, -e2):
                origins.append(start + off * radius)
                dirs.append(u)
                owner.append((i, j))
    origins, dirs = np.array(origins), np.array(dirs)
    blocked = [np.zeros(count, bool) for _ in P]
    if len(origins):
        loc, ray, _tri = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
        if len(ray):
            d = np.einsum("ij,ij->i", loc - origins[ray], dirs[ray])
            for r in ray[(d > 0) & (d < length)]:
                i, j = owner[r]
                blocked[i][j] = True
    return [U[~b] for U, b in zip(cand, blocked, strict=True)]


def bend_candidates(free: list[np.ndarray], P: np.ndarray, base: np.ndarray, k: int = 3) -> list[float]:
    """Candidate head bend angles (deg, ascending): per target with a free cone, the angle
    between the free direction closest to the base direction and the base direction itself;
    clustered (1-D k-means) into at most ``k`` values. Targets with no free direction are
    skipped (no tool reaches them)."""
    ang = []
    for U, p in zip(free, P, strict=True):
        if len(U) == 0:
            continue
        b = np.asarray(base, float) - p
        b /= np.linalg.norm(b) or 1.0
        c = U @ b
        ang.append(float(np.degrees(np.arccos(np.clip(c.max(), -1, 1)))))
    if not ang:
        return []
    a = np.array(ang)
    centres = np.quantile(a, np.linspace(0.1, 0.9, min(k, len(np.unique(a)))))
    for _ in range(20):
        lab = np.argmin(np.abs(a[:, None] - centres[None, :]), axis=1)
        new = np.array([a[lab == c].mean() if np.any(lab == c) else centres[c] for c in range(len(centres))])
        if np.allclose(new, centres):
            break
        centres = new
    return sorted({round(float(c), 1) for c in centres})


def shape_seeds(ev, v0, k: int = 3) -> list:
    """Tool values to try first (opt.tool_geometry evaluator ``ev``): ``v0`` with the bend
    (legacy bent tools: theta_deg; pose tools: the head turn rx / ry with the widest range) set
    to each candidate angle, both signs, clipped into the range. The free directions are kept
    on ``ev.cones``."""
    x = ev.ee.extension
    if x.type == "pose":
        keys = [kk for kk in ("rx_deg", "ry_deg") if getattr(x, kk)[1] - getattr(x, kk)[0] > 1e-9]
        key = max(keys, key=lambda kk: getattr(x, kk)[1] - getattr(x, kk)[0]) if keys else None
    else:
        key = "theta_deg" if x.type in ("bent", "bent2") and x.theta_deg[1] - x.theta_deg[0] > 1e-9 else None
    if key is None:
        return []
    T = np.linalg.inv(ev.T_part) @ ev.T_base0  # the robot base in the part frame
    free = free_directions(
        ev.part, ev.t.P_part, ev.t.N_part, length=max(0.15, float(ev.tm.length(v0))), radius=x.radius
    )
    ev.cones = free
    lo, hi = getattr(x, key)
    out, seen = [], set()
    for a in bend_candidates(free, ev.t.P_part, T[:3, 3], k):
        for val in (a, -a):
            val = float(np.clip(val, lo, hi))
            if round(val, 3) in seen or abs(val - getattr(v0, key)) < 1e-6:
                continue
            seen.add(round(val, 3))
            out.append(v0.model_copy(update={key: val}))
    return out
