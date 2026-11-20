"""Unreachable targets, made obvious (v6 feedback): a large red dot, a red cross on the surface
and an id label at every point the selected plan (or, without one, the feasibility check)
cannot reach. Drawn by ui.targets_ui after the target points. Coloured by why (``REASON_COLORS``:
out of reach, joint limits, singularity, collision) and the label names it, for a collision
with what the best arm pose hits (core.collision.describe_hits); the log (ui.log_panel) lists
them too.
"""

from __future__ import annotations

import numpy as np

RED = (235, 20, 20)
REASON_COLORS = {
    "no_ik": RED,  # out of reach
    "joint_limits": (245, 140, 0),
    "singular": (150, 60, 220),
    "collision": (225, 0, 170),
}
SHORT = {"no_ik": "out of reach", "joint_limits": "joint limits", "singular": "singularity"}
LEGEND = "colour = why: red out of reach, orange joint limits, violet singularity, magenta collision"
MAX_LABELS = 30  # beyond this only dots and crosses (labels would clutter the view)


def unreached_ids(ids, plan_results: dict, feasibility: dict) -> list[int]:
    """Point ids to flag: unreachable in the selected plan's results, else in the
    feasibility check (points without a result are not flagged)."""
    src = plan_results or feasibility or {}
    return [i for i in ids if (r := src.get(i)) is not None and r.status == "unreachable"]


def why(r) -> str:
    """Short reason of an unreachable result: "out of reach", "collision: part, floor", ..."""
    from core.collision import HITS

    code = getattr(r, "reason", "") or ""
    if code == "collision":
        note = next((n for n in getattr(r, "notes", []) if n.startswith(HITS)), "")
        objs = [p.split(" (")[0] for p in note[len(HITS) :].split("; ") if p]
        return "collision: " + ", ".join(objs) if objs else "collision"
    return SHORT.get(code, code.replace("_", " ") or "not covered")


def crosses(P: np.ndarray, N: np.ndarray, size: float) -> np.ndarray:
    """(2 n, 2, 3) line segments: an X of arm length ``size`` in each point's tangent plane,
    lifted 1 mm along the normal so it sits on the surface."""
    a = np.cross(N, np.where(np.abs(N[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = np.cross(N, a)
    c = P + 0.001 * N
    d1, d2 = (a + b) * size / np.sqrt(2), (a - b) * size / np.sqrt(2)
    return np.concatenate([np.stack([c - d1, c + d1], 1), np.stack([c - d2, c + d2], 1)])


GAP = (0, 170, 210)  # v10 T10: edge-check gaps (cyan, apart from the reason colours)


def draw_gaps(server, root: str, P: np.ndarray, N: np.ndarray, point_size: float) -> dict:
    """Edge-check gaps (verification points no stop reaches): cyan dots + crosses, one label."""
    if len(P) == 0:
        return {}
    P, N = np.asarray(P, np.float32), np.asarray(N, float)
    size = max(2 * point_size, 0.008)
    seg = crosses(P.astype(float), N, 1.2 * size).astype(np.float32)
    return {
        "edge_gaps": server.scene.add_point_cloud(
            f"{root}/edge_gaps",
            P,
            np.tile(np.array(GAP, np.uint8), (len(P), 1)),
            point_size=size,
            point_shape="circle",
        ),
        "edge_gaps_x": server.scene.add_line_segments(
            f"{root}/edge_gaps_x", seg, GAP, thickness=3.0, thickness_units="screen"
        ),
        "edge_gaps_label": server.scene.add_label(
            f"{root}/edge_gaps_label", f"✘ {len(P)} edge gap(s)", position=tuple(P[0] + 0.03 * N[0])
        ),
    }


UNFIT = (150, 90, 40)  # v10 T12: below the tool's minimum surface radius (brown)


def draw_unfit(server, root: str, ids, P, N, bad: dict, point_size: float) -> dict:
    """Points the tool cannot fit (core.surface_radius.Fit.bad): brown rings + crosses + labels."""
    index = {int(p): k for k, p in enumerate(ids)}
    keep = [p for p in sorted(bad) if p in index]
    if not keep:
        return {}
    k = np.array([index[p] for p in keep])
    Pb, Nb = np.asarray(P, float)[k], np.asarray(N, float)[k]
    size = max(2.5 * point_size, 0.01)
    out = {
        "unfit_x": server.scene.add_line_segments(
            f"{root}/unfit_x",
            crosses(Pb, Nb, 1.3 * size).astype(np.float32),
            UNFIT,
            thickness=3.0,
            thickness_units="screen",
        )
    }
    for j, pid in enumerate(keep[:MAX_LABELS]):
        _, r, rmin, kind = bad[pid]
        out[f"unfit_label_{pid}"] = server.scene.add_label(
            f"{root}/unfit_label_{pid}",
            f"✘ {pid} radius {1000 * r:.0f} mm {kind} < {1000 * rmin:.0f}",
            position=tuple(Pb[j] + 0.02 * Nb[j]),
        )
    return out


def unreached_key(ids, P, N, bad: list[int], point_size: float, results=None) -> bytes:
    """Digest of everything ``draw_unreached`` draws for these arguments (positions, normals,
    reasons and labels of the flagged points): the markers are redrawn only when it changes."""
    from ui.targets_draw import digest

    index = {pid: k for k, pid in enumerate(ids)}
    bad = [i for i in bad if i in index]
    k = np.array([index[i] for i in bad], int)
    res = results or {}
    texts = [(getattr(res.get(i), "reason", ""), why(res[i]) if i in res else "") for i in bad]
    return digest(np.asarray(P, float)[k], np.asarray(N, float)[k], bad, texts, float(point_size))


def draw_unreached(server, root: str, ids, P, N, bad: list[int], point_size: float, results=None) -> dict:
    """Scene handles {name: handle} for the flagged points (empty when none); ``results``
    (point id -> PointResult) colours and labels them by reason."""
    if not bad:
        return {}
    index = {pid: k for k, pid in enumerate(ids)}
    bad = [i for i in bad if i in index]
    k = np.array([index[i] for i in bad], int)
    if len(k) == 0:
        return {}
    Pb, Nb = np.asarray(P, float)[k], np.asarray(N, float)[k]
    size = max(3 * point_size, 0.012)
    res = results or {}
    C = np.array([REASON_COLORS.get(getattr(res.get(i), "reason", ""), RED) for i in bad], np.uint8)
    out = {
        "unreached": server.scene.add_point_cloud(
            f"{root}/unreached", Pb.astype(np.float32), C, point_size=size, point_shape="circle"
        ),
        "unreached_x": server.scene.add_line_segments(
            f"{root}/unreached_x",
            crosses(Pb, Nb, 1.5 * size).astype(np.float32),
            np.repeat(np.concatenate([C, C])[:, None, :], 2, axis=1),
            thickness=4.0,
            thickness_units="screen",
        ),
    }
    for j, (pid, p, n) in enumerate(zip(bad, Pb, Nb, strict=True)):
        if j >= MAX_LABELS:  # one summary label for the rest (their dots and crosses stay)
            out["unreached_label_more"] = server.scene.add_label(
                f"{root}/unreached_label_more",
                f"+{len(bad) - MAX_LABELS} more unreached",
                position=tuple(p + 0.02 * n),
            )
            break
        out[f"unreached_label_{pid}"] = server.scene.add_label(
            f"{root}/unreached_label_{pid}",
            f"✘ {pid} {why(res[pid])}" if pid in res else f"✘ {pid}",
            position=tuple(p + 0.02 * n),
        )
    return out


def draw_extras(tab, root: str, ids, P, N) -> dict:
    """The Targets tab's extra markers: edge-check gaps (v10 T10) and points below the tool's minimum
    surface radius (v10 T12)."""
    from ui.radius_ui import fit_of

    size = tab.point_size.value / 1000
    out = {}
    gaps = getattr(tab, "edge_gaps", None)
    if gaps is not None:
        out.update(draw_gaps(tab.server, root, *gaps(), size))
    out.update(draw_unfit(tab.server, root, ids, P, N, fit_of(tab.s).bad, size))
    return out
