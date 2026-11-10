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
        if j >= MAX_LABELS:
            break
        out[f"unreached_label_{pid}"] = server.scene.add_label(
            f"{root}/unreached_label_{pid}",
            f"✘ {pid} {why(res[pid])}" if pid in res else f"✘ {pid}",
            position=tuple(p + 0.02 * n),
        )
    return out
