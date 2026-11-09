"""Unreachable targets, made obvious (v6 feedback): a large red dot, a red cross on the surface
and an id label at every point the selected plan (or, without one, the feasibility check)
cannot reach. Drawn by ui.targets_ui after the target points; the log (ui.log_panel) says why.
"""

from __future__ import annotations

import numpy as np

RED = (235, 20, 20)
MAX_LABELS = 30  # beyond this only dots and crosses (labels would clutter the view)


def unreached_ids(ids, plan_results: dict, feasibility: dict) -> list[int]:
    """Point ids to flag: unreachable in the selected plan's results, else in the
    feasibility check (points without a result are not flagged)."""
    src = plan_results or feasibility or {}
    return [i for i in ids if (r := src.get(i)) is not None and r.status == "unreachable"]


def crosses(P: np.ndarray, N: np.ndarray, size: float) -> np.ndarray:
    """(2 n, 2, 3) line segments: an X of arm length ``size`` in each point's tangent plane,
    lifted 1 mm along the normal so it sits on the surface."""
    a = np.cross(N, np.where(np.abs(N[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = np.cross(N, a)
    c = P + 0.001 * N
    d1, d2 = (a + b) * size / np.sqrt(2), (a - b) * size / np.sqrt(2)
    return np.concatenate([np.stack([c - d1, c + d1], 1), np.stack([c - d2, c + d2], 1)])


def draw_unreached(server, root: str, ids, P, N, bad: list[int], point_size: float) -> dict:
    """Scene handles {name: handle} for the flagged points (empty when none)."""
    if not bad:
        return {}
    index = {pid: k for k, pid in enumerate(ids)}
    bad = [i for i in bad if i in index]
    k = np.array([index[i] for i in bad], int)
    if len(k) == 0:
        return {}
    Pb, Nb = np.asarray(P, float)[k], np.asarray(N, float)[k]
    size = max(3 * point_size, 0.012)
    out = {
        "unreached": server.scene.add_point_cloud(
            f"{root}/unreached", Pb.astype(np.float32), RED, point_size=size, point_shape="circle"
        ),
        "unreached_x": server.scene.add_line_segments(
            f"{root}/unreached_x",
            crosses(Pb, Nb, 1.5 * size).astype(np.float32),
            RED,
            thickness=4.0,
            thickness_units="screen",
        ),
    }
    for j, (pid, p, n) in enumerate(zip(bad, Pb, Nb, strict=True)):
        if j >= MAX_LABELS:
            break
        out[f"unreached_label_{pid}"] = server.scene.add_label(
            f"{root}/unreached_label_{pid}", f"✘ {pid}", position=tuple(p + 0.02 * n)
        )
    return out
