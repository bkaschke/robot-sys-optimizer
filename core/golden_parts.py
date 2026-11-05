"""Golden test parts, built procedurally so tests can compare against analytic geometry.

All parts are in meters, part frame z up, resting on z = 0 unless noted.

- flat_plate:   1.0 x 0.6 x 0.02 m plate; top face at z = 0.02.
- curved_panel: cylindrical shell, axis along x, convex up. Outer radius R, top of the
                arc at z = R_TOP_Z. Outer-surface normal at angle a is (0, sin a, cos a).
- open_box:     0.6 x 0.6 x 0.5 m box with a 0.4 x 0.4 x 0.45 m pocket (forces a long tool).
- long_rail:    6.0 x 0.2 x 0.15 m bar a fixed robot cannot cover (forces stations).
- deep_hole:    0.6 x 0.6 x 0.35 m block with a 0.1 x 0.1 m hole, 0.3 m deep (HOLE_*): narrower
                than the UR20 wrist, so the minimal tool length is known analytically (M8).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import trimesh

# curved panel parameters (shared with tests for analytic normals)
PANEL_RADIUS = 1.0
PANEL_THICKNESS = 0.01
PANEL_WIDTH = 0.6  # along x
PANEL_HALF_ANGLE = 0.4  # rad, about x
PANEL_TOP_Z = 0.3  # z of the outer surface at a = 0


def _box(extents: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(center)
    return m


def flat_plate() -> trimesh.Trimesh:
    return _box((1.0, 0.6, 0.02), (0.0, 0.0, 0.01))


def panel_outer_point(x: float, a: float) -> np.ndarray:
    """Point on the curved panel's outer surface at axial position x and arc angle a."""
    r = PANEL_RADIUS
    return np.array([x, r * np.sin(a), PANEL_TOP_Z - r + r * np.cos(a)])


def panel_outer_normal(a: float) -> np.ndarray:
    return np.array([0.0, np.sin(a), np.cos(a)])


def _shell_from_grids(top: np.ndarray, bottom: np.ndarray) -> trimesh.Trimesh:
    """Closed solid between two (nu, nv, 3) vertex grids with the same parameterization."""
    nu, nv, _ = top.shape
    idx = np.arange(nu * nv).reshape(nu, nv)
    off = nu * nv
    verts = np.vstack([top.reshape(-1, 3), bottom.reshape(-1, 3)])
    faces = []

    def quad(a, b, c, d):  # a-b-c-d counter-clockwise
        faces.append((a, b, c))
        faces.append((a, c, d))

    for i in range(nu - 1):
        for j in range(nv - 1):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            quad(a, b, c, d)
            quad(d + off, c + off, b + off, a + off)
    # side walls along the four grid borders
    border = (
        [idx[i, 0] for i in range(nu)]
        + [idx[nu - 1, j] for j in range(1, nv)]
        + [idx[i, nv - 1] for i in range(nu - 2, -1, -1)]
        + [idx[0, j] for j in range(nv - 2, 0, -1)]
    )
    for k in range(len(border)):
        p, q = border[k], border[(k + 1) % len(border)]
        quad(q, p, p + off, q + off)

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    trimesh.repair.fix_normals(mesh)  # make winding consistent and outward
    return mesh


def curved_panel(nx: int = 61, na: int = 81) -> trimesh.Trimesh:
    xs = np.linspace(-PANEL_WIDTH / 2, PANEL_WIDTH / 2, nx)
    angles = np.linspace(-PANEL_HALF_ANGLE, PANEL_HALF_ANGLE, na)
    top = np.array([[panel_outer_point(x, a) for a in angles] for x in xs])
    inward = np.array([[-panel_outer_normal(a) for a in angles] for _ in xs])
    bottom = top + PANEL_THICKNESS * inward
    return _shell_from_grids(top, bottom)


def open_box() -> trimesh.Trimesh:
    """Bottom slab + four walls, as non-overlapping boxes concatenated into one mesh."""
    outer, height, wall, floor = 0.6, 0.5, 0.1, 0.05
    inner = outer - 2 * wall
    wall_h = height - floor
    zc = floor + wall_h / 2
    parts = [
        _box((outer, outer, floor), (0, 0, floor / 2)),
        _box((outer, wall, wall_h), (0, (outer - wall) / 2, zc)),
        _box((outer, wall, wall_h), (0, -(outer - wall) / 2, zc)),
        _box((wall, inner, wall_h), ((outer - wall) / 2, 0, zc)),
        _box((wall, inner, wall_h), (-(outer - wall) / 2, 0, zc)),
    ]
    return trimesh.util.concatenate(parts)


HOLE_WIDTH = 0.1
HOLE_DEPTH = 0.3
HOLE_FLOOR = 0.05  # z of the hole bottom; rim at HOLE_FLOOR + HOLE_DEPTH


def deep_hole() -> trimesh.Trimesh:
    outer, w, h, f = 0.6, HOLE_WIDTH, HOLE_DEPTH, HOLE_FLOOR
    side = (outer - w) / 2
    zc = f + h / 2
    parts = [
        _box((outer, outer, f), (0, 0, f / 2)),
        _box((outer, side, h), (0, (w + side) / 2, zc)),
        _box((outer, side, h), (0, -(w + side) / 2, zc)),
        _box((side, w, h), ((w + side) / 2, 0, zc)),
        _box((side, w, h), (-(w + side) / 2, 0, zc)),
    ]
    return trimesh.util.concatenate(parts)


def long_rail() -> trimesh.Trimesh:
    return _box((6.0, 0.2, 0.15), (0.0, 0.0, 0.075))


GOLDEN_PARTS: dict[str, Callable[[], trimesh.Trimesh]] = {
    "flat_plate": flat_plate,
    "curved_panel": curved_panel,
    "open_box": open_box,
    "long_rail": long_rail,
    "deep_hole": deep_hole,
}
