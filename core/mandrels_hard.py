"""Hard synthetic mandrels: cases a naive rule cannot solve.

Steep-walled pockets, decoy flanges and pads, creased tool surfaces. Same conventions as
core.mandrels: metres, tool surface toward +Z, ``(mesh, truth, info)``.
"""

from __future__ import annotations

import numpy as np

from core.mandrels import _bump, _describe, _noise_and_tilt, _outward, _slab, merge_axis


def _cells_inside(xs, ys, hx, hy):
    CX, CY = np.meshgrid((xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2)
    return ((np.abs(CX) < hx) & (np.abs(CY) < hy)).ravel()


def _finish(mesh, ntop, inner, rng, noise, tilt, info):
    truth = np.zeros(len(mesh.faces), bool)
    truth[:ntop] = np.repeat(inner, 2)
    _noise_and_tilt(mesh, noise, tilt, rng)
    return mesh, truth, _describe(mesh, truth, dict(info, noise_m=noise, tilt_deg=tilt))


def steep_pocket(
    wall_deg: float = 80.0,
    depth: float = 0.10,
    floor=(0.22, 0.16),
    rim: float = 0.06,
    resolution: float = 0.015,
    seed: int = 0,
    noise: float = 0.0,
    tilt: float = 0.0,
    thickness: float = 0.004,
    base_height: float = 0.15,
):
    """Female pocket: floor, walls at ``wall_deg`` from horizontal (<= 88), flat rim.

    Truth = floor + walls (the laminate goes down the walls). The rim is an up-facing decoy.
    """
    if not 20.0 <= wall_deg <= 88.0:
        raise ValueError(f"wall_deg must be within 20..88 (no undercuts), got {wall_deg}")
    rng = np.random.default_rng(seed)
    fx, fy = floor
    dw = depth / np.tan(np.radians(wall_deg))
    ox, oy = fx + dw + rim, fy + dw + rim
    bx = [-ox, -(fx + dw), -fx, fx, fx + dw, ox]
    by = [-oy, -(fy + dw), -fy, fy, fy + dw, oy]
    xs = merge_axis(np.linspace(-ox, ox, int(2 * ox / resolution) + 1), bx, resolution)
    ys = merge_axis(np.linspace(-oy, oy, int(2 * oy / resolution) + 1), by, resolution)
    X, Y = np.meshgrid(xs, ys)
    s = np.maximum(np.clip((np.abs(X) - fx) / dw, 0, 1), np.clip((np.abs(Y) - fy) / dw, 0, 1))
    cx, cy = rng.uniform(-0.08, 0.08, 2)
    bump = 0.012 * _bump(X, Y, cx, cy, fx * 0.5, fy * 0.5) * (1 - s)
    v, f, ntop = _slab(xs, ys, base_height + depth * s + bump, thickness)
    inner = _cells_inside(xs, ys, fx + dw, fy + dw)
    return _finish(
        _outward(v, f),
        ntop,
        inner,
        rng,
        noise,
        tilt,
        dict(kind="steep_pocket", seed=seed, wall_deg=wall_deg, depth_m=depth, thickness_m=thickness),
    )


def decoy_shell(
    drop: float = 0.08,
    ambiguous: bool = False,
    inner=(0.30, 0.24),
    flange: float = 0.25,
    resolution: float = 0.01,
    seed: int = 0,
    noise: float = 0.0,
    tilt: float = 0.0,
    thickness: float = 0.004,
    relief: float = 0.04,
    base_height: float = 0.15,
):
    """Small raised male surface surrounded by a wide flat up-facing flange (a decoy).

    Default: the flange sits ``drop`` lower, joined by a 60 degree step. ``ambiguous=True``:
    the flange is at the same height, joined only by the dome's own edge slope (~35-40 degrees
    crease), so the tool surface boundary is genuinely unclear. Truth = the raised surface.
    """
    rng = np.random.default_rng(seed)
    Lx, Ly = inner
    drop = 0.0 if ambiguous else drop
    rw = drop / np.tan(np.radians(60.0)) if drop else 0.0
    hx, hy = Lx / 2 + rw + flange, Ly / 2 + rw + flange
    bx = [-hx, -(Lx / 2 + rw), -Lx / 2, Lx / 2, Lx / 2 + rw, hx]
    by = [-hy, -(Ly / 2 + rw), -Ly / 2, Ly / 2, Ly / 2 + rw, hy]
    xs = merge_axis(np.linspace(-hx, hx, int(2 * hx / (2 * resolution)) + 1), bx, 2 * resolution)
    ys = merge_axis(np.linspace(-hy, hy, int(2 * hy / (2 * resolution)) + 1), by, 2 * resolution)
    # fine grid over the raised surface itself
    xi = np.linspace(-Lx / 2, Lx / 2, int(Lx / resolution) + 1)
    yi = np.linspace(-Ly / 2, Ly / 2, int(Ly / resolution) + 1)
    xs = np.unique(np.round(np.concatenate([xs[np.abs(xs) >= Lx / 2 - 1e-9], xi]), 9))
    ys = np.unique(np.round(np.concatenate([ys[np.abs(ys) >= Ly / 2 - 1e-9], yi]), 9))
    X, Y = np.meshgrid(xs, ys)
    Xc, Yc = np.clip(X, -Lx / 2, Lx / 2), np.clip(Y, -Ly / 2, Ly / 2)
    d = np.hypot(X - Xc, Y - Yc)
    u, w = 2 * Xc / Lx, 2 * Yc / Ly
    win = (1 - u**2) * (1 - w**2)
    win = win if ambiguous else win**2  # squared window: zero edge slope
    ox, oy = rng.uniform(-0.1, 0.1, 2)
    dimple = -0.3 * relief * _bump(Xc, Yc, ox * Lx, oy * Ly, Lx * 0.07, Ly * 0.08) * (1 - u**2) * (1 - w**2)
    z = base_height + (relief * 1.5 if ambiguous else relief) * win + dimple
    if drop:
        z = z - drop * np.clip(d / rw, 0, 1)
    v, f, ntop = _slab(xs, ys, z, thickness)
    inner_c = _cells_inside(xs, ys, Lx / 2, Ly / 2)
    return _finish(
        _outward(v, f),
        ntop,
        inner_c,
        rng,
        noise,
        tilt,
        dict(
            kind="decoy_shell",
            seed=seed,
            drop_m=drop,
            flange_width_m=flange,
            ambiguous_flange=bool(ambiguous),
            thickness_m=thickness,
        ),
    )
