"""Synthetic composite-layup mandrels with ground-truth tool-surface masks.

Test data and demo material for tool-surface detection. Everything is in metres, the tool
surface faces +Z, and every generator returns ``(mesh, truth_mask, info)`` where ``truth_mask``
is a bool array per face of ``mesh``. Meshes are built without boolean operations: each
component (faceplate slab, rib plate, frame rail, boss) is a closed, consistently wound shell,
so the composite is watertight per component but components overlap where a real mandrel
would be one welded solid (``info["components"]`` says how many). No viser / ui imports.
"""

from __future__ import annotations

import numpy as np
import trimesh

_Z = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------- mesh building blocks
def _slab(xs: np.ndarray, ys: np.ndarray, ztop: np.ndarray, thickness: float, skip=None):
    """Closed slab under a height field. ``ztop`` is [len(ys), len(xs)].

    Returns (vertices, faces, n_top_faces); the first ``n_top_faces`` faces are the top
    surface, two per cell, ordered cell (j, i) -> 2 * (j * (nx - 1) + i).
    ``skip`` ([ny-1, nx-1] bool) cuts cells through the slab (perforation, with hole walls);
    skipped cells must not touch the border or each other, even diagonally.
    """
    nx, ny = len(xs), len(ys)
    X, Y = np.meshgrid(xs, ys)
    top = np.column_stack([X.ravel(), Y.ravel(), ztop.ravel()])
    bot = top - np.array([0.0, 0.0, thickness])
    verts = np.vstack([top, bot])
    n = nx * ny
    idx = np.arange(n).reshape(ny, nx)
    v00, v10, v11, v01 = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    top_f = np.stack(
        [np.stack([v00, v10, v11], -1).reshape(-1, 3), np.stack([v00, v11, v01], -1).reshape(-1, 3)], 1
    ).reshape(-1, 3)  # interleaved: cell k -> faces 2k, 2k+1
    if skip is not None:
        skip = np.asarray(skip, bool)
        top_f = top_f.reshape(-1, 2, 3)[~skip.ravel()].reshape(-1, 3)
    bot_f = top_f[:, ::-1] + n
    # CCW boundary loop (seen from +Z): bottom row -> right column -> top row -> left column
    loop = np.concatenate([idx[0, :-1], idx[:-1, -1], idx[-1, :0:-1], idx[:0:-1, 0]])
    nxt = np.roll(loop, -1)
    sides = np.vstack([np.column_stack([loop, loop + n, nxt + n]), np.column_stack([loop, nxt + n, nxt])])
    walls = []
    if skip is not None:
        for j, i in zip(*np.nonzero(skip), strict=True):
            ring = [idx[j, i], idx[j, i + 1], idx[j + 1, i + 1], idx[j + 1, i]]
            nbr = [(j - 1, i), (j, i + 1), (j + 1, i), (j, i - 1)]
            for k, (nj, ni) in enumerate(nbr):
                a, b = ring[k], ring[(k + 1) % 4]
                if not skip[nj, ni]:  # solid beyond this edge: wall faces into the hole
                    walls += [(a, b, b + n), (a, b + n, a + n)]
    faces = np.vstack([top_f, bot_f, sides] + ([np.array(walls)] if walls else []))
    return verts, faces, len(top_f)


def merge_axis(base: np.ndarray, breaks, res: float) -> np.ndarray:
    """Axis samples with exact grid lines at ``breaks`` (base samples next to a break are dropped)."""
    br = np.asarray(sorted(breaks), float)
    keep = (
        np.array([np.all(np.abs(br - x) > 0.3 * res) for x in base]) if len(br) else np.ones(len(base), bool)
    )
    return np.unique(np.round(np.concatenate([base[keep], br]), 9))


def _prism(s: np.ndarray, c0: float, thick: float, zbot: float, ztop: np.ndarray, along_x: bool):
    """Vertical plate (rib) running along s, centred at c0 across, top profile ``ztop(s)``."""
    n = len(s)
    v = np.zeros((4 * n, 3))
    for i in range(n):
        for side, off in enumerate((-thick / 2, thick / 2)):
            v[4 * i + 2 * side + 0] = (s[i], c0 + off, zbot)
            v[4 * i + 2 * side + 1] = (s[i], c0 + off, ztop[i])
    f = []

    def quad(a, b, c, d):
        f.extend([(a, b, c), (a, c, d)])

    def vid(i, side, t):
        return 4 * i + 2 * side + t

    for i in range(n - 1):
        quad(vid(i, 0, 1), vid(i + 1, 0, 1), vid(i + 1, 1, 1), vid(i, 1, 1))  # top
        quad(vid(i, 0, 0), vid(i, 1, 0), vid(i + 1, 1, 0), vid(i + 1, 0, 0))  # bottom
        quad(vid(i, 0, 0), vid(i + 1, 0, 0), vid(i + 1, 0, 1), vid(i, 0, 1))  # front
        quad(vid(i, 1, 0), vid(i, 1, 1), vid(i + 1, 1, 1), vid(i + 1, 1, 0))  # back
    quad(vid(0, 1, 0), vid(0, 0, 0), vid(0, 0, 1), vid(0, 1, 1))
    j = n - 1
    quad(vid(j, 0, 0), vid(j, 1, 0), vid(j, 1, 1), vid(j, 0, 1))
    f = np.array(f)
    if not along_x:
        v = v[:, [1, 0, 2]]
        f = f[:, ::-1]
    return v, f


def _outward(verts: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    m = trimesh.Trimesh(verts, faces, process=False)
    if m.volume < 0:
        m = trimesh.Trimesh(verts, faces[:, ::-1], process=False)
    return m


def _assemble(parts: list[tuple[np.ndarray, np.ndarray]]):
    """Concatenate closed components without merging vertices; returns mesh and face offsets."""
    vs, fs, off, nv = [], [], [], 0
    nf = 0
    for v, f in parts:
        m = _outward(v, f)
        vs.append(m.vertices)
        fs.append(m.faces + nv)
        off.append(nf)
        nv += len(v)
        nf += len(f)
    return trimesh.Trimesh(np.vstack(vs), np.vstack(fs), process=False), off


def _describe(mesh: trimesh.Trimesh, truth: np.ndarray, info: dict) -> dict:
    out = dict(info)
    out["n_faces"] = int(len(mesh.faces))
    out["truth_faces"] = int(truth.sum())
    out["truth_area_m2"] = float(mesh.area_faces[truth].sum())
    out["watertight"] = bool(mesh.is_watertight)
    out["winding_consistent"] = bool(mesh.is_winding_consistent)
    out["components"] = int(len(mesh.split(only_watertight=False))) if len(mesh.faces) < 200000 else -1
    zmean = float((mesh.face_normals[truth, 2] * mesh.area_faces[truth]).sum() / mesh.area_faces[truth].sum())
    out["truth_mean_nz"] = zmean
    return out


def _bump(x, y, cx, cy, wx, wy):
    return np.exp(-(((x - cx) / wx) ** 2) - ((y - cy) / wy) ** 2)


def _noise_and_tilt(mesh: trimesh.Trimesh, noise_m: float, tilt_deg: float, rng) -> trimesh.Trimesh:
    if noise_m > 0:
        mesh.vertices = mesh.vertices + rng.normal(0.0, noise_m, mesh.vertices.shape)
    if tilt_deg:
        phi = rng.uniform(0, 2 * np.pi)
        axis = np.array([np.cos(phi), np.sin(phi), 0.0])
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(tilt_deg), axis, mesh.centroid)
        )
    return mesh


# ---------------------------------------------------------------- generators
def faceplate_eggcrate(
    size=(1.2, 0.9),
    resolution: float = 0.02,
    seed: int = 0,
    noise: float = 0.0,
    tilt: float = 0.0,
    thickness: float = 0.012,
    margin: float = 0.05,
    saddle: float = 0.04,
    bump: float = 0.05,
    rib_pitch: float = 0.15,
    rib_thickness: float = 0.004,
    height: float = 0.25,
    frame_width: float = 0.04,
    frame_height: float = 0.05,
    bosses: int = 4,
    relief=None,
    breaks=None,
    plinth=None,
    pins: int = 0,
    perforate: int = 0,
):
    """Freeform faceplate (saddle + bump + dimple) on an egg-crate of ribs and a base frame.

    Truth = the whole top surface of the faceplate (the run-off ``margin`` is included and
    recorded in ``info``). Ribs end inside the plate, so their tops face up but are hidden.

    Hard-case options: ``relief(x, y)`` is added to the height (creases); ``breaks`` =
    (x_lines, y_lines) puts grid lines on the creases; ``plinth`` = (width, under) adds a flat
    machined pad beside the frame, extending ``under`` m beneath the plate (decoy, not truth);
    ``pins`` adds raised tooling pins on the surface (excluded from truth); ``perforate`` = k
    cuts square holes through the plate every k cells.
    """
    rng = np.random.default_rng(seed)
    Lx, Ly = size
    nx, ny = max(int(round(Lx / resolution)), 4) + 1, max(int(round(Ly / resolution)), 4) + 1
    xs, ys = np.linspace(-Lx / 2, Lx / 2, nx), np.linspace(-Ly / 2, Ly / 2, ny)
    if breaks is not None:
        xs = merge_axis(xs, [-Lx / 2, Lx / 2, *breaks[0]], resolution)
        ys = merge_axis(ys, [-Ly / 2, Ly / 2, *breaks[1]], resolution)
        nx, ny = len(xs), len(ys)
    bx, by = rng.uniform(-0.25, 0.25, 2) * (Lx, Ly)
    dx, dy = rng.uniform(-0.25, 0.25, 2) * (Lx, Ly)
    sad = saddle * rng.uniform(0.8, 1.2)

    def h(x, y):
        s = sad * ((2 * x / Lx) ** 2 - (2 * y / Ly) ** 2)
        z = (
            height
            + s
            + bump * _bump(x, y, bx, by, Lx * 0.18, Ly * 0.2)
            - 0.6 * bump * _bump(x, y, dx, dy, Lx * 0.12, Ly * 0.15)
        )
        return z if relief is None else z + relief(x, y)

    X, Y = np.meshgrid(xs, ys)
    skip = None
    if perforate:
        skip = np.zeros((ny - 1, nx - 1), bool)
        skip[2 : ny - 3 : perforate, 2 : nx - 3 : perforate] = True
    sv, sf, ntop = _slab(xs, ys, h(X, Y), thickness, skip)
    parts = [(sv, sf)]
    # ribs: along x and along y, plus the perimeter plates; tops sit mid-plate
    ix = np.linspace(-Lx / 2 + frame_width, Lx / 2 - frame_width, max(int(Lx / rib_pitch), 2) + 1)
    iy = np.linspace(-Ly / 2 + frame_width, Ly / 2 - frame_width, max(int(Ly / rib_pitch), 2) + 1)
    sx = np.linspace(-Lx / 2 + frame_width, Lx / 2 - frame_width, nx)
    sy = np.linspace(-Ly / 2 + frame_width, Ly / 2 - frame_width, ny)
    rib_off = 0.5 * thickness
    if breaks is not None:  # sample ribs on the slab's own grid lines so creases line up
        sx = xs[(xs >= -Lx / 2 + frame_width) & (xs <= Lx / 2 - frame_width)]
        sy = ys[(ys >= -Ly / 2 + frame_width) & (ys <= Ly / 2 - frame_width)]
        rib_off = 0.7 * thickness
    for y0 in iy:
        parts.append(_prism(sx, y0, rib_thickness, 0.0, h(sx, y0) - rib_off, True))
    for x0 in ix:
        parts.append(_prism(sy, x0, rib_thickness, 0.0, h(x0, sy) - rib_off, False))
    # base frame: four rails under the perimeter
    fw, fh = frame_width, frame_height
    boxes = []
    for sx_, sy_, ex, ey in [
        (0, -1, Lx, fw),
        (0, 1, Lx, fw),
        (-1, 0, fw, Ly - 2 * fw),
        (1, 0, fw, Ly - 2 * fw),
    ]:
        pos = np.array([sx_ * (Lx - fw) / 2, sy_ * (Ly - fw) / 2, fh / 2])
        if sx_:
            pos[1] = 0.0
        if sy_:
            pos[0] = 0.0
        b = trimesh.creation.box((ex, ey, fh), trimesh.transformations.translation_matrix(pos))
        boxes.append((b.vertices, b.faces))
    parts += boxes
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for k in range(bosses):  # lifting bosses on the frame corners
        sx_, sy_ = corners[k % 4]
        cyl = trimesh.creation.cylinder(radius=0.025, height=0.04, sections=24)
        cyl.apply_translation([sx_ * (Lx / 2 - fw / 2 - 0.0), sy_ * (Ly / 2 - fw / 2), fh + 0.02])
        parts.append((cyl.vertices, cyl.faces))
    if plinth is not None:
        pw, under = plinth
        pad = trimesh.creation.box((pw, Ly + 0.2, fh))
        pad.apply_translation([Lx / 2 - under + pw / 2, 0.0, fh / 2])
        parts.append((pad.vertices, pad.faces))
    pin_xy = np.zeros((0, 2))
    if pins:
        pin_xy = rng.uniform(-0.3, 0.3, (pins, 2)) * (Lx, Ly)
        for px, py in pin_xy:
            cyl = trimesh.creation.cylinder(radius=0.012, height=0.04, sections=20)
            cyl.apply_translation([px, py, float(h(px, py)) - 0.01 + 0.02])
            parts.append((cyl.vertices, cyl.faces))
    mesh, _ = _assemble(parts)
    truth = np.zeros(len(mesh.faces), bool)
    cx_, cy_ = (xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2
    CX, CY = np.meshgrid(cx_, cy_)
    ok = np.ones(CX.shape, bool)
    for px, py in pin_xy:  # surface under a pin is not tool surface
        ok &= np.hypot(CX - px, CY - py) > 0.012 + 0.7 * resolution
    if skip is not None:
        ok = ok[~skip]
    truth[:ntop] = np.repeat(ok.ravel(), 2)
    _noise_and_tilt(mesh, noise, tilt, rng)
    info = _describe(
        mesh,
        truth,
        dict(
            kind="faceplate_eggcrate",
            pins=int(pins),
            pin_xy=pin_xy.tolist(),
            perforated=bool(perforate),
            plinth=plinth,
            seed=seed,
            size=tuple(size),
            resolution=resolution,
            thickness_m=thickness,
            margin_m=margin,
            part_area=(-Lx / 2 + margin, -Ly / 2 + margin, Lx / 2 - margin, Ly / 2 - margin),
            tilt_deg=tilt,
            noise_m=noise,
        ),
    )
    return mesh, truth, info


def thin_shell(
    kind: str = "male",
    flange: bool = True,
    size=(0.8, 0.6),
    resolution: float = 0.01,
    seed: int = 0,
    noise: float = 0.0,
    tilt: float = 0.0,
    thickness: float = 0.004,
    flange_width: float = 0.02,
    flange_angle_deg: float = 60.0,
    relief: float = 0.06,
    base_height: float = 0.15,
):
    """Thin shell (3-6 mm) following a smooth surface, with an optional angled flange lip.

    male: a dome-like bump with a dimple (convex + concave); the tool surface is the outer top.
    female: a trough with a raised rib inside (concave + convex); the tool surface is the inner,
    up-facing side. Truth excludes the flange (the lip is a trim return, not laid up on).
    """
    if kind not in ("male", "female"):
        raise ValueError(f"kind must be 'male' or 'female', got {kind!r}")
    rng = np.random.default_rng(seed)
    Lx, Ly = size
    nx, ny = max(int(round(Lx / resolution)), 4) + 1, max(int(round(Ly / resolution)), 4) + 1
    xs, ys = np.linspace(-Lx / 2, Lx / 2, nx), np.linspace(-Ly / 2, Ly / 2, ny)
    nf = 3 if flange else 0
    if flange:
        ext = flange_width * np.arange(1, nf + 1) / nf
        xs = np.concatenate([-Lx / 2 - ext[::-1], xs, Lx / 2 + ext])
        ys = np.concatenate([-Ly / 2 - ext[::-1], ys, Ly / 2 + ext])
    cx, cy = rng.uniform(-0.1, 0.1, 2)
    jx = rng.uniform(0.9, 1.1)

    def s(x, y):
        if kind == "male":
            dome = relief * _bump(x, y, cx, cy, Lx * 0.32 * jx, Ly * 0.34)
            dimple = -0.3 * relief * _bump(x, y, cx + 0.12 * Lx, cy - 0.1 * Ly, Lx * 0.07, Ly * 0.08)
            return base_height + dome + dimple
        trough = -relief * np.exp(-(((x - cx) / (Lx * 0.28 * jx)) ** 2))
        ridge = (
            0.25 * relief * np.exp(-(((x - cx - 0.08 * Lx) / 0.03) ** 2)) * np.exp(-((y / (Ly * 0.3)) ** 4))
        )
        return base_height + trough + ridge + 0.02 * (2 * y / Ly) ** 2

    X, Y = np.meshgrid(xs, ys)
    Xc, Yc = np.clip(X, -Lx / 2, Lx / 2), np.clip(Y, -Ly / 2, Ly / 2)
    d = np.hypot(X - Xc, Y - Yc)
    z = s(Xc, Yc) - d * np.tan(np.radians(flange_angle_deg))
    v, f, ntop = _slab(xs, ys, z, thickness)
    mesh = _outward(v, f)
    cell_x = (xs[:-1] + xs[1:]) / 2
    cell_y = (ys[:-1] + ys[1:]) / 2
    CX, CY = np.meshgrid(cell_x, cell_y)
    inner = ((np.abs(CX) < Lx / 2) & (np.abs(CY) < Ly / 2)).ravel()
    truth = np.zeros(len(mesh.faces), bool)
    truth[:ntop] = np.repeat(inner, 2)
    _noise_and_tilt(mesh, noise, tilt, rng)
    info = _describe(
        mesh,
        truth,
        dict(
            kind=f"thin_shell_{kind}",
            seed=seed,
            flange=flange,
            size=tuple(size),
            resolution=resolution,
            thickness_m=thickness,
            flange_width_m=flange_width if flange else 0.0,
            tilt_deg=tilt,
            noise_m=noise,
        ),
    )
    return mesh, truth, info


# ---------------------------------------------------------------- variations
def vary(
    mesh: trimesh.Trimesh,
    truth: np.ndarray,
    noise_m: float = 0.0,
    tilt_deg: float = 0.0,
    remesh: str | None = None,
    holes: int = 0,
    hole_radius: float = 0.008,
    seed: int = 0,
):
    """Return a varied copy ``(mesh, truth, info)`` with the truth mask kept aligned.

    remesh: None | "coarse" (quadric decimation to ~25%) | "fine" (one midpoint subdivision)
    | "irregular" (subdivide, then decimate to ~60% of the original count). Labels are
    transferred by proximity (nearest original face to each new face centroid). ``holes``
    removes small discs of faces at random spots on the tool surface.
    """
    rng = np.random.default_rng(seed)
    src = mesh
    m = mesh.copy()
    t = np.asarray(truth, bool).copy()
    if remesh:
        n0 = len(m.faces)
        if remesh == "fine":
            m = m.subdivide()
        elif remesh == "coarse":
            m = m.simplify_quadric_decimation(face_count=max(n0 // 4, 100))
        elif remesh == "irregular":
            m = m.subdivide().simplify_quadric_decimation(face_count=max(int(n0 * 0.6), 100))
        else:
            raise ValueError(f"remesh must be None, 'coarse', 'fine' or 'irregular', got {remesh!r}")
        m = trimesh.Trimesh(m.vertices, m.faces, process=False)
        _, _, tid = trimesh.proximity.closest_point(src, m.triangles_center)
        t = np.asarray(truth, bool)[tid]
    if holes:
        from scipy.spatial import cKDTree

        cen = m.triangles_center
        tree = cKDTree(cen)
        keep = np.ones(len(m.faces), bool)
        cand = np.flatnonzero(t)
        for p in rng.choice(cand, size=min(holes, len(cand)), replace=False):
            keep[tree.query_ball_point(cen[p], hole_radius)] = False
        m = trimesh.Trimesh(m.vertices, m.faces[keep], process=False)
        t = t[keep]
    _noise_and_tilt(m, noise_m, tilt_deg, rng)
    info = _describe(m, t, dict(remesh=remesh, holes=holes, noise_m=noise_m, tilt_deg=tilt_deg, seed=seed))
    return m, t, info


# The case set lives in core.mandrel_cases; these names are forwarded for older imports.
def __getattr__(name):
    if name in ("SYNTHETIC_CASES", "MandrelCase", "build_case", "iter_cases"):
        from core import mandrel_cases

        return getattr(mandrel_cases, name)
    raise AttributeError(name)
