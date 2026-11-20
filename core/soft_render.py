"""Offline software renderer for part pictures (no GPU, no browser, no matplotlib, no viser).

``render(mesh, highlight=...)`` returns PNG bytes. A numpy z-buffer rasteriser: orthographic
camera fitted to the mesh bounds, Lambert shading from a fixed light plus ambient, double-sided
(back faces are darker), per-face colours, a highlight mask in the tool-surface magenta and an
optional second mask. Rendered at 2x and averaged down (anti-aliasing). Deterministic.

Triangles are bucketed by screen size. Each bucket is rasterised for all its triangles at once,
one pixel offset of the bounding box at a time (so a 500k-face part costs a few passes, not a
Python loop per triangle); triangles of two pixels or less also splat their centre so thin or
tiny faces leave no holes. Only triangles larger than ``BIG_PX`` use a per-triangle loop.
"""

from __future__ import annotations

import io

import numpy as np

SIZE = (480, 360)
BASE = (176, 182, 192)  # neutral grey
ACCENT = (230, 70, 200)  # ui/tool_surface_ui.PREVIEW_COLOR (the tool-surface magenta)
SECOND = (240, 160, 30)  # secondary candidates
SS = 2  # supersampling
BIG_PX = 32  # larger triangles are drawn one by one
BACK = 0.55  # shade factor of back faces
VIEWS = ("iso", "top", "front", "side")
_CHUNK = 1_500_000  # candidate pixels per pass (bounds memory)
_CLASSES = ((0, 2), (2, 4), (4, 8), (8, 16), (16, BIG_PX))


def _basis(view: str, up):
    """(toward-camera c, screen right r, screen up u) for a named view."""
    up = np.asarray(up, float)
    up = up / np.linalg.norm(up)
    ref = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = ref - up * (ref @ up)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    if view == "iso":
        c, hint = e1 - e2 + 0.9 * up, up
    elif view == "top":
        c, hint = up, e2
    elif view == "front":
        c, hint = -e2, up
    elif view == "side":
        c, hint = e1, up
    else:
        raise ValueError(f"view must be one of {VIEWS}, got {view!r}")
    c = c / np.linalg.norm(c)
    u = hint - c * (hint @ c)
    u /= np.linalg.norm(u)
    return c, np.cross(u, c), u


def _mask(sel, n: int) -> np.ndarray | None:
    if sel is None:
        return None
    a = np.asarray(sel)
    if a.dtype == bool:
        if len(a) != n:
            raise ValueError(f"highlight mask has {len(a)} entries, the mesh has {n} faces")
        return a
    m = np.zeros(n, bool)
    if a.size:
        m[a.astype(int)] = True
    return m


def _fragments(sx, sy, z, sel, size, w, h, eps):
    """Pixel fragments (pixel index, depth, face index) of triangles ``sel`` whose bounding
    boxes are at most ``size`` pixels wide and high."""
    ax, bx, cx = sx[sel, 0], sx[sel, 1], sx[sel, 2]
    ay, by, cy = sy[sel, 0], sy[sel, 1], sy[sel, 2]
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    x0 = np.floor(np.minimum(np.minimum(ax, bx), cx)).astype(np.int64)
    y0 = np.floor(np.minimum(np.minimum(ay, by), cy)).astype(np.int64)
    za, zb, zc = z[sel, 0], z[sel, 1], z[sel, 2]
    out = []
    for dy in range(size):
        for dx in range(size):
            px, py = x0 + dx, y0 + dy
            ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            qx, qy = px + 0.5, py + 0.5
            l0 = ((bx - qx) * (cy - qy) - (by - qy) * (cx - qx)) / area
            l1 = ((cx - qx) * (ay - qy) - (cy - qy) * (ax - qx)) / area
            ok &= (l0 >= -eps) & (l1 >= -eps) & (1 - l0 - l1 >= -eps)
            i = np.flatnonzero(ok)
            if len(i):
                zz = l0[i] * za[i] + l1[i] * zb[i] + (1 - l0[i] - l1[i]) * zc[i]
                out.append((py[i] * w + px[i], zz, sel[i]))
    return out


def _put(zbuf, fid, pix, zz, face) -> None:
    """Depth test + write; sorted by depth so that the nearest fragment of a pixel is last."""
    keep = zz > zbuf[pix]
    pix, zz, face = pix[keep], zz[keep], face[keep]
    o = np.argsort(zz, kind="stable")
    zbuf[pix[o]] = zz[o]
    fid[pix[o]] = face[o]


def _raster(sx, sy, z, w, h):
    """Face-id image (h, w), -1 = background; larger depth = nearer."""
    zbuf = np.full(w * h, -np.inf)
    fid = np.full(w * h, -1, np.int32)
    ax, ay = sx[:, 1] - sx[:, 0], sy[:, 1] - sy[:, 0]
    bx, by = sx[:, 2] - sx[:, 0], sy[:, 2] - sy[:, 0]
    live = np.abs(ax * by - ay * bx) > 1e-12
    on = (sx.max(1) >= 0) & (sx.min(1) < w) & (sy.max(1) >= 0) & (sy.min(1) < h)
    dim = np.maximum(np.ceil(sx.max(1)) - np.floor(sx.min(1)), np.ceil(sy.max(1)) - np.floor(sy.min(1)))
    ok = live & on
    for lo, size in _CLASSES:
        sel = np.flatnonzero(ok & (dim > lo) & (dim <= size))
        if size == 2 and len(sel):  # tiny: a centre splat and generous coverage, so nothing vanishes
            cx = np.clip(np.floor(sx[sel].mean(1)), 0, w - 1).astype(np.int64)
            cy = np.clip(np.floor(sy[sel].mean(1)), 0, h - 1).astype(np.int64)
            _put(zbuf, fid, cy * w + cx, z[sel].mean(1), sel)
        step = max(1, _CHUNK // (size * size))
        for i in range(0, len(sel), step):
            parts = _fragments(sx, sy, z, sel[i : i + step], size, w, h, 0.3 if size == 2 else 1e-9)
            if parts:
                _put(zbuf, fid, *(np.concatenate(p) for p in zip(*parts, strict=True)))
    for f in np.flatnonzero(ok & (dim > BIG_PX)):  # large triangles: bounding box per triangle
        x0, x1 = max(int(np.floor(sx[f].min())), 0), min(int(np.ceil(sx[f].max())), w - 1)
        y0, y1 = max(int(np.floor(sy[f].min())), 0), min(int(np.ceil(sy[f].max())), h - 1)
        ix, iy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        gx, gy = ix + 0.5, iy + 0.5
        (xa, xb, xc), (ya, yb, yc) = sx[f], sy[f]
        area = (xb - xa) * (yc - ya) - (yb - ya) * (xc - xa)
        l0 = ((xb - gx) * (yc - gy) - (yb - gy) * (xc - gx)) / area
        l1 = ((xc - gx) * (ya - gy) - (yc - gy) * (xa - gx)) / area
        ins = (l0 >= 0) & (l1 >= 0) & (l0 + l1 <= 1)
        zz = l0 * z[f, 0] + l1 * z[f, 1] + (1 - l0 - l1) * z[f, 2]
        _put(zbuf, fid, (iy * w + ix)[ins], zz[ins], np.full(int(ins.sum()), f))
    return fid.reshape(h, w)


def render(
    mesh,
    face_colors=None,
    highlight=None,
    view: str = "iso",
    size=SIZE,
    up=(0, 0, 1),
    background=(255, 255, 255),
    secondary=None,
) -> bytes:
    """PNG bytes of ``mesh`` (trimesh-like: ``vertices``, ``faces``) seen from ``view``.

    ``highlight`` / ``secondary``: bool mask over the faces or an array of face ids, drawn in the
    accent / second colour. ``face_colors``: (n, 3|4) uint8 base colours (default neutral grey).
    """
    from PIL import Image

    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, np.int64)
    W, H = int(size[0]), int(size[1])
    if len(F) == 0 or W < 8 or H < 8:
        raise ValueError("nothing to render: the mesh has no faces or the picture is too small")
    c, r, u = _basis(view, up)
    px, py, pz = V @ r, V @ u, V @ c
    lo, hi = np.array([px.min(), py.min()]), np.array([px.max(), py.max()])
    scale = 0.9 * min(W / max(hi[0] - lo[0], 1e-12), H / max(hi[1] - lo[1], 1e-12))
    ctr = (lo + hi) / 2
    w2, h2 = W * SS, H * SS
    X = ((px - ctr[0]) * scale + W / 2) * SS
    Y = (H / 2 - (py - ctr[1]) * scale) * SS
    fid = _raster(X[F], Y[F], pz[F], w2, h2)
    P = V[F]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    facing = n @ c
    n = np.where((facing < 0)[:, None], -n, n)
    light = 0.8 * c - 0.4 * r + 0.5 * u
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(n @ light, 0, 1)
    shade = np.where(facing < 0, shade * BACK, shade)
    col = np.tile(np.asarray(BASE, np.float32), (len(F), 1))
    if face_colors is not None:
        fc = np.asarray(face_colors)
        if len(fc) != len(F):
            raise ValueError(f"face_colors has {len(fc)} rows, the mesh has {len(F)} faces")
        col = fc[:, :3].astype(np.float32)
    for sel, rgb in ((_mask(secondary, len(F)), SECOND), (_mask(highlight, len(F)), ACCENT)):
        if sel is not None:
            col[sel] = rgb
    col = col * shade[:, None].astype(np.float32)
    img = np.empty((h2, w2, 3), np.float32)
    img[:] = np.asarray(background, np.float32)
    hit = fid >= 0
    img[hit] = col[fid[hit]]
    img = img.reshape(H, SS, W, SS, 3).mean((1, 3))
    buf = io.BytesIO()
    Image.fromarray(np.clip(img + 0.5, 0, 255).astype(np.uint8), "RGB").save(buf, "PNG", compress_level=6)
    return buf.getvalue()
