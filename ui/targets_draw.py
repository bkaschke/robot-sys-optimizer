"""Scene drawing of the Targets tab (R1a): the face selection, the target points and normals,
the unreachable markers and (via ui.part_heatmap) the part heatmap.

Each part is redrawn only when what it shows changed:
- the selection is drawn as chunks (one mesh node per added batch of faces) under one frame
  node, so an add sends only the new faces; a removal rebuilds it as one chunk, and so does an
  add beyond ``MAX_CHUNKS`` chunks (bounds the node count);
- the point cloud and the normals are updated in place when the point count is unchanged
  (viser sends nothing for an unchanged array);
- the unreached markers and the heatmap are redrawn only when a digest of what they draw
  changed (the heatmap: mode, mesh, points and scores).
"""

from __future__ import annotations

import hashlib

import numpy as np

from core import targets as tg
from ui.display_lod import overlay_lift

PART = "/cell/part"
SEL_COLOR = (60, 200, 120)
SEL_OFFSET = 5e-4  # m: the selection is drawn just above the part
MAX_CHUNKS = 16
PALETTE = [(230, 60, 60), (40, 120, 230), (240, 170, 20), (150, 70, 200), (20, 170, 170), (120, 120, 120)]
STATUS_COLORS = {"reachable": (40, 180, 70), "marginal": (240, 170, 20), "unreachable": (220, 40, 40)}
UNCHECKED = (150, 150, 150)
NORMAL_COLOR = (30, 30, 30)


def digest(*parts) -> bytes:
    """A short hash of arrays and plain values (exact for arrays: dtype, shape and bytes)."""
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        if isinstance(p, np.ndarray):
            a = np.ascontiguousarray(p)
            h.update(f"{a.dtype}{a.shape}".encode())
            h.update(a.data)
        else:
            h.update(repr(p).encode())
        h.update(b"|")
    return h.digest()


def alive(h) -> bool:
    return h is not None and not h._impl.removed


# --- selection ----------------------------------------------------------------------------------


def chunk_arrays(m, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (lifted along the part's vertex normals) and faces of a selection chunk. The
    whole mesh's normals are used, so neighbouring chunks share their seam vertices exactly."""
    used, inv = np.unique(m.faces[faces], return_inverse=True)
    V = m.vertices[used] + overlay_lift(m, SEL_OFFSET) * m.vertex_normals[used]
    return V.astype(np.float32), inv.reshape(-1, 3).astype(np.uint32)


class SelectionLayer:
    """The selected faces as mesh chunks under the frame ``{PART}/selection`` (the frame is the
    handle the Objects panel shows/hides; hiding it hides every chunk)."""

    def __init__(self, server, root: str = f"{PART}/selection"):
        self.server, self.root = server, root
        self.frame = None
        self.chunks: list[tuple[np.ndarray, object]] = []  # (face indices, mesh handle)
        self.drawn = np.zeros(0, int)
        self._mesh = None
        self._n = 0

    def _alive(self) -> bool:
        return alive(self.frame) and all(alive(h) for _, h in self.chunks)

    def _add_chunk(self, m, faces: np.ndarray) -> None:
        V, F = chunk_arrays(m, faces)
        h = self.server.scene.add_mesh_simple(
            f"{self.root}/c{self._n}", V, F, color=SEL_COLOR, opacity=0.6, side="double"
        )
        self._n += 1
        self.chunks.append((faces, h))

    def update(self, m, faces: np.ndarray):
        """Show ``faces`` of mesh ``m`` (None = nothing); returns the frame handle."""
        faces = np.unique(np.asarray(faces, int)) if m is not None else np.zeros(0, int)
        if m is self._mesh and self._alive():
            if np.array_equal(faces, self.drawn):
                return self.frame
            added = np.setdiff1d(faces, self.drawn, assume_unique=True)
            if len(added) + len(self.drawn) == len(faces) and len(self.chunks) < MAX_CHUNKS:
                self._add_chunk(m, added)  # only an add: send just the new faces
                self.drawn = faces
                return self.frame
        self._rebuild(m, faces)
        return self.frame

    def _rebuild(self, m, faces: np.ndarray) -> None:
        for _, h in self.chunks:
            if alive(h):
                h.remove()
        self.chunks = []
        if not alive(self.frame):
            self.frame = self.server.scene.add_frame(self.root, show_axes=False)
        self._mesh, self.drawn = m, faces
        if m is not None and len(faces):
            self._add_chunk(m, faces)


# --- points -------------------------------------------------------------------------------------


def _lookup(table, idx) -> np.ndarray:
    return np.asarray(table, np.uint8)[np.asarray(idx, int)]


def point_colors(color_by: str, ids, G: list[str], names: list[str], feas: dict, sel: dict) -> np.ndarray:
    """(n, 3) uint8 colour per point: by feasibility status, zone / configuration of the selected
    plan, or group (the colour table is indexed once, with numpy)."""
    if color_by == "feasibility" and feas:
        table = [UNCHECKED, *STATUS_COLORS.values()]
        code = {k: j + 1 for j, k in enumerate(STATUS_COLORS)}
        idx = [code.get(feas[i].status, 0) if i in feas else 0 for i in ids]
        return _lookup(table, idx).reshape(-1, 3)
    if color_by in ("zone", "configuration") and sel:
        labels = sorted({r.config for r in sel.values() if r.config})
        rank = {c: j for j, c in enumerate(labels)}
        key = (lambda r: r.zone) if color_by == "zone" else (lambda r: rank[r.config])
        n = len(PALETTE)
        idx = [
            key(r) % n if (r := sel.get(i)) is not None and r.config and r.zone is not None else n
            for i in ids
        ]
        return _lookup([*PALETTE, UNCHECKED], idx).reshape(-1, 3)
    where = {g: j % len(PALETTE) for j, g in reversed(list(enumerate(names)))}  # first match wins
    return _lookup(PALETTE, [where.get(g, len(PALETTE) - 1) for g in G]).reshape(-1, 3)


def _put(tab, key: str, n: int, update: dict, make) -> None:
    """Update handle ``key`` in place when it is alive and has ``n`` items, else re-create it."""
    h = tab._handles.get(key)
    if alive(h) and len(h.points) == n:
        for name, value in update.items():
            setattr(h, name, value)
        return
    tab._drop(key)
    tab._handles[key] = make()


EXTRAS = ("edge_gaps", "unfit")  # handle-key prefixes of ui.unreached_view.draw_extras (v10 T10, T12)


def _drop_unreached(tab) -> None:
    for k in [k for k in tab._handles if k.startswith(("unreached", *EXTRAS))]:
        tab._drop(k)
    tab._fp.pop("unreached", None)


def _draw_extras(tab, ids, P, N) -> None:
    """Edge-check gaps and points the tool cannot fit (few markers: redrawn every time)."""
    from ui.unreached_view import draw_extras

    for k in [k for k in tab._handles if k.startswith(EXTRAS)]:
        tab._drop(k)
    tab._handles.update(draw_extras(tab, PART, ids, P, N))


def heat_key(tab) -> tuple:
    """What the heatmap shows: the mode, the part mesh, the points and the per-point scores."""
    from ui.part_heatmap import SCORES, point_scores

    heat, s = tab.heat, tab.s
    what = heat.choice.value
    key: tuple = (what, s.part is None)
    if SCORES.get(what) is not None and s.part is not None:
        sel = tab.selected_results() if tab.selected_results else {}
        scores = point_scores(s, what, sel)
        ids, P, _, _ = tg.as_arrays(s.targets)
        sc = np.array(sorted(scores.items()), float).reshape(-1, 2)
        key += (s.part.mesh_path, s.part.units, digest(ids, P, sc))
    return key


def track_heat(tab) -> None:
    """Record what every heatmap draw shows (also the draws its own dropdown and the tolerance
    panel start), so ``draw_heat`` can skip a redraw of the same picture. Wraps the layer's
    ``draw`` on the instance; ui.part_heatmap is unchanged."""
    heat = tab.heat
    draw = heat.draw

    def tracked() -> None:
        key = heat_key(tab)
        draw()
        tab._fp["heat"] = (key, heat._handles)

    heat.draw = tracked


def draw_heat(tab) -> None:
    """Redraw the heatmap only when ``heat_key`` changed. A clear from elsewhere replaces the
    layer's handle list, which forces the next redraw here."""
    heat = tab.heat
    last = tab._fp.get("heat")
    if last is not None and last[1] is heat._handles and all(alive(h) for h in last[1]):
        if last[0] == heat_key(tab):
            return
    heat.draw()


def draw_points(tab) -> None:
    """The target points, normals and unreachable markers of the tab (only what changed)."""
    from ui.unreached_view import draw_unreached, unreached_ids, unreached_key

    s = tab.s
    draw_heat(tab)  # v8 T8: the part's colours follow the selected plan too
    ids, P, N, G = tg.as_arrays(s.targets)
    names = [g.name for g in s.targets.groups]
    counts = {g: G.count(g) for g in names}
    tab.count.content = f"**{len(P)} points**  \n" + ", ".join(f"{k}: {v}" for k, v in counts.items())
    tab.preview_idx.max = max(1, len(P) - 1)
    if len(P) == 0:
        tab._drop("points")
        tab._drop("normals")
        return _drop_unreached(tab)
    sel = tab.selected_results() if tab.selected_results else {}
    colors = point_colors(tab.color_by.value, ids, G, names, s.feasibility or {}, sel)
    P32, size = P.astype(np.float32), tab.point_size.value / 1000
    _put(
        tab,
        "points",
        len(P),
        {"points": P32, "colors": colors, "point_size": size},
        lambda: tab.server.scene.add_point_cloud(
            f"{PART}/targets", P32, colors, point_size=size, point_shape="circle", precision="float32"
        ),
    )
    if tab.show_normals.value:
        seg = np.stack([P, P + N * tab.normal_len.value / 1000], axis=1).astype(np.float32)
        _put(
            tab,
            "normals",
            len(P),
            {"points": seg},
            lambda: tab.server.scene.add_line_segments(
                f"{PART}/normals", seg, NORMAL_COLOR, thickness=1.5, thickness_units="screen"
            ),
        )
    else:
        tab._drop("normals")
    if not tab.show_unreached.value:  # v6 feedback: make the points the plan misses obvious
        return _drop_unreached(tab)
    from ui.staleness import feasibility_changes

    # an out-of-date feasibility check (points, tool, part or base changed since) would
    # mark the wrong points: only the selected plan's results are used then
    feas = s.feasibility if s.feasibility and not feasibility_changes(s) else {}
    _draw_extras(tab, ids, P, N)
    bad = unreached_ids(ids, sel, feas)
    key = unreached_key(ids, P, N, bad, size, sel or feas)
    old = [h for k, h in tab._handles.items() if k.startswith("unreached")]
    if tab._fp.get("unreached") == key and all(alive(h) for h in old):
        return
    for k in [k for k in tab._handles if k.startswith("unreached")]:
        tab._drop(k)
    tab._handles.update(draw_unreached(tab.server, PART, ids, P, N, bad, size, sel or feas))
    tab._fp["unreached"] = key
