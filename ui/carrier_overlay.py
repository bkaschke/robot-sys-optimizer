"""Floor overlays of the External axis tab (v9, issue #14): the standoff lines around the part
(minimum red, preferred amber), the **standoff range** of a contour plan as a filled band from
*Standoff from* to *Standoff up to* (blue), and the floor keep-outs (red boxes).

They follow the fields as they are typed (``preview``: the fields are read into a trial Base,
nothing is applied), and the committed base on every redraw of the panel (``draw``). Handles live
in ``panel._overlay`` ({"Standoff": [...], "Keep-outs": [...]}), so the left panel can show / hide
them as their own entries under External axis.
"""

from __future__ import annotations

import numpy as np
import trimesh
from shapely.geometry import Polygon

from core.standoff import Zones, footprint

ROOT = "/cell/carrier_overlay"
MIN_COLOR, PREF_COLOR = (220, 40, 40), (240, 170, 20)
BAND_COLOR, KEEPOUT_COLOR = (60, 130, 230), (230, 60, 60)
Z = 0.004  # just above the floor grid


def _ring_segments(ring: np.ndarray) -> np.ndarray:
    pts = np.column_stack([ring, np.full(len(ring), Z)])
    return np.stack([pts[:-1], pts[1:]], axis=1)


def band(part: Polygon, lo: float, hi: float) -> trimesh.Trimesh | None:
    """The floor area between the part outline grown by ``lo`` and by ``hi`` (m), triangulated."""
    if part is None or hi <= lo + 1e-6:
        return None
    area = part.buffer(hi).difference(part.buffer(max(lo, 0.0)))
    polys = list(area.geoms) if hasattr(area, "geoms") else [area]
    meshes = []
    for poly in polys:
        if poly.is_empty or poly.area < 1e-8:
            continue
        v, f = trimesh.creation.triangulate_polygon(poly, engine="earcut")
        meshes.append(trimesh.Trimesh(np.column_stack([v, np.full(len(v), Z - 0.001)]), f, process=False))
    return trimesh.util.concatenate(meshes) if meshes else None


def clear(panel) -> None:
    for hs in getattr(panel, "_overlay", {}).values():
        for h in hs:
            h.remove()
    panel._overlay = {"Standoff": [], "Keep-outs": []}


def draw(panel, base=None) -> Zones | None:
    """Redraw the overlays for ``base`` (default the session's); returns its standoff zones."""
    clear(panel)
    s, sc = panel.s, panel.server.scene
    b = base if base is not None else s.base
    zones = Zones(b, footprint(s.part_mesh(), s.part.T_world_part)) if s.part is not None else None
    out = panel._overlay
    if zones is not None and zones.part is not None:
        lo = b.fixed_standoff_m
        if lo is not None:  # contour plan: the stops keep a standoff in [lo, hi]
            hi = b.max_standoff_m if b.max_standoff_m is not None else lo
            m = band(zones.part, lo, hi)
            if m is not None:
                out["Standoff"].append(
                    sc.add_mesh_simple(
                        f"{ROOT}/standoff_band",
                        np.asarray(m.vertices, np.float32),
                        np.asarray(m.faces, np.uint32),
                        color=BAND_COLOR,
                        opacity=0.3,
                        side="double",
                    )
                )
            for name, d in (("from", lo), ("upto", hi)):
                ring = zones.shell(d) if d > 0 else None
                if ring is not None:
                    out["Standoff"].append(
                        sc.add_line_segments(
                            f"{ROOT}/standoff_{name}",
                            _ring_segments(ring),
                            BAND_COLOR,
                            thickness=2.0,
                            thickness_units="screen",
                        )
                    )
        for name, d, color in (("min", zones.min, MIN_COLOR), ("pref", zones.pref, PREF_COLOR)):
            ring = zones.shell(d) if d > 0 else None
            if ring is not None:
                out["Standoff"].append(
                    sc.add_line_segments(
                        f"{ROOT}/standoff_{name}",
                        _ring_segments(ring),
                        color,
                        thickness=2.0,
                        thickness_units="screen",
                    )
                )
    for k, ko in enumerate(b.keepouts):
        (x0, y0), (x1, y1) = ko.min_xy, ko.max_xy
        out["Keep-outs"].append(
            sc.add_box(
                f"{ROOT}/keepout{k}",
                color=KEEPOUT_COLOR,
                dimensions=(x1 - x0, y1 - y0, 0.02),
                position=((x0 + x1) / 2, (y0 + y1) / 2, 0.01),
                opacity=0.35,
            )
        )
    hook = getattr(panel, "on_overlay", None)
    if hook is not None:
        hook()  # e.g. the left panel re-applies its show / hide ticks
    return zones


def preview(panel) -> None:
    """Draw the overlays from the fields as typed (a trial Base; nothing is applied)."""
    if getattr(panel, "_busy", False):
        return
    try:
        base = panel.collect()
    except Exception:  # a half-typed field: keep the last drawing
        return
    try:
        draw(panel, base)
    except Exception:  # never break typing over a drawing problem
        pass


def wire(panel) -> None:
    """Live preview on the standoff / keep-out fields (and the mobile contour fields)."""
    widgets = [panel.so, panel.env, panel.keepouts]
    for name in ("fixed_so", "max_so"):
        if hasattr(panel, name):
            widgets.append(getattr(panel, name))
    widgets += list(getattr(panel, "side_boxes", {}).values())
    for w in widgets:
        w.on_update(lambda _e: preview(panel))
