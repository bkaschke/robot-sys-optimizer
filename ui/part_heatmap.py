"""Targets -> Display -> Colour part by (v8 T8): the part surface painted by a per-point score
of the selected plan (or the feasibility check), red = poor to green = good, with a legend.

Each face takes the score of the nearest target within 1.5 x the point spacing of its centre
(the station-area rule, ui.station_areas; coarse meshes are subdivided first). Scores are
binned into ``BINS`` colours and drawn as ONE vertex-coloured mesh (R1b, ui.colour_mesh) just
above the surface under the part. The last ``KEEP`` pictures stay in the scene hidden, so
switching back to a mode or plan shown before only toggles visibility (no re-upload).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from core import targets as tg

PATH = "/cell/part/heatmap"
BINS = 10
OFFSET = 1.1e-3  # m above the surface (station areas are at 0.8 mm)
KEEP = 4  # cached heatmap pictures (e.g. 2 plans x 2 modes)
OFF = "Reached / unreached (points)"
EDGE = "Edge check"
FIT = "Surface radius (tool fit)"  # no surface colour: the points' own colours (pre-v8)

# label -> (unit, lo, hi): scores are clipped to [lo, hi] for the colours
SCORES = {
    OFF: None,
    "Clearance": ("mm", 0.0, 50.0),
    "Manipulability": ("% of max", 0.0, 100.0),
    "Joint-limit margin": ("°", 0.0, 30.0),
    "Tolerance reach rate": ("%", 0.0, 100.0),
    EDGE: ("%", 0.0, 100.0),  # v10 T10: 0 = gap, 100 = reached
    FIT: ("% of the tool's minimum radius", 0.0, 200.0),  # v10 T12: red below the minimum
}


def colour(f: float) -> tuple[int, int, int]:
    """0 -> red, 0.5 -> yellow, 1 -> green."""
    f = min(1.0, max(0.0, float(f)))
    if f < 0.5:
        return (220, int(40 + 2 * f * 180), 40)
    return (int(220 - 2 * (f - 0.5) * 180), 220, 40)


def tolerance_for(s, sol, tol=None):
    """The session's tolerance result if it belongs to ``sol`` (a plan, or None = the result's own
    label) *as it is now*: same label and the plan stamp recorded when it was computed equals
    ``s.stamps[label]`` (review R8). No recorded stamp = unknown = not shown."""
    tol = getattr(s, "tolerance", None) if tol is None else tol
    if tol is None:
        return None
    own = getattr(tol, "key", "") or tol.label  # v12 #8: plans are keyed by id (label before v12)
    k = own if sol is None else (getattr(sol, "id", None) or sol.label)
    if own != k or not hasattr(tol, "plan_stamp"):
        return None
    return tol if (tol.plan_stamp or {}) == (s.stamps.get(k) or {}) else None


def point_scores(s, what: str, sel: dict, label: str | None = None) -> dict[int, float]:
    """{point id: score} for ``what`` from the selected plan's results (else the feasibility
    check); the tolerance rate from the session's last tolerance check."""
    res = sel or s.feasibility or {}
    ok = {int(p): r for p, r in res.items() if r.status != "unreachable" and r.q is not None}
    if what == "Clearance":
        return {p: (r.clearance * 1000 if math.isfinite(r.clearance) else 1e9) for p, r in ok.items()}
    if what == "Joint-limit margin":
        return {p: float(r.limit_margin_deg) for p, r in ok.items()}
    if what == "Manipulability":
        top = getattr(s, "max_manip", None) or max((r.manipulability for r in ok.values()), default=0.0)
        top = top or 1.0
        return {p: 100 * float(r.manipulability) / top for p, r in ok.items()}
    if what == FIT:
        return fit_scores(s)
    if what == "Tolerance reach rate":
        tol = getattr(s, "tolerance", None)
        if tol is not None:
            tol = tolerance_for(s, None if label is None else SimpleNamespace(label=label), tol)
        return {int(p): 100 * t.rate for p, t in tol.points.items()} if tol is not None else {}
    return {}


def fit_scores(s) -> dict[int, float]:
    """v10 T12: each point's surface radius (the curvature its limit applies to) as % of the
    tool's minimum radius (the first TCP with one); flat = far above."""
    from core.surface_radius import limit_radius
    from ui.radius_ui import fit_of

    t = next((t for t in (s.tool.tcps if s.tool else []) if t.min_radius_mm), None)
    if t is None:
        return {}
    out = {}
    for pid, rad in fit_of(s).radii.items():
        r, _ = limit_radius(rad, t.radius_applies)
        out[pid] = 100 * r / (t.min_radius_mm / 1000) if math.isfinite(r) else 1e9
    return out


def bins(scores: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Bin index 0..BINS-1 per score (clipped to [lo, hi]); monotonic in the score."""
    f = (np.clip(scores, lo, hi) - lo) / max(hi - lo, 1e-12)
    return np.minimum((f * BINS).astype(int), BINS - 1)


class HeatmapLayer:
    """The dropdown + legend (built inside the Display folder) and the coloured meshes."""

    def __init__(self, gui, tab):
        self.tab = tab
        self.choice = gui.add_dropdown("Colour part by", tuple(SCORES), hint="the selected plan (Results)")
        self.legend = gui.add_markdown("")
        self.choice.on_update(lambda _e: self.draw())
        self._handles: list = []  # the shown node (ui.targets_draw checks this list's identity)
        from ui.colour_mesh import NodeCache

        self._cache = NodeCache(KEEP)
        self.cones = gui.add_checkbox(  # v8 T15
            "Show approach cones",
            False,
            hint="directions a straight tool of the current length can come from",
        )
        self.cones_note = gui.add_markdown("")
        self.cones.on_update(lambda _e: self.draw_cones())
        self._cone_handle = None

    def draw_cones(self) -> None:
        """Short lines from each target along its free approach directions (opt.approach_cones):
        no lines = no straight tool reaches that point."""
        if self._cone_handle is not None:
            self._cone_handle.remove()
            self._cone_handle = None
        self.cones_note.content = ""
        s = self.tab.s
        if not self.cones.value or s.part is None or not s.targets.points:
            return
        from opt.approach_cones import free_directions

        ids, P, N, _ = tg.as_arrays(s.targets)
        L = float(s.tool_model.length()) if s.tool_model is not None else 0.3
        r = float(s.tool.extension.radius) if s.tool is not None else 0.03
        free = free_directions(s.part_mesh(), P, N, length=max(0.15, L), radius=r, count=40)
        seg = [np.stack([p, p + u * 0.06]) for p, U in zip(P, free, strict=True) for u in U[::2]]
        if seg:
            self._cone_handle = self.tab.server.scene.add_line_segments(
                f"{PATH}_cones", np.array(seg), colors=(60, 140, 230), line_width=1.5
            )
        blocked = [int(i) for i, U in zip(ids, free, strict=True) if len(U) == 0]
        self.cones_note.content = f"_Tool {1000 * L:.0f} mm, radius {1000 * r:.0f} mm: " + (
            f"**no free direction** at point(s) {', '.join(map(str, blocked[:20]))}._"
            if blocked
            else "every point has one._"
        )

    def _draw_edges(self, unit: str, lo: float, hi: float) -> bool:
        """v10 T10: faces near the edge check's points, red at gaps, green where reached."""
        get = getattr(self.tab, "edge_result", None)
        r = get() if get is not None else None
        if r is None:
            self.legend.content = "_Nothing to show yet: run Results → Edge check on the selected plan._"
            return False
        from ui.station_areas import display_mesh

        Pe = np.array([e.p for e in r.points])
        mesh = display_mesh(self.tab.s.part_mesh(), Pe)
        d, j = cKDTree(Pe).query(mesh.triangles_center, distance_upper_bound=0.5 * r.spacing)
        near = np.isfinite(d)
        val = np.array([100.0 if e.reached else 0.0 for e in r.points])
        fb = np.full(len(mesh.faces), -1)
        fb[near] = bins(val[j[near]], lo, hi)
        shown = self._paint(mesh, fb)
        self.legend.content = f"🟥 gap → 🟩 reached  \n_{r.summary()}._"
        return shown

    def _paint(self, mesh, fb: np.ndarray) -> bool:
        """The faces with a bin (>= 0) as ONE vertex-coloured GLB (cached by picture)."""
        from ui.colour_mesh import show_patch

        self.face_bins = fb
        faces = np.flatnonzero(fb >= 0)
        if len(faces):
            table = np.array([colour((b + 0.5) / BINS) for b in range(BINS)], np.uint8)
            scene = self.tab.server.scene
            self._handles = [show_patch(scene, self._cache, PATH, mesh, faces, table[fb[faces]], OFFSET)]
        return bool(len(faces))

    def clear(self) -> None:
        """Hide the heatmap (its node stays cached, hidden) and empty the legend."""
        self._cache.hide()
        self._handles = []
        self.legend.content = ""

    def draw(self) -> None:
        self._handles = []  # a new list: a redraw is visible to ui.targets_draw
        self.legend.content = ""
        if not self._draw():
            self._cache.hide()

    def _draw(self) -> bool:
        """Show the picture for the current mode and plan; False = nothing to show."""
        what = self.choice.value
        s = self.tab.s
        spec = SCORES.get(what)
        if spec is None or s.part is None:
            return False
        unit, lo, hi = spec
        sel = self.tab.selected_results() if self.tab.selected_results else {}
        if what == EDGE:
            return self._draw_edges(unit, lo, hi)
        scores = point_scores(s, what, sel)
        if not scores:
            tol = what == "Tolerance reach rate"
            src = "a tolerance check (Results)" if tol else "a plan or a feasibility check"
            self.legend.content = f"_Nothing to show yet: run {src}._"
            return False
        from ui.station_areas import display_mesh, face_labels

        ids, P, _, _ = tg.as_arrays(s.targets)
        mesh = display_mesh(s.part_mesh(), P)
        val = np.array([scores.get(int(i), np.nan) for i in ids])
        lab = face_labels(mesh, P, np.where(np.isfinite(val), 0, -1))
        painted = lab >= 0
        fb = np.full(len(mesh.faces), -1)
        fb[painted] = bins(val[lab[painted]], lo, hi)
        shown = self._paint(mesh, fb)
        vals = np.array(list(scores.values()))
        self.legend.content = (
            f"🟥 {lo:g} {unit} or less → 🟨 → 🟩 {hi:g} {unit} or more  \n"
            f"_{len(scores)} points: lowest {np.min(vals):.0f}, median {np.median(vals):.0f} {unit}._"
        )
        return shown
