"""Targets -> Display -> Colour part by (v8 T8): the part surface painted by a per-point score
of the selected plan (or the feasibility check), red = poor to green = good, with a legend.

Each face takes the score of the nearest target within 1.5 x the point spacing of its centre
(the station-area rule, ui.station_areas; coarse meshes are subdivided first). Scores are
binned into ``BINS`` colours, one mesh per bin, drawn just above the surface under the part.
"""

from __future__ import annotations

import math

import numpy as np

from core import targets as tg

PATH = "/cell/part/heatmap"
BINS = 10
OFFSET = 1.1e-3  # m above the surface (station areas are at 0.8 mm)
OFF = "Reached / unreached (points)"  # no surface colour: the points' own colours (pre-v8)

# label -> (unit, lo, hi): scores are clipped to [lo, hi] for the colours
SCORES = {
    OFF: None,
    "Clearance": ("mm", 0.0, 50.0),
    "Manipulability": ("% of max", 0.0, 100.0),
    "Joint-limit margin": ("°", 0.0, 30.0),
    "Tolerance reach rate": ("%", 0.0, 100.0),
}


def colour(f: float) -> tuple[int, int, int]:
    """0 -> red, 0.5 -> yellow, 1 -> green."""
    f = min(1.0, max(0.0, float(f)))
    if f < 0.5:
        return (220, int(40 + 2 * f * 180), 40)
    return (int(220 - 2 * (f - 0.5) * 180), 220, 40)


def point_scores(s, what: str, sel: dict) -> dict[int, float]:
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
    if what == "Tolerance reach rate":
        tol = getattr(s, "tolerance", None)
        return {int(p): 100 * t.rate for p, t in tol.points.items()} if tol is not None else {}
    return {}


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
        self._handles: list = []

    def clear(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self.legend.content = ""

    def draw(self) -> None:
        self.clear()
        what = self.choice.value
        s = self.tab.s
        spec = SCORES.get(what)
        if spec is None or s.part is None:
            return
        unit, lo, hi = spec
        sel = self.tab.selected_results() if self.tab.selected_results else {}
        scores = point_scores(s, what, sel)
        if not scores:
            tol = what == "Tolerance reach rate"
            src = "a tolerance check (Results)" if tol else "a plan or a feasibility check"
            self.legend.content = f"_Nothing to show yet: run {src}._"
            return
        from ui.station_areas import display_mesh, face_labels

        ids, P, _, _ = tg.as_arrays(s.targets)
        mesh = display_mesh(s.part_mesh(), P)
        val = np.array([scores.get(int(i), np.nan) for i in ids])
        lab = face_labels(mesh, P, np.where(np.isfinite(val), 0, -1))
        painted = lab >= 0
        self.face_bins = fb = np.full(len(mesh.faces), -1)
        fb[painted] = bins(val[lab[painted]], lo, hi)
        for b in range(BINS):
            faces = np.flatnonzero(fb == b)
            if len(faces) == 0:
                continue
            sub = mesh.submesh([faces], append=True)
            v = sub.vertices + sub.vertex_normals * OFFSET
            self._handles.append(
                self.tab.server.scene.add_mesh_simple(
                    f"{PATH}/b{b}",
                    v,
                    sub.faces,
                    color=colour((b + 0.5) / BINS),
                    side="double",
                    flat_shading=False,
                )
            )
        vals = np.array(list(scores.values()))
        self.legend.content = (
            f"🟥 {lo:g} {unit} or less → 🟨 → 🟩 {hi:g} {unit} or more  \n"
            f"_{len(scores)} points: lowest {np.min(vals):.0f}, median {np.median(vals):.0f} {unit}._"
        )
