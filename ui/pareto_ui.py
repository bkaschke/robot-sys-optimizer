"""Pareto fronts in the GUI (v8 T11): the measure checkboxes of the planners and the Results ->
Pareto front scatter (two chosen measures, one numbered dot per plan; the numbers are the
"Pareto i/n" labels in the Results list).

Clicking a dot selects that plan. viser 1.1.1 reports no clicks on images or plots, so the
chart is an SVG in an HTML element: each dot's onclick presses the matching button of the
*Select plan* button group under it (a normal viser button, so the click reaches the server
the usual way and the buttons also work by themselves).
"""

from __future__ import annotations

from collections.abc import Callable
from html import escape

from opt.pareto import OBJECTIVES, measures

DOT = "#3d6fc4"
PICKED = "#e8762c"
W, H, PAD_L, PAD_R, PAD_T, PAD_B = 320, 230, 52, 14, 12, 40


def pareto_keys_widgets(gui, default: tuple) -> dict:
    """One checkbox per measure (hidden until the objective is Pareto front)."""
    return {
        k: gui.add_checkbox(f"Pareto: {label}", k in default, visible=False)
        for k, (label, _sense) in OBJECTIVES.items()
    }


def button_label(n: int) -> str:
    return f"Plan {n}"


def _ticks(lo: float, hi: float) -> list[float]:
    import numpy as np

    if hi <= lo:
        return [lo]
    if float(lo).is_integer() and float(hi).is_integer() and hi - lo <= 6:  # e.g. stations
        return [float(v) for v in range(int(lo), int(hi) + 1)]
    return [float(v) for v in np.linspace(lo, hi, 4)]


def _fmt(v: float) -> str:
    a = abs(v)
    return f"{v:.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.2f}"


def scatter_svg(
    points: list[tuple[int, float, float]], xlabel: str, ylabel: str, picked: int | None = None
) -> str:
    """SVG scatter of (plan number, x, y). Text uses currentColor (light and dark panels).
    A dot's onclick climbs to the nearest element holding the button *Plan n* and clicks it.
    Points with a missing measure (inf, e.g. no cycle estimate) are left out."""
    import math

    points = [p for p in points if math.isfinite(p[1]) and math.isfinite(p[2])]
    if not points:
        return "<p><i>No plan has both measures.</i></p>"
    xs, ys = [p[1] for p in points], [p[2] for p in points]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dx, dy = (x1 - x0) or abs(x0) or 1.0, (y1 - y0) or abs(y0) or 1.0
    x0, x1, y0, y1 = x0 - 0.08 * dx, x1 + 0.08 * dx, y0 - 0.1 * dy, y1 + 0.1 * dy
    pw, ph = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def X(v: float) -> float:
        return PAD_L + (v - x0) / (x1 - x0) * pw

    def Y(v: float) -> float:
        return PAD_T + (1 - (v - y0) / (y1 - y0)) * ph

    out = [
        f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px;display:block;'
        'color:inherit;font:11px sans-serif" xmlns="http://www.w3.org/2000/svg" class="rso-pareto">',
        f'<rect x="{PAD_L}" y="{PAD_T}" width="{pw}" height="{ph}" fill="none" '
        'stroke="currentColor" stroke-opacity=".35"/>',
    ]
    for v in _ticks(min(xs), max(xs)):
        out.append(
            f'<text x="{X(v):.1f}" y="{PAD_T + ph + 13}" text-anchor="middle" fill="currentColor" '
            f'fill-opacity=".75">{_fmt(v)}</text>'
        )
    for v in _ticks(min(ys), max(ys)):
        out.append(
            f'<text x="{PAD_L - 4}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="currentColor" '
            f'fill-opacity=".75">{_fmt(v)}</text>'
        )
    out.append(
        f'<text x="{PAD_L + pw / 2}" y="{H - 6}" text-anchor="middle" fill="currentColor">'
        f"{escape(xlabel)}</text>"
    )
    out.append(
        f'<text transform="translate(12 {PAD_T + ph / 2}) rotate(-90)" text-anchor="middle" '
        f'fill="currentColor">{escape(ylabel)}</text>'
    )
    for n, x, y in points:
        lab = escape(button_label(n))
        js = (
            "var e=this,t='" + lab + "';while(e&&!e.querySelector){e=e.parentNode}"
            "for(;e;e=e.parentElement){var b=[...e.querySelectorAll('button')]"
            ".find(function(x){return x.textContent.trim()===t});if(b){b.click();break}}"
        )
        col = PICKED if n == picked else DOT
        out.append(
            f'<g style="cursor:pointer" onclick="{js}" data-plan="{n}"><title>{lab}</title>'
            f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="11" fill="transparent"/>'
            f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="{7 if n == picked else 5}" fill="{col}"/>'
            f'<text x="{X(x) + 8:.1f}" y="{Y(y) - 6:.1f}" fill="currentColor">{n}</text></g>'
        )
    out.append("</svg>")
    return "".join(out)


class ParetoView:
    """Results -> Pareto front (folder shown only when the list holds a front)."""

    def __init__(self, server, session, tool_length=None, select: Callable | None = None, picked=None):
        self.s, self.server = session, server
        self.tool_length = tool_length or (lambda sol: None)
        self.select = select  # (solution) -> None: make it the selected plan in Results
        self.picked = picked or (lambda: None)  # () -> the selected solution
        gui = server.gui
        labels = [label for label, _ in OBJECTIVES.values()]
        with gui.add_folder("Pareto front", expand_by_default=True, visible=False) as self.folder:
            self.x = gui.add_dropdown("X axis", labels, initial_value=labels[0])
            self.y = gui.add_dropdown("Y axis", labels, initial_value=labels[1])
            self.chart = gui.add_html("")
            self.buttons = None  # "Select plan" button group (rebuilt when the front changes)
            self.note = gui.add_markdown("")
        self._options: tuple = ()
        self.x.on_update(lambda _e: self.refresh())
        self.y.on_update(lambda _e: self.refresh())

    def plans(self) -> list:
        return [r for r in self.s.results if r.label.startswith("Pareto ")]

    @staticmethod
    def number(sol) -> int:
        """1 from "Pareto 1/5: ..." (the dot's number)."""
        head = sol.label.split(":")[0].replace("Pareto ", "")
        return int(head.split("/")[0])

    def pick(self, n: int) -> bool:
        """Select the plan with dot number ``n`` (what a click on the dot does)."""
        sol = next((p for p in self.plans() if self.number(p) == n), None)
        if sol is None or self.select is None:
            return False
        self.select(sol)
        self.refresh()
        return True

    def _button_group(self, options: tuple) -> None:
        if options == self._options and self.buttons is not None:
            return
        if self.buttons is not None:
            self.buttons.remove()
            self.buttons = None
        self._options = options
        if not options:
            return
        with self.folder:
            self.buttons = self.server.gui.add_button_group(
                "Select plan", options, order=self.chart.order + 0.01
            )

        @self.buttons.on_click
        def _(e) -> None:
            self.pick(int(str(e.target.value).split()[-1]))

    def refresh(self) -> None:
        plans = self.plans()
        self.folder.visible = bool(plans)
        if not plans:
            self._button_group(())
            self.chart.content = ""
            return
        key = {label: k for k, (label, _) in OBJECTIVES.items()}
        kx, ky = key[self.x.value], key[self.y.value]
        pts = []
        for sol in plans:
            m = measures(sol, self.s.result_points.get(sol.label, {}), self.tool_length(sol))
            pts.append((self.number(sol), m[kx], m[ky]))
        cur = self.picked()
        picked = self.number(cur) if cur is not None and cur.label.startswith("Pareto ") else None
        self.chart.content = scatter_svg(pts, self.x.value, self.y.value, picked)
        self._button_group(tuple(button_label(n) for n, _, _ in sorted(pts)))
        self.note.content = (
            f"_{len(plans)} plan(s) on the front: none is better than another on every measure. "
            "**Click a dot** (or a button) to select that plan; the numbers are the **Pareto i/n** "
            "plans in the list above._"
        )
