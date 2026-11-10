"""Pareto fronts in the GUI (v8 T11): the measure checkboxes of the planners and the Results ->
Pareto front scatter (two chosen measures, one numbered dot per plan, the numbers are the
"Pareto i/n" labels in the Results list; viser cannot report clicks on an image, so pick the
plan in the list)."""

from __future__ import annotations

import io

import numpy as np

from opt.pareto import OBJECTIVES, measures


def pareto_keys_widgets(gui, default: tuple) -> dict:
    """One checkbox per measure (hidden until the objective is Pareto front)."""
    return {
        k: gui.add_checkbox(f"Pareto: {label}", k in default, visible=False)
        for k, (label, _sense) in OBJECTIVES.items()
    }


def scatter_png(
    points: list[tuple[str, float, float]], xlabel: str, ylabel: str, dark: bool = False
) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fg, bg = ("#e6e8eb", "#16181c") if dark else ("#1d1f23", "#ffffff")
    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=100)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    xs, ys = np.array([p[1] for p in points]), np.array([p[2] for p in points])
    ax.scatter(xs, ys, s=42, color="#3d6fc4", zorder=3)
    for name, x, y in points:
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=9, color=fg)
    ax.set_xlabel(xlabel, color=fg)
    ax.set_ylabel(ylabel, color=fg)
    ax.tick_params(colors=fg)
    for sp in ax.spines.values():
        sp.set_color(fg)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=bg)
    plt.close(fig)
    return buf.getvalue()


class ParetoView:
    """Results -> Pareto front (folder shown only when the list holds a front)."""

    def __init__(self, server, session, tool_length=None):
        self.s, self.server = session, server
        self.tool_length = tool_length or (lambda sol: None)
        gui = server.gui
        labels = [label for label, _ in OBJECTIVES.values()]
        with gui.add_folder("Pareto front", expand_by_default=True, visible=False) as self.folder:
            self.x = gui.add_dropdown("X axis", labels, initial_value=labels[0])
            self.y = gui.add_dropdown("Y axis", labels, initial_value=labels[1])
            self.note = gui.add_markdown("")
            self.image = None
        self.x.on_update(lambda _e: self.refresh())
        self.y.on_update(lambda _e: self.refresh())

    def plans(self) -> list:
        return [r for r in self.s.results if r.label.startswith("Pareto ")]

    def refresh(self) -> None:
        plans = self.plans()
        self.folder.visible = bool(plans)
        if not plans:
            return
        key = {label: k for k, (label, _) in OBJECTIVES.items()}
        kx, ky = key[self.x.value], key[self.y.value]
        pts = []
        for sol in plans:
            m = measures(sol, self.s.result_points.get(sol.label, {}), self.tool_length(sol))
            pts.append((sol.label.split(":")[0].replace("Pareto ", "#"), m[kx], m[ky]))
        png = scatter_png(pts, self.x.value, self.y.value)
        from PIL import Image

        img = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        if self.image is None:
            with self.folder:
                self.image = self.server.gui.add_image(img, format="png")
        else:
            self.image.image = img
        self.note.content = (
            f"_{len(plans)} plan(s) on the front: none is better than another on every measure. "
            "The numbers are the **Pareto i/n** plans in the list above._"
        )
