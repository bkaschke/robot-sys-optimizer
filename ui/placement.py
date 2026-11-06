"""Base placement panel (Optimize tab, M6): build/load the reachability map for the current
robot + tool TCP, vote base positions for all targets, draw the heatmap on the floor and
move the base to the peak.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np
import viser

from core import targets as tg
from core.robot import Robot
from core.transforms import matrix_to_wxyz
from opt.reachability import MapSettings, ReachMap, VoteGrid, map_for_tool, vote_bases
from ui.session import Session
from ui.widgets import notify

HEATMAP = "/cell/heatmap"
STOPS = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], float)


def colormap(v: np.ndarray) -> np.ndarray:
    """Viridis-like colours for v in [0, 1] -> uint8 RGB."""
    x = np.clip(v, 0, 1) * (len(STOPS) - 1)
    i = np.minimum(x.astype(int), len(STOPS) - 2)
    t = (x - i)[..., None]
    return ((1 - t) * STOPS[i] + t * STOPS[i + 1]).astype(np.uint8)


class PlacementPanel:
    def __init__(self, server: viser.ViserServer, session: Session, set_base: Callable[[np.ndarray], None]):
        self.server = server
        self.s = session
        self.set_base = set_base
        self.rmap: ReachMap | None = None
        self.votes: VoteGrid | None = None
        self._map_key = None
        self._handles: list = []
        gui = server.gui
        with gui.add_folder("Base placement (reachability)"):
            self.res = gui.add_number("Map cell (mm)", 50.0, min=10.0, max=200.0, step=5.0)
            self.grid = gui.add_number("Heatmap cell (mm)", 25.0, min=5.0, max=200.0, step=5.0)
            self.half = gui.add_number("Search half-width (m)", 2.5, min=0.5, max=10.0, step=0.5)
            self.show = gui.add_button("Show base heatmap")
            self.progress = gui.add_progress_bar(0.0, visible=False)
            self.info = gui.add_markdown(
                "_Map not built. Building takes ~1 min once per tool length; it is cached afterwards._"
            )
            self.move = gui.add_button("Move base to best", disabled=True)
            self.hide = gui.add_button("Hide heatmap")
        self.show.on_click(lambda e: threading.Thread(target=self.compute, args=(e,), daemon=True).start())

        @self.move.on_click
        def _(e) -> None:
            if self.votes is None:
                return
            x, y = self.votes.best()
            T = self.s.T_world_base.copy()
            T[:2, 3] = (x, y)
            self.set_base(T)

        @self.hide.on_click
        def _(e) -> None:
            self._clear()

    # --- computation -----------------------------------------------------------------------

    def _tcp(self) -> str:
        groups = {g.name: g for g in self.s.targets.groups}
        for p in self.s.targets.points:
            g = groups.get(p.group)
            if g is not None and g.tcps:
                return g.tcps[0]
        return self.s.tool.tcps[0].name

    def compute(self, event=None) -> None:
        s = self.s
        if s.tool_model is None or s.part is None or not s.targets.points:
            return notify(event, "Cannot vote", "Need a tool, a part and targets")
        robot = Robot(s.robot.spec)
        T_ft = s.tool_model.T_flange_tcp(self._tcp())
        settings = MapSettings(res=self.res.value / 1000)
        self.show.disabled, self.progress.visible = True, True
        try:
            if s.tool.extension.type == "pose":  # any TCP offset: the shared flange map, no build
                self.rmap = map_for_tool(robot, T_ft, settings, progress=self._progress, any_offset=True)
            else:
                self.rmap = ReachMap.load_or_build(robot, T_ft, settings, progress=self._progress)
        except NotImplementedError as err:
            return notify(event, "No map for this robot", str(err))
        finally:
            self.show.disabled, self.progress.visible = False, False
        _, P, N, G = tg.as_arrays(s.targets)
        Tp = s.part.T_world_part
        Pw, Nw = P @ Tp[:3, :3].T + Tp[:3, 3], N @ Tp[:3, :3].T
        tilt = {g.name: g.tilt_deg for g in s.targets.groups}
        cones = np.array([tilt.get(g, 0.0) for g in G])
        c = Pw[:, :2].mean(axis=0)
        step, half = self.grid.value / 1000, self.half.value
        xs = np.arange(c[0] - half, c[0] + half + 1e-9, step)
        ys = np.arange(c[1] - half, c[1] + half + 1e-9, step)
        self.votes = vote_bases(
            self.rmap, Pw, Nw, cones, xs, ys, s.T_world_base[2, 3], s.T_world_base[:3, :3]
        )
        self._draw(len(P))

    def _progress(self, f: float) -> None:
        self.progress.value = 100.0 * f

    # --- drawing -------------------------------------------------------------------------------

    def _clear(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _draw(self, n_targets: int) -> None:
        self._clear()
        v = self.votes
        full = v.coverage >= n_targets - 1e-6
        frac = v.coverage / max(1, n_targets)
        # partial coverage in the lower half of the colormap, full coverage by quality on top
        val = np.where(
            full, 0.5 + 0.5 * v.quality / max(v.quality[full].max(), 1e-9) if full.any() else 0, 0.5 * frac
        )
        rgba = np.zeros(v.score.shape + (4,), np.uint8)
        rgba[..., :3] = colormap(val)
        rgba[..., 3] = np.where(v.coverage > 0, 170, 0)
        img = rgba  # Viser image: columns run along +x, row 0 at -y (checked with a test image)
        w, h = v.xs[-1] - v.xs[0], v.ys[-1] - v.ys[0]
        center = ((v.xs[0] + v.xs[-1]) / 2, (v.ys[0] + v.ys[-1]) / 2, 0.002)
        self._handles.append(
            self.server.scene.add_image(
                HEATMAP, img, w, h, position=center, cast_shadow=False, receive_shadow=False
            )
        )
        bx, by = v.best()
        self._handles.append(
            self.server.scene.add_frame(
                f"{HEATMAP}_best",
                axes_length=0.25,
                axes_radius=0.01,
                wxyz=matrix_to_wxyz(np.eye(3)),
                position=(bx, by, 0.01),
            )
        )
        k = np.unravel_index(np.argmax(v.score), v.score.shape)
        self.info.content = (
            f"Best base: **x {bx * 1000:.0f} mm, y {by * 1000:.0f} mm**  \n"
            f"covers {v.coverage[k]:.1f}/{n_targets} targets (map estimate), worst-point "
            f"manipulability {v.quality[k]:.2f}  \n_Yaw does not matter for this robot. Run the "
            f"feasibility check to confirm with collisions and torque._"
        )
        self.move.disabled = False
