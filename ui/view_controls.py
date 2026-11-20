"""Look and camera of the 3D view (R2): lighting presets, shadows, "Fit view" and the
Top / Front / Side / Iso buttons.

Lighting is built from viser's own pieces and works offline: the HDRI presets are JPEGs inside
the viser package (``viser/_assets/hdri``); ``configure_environment_map`` reads the file on the
server and sends the bytes over the websocket, so the browser never fetches anything from a CDN.

* Studio: a soft "studio" environment map (the theme background stays), a key light that can cast
  shadows onto the floor grid, and a weak fill light.
* Flat: no shading direction; even ambient light (good for reading colours).
* Off: viser's plain default lights and environment (the look before R2).

Shadows cost GPU time (software WebGL on office PCs can be slow), so they start off.

The camera fits the bounding box of the robot (at its current pose), the part, the carrier and the
stay-out zones. The view buttons keep the distance of the last fit.
"""

from __future__ import annotations

import itertools
import re

import numpy as np
import viser

from ui.log_panel import log

LIGHTING = ("Studio", "Flat", "Off")
HDRI = "studio"  # a bundled, neutral preset (viser/_assets/hdri/studio_small_03_1k.jpg)
ENV_INTENSITY = 0.5  # low: the part colours must not wash out
KEY_POSITION = (-1.5, -2.5, 4.0)  # the light shines from here toward the world origin
FILL_POSITION = (2.0, 2.0, 1.0)
KEY_INTENSITY, FILL_INTENSITY, AMBIENT_INTENSITY = 2.4, 0.5, 1.8
FOV_DEG = 45.0  # viser's default vertical field of view
FIT_MARGIN = 1.3  # room around the cell (the side panels cover some of the view)
ROBOT_PAD = 0.15  # m: links are thicker than their frame origins
TOOL_PAD = 0.3
PANEL_SHIFT = 0.06  # look-at shift to screen-right (of the distance): the cell clears the right-hand panel
TRAVEL_CLIP = 10.0  # m: a long rail shows its ends up to here

# view name -> (direction from the cell centre to the camera, up direction)
VIEWS = {
    "Top": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "Front": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "Side": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "Iso": ((1.0, -1.0, 0.75), (0.0, 0.0, 1.0)),
}


def _corners(box: np.ndarray, T: np.ndarray) -> np.ndarray:
    """The 8 corners of the (2, 3) box [min, max], moved by the pose ``T`` (world)."""
    pts = np.array(list(itertools.product(*zip(box[0], box[1], strict=True))))
    return pts @ T[:3, :3].T + T[:3, 3]


def cell_points(s) -> list[np.ndarray]:
    """(n, 3) point sets (world, m) that bound everything in the study."""
    out: list[np.ndarray] = []
    if s.part is not None:
        try:
            shown, _ = s.part_display()
            out.append(_corners(np.asarray(shown.bounds), s.part.T_world_part))
        except Exception:  # a part file that doesn't load: nothing to frame
            pass
    if s.robot is not None:
        try:
            q = s.q if s.q is not None else s.robot.home_q()
            pts = [s.T_world_base[:3, 3]]
            for f in s.robot.model.frames:
                pts.append((s.T_world_base @ s.robot.frame_pose(q, f.name))[:3, 3])
            P = np.array(pts)
            out.append(np.vstack([P - ROBOT_PAD, P + ROBOT_PAD]))
            tcp = s.T_world_flange()[:3, 3]
            out.append(np.array([tcp - TOOL_PAD, tcp + TOOL_PAD]))
        except Exception:
            pass
    try:
        c = s.carrier() if s.base.mode == "movable" else None
    except Exception:
        c = None
    if c is not None:
        try:
            for b in c.static_bodies():
                out.append(_corners(np.asarray(b.mesh.bounds), b.T))
            if len(c.dofs):
                lo, hi = (
                    np.clip(c.bounds[:, 0], -TRAVEL_CLIP, TRAVEL_CLIP),
                    np.clip(c.bounds[:, 1], -TRAVEL_CLIP, TRAVEL_CLIP),
                )
                for ends in (lo, hi):
                    v = dict(zip(c.dofs, (float(x) for x in ends), strict=True))
                    for b in c.moving_bodies(v):
                        out.append(_corners(np.asarray(b.mesh.bounds), b.T))
        except Exception:
            pass
    from core.carrier import primitive_mesh
    from core.transforms import pose_to_matrix

    for z in getattr(s, "stayouts", []) or []:
        try:
            out.append(_corners(np.asarray(primitive_mesh(z, s.resolve).bounds), pose_to_matrix(z.pose)))
        except (ValueError, OSError):
            continue
    return out


def cell_bounds(s) -> np.ndarray | None:
    """(2, 3) [min, max] box around the cell (world, m); None for an empty study."""
    pts = cell_points(s)
    if not pts:
        return None
    P = np.vstack(pts)
    return np.array([P.min(axis=0), P.max(axis=0)])


def fit_distance(box: np.ndarray) -> float:
    """Camera distance from the box centre at which its bounding sphere fills the view."""
    radius = max(0.5 * float(np.linalg.norm(box[1] - box[0])), 0.3)
    return FIT_MARGIN * radius / float(np.sin(np.radians(FOV_DEG) / 2.0))


def view_pose(
    centre, dist: float, view: str, shift: float = PANEL_SHIFT
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """(camera position, look-at point, up direction) for a named view at ``dist``."""
    d, up = VIEWS[view]
    d = np.asarray(d, float) / np.linalg.norm(d)
    right = np.cross(-d, up)
    c = np.asarray(centre, float) + shift * dist * right / np.linalg.norm(right)
    return c + dist * d, c, up


class ViewControls:
    """The "View" folder: Lighting, Shadows, Fit view, Top / Front / Side / Iso."""

    def __init__(self, server: viser.ViserServer, session):
        self.server, self.s = server, session
        self.frame: tuple[np.ndarray, float] | None = None  # (centre, distance) of the last fit
        self.fit_part: str | None = None  # part (mesh path, without a refined-size suffix) at the last fit
        scene = server.scene
        server.initial_camera.fov = float(np.radians(FOV_DEG))
        self.key = scene.add_light_directional(
            "/world/light_key", intensity=KEY_INTENSITY, cast_shadow=False, position=KEY_POSITION
        )
        self.fill = scene.add_light_directional(
            "/world/light_fill", intensity=FILL_INTENSITY, cast_shadow=False, position=FILL_POSITION
        )
        self.ambient = scene.add_light_ambient("/world/light_ambient", intensity=AMBIENT_INTENSITY)
        gui = server.gui
        with gui.add_folder("View", expand_by_default=True):
            self.lighting = gui.add_dropdown(
                "Lighting",
                LIGHTING,
                initial_value="Studio",
                hint="Studio: soft environment light and a key light; Flat: even light; "
                "Off: the plain viewer lights",
            )
            self.shadows = gui.add_checkbox(
                "Shadows",
                False,
                hint="the key light's shadow on the floor; slow on PCs without a graphics card",
            )
            self.fit_btn = gui.add_button("Fit view", hint="frame the robot, part, carrier and zones")
            self.views = gui.add_button_group("Direction", tuple(VIEWS))
        self.lighting.on_update(lambda _e: self.apply_lighting())
        self.shadows.on_update(lambda _e: self.apply_shadows())
        self.fit_btn.on_click(lambda _e: self.fit())
        self.views.on_click(lambda e: self.set_view(e.target.value))
        self.apply_lighting()

    # --- lighting --------------------------------------------------------------

    def apply_lighting(self) -> None:
        mode = self.lighting.value
        scene = self.server.scene
        if mode == "Studio":
            scene.configure_default_lights(enabled=False, cast_shadow=False)
            scene.configure_environment_map(
                HDRI, background=False, environment_intensity=ENV_INTENSITY
            )  # background=False: the theme background stays
        elif mode == "Flat":
            scene.configure_default_lights(enabled=False, cast_shadow=False)
            scene.configure_environment_map(None)
        else:  # "Off": viser's own default lights and environment map
            scene.configure_default_lights(enabled=True, cast_shadow=False)
            scene.configure_environment_map("city", background=False)
        self.key.visible = mode == "Studio"
        self.fill.visible = mode == "Studio"
        self.ambient.visible = mode == "Flat"
        self.apply_shadows()

    def apply_shadows(self) -> None:
        """The key light casts shadows when the box is ticked (and the light is in use)."""
        self.key.cast_shadow = bool(self.shadows.value)

    # --- camera ----------------------------------------------------------------

    def fit(self, view: str = "Iso", quiet: bool = False) -> bool:
        """Frame the cell from the named view. Returns False for an empty study."""
        box = cell_bounds(self.s)
        if box is None:
            if not quiet:
                log("Fit view: nothing to frame yet (no robot, part, carrier or zone)", "warn")
            return False
        self.fit_part = self.part_key()
        self.frame = ((box[0] + box[1]) / 2.0, fit_distance(box))
        self._go(view)
        return True

    def part_key(self) -> str | None:
        """The part's file, ignoring tessellation (``<stem>_refined_<N>mm``): refining keeps the view."""
        p = self.s.part
        return None if p is None else re.sub(r"_refined_[0-9.]+mm(?=\.)", "", p.mesh_path)

    def on_part_changed(self) -> None:
        """Part listener: refit only for a different part (not a pose, unit or tessellate edit)."""
        if self.part_key() != self.fit_part:
            self.fit(quiet=True)

    def set_view(self, view: str) -> None:
        """Look at the cell centre from the named direction, at the distance of the last fit."""
        if self.frame is None and not self.fit(view):
            return
        self._go(view)

    def _go(self, view: str) -> None:
        centre, dist = self.frame
        pos, look, up = view_pose(centre, dist, view)
        init = self.server.initial_camera
        init.position, init.look_at, init.up_direction = tuple(pos), tuple(look), up
        for client in self.server.get_clients().values():
            cam = client.camera
            cam.up_direction = up  # first: a top view looks along the old up
            cam.position = tuple(pos)
            cam.look_at = tuple(look)
