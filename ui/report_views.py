"""Framed 3D views for the report (R3).

``capture`` renders the cell from named directions (Iso, Top, ...) through the clicking
browser: the camera is placed with ``view_controls.view_pose`` from the cell bounds, the floor
grid, the world origin, the drag gizmos and the target selection are hidden, the PNG is checked
for the empty-scene race (CLAUDE.md, "Headless screenshots occasionally capture before the
scene streams in") and retried, and the camera and every hidden item are restored in a
``finally``. Nothing else in the scene changes, so the robot, part, carrier, stay-out zones,
result ghosts and the heatmap appear as the user has them.

The scene is shared by all browsers, so the hidden items are also hidden for them during the
capture (a few seconds).

Transform gizmos are not hidden with ``visible = False``: that would hide their whole subtree
(the robot hangs under the base gizmo, see CLAUDE.md "Viser gotcha"). Their handles are
switched off with ``disable_axes`` / ``disable_sliders`` / ``disable_rotations`` instead.
"""

from __future__ import annotations

import io
import logging
import time

import numpy as np

from ui.floor import NODE as FLOOR_NODE
from ui.view_controls import VIEWS, cell_bounds, fit_distance, view_pose

log = logging.getLogger(__name__)

DEFAULT_VIEWS = ("Iso", "Top")
ORDER = ("Iso", "Top", "Front", "Side")  # order in the report
SETTLE_S = 0.6  # after moving the camera: the browser applies it and streams the scene
RETRY_WAIT_S = 0.8  # after a blank capture
TRIES = 3
BLANK_STD = 0.5  # grey-level standard deviation (0..255) at or below which a capture is empty
HIDE_NODES = (FLOOR_NODE, "/world/origin")  # exact node names
HIDE_PREFIXES = ("/cell/part/selection",)  # the face selection layer
GIZMO_FLAGS = ("disable_axes", "disable_sliders", "disable_rotations")


class ViewList(list):
    """The captured ``(caption, png bytes)`` pairs; ``notes`` says why views are missing."""

    def __init__(self, items=(), notes=()):
        super().__init__(items)
        self.notes: list[str] = list(notes)


def to_png(render) -> tuple[bytes, float]:
    """(PNG bytes on a white background, grey-level std) of a ``get_render`` array.

    The PNG render has a transparent background; compositing on white gives the report a
    clean background and makes an empty scene a uniform image."""
    from PIL import Image

    a = np.asarray(render, dtype=np.uint8)
    if a.ndim == 3 and a.shape[2] == 4:
        al = a[..., 3:4].astype(float) / 255.0
        rgb = (a[..., :3] * al + 255.0 * (1.0 - al)).round().astype(np.uint8)
    else:
        rgb = a[..., :3]
    std = float(rgb.astype(float).mean(axis=2).std())
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue(), std


class _Hidden:
    """Hide the scene's helper items; ``restore`` puts every changed attribute back."""

    def __init__(self):
        self.saved: list[tuple[object, str, object]] = []

    def hide(self, server) -> None:
        nodes = getattr(server.scene, "_handle_from_node_name", {})  # viser's registry of handles
        for name, h in list(nodes.items()):
            try:
                if getattr(h, "disable_axes", None) is not None:  # a transform gizmo
                    for flag in GIZMO_FLAGS:
                        self._set(h, flag, True)
                elif name in HIDE_NODES or name.startswith(HIDE_PREFIXES):
                    self._set(h, "visible", False)
            except Exception as err:  # a node removed meanwhile
                log.debug("could not hide %s: %s", name, err)

    def _set(self, h, attr: str, value) -> None:
        old = getattr(h, attr)
        if old != value:
            self.saved.append((h, attr, old))
            setattr(h, attr, value)

    def restore(self) -> None:
        for h, attr, old in reversed(self.saved):
            try:
                setattr(h, attr, old)
            except Exception as err:  # removed handle: nothing to restore
                log.debug("could not restore %s: %s", attr, err)
        self.saved = []


def _look(cam, pos, look, up) -> None:
    cam.up_direction = tuple(up)  # first: a top view looks along the old up
    cam.position = tuple(pos)
    cam.look_at = tuple(look)


def _render(client, size) -> tuple[bytes | None, float]:
    """One view, retried while the capture is empty. (None, std) when it stayed empty."""
    std = 0.0
    for attempt in range(TRIES):
        time.sleep(SETTLE_S if attempt == 0 else RETRY_WAIT_S)
        png, std = to_png(client.camera.get_render(size[0], size[1], transport_format="png", timeout=10.0))
        if std > BLANK_STD:
            return png, std
        log.info("report view: empty capture (try %d of %d)", attempt + 1, TRIES)
    return None, std


def capture(server, client, session, plan=None, views=DEFAULT_VIEWS, size=(720, 1280)) -> ViewList:
    """Render the named ``views`` of the cell: a ``ViewList`` of ``(caption, png bytes)``.

    ``size`` is (height, width). With no client, an empty study, or a render failure the list
    is empty (or shorter) and ``.notes`` holds "3D views not included: <reason>"."""
    out = ViewList()
    if client is None:
        out.notes.append(
            "3D views not included: no browser is connected (reports are rendered only from the GUI)."
        )
        return out
    views = [v for v in views if v in VIEWS]
    if not views:
        out.notes.append("3D views not included: no view was chosen.")
        return out
    box = cell_bounds(session)
    if box is None:
        out.notes.append("3D views not included: nothing to show yet (no robot, part, carrier or zone).")
        return out
    centre, dist = (box[0] + box[1]) / 2.0, fit_distance(box)
    cam = client.camera
    saved_cam = (cam.position, cam.look_at, cam.up_direction)
    hidden = _Hidden()
    try:
        hidden.hide(server)
        for v in views:
            _look(cam, *view_pose(centre, dist, v, shift=0.0))  # no side panel in a render: centred
            try:
                png, _ = _render(client, size)
            except Exception as err:  # timeout / renderer unavailable: stop, keep what we have
                out.notes.append(f"3D views not included: {v} view failed ({type(err).__name__}: {err}).")
                break
            if png is None:
                out.notes.append(f"3D views not included: the {v} view stayed empty after {TRIES} tries.")
                continue
            out.append((f"{v} view", png))
    except Exception as err:
        log.exception("report views failed")
        out.notes.append(f"3D views not included: {type(err).__name__}: {err}")
    finally:
        try:
            hidden.restore()
        finally:
            _look(cam, saved_cam[0], saved_cam[1], saved_cam[2])
    return out
