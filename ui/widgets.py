"""Shared GUI helpers: notifications, safe upload names, pose fields + gizmo."""

from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np
import viser

from core.transforms import make_T, matrix_to_rpy, matrix_to_wxyz, rpy_to_matrix, wxyz_to_matrix

MAX_UPLOAD = 200 * 1024 * 1024


def safe_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name)
    return stem.lstrip(".") or "upload"


class Latest:
    """Wrap a callback so only the newest call runs (drag smoothing).

    Viser runs GUI/scene callbacks on a thread pool: during a drag, dozens of events arrive
    and run concurrently and out of order, and the browser waits for all of them. With this
    wrapper, a call that arrives while ``fn`` is running only records its arguments; when
    ``fn`` returns it runs once more with the newest ones. Intermediate events are dropped.
    A call on an idle wrapper runs synchronously in the caller's thread (tests rely on it).
    """

    def __init__(self, fn):
        self.fn = fn
        self._lock = threading.Lock()
        self._pending = None
        self._running = False

    def __call__(self, *args) -> None:
        with self._lock:
            self._pending = args
            if self._running:
                return
            self._running = True
        try:
            while True:
                with self._lock:
                    args, self._pending = self._pending, None
                    if args is None:
                        return
                self.fn(*args)
        finally:
            with self._lock:
                self._running = False


def notify(event, title: str, body: str, ok: bool = False) -> None:
    client = getattr(event, "client", None)
    if client is not None:
        client.add_notification(title, body, color="green" if ok else "red", auto_close_seconds=6)


class PoseWidgets:
    """xyz (mm) + rpy (deg) inputs, two-way synced with a transform gizmo."""

    def __init__(self, server: viser.ViserServer, node: str, T: np.ndarray, on_change, scale: float = 0.3):
        self._on_change = on_change
        self._syncing = False
        self.gizmo = server.scene.add_transform_controls(node, scale=scale, visible=True)
        self.xyz = server.gui.add_vector3("xyz (mm)", (0.0, 0.0, 0.0), step=1.0)
        self.rpy = server.gui.add_vector3("rpy (deg)", (0.0, 0.0, 0.0), step=0.5)
        self.show = server.gui.add_checkbox("Show gizmo", True)
        self.set(T)

        from_gizmo = Latest(
            lambda: self._apply(make_T(wxyz_to_matrix(self.gizmo.wxyz), self.gizmo.position), source="gizmo")
        )
        from_fields = Latest(lambda: self._apply(self._from_fields(), source="fields"))

        @self.gizmo.on_update
        def _(_e) -> None:
            if not self._syncing:
                from_gizmo()

        @self.xyz.on_update
        def _(_e) -> None:
            if not self._syncing:
                from_fields()

        @self.rpy.on_update
        def _(_e) -> None:
            if not self._syncing:
                from_fields()

        @self.show.on_update
        def _(_e) -> None:
            self.set_handles(self.show.value)

    def set_handles(self, shown: bool) -> None:
        """Show/hide only the drag handles. Never toggle ``gizmo.visible`` for this: in Viser
        that hides the whole subtree (the robot under the base gizmo, the part under its)."""
        self.gizmo.disable_axes = not shown
        self.gizmo.disable_sliders = not shown
        self.gizmo.disable_rotations = not shown

    def _from_fields(self) -> np.ndarray:
        return make_T(rpy_to_matrix(np.radians(self.rpy.value)), np.array(self.xyz.value) / 1000)

    def set(self, T: np.ndarray, gizmo: bool = True) -> None:
        """Show pose T. ``gizmo=False`` while the user drags the gizmo: writing its pose back
        mid-drag would fight the drag in the browser."""
        self._syncing = True
        try:
            if gizmo:
                self.gizmo.wxyz = matrix_to_wxyz(T[:3, :3])
                self.gizmo.position = T[:3, 3]
            # full precision: rounding here would leak into the pose on the next field edit
            self.xyz.value = tuple(float(v) for v in T[:3, 3] * 1000)
            self.rpy.value = tuple(float(v) for v in np.degrees(matrix_to_rpy(T[:3, :3])))
        finally:
            self._syncing = False

    def _apply(self, T: np.ndarray, source: str) -> None:
        self.set(T, gizmo=source != "gizmo")
        self._on_change(T)

    def remove(self) -> None:
        for h in (self.gizmo, self.xyz, self.rpy, self.show):
            h.remove()
