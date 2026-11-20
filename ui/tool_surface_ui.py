"""Targets tab -> "Detect tool surface" (plan Part 4, step T4).

*Detect* runs ``core.tool_surface_pick.detect_tool_surface`` in a background thread on the
part mesh (STEP parts pass their CAD face ids; the up vector is world +Z expressed in the part
frame, so a rotated part still looks from above). The result is only a PREVIEW (magenta
overlay) with the detector's one-line summary, its confidence, the warnings and the secondary
candidates (each with an *Add* button). *Replace selection* / *Add to selection* turn the
preview into the normal selection; *Discard* clears it. Below ``LOW_CONFIDENCE`` the panel says
so prominently. Errors are notified, never silent.

The pad radius for the sanding check is not available: the tool model has no pad or contact
radius (``Extension.radius`` is the arm's collision capsule), so None is passed.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np
import viser

from ui.batch_materialize import detect_for_pose
from ui.log_panel import log
from ui.targets_draw import PART, SelectionLayer, chunk_arrays
from ui.widgets import notify

if TYPE_CHECKING:
    from ui.targets_ui import TargetsTab

LOW_CONFIDENCE = 0.7
PREVIEW_COLOR = (230, 70, 200)
SLOTS = 4  # secondary candidates listed


class PreviewLayer(SelectionLayer):
    """A SelectionLayer in another colour (the detection preview)."""

    def _add_chunk(self, m, faces: np.ndarray) -> None:
        V, F = chunk_arrays(m, faces)
        h = self.server.scene.add_mesh_simple(
            f"{self.root}/c{self._n}", V, F, color=PREVIEW_COLOR, opacity=0.65, side="double"
        )
        self._n += 1
        self.chunks.append((faces, h))


def pad_radius_m(session) -> float | None:
    """The tool's pad radius (m) for the concave-radius check; None when unknown."""
    tool = getattr(session, "tool", None)
    return getattr(tool, "pad_radius_m", None) if tool is not None else None


class ToolSurfacePanel:
    def __init__(self, server: viser.ViserServer, tab: TargetsTab):
        self.tab = tab
        self.layer = PreviewLayer(server, f"{PART}/tool_surface_preview")
        self.guess = None
        self.extra = np.zeros(0, dtype=int)  # faces of the secondary candidates added so far
        self._mesh = None  # the mesh the preview belongs to
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        gui = server.gui
        with gui.add_folder("Detect tool surface", expand_by_default=False):
            gui.add_markdown("_Guesses the tool surface of a mandrel (looking from world +Z)._")
            self.detect_btn = gui.add_button("Detect tool surface", color="violet")
            self.info = gui.add_markdown("")
            self.low = gui.add_markdown("", visible=False)
            self.warn = gui.add_markdown("", visible=False)
            self.sec_md, self.sec_btn = [], []
            for _ in range(SLOTS):
                self.sec_md.append(gui.add_markdown("", visible=False))
                self.sec_btn.append(gui.add_button("Add", visible=False))
            self.replace_btn = gui.add_button("Replace selection", color="green", visible=False)
            self.add_btn = gui.add_button("Add to selection", visible=False)
            self.discard_btn = gui.add_button("Discard", color="red", visible=False)
        self.detect_btn.on_click(lambda e: self.detect(e))
        self.replace_btn.on_click(lambda e: self.accept("set", e))
        self.add_btn.on_click(lambda e: self.accept("add", e))
        self.discard_btn.on_click(lambda e: self.discard())
        for i, b in enumerate(self.sec_btn):
            b.on_click(lambda e, i=i: self.add_secondary(i, e))

    # --- state ----------------------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def preview_faces(self) -> np.ndarray:
        if self.guess is None:
            return np.zeros(0, dtype=int)
        return np.union1d(np.asarray(self.guess.faces, int), self.extra)

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def sync(self) -> None:
        """The part changed: a preview of the old mesh is dropped."""
        if self.guess is not None and self.tab.mesh() is not self._mesh:
            self.discard()

    # --- detect ---------------------------------------------------------------------------------

    def detect(self, event=None, block: bool = False) -> bool:
        if self.running:
            notify(event, "Detect tool surface", "Detection is already running")
            return False
        mesh = self.tab.mesh()
        if mesh is None or self.tab.s.part is None:
            notify(event, "Detect tool surface", "Load a part first")
            return False
        T = np.array(self.tab.s.part.T_world_part, float)
        self.discard()
        self.detect_btn.disabled = True
        self.info.content = "_Detecting…_"
        self._thread = threading.Thread(target=self._work, args=(event, mesh, T), daemon=True)
        self._thread.start()
        if block:
            self._thread.join()
        return True

    def _work(self, event, mesh, T) -> None:
        try:
            if mesh.metadata.get("cad_face_ids") is None:  # a STEP part whose copy lost the sidecar
                from ui.session import load_cad_face_ids

                part = self.tab.s.part
                cad = load_cad_face_ids(self.tab.s.resolve(part.mesh_path), part.units)
                if cad is not None:
                    mesh.metadata["cad_face_ids"] = cad
            g = detect_for_pose(mesh, T, pad_radius_m(self.tab.s))
        except Exception as err:  # noqa: BLE001 - reported, never silent
            self.info.content = f"⚠ Detection failed: {type(err).__name__}: {err}"
            notify(event, "Detect tool surface", f"detection failed: {type(err).__name__}: {err}")
        else:
            if mesh is not self.tab.mesh():
                self.info.content = "_The part changed during detection; run it again._"
            else:
                with self._lock:
                    self.guess, self._mesh, self.extra = g, mesh, np.zeros(0, dtype=int)
                self._show()
                log(f"Tool surface: {g.summary} (confidence {g.confidence:.2f})", "done")
        finally:
            self.detect_btn.disabled = False

    # --- preview --------------------------------------------------------------------------------

    def _show(self) -> None:
        g = self.guess
        if g is None:
            self.layer.update(None, np.zeros(0, int))
            self.info.content = ""
            for w in (self.low, self.warn, self.replace_btn, self.add_btn, self.discard_btn):
                w.visible = False
            for md, b in zip(self.sec_md, self.sec_btn, strict=True):
                md.visible = b.visible = False
            return
        faces = self.preview_faces
        self.layer.update(self._mesh, faces)
        self.info.content = (
            f"**{g.summary}**\n\nConfidence {g.confidence:.0%}; preview {len(faces)} faces (magenta)."
        )
        self.low.visible = g.confidence < LOW_CONFIDENCE
        self.low.content = "⚠ **Low confidence: review before accepting**"
        self.warn.visible = bool(g.warnings)
        self.warn.content = "\n\n".join(f"⚠ {w}" for w in g.warnings)
        for i, (md, b) in enumerate(zip(self.sec_md, self.sec_btn, strict=True)):
            c = g.secondary[i] if i < len(g.secondary) else None
            md.visible = b.visible = c is not None
            if c is not None:
                taken = len(np.setdiff1d(c.faces, faces)) == 0
                md.content = (
                    f"Candidate {i + 1}: {c.reason} ({len(c.faces)} faces{', added' if taken else ''})"
                )
                b.disabled = taken
        for w in (self.replace_btn, self.add_btn, self.discard_btn):
            w.visible = True

    def add_secondary(self, i: int, event=None) -> bool:
        if self.guess is None or i >= len(self.guess.secondary):
            return False
        self.extra = np.union1d(self.extra, np.asarray(self.guess.secondary[i].faces, int))
        self._show()
        return True

    def discard(self) -> None:
        self.guess, self._mesh, self.extra = None, None, np.zeros(0, dtype=int)
        self._show()

    def accept(self, how: str = "set", event=None) -> bool:
        """Turn the preview into the selection: ``how`` = set (replace) or add."""
        if self.guess is None:
            notify(event, "Detect tool surface", "Nothing to accept: run Detect first")
            return False
        if self._mesh is not self.tab.mesh():
            self.discard()
            notify(event, "Detect tool surface", "The part changed since the detection; run it again")
            return False
        faces = self.preview_faces
        self.tab.select(faces, how)
        verb = "replaced" if how == "set" else "added to"
        notify(event, "Detect tool surface", f"{len(faces)} faces {verb} the selection", ok=True)
        self.discard()
        return True
