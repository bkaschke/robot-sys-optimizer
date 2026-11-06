"""Scene display of the session's end effector, attached to the robot flange.

Scene tree (persistent frames; their children are rebuilt):
    /cell/tool                  pose = T_world_flange
      /cell/tool/mount          gizmo, local pose = T_flange_mount (owned by ToolSetupTab)
      /cell/tool/extension      capsule mesh(es), flange frame
      /cell/tool/head           pose = T_flange_head
        /cell/tool/head/<link>  clickable link mesh at T_head_link (+ /contact overlay)
        /cell/tool/head/tcp_gizmo   gizmo, local pose = T_head_tcp (owned by ToolSetupTab)
      /cell/tool/tcp/<name>     TCP axes at T_flange_tcp
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import viser

from core.mesh_select import pick_face
from core.transforms import make_T, matrix_to_wxyz, wxyz_to_matrix
from ui.session import Session

ROOT = "/cell/tool"
HEAD = f"{ROOT}/head"
LINK_COLOR = (170, 175, 185)
CONTACT_COLOR = (255, 140, 0)
EXT_COLOR = (90, 140, 220)


def _set_pose(h, T: np.ndarray) -> None:
    h.wxyz = matrix_to_wxyz(T[:3, :3])
    h.position = T[:3, 3]


class ToolView:
    def __init__(self, server: viser.ViserServer, session: Session):
        self.server = server
        self.s = session
        self.root = server.scene.add_frame(ROOT, show_axes=False)
        self.head = server.scene.add_frame(HEAD, show_axes=False)
        self.ext_handle = None
        self.link_handles: dict[str, object] = {}
        self.overlay_handles: dict[str, object] = {}
        self.tcp_handles: dict[str, object] = {}
        self.on_pick: Callable[[str, int, object], None] | None = None  # (link, face, event)
        self.active_tcp: str | None = None

    # --- full rebuild (tool loaded / TCP list changed) --------------------------------------

    def rebuild(self) -> None:
        for h in [*self.link_handles.values(), *self.overlay_handles.values(), *self.tcp_handles.values()]:
            h.remove()
        self.link_handles, self.overlay_handles, self.tcp_handles = {}, {}, {}
        tm = self.s.tool_model
        self.root.visible = tm is not None
        if tm is None:
            if self.ext_handle is not None:
                self.ext_handle.remove()
                self.ext_handle = None
            return
        for hl in tm.links:
            h = self.server.scene.add_mesh_simple(
                f"{HEAD}/{hl.name}",
                hl.mesh.vertices,
                hl.mesh.faces,
                color=LINK_COLOR,
                side="double",
                wxyz=matrix_to_wxyz(hl.T_head_link[:3, :3]),
                position=hl.T_head_link[:3, 3],
            )
            h.on_click(self._click_handler(hl.name))
            self.link_handles[hl.name] = h
        for t in tm.ee.tcps:
            self.tcp_handles[t.name] = self.server.scene.add_frame(
                f"{ROOT}/tcp/{t.name}", axes_length=0.06, axes_radius=0.003
            )
        self.update()

    def _click_handler(self, link: str):
        def handler(event) -> None:
            if self.on_pick is None:
                return
            tm = self.s.tool_model
            T_world_link = self.s.T_world_flange() @ tm.T_flange_head() @ tm.link(link).T_head_link
            T_inv = np.linalg.inv(T_world_link)
            o = T_inv[:3, :3] @ np.asarray(event.ray_origin) + T_inv[:3, 3]
            d = T_inv[:3, :3] @ np.asarray(event.ray_direction)
            face = pick_face(tm.link(link).mesh, o, d)
            if face is not None:
                self.on_pick(link, face, event)

        return handler

    # --- cheap updates (parameters changed) ---------------------------------------------------

    def set_flange(self, T_world_flange: np.ndarray) -> None:
        _set_pose(self.root, T_world_flange)

    def update(self) -> None:
        """Re-pose head and TCPs, regenerate the extension mesh and contact overlays."""
        tm = self.s.tool_model
        if tm is None:
            return
        self.set_flange(self.s.T_world_flange())
        _set_pose(self.head, tm.T_flange_head())
        mesh = tm.extension_mesh()
        if self.ext_handle is not None:
            self.ext_handle.remove()
            self.ext_handle = None
        if mesh is not None:
            self.ext_handle = self.server.scene.add_mesh_simple(
                f"{ROOT}/extension", mesh.vertices, mesh.faces, color=EXT_COLOR, opacity=0.85
            )
        for t in tm.ee.tcps:
            if t.name in self.tcp_handles:
                _set_pose(self.tcp_handles[t.name], tm.T_flange_tcp(t.name))
        self._update_contacts()

    def _update_contacts(self) -> None:
        for h in self.overlay_handles.values():
            h.remove()
        self.overlay_handles = {}
        tm = self.s.tool_model
        tcp = next((t for t in tm.ee.tcps if t.name == self.active_tcp), None)
        whole = set(tcp.contact_links) if tcp else set()
        faces = tcp.contact_faces if tcp else {}
        for name, h in self.link_handles.items():
            h.color = CONTACT_COLOR if name in whole else LINK_COLOR
            ids = faces.get(name, [])
            if ids and name not in whole:
                hl = tm.link(name)
                sub = hl.mesh.submesh([np.asarray(ids)], append=True)
                # lift the overlay 0.2 mm along its normals so it draws on top of the link
                verts = sub.vertices + 2e-4 * sub.vertex_normals
                self.overlay_handles[name] = self.server.scene.add_mesh_simple(
                    f"{HEAD}/{name}/contact", verts, sub.faces, color=CONTACT_COLOR
                )

    def T_world_tcp(self, name: str) -> np.ndarray:
        """TCP pose as the browser receives it (composed from scene handles; for tests)."""
        h, r = self.tcp_handles[name], self.root
        return make_T(wxyz_to_matrix(r.wxyz), r.position) @ make_T(wxyz_to_matrix(h.wxyz), h.position)
