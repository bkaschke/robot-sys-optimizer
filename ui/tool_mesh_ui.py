"""Tool tab -> *Mesh / STEP tool*: a tool from one STL / OBJ / PLY / STEP file (core.tool_import),
and for STEP files the TCP or the flange mount placed on one of the model's named coordinate
systems (datum CSYS, core.step_datums).

*Set TCP to datum* puts the TCP selected in the TCPs folder on the chosen datum (*Flip z*:
turned 180° about its x axis, because the TCP's z must point out of the tool toward the part).
*Mount datum on flange* makes the chosen datum coincide with the robot flange (the mesh is
the head and sits on the mount), *Flip z* likewise. Both go through ``ToolSetupTab._commit``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.tool_import import (
    MESH_EXTS,
    STEP_EXTS,
    ToolImportError,
    import_mesh_tool,
    load_datums,
    mesh_tool_ee,
)
from core.transforms import matrix_to_pose
from ui.widgets import MAX_UPLOAD, notify, safe_name

NONE = "(none)"
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # 180 deg about x: z reversed, x kept


class ToolMeshPanel:
    def __init__(self, server, tab):
        self.server, self.tab, self.s = server, tab, tab.s
        self.datums: list = []
        gui = server.gui
        with gui.add_folder("Mesh / STEP tool", expand_by_default=False):
            gui.add_markdown(
                "_A tool from one file: the mesh is the tool head, on the flange mount. A STEP "
                "file's named coordinate systems can place the TCP and the mount._"
            )
            self.units = gui.add_dropdown(
                "Mesh units", ("mm", "m", "in"), hint="for STL / OBJ / PLY (a STEP file has its own)"
            )
            self.upload = gui.add_upload_button("Upload tool mesh (STL/OBJ/PLY/STEP)")
            self.info = gui.add_markdown("")
            self.tcp_datum = gui.add_dropdown("TCP datum", (NONE,), visible=False)
            self.tcp_flip = gui.add_checkbox(
                "Flip TCP z",
                False,
                visible=False,
                hint="the TCP z must point out of the tool, toward the part",
            )
            self.contact_r = gui.add_number(
                "Contact faces within (mm)",
                5.0,
                min=0.0,
                max=200.0,
                step=1.0,
                visible=False,
                hint="mesh faces this close to the TCP may touch the part (0 = none: a non-contact tool)",
            )
            self.set_tcp = gui.add_button("Set TCP to datum", visible=False)
            self.mount_datum = gui.add_dropdown("Flange datum", (NONE,), visible=False)
            self.mount_flip = gui.add_checkbox(
                "Flip flange z",
                False,
                visible=False,
                hint="the flange z points away from the robot, into the tool",
            )
            self.set_mount = gui.add_button("Mount datum on flange", visible=False)
        self.upload.on_upload(lambda e: self.add_file(self.upload.value.name, self.upload.value.content, e))
        self.set_tcp.on_click(lambda e: self.apply_tcp(e))
        self.set_mount.on_click(lambda e: self.apply_mount(e))
        tab.refresh_listeners.append(self.refresh)
        self.refresh()

    # --- actions ---------------------------------------------------------------------------

    def add_file(self, filename: str, data: bytes, event=None) -> bool:
        ext = Path(filename).suffix.lower()
        if ext not in MESH_EXTS + STEP_EXTS or len(data) > MAX_UPLOAD:
            notify(event, "Unsupported file", f"Use {', '.join(MESH_EXTS + STEP_EXTS)} under 200 MB")
            return False
        name = safe_name(Path(filename).stem) or "tool"
        dest = self.s.project_dir / "assets" / "tools" / name
        try:
            urdf, datums = import_mesh_tool(filename, data, dest, name, self.units.value)
            self.s.set_tool(mesh_tool_ee(name, self.s.relative(urdf)))
        except (ToolImportError, ValueError, OSError) as err:
            notify(event, "Tool import failed", str(err))
            return False
        self.tab.refresh()  # refreshes this panel too (datums)
        more = f"; {len(datums)} coordinate system(s): pick the TCP datum" if datums else ""
        notify(event, "Tool loaded", f"{name}{more}", ok=True)
        return True

    def _datum(self, name: str):
        return next((d for d in self.datums if d.name == name), None)

    def apply_tcp(self, event=None) -> bool:
        d = self._datum(self.tcp_datum.value)
        if d is None or self.s.tool is None:
            return False
        T = d.T @ FLIP if self.tcp_flip.value else d.T
        cur = self.tab.tcp_choice.value
        pose = matrix_to_pose(T).model_dump()
        faces = self.faces_near(T[:3, 3], float(self.contact_r.value) / 1000)

        def fn(dd: dict) -> None:
            tcp = self.tab._tcp(dd, cur)
            tcp["T_head_tcp"] = pose
            tcp["contact_faces"] = faces  # the tip may touch the part (replaces earlier picks)

        if not self.tab._commit(fn, event):
            return False
        self.tab._after(full=True)
        n = sum(len(v) for v in faces.values())
        notify(
            event,
            "TCP placed",
            f"{cur} on {d.name}" + (f"; {n} contact face(s) at the tip" if n else ""),
            ok=True,
        )
        return True

    def apply_mount(self, event=None) -> bool:
        d = self._datum(self.mount_datum.value)
        if d is None or self.s.tool is None:
            return False
        T = d.T @ FLIP if self.mount_flip.value else d.T
        pose = matrix_to_pose(np.linalg.inv(T)).model_dump()  # the datum lands on the flange
        if not self.tab._commit(lambda dd: dd.update(T_flange_mount=pose), event):
            return False
        self.tab._after(full=True)
        notify(event, "Mounted", f"{d.name} is on the flange", ok=True)
        return True

    def faces_near(self, p, r: float) -> dict[str, list[int]]:
        """{head link: face ids}: the faces within ``r`` of ``p`` (head frame), grown over the
        smooth surface around them (the flat end of a probe) like a click with *Pick contact
        faces* (its *Region angle*)."""
        import trimesh

        out: dict[str, list[int]] = {}
        if r <= 0 or self.s.tool_model is None:
            return out
        for hl in self.s.tool_model.links:
            V = np.asarray(hl.mesh.vertices) @ hl.T_head_link[:3, :3].T + hl.T_head_link[:3, 3]
            tri = V[np.asarray(hl.mesh.faces)]
            c = trimesh.triangles.closest_point(tri, np.repeat(np.asarray(p, float)[None], len(tri), axis=0))
            f = set(np.flatnonzero(np.linalg.norm(c - p, axis=1) <= r).tolist())
            if not f:
                continue
            from core.mesh_select import grow_region

            angle = float(self.tab.pick_angle.value)  # as a click on the tool would pick
            for seed in list(f):
                f |= set(grow_region(hl.mesh, seed, angle).tolist())
            out[hl.name] = sorted(int(i) for i in f)
        return out

    # --- widgets -------------------------------------------------------------------------------

    def refresh(self) -> None:
        ee = self.s.tool
        self.datums = load_datums(self.s.resolve(ee.urdf)) if ee is not None and ee.urdf else []
        names = tuple(d.name for d in self.datums)
        has = bool(names)
        for w in (
            self.tcp_datum,
            self.tcp_flip,
            self.contact_r,
            self.set_tcp,
            self.mount_datum,
            self.mount_flip,
            self.set_mount,
        ):
            w.visible = has
        if has:
            for w in (self.tcp_datum, self.mount_datum):
                keep = w.value if w.value in names else names[0]
                w.options = names
                w.value = keep
        self.info.content = (
            f"_{len(names)} coordinate system(s) in this tool's STEP file._"
            if has
            else ("_This tool has no STEP coordinate systems._" if ee is not None and ee.urdf else "")
        )
