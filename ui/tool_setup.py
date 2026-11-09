"""Tool Setup tab: tool library, process preset, flange mount, extension, TCPs, contact
faces and mass. Edits go through ``_commit`` which re-validates the whole EndEffector."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import viser
from pydantic import ValidationError

from core.mesh_select import grow_region
from core.process import PRESETS, apply_preset
from core.schema import TCP, EndEffector, Extension, Tolerances
from core.tool import list_tools, save_tool
from core.transforms import matrix_to_pose, pose_to_matrix
from core.urdf_import import URDFImportError, import_urdf_zip
from ui import tool_extension
from ui.session import REPO, Session
from ui.tool_view import HEAD, ROOT, ToolView
from ui.widgets import MAX_UPLOAD, PoseWidgets, notify, safe_name

NONE = "(none)"


class ToolSetupTab:
    def __init__(self, server: viser.ViserServer, session: Session):
        self.server = server
        self.s = session
        self.view = ToolView(server, session)
        self.view.on_pick = self._on_pick
        self._busy = False
        self._dyn: list = []  # widgets rebuilt per tool (sliders, link checkboxes)
        self._nominal: dict = {}
        self._link_boxes: dict = {}
        self.tcp_choice_value: str | None = None  # TCP to keep selected across refreshes
        gui = server.gui

        with gui.add_folder("Library"):
            self.choice = gui.add_dropdown("Tool", (NONE, *list_tools()), initial_value=NONE)
            self.upload = gui.add_upload_button("Upload tool URDF (.zip)", mime_type="application/zip")
            self.name = gui.add_text("Name", "")
            self.save = gui.add_button("Save to library")
        with gui.add_folder("Process"):
            self.process = gui.add_dropdown("Type", tuple(PRESETS), initial_value="custom")
            self.notes = gui.add_markdown("")
            self.force = gui.add_number("Process force (N)", 0.0, min=0.0, step=1.0)
            self.tilt = gui.add_number("Tilt cone (deg)", 0.0, min=0.0, max=90.0, step=0.5)
            self.standoff = gui.add_vector2("Standoff min/max (mm)", (0.0, 0.0), step=1.0)
            self.spin = gui.add_text("Spin (deg)", "free", hint="'free', a half-width, or 'lo,hi'")
        with gui.add_folder("Flange mount"):
            self.mount = PoseWidgets(server, f"{ROOT}/mount", np.eye(4), self._on_mount, scale=0.12)
        with gui.add_folder("Extension"):
            tool_extension.build(gui, self)  # pose (x y z rx ry rz) or legacy straight / bent
        with gui.add_folder("TCPs"):
            self.tcp_choice = gui.add_dropdown("TCP", ("tcp",))
            self.tcp_name = gui.add_text("Rename", "")
            self.tcp_add = gui.add_button("Add TCP")
            self.tcp_del = gui.add_button("Remove TCP")
            self.tcp_lock = gui.add_checkbox(
                "Lock TCP position", True, hint="untick to move the TCP (fields or gizmo)"
            )
            self.tcp_pose = PoseWidgets(server, f"{HEAD}/tcp_gizmo", np.eye(4), self._on_tcp_pose, scale=0.08)
            self.tcp_lock.on_update(lambda _e: self._apply_tcp_lock())
            self._apply_tcp_lock()
            self.pick = gui.add_checkbox("Pick contact faces (click tool)", False)
            self.pick_angle = gui.add_slider("Region angle (deg)", 1.0, 45.0, 1.0, 15.0)
            self.clear_faces = gui.add_button("Clear contact faces")
            self.links_folder = gui.add_folder("Whole-link contact")
            self.contact_info = gui.add_markdown("")
        with gui.add_folder("Mass"):
            self.mass_info = gui.add_markdown("")
            self.mass_override = gui.add_checkbox("Override head mass/COM", False)
            self.mass = gui.add_number("Head mass (kg)", 0.0, min=0.0, step=0.1)
            self.com = gui.add_vector3("Head COM (mm)", (0.0, 0.0, 0.0), step=1.0)
        self._wire()
        self.refresh()

    # --- editing core -----------------------------------------------------------------------

    def _commit(self, fn: Callable[[dict], None], event=None) -> bool:
        """Apply ``fn`` to a dict of the tool, re-validate, and update model + scene."""
        if self.s.tool is None:
            return False
        data = self.s.tool.model_dump(mode="python")
        fn(data)
        try:
            ee = EndEffector.model_validate(data)
        except ValidationError as err:
            notify(event, "Invalid tool setting", str(err.errors()[0]["msg"]))
            self.refresh()  # put widgets back to the valid state
            return False
        self.s.tool = ee
        self.s.tool_model.ee = ee
        return True

    def _after(self, full: bool = False) -> None:
        if full:
            self.refresh()
        else:
            self.view.update()
            self._update_info()

    # --- callbacks ----------------------------------------------------------------------------

    def _guard(self, fn):
        def wrapped(e) -> None:
            if not self._busy:
                fn(e)

        return wrapped

    def _wire(self) -> None:
        g = self._guard

        @self.choice.on_update
        @g
        def _(e) -> None:
            if self.choice.value == NONE:
                self.s.set_tool(None)
            else:
                try:
                    self.s.load_tool_from_library(self.choice.value)
                except (OSError, ValueError) as err:
                    return notify(e, "Tool load failed", str(err))
            self.refresh()

        self.upload.on_upload(g(self._on_upload))

        @self.save.on_click
        def _(e) -> None:
            self._save(e)

        @self.process.on_update
        @g
        def _(e) -> None:
            if self.s.tool is not None:
                self.s.tool = apply_preset(self.s.tool, self.process.value)
                self.s.tool_model.ee = self.s.tool
                self._after(full=True)

        def tol(d: dict) -> None:
            lo, hi = (v / 1000 for v in self.standoff.value)
            spin = self.spin.value.strip()
            if spin != "free":
                parts = [float(x) for x in spin.split(",")]
                spin = parts[0] if len(parts) == 1 else tuple(parts)
            d["tolerances"] = Tolerances(
                spin_deg=spin, tilt_deg=self.tilt.value, standoff_m=lo if lo == hi else (lo, hi)
            ).model_dump()
            d["process_force_N"] = self.force.value

        for w in (self.force, self.tilt, self.standoff, self.spin):

            @w.on_update
            @g
            def _(e) -> None:
                try:
                    ok = self._commit(tol, e)
                except ValueError:
                    return notify(e, "Spin", "use 'free', a number, or 'lo,hi'")
                if ok:
                    self._update_info()

        def ext(d: dict) -> None:
            tool_extension.commit(self, d)

        for w in tool_extension.widgets(self):

            @w.on_update
            @g
            def _(e) -> None:
                if self._commit(ext, e):
                    self._after(full=True)  # slider bounds may have changed

        @self.tcp_choice.on_update
        @g
        def _(e) -> None:
            self._show_tcp()

        @self.tcp_name.on_update
        @g
        def _(e) -> None:
            new, old = self.tcp_name.value.strip(), self.tcp_choice.value
            if new and new != old and self._commit(lambda d: self._rename_tcp(d, old, new), e):
                self.tcp_choice_value = new
                self._after(full=True)

        @self.tcp_add.on_click
        def _(e) -> None:
            names = {t.name for t in self.s.tool.tcps} if self.s.tool else set()
            name = next(f"tcp{i}" for i in range(1, 100) if f"tcp{i}" not in names)
            if self._commit(lambda d: d["tcps"].append(TCP(name=name).model_dump()), e):
                self.tcp_choice_value = name
                self._after(full=True)

        @self.tcp_del.on_click
        def _(e) -> None:
            cur = self.tcp_choice.value
            if self._commit(lambda d: d.update(tcps=[t for t in d["tcps"] if t["name"] != cur]), e):
                self._after(full=True)

        @self.clear_faces.on_click
        def _(e) -> None:
            cur = self.tcp_choice.value
            if self._commit(lambda d: self._tcp(d, cur).update(contact_faces={}), e):
                self._after()

        @self.pick.on_update
        def _(e) -> None:
            self.view.on_pick = self._on_pick if self.pick.value else None

        def mass(d: dict) -> None:
            on = self.mass_override.value
            d["mass_kg"] = self.mass.value if on else None
            d["com_head"] = tuple(v / 1000 for v in self.com.value) if on else None

        for w in (self.mass_override, self.mass, self.com):

            @w.on_update
            @g
            def _(e) -> None:
                if self._commit(mass, e):
                    self._update_info()

        self.view.on_pick = None

    @staticmethod
    def _tcp(d: dict, name: str) -> dict:
        return next(t for t in d["tcps"] if t["name"] == name)

    @staticmethod
    def _rename_tcp(d: dict, old: str, new: str) -> None:
        ToolSetupTab._tcp(d, old)["name"] = new

    def _on_mount(self, T: np.ndarray) -> None:
        if not self._busy and self._commit(lambda d: d.update(T_flange_mount=matrix_to_pose(T).model_dump())):
            self._after()

    def _apply_tcp_lock(self) -> None:
        """Locked (the default): the TCP pose fields are read-only and the gizmo has no handles."""
        self.tcp_pose.set_enabled(not self.tcp_lock.value)

    def _on_tcp_pose(self, T: np.ndarray) -> None:
        cur = self.tcp_choice.value
        if self.tcp_lock.value:  # locked: put the fields / gizmo back to the stored pose
            if not self._busy and self.s.tool is not None:
                self._show_tcp()
            return
        if not self._busy and self._commit(
            lambda d: self._tcp(d, cur).update(T_head_tcp=matrix_to_pose(T).model_dump())
        ):
            self._after()

    def _on_nominal(self, key: str, scale: float):
        def handler(e) -> None:
            if self._busy:
                return
            val = self._nominal[key].value * scale
            if self._commit(lambda d: d["extension"]["nominal"].update({key: val}), e):
                self._after()

        return handler

    def _on_link_contact(self, link: str):
        def handler(e) -> None:
            if self._busy:
                return
            cur, on = self.tcp_choice.value, self._link_boxes[link].value

            def fn(d: dict) -> None:
                t = self._tcp(d, cur)
                links = set(t["contact_links"])
                links.add(link) if on else links.discard(link)
                t["contact_links"] = sorted(links)

            if self._commit(fn, e):
                self._after()

        return handler

    def _on_pick(self, link: str, face: int, event) -> None:
        region = grow_region(self.s.tool_model.link(link).mesh, face, self.pick_angle.value)
        cur = self.tcp_choice.value

        def fn(d: dict) -> None:
            faces = self._tcp(d, cur)["contact_faces"]
            have = set(faces.get(link, []))
            have = have - set(region.tolist()) if face in have else have | set(region.tolist())
            faces[link] = sorted(have)
            if not faces[link]:
                del faces[link]

        if self._commit(fn, event):
            self._after()

    def _on_upload(self, e) -> None:
        f = self.upload.value
        if len(f.content) > MAX_UPLOAD:
            return notify(e, "Upload too large", "Limit is 200 MB")
        name = safe_name(Path(f.name).stem)
        dest = self.s.project_dir / "assets" / "tools" / name
        try:
            info = import_urdf_zip(f.content, dest, require_actuated=False)
            ee = EndEffector(  # new tools use the v2 pose extension (head pose with ranges)
                name=name,
                urdf=self.s.relative(info.path),
                tcps=[TCP(name="tcp")],
                extension=Extension(type="pose"),
            )
            self.s.set_tool(ee)
        except (URDFImportError, ValueError, OSError) as err:
            return notify(e, "Tool import failed", str(err))
        self.refresh()
        notify(e, "Tool loaded", f"{name}: place the TCP, then save to the library", ok=True)

    def _save(self, e) -> None:
        ee = self.s.tool
        if ee is None:
            return notify(e, "No tool", "Load or upload a tool first")
        name = self.name.value.strip() or ee.name
        try:
            urdf = self.s.resolve(ee.urdf) if ee.urdf else None
            if urdf is not None and not urdf.is_relative_to(REPO / "assets" / "tools"):
                # copy uploaded geometry into the repo so the library entry is self-contained
                dest = REPO / "assets" / "tools" / safe_name(name)
                shutil.copytree(urdf.parent, dest, dirs_exist_ok=True)
                ee = ee.model_copy(update={"urdf": (dest / urdf.name).relative_to(REPO).as_posix()})
            path = save_tool(ee.model_copy(update={"name": name}), name)
        except (OSError, ValueError) as err:
            return notify(e, "Save failed", str(err))
        self.s.tool = ee.model_copy(update={"name": name, "tool": path.relative_to(REPO).as_posix()})
        self.s.tool_model.ee = self.s.tool
        self._busy = True
        try:
            self.choice.options = (NONE, *list_tools())
            self.choice.value = name
        finally:
            self._busy = False
        notify(e, "Saved", str(path.relative_to(REPO)), ok=True)

    # --- widget refresh -------------------------------------------------------------------------

    def on_robot_moved(self, T_world_flange: np.ndarray) -> None:
        self.view.set_flange(T_world_flange)

    def refresh(self) -> None:
        self._busy = True
        try:
            self._refresh()
        finally:
            self._busy = False

    def _refresh(self) -> None:
        ee = self.s.tool
        for h in self._dyn:
            h.remove()
        self._dyn, self._nominal, self._link_boxes = [], {}, {}
        self.view.active_tcp = None
        if ee is None:
            self.choice.value = NONE
            self.view.rebuild()
            self.mount.gizmo.visible = self.tcp_pose.gizmo.visible = False  # no tool: nothing below them
            self.mass_info.content = self.contact_info.content = "_No tool loaded._"
            return
        self.choice.value = ee.name if ee.name in list_tools() else NONE
        self.name.value = ee.name
        self.process.value = ee.process
        self.notes.content = f"_{PRESETS[ee.process].notes}_"
        self.force.value = ee.process_force_N
        tol = ee.tolerances
        self.tilt.value = tol.tilt_deg
        so = tol.standoff_m if isinstance(tol.standoff_m, tuple) else (tol.standoff_m, tol.standoff_m)
        self.standoff.value = tuple(v * 1000 for v in so)
        sp = tol.spin_deg
        self.spin.value = (
            sp if sp == "free" else (",".join(map(str, sp)) if isinstance(sp, tuple) else str(sp))
        )
        self.mount.set(pose_to_matrix(ee.T_flange_mount))
        tool_extension.refresh(self, ee.extension, self._on_nominal)
        names = [t.name for t in ee.tcps]
        cur = self.tcp_choice_value
        self.tcp_choice.options = tuple(names)
        self.tcp_choice.value = cur if cur in names else names[0]
        with self.links_folder:
            for hl in self.s.tool_model.links:
                cb = self.server.gui.add_checkbox(hl.name, False)
                cb.on_update(self._on_link_contact(hl.name))
                self._link_boxes[hl.name] = cb
                self._dyn.append(cb)
        self.mass_override.value = ee.mass_kg is not None
        self.mass.value = ee.mass_kg or 0.0
        self.com.value = tuple(v * 1000 for v in (ee.com_head or (0.0, 0.0, 0.0)))
        self.mount.gizmo.visible = True
        self.mount.set_handles(self.mount.show.value)
        self.view.rebuild()
        self._show_tcp()

    def _show_tcp(self) -> None:
        tcp = next(t for t in self.s.tool.tcps if t.name == self.tcp_choice.value)
        self.tcp_choice_value = tcp.name
        busy, self._busy = self._busy, True
        try:
            self.tcp_name.value = tcp.name
            self.tcp_pose.set(pose_to_matrix(tcp.T_head_tcp))
            self.tcp_pose.gizmo.visible = True
            self.tcp_pose.set_handles(self.tcp_pose.show.value and not self.tcp_lock.value)
            for link, cb in self._link_boxes.items():
                cb.value = link in tcp.contact_links
        finally:
            self._busy = busy
        self.view.active_tcp = tcp.name
        self.view.update()
        self._update_info()

    def _update_info(self) -> None:
        tm, ee = self.s.tool_model, self.s.tool
        if tm is None:
            return
        mp = tm.mass_properties()
        warn = "" if mp.head_mass_known else "  \n⚠ URDF has no inertials: set the head mass."
        c = mp.com_flange * 1000
        self.mass_info.content = (
            f"**Total {mp.mass_kg:.2f} kg** (head + extension {tm.length() * 1000:.0f} mm)  \n"
            f"COM in flange: {c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f} mm{warn}"
        )
        tcp = next((t for t in ee.tcps if t.name == self.tcp_choice.value), None)
        if tcp is not None:
            n = sum(len(v) for v in tcp.contact_faces.values())
            links = ", ".join(tcp.contact_links) or "none"
            hint = (
                " (contact-free process: no faces may touch)" if PRESETS[ee.process].contact == "none" else ""
            )
            self.contact_info.content = f"Contact: whole links {links}; {n} picked faces{hint}"
