"""Package builder actions (split from ui.package_builder): check, export, use in the project,
open a zip, start a new package. ``self`` is the PackageBuilder."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

from core.safe_io import replace_dir
from core.urdf_package import (
    Frame,
    Package,
    Part,
    check,
    collision_files,
    safe,
    to_base,
    to_tool,
    write_package,
    zip_folder,
)
from ui.widgets import MAX_UPLOAD, notify


def write(self, dest: Path) -> Path:
    # built in a temp folder and swapped in: a failed export keeps the previous one
    replace_dir(dest, lambda tmp: write_package(self.pkg, self.work / "meshes", tmp))
    return dest / f"{safe(self.pkg.name)}.urdf"


def import_mesh(self, e) -> None:
    """Add the uploaded mesh as a part (v9 #2). STEP files are meshed with CadQuery, which takes
    seconds to minutes: that runs in a thread with a status line and a progress bar; a failure is
    shown there, kept for *Check*, and leaves the package as it was."""
    f = self.upload.value
    if len(f.content) > MAX_UPLOAD:
        return notify(e, "Upload too large", "Limit is 200 MB")
    ext = Path(f.name).suffix.lower()
    if ext not in (".step", ".stp", ".stl", ".obj", ".ply"):
        return notify(e, "Unsupported file", "Use STL, OBJ, PLY or STEP")
    name = safe(Path(f.name).stem)
    units = self.units.value
    content = f.content
    stop = threading.Event()  # v10 T20: Cancel import (STEP: the meshing process is ended)
    self.import_cancel.on_click(lambda _e: stop.set())
    self.import_cancel.visible = ext in (".step", ".stp")
    self.import_bar.visible = True
    self.import_info.content = (
        f"_Importing **{f.name}**{' (meshing the STEP file)' if ext in ('.step', '.stp') else ''}…_"
    )

    def work() -> None:
        from core.carrier import primitive_mesh

        mesh_dir = self.work / "meshes"
        try:
            mesh_dir.mkdir(parents=True, exist_ok=True)
            sure, guess_note = True, ""
            found: dict = {}
            if ext in (".step", ".stp"):
                from core.mesh_units import step_in_process

                with tempfile.TemporaryDirectory() as tmp:
                    src = Path(tmp) / f"{name}{ext}"
                    src.write_bytes(content)
                    step_in_process(src, mesh_dir / f"{name}.stl", stop.is_set)
                fname, u = f"{name}.stl", "m"
                from core.step_datums import read_datums
                from core.transforms import matrix_to_pose

                # v11 #10: its named coordinate systems, in the mesh's (file) frame
                found = {d.name: matrix_to_pose(d.T) for d in read_datums(content.decode("utf-8", "ignore"))}
            else:
                fname = f"{name}{ext}"
                (mesh_dir / fname).write_bytes(content)
                u, sure, guess_note = _units(mesh_dir / fname, units)
            part = Part(name=name, stage=self._default_stage(), shape="mesh", mesh=fname, units=u)
            if ext in (".step", ".stp") and found:
                part = part.model_copy(update={"datums": found})
            m = primitive_mesh(self.primitive(part))  # fail now on a broken file
        except Exception as err:  # a broken mesh / STEP file: report, keep the package
            from core.mesh_units import ImportCancelledError

            if isinstance(err, ImportCancelledError):
                self.import_info.content = f"_Import of **{f.name}** cancelled; nothing was added._"
                return
            why = f"{type(err).__name__}: {err}"[:300]
            self.import_errors[f.name] = why
            self.import_info.content = f"✘ **{f.name} was not added**: {why}"
            notify(e, "Mesh not added", why)
        else:
            self.import_errors.pop(f.name, None)
            n_d = len(part.datums)
            frames = f"; {n_d} STEP frame(s): *Frames and travel → Frame from STEP*" if n_d else ""
            self._add(part)
            from ui.package_load import after_import

            where = after_import(self, e, part, ext in (".step", ".stp"))  # v12 #27
            where = f"; {where}" if where else ""
            self.import_info.content = (
                f"✔ Added **{name}** ({len(m.faces)} triangles){guess_note}{where}{frames}."
            )
            if not sure:  # v10 T20: ask which units it is in
                ask_units(self, e, part.name)
        finally:
            self.import_bar.visible = False
            self.import_cancel.visible = False

    threading.Thread(target=work, daemon=True).start()


def _units(path: Path, chosen: str) -> tuple[str, bool, str]:
    """(units, sure, note) of a mesh file: the chosen units, or with *auto* a guess from its size."""
    from ui.package_builder import AUTO

    if chosen != AUTO:
        return chosen, True, ""
    import trimesh

    from core.mesh_units import describe, guess_units

    m = trimesh.load(str(path), force="mesh")
    ext = float(max(m.extents)) if len(m.faces) else 0.0
    u, sure = guess_units(ext)
    if sure:
        return (
            u,
            True,
            f"; units **{u}** from its size ({describe(ext, u)}; change with *Units of this mesh*)",
        )
    return u, False, f"; units **unsure**: taken as **{u}** ({describe(ext, u)}), pick them in the dialog"


def set_units(self, part_name: str, units: str) -> None:
    """The units of mesh part ``part_name`` (v10 T20)."""
    parts = [x.model_copy(update={"units": units}) if x.name == part_name else x for x in self.pkg.parts]
    self.pkg = self.pkg.model_copy(update={"parts": parts})
    self.load_widgets()
    self.import_info.content = f"✔ **{part_name}**: units set to **{units}**."


def ask_units(self, e, part_name: str) -> None:
    """A dialog: which units is the mesh in? (no browser client: the guess stays)."""
    client = getattr(e, "client", None)
    if client is None:
        return
    with client.gui.add_modal(f"Units of {part_name}?") as modal:
        client.gui.add_markdown("_Its size fits more than one unit. Which units is the file in?_")
        for u in ("mm", "m", "in"):
            b = client.gui.add_button(u)

            def pick(_e, u=u) -> None:
                set_units(self, part_name, u)
                modal.close()

            b.on_click(pick)


def check_package(self, e) -> None:
    problems = check(self.pkg) + [
        f"{f}: import failed ({why})" for f, why in getattr(self, "import_errors", {}).items()
    ]
    self.status.content = "✔ **Ready to export.**" if not problems else "✘ " + "  \n✘ ".join(problems)
    if problems:
        notify(e, "Package not ready", "; ".join(problems))


def export(self, e) -> None:
    try:
        urdf = write(self, self.s.project_dir / "export" / "packages" / safe(self.pkg.name))
    except (ValueError, OSError) as err:
        return notify(e, "Export failed", str(err))
    data = zip_folder(urdf.parent)
    zp = urdf.parent.parent / f"{safe(self.pkg.name)}.zip"
    zp.write_bytes(data)
    self.status.content = f"Saved `{zp}`"
    if getattr(e, "client", None) is not None:
        e.client.send_file_download(zp.name, data)
    notify(e, "Package exported", zp.name, ok=True)


def _tool_from(self, urdf_rel: str, urdf: Path | None = None):
    """The package as the project's tool. Editing the project's tool (``edit_source``) keeps its
    process and extension; the geometry, the TCP poses and (v10 #11) the contacts picked in the
    builder come from the package. v10 #1: contact parts / faces that the new URDF no longer has are
    dropped and listed in ``self.contact_lost``."""
    from core.tool_contacts import carry_contacts, link_faces

    self.contact_lost = []
    new = to_tool(self.pkg, urdf_rel)
    src = getattr(self, "edit_source", None)
    if not src or src[0] != "tool" or src[1] is None:
        if urdf is not None:  # v10 #11: picked faces must fit the written meshes
            tcps, self.contact_lost = carry_contacts(new.tcps, None, link_faces(new, urdf))
            new = new.model_copy(update={"tcps": tcps})
        return new
    old = src[1]
    by_name = {safe(t.name): t for t in old.tcps}
    keep = ("T_head_tcp", "contact_links", "contact_faces", "min_radius_mm", "radius_applies")
    keep += ("travel", "scan_width_mm", "overlap_pct", "overrun_mm")  # v11 #7
    tcps = [
        by_name[t.name].model_copy(update={k: getattr(t, k) for k in keep}) if t.name in by_name else t
        for t in new.tcps
    ]
    if urdf is not None:
        tcps, self.contact_lost = carry_contacts(
            tcps, getattr(self, "edit_faces", None), link_faces(new, urdf)
        )
    return old.model_copy(update={"urdf": urdf_rel, "tcps": tcps})


def use(self, e) -> bool:
    """The package into the project (True when it was loaded)."""
    from ui.package_edit import use_part

    dest = self.s.project_dir / "assets" / "packages" / safe(self.pkg.name)
    try:
        urdf = write(self, dest)
        if self.pkg.kind == "tool":
            self.s.set_tool(_tool_from(self, self.s.relative(urdf), urdf))
            if self.tool_tab is not None:
                self.tool_tab.refresh()
            where = "Tool tab: set the process and the extension ranges, then **Save to library**"
            if self.contact_lost:  # v10 #1
                from ui.log_panel import log

                for msg in self.contact_lost:
                    log(f"Tool contact not carried over: {msg}", "warn")
                where += ".  \n⚠ **Contact settings not carried over:** " + "; ".join(self.contact_lost)
        elif self.pkg.kind == "part":
            where = use_part(self, dest)
        else:
            base = to_base(
                self.pkg, self.s.relative(dest / "meshes"), self.s.base, collision_files(self.pkg, dest)
            )
            if self.carrier is not None:
                self.carrier.load(base)
                if not self.carrier.commit(e):
                    return False
            else:
                self.s.base = base
            where = "External axis tab: set the standoffs and keep-outs, then plan"
    except (ValueError, OSError) as err:
        notify(e, "Not loaded", str(err))
        return False
    self.status.content = f"Loaded **{self.pkg.name}** ({self.pkg.kind}). {where}."
    notify(e, "Package loaded", self.pkg.name, ok=True)
    return True


def open_zip(self, e) -> None:
    from ui.package_edit import load_zip

    f = self.open.value
    if len(f.content) > MAX_UPLOAD:
        return notify(e, "Upload too large", "Limit is 200 MB")
    try:
        pkg = load_zip(self, f.content)
    except Exception as err:  # malformed zips / URDFs: report, keep the current package
        return notify(e, "Could not open the package", f"{type(err).__name__}: {err}")
    self.edit_source = None
    self.status.content = f"Opened **{pkg.name}**: a {pkg.kind} with {len(pkg.parts)} part(s)."
    notify(e, "Package opened", pkg.name, ok=True)


def reset(self) -> None:
    self.edit_source = None
    kind = self.pkg.kind
    self.pkg = Package(kind=kind, name=f"my_{kind}", tcps=[Frame(name="tcp")] if kind == "tool" else [])
    self._set_kind(kind)
    self.status.content = ""
