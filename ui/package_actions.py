"""Package builder actions (split from ui.package_builder): check, export, use in the project,
open a zip, start a new package. ``self`` is the PackageBuilder."""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

from core.urdf_package import (
    Frame,
    Package,
    Part,
    check,
    collision_files,
    open_package,
    safe,
    to_base,
    to_tool,
    write_package,
    zip_folder,
)
from ui.widgets import MAX_UPLOAD, notify


def write(self, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    return write_package(self.pkg, self.work / "meshes", dest)


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
    self.import_bar.visible = True
    self.import_info.content = (
        f"_Importing **{f.name}**{' (meshing the STEP file)' if ext in ('.step', '.stp') else ''}…_"
    )

    def work() -> None:
        from core.carrier import primitive_mesh

        mesh_dir = self.work / "meshes"
        try:
            mesh_dir.mkdir(parents=True, exist_ok=True)
            if ext in (".step", ".stp"):
                from core.tool_import import step_mesh

                with tempfile.TemporaryDirectory() as tmp:
                    src = Path(tmp) / f"{name}{ext}"
                    src.write_bytes(content)
                    step_mesh(src).export(mesh_dir / f"{name}.stl")
                fname, u = f"{name}.stl", "m"
            else:
                fname, u = f"{name}{ext}", units
                (mesh_dir / fname).write_bytes(content)
            part = Part(name=name, stage=self._default_stage(), shape="mesh", mesh=fname, units=u)
            m = primitive_mesh(self.primitive(part))  # fail now on a broken file
        except Exception as err:  # a broken mesh / STEP file: report, keep the package
            why = f"{type(err).__name__}: {err}"[:300]
            self.import_errors[f.name] = why
            self.import_info.content = f"✘ **{f.name} was not added**: {why}"
            notify(e, "Mesh not added", why)
        else:
            self.import_errors.pop(f.name, None)
            self.import_info.content = f"✔ Added **{name}** ({len(m.faces)} triangles)."
            self._add(part)
        finally:
            self.import_bar.visible = False

    threading.Thread(target=work, daemon=True).start()


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


def use(self, e) -> None:
    dest = self.s.project_dir / "assets" / "packages" / safe(self.pkg.name)
    try:
        urdf = write(self, dest)
        if self.pkg.kind == "tool":
            self.s.set_tool(to_tool(self.pkg, self.s.relative(urdf)))
            if self.tool_tab is not None:
                self.tool_tab.refresh()
            where = "Tool tab: set the process and the extension ranges, then **Save to library**"
        else:
            base = to_base(
                self.pkg, self.s.relative(dest / "meshes"), self.s.base, collision_files(self.pkg, dest)
            )
            if self.carrier is not None:
                self.carrier.load(base)
                if not self.carrier.commit(e):
                    return
            else:
                self.s.base = base
            where = "External axis tab: set the standoffs and keep-outs, then plan"
    except (ValueError, OSError) as err:
        return notify(e, "Not loaded", str(err))
    self.status.content = f"Loaded **{self.pkg.name}** ({self.pkg.kind}). {where}."
    notify(e, "Package loaded", self.pkg.name, ok=True)


def open_zip(self, e) -> None:
    f = self.open.value
    if len(f.content) > MAX_UPLOAD:
        return notify(e, "Upload too large", "Limit is 200 MB")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pkg, urdf = open_package(f.content, Path(tmp) / "pkg")
            mesh_dir = self.work / "meshes"
            if mesh_dir.exists():
                shutil.rmtree(mesh_dir)
            mesh_dir.mkdir(parents=True)
            if (urdf.parent / "meshes").is_dir():
                shutil.copytree(urdf.parent / "meshes", mesh_dir, dirs_exist_ok=True)
    except Exception as err:  # malformed zips / URDFs: report, keep the current package
        return notify(e, "Could not open the package", f"{type(err).__name__}: {err}")
    self.pkg, self.sel, self.sel_tcp = pkg, 0, 0
    self.show.value = True
    self.load_widgets()
    self.status.content = f"Opened **{pkg.name}**: a {pkg.kind} with {len(pkg.parts)} part(s)."
    notify(e, "Package opened", pkg.name, ok=True)


def reset(self) -> None:
    kind = self.pkg.kind
    self.pkg = Package(kind=kind, name=f"my_{kind}", tcps=[Frame(name="tcp")] if kind == "tool" else [])
    self._set_kind(kind)
    self.status.content = ""
