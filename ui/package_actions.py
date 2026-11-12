"""Package builder actions (split from ui.package_builder): check, export, use in the project,
open a zip, start a new package. ``self`` is the PackageBuilder."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from core.urdf_package import (
    Frame,
    Package,
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


def check_package(self, e) -> None:
    problems = check(self.pkg)
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
