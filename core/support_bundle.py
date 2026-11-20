"""Support bundle: one zip to send when something goes wrong on a PC nobody can log in to.

Contents: ``info.txt`` (program version, Python, operating system, memory, package versions,
folders), ``panel_log.txt`` (the messages of the GUI's log window), ``logs/`` (the log file and
its rotated copies), ``project/project.json`` of the open study, and for a batch the
``batch.json`` plus every job's ``error.txt``. **No part, tool or carrier meshes** are included,
and nothing is sent anywhere: the zip is written to ``<data folder>/support/`` for the user to
pass on. Every piece is optional: what cannot be read is listed in ``info.txt``.
"""

from __future__ import annotations

import io
import os
import platform
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

from core import app_log, data_paths

PACKAGES = (
    "viser",
    "numpy",
    "scipy",
    "trimesh",
    "pinocchio",
    "coal",
    "numba",
    "pydantic",
    "shapely",
    "ortools",
    "cma",
    "cadquery",
    "yourdfpy",
    "rtree",
    "manifold3d",
)
MAX_FILE = 5_000_000  # a larger file is cut to its last MAX_FILE bytes


def _memory() -> str:
    try:
        import psutil

        m = psutil.virtual_memory()
        return f"{m.total / 2**30:.1f} GB total, {m.available / 2**30:.1f} GB free"
    except Exception:  # noqa: BLE001 - optional
        return "unknown"


def _versions() -> list[str]:
    from importlib import metadata

    out = []
    for name in PACKAGES:
        try:
            out.append(f"  {name} {metadata.version(name)}")
        except Exception:  # noqa: BLE001 - a conda package or a frozen build without metadata
            v = getattr(sys.modules.get(name), "__version__", None)  # only if already loaded
            out.append(f"  {name} {v}" if v else f"  {name} (version not recorded)")
    return out


def system_info(project_dir: Path | None = None, notes: list[str] | None = None) -> str:
    from core.version import app_version

    rows = [
        f"Robot Cell Optimizer {app_version()}",
        f"Bundle made: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Python: {sys.version.split()[0]} ({'installed program' if data_paths.is_frozen() else 'source'})",
        f"System: {platform.platform()} ({platform.machine()})",
        f"CPUs: {os.cpu_count()}",
        f"Memory: {_memory()}",
        f"Program folder: {data_paths.resources()}",
        f"Data folder: {data_paths.data_dir()}",
        f"Workspace: {data_paths.workspace_dir()}",
        f"Open study: {project_dir if project_dir else '-'}",
        "Settings from the environment: "
        + (", ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("RSO_")) or "none"),
        "Packages:",
        *_versions(),
    ]
    if notes:
        rows += ["", "Not included:", *(f"  {n}" for n in notes)]
    return "\n".join(rows) + "\n"


def _read(path: Path) -> bytes:
    size = path.stat().st_size
    with open(path, "rb") as f:
        if size > MAX_FILE:
            f.seek(size - MAX_FILE)
        return f.read()


def panel_text(entries) -> str:
    """Log-panel entries [(time, level, message)] as text, oldest first."""
    return "".join(
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))} {level:<5} {msg}\n"
        for t, level, msg in entries
    )


def _batch_dir(project_dir: Path | None) -> Path | None:
    """The batch folder a job study lives in (its parent has batch.json), else None."""
    if project_dir is None:
        return None
    for d in (Path(project_dir), Path(project_dir).parent):
        if (d / "batch.json").is_file():
            return d
    return None


def collect(project_dir: Path | None = None, batch_dir: Path | None = None) -> list[tuple[str, Path]]:
    """(name in the zip, file) of everything that exists. Meshes are never listed."""
    items = [(f"logs/{p.name}", p) for p in app_log.log_files()]
    if project_dir is not None:
        p = Path(project_dir) / "project.json"
        if p.is_file():
            items.append(("project/project.json", p))
    recent = data_paths.last_project_file()
    if recent.is_file():
        items.append(("workspace/last_project.txt", recent))
    b = batch_dir or _batch_dir(project_dir)
    if b is not None and (Path(b) / "batch.json").is_file():
        items.append(("batch/batch.json", Path(b) / "batch.json"))
        for err in sorted(Path(b).glob("*/error.txt")):
            items.append((f"batch/{err.parent.name}/error.txt", err))
    return items


def make_bundle(
    project_dir: Path | None = None,
    panel_entries=(),
    out_dir: Path | None = None,
    batch_dir: Path | None = None,
) -> tuple[Path, bytes]:
    """Write ``support_<date>_<time>.zip`` and return (path, bytes). Never raises for a piece
    that cannot be read (it is listed in info.txt); OSError only when the zip cannot be written."""
    notes: list[str] = []
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("panel_log.txt", panel_text(panel_entries) or "(no messages)\n")
        try:
            items = collect(project_dir, batch_dir)
        except OSError as err:
            items, _ = [], notes.append(f"files could not be listed ({err})")
        for name, path in items:
            try:
                z.writestr(name, _read(path))
            except OSError as err:
                notes.append(f"{name} ({err})")
        if not any(n.startswith("logs/") for n, _ in items):
            notes.append("logs/ (no log file yet: it is written by the running program, not by scripts)")
        z.writestr("info.txt", system_info(project_dir, notes))
    out = Path(out_dir) if out_dir else data_paths.data_dir() / "support"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"support_{datetime.now():%Y%m%d_%H%M%S}.zip"
    path.write_bytes(buf.getvalue())
    return path, buf.getvalue()
