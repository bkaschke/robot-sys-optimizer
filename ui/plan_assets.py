"""Asset files a worker session reads (review L5): their stamp makes replaced files rebuild it."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

_ASSET_KEYS = ("urdf", "mesh_path", "mesh", "collision_mesh")  # project fields naming a file
_MESH_REF = re.compile(r"""filename\s*=\s*["']([^"']+)["']""")


def _asset_files(project_json: str, project_dir: str) -> list[Path]:
    """Files the rebuilt session reads: the tool / robot URDFs with the meshes they name, part,
    carrier-body and stay-out meshes (resolved like ``Session.resolve``)."""
    from core import data_paths

    names: list[str] = []

    def walk(x) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if k in _ASSET_KEYS and isinstance(v, str) and v:
                    names.append(v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(json.loads(project_json))
    out: list[Path] = []
    for n in names:
        p = Path(n)
        f = (
            p
            if p.is_absolute()
            else next(
                (b / p for b in (Path(project_dir), *data_paths.asset_bases()) if (b / p).exists()), None
            )
        )
        if f is None or not f.is_file():
            continue
        out.append(f)
        if f.suffix.lower() == ".urdf":  # its meshes (relative to the URDF; package:// stripped)
            try:
                refs = _MESH_REF.findall(f.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                refs = []
            for r in refs:
                r = re.sub(r"^(package|file)://", "", r)
                for base in (f.parent, *f.parents):
                    if (base / r).is_file():
                        out.append(base / r)
                        break
    return out


def _asset_stamp(project_json: str, project_dir: str) -> bytes:
    """Review L5: mtime_ns and size of every asset file, so replaced files rebuild the session."""
    parts = []
    for f in _asset_files(project_json, project_dir):
        try:
            st = os.stat(f)
            parts.append(f"{f}|{st.st_mtime_ns}|{st.st_size}")
        except OSError:
            parts.append(f"{f}|missing")
    return "\n".join(parts).encode()
