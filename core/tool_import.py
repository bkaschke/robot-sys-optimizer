"""A tool from one mesh: an STL / OBJ / PLY file or a STEP model (Tool -> Library -> *Upload
tool mesh*).

The file becomes a one-link head (``head``) with a URDF written next to it, in metres:
``<dest>/head.stl`` + ``<dest>/<name>.urdf``. STEP models are tessellated with CadQuery (OCP
reads them in mm whatever the file's unit); their named coordinate systems
(core.step_datums) are saved as ``<dest>/datums.json`` (4x4 poses in the head frame, m) so the
GUI can put the TCP or the flange mount on one. The new tool is a *pose* tool with every
extension range at 0: the mesh sits directly on the flange mount (widen the ranges to let the
tool optimizer add a connector).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import trimesh

from core.schema import TCP, EndEffector, Extension, ExtensionValues
from core.step_datums import Datum, read_datums

MESH_EXTS = (".stl", ".obj", ".ply")
STEP_EXTS = (".step", ".stp")
UNIT_SCALE = {"m": 1.0, "mm": 1e-3, "in": 0.0254}
MAX_FACES = 2_000_000
MAX_EXTENT_M = 10.0  # a tool bigger than this is a unit mistake
DATUMS = "datums.json"


class ToolImportError(ValueError):
    pass


def step_mesh(path: Path, tolerance_mm: float = 0.1) -> trimesh.Trimesh:
    """Tessellate a STEP file (metres)."""
    import cadquery as cq

    shape = cq.importers.importStep(str(path)).val()
    verts, tris = shape.tessellate(tolerance_mm, 0.2)
    if not tris:
        raise ToolImportError("the STEP file has no solid faces to mesh")
    V = np.array([(v.x, v.y, v.z) for v in verts], float) / 1000.0  # OCP works in mm
    return trimesh.Trimesh(V, np.asarray(tris, np.int64), process=True)


def _urdf(name: str) -> str:
    return (
        f'<?xml version="1.0"?>\n<robot name="{name}">\n  <link name="head">\n'
        '    <visual><geometry><mesh filename="head.stl"/></geometry></visual>\n'
        '    <collision><geometry><mesh filename="head.stl"/></geometry></collision>\n'
        "  </link>\n</robot>\n"
    )


def import_mesh_tool(
    filename: str, data: bytes, dest: Path, name: str, units: str = "mm"
) -> tuple[Path, list[Datum]]:
    """Write the head mesh + URDF (+ STEP datums) into ``dest``; returns (URDF path, datums).
    ``units`` applies to STL / OBJ / PLY (a STEP file carries its own)."""
    ext = Path(filename).suffix.lower()
    dest.mkdir(parents=True, exist_ok=True)
    datums: list[Datum] = []
    if ext in STEP_EXTS:
        src = dest / f"source{ext}"
        src.write_bytes(data)
        try:
            mesh = step_mesh(src)
        except ToolImportError:
            raise
        except Exception as err:  # OCP raises its own types on a broken file
            raise ToolImportError(f"could not read the STEP file ({type(err).__name__})") from err
        datums = read_datums(data.decode("latin-1", errors="replace"))
    elif ext in MESH_EXTS:
        if units not in UNIT_SCALE:
            raise ToolImportError(f"unknown units {units!r}")
        try:
            mesh = trimesh.load(io.BytesIO(data), file_type=ext[1:], force="mesh")
        except Exception as err:
            raise ToolImportError(f"could not read the mesh ({type(err).__name__})") from err
        mesh.apply_scale(UNIT_SCALE[units])
    else:
        raise ToolImportError(f"use {', '.join(MESH_EXTS + STEP_EXTS)}")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ToolImportError("the file holds no triangles")
    if len(mesh.faces) > MAX_FACES:
        raise ToolImportError(f"{len(mesh.faces)} triangles: more than {MAX_FACES}")
    extent = float(np.max(mesh.extents))
    if extent > MAX_EXTENT_M:
        raise ToolImportError(f"the tool is {extent:.1f} m across: check the units")
    mesh.export(dest / "head.stl")
    path = dest / f"{name}.urdf"
    path.write_text(_urdf(name), encoding="utf-8")
    (dest / DATUMS).write_text(
        json.dumps([{"name": d.name, "T": np.asarray(d.T).tolist()} for d in datums], indent=1),
        encoding="utf-8",
    )
    return path, datums


def load_datums(urdf_path: Path | None) -> list[Datum]:
    """The STEP datums saved next to a mesh tool's URDF ([] when there are none)."""
    if urdf_path is None:
        return []
    f = Path(urdf_path).parent / DATUMS
    if not f.is_file():
        return []
    try:
        return [
            Datum(str(d["name"]), np.asarray(d["T"], float).reshape(4, 4)) for d in json.loads(f.read_text())
        ]
    except (ValueError, KeyError, TypeError):
        return []


def mesh_tool_ee(name: str, urdf_rel: str) -> EndEffector:
    """A pose tool whose head is the mesh, on the flange mount (every range 0), TCP at the
    head origin."""
    zero = (0.0, 0.0)
    x = Extension(type="pose", x=zero, y=zero, z=zero, nominal=ExtensionValues(z=0.0))
    return EndEffector(name=name, urdf=urdf_rel, tcps=[TCP(name="tcp")], extension=x)
