"""STEP -> triangle mesh that remembers which CAD face every triangle came from.

``read_step(path)`` reads a STEP file with OCP (the OpenCascade bindings behind CadQuery, bundled
in the Windows build, nothing is downloaded), tessellates every B-rep face and returns a
``StepMesh``: a ``trimesh.Trimesh`` plus ``face_ids`` (per triangle: index of its CAD face) and
``face_types`` (per CAD face: "plane", "cylinder", "bspline", ...). The CAD faces are real region
boundaries, so a detector can select whole faces instead of guessing from dihedral angles.

Units: OCP always delivers STEP geometry in millimetres, whatever length unit the file declares
(a file in m or inch is converted by the reader). ``units`` is the unit of the *returned* mesh:
"m" (default, SI as everywhere in the code), "mm" or "in". The file itself is self-describing,
so a part's ``units`` setting (meant for STL / OBJ / PLY) is not applied to STEP.

Tolerance: chord deviation = min(0.5 mm, 0.1 % of the bounding-box diagonal), not below 0.02 mm;
angular deviation 0.2 rad (about 11 degrees). A 1 m part is meshed at 0.5 mm, a 100 mm tool
at 0.1 mm. If that gives more than ``MAX_TRIANGLES`` the chord deviation doubles (up to 4 times);
beyond that the import fails with a one-line message.

Tessellating big files is slow, so the result is cached as npz (vertices, faces, face_ids,
face_types, n_solids) in ``<collision cache>/step`` keyed by the file hash, tolerances and units.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from core.data_paths import cache_dir

STEP_EXTS = (".step", ".stp")
UNIT_SCALE_FROM_MM = {"mm": 1.0, "m": 1e-3, "in": 1.0 / 25.4}
MAX_TRIANGLES = 2_000_000
MAX_CHORD_MM = 0.5
MIN_CHORD_MM = 0.02
CHORD_FRACTION = 1e-3  # of the bounding-box diagonal
ANGULAR_RAD = 0.2


class StepImportError(ValueError):
    """One-line, user-readable reason a STEP file could not be turned into a mesh."""


@dataclass
class StepMesh:
    mesh: trimesh.Trimesh
    face_ids: np.ndarray  # (n_triangles,) CAD face index of every triangle
    face_types: list[str]  # per CAD face
    n_solids: int = 1
    linear_deflection_mm: float = 0.0
    from_cache: bool = False
    info: dict = field(default_factory=dict)

    @property
    def n_cad_faces(self) -> int:
        return len(self.face_types)


def default_deflection_mm(diagonal_mm: float) -> float:
    return float(min(MAX_CHORD_MM, max(MIN_CHORD_MM, CHORD_FRACTION * diagonal_mm)))


def _ocp():
    try:
        import OCP  # noqa: F401
    except ImportError as err:
        frozen = getattr(sys, "frozen", False)
        raise StepImportError(
            "STEP import is not included in this build; use an STL / OBJ / PLY mesh"
            if frozen
            else "STEP import needs CadQuery / OCP (pip install cadquery); STL / OBJ / PLY work without"
        ) from err


def _read_shape(path: Path):
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    try:
        status = reader.ReadFile(str(path))
    except Exception as err:
        raise StepImportError(f"{path.name}: not a readable STEP file ({type(err).__name__})") from err
    if status != IFSelect_RetDone:
        raise StepImportError(f"{path.name}: not a readable STEP file")
    try:
        if reader.TransferRoots() == 0:
            raise StepImportError(f"{path.name}: the STEP file holds no shapes")
        shape = reader.OneShape()
    except StepImportError:
        raise
    except Exception as err:
        raise StepImportError(f"{path.name}: could not read the STEP shapes ({type(err).__name__})") from err
    if shape.IsNull():
        raise StepImportError(f"{path.name}: the STEP file holds no shapes")
    return shape


def _count(shape, kind) -> int:
    from OCP.TopExp import TopExp_Explorer

    n, it = 0, TopExp_Explorer(shape, kind)
    while it.More():
        n += 1
        it.Next()
    return n


def _tessellate(shape, lin_mm: float, ang: float):
    """(vertices mm, triangles, face_ids, face_types) with duplicate vertices still unmerged."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedMapOfShape

    BRepMesh_IncrementalMesh(shape, lin_mm, False, ang, True)
    faces = TopTools_IndexedMapOfShape()  # unique faces (an explorer can repeat shared ones)
    TopExp.MapShapes_s(shape, TopAbs_FACE, faces)
    verts, tris, ids, types = [], [], [], []
    base = 0
    for i in range(1, faces.Extent() + 1):
        face = TopoDS.Face_s(faces.FindKey(i))
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        types.append(str(BRepAdaptor_Surface(face).GetType()).split(".")[-1].replace("GeomAbs_", "").lower())
        if tri is None:
            continue
        trsf = loc.Transformation()
        n_nodes = tri.NbNodes()
        P = np.empty((n_nodes, 3))
        for k in range(n_nodes):
            p = tri.Node(k + 1).Transformed(trsf)
            P[k] = (p.X(), p.Y(), p.Z())
        n_tri = tri.NbTriangles()
        T = np.empty((n_tri, 3), np.int64)
        for k in range(n_tri):
            T[k] = tri.Triangle(k + 1).Get()
        T -= 1
        if face.Orientation() == TopAbs_REVERSED:  # keep outward-facing winding
            T = T[:, ::-1]
        verts.append(P)
        tris.append(T + base)
        ids.append(np.full(n_tri, len(types) - 1, np.int64))
        base += n_nodes
    if not tris:
        raise StepImportError("the STEP file has no faces to mesh (surfaces or solids needed)")
    return np.vstack(verts), np.vstack(tris), np.concatenate(ids), types


def _diagonal_mm(shape) -> float:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    if box.IsVoid():
        raise StepImportError("the STEP file has no geometry")
    x0, y0, z0, x1, y1, z1 = box.Get()
    return float(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2))


def _cache_path(path: Path, lin, ang, units) -> Path:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    h.update(f"|{lin!r}|{ang!r}|{units}|v1".encode())
    return cache_dir("collision") / "step" / f"{h.hexdigest()}.npz"


def _from_arrays(V, F, ids, types, n_solids, lin, cached) -> StepMesh:
    mesh = trimesh.Trimesh(V, F, process=False)
    return StepMesh(mesh, ids, list(types), int(n_solids), float(lin), cached, {})


def read_step(
    path,
    units: str = "m",
    linear_deflection: float | None = None,
    angular_deflection: float = ANGULAR_RAD,
    use_cache: bool = True,
) -> StepMesh:
    """Read and tessellate a STEP file. ``linear_deflection`` is in millimetres (None = chosen
    from the model size, see the module docstring). Raises ``StepImportError`` (one line)."""
    path = Path(path)
    if units not in UNIT_SCALE_FROM_MM:
        raise StepImportError(f"unknown units {units!r}; use {', '.join(UNIT_SCALE_FROM_MM)}")
    if not path.is_file():
        raise StepImportError(f"{path.name}: file not found")
    _ocp()
    cache = None
    if use_cache:
        try:
            cache = _cache_path(path, linear_deflection, angular_deflection, units)
            if cache.is_file():
                with np.load(cache, allow_pickle=False) as z:
                    return _from_arrays(
                        z["vertices"],
                        z["faces"],
                        z["face_ids"],
                        z["face_types"],
                        z["n_solids"],
                        z["lin"],
                        True,
                    )
        except Exception:  # unreadable or corrupt cache entry: just rebuild
            pass
    shape = _read_shape(path)
    from OCP.TopAbs import TopAbs_SOLID

    n_solids = _count(shape, TopAbs_SOLID)
    lin = linear_deflection or default_deflection_mm(_diagonal_mm(shape))
    for _ in range(5):
        try:
            V, F, ids, types = _tessellate(shape, lin, angular_deflection)
        except StepImportError:
            raise
        except Exception as err:
            raise StepImportError(f"{path.name}: tessellation failed ({type(err).__name__})") from err
        if len(F) <= MAX_TRIANGLES:
            break
        lin *= 2.0
    else:
        raise StepImportError(f"{path.name}: more than {MAX_TRIANGLES} triangles even at a coarse tolerance")
    # faces meet at identical points (shared edge discretization): merge them so the mesh is closed
    keep = np.round(V / 1e-6).astype(np.int64)  # 1 nm grid in mm
    _, first, inverse = np.unique(keep, axis=0, return_index=True, return_inverse=True)
    V = V[first] * UNIT_SCALE_FROM_MM[units]
    F = inverse.reshape(-1)[F]
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    F, ids = F[ok], ids[ok]
    if len(F) == 0:
        raise StepImportError(f"{path.name}: the STEP file has no faces to mesh")
    out = _from_arrays(V, F, ids, types, n_solids, lin, False)
    out.info = {"solids": n_solids, "cad_faces": len(types), "triangles": len(F)}
    if cache is not None:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=cache.parent, suffix=".tmp.npz")
            os.close(fd)
            np.savez(
                tmp,
                vertices=V,
                faces=F,
                face_ids=ids,
                face_types=np.array(types, dtype="U"),
                n_solids=n_solids,
                lin=lin,
            )
            os.replace(tmp, cache)
        except OSError:
            pass
    return out
