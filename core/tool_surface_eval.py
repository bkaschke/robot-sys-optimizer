"""Tool-surface detection metrics, labelled-sample loading and two trivial baselines.

Sample format (fixed, so real data can follow the synthetic set)::

    <name>/mandrel.(stl|step)        the whole mandrel
    <name>/tool_surface.(stl|step)   the tool surface on its own

Labels come from proximity: a mandrel face is truth if its centroid is within a tolerance of
the tool_surface mesh and its normal agrees with the nearest tool_surface face's normal.
A detector is ``detect(mesh) -> bool mask per face``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from core.mesh_select import grow_region

STEP_EXTS = (".step", ".stp")
MESH_EXTS = (".stl", ".ply", ".obj", ".off")


def score(pred_mask, truth_mask, mesh: trimesh.Trimesh) -> dict:
    """Area-weighted IoU, precision and recall of ``pred_mask`` against ``truth_mask``."""
    pred, truth = np.asarray(pred_mask, bool), np.asarray(truth_mask, bool)
    if pred.shape != truth.shape or len(pred) != len(mesh.faces):
        raise ValueError(
            f"mask sizes differ: pred {pred.shape}, truth {truth.shape}, mesh has {len(mesh.faces)} faces"
        )
    a = mesh.area_faces
    inter = a[pred & truth].sum()
    union = a[pred | truth].sum()
    ap, at = a[pred].sum(), a[truth].sum()
    return {
        "iou": float(inter / union) if union > 0 else 1.0,
        "precision": float(inter / ap) if ap > 0 else (1.0 if at == 0 else 0.0),
        "recall": float(inter / at) if at > 0 else 1.0,
    }


def labels_from_proximity(
    mandrel: trimesh.Trimesh,
    tool_surface: trimesh.Trimesh,
    tol: float = 0.0005,
    edge_scale: float = 0.15,
    min_dot: float = 0.7,
) -> np.ndarray:
    """Truth mask on ``mandrel`` faces from the separately given tool-surface mesh.

    Distance tolerance = max(tol, edge_scale * median mandrel edge length); normals must agree.
    """
    if len(tool_surface.faces) == 0:
        raise ValueError("tool_surface mesh has no faces")
    edge = float(np.median(mandrel.edges_unique_length))
    eff = max(tol, edge_scale * edge)
    cen = mandrel.triangles_center
    closest, dist, tid = trimesh.proximity.closest_point(tool_surface, cen)
    agree = np.einsum("ij,ij->i", mandrel.face_normals, tool_surface.face_normals[tid]) > min_dot
    return (dist <= eff) & agree


def _load_mesh(path: Path, scale: float) -> trimesh.Trimesh:
    ext = path.suffix.lower()
    if ext in STEP_EXTS:  # STEP carries its own unit (read in metres); ``scale`` is for mesh files
        from core.step_import import StepImportError, read_step

        try:
            return read_step(path, "m").mesh
        except StepImportError as err:
            raise ValueError(str(err)) from err
    m = trimesh.load(path, force="mesh")
    if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
        raise ValueError(f"{path} did not load as a triangle mesh")
    if scale != 1.0:
        m.apply_scale(scale)
    return m


def _find(folder: Path, stem: str) -> Path:
    for ext in MESH_EXTS + STEP_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{folder}: no {stem}.(stl|step) found (sample format: mandrel + tool_surface)")


def load_sample(folder, scale: float = 1.0, tol: float = 0.0005):
    """Load ``<folder>/mandrel.*`` + ``tool_surface.*``; returns ``(mesh, truth_mask, info)``.

    ``scale`` converts file units to metres (0.001 for millimetre STL files); STEP files carry
    their own unit and are always read in metres.
    """
    folder = Path(folder)
    mandrel = _load_mesh(_find(folder, "mandrel"), scale)
    tool = _load_mesh(_find(folder, "tool_surface"), scale)
    truth = labels_from_proximity(mandrel, tool, tol=tol)
    info = {
        "name": folder.name,
        "n_faces": int(len(mandrel.faces)),
        "truth_faces": int(truth.sum()),
        "truth_area_m2": float(mandrel.area_faces[truth].sum()),
        "tool_surface_area_m2": float(tool.area),
        "watertight": bool(mandrel.is_watertight),
    }
    if not truth.any():
        raise ValueError(
            f"{folder}: no mandrel face lies on the tool_surface mesh; "
            "are the two files in the same frame and units?"
        )
    return mandrel, truth, info


def save_sample(folder, mesh: trimesh.Trimesh, truth) -> Path:
    """Write the two-file sample format (binary STL) from a mesh and a truth mask."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    mesh.export(folder / "mandrel.stl")
    sub = trimesh.Trimesh(mesh.vertices, mesh.faces[np.asarray(truth, bool)], process=False)
    sub.export(folder / "tool_surface.stl")
    return folder


# ---------------------------------------------------------------- baselines
def normal_up(mesh: trimesh.Trimesh) -> np.ndarray:
    """Every face with n.z > 0.5."""
    return mesh.face_normals[:, 2] > 0.5


def largest_smooth(mesh: trimesh.Trimesh, max_angle_deg: float = 30.0, up_only: bool = False) -> np.ndarray:
    """The largest (by area) region growable across edges with dihedral <= max_angle_deg.

    ``up_only`` ranks only regions whose area-weighted mean n.z is positive.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    adj = mesh.face_adjacency[mesh.face_adjacency_angles <= np.radians(max_angle_deg)]
    n = len(mesh.faces)
    g = coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    area = np.bincount(lab, weights=mesh.area_faces)
    if up_only:
        nz = np.bincount(lab, weights=mesh.area_faces * mesh.face_normals[:, 2])
        area = np.where(nz > 0, area, -1.0)
    seed = int(np.flatnonzero(lab == int(np.argmax(area)))[0])
    mask = np.zeros(n, bool)
    mask[grow_region(mesh, seed, max_angle_deg)] = True
    return mask


def largest_up(mesh: trimesh.Trimesh) -> np.ndarray:
    """largest_smooth restricted to up-facing regions (breaks the top/bottom tie)."""
    return largest_smooth(mesh, up_only=True)


BASELINES = {"normal_up": normal_up, "largest_smooth": largest_smooth, "largest_up": largest_up}
