"""Results -> station areas on the part (v4 plan item 3): the part surface painted by stop in
the station colours (the colours of the lines to the served points), groups (zones) of a stop
in lighter shades of its colour. A face takes the stop of the nearest target within 1.5 x
the point spacing of its centre; faces that hold a target always take that target's stop.
Drawn under /cell/part so it follows the part; on / off in the object tree."""

from __future__ import annotations

import numpy as np
import trimesh

PATH = "/cell/part/areas"
SHADES = (0.0, 0.35, 0.6)  # mixed towards white for the 2nd, 3rd .. group of a stop
OFFSET = 8e-4  # m above the surface (the selection is drawn at 5e-4)


_REFINED: dict = {}  # (id(mesh), spacing) -> (mesh, refined): display only


def display_mesh(mesh: trimesh.Trimesh, P: np.ndarray) -> trimesh.Trimesh:
    """The mesh, subdivided to about the point spacing when its triangles are larger (a coarse
    CAD export would otherwise be painted one whole face at a time). Cached per mesh."""
    if len(P) < 2:
        return mesh
    from scipy.spatial import cKDTree

    sp = float(np.median(cKDTree(P).query(P, k=2)[0][:, 1]))
    if sp <= 0 or mesh.edges_unique_length.max() <= sp:
        return mesh
    key = (id(mesh), round(sp, 6))
    hit = _REFINED.get(key)
    if hit is not None and hit[0] is mesh:
        return hit[1]
    from core.target_tools import refine_mesh

    try:
        fine, _ = refine_mesh(mesh, sp)
    except ValueError:  # too many faces: paint the mesh as it is
        fine = mesh
    _REFINED.clear()
    _REFINED[key] = (mesh, fine)
    return fine


def face_labels(mesh: trimesh.Trimesh, P: np.ndarray, point_label: np.ndarray) -> np.ndarray:
    """Per face: index into P of its target, or -1 (not painted)."""
    out = np.full(len(mesh.faces), -1)
    if len(P) == 0:
        return out
    from scipy.spatial import cKDTree

    tree = cKDTree(P)
    sp = float(np.median(tree.query(P, k=2)[0][:, 1])) if len(P) > 1 else float(mesh.scale) * 0.05
    d, j = tree.query(mesh.triangles_center, distance_upper_bound=1.5 * sp)
    near = np.isfinite(d)
    out[near] = j[near]
    _, _, tri = trimesh.proximity.closest_point(mesh, P)  # faces that hold a target
    out[tri] = np.arange(len(P))
    out[out >= 0] = np.where(point_label[out[out >= 0]] >= 0, out[out >= 0], -1)
    return out


def area_meshes(
    mesh: trimesh.Trimesh, ids, P: np.ndarray, sol, colors
) -> list[tuple[str, np.ndarray, np.ndarray, tuple]]:
    """(name, vertices, faces, colour) per (stop, group) of solution ``sol``."""
    stop = np.array([sol.assignment.get(int(i), -1) for i in ids])
    zone_of: dict[int, int] = {}
    for k in {z.station for z in sol.zones}:
        for j, z in enumerate([z for z in sol.zones if z.station == k]):
            for p in z.points:
                zone_of[int(p)] = j
    group = np.array([zone_of.get(int(i), 0) for i in ids])
    lab = face_labels(mesh, P, stop)
    out = []
    painted = lab >= 0
    keys = (
        np.stack([stop[lab[painted]], group[lab[painted]]], axis=1)
        if painted.any()
        else np.zeros((0, 2), int)
    )
    faces_idx = np.flatnonzero(painted)
    for k, g in sorted({(int(a), int(b)) for a, b in keys}):
        sel = faces_idx[(keys[:, 0] == k) & (keys[:, 1] == g)]
        sub = mesh.submesh([sel], append=True)
        base = np.array(colors[k % len(colors)], float)
        c = base + (255 - base) * SHADES[g % len(SHADES)]
        out.append(
            (
                f"S{k + 1}_G{g + 1}",
                sub.vertices + OFFSET * sub.vertex_normals,
                sub.faces,
                tuple(int(v) for v in c),
            )
        )
    return out
