"""Results -> station areas on the part (v4 plan item 3): the part surface painted by stop in
the station colours (the colours of the lines to the served points), groups (zones) of a stop
in lighter shades of its colour. Without zones, each joint configuration of a stop is its own
group (v6: the paths do a stop one configuration at a time). A face takes the stop of the
nearest target within 1.5 x the point spacing of its centre (coarse meshes are subdivided to
the spacing first, so every target has faces around it).
Drawn under /cell/part so it follows the part; on / off in the object tree."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import trimesh

PATH = "/cell/part/areas"
SHADES = (0.0, 0.35, 0.6)  # mixed towards white for the 2nd, 3rd .. group of a stop
OFFSET = 8e-4  # m above the surface (the selection is drawn at 5e-4)


REFINED_KEEP = 4  # refined display meshes kept (a few parts / spacings: switching back is free)
_REFINED: OrderedDict = OrderedDict()  # (id(mesh), spacing) -> (mesh, refined): display only


def display_mesh(mesh: trimesh.Trimesh, P: np.ndarray) -> trimesh.Trimesh:
    """The mesh, subdivided to about the point spacing when its triangles are larger (a coarse
    CAD export would otherwise be painted one whole face at a time). The last ``REFINED_KEEP``
    (mesh, spacing) pairs are cached (least recently used dropped first)."""
    if len(P) < 2:
        return mesh
    from scipy.spatial import cKDTree

    sp = float(np.median(cKDTree(P).query(P, k=2)[0][:, 1]))
    if sp <= 0 or mesh.edges_unique_length.max() <= sp:
        return mesh
    key = (id(mesh), round(sp, 6))
    hit = _REFINED.get(key)
    if hit is not None and hit[0] is mesh:
        _REFINED.move_to_end(key)
        return hit[1]
    from core.target_tools import refine_mesh

    try:
        fine, _ = refine_mesh(mesh, sp)
    except ValueError:  # too many faces: paint the mesh as it is
        fine = mesh
    _REFINED[key] = (mesh, fine)  # holds ``mesh``, so its id cannot be reused while cached
    while len(_REFINED) > REFINED_KEEP:
        _REFINED.popitem(last=False)
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
    out[out >= 0] = np.where(point_label[out[out >= 0]] >= 0, out[out >= 0], -1)
    return out


def config_groups(sol, config_of: dict[int, str]) -> dict[int, int]:
    """Point id -> group index at its stop by joint configuration (largest group first, as
    core.paths.sequences orders them)."""
    by: dict[int, dict[str, list[int]]] = {}
    for pid, k in sol.assignment.items():
        if int(pid) in config_of:
            by.setdefault(k, {}).setdefault(config_of[int(pid)], []).append(int(pid))
    out: dict[int, int] = {}
    for groups in by.values():
        for j, pids in enumerate(sorted(groups.values(), key=len, reverse=True)):
            out.update(dict.fromkeys(pids, j))
    return out


def _stop_groups(mesh, ids, P: np.ndarray, sol, config_of: dict[int, str] | None):
    """(painted face indices, (stop, group) per painted face) of solution ``sol``."""
    stop = np.array([sol.assignment.get(int(i), -1) for i in ids])
    zone_of: dict[int, int] = {}
    for k in {z.station for z in sol.zones}:
        for j, z in enumerate([z for z in sol.zones if z.station == k]):
            for p in z.points:
                zone_of[int(p)] = j
    if not sol.zones and config_of:
        zone_of = config_groups(sol, config_of)
    group = np.array([zone_of.get(int(i), 0) for i in ids])
    lab = face_labels(mesh, P, stop)
    painted = lab >= 0
    keys = (
        np.stack([stop[lab[painted]], group[lab[painted]]], axis=1)
        if painted.any()
        else np.zeros((0, 2), int)
    )
    return np.flatnonzero(painted), keys


def group_colour(k: int, g: int, colors) -> tuple[int, int, int]:
    """Stop ``k``'s station colour, mixed towards white for its 2nd, 3rd .. group."""
    base = np.array(colors[k % len(colors)], float)
    c = base + (255 - base) * SHADES[g % len(SHADES)]
    return tuple(int(v) for v in c)


def area_meshes(
    mesh: trimesh.Trimesh, ids, P: np.ndarray, sol, colors, config_of: dict[int, str] | None = None
) -> list[tuple[str, np.ndarray, np.ndarray, tuple]]:
    """(name, vertices, faces, colour) per (stop, group) of solution ``sol``. Groups are the
    zones, or without zones the joint configurations in ``config_of`` (point id -> label).
    (Reference / tests; the scene draws ``area_faces`` as one vertex-coloured mesh.)"""
    faces_idx, keys = _stop_groups(mesh, ids, P, sol, config_of)
    out = []
    for k, g in sorted({(int(a), int(b)) for a, b in keys}):
        sel = faces_idx[(keys[:, 0] == k) & (keys[:, 1] == g)]
        sub = mesh.submesh([sel], append=True)
        out.append(
            (
                f"S{k + 1}_G{g + 1}",
                sub.vertices + OFFSET * sub.vertex_normals,
                sub.faces,
                group_colour(k, g, colors),
            )
        )
    return out


def area_faces(
    mesh: trimesh.Trimesh, ids, P: np.ndarray, sol, colors, config_of: dict[int, str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(painted face indices, (n, 3) uint8 colour per painted face, sorted group names
    ``S<k>_G<g>``) of solution ``sol``: the same faces and colours as ``area_meshes``."""
    faces_idx, keys = _stop_groups(mesh, ids, P, sol, config_of)
    rgb = np.zeros((len(faces_idx), 3), np.uint8)
    names = []
    for k, g in sorted({(int(a), int(b)) for a, b in keys}):
        rgb[(keys[:, 0] == k) & (keys[:, 1] == g)] = group_colour(k, g, colors)
        names.append(f"S{k + 1}_G{g + 1}")
    return faces_idx, rgb, names
