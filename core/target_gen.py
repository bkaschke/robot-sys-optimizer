"""Target generation from a surface selection, without the GUI.

``GenSettings`` holds every parameter of the Targets tab's *Generate points* panel (SI
units: m, degrees for the row angle) and is JSON-serialisable, so a batch manifest can
store it. ``generate_raw`` / ``generate_targets`` reproduce ``TargetsTab.run_generate``
exactly: per-side generation (``split_sides``), then the dedupe against existing points
(tolerance 0.4 x the minimum spacing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from pydantic import BaseModel, ConfigDict, Field

from core import target_tools as tt
from core import targets as tg
from core.facing import facing_sides
from core.surface_grid import surface_grid

GRID = "Surface grid (rows/columns)"
LATTICE = "Part-anchored lattice"
PROJECTED = "Projected grid"
POISSON = "Poisson (even)"
RASTER = "Raster lines"
EDGES = "Selection edges"
METHODS = (GRID, LATTICE, PROJECTED, POISSON, RASTER, EDGES)
SPLIT_METHODS = (GRID, LATTICE, PROJECTED, RASTER)  # methods that honour split_sides


class GenSettings(BaseModel):
    """Parameters of one generation run (defaults = the GUI widgets, in SI units)."""

    model_config = ConfigDict(extra="forbid")

    method: str = GRID
    row_m: float = Field(0.025, gt=0)
    col_m: float = Field(0.025, gt=0)
    inset_m: float = Field(0.0, ge=0)
    row_angle_deg: float = 0.0
    row_dir: tuple[float, float, float] = (1.0, 0.0, 0.0)
    spacing_m: float = Field(0.010, gt=0)
    stepover_m: float = Field(0.025, gt=0)
    direction: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (0,0,0) = automatic
    normals_mode: str = "interpolated"
    split_sides: bool = True
    group: str = "default"


@dataclass
class Generated:
    P: np.ndarray
    N: np.ndarray
    note: str = ""


def dedupe_tol(s: GenSettings) -> float:
    if s.method == RASTER:
        return 0.4 * min(s.spacing_m, s.stepover_m)
    return 0.4 * (min(s.row_m, s.col_m) if s.method in (GRID, LATTICE) else s.spacing_m)


def _direction(s: GenSettings) -> np.ndarray | None:
    d = np.asarray(s.direction, float)
    return d if np.linalg.norm(d) > 0 else None


def _one(m: trimesh.Trimesh, faces: np.ndarray, s: GenSettings, d) -> tuple[np.ndarray, np.ndarray, str]:
    method, step, nm = s.method, s.spacing_m, s.normals_mode
    if method == GRID:
        P, N, info = surface_grid(m, faces, s.row_m, s.col_m, s.inset_m, s.row_angle_deg, nm)
        txt = f"{info.rows} rows on {info.area_m2 * 1e4:.0f} cm²"
        if info.dropped_at_edge:
            txt += f"; {info.dropped_at_edge} within the edge inset left out"
        return P, N, txt + "".join(f"  \n⚠ {w}" for w in info.warnings)
    if method == LATTICE:
        rd = np.asarray(s.row_dir, float)
        rd = rd if np.linalg.norm(rd) > 0 else (1.0, 0.0, 0.0)
        return (*tt.lattice_grid(m, faces, s.row_m, s.col_m, d, rd, normals=nm), "")
    if method == PROJECTED:
        return (*tg.projected_grid(m, faces, step, direction=d, normals=nm), "")
    if method == POISSON:
        return (*tg.poisson_disk(m, faces, step, normals=nm), "")
    if method == RASTER:
        return (*tg.raster(m, faces, s.stepover_m, step, d, nm), "")
    if method == EDGES:
        return (*tg.edge_paths(m, faces, step), "")
    raise ValueError(f"Unknown generator '{method}'")


def generate_raw(mesh: trimesh.Trimesh, faces, s: GenSettings) -> Generated:
    """All points of the generator on ``faces`` (no dedupe). Raises ValueError/IndexError."""
    faces = np.asarray(faces, dtype=int)
    d = _direction(s)
    per_side = s.split_sides and d is None and s.method in SPLIT_METHODS
    cell = s.row_m * s.col_m if s.method in (GRID, LATTICE) else s.spacing_m**2
    sides = facing_sides(mesh, faces, min_area=cell) if per_side else [faces]
    if not sides:
        raise ValueError("Every selected side is smaller than one grid cell: reduce the spacing.")
    outs = [_one(mesh, f, s, d) for f in sides]
    P = np.concatenate([o[0] for o in outs]) if outs else np.zeros((0, 3))
    N = np.concatenate([o[1] for o in outs]) if outs else np.zeros((0, 3))
    note = "; ".join(o[2] for o in outs if o[2])
    if len(sides) > 1:
        note = f"{len(sides)} sides" + (f": {note}" if note else "")
    return Generated(P, N, note)


def generate_targets(
    mesh: trimesh.Trimesh,
    faces,
    settings: GenSettings,
    existing_points: np.ndarray | None = None,
    existing_normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate points on ``faces`` and drop those duplicating ``existing_points``.

    Returns (P, N, group name per point); every point goes into ``settings.group``.
    """
    g = generate_raw(mesh, faces, settings)
    P, N = np.asarray(g.P).reshape(-1, 3), np.asarray(g.N).reshape(-1, 3)
    old = np.zeros((0, 3)) if existing_points is None else np.asarray(existing_points).reshape(-1, 3)
    n_old = None if existing_normals is None else np.asarray(existing_normals).reshape(-1, 3)
    P, N = tt.dedupe(P, N, old, dedupe_tol(settings), n_old)
    return P, N, [settings.group] * len(P)
