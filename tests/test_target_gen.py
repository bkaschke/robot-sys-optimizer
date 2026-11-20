"""core.target_gen: the GUI-free generator gives exactly what TargetsTab.run_generate gave
before the refactor (the old logic is copied below as ``old_generate``)."""

from types import SimpleNamespace

import numpy as np
import pytest

from core import golden_parts as gp
from core import target_tools as tt
from core import targets as tg
from core.facing import facing_sides
from core.surface_grid import surface_grid
from core.target_gen import GRID, LATTICE, METHODS, GenSettings, generate_raw, generate_targets
from ui.targets_ui import METHOD_FIELDS, TargetsTab


def old_generate(m, selection, w):
    """Verbatim logic of the pre-refactor run_generate; ``w`` = widget values (mm, deg)."""
    step = w["spacing"] / 1000
    d = np.asarray(w["direction"], float)
    d = d if np.linalg.norm(d) > 0 else None
    method = w["method"]
    row, col = w["row"] / 1000, w["col"] / 1000
    tol = 0.4 * (min(row, col) if method in (GRID, LATTICE) else step)
    if method == "Raster lines":
        tol = 0.4 * min(step, w["stepover"] / 1000)

    def one(faces):
        if method == GRID:
            P, N, _ = surface_grid(m, faces, row, col, w["inset"] / 1000, w["row_angle"], w["normals_mode"])
            return P, N
        if method == LATTICE:
            r = np.asarray(w["row_dir"], float)
            rd = r if np.linalg.norm(r) > 0 else (1.0, 0.0, 0.0)
            return tt.lattice_grid(m, faces, row, col, d, rd, normals=w["normals_mode"])
        if method == "Projected grid":
            return tg.projected_grid(m, faces, step, direction=d, normals=w["normals_mode"])
        if method == "Poisson (even)":
            return tg.poisson_disk(m, faces, step, normals=w["normals_mode"])
        if method == "Raster lines":
            return tg.raster(m, faces, w["stepover"] / 1000, step, d, w["normals_mode"])
        return tg.edge_paths(m, faces, step)

    per_side = w["split_sides"] and d is None and "split_sides" in METHOD_FIELDS.get(method, ())
    cell = row * col if method in (GRID, LATTICE) else step * step
    sides = facing_sides(m, selection, min_area=cell) if per_side else [selection]
    if not sides:
        raise ValueError("small")
    outs = [one(f) for f in sides]
    P = np.concatenate([o[0] for o in outs])
    N = np.concatenate([o[1] for o in outs])
    return P, N, tol


WIDGETS = dict(
    method=GRID,
    row=20.0,
    col=25.0,
    inset=5.0,
    row_angle=10.0,
    row_dir=(1.0, 0.0, 0.0),
    spacing=15.0,
    stepover=30.0,
    direction=(0.0, 0.0, 0.0),
    normals_mode="interpolated",
    split_sides=True,
    gen_group="default",
)


def fake_tab(**over):
    w = {**WIDGETS, **over}
    ns = SimpleNamespace(**{k: SimpleNamespace(value=v) for k, v in w.items()})
    return ns, w


def _sorted(P):
    return P[np.lexsort(P.T[::-1])]


def _plate_sel(both_sides=False):
    m = gp.flat_plate()
    sel = tg.select_by_normal(m, (0, 0, 1), 5)
    if both_sides:
        sel = np.union1d(sel, tg.select_by_normal(m, (0, 0, -1), 5))
    return m, sel


def test_methods_match_gui_constants():
    assert set(METHODS) == set(METHOD_FIELDS)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("both", [False, True])
def test_matches_old_run_generate(method, both):
    m, sel = _plate_sel(both)
    ns, w = fake_tab(method=method)
    st = TargetsTab.gen_settings(ns)
    P0, N0, tol0 = old_generate(m, sel, w)
    g = generate_raw(m, sel, st)
    assert len(g.P) == len(P0) > 0
    assert np.allclose(_sorted(g.P), _sorted(P0))
    assert np.allclose(g.N[np.lexsort(g.P.T[::-1])], N0[np.lexsort(P0.T[::-1])])
    # full path incl. dedupe: against nothing, then against the points themselves
    P, N, groups = generate_targets(m, sel, st)
    Pd, _ = tt.dedupe(P0, N0, np.zeros((0, 3)), tol0, None)
    assert np.allclose(_sorted(P), _sorted(Pd)) and groups == ["default"] * len(P)
    P2, _, g2 = generate_targets(m, sel, st, existing_points=P, existing_normals=N)
    assert len(P2) == 0 and g2 == []


def test_face_normals_and_curved_part():
    m = gp.curved_panel()
    sel = tg.select_by_normal(m, (0, 0, 1), 60)
    ns, w = fake_tab(method="Projected grid", normals_mode="face", spacing=20.0)
    st = TargetsTab.gen_settings(ns)
    P0, _, _ = old_generate(m, sel, w)
    assert np.allclose(_sorted(generate_raw(m, sel, st).P), _sorted(P0))


def test_dedupe_against_existing_uses_tolerance():
    m, sel = _plate_sel()
    st = GenSettings(method="Projected grid", spacing_m=0.02, split_sides=False)
    P, N, _ = generate_targets(m, sel, st)
    keep = len(P) // 2
    P2, _, _ = generate_targets(m, sel, st, existing_points=P[:keep], existing_normals=N[:keep])
    assert len(P2) == len(P) - keep


def test_group_name_and_errors():
    m, sel = _plate_sel()
    st = GenSettings(method="Poisson (even)", spacing_m=0.05, group="top")
    assert set(generate_targets(m, sel, st)[2]) == {"top"}
    with pytest.raises(ValueError):
        generate_raw(m, sel, GenSettings(row_m=5.0, col_m=5.0))  # every side < one cell


def test_json_round_trip():
    st = GenSettings(method=LATTICE, row_dir=(0, 1, 0), direction=(0, 0, -1), group="g2", split_sides=False)
    back = GenSettings.model_validate_json(st.model_dump_json())
    assert back == st
    assert GenSettings.model_validate(__import__("json").loads(st.model_dump_json())) == st
    assert GenSettings().row_m == 0.025 and GenSettings().spacing_m == 0.01
