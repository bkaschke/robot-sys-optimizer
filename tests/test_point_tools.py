"""Targets tab extras: single points, remove by click, row/column brush, fill selection,
undo, and part mesh refinement (tessellation)."""

from dataclasses import dataclass

import numpy as np
import pytest
import viser

from core import golden_parts as gp
from core import target_tools as tt
from core import targets as tg
from core.transforms import make_T

PLATE_TOP = 0.02


# --- core ---------------------------------------------------------------------------------------


def test_lattice_has_exact_row_and_column_spacing():
    m = gp.flat_plate()
    top = tg.select_by_normal(m, (0, 0, 1), 5.0)
    P, N = tt.lattice_grid(m, top, row=0.1, col=0.04)
    assert np.allclose(P[:, 2], PLATE_TOP) and np.allclose(N, (0, 0, 1))
    xs, ys = np.unique(P[:, 0].round(9)), np.unique(P[:, 1].round(9))
    assert np.allclose(np.diff(xs), 0.04) and np.allclose(np.diff(ys), 0.1)  # columns along x
    assert len(P) == len(xs) * len(ys)
    # rows along y instead: swap roles
    P2, _ = tt.lattice_grid(m, top, row=0.1, col=0.04, row_dir=(0, 1, 0))
    assert np.allclose(np.diff(np.unique(P2[:, 1].round(9))), 0.04)
    # serpentine: consecutive points within a row are one column apart
    assert np.allclose(np.linalg.norm(np.diff(P[: len(xs)], axis=0), axis=1), 0.04)
    with pytest.raises(ValueError):
        tt.lattice_grid(m, top, 0.0, 0.1)


def test_refine_keeps_surface_and_maps_faces():
    m = gp.curved_panel()
    r, parent = tt.refine_mesh(m, 0.01)
    assert len(r.faces) > len(m.faces) and len(parent) == len(r.faces)
    assert abs(r.area - m.area) < 1e-9  # flat subdivision: same surface
    assert r.edges_unique_length.max() <= 0.01 + 1e-9
    # every child face lies in its parent's plane
    assert np.allclose((r.triangles_center - m.triangles_center[parent]) @ np.zeros(3), 0)
    assert np.allclose(np.einsum("ij,ij->i", r.face_normals, m.face_normals[parent]), 1, atol=1e-9)
    with pytest.raises(ValueError):
        tt.refine_mesh(m, 1e-5)  # far too many faces


def test_brush_dab_stays_inside_radius_and_off_the_edge():
    m = gp.flat_plate()
    c = np.array([0.45, 0.0, PLATE_TOP])  # 50 mm from the +x edge
    f = int(m.ray.intersects_first([c + (0, 0, 1)], [(0, 0, -1)])[0])
    P, N = tt.brush_dab(m, f, c, radius=0.08, row=0.02, col=0.02, max_angle_deg=45)
    assert len(P) > 10
    assert np.all(np.linalg.norm(P - c, axis=1) <= 0.08 + 1e-9)
    assert np.allclose(P[:, 2], PLATE_TOP)  # nothing on the side face past the edge
    # a second, overlapping dab adds only the new lattice points
    c2 = c - (0.05, 0, 0)
    P2, N2 = tt.brush_dab(m, f, c2, radius=0.08, row=0.02, col=0.02)
    new, _ = tt.dedupe(P2, N2, P, 0.005)
    both = np.vstack([P, new])
    d = np.linalg.norm(both[:, None] - both[None], axis=2) + np.eye(len(both))
    assert d.min() > 0.019  # still one clean 20 mm lattice


def test_nearest_point():
    P = np.array([[0, 0, 0], [1, 0, 0.0]])
    assert tt.nearest_point(P, (0.9, 0, 0), 0.2) == 1
    assert tt.nearest_point(P, (0.5, 0, 0), 0.2) is None


# --- GUI ----------------------------------------------------------------------------------------


@dataclass
class FakeClick:
    ray_origin: tuple
    ray_direction: tuple
    modifier: str | None = None
    client = None


T_PART = make_T(p=(1.0, 0.0, 0.5))


def click_top(x, y):
    return FakeClick((1.0 + x, y, 2.0), (0.0, 0.0, -1.0))


@pytest.fixture
def app(tmp_path):
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8774, verbose=False)
    state = build_app(server)  # flat plate at (1, 0, 0.5)
    state["session"].project_dir = tmp_path
    state["session"].targets = state["session"].targets.model_copy(update={"points": []})
    state["targets"].refresh()
    yield state
    server.stop()


def test_single_point_remove_and_undo(app):
    tab, s = app["targets"], app["session"]
    tools = tab.tools
    tab.click_mode.value = "Single point"
    tab.on_part_click(click_top(0.1, 0.05))
    tab.on_part_click(click_top(-0.2, 0.0))
    assert len(s.targets.points) == 2
    assert np.allclose(s.targets.points[0].p, (0.1, 0.05, PLATE_TOP), atol=1e-9)
    tab.on_part_click(click_top(0.1, 0.05))  # same spot again: skipped as a duplicate
    assert len(s.targets.points) == 2
    tab.click_mode.value = "Remove point"
    tab.on_part_click(click_top(0.11, 0.05))
    assert len(s.targets.points) == 1 and np.allclose(s.targets.points[0].p[0], -0.2)
    assert tools.undo_last() == 1  # undoes the second add
    assert len(s.targets.points) == 0


def test_brush_dabs_and_fill_selection(app):
    tab, s = app["targets"], app["session"]
    tools = tab.tools
    tools.row.value, tools.col.value, tools.radius.value = 20.0, 30.0, 70.0
    tab.click_mode.value = "Point brush"
    tab.on_part_click(click_top(0.0, 0.0))
    n1 = len(s.targets.points)
    assert n1 > 5 and tools._ring is not None
    tab.on_part_click(click_top(0.03, 0.0))  # overlapping dab, shifted by one column
    n2 = len(s.targets.points)
    assert n1 < n2 < 2 * n1
    _, P, _, _ = tg.as_arrays(s.targets)
    d = np.linalg.norm(P[:, None] - P[None], axis=2) + np.eye(len(P))
    assert d.min() > 0.019  # no doubled points
    # lattice anchored at the part origin: x on multiples of the column spacing, y of the row
    for coord, step in ((P[:, 0], 0.03), (P[:, 1], 0.02)):
        k = coord / step
        assert np.allclose(k, np.round(k), atol=1e-6)
    tools.undo_last()
    assert len(s.targets.points) == n1

    # fill the whole top face with rows 50 mm / columns 100 mm
    tab.click_mode.value = "Fill +"
    tab.on_part_click(click_top(0.0, 0.0))
    assert len(tab.selection) > 0
    s.targets = s.targets.model_copy(update={"points": []})
    tools.row.value, tools.col.value = 50.0, 100.0
    added = tools.fill_selection()
    _, P, _, _ = tg.as_arrays(s.targets)
    assert added == len(P) == len(np.unique(P[:, 0].round(9))) * len(np.unique(P[:, 1].round(9)))
    assert np.allclose(np.diff(np.unique(P[:, 0].round(9))), 0.1)
    assert np.allclose(np.diff(np.unique(P[:, 1].round(9))), 0.05)


def test_refine_part_mesh_keeps_selection_and_points(app, tmp_path):
    tab, s, cell = app["targets"], app["session"], app["cell"]
    tab.click_mode.value = "Fill +"
    tab.on_part_click(click_top(0.0, 0.0))
    area = tab.mesh().area_faces[tab.selection].sum()
    tab.click_mode.value = "Single point"
    tab.on_part_click(click_top(0.2, 0.1))
    n_faces = len(tab.mesh().faces)
    tab.tools.max_edge.value = 25.0
    assert tab.tools.refine_part()
    m = tab.mesh()
    assert len(m.faces) > 10 * n_faces and m.edges_unique_length.max() <= 0.025 + 1e-9
    assert s.part.mesh_path.endswith("_refined_25mm.stl") and (tmp_path / s.part.mesh_path).exists()
    assert np.isclose(m.area_faces[tab.selection].sum(), area)  # same surface still selected
    assert len(s.targets.points) == 1
    assert cell.part_choice.value == "(uploaded mesh)" and cell.part_units.value == "m"
    assert len(s.part_mesh().faces) == len(m.faces)  # collision / scene use the refined mesh
    assert np.allclose(s.part.T_world_part, T_PART)
    # finer mesh: a brush dab now follows the refined faces and still works
    tab.click_mode.value = "Point brush"
    tab.on_part_click(click_top(-0.2, -0.1))
    assert len(s.targets.points) > 1
