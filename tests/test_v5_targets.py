"""v4 plan item 1: sand-optimizer-style selection tools (fill, brush handle, box, full surface,
select all / clear / invert), the rows/columns surface grid, and per-tool / per-generator
settings that hide when not needed."""

from dataclasses import dataclass

import numpy as np
import pytest
import viser

from core import golden_parts as gp
from core import target_tools as tt
from core import targets as tg
from core.surface_grid import FacePainter, surface_grid

PLATE_TOP = 0.02


def _top(m):
    return tg.select_by_normal(m, (0, 0, 1), 5.0)


# --- core ---------------------------------------------------------------------------------------


def test_surface_grid_rows_columns_centred_and_ordered():
    m = gp.flat_plate()  # 1.0 x 0.6
    P, N, info = surface_grid(m, _top(m), row=0.05, col=0.1)
    xs, ys = np.unique(P[:, 0].round(9)), np.unique(P[:, 1].round(9))
    assert (len(xs), len(ys)) == (11, 13) and len(P) == 143 and info.rows == 13 and info.cols == 11
    assert np.allclose(np.diff(xs), 0.1) and np.allclose(np.diff(ys), 0.05)
    assert np.isclose(xs.mean(), 0) and np.isclose(ys.mean(), 0)  # centred on the selection
    assert np.allclose(P[:, 2], PLATE_TOP) and np.allclose(N, (0, 0, 1))
    assert np.all(np.diff(P[:, 1]) >= -1e-12)  # row by row (rows along part x)
    assert np.isclose(info.area_m2, 0.6) and info.spread == pytest.approx(1.0)


def test_surface_grid_inset_and_row_angle():
    m = gp.flat_plate()
    P, _, info = surface_grid(m, _top(m), row=0.05, col=0.1, inset=0.03)
    assert info.dropped_at_edge == 143 - len(P) > 0
    assert np.all(np.abs(P[:, 0]) <= 0.5 - 0.03 + 1e-9) and np.all(np.abs(P[:, 1]) <= 0.3 - 0.03 + 1e-9)
    P90, _, _ = surface_grid(m, _top(m), row=0.05, col=0.1, angle_deg=90)  # rows along part y
    assert np.allclose(np.diff(np.unique(P90[:, 1].round(9))), 0.1)
    assert np.allclose(np.diff(np.unique(P90[:, 0].round(9))), 0.05)


def test_surface_grid_curved_normals_and_errors():
    c = gp.curved_panel()
    faces = tg.select_by_normal(c, (0, 0, 1), 60)
    P, N, info = surface_grid(c, faces, 0.02, 0.02, inset=0.01)
    assert len(P) > 500 and not info.warnings
    a = np.arctan2(P[:, 1], P[:, 2] - (gp.PANEL_TOP_Z - gp.PANEL_RADIUS))
    exp = np.column_stack([np.zeros_like(a), np.sin(a), np.cos(a)])
    assert np.degrees(np.arccos(np.clip((N * exp).sum(1), -1, 1))).max() < 1.0
    with pytest.raises(ValueError, match="No surface"):
        surface_grid(c, [], 0.02, 0.02)
    with pytest.raises(ValueError, match="spacing"):
        surface_grid(c, faces, 0.0, 0.02)
    box = gp.flat_plate()
    with pytest.raises(ValueError, match="opposite"):
        surface_grid(box, np.arange(len(box.faces)), 0.05, 0.05)  # a closed solid


def test_painter_brush_component_and_project():
    m, _ = tt.refine_mesh(gp.flat_plate(), 0.02)
    fp = FacePainter(m)
    c = np.array([0.0, 0.0, PLATE_TOP])
    f = fp.brush(c, (0, 0, 1), 0.05, 45)
    d = np.linalg.norm(m.triangles_center[f] - c, axis=1)
    assert len(f) > 10 and d.max() < 0.05 + 0.02 and np.allclose(m.face_normals[f], (0, 0, 1))
    back = fp.brush(c, (0, 0, -1), 0.05, 45)  # brushing the back: only the bottom faces
    assert len(back) and np.allclose(m.face_normals[back], (0, 0, -1))
    assert len(fp.component(int(f[0]))) == len(m.faces)  # one body
    p, n, _ = fp.project((0.1, 0.05, 0.4))
    assert np.allclose(p, (0.1, 0.05, PLATE_TOP)) and np.allclose(n, (0, 0, 1))
    coarse = gp.flat_plate()  # two triangles on top: a brush that touches one takes it
    assert len(FacePainter(coarse).brush(c, (0, 0, 1), 0.01, 45)) >= 1


# --- GUI ----------------------------------------------------------------------------------------


@dataclass
class FakeClick:
    ray_origin: tuple
    ray_direction: tuple
    modifier: str | None = None
    client = None


def click_top(x, y, modifier=None):
    return FakeClick((1.0 + x, y, 2.0), (0.0, 0.0, -1.0), modifier)


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8803, verbose=False)
    state = build_app(server)  # flat plate at (1, 0, 0.5)
    state["session"].project_dir = tmp_path_factory.mktemp("proj")
    state["session"].targets = state["session"].targets.model_copy(update={"points": []})
    tab = state["targets"]
    tab.click_mode.value = "Fill +"
    tab.on_part_click(click_top(0, 0))
    tab.tools.max_edge.value = 20.0
    assert tab.tools.refine_part()  # a fine mesh for the brush
    tab.select([], "set")
    yield state
    server.stop()


def test_tool_dropdown_shows_only_its_settings(app):
    tab = app["targets"]
    sf = tab.surf
    tab.click_mode.value = "Fill +"
    assert sf.angle.visible and not sf.radius.visible and not sf.normal_dir.visible and sf.handle is None
    tab.click_mode.value = "Brush +"
    assert sf.radius.visible and sf.max_angle.visible and not sf.angle.visible and sf.handle is not None
    tab.click_mode.value = "By normal"
    assert sf.normal_dir.visible and sf.by_normal.visible and sf.handle is None
    tab.click_mode.value = "Box +"
    assert sf._rect_on
    tab.click_mode.value = "Remove point"
    assert sf.radius.visible and not sf.max_angle.visible and not sf._rect_on
    tab.method.value = "Surface grid (rows/columns)"
    assert tab.inset.visible and tab.row_angle.visible and not tab.spacing.visible and not tab.row_dir.visible
    tab.method.value = "Poisson (even)"
    assert tab.spacing.visible and not tab.row.visible and not tab.direction.visible
    tab.method.value = "Part-anchored lattice"
    assert tab.row_dir.visible and not tab.inset.visible


def test_brush_handle_paints_and_erases_along_its_path(app):
    tab = app["targets"]
    sf = tab.surf
    tab.select([], "set")
    tab.click_mode.value = "Brush +"
    sf.radius.value = 30.0
    sf.dab((-0.3, 0.0, PLATE_TOP))  # a click: one dab, handle moved there
    n_dab = len(tab.selection)
    assert n_dab > 0 and np.allclose(sf.handle.position, (-0.3, 0.0, PLATE_TOP))
    sf.brush_to(np.array([0.3, 0.0, 0.25]))  # dragged above the part: follows the surface
    m = tab.mesh()
    c = m.triangles_center[tab.selection]
    assert len(tab.selection) > 5 * n_dab
    assert np.all(np.abs(c[:, 1]) < 0.03 + 0.02) and np.allclose(c[:, 2], PLATE_TOP)  # a strip on top
    assert c[:, 0].min() < -0.28 and c[:, 0].max() > 0.28  # no gaps along the path
    sf.snap()
    assert np.allclose(sf.handle.position, (0.3, 0.0, PLATE_TOP))
    tab.click_mode.value = "Brush −"
    before = len(tab.selection)
    sf.dab((0.0, 0.0, PLATE_TOP))
    assert 0 < len(tab.selection) < before
    tab.click_mode.value = "Fill +"
    assert sf.handle is None


def test_fill_full_surface_and_bulk_buttons(app):
    tab = app["targets"]
    m = tab.mesh()
    tab.select([], "set")
    tab.click_mode.value = "Fill +"
    tab.on_part_click(click_top(0.1, 0.1))
    top = len(tab.selection)
    assert np.allclose(m.face_normals[tab.selection], (0, 0, 1))
    tab.on_part_click(click_top(0.1, 0.1, "ctrl"))  # ctrl inverts: removes
    assert len(tab.selection) == 0
    tab.click_mode.value = "Full surface"
    tab.on_part_click(click_top(0.1, 0.1))
    assert len(tab.selection) == len(m.faces)
    tab.surf.bulk("Clear")
    assert len(tab.selection) == 0
    tab.click_mode.value = "Fill +"
    tab.on_part_click(click_top(0.1, 0.1))
    tab.surf.bulk("Invert")
    assert len(tab.selection) == len(m.faces) - top
    tab.surf.bulk("Select all")
    assert len(tab.selection) == len(m.faces)


def test_generate_surface_grid_from_the_gui(app):
    tab, s = app["targets"], app["session"]
    s.targets = s.targets.model_copy(update={"points": []})
    tab.select([], "set")
    tab.click_mode.value = "Fill +"
    tab.on_part_click(click_top(0, 0))
    tab.method.value = "Surface grid (rows/columns)"
    tab.row.value, tab.col.value, tab.inset.value, tab.row_angle.value = 50.0, 100.0, 30.0, 0.0
    n = tab.run_generate()
    _, P, N, _ = tg.as_arrays(s.targets)
    assert n == len(P) > 0 and np.allclose(N, (0, 0, 1)) and np.allclose(P[:, 2], PLATE_TOP)
    assert "13 rows" not in tab.gen_info.content and "edge inset" in tab.gen_info.content
    assert tab.run_generate() == 0  # same grid again: every point already exists
    assert tab.tools.undo_last() == n and len(s.targets.points) == 0
