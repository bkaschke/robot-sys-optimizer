"""M4 core checks. Done-when: a 10 mm grid on the curved test panel has normals within 1 deg
of the analytic normals."""

import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree

from core import golden_parts as gp
from core.targets import (
    edge_paths,
    grow_region,
    parse_csv,
    pick_face,
    poisson_disk,
    projected_grid,
    raster,
    select_by_normal,
    select_in_screen_rect,
    surface_normals,
)


@pytest.fixture(scope="module")
def panel():
    m = gp.curved_panel()
    top = grow_region(m, pick_face(m, (0.05, 0.02, 2.0), (0, 0, -1)), 15.0)
    return m, top


def _analytic(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Angle a and analytic outer normal for points on the panel's outer surface."""
    a = np.arctan2(p[:, 1], p[:, 2] - (gp.PANEL_TOP_Z - gp.PANEL_RADIUS))
    return a, np.column_stack([np.zeros_like(a), np.sin(a), np.cos(a)])


def _angle_deg(n1, n2) -> np.ndarray:
    return np.degrees(np.arccos(np.clip((n1 * n2).sum(axis=1), -1, 1)))


def test_click_region_is_exactly_the_outer_surface(panel):
    m, top = panel
    assert np.all(m.face_normals[top][:, 2] > np.cos(gp.PANEL_HALF_ANGLE + 0.02))
    expected = 2 * (61 - 1) * (81 - 1)  # outer grid: (nx-1)*(na-1) quads, 2 triangles each
    assert len(top) == expected


def test_10mm_grid_normals_within_1deg(panel):
    m, top = panel
    pts, nrm = projected_grid(m, top, 0.010)
    assert len(pts) > 0.9 * (0.6 / 0.01) * (2 * np.sin(gp.PANEL_HALF_ANGLE) / 0.01)
    a, expected = _analytic(pts)
    err = _angle_deg(nrm, expected)
    assert err.max() < 1.0, f"worst normal error {err.max():.3f} deg"
    # points really lie on the outer surface (radius R about the panel axis)
    r = np.hypot(pts[:, 1], pts[:, 2] - (gp.PANEL_TOP_Z - gp.PANEL_RADIUS))
    assert np.allclose(r, gp.PANEL_RADIUS, atol=2e-4)  # chord sag of the tessellation
    # spacing is 10 mm in the projection plane
    d, _ = cKDTree(pts[:, :2]).query(pts[:, :2], k=2)
    assert np.allclose(np.median(d[:, 1]), 0.010, atol=1e-6)


def test_interpolated_normals_beat_face_normals(panel):
    m, top = panel
    pts, n_i = projected_grid(m, top, 0.02)
    _, n_f = projected_grid(m, top, 0.02, normals="face")
    _, expected = _analytic(pts)
    assert _angle_deg(n_i, expected).max() < _angle_deg(n_f, expected).max()
    assert _angle_deg(n_i, expected).max() < 0.05


def test_selection_border_normals_not_blended_across_edge(panel):
    m, top = panel
    # sample exactly on the selection's border vertices: normals must stay on the surface
    border = np.unique(m.faces[top])
    pts = m.vertices[border][:50]
    _, _, tri = m.nearest.on_surface(pts)
    in_sel = np.isin(tri, top)
    n = surface_normals(m, pts[in_sel], tri[in_sel], top)
    _, expected = _analytic(pts[in_sel])
    assert _angle_deg(n, expected).max() < 0.5


def test_grid_ignores_back_faces_and_unselected():
    m = gp.flat_plate()
    top = select_by_normal(m, (0, 0, 1), 1.0)
    pts, n = projected_grid(m, top, 0.05, direction=(0, 0, -1))
    assert np.allclose(pts[:, 2], 0.02) and np.allclose(n, [0, 0, 1])
    bottom = select_by_normal(m, (0, 0, -1), 1.0)
    pts_b, _ = projected_grid(m, bottom, 0.05, direction=(0, 0, -1))  # bottom faces away
    assert len(pts_b) == 0


def test_poisson_spacing(panel):
    m, top = panel
    pts, nrm = poisson_disk(m, top, 0.02, seed=3)
    d, _ = cKDTree(pts).query(pts, k=2)
    assert d[:, 1].min() >= 0.02 - 1e-9
    assert d[:, 1].mean() < 0.02 * 1.5  # dense: no big holes
    _, expected = _analytic(pts)
    # <= half a facet angle (0.29 deg): selection-border vertices only see inner faces
    assert _angle_deg(nrm, expected).max() < 0.35


def test_raster_lines(panel):
    m, top = panel
    pts, nrm = raster(m, top, stepover=0.05, step=0.01, direction=(1, 0, 0))
    # lines run along x (panel axis): 0.6 m long -> ~60 points per line, ~16 lines across
    rows = np.round(pts[:, 1], 3)
    assert 14 <= len(np.unique(rows)) <= 18
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert np.isclose(np.median(steps), 0.01, atol=1e-6)  # along-line spacing
    # serpentine: first two rows run in opposite x directions
    row_ids = list(dict.fromkeys(rows.tolist()))  # rows in visiting order
    first, second = pts[rows == row_ids[0]], pts[rows == row_ids[1]]
    assert np.sign(first[-1, 0] - first[0, 0]) == -np.sign(second[-1, 0] - second[0, 0])
    _, expected = _analytic(pts)
    assert _angle_deg(nrm, expected).max() < 0.5


def test_edge_paths_follow_border():
    m = gp.flat_plate()
    top = select_by_normal(m, (0, 0, 1), 1.0)
    pts, nrm = edge_paths(m, top, 0.05)
    # perimeter 2*(1.0 + 0.6) = 3.2 m at 50 mm steps
    assert 60 <= len(pts) <= 70
    on_border = np.isclose(np.abs(pts[:, 0]), 0.5) | np.isclose(np.abs(pts[:, 1]), 0.3)
    assert on_border.all() and np.allclose(nrm, [0, 0, 1])


def test_screen_rect_selection():
    m = trimesh.creation.box(extents=(1, 1, 1))
    # OpenCV camera 3 m above the box looking down -z: R maps cam x->+x, y->-y, z->-z
    T_cam_world = np.eye(4)
    T_cam_world[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T_cam_world[:3, 3] = [0, 0, 3.0]
    T_cam_mesh = np.linalg.inv(T_cam_world)
    sel = select_in_screen_rect(m, T_cam_mesh, np.radians(60), 1.0, (0.0, 0.0), (1.0, 1.0))
    assert np.allclose(m.face_normals[sel], [0, 0, 1])  # only the top faces the camera
    none = select_in_screen_rect(m, T_cam_mesh, np.radians(60), 1.0, (0.0, 0.0), (0.05, 0.05))
    assert len(none) == 0


def test_csv_parse():
    rows = parse_csv("x,y,z,nx,ny,nz,group\n10,20,30,0,0,2,top\n# comment\n1,2,3,1,0,0\n", scale=1e-3)
    assert len(rows) == 2
    assert np.allclose(rows[0][0], [0.01, 0.02, 0.03]) and np.allclose(rows[0][1], [0, 0, 1])
    assert rows[0][2] == "top" and rows[1][2] is None
    for bad in (
        "1,2,3,0,0,0\n",
        "1,2,3,0,0\n",
        "x,y\n1,2,3,a,0,1\n",
        "1,2,3,0,0,1,g,extra\n",
        "1,2,nan,0,0,1\n",
    ):
        with pytest.raises(ValueError):
            parse_csv(bad)
