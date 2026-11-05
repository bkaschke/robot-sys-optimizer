import numpy as np
import pytest

from core import golden_parts as gp


@pytest.mark.parametrize("name", ["flat_plate", "curved_panel", "long_rail"])
def test_watertight_parts(name):
    mesh = gp.GOLDEN_PARTS[name]()
    assert mesh.is_watertight
    assert mesh.volume > 0


def test_flat_plate_dims():
    assert np.allclose(gp.flat_plate().extents, [1.0, 0.6, 0.02])


def test_open_box_pocket_depth():
    mesh = gp.open_box()
    assert np.allclose(mesh.extents, [0.6, 0.6, 0.5])
    # a ray straight down the middle hits the pocket floor 0.45 m below the rim
    hits, _, _ = mesh.ray.intersects_location([[0, 0, 1.0]], [[0, 0, -1.0]])
    assert np.isclose(hits[:, 2].max(), 0.05)


def test_curved_panel_normals_match_analytic():
    mesh = gp.curved_panel()
    a = np.linspace(-0.35, 0.35, 15)
    pts = np.array([gp.panel_outer_point(0.1, ai) for ai in a])
    expected = np.array([gp.panel_outer_normal(ai) for ai in a])
    _, _, tri = mesh.nearest.on_surface(pts)
    got = mesh.face_normals[tri]
    angle = np.degrees(np.arccos(np.clip((got * expected).sum(1), -1, 1)))
    assert angle.max() < 1.0  # faces are outward and within 1 degree


def test_long_rail_exceeds_ur20_reach():
    # UR20 reach is 1.75 m; a 6 m rail cannot be covered from one base pose
    assert gp.long_rail().extents[0] > 2 * 1.75
