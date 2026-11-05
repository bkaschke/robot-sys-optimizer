"""M4 through the GUI: click-select on the curved panel (moved and rotated in the cell),
generate a 10 mm grid, check normals against the analytic panel normals."""

from dataclasses import dataclass

import numpy as np
import pytest
import viser

from core import golden_parts as gp
from core import targets as tg
from core.transforms import make_T, rpy_to_matrix
from ui.session import Session, load_project, save_project
from ui.targets_ui import target_tcp_pose


@dataclass
class FakeClick:
    ray_origin: tuple
    ray_direction: tuple
    modifier: str | None = None
    client = None


T_PART = make_T(rpy_to_matrix((0.2, -0.1, 0.8)), (1.1, -0.2, 0.6))


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8767, verbose=False)
    state = build_app(server)
    cell = state["cell"]
    cell.part_choice.value = "curved_panel"
    cell.part_pose.xyz.value = tuple(T_PART[:3, 3] * 1000)
    cell.part_pose.rpy.value = (np.degrees(0.2), np.degrees(-0.1), np.degrees(0.8))
    yield state
    server.stop()


def _click_world(part_point, part_dir, modifier=None) -> FakeClick:
    """A click whose ray (world frame) passes through ``part_point`` along ``part_dir``."""
    o = T_PART[:3, :3] @ (np.asarray(part_point) - 2.0 * np.asarray(part_dir)) + T_PART[:3, 3]
    return FakeClick(tuple(o), tuple(T_PART[:3, :3] @ np.asarray(part_dir)), modifier)


def test_part_pose_applied(app):
    assert np.allclose(app["session"].part.T_world_part, T_PART, atol=1e-9)


def test_click_selects_outer_surface_and_grid_normals(app):
    tab, s = app["targets"], app["session"]
    tab.click_mode.value = "select surface"
    tab.on_part_click(_click_world((0.05, 0.03, gp.PANEL_TOP_Z), (0, 0, -1)))
    m = tab.mesh()
    assert len(tab.selection) == 2 * 60 * 80  # the whole outer surface, nothing else
    tab.method.value = "Projected grid"
    tab.spacing.value = 10.0
    tab.direction.value = (0.0, 0.0, 0.0)
    tab.run_generate()
    _, P, N, G = tg.as_arrays(s.targets)
    assert len(P) > 2000 and set(G) == {"default"}
    a = np.arctan2(P[:, 1], P[:, 2] - (gp.PANEL_TOP_Z - gp.PANEL_RADIUS))
    expected = np.column_stack([np.zeros_like(a), np.sin(a), np.cos(a)])
    err = np.degrees(np.arccos(np.clip((N * expected).sum(1), -1, 1)))
    assert err.max() < 1.0, f"worst normal error {err.max():.3f} deg"
    # drawn in the part frame under the part node
    assert np.allclose(np.asarray(tab._handles["points"].points), P, atol=1e-6)
    assert m is tab.mesh()


def test_group_tolerances_default_from_tool(app):
    s = app["session"]
    g = s.targets.groups[0]
    assert g.tilt_deg == s.tool.tolerances.tilt_deg  # sander preset: 10 deg


def test_shift_click_adds_ctrl_click_removes(app):
    tab = app["targets"]
    before = tab.selection.copy()
    tab.on_part_click(_click_world((0.0, 0.0, -1.0), (0, 0, 1), modifier="shift"))  # inner surface
    assert len(tab.selection) == 2 * len(before)
    tab.on_part_click(_click_world((0.0, 0.0, -1.0), (0, 0, 1), modifier="ctrl"))
    assert np.array_equal(tab.selection, before)


def test_flip_and_delete_on_selection(app):
    tab, s = app["targets"], app["session"]
    n0 = len(s.targets.points)
    tab._edit_on_selection(lambda m, i, P, N, G: (i, P, np.where(m[:, None], -N, N), G), None)
    assert all(p.n[2] < 0 for p in s.targets.points)
    tab._edit_on_selection(
        lambda m, i, P, N, G: (i[~m], P[~m], N[~m], [g for k, g in zip(m, G, strict=True) if not k]), None
    )
    assert len(s.targets.points) == 0 and n0 > 0


def test_place_point_mode(app):
    tab, s = app["targets"], app["session"]
    tab.click_mode.value = "place point"
    tab.on_part_click(_click_world((0.1, 0.0, gp.PANEL_TOP_Z), (0, 0, -1)))
    p = s.targets.points[-1]
    assert np.allclose(p.p, (0.1, 0.0, gp.PANEL_TOP_Z), atol=1e-6)
    assert np.allclose(p.n, (0, 0, 1), atol=1e-3)
    tab.click_mode.value = "select surface"


def test_preview_pose_points_tool_into_surface(app):
    tab, s = app["targets"], app["session"]
    tab.preview_on.value = True
    _, P, N, _ = tg.as_arrays(s.targets)
    tm = s.tool_model
    root = tab._handles["pv_root"]
    from core.transforms import wxyz_to_matrix

    T_world_flange = make_T(wxyz_to_matrix(root.wxyz), root.position)
    T_world_tcp = T_world_flange @ tm.T_flange_tcp("pad")
    assert np.allclose(T_world_tcp[:3, 3], T_PART[:3, :3] @ P[-1] + T_PART[:3, 3], atol=1e-9)
    assert np.allclose(T_world_tcp[:3, 2], -(T_PART[:3, :3] @ N[-1]), atol=1e-9)  # tcp.z = -target.z
    tab.preview_on.value = False


def test_target_tcp_pose_standoff():
    T = target_tcp_pose(np.array([0, 0, 1.0]), np.array([0, 0, 1.0]), 0.2)
    assert np.allclose(T[:3, 3], [0, 0, 1.2]) and np.allclose(T[:3, 2], [0, 0, -1])
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3))


def test_targets_saved_in_project(app, tmp_path):
    s = app["session"]
    save_project(s, tmp_path)
    s2 = Session()
    load_project(s2, tmp_path)
    assert len(s2.targets.points) == len(s.targets.points) > 0
    assert s2.targets.groups == s.targets.groups
