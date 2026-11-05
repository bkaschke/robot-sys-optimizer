"""M2 checks. Done-when: robot pose in the viewer matches Pinocchio FK for the same q."""

import urllib.request

import numpy as np
import pytest
import viser

from app import TABS, build_app
from core.transforms import make_T, rpy_to_matrix, wxyz_to_matrix
from ui.session import Session, load_project, save_project


@pytest.fixture(scope="module")
def app():
    server = viser.ViserServer(host="127.0.0.1", port=8765, verbose=False)
    state = build_app(server)
    yield server, state
    server.stop()


def _viewer_pose(state, link: str) -> np.ndarray:
    """World pose of a link as the browser receives it: base gizmo handle x URDF frames."""
    cell = state["cell"]
    g = cell.base.gizmo
    T_world_base = make_T(wxyz_to_matrix(g.wxyz), g.position)
    return T_world_base @ cell.view.link_pose_in_parent(link)


def test_page_served_with_tabs(app):
    server, state = app
    assert set(state["tabs"]) == set(TABS)
    with urllib.request.urlopen(f"http://127.0.0.1:{server.get_port()}/", timeout=10) as r:
        assert r.status == 200


def test_viewer_matches_pinocchio_fk(app):
    _, state = app
    cell, s = state["cell"], state["session"]
    rng = np.random.default_rng(7)
    for _ in range(25):
        # move the base through the GUI fields (mm / deg), then jog through the sliders
        xyz = rng.uniform(-1000, 1000, 3)
        rpy = rng.uniform(-180, 180, 3)
        cell.base.rpy.value = tuple(rpy)
        cell.base.xyz.value = tuple(xyz)
        q = s.robot.random_q(rng)
        cell.set_q(q)
        q_used = np.radians([sl.value for sl in cell.sliders])  # slider step is 0.1 deg
        assert np.allclose(s.q, q_used)
        expected_base = make_T(rpy_to_matrix(np.radians(rpy)), xyz / 1000)
        assert np.allclose(s.T_world_base, expected_base, atol=1e-9)
        for link in ("shoulder_link", "forearm_link", "wrist_3_link", s.robot.spec.flange_link):
            T_fk = s.T_world_base @ s.robot.frame_pose(s.q, link)
            T_view = _viewer_pose(state, link)
            assert np.allclose(T_view, T_fk, atol=1e-6), link


def test_mount_preset_keeps_yaw(app):
    _, state = app
    cell, s = state["cell"], state["session"]
    cell.mount.value = "floor"
    cell.base.rpy.value = (0.0, 0.0, 30.0)
    assert cell.mount.value == "floor"
    cell.mount.value = "ceiling"
    R = s.T_world_base[:3, :3]
    assert np.allclose(R @ np.array([0, 0, 1]), [0, 0, -1])  # base z points down
    assert np.isclose(np.degrees(np.arctan2(R[1, 0], R[0, 0])), 30.0)
    cell.base.rpy.value = (10.0, 0.0, 0.0)
    assert cell.mount.value == "custom"


def test_part_switch(app):
    _, state = app
    cell, s = state["cell"], state["session"]
    cell.part_choice.value = "curved_panel"
    assert s.part.mesh_path == "assets/parts/curved_panel.stl"
    cell.part_choice.value = "(none)"
    assert s.part is None and cell.part_handle is None


def test_project_save_load_round_trip(tmp_path):
    s = Session()
    s.load_robot_from_registry("ur20")
    s.T_world_base = make_T(rpy_to_matrix((0, 0, 0.7)), (1.2, -0.3, 0.1))
    from ui.session import PartState

    s.part = PartState("assets/parts/open_box.stl", "m", make_T(p=(0.5, 0.5, 0.8)))
    save_project(s, tmp_path)

    s2 = Session()
    load_project(s2, tmp_path)
    assert s2.robot_name == "ur20"
    assert np.allclose(s2.T_world_base, s.T_world_base)
    assert s2.part.mesh_path == "assets/parts/open_box.stl"
    assert np.allclose(s2.part.T_world_part, s.part.T_world_part)


def test_broken_project_leaves_session_untouched(tmp_path):
    s = Session()
    s.load_robot_from_registry("ur20")
    (tmp_path / "project.json").write_text(
        '{"version": 1, "robot": {"name": "ur20"}, "part": {"mesh": "missing.stl"}}'
    )
    with pytest.raises(FileNotFoundError):
        load_project(s, tmp_path)
    assert s.robot_name == "ur20" and s.part is None
