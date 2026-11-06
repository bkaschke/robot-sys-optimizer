"""v3 feedback item 4: New part in the same cell, and a completely new empty study (no robot)."""

import shutil

import numpy as np
import pytest
import viser

from core.schema import load_project


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8798, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


@pytest.fixture
def made():
    """Study folders created by a test (removed afterwards)."""
    dirs = []
    yield dirs
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


def test_new_part_same_cell(app, made):
    s, pp = app["session"], app["project"]
    assert pp.load_example("rail_long_part")
    app["stations"].start(block=True)
    old_dir, base, tool, robot = s.project_dir, s.base, s.tool, s.robot_name
    groups = s.targets.groups
    assert s.part is not None and s.targets.points and s.results
    dest = pp.new_study(keep_cell=True)
    made.append(dest)
    from ui.session import REPO, WORKSPACE

    assert s.project_dir == dest and dest.parent == WORKSPACE and dest.name.startswith("study_")
    assert not WORKSPACE.is_relative_to(REPO / "workspace")  # tests never write the user's projects
    assert s.part is None and not s.targets.points and s.targets.groups == groups
    assert not s.results and not s.result_points and not s.feasibility and s.selection is None
    assert s.base == base and s.tool == tool and s.robot_name == robot  # the cell stays
    saved = load_project(dest / "project.json")
    assert saved.part is None and saved.robot.name == robot and saved.base == base
    old = load_project(old_dir / "project.json")
    assert old.part is not None and old.targets.points  # the previous study was kept
    assert app["cell"].part_handle is None and app["results"].choice.options == ("(none)",)


def test_new_empty_study_without_robot(app, made):
    s, pp, cell = app["session"], app["project"], app["cell"]
    dest = pp.new_study(keep_cell=False)
    made.append(dest)
    assert s.robot is None and s.tool is None and s.part is None and s.base.mode == "static"
    assert not s.targets.points and not s.targets.groups and not s.results
    assert cell.view is None and not cell.sliders and cell.robot_choice.value.startswith("(no robot")
    assert "No robot" in cell.readout.content
    # nothing crashes without a robot: planner / check / paths refuse politely
    app["stations"].start(block=True)
    app["optimize"].start(block=True)
    app["sim"].build(block=True)
    assert app["stations"]._thread is None or not app["stations"]._thread.is_alive()
    saved = load_project(dest / "project.json")
    assert saved.robot is None
    from ui.session import Session
    from ui.session import load_project as load_session

    s2 = Session()
    load_session(s2, dest)
    assert s2.robot is None and s2.q is None
    # choosing a robot brings it in
    cell.robot_choice.value = "ur20"
    assert s.robot is not None and cell.view is not None and len(cell.sliders) == s.robot.nq
    assert np.allclose(s.T_world_flange()[:3, 3], (s.T_world_base @ s.robot.fk(s.q))[:3, 3])
