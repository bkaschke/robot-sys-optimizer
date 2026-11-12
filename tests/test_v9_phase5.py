"""V9 phase 5 (issue #9): Optimize -> Gantry placement (opt.frame_place, ui.frame_ui)."""

import numpy as np
import pytest
import viser
from shapely.geometry import LineString, Point

from core.schema import Base, CarrierJoints, Pose
from core.transforms import pose_to_matrix
from opt.frame_place import FrameSettings, candidate_frames
from tests.helpers import confirm_all


def _fp():
    from shapely.geometry import box

    return box(-1.0, -0.2, 1.0, 0.2)  # a long part along x


def test_rail_candidates_beside_the_part():
    base = Base(
        mode="movable", carrier="rail", joints=CarrierJoints(x=(-1.5, 1.5)), T_mount=Pose(xyz=(0, 0, 0.7))
    )
    st = FrameSettings(d_min=0.3, d_max=0.9, step=0.3, z_lo=0.5, z_hi=0.9, z_steps=3)
    c = candidate_frames(base, _fp(), st)
    assert len(c) == 2 * 3 * 3  # two sides, three standoffs, three mount heights
    ds = set()
    for pose, h in c:
        T = pose_to_matrix(pose)
        assert abs(abs(T[0, 0]) - 1) < 1e-9  # the track along the part's long axis
        line = LineString([T[:2, 3] - 2 * T[:2, 0], T[:2, 3] + 2 * T[:2, 0]])
        ds.add(round(line.distance(_fp()), 3))
        assert 0.5 - 1e-9 <= h <= 0.9 + 1e-9
    assert ds == {0.3, 0.6, 0.9}


def test_gantry_candidates_along_and_across():
    base = Base(
        mode="movable",
        carrier="gantry",
        joints=CarrierJoints(x=(-1, 1), y=(-1, 1)),
        T_mount=Pose(xyz=(0, 0, -0.8)),
    )
    c = candidate_frames(base, _fp(), FrameSettings(shift=0.3, shift_step=0.3))
    assert len(c) == 2 * 3 * 3  # along / across, a 3 x 3 grid of shifts, the current mount height
    yaws = {round(np.degrees(np.arctan2(*pose_to_matrix(p)[[1, 0], 0])) % 180) for p, _ in c}
    assert yaws == {0, 90}
    assert all(h == pytest.approx(-0.8) for _, h in c)
    assert max(Point(p.xyz[:2]).distance(Point(0, 0)) for p, _ in c) == pytest.approx(0.3 * np.sqrt(2))


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8849, verbose=False)
    st = build_app(server)
    yield st
    st["builder_window"].stop()
    server.stop()


def test_place_the_rail_and_use_it(app):
    from ui.staleness import solution_changes

    assert app["project"].load_example("rail_long_part")
    s, frame, stations = app["session"], app["frame"], app["stations"]
    app["picker"].show("Gantry placement")
    assert frame.folder.visible and frame.standoff.visible and not frame.shift.visible
    stations.refine.value = "none"
    confirm_all(stations)
    frame.top.value = 2
    frame.start(block=True)
    assert 1 <= len(frame.rows) <= 2 and "| 1 |" in frame.status.content
    best = frame.rows[0]
    assert best["reached"] == len(s.targets.points) and best["plans"]
    assert frame.use_frame(index=0)
    assert s.base.pose == best["base"].pose and not s.base.align_to_part
    assert s.results and solution_changes(s, s.results[0]) == []  # its plans, up to date
    assert s.results[0].label in s.result_points
