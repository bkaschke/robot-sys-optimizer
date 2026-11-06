"""Cancel stops the station planner and the tool optimizer within about a second (v1 waited
for a whole station-count round, 5-30 s, or a whole plan per tool length)."""

import threading
import time

import numpy as np
import pytest
import viser

from core.feasibility import CancelledError
from opt.placement import _solve


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8789, verbose=False)
    st = build_app(server)
    st["project"].load_example("mobile_amr")
    yield st
    server.stop()


def _cancel_after(panel, delay: float) -> float:
    """Start the panel's job, press Cancel after ``delay`` s; seconds from Cancel to stop."""
    panel.start()
    time.sleep(delay)
    assert panel._thread.is_alive(), "job finished before Cancel: test scene too small"
    t0 = time.perf_counter()
    panel._cancel.set()
    panel._thread.join(timeout=30)
    assert not panel._thread.is_alive()
    return time.perf_counter() - t0


def test_station_planner_cancels_fast(app):
    panel = app["stations"]
    dt = _cancel_after(panel, 6.0)  # past the map load and set cover, inside refinement
    assert dt < 2.0, f"Cancel took {dt:.1f} s"
    assert "Cancelled" in panel.status.content


def test_tool_optimizer_cancels_fast(app):
    panel = app["tool_opt"]
    dt = _cancel_after(panel, 6.0)
    assert dt < 2.0, f"Cancel took {dt:.1f} s"


def test_checker_raises_when_cancelled(app):
    from opt.placement import Targets  # noqa: F401  (module import cost outside the timing)

    s = app["session"]
    from core.collision import CollisionScene
    from core.feasibility import FeasibilityChecker
    from core.ik import make_ik

    scene = CollisionScene(s.robot, s.tool_model, s.part_mesh(), s.T_world_base, s.part.T_world_part)
    ch = FeasibilityChecker(
        s.robot, make_ik(s.robot), s.tool_model, scene, s.T_world_base, s.part.T_world_part
    )
    ch.cancelled = lambda: True
    p = s.targets.points[0]
    with pytest.raises(CancelledError):
        ch.check_point(0, np.array(p.p), np.array(p.n), s.targets.groups[0])


def test_cp_sat_stops_on_cancel():
    from ortools.sat.python import cp_model

    # a hard model (pigeonhole 14 -> 13) that runs until the time limit unless stopped
    m = cp_model.CpModel()
    x = [[m.NewBoolVar(f"x{i}{j}") for j in range(13)] for i in range(14)]
    for i in range(14):
        m.AddExactlyOne(x[i])
    for j in range(13):
        m.AddAtMostOne(x[i][j] for i in range(14))
    flag = threading.Event()
    threading.Timer(0.5, flag.set).start()
    t0 = time.perf_counter()
    _, status = _solve(m, 30.0, flag.is_set)
    assert time.perf_counter() - t0 < 2.0
    assert status != cp_model.OPTIMAL
