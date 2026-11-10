"""Cancel stops the station planner and the tool optimizer within about a second (v1 waited
for a whole station-count round, 5-30 s, or a whole plan per tool length)."""

import threading
import time

import numpy as np
import pytest
import viser

from core.feasibility import CancelledError
from opt.placement import _solve
from tests.helpers import confirm_all


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8789, verbose=False)
    st = build_app(server)
    st["project"].load_example("mobile_amr")
    yield st
    server.stop()


def _cancel_after(panel, stage: str, text) -> float:
    """Start the panel's job, press Cancel once its progress text shows ``stage`` (inside the
    exact-check stages, whatever the machine load); seconds from Cancel to stop."""
    confirm_all(panel)
    panel.start()
    t_end = time.perf_counter() + 120
    while stage not in text().lower() and panel._thread.is_alive() and time.perf_counter() < t_end:
        time.sleep(0.05)
    assert panel._thread.is_alive(), f"job finished before Cancel (last status: {text()})"
    where = text()
    t0 = time.perf_counter()
    panel._cancel.set()
    panel._thread.join(timeout=30)
    assert not panel._thread.is_alive(), f"still running 30 s after Cancel at: {where}"
    dt = time.perf_counter() - t0
    assert dt < 2.0, f"Cancel took {dt:.1f} s at: {where}"
    return dt


def test_station_planner_cancels_fast(app):
    panel = app["stations"]
    _cancel_after(panel, "quick plan", lambda: panel.status.content)  # inside the exact checks
    assert "Cancelled" in panel.status.content


def test_tool_optimizer_cancels_fast(app):
    panel = app["tool_opt"]
    _cancel_after(panel, "l scan", lambda: panel.info.content)  # inside the first plan of the L scan


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


def test_map_build_stops_on_cancel(tmp_path):
    """A first-time reachability map build (about a minute) also honours Cancel."""
    from core.robot import Robot
    from opt.reachability import map_for_tool

    flag = threading.Event()
    threading.Timer(0.5, flag.set).start()
    t0 = time.perf_counter()
    with pytest.raises(CancelledError):
        map_for_tool(Robot.from_registry("ur20"), np.eye(4), cache_dir=tmp_path, cancelled=flag.is_set)
    assert time.perf_counter() - t0 < 3.0 and not list(tmp_path.glob("*.npz"))


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
