"""v8 T2: exact checks in a process pool (opt.parallel)."""

import pickle

import numpy as np
import pytest
import viser

from tests.helpers import confirm_all
from ui import jobs


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8852, verbose=False)
    st = build_app(server)
    yield st
    from opt.parallel import shutdown_all

    jobs.CHECK_WORKERS = 1
    shutdown_all()
    server.stop()


def _spec(s):
    from core.robot import Robot
    from opt.parallel import CheckerSpec

    sites = s.sites()
    bb, wb = sites.scene_bodies() if sites is not None else ([], [])
    robot = Robot(s.robot.spec)
    return CheckerSpec(
        robot.spec,
        s.tool_model,
        s.part_mesh(),
        s.T_world_base,
        s.part.T_world_part,
        bb,
        wb,
        s.obstacle_entries(),
    )


def test_gantry_setup_pickles(app):
    """Every collision-scene input must pickle (a closure in Carrier.axis_follow didn't)."""
    assert app["project"].load_example("gantry_overhead")
    spec = _spec(app["session"])
    checker = pickle.loads(pickle.dumps(spec)).build()  # a worker can rebuild the scene
    assert isinstance(checker.scene.collides(checker.robot.home_q(), checker.tool.ee.tcps[0].name), bool)


def test_pool_results_equal_serial_checks(app):
    from opt.parallel import CheckPool
    from ui.stations import world_targets

    assert app["project"].load_example("rail_long_part")
    s = app["session"]
    spec = _spec(s)
    checker = spec.build()
    t = world_targets(s)
    T = []
    for dx in (0.0, 1.0, 2.0):  # three stops along the rail
        Tk = s.T_world_base.copy()
        Tk[0, 3] += dx
        T.append(Tk)
    calls = [
        (
            Tk,
            kind,
            int(t.ids[i]),
            t.P_part[i],
            t.N_part[i],
            t.groups[i],
            40 if kind == "configs" else None,
            True,
        )
        for Tk in T
        for i in range(0, len(t.ids), 3)
        for kind in ("point", "configs")
    ]
    pool = CheckPool(spec, 2)
    try:
        par = pool.run(checker.settings, calls)
    finally:
        pool.shutdown()
    for c, r in zip(calls, par, strict=True):
        Tk, kind, pid, p, n, g, mp_, _ = c
        checker.set_base(Tk)
        ref = (
            checker._check_point(pid, p, n, g, mp_, True)
            if kind == "point"
            else checker._check_point_configs(pid, p, n, g, mp_)
        )
        if kind == "point":
            assert (r.status, r.reason) == (ref.status, ref.reason)
            assert (r.q is None) == (ref.q is None) and (r.q is None or np.allclose(r.q, ref.q))
        else:
            assert sorted(r) == sorted(ref)
            assert all(np.allclose(r[k].q, ref[k].q) for k in r)


@pytest.mark.slow
def test_plan_with_check_workers_equals_serial(app):
    st, s = app["stations"], app["session"]
    assert app["project"].load_example("rail_long_part")

    def plan(workers):
        jobs.CHECK_WORKERS = workers
        confirm_all(st)
        st.start(block=True)
        return [(r.label, round(r.cost, 6), len(r.zones), sorted(r.assignment.items())) for r in s.results]

    ref = plan(1)
    assert plan(4) == ref
    assert plan(4) == ref  # the pool is reused
