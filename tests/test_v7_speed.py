"""v7 round 2, phase A (docs/V7_PLAN.md T1-T4): faster zoning checks with identical results."""

import numpy as np
import pytest
import viser

from core.collision import CollisionScene
from core.configs import config_label
from core.feasibility import FeasibilityChecker
from core.ik import make_ik
from core.robot import Robot
from tests.helpers import confirm_all
from ui.stations import world_targets


@pytest.fixture(scope="module")
def rail():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8781, verbose=False)
    app = build_app(server)
    assert app["project"].load_example("rail_long_part")
    s = app["session"]
    sites = s.sites()
    robot = Robot(s.robot.spec)
    bb, wb = sites.scene_bodies()
    T0, Tp = s.T_world_base.copy(), s.part.T_world_part.copy()
    scene = CollisionScene(robot, s.tool_model, s.part_mesh(), T0, Tp, base_bodies=bb, world_bodies=wb)
    checker = FeasibilityChecker(robot, make_ik(robot), s.tool_model, scene, T0, Tp)
    yield checker, world_targets(s), sites
    server.stop()


def _old_configs(checker, pid, p, n, g, max_poses=40):
    """check_point_configs before T1/T2: every branch collision-checked, clearance for each."""
    out = {}
    for r in checker._feasible(pid, p, n, g, max_poses, [0], all_branches=True):
        lab = config_label(checker.ik, r.q)
        if lab not in out:
            r.config = lab
            out[lab] = r
            if len(out) == checker.ik.n_branches:
                break
    return out


class _Count:
    def __init__(self, scene):
        self.scene, self.n = scene, {"collides": 0, "part_clearance": 0}
        for name in self.n:
            fn = getattr(scene, name)

            def wrap(*a, _fn=fn, _name=name, **k):
                self.n[_name] += 1
                return _fn(*a, **k)

            setattr(scene, name, wrap)

    def reset(self):
        for name in self.n:
            delattr(self.scene, name)


@pytest.mark.slow
def test_configs_unchanged_and_fewer_checks(rail):
    checker, t, sites = rail
    lo, hi = sites.c.bounds[0]
    pts = range(0, len(t.ids), 3)
    calls = {}
    for impl in ("old", "new"):
        count = _Count(checker.scene)
        got = []
        try:
            for s in np.linspace(lo, hi, 3):
                checker.set_base(sites.T({"s": float(s)}))
                for i in pts:
                    args = (int(t.ids[i]), t.P_part[i], t.N_part[i], t.groups[i])
                    got.append(
                        _old_configs(checker, *args) if impl == "old" else checker.check_point_configs(*args)
                    )
        finally:
            count.reset()
        calls[impl] = (dict(count.n), got)
    old, new = calls["old"][1], calls["new"][1]
    assert sum(len(d) for d in old) > 0
    for a, b in zip(old, new, strict=True):
        assert list(a) == list(b)  # same configurations, same order
        for lab in a:
            ra, rb = a[lab], b[lab]
            assert np.array_equal(ra.q, rb.q) and ra.tcp == rb.tcp and ra.status == rb.status
            assert ra.clearance == pytest.approx(rb.clearance, abs=1e-12)
            assert (ra.spin_deg, ra.tilt_deg, ra.standoff) == (rb.spin_deg, rb.tilt_deg, rb.standoff)
    n_old, n_new = calls["old"][0], calls["new"][0]
    assert n_new["collides"] < 0.5 * n_old["collides"], (n_old, n_new)
    # T2: the clearance only for each configuration's kept solution (at most 8 per point)
    assert n_new["part_clearance"] <= 8 * len(new), (n_old, n_new)
    assert n_new["part_clearance"] == sum(len(d) for d in new)


class _NoMemo:
    def get(self, checker, call, compute):
        return compute()


def _plan(app):
    import time

    st, s = app["stations"], app["session"]
    t0 = time.monotonic()
    confirm_all(st)
    st.start(block=True)
    return time.monotonic() - t0, [(len(r.stations), tuple(r.unreached), len(r.zones)) for r in s.results]


@pytest.mark.slow
def test_memo_keeps_plans(monkeypatch):
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8782, verbose=False)
    try:
        app = build_app(server)
        assert app["project"].load_example("rail_long_part")
        _plan(app)  # warm up (reachability map, meshes)
        _, with_memo = _plan(app)
        monkeypatch.setattr("opt.placement.CheckMemo", _NoMemo)
        _, without = _plan(app)
    finally:
        server.stop()
    assert with_memo and sorted(with_memo) == sorted(without)
