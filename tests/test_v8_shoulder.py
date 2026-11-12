"""One shoulder side per plan for arms whose shoulder cannot turn a full circle (opt.shoulder),
the path tracer at joint limits (core.paths) and the home on the plan's side (core.home_search)."""

from types import SimpleNamespace

import numpy as np
import pytest

import opt.shoulder as sh
from core.feasibility import PointResult
from core.home_search import choose_home, plan_j2_sign
from core.robot import Robot
from opt.shoulder import one_side, shoulder_locked, side


def test_side_and_locked():
    assert side("S+E-W+") == "S+" and side("S-E+W-") == "S-" and side("E+W+") == ""
    assert shoulder_locked(Robot.from_registry("crx30ia")) and shoulder_locked(Robot.from_registry("gofa12"))
    assert not shoulder_locked(Robot.from_registry("ur20"))


def _r(pid, j2, mu=1.0):
    q = np.zeros(6)
    q[1] = j2
    return PointResult(pid, "reachable", q=q, tcp="tcp", manipulability=mu, limit_margin_deg=30.0)


class _Ik:
    def config_label(self, q):
        return ("S+" if q[1] > 0 else "S-") + "E+W+"


def _setup(options):
    """Two stops; ``options[stop][pid]`` = shoulder angles that point can use there."""
    state = {"k": 0}
    ik = _Ik()

    def configs(pid, *_a):
        out = {}
        for j2 in options[state["k"]].get(pid, []):
            r = _r(pid, j2)
            out[ik.config_label(r.q)] = r
        return out

    checker = SimpleNamespace(
        robot=Robot.from_registry("crx30ia"),
        ik=ik,
        set_base=lambda T: state.update(k=int(T[0, 3])),
        check_point_configs=configs,
    )
    sites = SimpleNamespace(T=lambda v: np.diag([1.0, 1, 1, 1]) + np.eye(4, k=3) * v["k"],
                            perturbations=lambda v: [])  # fmt: skip
    ids = sorted({p for o in options.values() for p in o})
    t = SimpleNamespace(
        ids=np.array(ids),
        P_part=np.zeros((len(ids), 3)),
        N_part=np.zeros((len(ids), 3)),
        groups=[None] * len(ids),
    )
    return checker, sites, t, [{"k": 0}, {"k": 1}]


def test_one_side_moves_points_to_the_kept_side(monkeypatch):
    monkeypatch.setattr(sh, "can_swing", lambda *a, **k: False)
    options = {
        0: {1: [1.0], 2: [1.0], 3: [1.0], 4: [-1.0, 1.0], 5: [-1.0]},  # 4 can switch, 5 cannot here
        1: {5: [1.0], 6: [-1.0], 7: [1.0], 8: [1.0]},  # 5 can be done here on S+, 6 nowhere
    }
    checker, sites, t, stations = _setup(options)
    assignment = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1}
    results = {
        1: _r(1, 1.0), 2: _r(2, 1.0), 3: _r(3, 1.0), 4: _r(4, -1.0), 5: _r(5, -1.0),
        6: _r(6, -1.0), 7: _r(7, 1.0), 8: _r(8, 1.0),
    }  # fmt: skip
    sides = one_side(checker, t, stations, sites, assignment, results)
    assert sides == {0: "S+", 1: "S+"}
    assert results[4].q[1] > 0 and assignment[4] == 0  # switched side at its stop
    assert results[5].q[1] > 0 and assignment[5] == 1  # moved to the other stop
    assert results[6].status == "unreachable" and results[6].reason == sh.REASON
    assert all(results[p].q[1] > 0 for p in (1, 2, 3, 7, 8))


def test_one_side_leaves_stops_that_can_swing(monkeypatch):
    monkeypatch.setattr(sh, "can_swing", lambda *a, **k: True)
    options = {0: {1: [1.0], 2: [-1.0]}, 1: {}}
    checker, sites, t, stations = _setup(options)
    results = {1: _r(1, 1.0), 2: _r(2, -1.0)}
    assert one_side(checker, t, stations, sites, {1: 0, 2: 0}, results) == {}
    assert results[2].q[1] < 0
    # a robot whose shoulder turns fully is never changed
    checker.robot = Robot.from_registry("ur20")
    monkeypatch.setattr(sh, "can_swing", lambda *a, **k: pytest.fail("not needed"))
    assert one_side(checker, t, stations, sites, {1: 0, 2: 0}, results) == {}


def test_home_on_the_plans_shoulder_side():
    r = Robot.from_registry("crx30ia")
    assert plan_j2_sign(r, [np.radians([0, -120, 0, 0, 0, 0]), np.radians([0, -100, 10, 0, 0, 0])]) == -1.0
    assert plan_j2_sign(Robot.from_registry("ur20"), [np.zeros(6)]) is None
    checker = SimpleNamespace(
        robot=r, set_base=lambda T: None, scene=SimpleNamespace(collides=lambda q, tcp=None: False)
    )
    home = np.radians([0.0, 60.0, -30.0, 0.0, 0.0, 0.0])
    assert choose_home(checker, home, [np.eye(4)], None)[0] is home
    q, note = choose_home(checker, home, [np.eye(4)], -1.0)
    assert q[1] < 0 and "other shoulder side" in note


def test_tracer_takes_the_copy_inside_the_limits():
    """A wrist joint that would pass its limit along the surface: the tracer takes the copy of
    that joint inside the limits (a jump the path then unwinds) instead of no solution."""
    from core.paths import _Tracer

    lo, hi = np.radians([-180, -180, -270, -190, -180, -225]), np.radians([180, 180, 270, 190, 180, 225])
    target = np.radians([10.0, -20.0, 30.0, 200.0 - 360.0, 40.0, 5.0])  # J4 = 200 deg = -160 deg
    ik = SimpleNamespace(solve_batch=None, solve=lambda T: [target.copy()], config_label=lambda q: "S+E+W+")
    ch = SimpleNamespace(
        ik=ik, lo=lo, hi=hi, T_base_part=np.eye(4), _T_tcp_flange={"tcp": np.eye(4)},
        scene=SimpleNamespace(collides=lambda q, tcp=None: False),
    )  # fmt: skip
    tr = _Tracer(ch, None, 0.02, 20.0)
    tr.batch = False
    prev = np.radians([10.0, -20.0, 30.0, 185.0, 40.0, 5.0])
    q, issue = tr.solve(np.eye(4), "tcp", prev, "S+E+W+")
    assert q is not None and issue == "jump"
    assert np.degrees(q[3]) == pytest.approx(-160.0)
