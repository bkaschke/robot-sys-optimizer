"""v2 phase 3 (docs/V2_PLAN.md, branch v3): mobile contour stops, faster exact checker, quick
plan first. The checker changes must not change results (except exact quality ties)."""

import time

import numpy as np
import pytest
import viser

from core.ik import best_copies, expand_to_limits, make_ik
from core.ik.ur import solve_batch
from core.robot import Robot
from opt.contour import SIDES, ContourSites


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8792, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


# --- checker internals ---------------------------------------------------------------------------


def test_best_copies_matches_expand_to_limits_margin():
    robot = Robot.from_registry("ur20")
    m = np.radians(5.0)
    lo, hi = robot.lower + m, robot.upper - m
    rng = np.random.default_rng(0)
    for _ in range(300):
        s = rng.uniform(-np.pi, np.pi, 6)
        q, ok = best_copies(s[None], lo, hi, robot.lower, robot.upper)
        ex = expand_to_limits(s[None], lo, hi)
        assert ok[0] == (len(ex) > 0)
        if len(ex):
            best_old = np.minimum(ex - robot.lower, robot.upper - ex).min(axis=1).max()
            best_new = np.minimum(q[0] - robot.lower, robot.upper - q[0]).min()
            assert best_new == pytest.approx(best_old, abs=1e-12)  # same margin (ties may differ)
            assert np.allclose(np.mod(q[0] - s + np.pi, 2 * np.pi) - np.pi, 0, atol=1e-9)


def test_batch_ik_path_equals_scalar_near_singularities():
    """Poses at and near the wrist singularity (q5 -> 0): the batched path (solve_batch, with
    the scalar fallback for flagged poses) yields exactly the scalar solver's solutions."""
    from types import SimpleNamespace

    from core.feasibility import FeasibilityChecker

    robot = Robot.from_registry("ur20")
    ik = make_ik(robot)
    ch = SimpleNamespace(ik=ik, batch_ik=True)
    poses = []
    for q5 in (0.0, 1e-12, 1e-10, 5e-10, 1e-9, 1e-7, 0.3):
        T = robot.fk(np.array([0.3, -1.2, 1.1, -0.5, q5, 0.4]))
        poses.append(("tcp", 0.0, 0.0, 0.0, T))
    stream = list(FeasibilityChecker._ik_stream(ch, iter(poses)))
    for (_, sols), pose in zip(stream, poses, strict=True):
        ref = ik.solve(pose[-1])
        key = lambda a: np.round(np.asarray(a), 9).tolist()  # noqa: E731
        assert sorted(key(s) for s in sols) == sorted(key(s) for s in ref)
    _, _, sing = solve_batch(ik, np.array([p[-1] for p in poses]), return_singular=True)
    assert not sing[-1]  # a regular pose is not deferred


def test_batched_checker_equals_scalar(app):
    from core.collision import CollisionScene
    from core.feasibility import FeasibilityChecker
    from core.transforms import make_T

    s = app["session"]
    assert app["project"].load_example("rail_long_part")
    groups = {g.name: g for g in s.targets.groups}
    T = make_T(p=(0.0, -0.9, 0.7))
    out = {}
    for batch in (False, True):
        scene = CollisionScene(s.robot, s.tool_model, s.part_mesh(), T, s.part.T_world_part)
        ch = FeasibilityChecker(s.robot, make_ik(s.robot), s.tool_model, scene, T, s.part.T_world_part)
        ch.batch_ik = batch
        out[batch] = [
            ch.check_point(p.id, np.array(p.p), np.array(p.n), groups[p.group]) for p in s.targets.points
        ]
    for a, b in zip(out[False], out[True], strict=True):
        assert (a.status, a.reason, a.tcp, a.spin_deg, a.tilt_deg) == (
            b.status,
            b.reason,
            b.tcp,
            b.spin_deg,
            b.tilt_deg,
        )
        if a.q is not None:
            assert np.allclose(a.q, b.q, atol=1e-9) and a.clearance == pytest.approx(b.clearance, abs=1e-9)


# --- mobile contour ---------------------------------------------------------------------------------


def _contour(s):
    s.base = s.base.model_copy(update={"facing": "front", "fixed_standoff_m": 0.4})
    sites = s.sites()
    assert isinstance(sites, ContourSites)
    return sites


def test_contour_keeps_the_gap_and_faces_the_part(app):
    s = app["session"]
    assert app["project"].load_example("mobile_amr")
    sites = _contour(s)
    c, part = sites.c, sites.zones.part
    side = np.array(SIDES["front"])
    for t in np.linspace(0, sites.bounds[0, 1], 40, endpoint=False):
        v = sites.full({"t": float(t)})
        assert c.footprint(v).distance(part) == pytest.approx(
            0.4, abs=0.002
        )  # exact on samples, 2 mm between
        R = np.array([[np.cos(v["yaw"]), -np.sin(v["yaw"])], [np.sin(v["yaw"]), np.cos(v["yaw"])]])
        from shapely.geometry import Point

        near = part.exterior.interpolate(part.exterior.project(Point(v["x"], v["y"])))
        to_part = np.array([near.x - v["x"], near.y - v["y"]])
        cosang = (R @ side) @ to_part / np.linalg.norm(to_part)
        assert cosang > np.cos(np.radians(5))  # the chosen side looks at the part


def test_contour_plan_full_coverage_fast_and_safe(app):
    s = app["session"]
    assert app["project"].load_example("mobile_amr")  # the example now uses the contour
    assert s.base.facing == "front" and s.base.fixed_standoff_m == pytest.approx(0.4)
    seen = []
    st = app["stations"]
    orig = st.on_solution
    st.on_solution = lambda: (seen.append(len(s.results)), orig())
    t0 = time.perf_counter()
    st.start(block=True)
    dt = time.perf_counter() - t0
    st.on_solution = orig
    best = min(s.results, key=lambda r: (len(r.unreached), r.cost))
    assert not best.unreached and len(best.stations) <= 3
    sites = s.sites()
    for stn in best.stations:
        gap = sites.c.footprint(stn.carrier).distance(sites.zones.part)
        assert gap == pytest.approx(0.4, abs=0.002)
        assert not stn.carrier or "t" in stn.carrier
    # one plan per station count in the list (refined plans replace quick ones)
    ks = [len(r.stations) for r in s.results]
    assert len(ks) == len(set(ks))
    assert dt < 40.0, f"plan took {dt:.1f} s"  # about 12 s on the dev PC (47 s in v1)


def test_quick_plans_stream_before_refinement():
    from opt.placement import PlannerSettings

    st = PlannerSettings()
    assert st.quick_first and st.refine_top >= 1


def test_gui_facing_fields(app):
    car, s = app["carrier"], app["session"]
    assert app["project"].load_example("mobile_amr")
    box = car.side_boxes  # v3 feedback: robot sides are checkboxes (several allowed)
    assert box["front"].visible and box["front"].value and car.fixed_so.visible and not car.so.visible
    box["front"].value = False  # none ticked: free heading search (v1)
    assert not car.fixed_so.visible and car.so.visible
    assert car.commit() and s.base.robot_sides() == [] and s.base.fixed_standoff_m is None
    box["left"].value = True
    car.fixed_so.value = 500.0
    assert car.commit() and s.base.robot_sides() == ["left"] and s.base.fixed_standoff_m == pytest.approx(0.5)
    car.kind.value = "rail"
    assert not box["front"].visible and not car.fixed_so.visible and car.align.visible
