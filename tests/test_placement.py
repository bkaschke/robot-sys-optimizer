"""M7 checks. Done-when: on a scene designed to need 2 stations, the planner finds 2 and
full coverage."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from core import golden_parts as gp
from core.collision import CollisionScene
from core.feasibility import FeasibilityChecker, PointResult
from core.ik import make_ik
from core.robot import Robot
from core.schema import Solution, Station, TargetGroup, Weights
from core.tool import ToolModel, load_tool
from core.transforms import make_T
from opt.cost import cost_breakdown, path_length
from opt.placement import (
    PlannerSettings,
    Targets,
    candidate_bases,
    greedy_cover,
    min_stations,
    order_stations,
    plan_stations,
    set_cover,
)
from opt.reachability import ReachMap, coverage_matrix

REPO = Path(__file__).resolve().parents[1]


# --- small units ----------------------------------------------------------------------------


def test_set_cover_finds_minimum():
    # 4 targets; base 0 covers {0,1}, base 1 covers {2,3}, base 2 covers {1,2}, base 3 covers {0}
    cov = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0], [1, 0, 0, 0]], bool)
    mu = np.ones(cov.shape) * 0.5
    assert sorted(set_cover(cov, mu, 2)) == [0, 1]
    assert min_stations(cov) == 2
    part = set_cover(cov, mu, 1)  # max coverage with one base
    assert len(part) == 1 and cov[part[0]].sum() == 2


def test_set_cover_prefers_quality_at_equal_coverage():
    cov = np.array([[1, 1], [1, 1]], bool)
    mu = np.array([[0.2, 0.2], [0.9, 0.8]])
    assert set_cover(cov, mu, 1) == [1]


def test_greedy_cover():
    cov = np.array([[1, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]], bool)
    assert greedy_cover(cov, np.zeros(cov.shape), 3) == [0, 1]


def test_order_stations_shortest_open_path():
    xy = np.array([[0, 0], [3, 0], [1, 0], [2, 0]], float)
    order = order_stations(xy, start=np.array([-1.0, 0]))
    assert order == [0, 2, 3, 1]
    assert path_length(xy[order]) == pytest.approx(3.0)


def test_cost_breakdown_terms():
    w = Weights()
    ok = PointResult(0, "reachable", clearance=0.05, manipulability=0.5, torque_util=0.3, limit_margin_deg=40)
    cost, terms = cost_breakdown([ok], 2, np.array([[0, 0], [1, 0]]), w, max_manip=1.0)
    assert terms["unreached"] == pytest.approx(2 * w.unreached)
    assert terms["stations"] == pytest.approx(w.stations)
    assert terms["travel"] == pytest.approx(w.travel)
    assert terms["manip"] == pytest.approx(w.manip / 1.5)
    assert terms["clearance"] == 0 and terms["limits"] == 0
    assert cost == pytest.approx(sum(terms.values()))


def test_solution_schema_rejects_bad_station_index():
    with pytest.raises(ValidationError):
        Solution(stations=[Station(xyz=(0, 0, 0))], assignment={3: 1})


def test_candidates_avoid_part_footprint():
    t = Targets(
        np.arange(1), np.array([[0.0, 0.0, 0.5]]), np.array([[0, 0, 1.0]]), None, None, [], np.zeros(1)
    )
    fp = np.array([[-0.5, -0.5], [0.5, 0.5]])
    st = PlannerSettings(candidate_step=0.1, search_margin=1.0, footprint_margin=0.2)
    b = candidate_bases(t, fp, 0.0, st)
    assert not np.any(np.all((b[:, :2] > -0.7 + 1e-9) & (b[:, :2] < 0.7 - 1e-9), axis=1))
    assert len(b) > 0


# --- done-when: long rail needs 2 stations ------------------------------------------------------


@pytest.fixture(scope="module")
def rail_scene():
    robot = Robot.from_registry("ur20")
    ee = load_tool("orbital_sander")
    tm = ToolModel(ee, REPO / ee.urdf)
    rmap = ReachMap.load_or_build(robot, tm.T_flange_tcp("pad"))
    T_part = make_T(p=(0, 0.8, 0.5))  # 6 m rail, top face at z = 0.65
    part = gp.long_rail()
    checker = FeasibilityChecker(
        robot, make_ik(robot), tm, CollisionScene(robot, tm, part, np.eye(4), T_part), np.eye(4), T_part
    )
    xs = np.arange(-2.5, 2.501, 0.25)  # 21 points over 5 m: beyond one base (reach ~2.1 m)
    Pp = np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, 0.15)])
    Np = np.tile([0, 0, 1.0], (len(xs), 1))
    g = TargetGroup(name="top", tilt_deg=10.0)
    t = Targets(np.arange(len(xs)), Pp + T_part[:3, 3], Np, Pp, Np, [g] * len(xs), np.full(len(xs), 10.0))
    fp = np.array([part.bounds[0, :2] + T_part[:2, 3], part.bounds[1, :2] + T_part[:2, 3]])
    return rmap, checker, t, fp


def test_one_station_cannot_cover_the_rail(rail_scene):
    rmap, checker, t, fp = rail_scene
    st = PlannerSettings()
    bases = candidate_bases(t, fp, 0.0, st)
    score, _ = coverage_matrix(rmap, t.P_world, t.N_world, t.cones, bases)
    covered = score >= st.reach_threshold
    assert covered.sum(1).max() < len(t.ids)  # map: no base reaches all 21
    assert min_stations(covered) == 2
    # exact check from the best single base confirms it
    b = bases[np.argmax(covered.sum(1))]
    checker.set_base(make_T(p=b))
    res = [checker.check_point(int(i), t.P_part[i], t.N_part[i], t.groups[i]) for i in range(len(t.ids))]
    assert sum(r.status != "unreachable" for r in res) < len(t.ids)


def test_planner_finds_two_stations_full_coverage(rail_scene):
    rmap, checker, t, fp = rail_scene
    streamed = []
    out = plan_stations(
        rmap,
        checker,
        t,
        np.eye(4),
        fp,
        Weights(),
        PlannerSettings(max_stations=3),
        on_solution=lambda s, r: streamed.append(s),
    )
    best, results = out[0]
    assert len(best.stations) == 2
    assert best.unreached == [] and len(best.assignment) == len(t.ids)
    assert len(streamed) == len(out) >= 1  # solutions were streamed as they finished
    # every assignment was verified by the exact checker (collision + torque) from its station
    for pid, k in best.assignment.items():
        checker.set_base(make_T(p=best.stations[k].xyz))
        r = checker.check_point(pid, t.P_part[pid], t.N_part[pid], t.groups[pid])
        assert r.status != "unreachable"
        assert results[pid].status != "unreachable"
    # stations stay off the rail footprint (+ base radius)
    for s in best.stations:
        xy = np.array(s.xyz[:2])
        assert not np.all((xy > fp[0] - 0.25) & (xy < fp[1] + 0.25))
