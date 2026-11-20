"""Review v8 fixes, work package I: M5 (contour MultiPolygon), L3 (deterministic numeric IK),
L4 (CSV BOM), M7 (tool evaluator cost terms), R17 (quick vs refined plan lookup)."""

from types import SimpleNamespace

import numpy as np
import pytest

from core.targets import parse_csv
from tests.test_placement import rail_scene  # noqa: F401  (fixture)


def test_contour_sites_accept_a_multipolygon_footprint():  # M5
    from core.carrier import Carrier
    from core.golden_parts import flat_plate
    from core.schema import Base, Pose, Primitive
    from core.transforms import make_T
    from opt.contour import ContourSites

    base = Base(
        mode="movable",
        carrier="mobile",
        joints={"x": (-5, 5), "y": (-5, 5)},
        bodies=[
            Primitive(shape="box", size=(0.2, 0.2, 0.2), pose=Pose(xyz=(-0.4, 0, 0)), name="a"),
            Primitive(shape="box", size=(0.2, 0.2, 0.2), pose=Pose(xyz=(0.4, 0, 0)), name="b"),
        ],
        facing="front",
        fixed_standoff_m=0.4,
    )
    carrier = Carrier(base)
    assert carrier.local_footprint().geom_type == "MultiPolygon"
    sites = ContourSites(carrier, flat_plate(), make_T(p=(2, 0, 0.5)))
    assert len(sites.table) > 0
    # the facing edge is the front of the front body: x = 0.5
    assert sites.sides == ["front"]


def test_numeric_ik_is_deterministic_without_rng():  # L3
    from core.ik.numeric import NumericIK
    from core.robot import Robot

    robot = Robot.from_registry("ur20")
    ik = NumericIK(robot, n_starts=6)
    q = np.array([0.3, -1.2, 1.4, -0.5, 0.9, 0.2])
    T = robot.fk(q)
    a, b = ik.solve(T), ik.solve(T)
    assert a.shape == b.shape and a.shape[0] >= 1
    assert np.array_equal(a, b)
    # an explicit rng still works
    c = ik.solve(T, rng=np.random.default_rng(5))
    assert c.ndim == 2


def test_parse_csv_bom_without_header_keeps_first_row():  # L4
    rows = parse_csv("﻿1,2,3,0,0,1\n4,5,6,0,0,1\n")
    assert len(rows) == 2
    assert np.allclose(rows[0][0], [1, 2, 3])
    assert len(parse_csv("﻿x,y,z,nx,ny,nz\n1,2,3,0,0,1\n")) == 1  # header still skipped


def test_tool_evaluator_cost_has_travel_and_standoff_terms(monkeypatch):  # M7
    import opt.cost
    import opt.shoulder
    import opt.tool_eval as te
    import opt.zoning
    from core.schema import Solution, Station, Weights
    from opt.station_checks import path_travel

    class Sites:
        kind = "rail"
        dofs = ["s"]

        def travel(self, a, b):
            return 3.0 * abs(a["s"] - b["s"])  # differs from the straight-line distance

        def accept(self, v):
            return True, 0.25

    seen = {}

    def fake_cost(*a, **kw):
        seen.update(kw)
        return 1.0, {}

    monkeypatch.setattr(te, "cost_breakdown", fake_cost)
    monkeypatch.setattr(te, "verify", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(opt.shoulder, "one_side", lambda *a, **k: None)
    monkeypatch.setattr(opt.zoning, "stop_sequences", lambda *a, **k: [])
    monkeypatch.setattr(opt.zoning, "config_changes", lambda *a, **k: 0)
    monkeypatch.setattr(opt.zoning, "arm_motion", lambda *a, **k: 0.0)
    monkeypatch.setattr(opt.cost, "cycle_estimate", lambda *a, **k: {"total": 0.0})
    monkeypatch.setattr(te, "wrap_excess", lambda *a, **k: 0.0)
    ev = object.__new__(te._Evaluator)
    ev.sites, ev.w, ev.L_ref, ev.max_manip, ev.seen, ev.family = Sites(), Weights(), 1.0, 1.0, [], []
    ev.t = SimpleNamespace(ids=np.array([0]))
    ev.st = SimpleNamespace(
        planner=SimpleNamespace(
            zoning=False, zoning_poses=1, cycle=None, speed_zones=None, wrap_limit_deg=0.0
        ),
        bend_step=0,
    )
    ev.tm = SimpleNamespace(length=lambda v: 0.1)
    ev.checker = lambda v: SimpleNamespace(robot=None)
    sol = Solution(
        label="x",
        stations=[
            Station(xyz=(0.0, 0.0, 0.0), yaw=0.0, carrier={"s": 0.0}),
            Station(xyz=(1.0, 0.0, 0.0), yaw=0.0, carrier={"s": 1.0}),
        ],
        assignment={0: 0},
        unreached=[],
        cost=0.0,
        breakdown={},
    )
    ev.cost_at_stations(object(), sol)
    stops = [{"s": 0.0}, {"s": 1.0}]
    assert path_travel(stops, Sites()) == 3.0
    assert seen["travel"] == pytest.approx(3.0)
    assert seen["standoff_penalty"] == pytest.approx(0.25)


@pytest.mark.slow
def test_refined_plan_is_compared_with_quick_plan_of_same_station_count(rail_scene, monkeypatch):  # noqa: F811
    import opt.placement as pl
    import opt.sampling as sampling

    rmap, checker, t, fp = rail_scene
    real_repair = sampling.repair_cover
    monkeypatch.setattr(sampling, "sample_size", lambda *a, **k: 5)  # m < n: repair_cover runs

    def repair(covered, mu, chosen):  # the repair adds a stop: k=2 gives 3 stations, k=3 gives 4
        chosen = real_repair(covered, mu, chosen)
        extra = [i for i in range(len(covered)) if i not in chosen]
        return chosen + extra[:1]

    monkeypatch.setattr(sampling, "repair_cover", repair)
    # refinement makes every stop worse (moved far along the rail)
    monkeypatch.setattr(pl, "refine_station", lambda c, t_, idx, v, *a, **k: {**v, "x": v["x"] + 4.0})
    streamed = []
    out = pl.plan_stations(
        rmap,
        checker,
        t,
        np.eye(4),
        fp,
        pl.Weights(),
        pl.PlannerSettings(max_stations=4, patience=1),
        on_solution=lambda s, r: streamed.append(s),
    )

    def rank(s):
        return (bool(s.violations), len(s.unreached), s.cost)

    for sol, _ in out:
        same = [s for s in streamed if len(s.stations) == len(sol.stations)]
        assert rank(sol) <= min(rank(s) for s in same), "a worse refined plan replaced a better quick one"
