"""M8 checks. Done-when: on a scene with a known minimal L (a pocket of known depth), the
result is within 5 mm.

Scene: golden `deep_hole`, a 0.1 m square hole 0.3 m deep, narrower than the UR20 wrist_3
(radius 55.7 mm), so the wrist must stay `clearance` above the rim while a thin straight
probe (capsule radius 20 mm) reaches down. Target: hole bottom, tool axis vertical
(tilt 0), standoff s = 50 mm. Then

    L_min = depth - s + clearance = 0.30 - 0.05 + 0.01 = 0.260 m

(wrist_3's lowest point sits 0.3 mm behind the flange face, so the exact value is 259.7 mm).
"""

from pathlib import Path

import numpy as np
import pytest

from core import golden_parts as gp
from core.collision import CollisionScene
from core.feasibility import FeasibilityChecker
from core.ik import make_ik
from core.robot import Robot
from core.schema import TCP, EndEffector, Extension, ExtensionValues, TargetGroup, Weights
from core.tool import ToolModel, load_tool
from core.transforms import make_T
from opt.placement import Targets
from opt.reachability import OffsetMap, ReachMap, is_straight, map_for_tool
from opt.tool_geometry import GeometrySettings, optimize_tool

REPO = Path(__file__).resolve().parents[1]
CLEARANCE = 0.01
STANDOFF = 0.05


@pytest.fixture(scope="module")
def robot():
    return Robot.from_registry("ur20")


def _probe(L: float = 0.3) -> EndEffector:
    return EndEffector(
        name="probe",
        process="scanning",
        extension=Extension(L=(0.05, 0.6), radius=0.02, nominal=ExtensionValues(L=L)),
        tcps=[TCP(name="tip")],
    )


@pytest.fixture(scope="module")
def hole_scene():
    T_part = make_T(p=(0.8, 0.0, 0.0))
    part = gp.deep_hole()
    g = TargetGroup(name="hole", tilt_deg=0.0, standoff_m=STANDOFF)
    Pp, Np = np.array([[0, 0, gp.HOLE_FLOOR]]), np.array([[0, 0, 1.0]])
    t = Targets(np.array([0]), Pp + T_part[:3, 3], Np, Pp, Np, [g], np.array([0.0]))
    b = part.bounds
    fp = np.array([b[0, :2] + T_part[:2, 3], b[1, :2] + T_part[:2, 3]])
    return part, T_part, t, fp


def test_minimal_L_in_deep_hole(robot, hole_scene):
    part, T_part, t, fp = hole_scene
    res = optimize_tool(robot, _probe(), None, part, T_part, np.eye(4), t, fp, Weights(), 1)
    expected = gp.HOLE_DEPTH - STANDOFF + CLEARANCE
    assert res.feasible, res.message
    assert abs(res.values.L - expected) <= 0.005, (
        f"L = {res.values.L * 1000:.1f} mm, expected {expected * 1000:.1f}"
    )
    assert res.solution.unreached == [] and len(res.solution.stations) == 1
    # the coverage-vs-length samples flip from infeasible to feasible exactly once
    flags = [s.feasible for s in res.samples]
    assert flags == sorted(flags)

    # independent exact check at the chosen station: slightly shorter fails, the result passes
    st = res.solution.stations[0]
    T_base = make_T(p=st.xyz)
    for L, ok in ((res.values.L - 0.003, False), (res.values.L, True)):
        tm = ToolModel(_probe(L), None)
        ch = FeasibilityChecker(
            robot, make_ik(robot), tm, CollisionScene(robot, tm, part, T_base, T_part), T_base, T_part
        )
        r = ch.check_point(0, t.P_part[0], t.N_part[0], t.groups[0])
        assert (r.status != "unreachable") == ok, (L, r.reason)


def test_unreachable_range_reported(robot, hole_scene):
    part, T_part, t, fp = hole_scene
    ee = _probe()
    ee = ee.model_copy(
        update={
            "extension": ee.extension.model_copy(update={"L": (0.05, 0.2), "nominal": ExtensionValues(L=0.1)})
        }
    )
    res = optimize_tool(
        robot, ee, None, part, T_part, np.eye(4), t, fp, Weights(), 1, GeometrySettings(coarse_samples=3)
    )
    assert not res.feasible and "no tool length" in res.message


# --- shared flange map ---------------------------------------------------------------------


def test_straight_tool_uses_shared_flange_map(robot):
    ee = load_tool("orbital_sander")
    tm = ToolModel(ee, REPO / ee.urdf)
    assert is_straight(tm.T_flange_tcp("pad"))
    m = map_for_tool(robot, tm.T_flange_tcp("pad"))
    assert isinstance(m, OffsetMap) and m.offset == pytest.approx(0.02 + 0.15 + 0.17)
    bent = tm.T_flange_tcp("pad", ExtensionValues(L=0.15, theta_deg=30))
    assert not is_straight(bent)


def test_offset_map_zero_is_flange_map(robot):
    flange = ReachMap.load_or_build(robot, np.eye(4))
    om = OffsetMap(flange, 0.0)
    rng = np.random.default_rng(3)
    p = rng.uniform(-1.5, 1.5, (500, 3))
    a = rng.normal(size=(500, 3))
    assert np.allclose(
        om.lookup(p, a, 5.0)[0], flange.lookup(p, a / np.linalg.norm(a, axis=1, keepdims=True), 5.0)[0]
    )


def test_offset_map_moves_tcp_along_axis(robot):
    """A TCP 0.3 m beyond the flange is reachable exactly where the flange 0.3 m back is."""
    flange = ReachMap.load_or_build(robot, np.eye(4))
    om = OffsetMap(flange, 0.3)
    p = np.array([[0.9, 0.2, 0.2]])
    a = np.array([[0.0, 0.0, -1.0]])
    assert np.allclose(om.lookup(p, a)[0], flange.lookup(p - 0.3 * a, a)[0])


# --- stage B ----------------------------------------------------------------------------------


def test_shape_stage_respects_bounds_and_never_worsens(robot):
    ee = load_tool("orbital_sander")  # bent extension: theta in [0, 45] deg
    T_part = make_T(p=(0.9, 0.0, 0.3))
    part = gp.flat_plate()
    g = TargetGroup(name="top", tilt_deg=10.0)
    Pp = np.array([(x, 0.0, 0.02) for x in (-0.3, 0.0, 0.3)])
    Np = np.tile([0, 0, 1.0], (3, 1))
    t = Targets(np.arange(3), Pp + T_part[:3, 3], Np, Pp, Np, [g] * 3, np.full(3, 10.0))
    b = part.bounds
    fp = np.array([b[0, :2] + T_part[:2, 3], b[1, :2] + T_part[:2, 3]])
    st = GeometrySettings(coarse_samples=3, shape_evals=12)
    res = optimize_tool(robot, ee, REPO / ee.urdf, part, T_part, np.eye(4), t, fp, Weights(), 1, st)
    assert res.feasible
    assert 0.0 <= res.values.theta_deg <= 45.0
    straight = optimize_tool(
        robot,
        ee,
        REPO / ee.urdf,
        part,
        T_part,
        np.eye(4),
        t,
        fp,
        Weights(),
        1,
        GeometrySettings(coarse_samples=3, shape_evals=0),
    )
    assert res.solution.cost <= straight.solution.cost + 1e-9
    assert res.solution.tool_values == res.values


def test_gui_finds_and_applies_shortest_tool():
    import viser

    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8770, verbose=False)
    try:
        state = build_app(server)
        s, cell, tab = state["session"], state["cell"], state["targets"]
        s.set_tool(_probe())  # straight probe, no head geometry
        state["tool"].refresh()
        cell.part_choice.value = "deep_hole"
        cell.part_pose.xyz.value = (800.0, 0.0, 0.0)
        from core import targets as tg

        s.targets = s.targets.model_copy(update={"groups": [TargetGroup(name="hole", standoff_m=STANDOFF)]})
        s.targets = tg.append_points(
            s.targets, np.array([[0, 0, gp.HOLE_FLOOR]]), np.array([[0, 0, 1.0]]), "hole"
        )
        tab.refresh()
        panel = state["tool_opt"]
        panel.max_stations.value = 1
        panel.start(block=True)
        assert panel.result.feasible, panel.info.content
        assert abs(panel.result.values.L - 0.26) <= 0.005
        assert "minimal L" in panel.info.content and not panel.apply.disabled
        assert s.results and s.results[0].tool_values is not None
        panel.apply_values(panel.result.values)
        assert s.tool.extension.nominal.L == pytest.approx(panel.result.values.L)
        assert s.tool_model.T_flange_tcp("tip")[2, 3] == pytest.approx(panel.result.values.L)
    finally:
        server.stop()
