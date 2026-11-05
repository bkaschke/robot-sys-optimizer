"""M9 checks (revised 2026-11-04 with the user: the plan's "rail needs fewer stations than a
mobile base" was an oversight; it holds for some tasks only).

Done-when, for BOTH carriers on the long-rail scene (6 m rail 0.4 m in front of a wall
keep-out), on two tasks (top face; wall-side face, where the robot reaches over the part):
  * every target reached, verified by the exact checker (collision, torque, payload curve);
  * every station keeps >= 300 mm (3D, as the plan defines it) from the part and stays out
    of the keep-out;
  * carrier DOFs inside their travel ranges;
  * mobile results hold under +-10 mm / +-1 deg localization error (built into verify);
  * optimality: the planner starts from the exact minimum set cover over ALL candidates
    (so candidate pruning never costs a station);
  * regression: station counts no worse than recorded (top face rail 2 / mobile 2,
    wall-side face rail 4 / mobile 5). Rail < mobile is not a requirement.
"""

from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest
from shapely.geometry import box

from core import golden_parts as gp
from core.carrier import Carrier
from core.collision import CollisionScene
from core.feasibility import FeasibilityChecker
from core.ik import make_ik
from core.robot import Robot
from core.schema import Base, CarrierJoints, Keepout, Pose, Primitive, Standoffs, TargetGroup, Weights
from core.standoff import Zones, carrier_ok, distance_3d, footprint
from core.tool import ToolModel, load_tool
from core.transforms import make_T
from opt.placement import PlannerSettings, Targets, min_stations, plan_stations
from opt.reachability import coverage_matrix, map_for_tool
from opt.sites import CarrierSites

REPO = Path(__file__).resolve().parents[1]
SO = Standoffs(carrier_part_min=0.3, carrier_part_preferred=0.4)
WALL = [Keepout(min_xy=(-5.0, 1.3), max_xy=(5.0, 3.0))]


def rail_base(**kw) -> Base:
    return Base(
        mode="movable",
        carrier="rail",
        pose=Pose(xyz=(0, 0.15, 0)),
        track=Primitive(shape="box", size=(7.0, 0.3, 0.1)),
        chassis=Primitive(shape="box", size=(0.5, 0.5, 0.15)),
        riser=Primitive(shape="cylinder", radius=0.15, height=0.55),
        T_mount=Pose(xyz=(0, 0, 0.7)),
        joints=CarrierJoints(x=(-3.0, 3.0)),
        standoff_m=SO,
        keepouts=WALL,
        **kw,
    )


def mobile_base(**kw) -> Base:
    return Base(
        mode="movable",
        carrier="mobile",
        chassis=Primitive(shape="box", size=(1.2, 0.8, 0.4), kg=300),
        riser=Primitive(shape="cylinder", radius=0.15, height=0.3),
        T_mount=Pose(xyz=(0.2, 0, 0.7)),
        joints=CarrierJoints(x=(-4.0, 4.0), y=(-3.0, 3.0), yaw_deg=(-180, 180)),
        standoff_m=SO,
        keepouts=WALL,
        **kw,
    )


# --- kinematics and URDF -----------------------------------------------------------------------


@pytest.mark.parametrize("make", [rail_base, mobile_base])
def test_urdf_fk_matches_carrier(make):
    base = make()
    c = Carrier(base)
    model = pin.buildModelFromXML(c.urdf())
    data = model.createData()
    rng = np.random.default_rng(1)
    fid = model.getFrameId("mount")
    for _ in range(10):
        v = {n: float(rng.uniform(lo, hi)) for n, (lo, hi) in zip(c.dofs, c.bounds, strict=True)}
        q = np.array([v[n] for n in c.dofs])
        pin.framesForwardKinematics(model, data, q)
        assert np.allclose(data.oMf[fid].homogeneous, c.T_world_base(v), atol=1e-9)


def test_gantry_inverted_mount():
    base = Base(
        mode="movable",
        carrier="gantry",
        pose=Pose(xyz=(0, 0, 2.5)),
        chassis=Primitive(shape="box", size=(0.4, 0.4, 0.1)),
        T_mount=Pose(rpy=(np.pi, 0, 0)),
        joints=CarrierJoints(x=(0, 3), y=(-1, 1)),
    )
    c = Carrier(base)
    T = c.T_world_base({"x": 1.0, "y": 0.5})
    assert np.allclose(T[:3, 3], [1.0, 0.5, 2.5]) and np.allclose(T[:3, 2], [0, 0, -1])
    assert c.travel({"x": 0, "y": 0}, {"x": 2, "y": 1}) == 2  # both axes move together


def test_values_for_base_inverts():
    c = Carrier(mobile_base())
    v = {"x": 1.2, "y": -0.7, "yaw": 0.6}
    assert c.values_for_base(c.T_world_base(v)) == pytest.approx(v)
    r = Carrier(rail_base())
    assert r.values_for_base(r.T_world_base({"s": -1.4}))["s"] == pytest.approx(-1.4)


# --- zones -------------------------------------------------------------------------------------


def test_standoff_and_keepout():
    part_fp = box(0, 0, 1, 1)
    zones = Zones(Base(standoff_m=SO, keepouts=[Keepout(min_xy=(5, 5), max_xy=(6, 6))]), part_fp)
    assert not zones.check(box(1.2, 0, 2, 1)).ok  # 0.2 m away < 0.3 min
    ok = zones.check(box(1.35, 0, 2, 1))
    assert ok.ok and ok.penalty == pytest.approx(0.5)  # halfway between min and preferred
    assert zones.check(box(1.5, 0, 2, 1)).penalty == 0.0
    ko = zones.check(box(4.0, 4.0, 5.2, 5.2))
    assert not ko.ok and ko.reason == "keep-out"


def test_gantry_above_part_passes_in_3d():
    """2D footprints overlap, but the overhead carriage is 1.5 m above the part."""
    part = gp.flat_plate()
    T_part = make_T(p=(1.0, 0.0, 0.5))
    base = Base(
        mode="movable",
        carrier="gantry",
        pose=Pose(xyz=(0, 0, 2.0)),
        chassis=Primitive(shape="box", size=(0.4, 0.4, 0.1)),
        T_mount=Pose(rpy=(np.pi, 0, 0)),
        joints=CarrierJoints(x=(0, 2), y=(-1, 1)),
        standoff_m=SO,
    )
    c = Carrier(base)
    zones = Zones(base, footprint(part, T_part))
    assert not zones.check(c.footprint({"x": 1.0, "y": 0.0})).ok  # 2D alone would reject
    assert carrier_ok(c, zones, {"x": 1.0, "y": 0.0}, part, T_part).ok


# --- tip-over and localization -------------------------------------------------------------------


def test_tip_over_centre_of_pressure():
    robot = Robot.from_registry("ur20")
    c = Carrier(mobile_base(tip_over=True))
    v = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    q = np.radians([0, -90, 90, -90, -90, 0])  # arm folded above the chassis
    ok, margin = c.tip_over(v, robot, q, 3.0, np.zeros(3), None, None)
    assert ok and margin > 0.2
    # pulling the tool sideways with 3 kN at 1.5 m height tips a 300 kg base
    tcp = np.array([0.0, 0.0, 1.5])
    ok, _ = c.tip_over(v, robot, q, 3.0, np.zeros(3), np.array([0, 3000.0, 0]), tcp)
    assert not ok


def test_localization_perturbations():
    sites = CarrierSites(Carrier(mobile_base(localization_m=0.01, localization_deg=1.0)), None, None)
    ps = sites.perturbations({"x": 0, "y": 0, "yaw": 0})
    assert len(ps) == 6
    assert max(abs(p["x"]) for p in ps) == pytest.approx(0.01)
    assert max(abs(p["yaw"]) for p in ps) == pytest.approx(np.radians(1.0))
    assert CarrierSites(Carrier(rail_base()), None, None).perturbations({"s": 0}) == []


def test_arm_collides_with_own_chassis():
    robot = Robot.from_registry("ur20")
    ee = load_tool("orbital_sander")
    tm = ToolModel(ee, REPO / ee.urdf)
    sites = CarrierSites(Carrier(mobile_base()), None, None)
    bb, wb = sites.scene_bodies()
    for v in ({"x": 0.0, "y": 0.0, "yaw": 0.0}, {"x": 0.7, "y": -0.4, "yaw": 1.2}):
        sc = CollisionScene(robot, tm, None, sites.T(v), None, base_bodies=bb, world_bodies=wb, floor=False)
        assert sc.collides(np.array([0, np.pi / 2, 0, 0, 0, 0]), "pad")  # upper arm straight down
        assert not sc.collides(np.radians([0, -90, 90, -90, -90, 0]), "pad")


# --- done-when ------------------------------------------------------------------------------------

TASKS = {
    # name: (point y, point z in the part frame, normal) along x in [-3, 3]
    "top": (0.0, 0.15, (0, 0, 1.0)),
    "wall_side": (0.1, 0.075, (0, 1.0, 0)),
}
REGRESSION = {("top", "rail"): 2, ("top", "mobile"): 2, ("wall_side", "rail"): 4, ("wall_side", "mobile"): 5}


@pytest.fixture(scope="module")
def rail_scene():
    robot = Robot.from_registry("ur20")
    ee = load_tool("orbital_sander")
    tm = ToolModel(ee, REPO / ee.urdf)
    rmap = map_for_tool(robot, tm.T_flange_tcp("pad"))
    T_part = make_T(p=(0, 0.8, 0.5))
    part = gp.long_rail()
    cache: dict = {}

    def targets(task: str) -> Targets:
        y, z, n = TASKS[task]
        xs = np.arange(-3.0, 3.001, 0.25)
        Pp = np.column_stack([xs, np.full_like(xs, y), np.full_like(xs, z)])
        Np = np.tile(n, (len(xs), 1))
        g = TargetGroup(name=task, tilt_deg=10.0)
        return Targets(
            np.arange(len(xs)), Pp + T_part[:3, 3], Np, Pp, Np, [g] * len(xs), np.full(len(xs), 10.0)
        )

    def plan(task: str, kind: str):
        if (task, kind) not in cache:
            base = rail_base() if kind == "rail" else mobile_base(localization_m=0.01, localization_deg=1.0)
            sites = CarrierSites(Carrier(base), part, T_part)
            bb, wb = sites.scene_bodies()  # the arm must also clear its own carriage / riser / track
            scene = CollisionScene(robot, tm, part, np.eye(4), T_part, base_bodies=bb, world_bodies=wb)
            ch = FeasibilityChecker(robot, make_ik(robot), tm, scene, np.eye(4), T_part)
            t = targets(task)
            out = plan_stations(
                rmap, ch, t, np.eye(4), None, Weights(), PlannerSettings(max_stations=5), sites=sites
            )
            cache[(task, kind)] = (out, sites, t)
        return cache[(task, kind)]

    return plan, rmap, part, T_part


@pytest.mark.parametrize("task", list(TASKS))
@pytest.mark.parametrize("kind", ["rail", "mobile"])
def test_carrier_plan_is_valid_and_optimal(rail_scene, task, kind):
    plan, rmap, part, T_part = rail_scene
    out, sites, t = plan(task, kind)
    full = [s for s, _ in out if not s.unreached]
    assert full, f"{kind}/{task}: no full-coverage solution"
    best = min(full, key=lambda s: len(s.stations))
    assert len(best.assignment) == len(t.ids)

    # standoff, keep-out and travel range at every station
    c = sites.c
    wall = box(-5, 1.3, 5, 3.0)
    for st in best.stations:
        v = {d: st.carrier[d] for d in c.dofs if d in st.carrier}
        for d, (lo, hi) in zip(c.dofs, c.bounds, strict=True):
            if d != "yaw":
                assert lo - 1e-9 <= v[d] <= hi + 1e-9, (d, v[d])
        fp = c.footprint(v)
        # 300 mm standoff, in 3D as the plan defines it (Coal distance). In this scene the rail
        # floats 0.5 m up with no stand, so a 0.4 m chassis may tuck partly under it: footprint
        # distance can be < 300 mm while the 3D distance is not. Model real fixtures as part
        # of the part mesh or as keep-outs.
        d3 = distance_3d(c.moving_bodies(v) + c.static_bodies(), part, T_part)
        assert d3 >= 0.3 - 1e-6, d3
        assert not fp.intersects(wall)
        assert np.allclose(st.xyz, c.T_world_base(v)[:3, 3])

    # optimality: the first k tried is the exact minimum cover over ALL candidates
    cand = sites.candidates(t.P_world, PlannerSettings().candidate_step)
    Ts = np.array([sites.T(v) for v in cand])
    score, _ = coverage_matrix(rmap, t.P_world, t.N_world, t.cones, Ts[:, :3, 3], Ts[0, :3, :3])
    k_full = min_stations(score >= PlannerSettings().reach_threshold)
    assert min(len(s.stations) for s, _ in out) == k_full

    # regression: never worse than the recorded count
    assert len(best.stations) <= REGRESSION[(task, kind)], f"{kind}/{task}: {len(best.stations)} stations"


# --- GUI ----------------------------------------------------------------------------------------


def test_gui_base_mode_panel():
    import viser

    from app import build_app
    from ui.carrier_ui import DEFAULTS

    server = viser.ViserServer(host="127.0.0.1", port=8771, verbose=False)
    try:
        state = build_app(server)
        panel, s = state["carrier"], state["session"]
        assert s.base.mode == "static" and "one base" in panel.info.content
        panel.load(DEFAULTS["rail"])
        assert panel.commit() and s.base.carrier == "rail"
        assert set(panel._dof) == {"s"}
        panel._dof["s"].value = 1250.0  # jog the carriage along the track
        c = Carrier(s.base)
        assert np.allclose(s.T_world_base, c.T_world_base({"s": 1.25}), atol=1e-9)
        assert np.allclose(state["cell"].base.gizmo.position, s.T_world_base[:3, 3])
        assert "Carrier footprint to part" in panel.info.content
        assert any(getattr(h, "name", "").endswith("standoff_min") for h in panel._handles)
        panel.so.value = (400.0, 300.0)  # preferred < minimum: rejected, base unchanged
        assert not panel.commit() and s.base.standoff_m.carrier_part_min == pytest.approx(0.3)
        panel.mode.value = "static"
        panel.so.value = (300.0, 400.0)
        assert panel.commit() and s.base.mode == "static" and panel._dof == {}
    finally:
        server.stop()
