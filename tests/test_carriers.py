"""M9 checks. Done-when: on the long-rail scene, a 1-axis rail solution respects a 300 mm
standoff and needs fewer stations than a mobile base.

The scene is the golden 6 m rail standing 0.4 m in front of a wall (keep-out); the task is
deburring the wall-side face, so the robot must reach over the part. Both carriers carry
the UR20 at the same 0.7 m mount height and keep 300 mm from the part. The rail carriage
(0.5 m wide) runs right along the part; the mobile base (plan default 1.2 x 0.8 m chassis)
keeps its robot further away and must also work under +-10 mm / +-1 deg localization error.
(On the rail's *top* face both need 2 stations: the UR20's reach hides the difference.)
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
from core.standoff import Zones, carrier_ok, footprint
from core.tool import ToolModel, load_tool
from core.transforms import make_T
from opt.placement import PlannerSettings, Targets, plan_stations
from opt.reachability import map_for_tool
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


@pytest.fixture(scope="module")
def wall_scene():
    robot = Robot.from_registry("ur20")
    ee = load_tool("orbital_sander")
    tm = ToolModel(ee, REPO / ee.urdf)
    rmap = map_for_tool(robot, tm.T_flange_tcp("pad"))
    T_part = make_T(p=(0, 0.8, 0.5))
    part = gp.long_rail()
    xs = np.arange(-3.0, 3.001, 0.25)  # 25 points along the wall-side face
    Pp = np.column_stack([xs, np.full_like(xs, 0.1), np.full_like(xs, 0.075)])
    Np = np.tile([0, 1.0, 0], (len(xs), 1))
    g = TargetGroup(name="far face", tilt_deg=10.0)
    t = Targets(np.arange(len(xs)), Pp + T_part[:3, 3], Np, Pp, Np, [g] * len(xs), np.full(len(xs), 10.0))

    def plan(base: Base):
        sites = CarrierSites(Carrier(base), part, T_part)
        bb, wb = sites.scene_bodies()  # the arm must also clear its own carriage / riser / track
        scene = CollisionScene(robot, tm, part, np.eye(4), T_part, base_bodies=bb, world_bodies=wb)
        ch = FeasibilityChecker(robot, make_ik(robot), tm, scene, np.eye(4), T_part)
        out = plan_stations(
            rmap, ch, t, np.eye(4), None, Weights(), PlannerSettings(max_stations=5), sites=sites
        )
        full = [s for s, _ in out if not s.unreached]
        return (min(full, key=lambda s: len(s.stations)) if full else None), sites

    return plan, part, T_part, t


def test_rail_needs_fewer_stations_than_mobile(wall_scene):
    plan, part, T_part, t = wall_scene
    rail, rail_sites = plan(rail_base())
    mobile, _ = plan(mobile_base(localization_m=0.01, localization_deg=1.0))
    assert rail is not None and len(rail.assignment) == len(t.ids)
    assert mobile is None or len(rail.stations) < len(mobile.stations), (
        f"rail {len(rail.stations)} vs mobile {len(mobile.stations) if mobile else None}"
    )
    part_fp = footprint(part, T_part)
    c = rail_sites.c
    for st in rail.stations:
        v = {"s": st.carrier["s"]}
        assert -3.0 - 1e-9 <= v["s"] <= 3.0 + 1e-9
        assert c.footprint(v).distance(part_fp) >= 0.3 - 1e-6  # 300 mm standoff respected
        assert not c.footprint(v).intersects(box(-5, 1.3, 5, 3.0))  # never in the wall zone
        assert np.allclose(st.xyz, c.T_world_base(v)[:3, 3])


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
