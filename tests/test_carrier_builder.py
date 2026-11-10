"""Carrier builder: carriers made of several boxes / cylinders / meshes (External Axis tab).

Checks: legacy chassis/riser/track convert to the same geometry; multi-body carriers give
the right support polygon, URDF, mount snapping, collision and standoff; mesh
bodies load with units; the GUI editor adds, edits, drags, snaps, uploads and round-trips
through the project file; the planner works with a multi-body mobile robot.
"""

from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest
import trimesh
from pydantic import ValidationError

from core import golden_parts as gp
from core.carrier import Carrier, body_list, body_top, migrate
from core.collision import CollisionScene
from core.robot import Robot
from core.schema import Base, CarrierJoints, Pose, Primitive, Standoffs
from core.standoff import distance_3d
from core.tool import ToolModel, load_tool
from core.transforms import make_T
from tests.test_carriers import mobile_base, rail_base

REPO = Path(__file__).resolve().parents[1]


def amr(**kw) -> Base:
    """A small AMR: chassis on four wheels, a column on top, robot mounted on the column."""
    wheel = dict(shape="cylinder", radius=0.1, height=0.08)
    wheels = [
        Primitive(name=f"wheel{i}", pose=Pose(xyz=(x, y, 0.1), rpy=(np.pi / 2, 0, 0)), **wheel)
        for i, (x, y) in enumerate(((0.45, 0.36), (0.45, -0.28), (-0.45, 0.36), (-0.45, -0.28)))
    ]
    return Base(
        mode="movable",
        carrier="mobile",
        bodies=[
            *wheels,
            Primitive(name="chassis", shape="box", size=(1.1, 0.7, 0.3), pose=Pose(xyz=(0, 0, 0.1))),
            Primitive(name="column", shape="box", size=(0.3, 0.3, 0.4), pose=Pose(xyz=(0.2, 0, 0.4))),
        ],
        T_mount=Pose(xyz=(0.2, 0, 0.8)),
        joints=CarrierJoints(x=(-4.0, 4.0), y=(-4.0, 4.0), yaw_deg=(-180, 180)),
        standoff_m=Standoffs(carrier_part_min=0.3, carrier_part_preferred=0.4),
        **kw,
    )


def world_vertices(bodies) -> dict:
    return {b.name: b.mesh.vertices @ b.T[:3, :3].T + b.T[:3, 3] for b in bodies}


# --- legacy conversion ---------------------------------------------------------------------------


@pytest.mark.parametrize("make", [rail_base, mobile_base])
def test_legacy_fields_convert_to_same_geometry(make):
    legacy = make()
    new = migrate(legacy)
    assert new.chassis is None and new.riser is None and new.track is None and new.bodies
    a, b = Carrier(legacy), Carrier(new)
    v = {n: float(np.mean(r)) + 0.3 for n, r in zip(a.dofs, a.bounds, strict=True)}
    for get in (lambda c: c.moving_bodies(v), lambda c: c.static_bodies()):
        wa, wb = world_vertices(get(a)), world_vertices(get(b))
        assert wa.keys() == wb.keys()
        for k in wa:
            assert np.allclose(wa[k], wb[k])
    assert a.footprint(v).equals(b.footprint(v))
    names = [p.name for p in body_list(legacy)]
    assert names[:2] == ["chassis", "riser"]


def test_body_names_are_unique_and_safe():
    base = Base(
        mode="movable",
        carrier="rail",
        joints=CarrierJoints(x=(0, 1)),
        bodies=[Primitive(shape="box", size=(1, 1, 1), name="a b")] * 2
        + [Primitive(shape="box", size=(1, 1, 1))],
    )
    names = [p.name for p in body_list(base)]
    assert len(set(names)) == 3 and names[0] == "a_b"


def test_schema_rejects_bad_bodies():
    with pytest.raises(ValidationError):
        Primitive(shape="mesh")  # no file


# --- multi-body carrier ---------------------------------------------------------------------------


def test_support_polygon_is_the_wheels():
    c = Carrier(amr())
    poly = c.footprint_moving({"x": 0.0, "y": 0.0, "yaw": 0.0})
    xmin, ymin, xmax, ymax = poly.bounds
    assert xmax == pytest.approx(0.55) and xmin == pytest.approx(-0.55)  # wheel radius past axle
    assert ymax == pytest.approx(0.36) and ymin == pytest.approx(-0.28 - 0.08)  # cylinder extends -y
    # the standoff footprint is the union of each body's hull (not one hull of everything)
    from shapely.geometry import box

    fp = c.footprint({"x": 0.0, "y": 0.0, "yaw": 0.0})
    assert fp.contains(box(-0.54, -0.34, 0.54, 0.34)) and not fp.contains(poly)


def test_urdf_body_frames_match():
    c = Carrier(amr())
    model = pin.buildModelFromXML(c.urdf())
    data = model.createData()
    v = {"x": 1.0, "y": -0.5, "yaw": 0.7}
    pin.framesForwardKinematics(model, data, np.array([v[n] for n in c.dofs]))
    for body in c.moving_bodies(v):
        T = data.oMf[model.getFrameId(f"body_{body.name}")].homogeneous
        assert np.allclose(T, body.T, atol=1e-8)  # URDF numbers are written to 9 digits
    assert np.allclose(data.oMf[model.getFrameId("mount")].homogeneous, c.T_world_base(v), atol=1e-8)


def test_snap_mount_to_body_top():
    base = amr()
    col = next(b for b in base.bodies if b.name == "column")
    top = body_top(col)
    assert np.allclose(top, [0.2, 0.0, 0.8])
    tilted = col.model_copy(update={"pose": Pose(xyz=(0, 0, 0.4), rpy=(0, 0, np.pi / 4))})
    assert body_top(tilted)[2] == pytest.approx(0.8)


def test_arm_collides_with_custom_column():
    robot = Robot.from_registry("ur20")
    ee = load_tool("orbital_sander")
    tm = ToolModel(ee, REPO / ee.urdf)
    from opt.sites import CarrierSites

    sites = CarrierSites(Carrier(amr()), None, None)
    bb, wb = sites.scene_bodies()
    assert {n for n, *_ in bb} >= {"chassis", "column", "wheel0"}
    sc = CollisionScene(
        robot, tm, None, sites.T({"x": 0, "y": 0, "yaw": 0}), None, base_bodies=bb, world_bodies=wb
    )
    assert sc.collides(np.array([0, np.pi / 2, 0, 0, 0, 0]), "pad")  # upper arm down into the AMR
    assert not sc.collides(np.radians([0, -90, 90, -90, -90, 0]), "pad")


# --- mesh bodies ------------------------------------------------------------------------------------


def test_mesh_body_units_and_standoff(tmp_path):
    stl = tmp_path / "cart.stl"
    trimesh.creation.box(extents=(800, 600, 300)).export(stl)  # mm, centred on its origin
    body = Primitive(shape="mesh", mesh=str(stl), units="mm", name="cart", pose=Pose(xyz=(0, 0, 0.15)))
    base = Base(
        mode="movable",
        carrier="mobile",
        bodies=[body],
        T_mount=Pose(xyz=(0, 0, 0.3)),
        joints=CarrierJoints(x=(-3, 3), y=(-3, 3), yaw_deg=(-180, 180)),
    )
    c = Carrier(base)
    (b,) = c.moving_bodies({"x": 0, "y": 0, "yaw": 0})
    w = b.mesh.vertices @ b.T[:3, :3].T + b.T[:3, 3]
    assert np.allclose(w.min(axis=0), [-0.4, -0.3, 0]) and np.allclose(w.max(axis=0), [0.4, 0.3, 0.3])
    assert np.allclose(body_top(body), [0, 0, 0.3])
    part = gp.flat_plate()  # 1 x 0.6 x 0.02, centred
    d = distance_3d(c.moving_bodies({"x": 0, "y": 0, "yaw": 0}), part, make_T(p=(1.5, 0, 0.3)))
    assert d == pytest.approx(1.5 - 0.5 - 0.4, abs=1e-6)
    assert 'filename="' in c.urdf() and 'scale="0.001' in c.urdf()


def test_missing_mesh_is_reported():
    base = Base(
        mode="movable",
        carrier="rail",
        joints=CarrierJoints(x=(0, 1)),
        bodies=[Primitive(shape="mesh", mesh="nowhere/cart.stl")],
    )
    with pytest.raises(FileNotFoundError):
        Carrier(base).moving_bodies({"s": 0.0})


# --- planner with a multi-body mobile robot -------------------------------------------------------


def test_plan_with_custom_amr():
    from core.collision import CollisionScene
    from core.feasibility import FeasibilityChecker
    from core.ik import make_ik
    from core.schema import TargetGroup, Weights
    from opt.placement import PlannerSettings, Targets, plan_stations
    from opt.reachability import map_for_tool
    from opt.sites import CarrierSites

    robot = Robot.from_registry("ur20")
    ee = load_tool("orbital_sander")
    tm = ToolModel(ee, REPO / ee.urdf)
    rmap = map_for_tool(robot, tm.T_flange_tcp("pad"))
    part, T_part = gp.flat_plate(), make_T(p=(0.0, 0.0, 0.6))
    xs = np.linspace(-0.4, 0.4, 5)
    Pp = np.column_stack([xs, np.zeros(5), np.full(5, 0.02)])
    Np = np.tile((0, 0, 1.0), (5, 1))
    g = TargetGroup(name="top", tilt_deg=10.0)
    t = Targets(np.arange(5), Pp + T_part[:3, 3], Np, Pp, Np, [g] * 5, np.full(5, 10.0))
    sites = CarrierSites(Carrier(amr()), part, T_part)
    bb, wb = sites.scene_bodies()
    scene = CollisionScene(robot, tm, part, np.eye(4), T_part, base_bodies=bb, world_bodies=wb)
    ch = FeasibilityChecker(robot, make_ik(robot), tm, scene, np.eye(4), T_part)
    out = plan_stations(rmap, ch, t, np.eye(4), None, Weights(), PlannerSettings(max_stations=2), sites=sites)
    best = min((s for s, _ in out if not s.unreached), key=lambda s: len(s.stations))
    assert len(best.assignment) == 5
    c = sites.c
    for st in best.stations:
        v = {d: st.carrier[d] for d in c.dofs}
        assert distance_3d(c.moving_bodies(v), part, T_part) >= 0.3 - 1e-6


# --- GUI ------------------------------------------------------------------------------------------


@pytest.fixture
def gui(tmp_path):
    import viser

    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8772, verbose=False)
    try:
        state = build_app(server)
        state["session"].project_dir = tmp_path
        yield state
    finally:
        server.stop()


def test_gui_external_axis_tab(gui):
    from ui.carrier_ui import DEFAULTS

    panel, s = gui["carrier"], gui["session"]
    assert "External Axis" in gui["tabs"]
    ed = panel.editor
    panel.load(DEFAULTS["mobile"])
    assert panel.commit()
    assert [b.name for b in s.base.bodies] == ["chassis", "riser"] and s.base.chassis is None

    # add a box, edit its size and position
    assert ed.add_shape("box") and ed.add_shape("cylinder")
    assert len(s.base.bodies) == 4 and s.base.bodies[-1].shape == "cylinder"
    ed.pick.value = s.base.bodies[2].name
    ed._commit(None, [b for b in ed.bodies if b.name != "cylinder1"], select=s.base.bodies[2].name)
    assert len(s.base.bodies) == 3
    new = s.base.bodies[-1].name
    assert new == "box1" and ed.pick.value == new
    ed.box.value = (300.0, 300.0, 500.0)
    ed.xyz.value = (-400.0, 0.0, 400.0)
    b = s.base.bodies[-1]
    assert b.size == pytest.approx((0.3, 0.3, 0.5)) and b.pose.xyz == pytest.approx((-0.4, 0.0, 0.4))

    # invalid size is rejected and the fields revert
    ed.box.value = (0.0, 300.0, 500.0)
    assert s.base.bodies[-1].size == pytest.approx((0.3, 0.3, 0.5))
    assert ed.box.value[0] == pytest.approx(300.0)

    # snap the robot onto the new body: base sits on its top
    ed.mount_on.value = new
    assert ed.snap_mount()
    assert s.base.T_mount.xyz == pytest.approx((-0.4, 0.0, 0.9))
    c = Carrier(s.base)
    v = c.clip(panel.values())
    assert np.allclose(s.T_world_base, c.T_world_base(v), atol=1e-9)
    assert s.T_world_base[2, 3] == pytest.approx(0.9)

    # drag the body with its gizmo
    ed.pick.value = new
    ed.drag.value = True
    assert ed.gizmo is not None
    T_parent = panel.parent_T("carriage")
    ed.gizmo.position = (T_parent @ np.array([-0.3, 0.1, 0.4, 1.0]))[:3]
    ed._on_body_drag(None)
    assert s.base.bodies[-1].pose.xyz == pytest.approx((-0.3, 0.1, 0.4), abs=1e-6)

    # drag the mount gizmo
    ed.drag_mount.value = True
    assert ed.mount_gizmo is not None
    ed.mount_gizmo.position = (panel.parent_T("mount") @ np.array([-0.3, 0.1, 0.9, 1.0]))[:3]
    ed._on_mount_drag(None)
    assert s.base.T_mount.xyz == pytest.approx((-0.3, 0.1, 0.9), abs=1e-6)

    # remove
    ed.pick.value = new
    k = len(s.base.bodies)
    ed._commit(None, [x for x in ed.bodies if x.name != new])
    assert len(s.base.bodies) == k - 1


def test_gui_mesh_upload_and_project_round_trip(gui, tmp_path):
    from ui.carrier_ui import DEFAULTS
    from ui.session import Session, load_project, save_project

    panel, s = gui["carrier"], gui["session"]
    panel.load(DEFAULTS["rail"])
    assert panel.commit()
    ed = panel.editor
    data = trimesh.exchange.stl.export_stl(trimesh.creation.box(extents=(400, 400, 200)))
    ed.mesh_units.value = "mm"
    assert ed.add_mesh_bytes("my cart.stl", data)
    body = s.base.bodies[-1]
    assert body.shape == "mesh" and body.units == "mm" and (tmp_path / body.mesh).exists()
    assert not ed.add_mesh_bytes("broken.stl", b"not a mesh")
    assert not (tmp_path / "assets" / "carriers" / "broken.stl").exists()
    assert not ed.add_mesh_bytes("notes.txt", b"x")

    # fixed body (e.g. a second track) moves with the carrier frame, not the carriage
    ed.pick.value = body.name
    ed.attach.value = "fixed (track / frame)"
    assert s.base.bodies[-1].attach == "fixed"
    assert any(h.name.endswith(body.name) for h in panel._handles)

    save_project(s, tmp_path)
    fresh = Session()
    load_project(fresh, tmp_path)
    assert fresh.base == s.base
    assert len(fresh.carrier().static_bodies()) == 2  # track + uploaded mesh
