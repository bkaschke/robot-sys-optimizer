"""v8 follow-ups: reduced-speed zones as their own panel and in the planner's cycle estimate,
point order over every arm configuration (see also tests/test_v8_gtsp.py)."""

import numpy as np
import pytest
import viser

from core.cycle_time import CycleSettings, plan_estimate
from core.schema import Pose, Primitive
from core.speed_zones import share_at, speed_zones
from tests.helpers import confirm_all


class _R:
    def __init__(self, q):
        self.q = np.asarray(q, float)


def test_share_at_and_plan_estimate_stretch_moves_into_a_zone():
    z = Primitive(shape="box", size=(1, 1, 1), name="s", speed_pct=25.0, pose=Pose(xyz=(2.0, 0.0, 0.0)))
    hard = Primitive(shape="box", size=(1, 1, 1), name="h")
    zones = speed_zones([z, hard])
    assert [x.name for x in zones] == ["s"]
    f = share_at(zones, [[2.0, 0.0, 0.5], [0.0, 0.0, 0.5], [2.0, 0.0, 1.5]])
    assert f.tolist() == [0.25, 1.0, 1.0]
    res = {1: _R(np.zeros(6)), 2: _R(np.full(6, 0.5)), 3: _R(np.full(6, 1.0))}
    seqs = {0: [[1, 2, 3]]}
    cs = CycleSettings(settle_s=0.0)
    base = plan_estimate(seqs, res, [], cs)
    slow = plan_estimate(seqs, res, [], cs, share={3: 0.25})
    assert "reduced" not in base
    assert slow["reduced"] == pytest.approx(base["joint"] / 2 * 3)  # the second of two equal moves, x4
    assert slow["total"] == pytest.approx(base["total"] + slow["reduced"])


def test_zone_pickles_without_its_mesh():
    import pickle

    z = speed_zones([Primitive(shape="cylinder", radius=0.5, height=1.0, name="c", speed_pct=50.0)])[0]
    assert z.contains([[0.0, 0.0, 0.5]])[0]
    z2 = pickle.loads(pickle.dumps(z))
    assert z2._mesh is None and z2.contains([[0.0, 0.0, 0.5]])[0] and not z2.contains([[0.9, 0.0, 0.5]])[0]


def test_stayout_and_speed_panels_keep_their_own_zones():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8883, verbose=False)
    try:
        app = build_app(server)
        s, so, sz = app["session"], app["stayouts"], app["slowzones"]
        assert sz.folder is not so.folder and not sz.zones and not so.zones
        so.add_shape("box")
        sz.add_shape("cylinder")
        sz.add_shape("box")
        assert [z.speed_pct for z in so.zones] == [None]
        assert [z.speed_pct for z in sz.zones] == [25.0, 25.0]
        assert len(s.stayouts) == 3 and len(s.obstacles()) == 1
        sz.speed.value = 40.0  # edits the selected (newest) slow zone only
        assert sorted(z.speed_pct for z in sz.zones) == [25.0, 40.0]
        so.xyz.value = (0.0, 2000.0, 0.0)  # a stay-out edit keeps the slow zones
        assert len(s.stayouts) == 3 and so.zones[0].pose.xyz[1] == pytest.approx(2.0)
        so.remove_zone()
        assert not so.zones and len(sz.zones) == 2
        names = {z.name for z in s.stayouts}
        assert len(names) == 2
        objects = app["objects"]
        objects.rebuild_tree()
        assert set(objects._children("Reduced-speed zones")) == names
        assert objects._children("Stay-out zones") == []
        from ui.move_targets import move_keys, move_target

        keys = [k for k in move_keys(objects) if k.startswith("Reduced-speed zone/")]
        assert len(keys) == 2 and move_target(objects, keys[0]) is not None
    finally:
        server.stop()


@pytest.mark.slow
def test_plan_cycle_estimate_counts_reduced_speed_zones():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8884, verbose=False)
    try:
        app = build_app(server)
        assert app["project"].load_example("rail_long_part")
        s, st = app["session"], app["stations"]
        confirm_all(st)
        st.start(block=True, quick=True)
        fast = s.results[0]
        P = np.array([p.p for p in s.targets.points], float)
        Pw = P @ s.part.T_world_part[:3, :3].T + s.part.T_world_part[:3, 3]
        lo, hi = Pw.min(axis=0) - 0.2, Pw.max(axis=0) + 0.2
        sz = app["slowzones"]
        sz.add_shape("box")
        sz.box.value = tuple((hi - lo) * 1000)
        c = (lo + hi) / 2 * 1000
        sz.xyz.value = (float(c[0]), float(c[1]), float(lo[2] * 1000))
        sz.speed.value = 50.0
        assert len(speed_zones(s.stayouts)) == 1 and not s.obstacles()
        confirm_all(st)
        st.start(block=True, quick=True)
        slow = s.results[0]
        assert len(slow.stations) == len(fast.stations) and not slow.unreached
        assert slow.cycle_s > fast.cycle_s + 1.0
    finally:
        server.stop()


def _scene_with_box(robot, centre, hose):
    import trimesh

    from core.collision import CollisionScene
    from core.transforms import make_T

    box = trimesh.creation.box((0.03, 0.03, 0.03))
    return CollisionScene(robot, None, box, np.eye(4), make_T(p=centre), floor=False, hose=hose)


def test_hose_is_a_shape_on_the_arm():
    import pinocchio as pin

    from core.hose import Hose, segments
    from core.robot import Robot

    robot = Robot.from_registry("ur20")
    hose = Hose(radius=0.03, offset=0.3, side=1.0)
    segs = segments(robot, hose)
    assert [j for j, _, _ in segs] == [2, 3, 4, 5]
    assert segs[0][2] == pytest.approx(0.862, abs=1e-3)
    q = np.radians([0.0, -60.0, 70.0, -90.0, -90.0, 0.0])
    pin.forwardKinematics(robot.model, robot.data, q)
    j, T, _ = segs[0]  # upper-arm hose: its middle, in the world (base) frame
    c = (np.asarray(robot.data.oMi[j].homogeneous) @ T)[:3, 3]
    without, with_hose = _scene_with_box(robot, c, None), _scene_with_box(robot, c, hose)
    assert len(with_hose.hose_ids) == 4 and not without.hose_ids
    assert not without.collides(q)  # the box is 0.3 m off the arm: clear of the links
    assert with_hose.collides(q)  # but the hose runs through it
    other = _scene_with_box(robot, c, Hose(radius=0.03, offset=0.3, side=-1.0))
    assert not other.collides(q)  # the other side of the arm


def test_cable_panel_modes_and_drawing():
    from app import build_app
    from core import collision

    server = viser.ViserServer(host="127.0.0.1", port=8885, verbose=False)
    try:
        app = build_app(server)
        cable = app["cable"]
        cable.mode.value = "hose along the arm"
        assert collision.HOSE is not None and collision.DRESS_M == 0.0
        assert collision.HOSE.radius == pytest.approx(0.03) and len(cable._handles) == 4
        cable.side.value = "−"
        assert collision.HOSE.side == -1.0
        cable.show.value = False
        assert not cable._handles
        cable.mode.value = "clearance margin"
        cable.dress.value = 40.0
        assert collision.HOSE is None and collision.DRESS_M == pytest.approx(0.04)
        assert not cable.diameter.visible and cable.dress.visible
    finally:
        collision.HOSE, collision.DRESS_M = None, 0.0
        server.stop()


def test_tool_geometry_confirm_and_extension_copy():
    from app import build_app
    from ui.staleness import step_state

    server = viser.ViserServer(host="127.0.0.1", port=8886, verbose=False)
    try:
        app = build_app(server)
        assert app["project"].load_example("rail_long_part")
        s, tool, opt, copy = app["session"], app["tool"], app["tool_opt"], app["tool_ext_copy"]
        confirm_all(s)
        opt.update_confirm()
        assert not opt.confirm_btn.visible  # step 3 confirmed: nothing to confirm
        # the copy shows the tool's extension
        x = s.tool.extension
        assert copy.ext_type.value == x.type and copy.radius.value == pytest.approx(x.radius * 1000)
        # an edit in the copy changes the tool and the Tool tab
        copy.radius.value = 33.0
        assert s.tool.extension.radius == pytest.approx(0.033) and tool.radius.value == pytest.approx(33.0)
        # ... and its own Confirm tool shows at once, right under the copied fields
        assert copy.confirm_btn.visible and "changed" in copy.confirm_md.content
        opt.confirm_now()
        assert step_state(s, "Tool Setup") == "confirmed"
        assert not copy.confirm_btn.visible and not opt.confirm_btn.visible
        # an edit in the Tool tab shows in the copy
        tool.radius.value = 37.0
        assert copy.radius.value == pytest.approx(37.0)
        # rough placement slider in the copy -> nominal + Tool tab slider
        key = next(iter(copy._nominal))
        w = copy._nominal[key]
        w.value = (w.min + w.max) / 2
        f = 1000.0 if key in ("L", "d", "x", "y", "z", "d2") else 1.0
        assert getattr(s.tool.extension.nominal, key) * f == pytest.approx(w.value)
        assert tool._nominal[key].value == pytest.approx(w.value)
        # the tool changed: Confirm tool shows, and confirms step 3
        assert step_state(s, "Tool Setup") == "changed"
        opt.update_confirm()
        assert opt.confirm_btn.visible and "changed" in opt.confirm_md.content
        opt.confirm_tool()
        opt.update_confirm()
        assert step_state(s, "Tool Setup") == "confirmed" and not opt.confirm_btn.visible
    finally:
        server.stop()
