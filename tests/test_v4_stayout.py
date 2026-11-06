"""v3 feedback item 5: stay-out zones (boxes, cylinders, meshes) that the arm, the tool and the
carrier keep clear of, in the collision scene, the planner and the GUI."""

import numpy as np
import pytest
import viser

from core.collision import CollisionScene
from core.robot import Robot
from core.schema import Pose, Primitive
from core.stayout import carrier_clear, obstacles, scene_entries
from ui.staleness import solution_changes


def _box_at(p, size=0.3, name="z") -> Primitive:
    return Primitive(
        shape="box",
        name=name,
        attach="fixed",
        size=(size, size, size),
        pose=Pose(xyz=(p[0], p[1], p[2] - size / 2)),
    )


def test_arm_collides_with_a_zone_around_its_wrist():
    robot = Robot.from_registry("ur20")
    q = np.radians([0, -90, 90, -90, -90, 0])
    T = np.eye(4)
    wrist = (T @ robot.fk(q))[:3, 3]
    free = CollisionScene(robot, None, None, T, None)
    assert not free.collides(q)
    zone = obstacles([_box_at(wrist)])
    hit = CollisionScene(robot, None, None, T, None, obstacles=scene_entries(zone))
    assert hit.collides(q)
    far = obstacles([_box_at(wrist + np.array([2.0, 2.0, 0.0]))])
    assert not CollisionScene(robot, None, None, T, None, obstacles=scene_entries(far)).collides(q)


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8799, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


def test_planner_keeps_carrier_and_arm_clear(app):
    s = app["session"]
    assert app["project"].load_example("rail_long_part")
    app["stations"].start(block=True)
    sol0 = min(s.results, key=lambda r: (len(r.unreached), r.cost))
    c = s.carrier()
    first = sol0.stations[0].carrier
    body = c.moving_bodies(c.clip(first))[0]
    centre = body.T[:3, 3] + np.array([0.0, 0.0, 0.3])
    s.stayouts = [_box_at(centre, 0.4, "on_the_rail")]  # right where the first carriage stood
    assert "stay-out zones" in solution_changes(s, sol0)  # the old plan is now out of date
    app["stations"].start(block=True)
    sol = min(s.results, key=lambda r: (len(r.unreached), r.cost))
    obs = s.obstacles()
    clearance = s.base.standoff_m.carrier_env_min
    for st in sol.stations:
        v = c.clip(st.carrier)
        ok, why = carrier_clear(c.moving_bodies(v) + c.static_bodies(), c.footprint(v), obs, clearance)
        assert ok, why
    # every planned joint solution is clear of the zone (arm and tool)
    from core.export import station_pose

    scene = CollisionScene(
        Robot(s.robot.spec),
        s.tool_model,
        None,
        s.T_world_base,
        None,
        floor=False,
        obstacles=s.obstacle_entries(),
    )
    res = s.result_points[sol.label]
    for pid, k in sol.assignment.items():
        scene.set_base(station_pose(s.base, sol.stations[k], s.T_world_base))
        assert not scene.collides(res[pid].q, res[pid].tcp)
    s.stayouts = []


def test_gui_add_edit_drag_save_remove(app, tmp_path):
    s, panel = app["session"], app["stayouts"]
    assert app["project"].load_example("rail_long_part")
    assert panel.add_shape("box")
    assert len(s.stayouts) == 1 and panel.pick.value == s.stayouts[0].name and panel._handles
    panel.box.value = (300.0, 400.0, 1500.0)
    panel.xyz.value = (2000.0, 1500.0, 0.0)
    z = s.stayouts[0]
    assert z.size == pytest.approx((0.3, 0.4, 1.5)) and z.pose.xyz == pytest.approx((2.0, 1.5, 0.0))
    panel.drag.value = True
    assert panel.gizmo is not None
    panel.gizmo.position = (2.5, 1.0, 0.2)
    panel._from_gizmo()
    assert s.stayouts[0].pose.xyz == pytest.approx((2.5, 1.0, 0.2)) and panel.xyz.value[0] == pytest.approx(
        2500
    )
    assert panel.add_shape("cylinder") and len(s.stayouts) == 2
    stl = (
        b"solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 100 0 0\nvertex 0 100 0\n"
        b"endloop\nendfacet\nendsolid t\n"
    )
    assert panel.add_mesh_bytes("fixture.stl", stl)
    assert s.stayouts[-1].shape == "mesh" and s.stayouts[-1].units == "mm"
    from ui.session import Session, load_project, save_project

    save_project(s)
    s2 = Session()
    load_project(s2, s.project_dir)
    assert [z.name for z in s2.stayouts] == [z.name for z in s.stayouts] and len(s2.obstacles()) == 3
    panel.pick.value = s.stayouts[0].name
    assert panel.remove_zone() and len(s.stayouts) == 2
    s.stayouts = []
    panel.refresh()
    assert not panel._handles
