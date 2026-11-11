"""Left panel Jog (2026-11-11): mobile moves without limits, TCP and carrier drag gizmos,
carrier joystick (ui.jog_gizmos, ui.joystick)."""

import time

import numpy as np
import pytest
import viser

from core.transforms import make_T, rpy_to_matrix
from ui.joystick import PREFIX, pad_html, parse, step_values


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8827, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


def test_step_values_per_carrier():
    v = {"x": 1.0, "y": 2.0, "yaw": np.pi / 2}
    fwd = step_values("mobile", "Drive and turn", v, 0.0, 1.0, 0.5, 1.0, 2.0)
    assert np.allclose((fwd["x"], fwd["y"], fwd["yaw"]), (1.0, 3.0, np.pi / 2))  # along the heading (+y)
    turn = step_values("mobile", "Drive and turn", v, 1.0, 0.0, 0.5, 0.3, 1.0)
    assert turn["yaw"] == pytest.approx(np.pi / 2 - 0.3) and turn["x"] == 1.0  # right = clockwise
    side = step_values("mobile", "Strafe", v, 1.0, 0.0, 0.5, 1.0, 1.0)
    assert np.allclose((side["x"], side["y"]), (1.5, 2.0))  # stick right = the robot's right (-left)
    assert step_values("rail", "", {"s": 0.0, "lift": 0.1}, 1.0, -1.0, 0.2, 1.0, 1.0) == {
        "s": 0.2,
        "lift": pytest.approx(-0.1),
    }
    assert step_values("mobile", "", v, 0.01, -0.02, 1.0, 1.0, 1.0) == v  # dead zone


def test_pad_and_parse():
    assert parse(PREFIX + "0.50,-1.00,17") == (0.5, -1.0)
    assert parse("joy:3,0") == (1.0, 0.0)  # clamped
    assert parse("hello") is None and parse("joy:a,b") is None
    html = pad_html()
    assert 'onpointerdown="' in html and html.count('"') % 2 == 0
    js = html.split('onpointerdown="')[1].split('"')[0]
    assert "joy:" in js and "<" not in js  # the handler stays one attribute


def test_mobile_moves_have_no_limits(app):
    assert app["project"].load_example("mobile_amr")
    s, car = app["session"], app["carrier"]
    assert car._dof["x"].min is None and car._dof["x"].max is None
    car._dof["x"].value = 25000.0  # 25 m: far outside the planner's search range
    c = s.carrier()
    assert car.current(c)["x"] == pytest.approx(25.0)
    assert np.allclose(s.T_world_base, c.T_world_base(car.current(c)), atol=1e-9)
    car.set_values({"x": -40.0, "y": 30.0, "yaw": np.radians(270.0)})
    v = car.current(c)
    assert (v["x"], v["y"]) == pytest.approx((-40.0, 30.0)) and np.degrees(v["yaw"]) == pytest.approx(-90.0)


def test_joystick_drives_the_mobile_base(app):
    assert app["project"].load_example("mobile_amr")
    s, car, drive = app["session"], app["carrier"], app["drive"]
    c = s.carrier()
    v0 = car.current(c)
    drive.mode.value = "Drive and turn"
    drive.speed.value = 500.0
    drive.cmd = (0.0, 1.0)
    assert drive.step(1.0)
    v1 = car.current(c)
    heading = np.array([np.cos(v0["yaw"]), np.sin(v0["yaw"])])
    assert np.allclose([v1["x"] - v0["x"], v1["y"] - v0["y"]], 0.5 * heading, atol=1e-9)
    # through the text input, as the browser pad sends it; the drive thread stops on release
    drive.value.value = PREFIX + "0.00,1.00,1"
    t_end = time.time() + 3
    while (
        time.time() < t_end and np.hypot(car.current(c)["x"] - v1["x"], car.current(c)["y"] - v1["y"]) < 0.05
    ):
        drive.command(0.0, 1.0)  # the pad repeats every 100 ms while held
        time.sleep(0.1)
    drive.value.value = PREFIX + "0.00,0.00,2"
    time.sleep(0.3)
    v2 = car.current(c)
    time.sleep(0.4)
    assert car.current(c) == v2  # stopped
    assert np.hypot(v2["x"] - v1["x"], v2["y"] - v1["y"]) > 0.05


def test_joystick_rail_stays_in_travel(app):
    assert app["project"].load_example("rail_long_part")
    car, drive = app["carrier"], app["drive"]
    drive.cmd = (1.0, 0.0)
    drive.speed.value = 5000.0
    for _ in range(10):
        drive.step(1.0)
    assert car._dof["s"].value == pytest.approx(car._dof["s"].max)


def test_carrier_gizmo_moves_the_carrier(app):
    assert app["project"].load_example("mobile_amr")
    s, car, drive = app["session"], app["carrier"], app["drive"]
    drive.show_gizmo.value = True
    g = drive.gizmo
    assert g.handle is not None
    c = s.carrier()
    v = car.current(c)
    G = g.mount_parent(c, v)
    assert np.allclose(g.handle.position, G[:3, 3])
    target = make_T(rpy_to_matrix((0, 0, v["yaw"] + 0.5)), G[:3, 3] + (1.0, -2.0, 0.0))
    g._dragged(target)
    w = car.current(c)
    assert (w["x"], w["y"]) == pytest.approx((v["x"] + 1.0, v["y"] - 2.0))
    assert np.allclose(g.mount_parent(c, w)[:3, :3], target[:3, :3], atol=1e-9)
    # a rail: only s follows the drag, clamped to the travel
    assert app["project"].load_example("rail_long_part")
    c = s.carrier()
    v = car.current(c)
    G = g.mount_parent(c, v)
    g._dragged(G @ make_T(p=(0.4, 0.3, 0.0)))  # sideways part is dropped
    assert car.current(c)["s"] == pytest.approx(v["s"] + 0.4)
    assert np.allclose(g.handle.position, g.mount_parent(c, car.current(c))[:3, 3])  # gizmo follows
    drive.show_gizmo.value = False
    assert g.handle is None


def test_tcp_gizmo_moves_the_jog_point(app):
    assert app["project"].load_example("rail_long_part")
    jog = app["cell"].jog_panel
    jog.allow.value = True
    app["cell"].set_q(np.radians([0.0, -100.0, -100.0, -70.0, 90.0, 0.0]))  # home is wrist-singular
    for frame in ("World", "Tool"):
        jog.frame.value = frame
        jog.show_gizmo.value = True
        g = jog.gizmo
        T0 = jog.T_world_point()
        G0 = g.frame_pose()
        assert np.allclose(g.handle.position, T0[:3, 3])
        R = np.eye(3) if frame == "World" else T0[:3, :3]
        assert np.allclose(G0[:3, :3], R, atol=1e-9)
        g._start(G0)
        assert g._dragged(G0 @ make_T(p=(0.0, 0.0, 0.02)))  # 20 mm along the frame's z
        T1 = jog.T_world_point()
        assert np.allclose(T1[:3, 3] - T0[:3, 3], 0.02 * R[:, 2], atol=1e-6)
        assert np.allclose(T1[:3, :3], T0[:3, :3], atol=1e-6)
        g._start(g.frame_pose())
        assert not g._dragged(g.frame_pose() @ make_T(p=(5.0, 0.0, 0.0)))  # out of reach: refused
        assert np.allclose(jog.T_world_point(), T1, atol=1e-9)
    jog.show_gizmo.value = False
    assert jog.gizmo.handle is None
    jog.allow.value = False
