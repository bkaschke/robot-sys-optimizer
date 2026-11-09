"""v6: collision-free joint moves (home, configuration changes), the lift moving at the stops
before the carrier drives, and Clear paths."""

import json

import numpy as np
import pytest

from core.carrier import Carrier
from core.golden_parts import tall_wall
from core.joint_plan import densify, plan_joint_path
from core.schema import Project
from ui.project_panel import EXAMPLES


def _wall(q) -> bool:
    """A wall in joint space: joint 0 in (-0.3, 0.3) is blocked unless joint 1 > 1.0."""
    return abs(q[0]) < 0.3 and q[1] < 1.0


def test_joint_plan_goes_around_and_keeps_straight_moves():
    lo, hi = np.full(2, -np.pi), np.full(2, np.pi)
    qa, qb = np.array([-1.0, 0.0]), np.array([1.0, 0.0])
    way = plan_joint_path(_wall, qa, qb, lo, hi, step=0.02)
    assert way is not None and np.allclose(way[0], qa) and np.allclose(way[-1], qb)
    qs = densify(way, 0.02)
    assert not any(_wall(q) for q in qs) and np.allclose(qs[-1], qb)
    assert max(q[1] for q in qs) > 1.0  # it went over the wall
    free = plan_joint_path(lambda q: False, qa, qb, lo, hi)
    assert len(free) == 2  # clear: the straight move
    assert plan_joint_path(lambda q: abs(q[0]) < 0.3, qa, qb, lo, hi, time_limit=0.5) is None


def test_lift_moves_at_the_stops_and_the_carrier_drives_at_one_height():
    from opt.sites import CarrierSites

    base = Project.model_validate(json.loads((EXAMPLES / "mobile_lift" / "project.json").read_text())).base
    sites = CarrierSites(Carrier(base), tall_wall(), np.eye(4))
    a = {"x": -0.5, "y": -1.2, "yaw": np.pi / 2, "lift": 0.75}
    b = {"x": 0.5, "y": -1.2, "yaw": np.pi / 2, "lift": 0.5}
    moves = sites.transit_staged(a, b, 0.05, 0.0)
    xy = np.array([[v["x"], v["y"]] for v, _ in moves])
    lift = np.array([v["lift"] for v, _ in moves])
    moved = np.any(np.abs(np.diff(xy, axis=0)) > 1e-12, axis=1)  # sample i+1 drove
    assert moved.any() and np.allclose(lift[1:][moved], 0.0)  # it drives at the travel height
    assert np.allclose(lift[:-1][moved], 0.0)  # ... and only once the lift is down
    i0 = int(np.argmax(lift < 1e-12))  # the lift went down at a, before any drive
    assert np.allclose(xy[: i0 + 1], (a["x"], a["y"])) and np.all(np.diff(lift[: i0 + 1]) < 0)
    assert lift[-1] == pytest.approx(0.5) and np.allclose(xy[-1], (0.5, -1.2))
    assert all(ok for _, ok in moves)
    assert sites.transit_staged(a, b, 0.05, None) == sites.transit(a, b, 0.05)


@pytest.fixture(scope="module")
def app():
    import viser

    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8812, verbose=False)
    yield build_app(server)
    server.stop()


def test_home_moves_are_clear_and_clear_paths(app):
    s = app["session"]
    assert app["project"].load_example("rail_long_part")
    stp = app["stations"]
    stp.zoning.value = stp.zoning.options[0]  # no zoning: configuration changes inside a stop
    stp.start(block=True)
    sol = min(s.results, key=lambda r: (len(r.unreached), r.cost))
    app["results"].choice.value = next(o for o in app["results"].choice.options if sol.label in o)
    sim = app["sim"]
    sim.home.value, sim.step.value = True, 50.0
    sim.build(block=True)
    plan = sim.plan
    assert plan is not None and not sim.clear_btn.disabled
    home = [p for p in plan.samples if p.kind == "home"]
    assert home and not [p for p in home if p.issue]
    ch = sim._checker()
    for st in plan.stations:  # joint moves are continuous (3 deg, back-off at the path step), clear
        ch.set_base(st.T_world_base)
        prev = None
        for p in st.samples:
            if p.kind == "home" and prev is not None and prev.q is not None:
                assert np.max(np.abs(p.q - prev.q)) <= np.radians(sim.jump.value)  # back-off: path steps
            if p.kind == "home":
                assert not ch.scene.collides(p.q, p.tcp)
            prev = p
    n_reconf = plan.counts()["reconfigure"]
    if n_reconf:  # each configuration change is reached by a traced move through home
        for st in plan.stations:
            for i, p in enumerate(st.samples):
                if p.issue == "reconfigure" and p.kind == "":
                    assert st.samples[i - 1].kind == "home"
    sim.clear_paths()
    assert sim.plan is None and sim.clear_btn.disabled and sim.export.disabled
    assert "cleared" in sim.info.content


def test_sim_lowers_the_lift_to_drive_between_stops(app):
    from types import SimpleNamespace

    s, sim = app["session"], app["sim"]
    assert app["project"].load_example("mobile_lift")
    a, b = {"t": 1.0, "lift": 0.75}, {"t": 1.6, "lift": 0.5}
    sol = SimpleNamespace(stations=[SimpleNamespace(carrier=a), SimpleNamespace(carrier=b)])
    path = sim._carrier_path(sol, 0.05, sim._checker(), s.home_q())
    moves = path(0, 1)
    lift = np.array([d["lift"] for d, _, _ in moves])
    xy = np.array([[d["x"], d["y"]] for d, _, _ in moves])
    moved = np.any(np.abs(np.diff(xy, axis=0)) > 1e-12, axis=1)
    assert moved.any() and np.allclose(lift[1:][moved], 0.0)  # lowest height while driving
    assert lift[-1] == pytest.approx(0.5) and all(ok for _, _, ok in moves)


# --- joint limits ------------------------------------------------------------------------------


def test_robot_joint_limits_clip_to_the_urdf_and_copies_keep_them():
    from core.robot import Robot

    r = Robot.from_registry("ur20")
    full = (r.lower.copy(), r.upper.copy())
    lims = [(-180, 180), (-200, 20), (-170, 170), (-360, 360), (-90, 90), (-720, 720)]
    r.set_limits(lims)
    assert np.allclose(r.lower[[0, 1, 4]], np.radians([-180, -200, -90]))
    assert r.lower[5] == pytest.approx(full[0][5]) and r.upper[5] == pytest.approx(full[1][5])  # URDF
    assert np.allclose(r.model.upperPositionLimit, r.upper)
    copy = Robot(r.spec)  # threads build their own Robot from the spec
    assert np.allclose(copy.lower, r.lower) and np.allclose(copy.upper, r.upper)
    with pytest.raises(ValueError, match="empty joint range"):
        r.set_limits([(10, 0), *lims[1:]])
    with pytest.raises(ValueError, match="each of the 6"):
        r.set_limits(lims[:3])
    r.set_limits(None)
    assert np.allclose(r.lower, full[0]) and r.spec.limits_deg is None

    from opt.reachability import map_limits

    r.set_limits(lims)
    lo, hi = map_limits(r)  # the map keeps joint 1's full turn, the rest narrowed
    assert lo[0] == pytest.approx(full[0][0]) and lo[4] == pytest.approx(np.radians(-90))


def test_joint_limits_panel_sliders_checker_and_project(app, tmp_path):
    from core.feasibility import FeasibilityChecker
    from ui.session import Session, load_project, save_project

    s, cell = app["session"], app["cell"]
    lp = cell.limits_panel
    assert app["project"].load_example("rail_long_part")
    assert len(lp.fields) == 6 and "URDF limits" in lp.info.content
    lp.fields[4].value = (-120.0, 120.0)
    lp.fields[0].value = (-90.0, 270.0)
    assert lp.apply_fields()
    assert s.robot.upper[4] == pytest.approx(np.radians(120)) and "saved in this project" in lp.info.content
    assert cell.sliders[4].min == pytest.approx(-120) and cell.sliders[4].max == pytest.approx(120)
    assert np.all(s.q <= s.robot.upper + 1e-12) and np.all(s.q >= s.robot.lower - 1e-12)
    sim = app["sim"]
    ch = sim._checker()
    assert isinstance(ch, FeasibilityChecker) and ch.hi[4] < np.radians(120)  # minus the margin
    lp.fields[1].value = (50.0, 40.0)
    assert not lp.apply_fields() and lp.fields[1].value[0] < lp.fields[1].value[1]  # reset
    d = save_project(s, tmp_path / "lim")
    s2 = Session()
    load_project(s2, d.parent if d.is_file() else d)
    assert s2.robot.upper[4] == pytest.approx(np.radians(120))
    assert s2.robot.spec.limits_deg[0] == (-90.0, 270.0)
    assert lp.set_limits(None) and s.robot.spec.limits_deg is None and "URDF limits" in lp.info.content


def test_areas_without_zones_get_one_group_per_configuration():
    from types import SimpleNamespace

    import trimesh

    from ui.station_areas import area_meshes, config_groups

    sol = SimpleNamespace(assignment={0: 0, 1: 0, 2: 0, 3: 1}, zones=[])
    cfg = {0: "S+E+W+", 1: "S+E+W-", 2: "S+E+W-", 3: "S+E+W+"}
    assert config_groups(sol, cfg) == {1: 0, 2: 0, 0: 1, 3: 0}  # largest group first
    mesh = trimesh.creation.box((4, 1, 0.01))
    mesh = mesh.subdivide().subdivide().subdivide()
    P = np.array([[-1.5, 0, 0.005], [-0.5, 0, 0.005], [0.5, 0, 0.005], [1.5, 0, 0.005]])
    names = {n for n, *_ in area_meshes(mesh, [0, 1, 2, 3], P, sol, [(200, 0, 0), (0, 0, 200)], cfg)}
    assert names == {"S1_G1", "S1_G2", "S2_G1"}
    plain = {n for n, *_ in area_meshes(mesh, [0, 1, 2, 3], P, sol, [(200, 0, 0), (0, 0, 200)])}
    assert plain == {"S1_G1", "S2_G1"}


def test_tcp_position_is_locked_by_default(app):
    tool, s = app["tool"], app["session"]
    assert app["project"].load_example("rail_long_part")
    tool.refresh()
    assert tool.tcp_lock.value and tool.tcp_pose.xyz.disabled and tool.tcp_pose.gizmo.disable_axes
    before = s.tool.tcps[0].T_head_tcp.xyz
    tool.tcp_pose._apply(np.eye(4), source="fields")  # an edit while locked is undone
    assert s.tool.tcps[0].T_head_tcp.xyz == before
    assert np.allclose(np.array(tool.tcp_pose.xyz.value) / 1000, before)
    tool.tcp_lock.value = False  # unlocked: the edit goes through
    assert not tool.tcp_pose.xyz.disabled and not tool.tcp_pose.gizmo.disable_axes
    T = np.eye(4)
    T[:3, 3] = (0.0, 0.0, 0.123)
    tool.tcp_pose._apply(T, source="fields")
    assert np.allclose(s.tool.tcps[0].T_head_tcp.xyz, (0, 0, 0.123))
    tool.tcp_lock.value = True


# --- jogging in frames -------------------------------------------------------------------------


def test_jog_pose_translates_along_and_turns_about_frame_axes_through_the_tcp():
    from core.jog import jog_pose
    from core.transforms import make_T, rpy_to_matrix

    R_f = rpy_to_matrix((0.0, 0.0, np.pi / 2))  # frame x = world y
    T = make_T(rpy_to_matrix((0.3, -0.2, 0.1)), (1.0, 2.0, 3.0))
    T1 = jog_pose(T, R_f, "X", 0.01)
    assert np.allclose(T1[:3, 3] - T[:3, 3], (0, 0.01, 0)) and np.allclose(T1[:3, :3], T[:3, :3])
    T2 = jog_pose(T, R_f, "Rz", 0.2)
    assert np.allclose(T2[:3, 3], T[:3, 3])  # turns in place
    assert np.allclose(T2[:3, :3], rpy_to_matrix((0, 0, 0.2)) @ T[:3, :3])  # frame z = world z
    T3 = jog_pose(T, T[:3, :3], "Z", 0.05)  # tool frame: along the TCP's own z
    assert np.allclose(T3[:3, 3] - T[:3, 3], 0.05 * T[:3, 2])


@pytest.mark.parametrize("frame", ["World", "Robot base", "Part", "Tool"])
def test_jog_panel_moves_the_tcp_in_its_frame(app, frame):
    s, cell = app["session"], app["cell"]
    if s.part is None or s.robot is None:
        assert app["project"].load_example("rail_long_part")
    jp = cell.jog_panel
    jp.frame.value = frame
    jp.refresh()
    jp.point.value = jp.point.options[-1]  # the tool's TCP
    cell.set_q(np.radians([0, -100, 100, -90, -90, 0]))
    T0 = jp.T_world_point()
    R = jp.R_frame(T0)
    jp.step.value = 20.0
    assert jp.jog("Y", +1)
    T1 = jp.T_world_point()
    assert np.allclose(T1[:3, 3] - T0[:3, 3], 0.02 * R[:, 1], atol=1e-6)
    assert np.allclose(T1[:3, :3], T0[:3, :3], atol=1e-6)
    jp.turn.value = 10.0
    assert jp.jog("Rx", -1)
    T2 = jp.T_world_point()
    assert np.allclose(T2[:3, 3], T1[:3, 3], atol=1e-6)  # turns about the TCP
    assert "xyz mm" in jp.readout.content


def test_jog_panel_joint_steps_and_refusals(app):
    s, cell = app["session"], app["cell"]
    if s.robot is None:
        assert app["project"].load_example("rail_long_part")
    jp = cell.jog_panel
    jp.frame.value = "Joint"
    jp.refresh()
    assert jp.j_plus.visible and not jp.plus.visible
    cell.set_q(np.radians([0, -100, 100, -90, -90, 0]))
    jp.joint_step.value = 5.0
    assert jp.jog("J3", +1) and np.degrees(s.q[2]) == pytest.approx(105)
    jp.frame.value = "World"
    jp.refresh()
    jp.step.value = 5000.0  # 5 m: out of reach
    assert not jp.jog("X", +1) and np.degrees(s.q[2]) == pytest.approx(105)


def test_jog_refuses_steps_into_the_floor_or_the_external_axis(app):
    s, cell = app["session"], app["cell"]
    assert app["project"].load_example("rail_long_part")
    jp = cell.jog_panel
    jp.point.value = jp.point.options[0]  # the flange
    up = np.radians([0, -90, 0, -90, 0, 0])
    down = np.radians([0, 90, 0, -90, 0, 0])  # upper arm pointing down: through the floor / rail
    cell.set_q(up)
    assert jp.coll.hits() == [] and "Collision" not in jp.readout.content
    hits = jp.coll.hits(down)
    assert hits and any(b == "floor" or b.startswith("carrier_") for _, b in hits)
    assert not jp._go(down) and np.allclose(s.q, up)  # refused
    cell.set_q(down)  # a joint-slider move is not refused, but warned about
    assert "Collision" in jp.readout.content
    jp.allow.value = True
    cell.set_q(up)
    assert jp._go(down) and np.allclose(s.q, down)
    jp.allow.value = False
    cell.set_q(up)


def test_sequences_keep_each_face_on_its_own():
    from types import SimpleNamespace

    from core.paths import sequences

    zone = SimpleNamespace(station=0, points=[0, 1, 2, 3, 4, 5])
    sol = SimpleNamespace(zones=[zone], assignment=dict.fromkeys(range(6), 0))
    res = {i: SimpleNamespace(q=np.zeros(6)) for i in range(6)}
    P = {i: np.array([i * 0.1, 0.0, 0.0]) for i in range(6)}
    group = {0: "top", 1: "side", 2: "top", 3: "side", 4: "top", 5: "side"}
    plain = sequences(sol, res, P)
    assert plain == {0: [(0, [0, 1, 2, 3, 4, 5])]}
    split = sequences(sol, res, P, group_of=group)
    assert [sorted(ids) for _, ids in split[0]] == [[0, 2, 4], [1, 3, 5]]
    assert all(z == 0 for z, _ in split[0])  # still the same zone (configuration)


def test_arm_motion_and_unzoned_config_cost():
    from types import SimpleNamespace

    from core.schema import Weights
    from opt.cost import cost_breakdown
    from opt.zoning import arm_motion, config_changes

    res = {
        0: SimpleNamespace(q=np.radians([0, 0, 0, 0, 0, 0])),
        1: SimpleNamespace(q=np.radians([90, 0, 0, 0, 0, 0])),
        2: SimpleNamespace(q=np.radians([90, 0, 0, 0, 0, 350])),  # -10 deg the short way
    }
    seqs = {0: [[0, 1, 2]], 1: [[0], [2]]}
    assert arm_motion(seqs, res) == pytest.approx((1.0 + 10 / 90) / 2)
    assert config_changes(seqs) == 1
    _, terms = cost_breakdown([], 0, [[0, 0]], Weights(motion=2.0, group=3.0), groups=1, motion=0.5)
    assert terms["motion"] == pytest.approx(1.0) and terms["group"] == pytest.approx(3.0)
