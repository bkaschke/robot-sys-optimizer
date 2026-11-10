"""v7: points on opposite sides of a part (core.facing + Targets -> Each side separately)."""

import numpy as np
import pytest
import viser

from core import golden_parts as gp
from core import targets as tg
from core.facing import facing_sides
from tests.helpers import confirm_all


def _box():
    import trimesh

    return trimesh.creation.box((1.0, 0.6, 0.02))


# --- core -----------------------------------------------------------------------------------------


def test_closed_box_splits_into_its_six_sides():
    b = _box()
    sides = facing_sides(b, np.arange(len(b.faces)))
    assert len(sides) == 6 and sorted(np.concatenate(sides)) == list(range(len(b.faces)))
    normals = [b.face_normals[s].mean(0).round(6) for s in sides]
    assert {tuple(n) for n in normals} == {
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
    }  # fmt: skip
    assert np.isclose(b.area_faces[sides[0]].sum(), 0.6)  # largest first (top or bottom)
    # the thin edge faces (<= 0.02 m2) drop out below a 150 mm grid cell
    assert len(facing_sides(b, np.arange(len(b.faces)), min_area=0.15**2)) == 2


def test_curved_panel_stays_one_side_and_cylinder_wraps_into_quarters():
    import trimesh

    p = gp.curved_panel()
    outer = tg.select_by_normal(p, (0, 0, 1), 60)
    assert len(facing_sides(p, outer)) == 1  # smooth and facing one way: unchanged behaviour
    c = trimesh.creation.cylinder(0.3, 1.0, sections=64)
    wall = np.flatnonzero(np.abs(c.face_normals[:, 2]) < 0.1)
    sides = facing_sides(c, wall)
    assert len(sides) == 4 and sum(map(len, sides)) == len(wall)


# --- GUI ------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8851, verbose=False)
    st = build_app(server)  # flat plate 1.0 x 0.6 x 0.02
    yield st
    server.stop()


def test_grid_on_top_and_bottom_of_the_plate(app):
    tab, s = app["targets"], app["session"]
    s.targets = s.targets.model_copy(update={"points": []})
    m = tab.mesh()
    top_bottom = np.flatnonzero(np.abs(m.face_normals[:, 2]) > 0.99)
    tab.select(top_bottom, "set")
    tab.method.value = "Surface grid (rows/columns)"
    tab.row.value, tab.col.value, tab.inset.value, tab.row_angle.value = 100.0, 100.0, 0.0, 0.0
    tab.direction.value = (0.0, 0.0, 0.0)
    tab.split_sides.value = False  # one direction for both sides: refused, as before
    assert tab.run_generate() == 0 and "opposite" in tab.gen_info.content
    tab.split_sides.value = True
    n = tab.run_generate()
    _, P, N, _ = tg.as_arrays(s.targets)
    up, down = N[:, 2] > 0.99, N[:, 2] < -0.99
    assert n == len(P) and up.sum() == down.sum() == n // 2 > 0 and (up | down).all()
    assert np.allclose(P[up, 2] - P[down, 2].mean(), 0.02)  # 20 mm apart: not duplicates
    assert "2 sides" in tab.gen_info.content
    for method in ("Projected grid", "Raster lines"):
        s.targets = s.targets.model_copy(update={"points": []})
        tab.method.value = method
        tab.spacing.value, tab.stepover.value = 100.0, 100.0
        assert tab.run_generate() > 0
        _, P, N, _ = tg.as_arrays(s.targets)
        assert (N[:, 2] > 0.99).any() and (N[:, 2] < -0.99).any(), method


# --- robot-agnostic IK (OPW) ----------------------------------------------------------------------


def test_opw_solver_round_trip_labels_and_manipulability():
    from core.ik import angle_diff, make_ik
    from core.robot import Robot

    robot = Robot.from_registry("generic_6r")  # made-up arm, KUKA-style offsets and sign flips
    ik = make_ik(robot)
    assert ik.batch and ik.n_branches == 8
    rng = np.random.default_rng(3)
    for _ in range(300):
        q = robot.random_q(rng)
        T = robot.fk(q)
        sols = ik.solve(T)
        assert any(np.abs(angle_diff(s, q)).max() < 1e-6 for s in sols)
        assert max(np.abs(robot.fk(s) - T).max() for s in sols) < 1e-9
    Ts = np.array([robot.fk(robot.random_q(rng)) for _ in range(50)])
    Q, V = ik.solve_batch(Ts)
    labels = {ik.config_label(Q[0, b]) for b in range(8) if V[0, b]}
    assert len(labels) == V[0].sum()  # every branch its own configuration
    for _ in range(10):  # |det J| from the solver = Pinocchio's
        q = robot.random_q(rng)
        assert ik.manipulability(q) == pytest.approx(abs(np.linalg.det(robot.jacobian(q))), rel=1e-6)


def test_opw_parameters_must_match_the_urdf():
    from core.ik.opw import OPWIK
    from core.robot import Robot

    robot = Robot.from_registry("generic_6r")
    robot.spec = robot.spec.model_copy(update={"opw": robot.spec.opw.model_copy(update={"c2": 0.65})})
    with pytest.raises(ValueError, match="do not match the URDF"):
        OPWIK(robot)


@pytest.mark.slow
def test_plan_with_a_non_ur_robot(app):
    s, cell = app["session"], app["cell"]
    cell.robot_choice.value = "generic_6r"
    assert s.robot.spec.name == "generic_6r" and np.allclose(s.q, s.robot.home_q())
    tab = app["targets"]
    s.targets = s.targets.model_copy(update={"points": []})
    m = tab.mesh()
    tab.select(np.flatnonzero(m.face_normals[:, 2] > 0.99), "set")
    tab.method.value = "Surface grid (rows/columns)"
    tab.row.value = tab.col.value = 150.0
    assert tab.run_generate() > 0
    st = app["stations"]
    confirm_all(st)
    st.start(block=True)  # builds (or loads) the generic arm's reachability map
    assert s.results, "the station planner refused or failed"
    best = s.results[0]
    reached = len(s.targets.points) - len(best.unreached)
    assert reached > 0 and all(
        r.config.startswith("S") for r in s.result_points[best.label].values() if r.config
    )
    cell.robot_choice.value = "ur20"


# --- clearer status, bug fixes ----------------------------------------------------------------


def test_log_minimize_and_run_durations(app):
    import time

    from ui import log_panel

    log_panel.clear()
    log_panel.log("older", "info")
    log_panel.log("Planning finished", "done", since=time.monotonic() - 65)
    assert log_panel.entries()[-1][2].endswith("_(1 min 5 s)_")
    panel = app["log"]
    panel.minimize(True)
    assert "Planning finished" in panel.body.content and "older" not in panel.body.content
    assert not panel.setup.visible and panel.min_btn.label == "Show full log"
    panel.minimize(False)
    assert "older" in panel.body.content and panel.setup.visible


def test_out_of_date_feasibility_is_not_marked(app):
    from core.feasibility import PointResult
    from ui.staleness import stamp_for

    s, t = app["session"], app["targets"]
    if not s.targets.points:
        m = t.mesh()
        t.select(np.flatnonzero(m.face_normals[:, 2] > 0.99), "set")
        t.method.value = "Surface grid (rows/columns)"
        t.row.value = t.col.value = 150.0
        t.run_generate()
    ids = [p.id for p in s.targets.points]
    s.feasibility = {i: PointResult(i, "unreachable", "no_ik") for i in ids[:2]}
    s.stamps["feasibility"] = stamp_for(s, "feasibility")
    t.selected_results = lambda: {}
    t._draw_points()
    assert len(t._handles["unreached"].points) == 2
    T = s.part.T_world_part.copy()
    T[0, 3] += 0.05  # the part moves: the check no longer applies
    s.part.T_world_part = T
    t._draw_points()
    assert "unreached" not in t._handles
    T[0, 3] -= 0.05
    s.part.T_world_part = T


def test_heatmap_errors_are_reported_not_lost(app):
    from ui import log_panel

    pl = app["placement"]

    def boom(event=None):
        raise RuntimeError("map exploded")

    pl.compute, saved = boom, pl.compute
    try:
        log_panel.clear()
        pl._compute_safe(None)  # runs in a thread in the app: must not raise there
        assert any("Base placement failed" in e[2] and "map exploded" in e[2] for e in log_panel.entries())
    finally:
        pl.compute = saved
