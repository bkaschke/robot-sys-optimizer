"""v6 feedback: carrier moves planned without home moves, colliding surface moves replaced by
planned joint moves, the log panel, unreachable points made obvious, the best few tool designs."""

import numpy as np
import pytest
import viser

from core.export import station_pose
from core.feasibility import PointResult
from core.paths import PathSample, _Tracer, trace
from opt.tool_geometry import top_designs
from tests.helpers import confirm_all


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8846, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


# --- log ------------------------------------------------------------------------------------------


def test_log_panel_messages_and_setup_status(app):
    from ui import log_panel
    from ui.widgets import notify

    log_panel.clear()
    notify(None, "Cannot plan", "Need a tool")
    log_panel.log("done thing", "done")
    rows = log_panel.entries()
    assert [r[1] for r in rows] == ["error", "done"] and "Cannot plan" in rows[0][2]
    body = app["log"].body.content
    assert body.index("done thing") < body.index("Cannot plan")  # newest first
    app["workflow"].refresh(force=True)
    assert "Setup not confirmed" in app["log"].setup.content
    for step in ("Cell", "External Axis", "Tool Setup", "Targets"):
        app["workflow"].confirm(step)
    assert "All setup steps confirmed" in app["log"].setup.content
    assert any("All setup steps confirmed" in r[2] for r in log_panel.entries())


def test_unreached_report_names_reasons_and_points():
    from ui.log_panel import unreached_report

    res = {
        1: PointResult(1, "reachable"),
        2: PointResult(2, "unreachable", "collision"),
        3: PointResult(3, "unreachable", "no_ik"),
        4: PointResult(4, "unreachable", "collision"),
    }
    msg = unreached_report(res, [1, 2, 3, 4], "Plan A")
    assert "3 point(s) unreachable" in msg and "2 × every arm pose collides" in msg
    assert "points 2, 4" in msg and "out of reach" in msg
    assert unreached_report(res, [1], "Plan A") is None


# --- unreachable markers ------------------------------------------------------------------------


def test_unreachable_points_are_marked(app):
    s, t = app["session"], app["targets"]
    assert app["project"].load_example("rail_long_part")
    ids = [p.id for p in s.targets.points]
    s.feasibility = {i: PointResult(i, "unreachable" if i in ids[:3] else "reachable", "no_ik") for i in ids}
    t.selected_results = lambda: {}
    t._draw_points()
    assert "unreached" in t._handles and "unreached_x" in t._handles
    labels = [k for k in t._handles if k.startswith("unreached_label_")]
    assert sorted(int(k.rsplit("_", 1)[1]) for k in labels) == sorted(ids[:3])
    assert len(t._handles["unreached"].points) == 3
    t.show_unreached.value = False
    t._draw_points()
    assert not any(k.startswith("unreached") for k in t._handles)
    t.show_unreached.value = True


# --- tool designs ---------------------------------------------------------------------------------


def test_top_designs_rank_full_coverage_then_unreached_then_cost():
    from core.schema import ExtensionValues, Solution, Station

    def sol(label, unreached, cost):
        return Solution(
            label=label, stations=[Station(xyz=(0, 0, 0))], assignment={}, unreached=unreached, cost=cost
        )

    v = [ExtensionValues(L=x) for x in (0.1, 0.2, 0.3, 0.4, 0.5)]
    cands = [
        (v[0], sol("a", [1, 2], 1.0), {}, False),
        (v[1], sol("b", [1], 9.0), {}, False),
        (v[2], sol("c", [1], 2.0), {}, False),
        (v[3], None, {}, False),  # nothing planned
        (v[4], sol("e", [], 50.0), {}, True),
    ]
    ds = top_designs(cands, lambda u: f"L {u.L * 1000:.0f} mm")
    assert [d.solution.label for d in ds] == ["L 500 mm: e", "L 300 mm: c", "L 200 mm: b"]
    assert [d.feasible for d in ds] == [True, False, False]
    assert ds[1].solution.tool_values.L == pytest.approx(0.3)


def test_tool_panel_offers_designs_that_miss_points(app):
    from core.schema import ExtensionValues, Solution, Station
    from opt.tool_geometry import Design, GeometryResult

    s, panel = app["session"], app["tool_opt"]
    sols = [
        Solution(
            label=f"L {L} mm: 1 station",
            stations=[Station(xyz=(0, 0, 0))],
            assignment={},
            unreached=[1] * k,
            cost=float(k),
        )
        for L, k in ((300, 1), (350, 2))
    ]
    ds = [Design(ExtensionValues(L=0.3 + 0.05 * i), x, {}, False) for i, x in enumerate(sols)]
    panel.result = GeometryResult(
        False, ds[0].values, None, {}, [], "no tool length reaches every target", ds
    )
    panel._pose, panel._stamp = False, {}
    panel._show()
    assert panel.design.visible and len(panel.design.options) == 2 and not panel.apply.disabled
    assert "miss points" in panel.info.content
    assert {x.label for x in sols} <= {r.label for r in s.results}
    panel.design.value = panel.design.options[1]
    assert panel.chosen() is ds[1]


# --- paths ----------------------------------------------------------------------------------------


def test_colliding_surface_move_becomes_a_joint_move():
    tr = _Tracer.__new__(_Tracer)
    tr.park = None
    hit = PathSample(0, None, 1, 2, 0.5, np.eye(4), "tcp", np.zeros(6), "collision")
    ok = PathSample(0, None, 1, 2, 1.0, np.eye(4), "tcp", np.zeros(6))
    detour = [PathSample(0, None, 1, 2, 1.0, np.eye(4), "tcp", np.ones(6), kind="home")]
    tr.joint_move = lambda *a, **k: detour
    tr.nearest_copy = lambda q, ref: np.ones(6)
    tr.tcp_pose = lambda r: np.eye(4)
    rb = PointResult(2, "reachable", q=np.ones(6), tcp="tcp")
    out = tr.around([hit, ok], 0, None, np.zeros(6), rb)
    assert out[:-1] == detour and out[-1].a == out[-1].b == 2 and out[-1].kind == ""
    assert tr.around([ok], 0, None, np.zeros(6), rb) == [ok]  # clear: kept on the surface
    tr.joint_move = lambda *a, **k: [hit, hit]  # the detour is no better: keep the move
    assert tr.around([hit, ok], 0, None, np.zeros(6), rb) == [hit, ok]


@pytest.mark.slow
def test_carrier_drives_between_stops_without_home(app):
    s = app["session"]
    assert app["project"].load_example("rail_long_part")
    stp = app["stations"]
    stp.zoning.value = stp.zoning.options[1]
    confirm_all(stp)
    stp.start(block=True)
    sol = min(s.results, key=lambda r: (len(r.unreached), r.cost))
    assert len(sol.stations) >= 2
    app["results"].choice.value = next(o for o in app["results"].choice.options if sol.label in o)
    res = s.result_points[sol.label]
    sim = app["sim"]
    ch = sim._checker()
    pts = {p.id: (np.array(p.p, float), np.array(p.n, float)) for p in s.targets.points}
    poses = [station_pose(s.base, st, s.T_world_base) for st in sol.stations]
    from ui.sim_build import carrier_path

    plan = trace(
        ch, s.part_mesh(), sol, res, pts, poses, step=0.05, carrier_path=carrier_path(s, sol, 0.05, ch)
    )
    assert plan.counts()["carrier_blocked"] == 0 and plan.counts()["collision"] == 0
    for n, st in enumerate(plan.stations):
        kinds = [p.kind for p in st.samples]
        if n:  # drives in, then a planned joint move to the first point
            assert kinds[0] == "carrier" and "home" in kinds
            car = [p for p in st.samples if p.kind == "carrier"]
            back = [p for p in plan.stations[n - 1].samples if p.kind == "retreat"]
            assert back and all(np.allclose(p.q, back[-1].q) for p in car)  # arm parked, backed off
            assert np.allclose(car[-1].T_world_base, st.T_world_base, atol=1e-9)
        if n + 1 < len(plan.stations):
            assert kinds[-1] == "retreat"
    # the GUI: home off, drive on -> carrier moves in the built plan; drive off -> none
    sim.home.value, sim.drive.value = False, True
    sim.build(block=True)
    assert any(p.kind == "carrier" for p in sim.plan.samples)
    sim.drive.value = False
    sim.build(block=True)
    assert not any(p.kind == "carrier" for p in sim.plan.samples)
    sim.home.value = sim.drive.value = True


# --- stale results, tool costs -------------------------------------------------------------------


def test_results_are_cleared_when_part_tool_or_targets_change(app):
    from core.schema import Solution, Station
    from ui import log_panel
    from ui.staleness import stamp_for

    s = app["session"]
    assert app["project"].load_example("rail_long_part")

    def add(label: str, stamp: dict) -> None:
        s.results = [*s.results, Solution(label=label, stations=[Station(xyz=(0, 0, 0))])]
        s.result_points[label] = {}
        s.stamps[label] = stamp

    old = {**stamp_for(s, "plan"), "part": "saved-by-an-older-version"}
    add("fresh plan", stamp_for(s, "plan"))
    add("loaded out of date", old)
    app["clear_stale"]()  # seen up to date: the fresh plan is now watched
    assert {"fresh plan", "loaded out of date"} <= {r.label for r in s.results}
    T = s.part.T_world_part.copy()
    T[0, 3] += 0.1  # the part moves
    s.part.T_world_part = T
    app["clear_stale"]()
    labels = {r.label for r in s.results}
    assert "fresh plan" not in labels and "fresh plan" not in s.result_points
    assert "loaded out of date" in labels  # never seen up to date: only flagged
    assert any("Cleared 1 result(s): the part changed" in e[2] for e in log_panel.entries())


def test_tool_optimizer_has_its_own_costs(app):
    s = app["session"]
    assert app["project"].load_example("rail_long_part")
    tc, pc = app["tool_opt"].costs, app["stations"].costs
    assert tc is not pc and tc.field == "tool_weights"
    tc.fields["L"].value = 7.5
    tc.fields["stations"].value = 0.25
    opt = s.extra["optimizer"]
    assert opt.tool_weights.L == pytest.approx(7.5) and opt.tool_weights.stations == pytest.approx(0.25)
    assert opt.weights.L != pytest.approx(7.5)  # the planner's costs are unchanged
    assert tc.requirements() == {"require_all": False, "require_one_config": False}
