"""v4 plan items 2 and 4: path patterns per stop and group (sand-optimizer orderings),
lifts over crossings, and the arm at home while the carrier moves between stops."""

import numpy as np
import pytest
import viser

from core.export import station_pose
from core.paths import CSV_COLUMNS, ordered_sequences, path_csv, trace
from core.patterns import PATTERNS, PatternSettings, crossing_lifts, order_points

GRID = np.array([(x, y, 0.0) for y in np.arange(0, 0.2, 0.02) for x in np.arange(0, 0.3, 0.02)])
UP = np.tile((0, 0, 1.0), (len(GRID), 1))


# --- patterns -------------------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", PATTERNS)
def test_every_pattern_visits_every_point(pattern):
    o = order_points(GRID, UP, PatternSettings(pattern=pattern))
    visits = 2 if pattern == "Cross-hatch" else 1
    assert len(o.order) == visits * len(GRID)
    assert np.array_equal(np.bincount(o.order, minlength=len(GRID)), np.full(len(GRID), visits))
    assert np.allclose(GRID[o.order[0], :2], (0, 0))  # min-min corner


def test_raster_rows_corner_and_angle():
    zz = order_points(GRID, UP, PatternSettings("Zigzag"))
    steps = np.linalg.norm(np.diff(GRID[zz.order], axis=0), axis=1)
    assert np.allclose(steps, 0.02) and zz.rows.max() + 1 == 10  # rows along x, 20 mm apart
    uni = order_points(GRID, UP, PatternSettings("Unidirectional"))
    row0 = GRID[uni.order[:15]]
    row1 = GRID[uni.order[15:30]]
    assert np.all(np.diff(row0[:, 0]) > 0) and np.all(np.diff(row1[:, 0]) > 0)  # same direction
    o = order_points(GRID, UP, PatternSettings("Zigzag", angle_deg=90, corner="max-max"))
    assert np.allclose(GRID[o.order[:2], :2], [(0, 0.18), (0, 0.16)])  # rows along y, from the top
    conc = order_points(GRID, UP, PatternSettings("Concentric"))
    ring0 = set(conc.order[conc.rows == 0])
    edge = {i for i, p in enumerate(GRID) if p[0] in (0, 0.28) or round(p[1], 6) in (0, 0.18)}
    assert ring0 == edge  # the outer ring first
    out = order_points(GRID, UP, PatternSettings("Concentric", inward=False))
    assert set(out.order[-len(edge) :]) == edge  # inside -> out ends on the border


def test_crossing_moves_lift():
    uv = np.array([[0, 0], [1, 1], [0, 1], [1, 0.0]])
    lifts = crossing_lifts(uv, np.array([0, 1, 2, 3]))
    assert list(lifts) == [False, False, True]  # (0,1)->(1,0) crosses the first move
    zz = order_points(GRID, UP, PatternSettings("Zigzag"))
    assert not crossing_lifts(zz.uv, zz.order).any()
    ch = order_points(GRID, UP, PatternSettings("Cross-hatch"))
    assert not crossing_lifts(ch.uv, ch.order, ch.passes).any()  # passes are checked separately


# --- tracing with the rail example -----------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8804, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


def _plan(app, example: str):
    s = app["session"]
    assert app["project"].load_example(example)
    stp = app["stations"]
    stp.zoning.value = stp.zoning.options[1]  # zones: one configuration per group
    stp.start(block=True)
    sol = min(s.results, key=lambda r: (len(r.unreached), r.cost))
    assert not sol.unreached and len(sol.stations) >= 2
    app["results"].choice.value = next(o for o in app["results"].choice.options if sol.label in o)
    return s, sol, s.result_points[sol.label]


@pytest.fixture(scope="module")
def rail(app):
    return _plan(app, "rail_long_part")


def test_rail_paths_with_pattern_home_and_carrier_moves(app, rail):
    s, sol, res = rail
    sim = app["sim"]
    ch = sim._checker()
    pts = {p.id: (np.array(p.p, float), np.array(p.n, float)) for p in s.targets.points}
    poses = [station_pose(s.base, st, s.T_world_base) for st in sol.stations]
    home = ch.robot.home_q()
    pat = PatternSettings("Zigzag")
    plan = trace(
        ch, s.part_mesh(), sol, res, pts, poses, step=0.05, pattern=pat, home=home,
        carrier_path=sim._carrier_path(sol, 0.05),
    )  # fmt: skip
    assert plan.counts()["carrier_blocked"] == 0
    seqs = ordered_sequences(sol, res, pts, pat)
    assert sum(len(q.pids) for v in seqs.values() for q in v) == len(sol.assignment)
    stops = [st.station for st in plan.stations]
    assert stops == sorted(stops)
    for n, st in enumerate(plan.stations):
        kinds = [p.kind for p in st.samples]
        # carrier move (not at the first stop) -> home move -> surface -> home move
        first_home = kinds.index("home")
        assert set(kinds[:first_home]) == ({"carrier"} if n else set())
        assert kinds[-1] == "home" and np.allclose(st.samples[-1].q, home)
        assert np.allclose(st.samples[first_home].q, home, atol=0.2)  # starts at home
        surface = [p for p in st.samples if p.kind == ""]
        assert {p.b for p in surface} == {pid for pid, k in sol.assignment.items() if k == st.station}
        carrier = [p for p in st.samples if p.kind == "carrier"]
        if carrier:  # the rail moves monotonically from the previous stop to this one
            sv = [p.carrier["s"] for p in carrier]
            prev = sol.stations[plan.stations[n - 1].station].carrier["s"]
            assert np.all(np.diff(np.r_[prev, sv]) * np.sign(sv[-1] - prev) > 0)
            assert np.allclose(carrier[-1].T_world_base, st.T_world_base, atol=1e-9)
            for p in carrier:
                assert np.allclose(p.q, home)
                T = (
                    np.linalg.inv(s.part.T_world_part)
                    @ p.T_world_base
                    @ ch.robot.fk(home)
                    @ ch.T_flange_tcp[p.tcp]
                )
                assert np.allclose(T, p.T_part_tcp, atol=1e-9)
    rows = path_csv(plan, s.part.T_world_part).strip().splitlines()
    assert rows[0].split(",") == list(CSV_COLUMNS) and {"kind", "carrier"} <= set(CSV_COLUMNS)
    kinds = {r.split(",")[1] for r in rows[1:]}
    assert kinds == {"surface", "home", "carrier"}
    assert all(r.split(",")[2].startswith("s=") for r in rows[1:])  # carrier values on every row


def test_lifted_moves_leave_the_surface(app, rail, monkeypatch):
    """Every move flagged as crossing (forced here: the rail's rows never cross) is lifted by
    the lift height in its middle and lands exactly on its points."""
    import core.path_order as cp

    s, sol, res = rail
    ch = app["sim"]._checker()
    pts = {p.id: (np.array(p.p, float), np.array(p.n, float)) for p in s.targets.points}
    poses = [station_pose(s.base, st, s.T_world_base) for st in sol.stations]
    monkeypatch.setattr(cp, "crossing_lifts", lambda uv, order, passes=None: np.ones(len(order) - 1, bool))
    pat = PatternSettings("Zigzag", no_crossing=True)
    plan = trace(ch, s.part_mesh(), sol, res, pts, poses, step=0.01, pattern=pat, lift=0.04)
    lifted = [p for p in plan.samples if p.kind == "lift"]
    assert lifted and not [p for p in plan.samples if p.kind == "" and p.a != p.b]
    mesh = s.part_mesh()
    mid = [p for p in lifted if 0.3 <= p.s <= 0.7]
    _, d, _ = __import__("trimesh").proximity.closest_point(
        mesh, np.array([p.T_part_tcp[:3, 3] for p in mid])
    )
    so = np.array([res[p.b].standoff for p in mid])
    assert np.all(d >= so + 0.04 - 2e-3)  # lifted by the lift height (curvature aside)
    for p in lifted:
        if p.s == 1.0:
            P, N = pts[p.b]
            assert np.linalg.norm(p.T_part_tcp[:3, 3] - (P + res[p.b].standoff * N)) < 1e-9
    assert plan.length() == 0.0  # lifted moves are not surface path


def test_gui_pattern_fields_and_playback_moves_the_carrier(app, rail):
    s, sol, _ = rail
    sim = app["sim"]
    sim.pattern.value = "Concentric"
    assert sim.inward.visible and sim.row_tol.visible and not sim.lift.visible
    sim.no_cross.value = True
    assert sim.lift.visible
    sim.pattern.value = "Shortest path (TSP)"
    assert not sim.inward.visible and not sim.row_tol.visible and sim.corner.visible
    sim.pattern.value = "Planner order"
    assert not sim.corner.visible and not sim.lift.visible
    sim.pattern.value, sim.no_cross.value, sim.home.value = "Zigzag", False, True
    sim.step.value = 50.0
    sim.build(block=True)
    assert sim.plan is not None and "home between stops" in sim.info.content
    i = next(k for k, p in enumerate(sim.plan.samples) if p.kind == "carrier")
    p = sim.plan.samples[i]
    sim.frame.value = i
    assert app["carrier"].current(s.carrier())["s"] == pytest.approx(p.carrier["s"], abs=1e-6)
    assert np.allclose(s.q, p.q) and "carrier moving" in sim.status.content
    sim.frame.value = sim.frame.max  # back home at the last stop
    assert np.allclose(s.q, sim.plan.samples[-1].q)


def test_mobile_contour_transit_follows_the_path(app):
    s, sol, _ = _plan(app, "mobile_amr")
    sites = s.sites()
    a, b = sol.stations[0].carrier, sol.stations[1].carrier
    moves = sites.transit(a, b, 0.05)
    assert moves and all(ok for _, ok in moves)
    last = moves[-1][0]
    assert np.allclose([last[k] for k in ("x", "y")], [b["x"], b["y"]], atol=1e-6)
    if sites.side_at(a) == sites.side_at(b):  # same robot side: along t, keeping the standoff
        ts = [d["t"] for d, _ in moves]
        assert np.all(np.diff(np.r_[a["t"], ts]) * np.sign(b["t"] - a["t"]) > 0)
