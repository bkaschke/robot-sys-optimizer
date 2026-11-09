"""v2 phase 5 (docs/V2_PLAN.md): paths between planned points, joint motion along them,
flagged problems, playback in the Results tab, path CSV."""

import time

import numpy as np
import pytest
import viser

from core.configs import config_label
from core.export import station_pose
from core.paths import CSV_COLUMNS, path_csv, sequences, trace


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8795, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


def _plan(app, zoning: bool):
    s = app["session"]
    assert app["project"].load_example("rail_long_part")
    stp = app["stations"]
    stp.zoning.value = stp.zoning.options[1 if zoning else 0]
    stp.start(block=True)
    sol = min(s.results, key=lambda r: (len(r.unreached), r.cost))
    assert not sol.unreached
    app["results"].choice.value = next(o for o in app["results"].choice.options if sol.label in o)
    return s, sol, s.result_points[sol.label]


def _trace(app, s, sol, res, **kw):
    sim = app["sim"]
    pts = {p.id: (np.array(p.p, float), np.array(p.n, float)) for p in s.targets.points}
    poses = [station_pose(s.base, st, s.T_world_base) for st in sol.stations]
    ch = sim._checker()
    return ch, trace(ch, s.part_mesh(), sol, res, pts, poses, **kw), pts


@pytest.fixture(scope="module")
def zoned(app):
    return _plan(app, zoning=True)


def test_zoned_paths_are_continuous_and_exact_at_the_points(app, zoned):
    s, sol, res = zoned
    ch, plan, pts = _trace(app, s, sol, res, step=0.02)
    assert plan.counts() == dict.fromkeys(plan.counts(), 0)  # nothing flagged on this plan
    seqs = sequences(sol, res, {p: v[0] for p, v in pts.items()})
    assert sum(len(z) for v in seqs.values() for _, z in v) == len(sol.assignment)
    for st in plan.stations:
        ch.set_base(st.T_world_base)
        prev = None
        for p in st.samples:
            assert p.q is not None
            if prev is not None and p.a != p.b:
                assert np.max(np.abs(p.q - prev)) <= np.radians(20) + 1e-9  # no joint jumps
                assert config_label(ch.ik, p.q) == config_label(ch.ik, prev)
            if p.s == 1.0 or p.a == p.b:  # at a planned point: the planned TCP position
                P, N = pts[p.b]
                assert np.linalg.norm(p.T_part_tcp[:3, 3] - (P + res[p.b].standoff * N)) < 1e-9
            # the TCP pose the joints give is the sample pose
            T = np.linalg.inv(ch.T_base_part) @ ch.robot.fk(p.q) @ ch.T_flange_tcp[p.tcp]
            assert np.allclose(T, p.T_part_tcp, atol=1e-6)
            prev = p.q
    assert plan.length() > 0.9 * sum(
        np.linalg.norm(pts[b][0] - pts[a][0])
        for v in seqs.values()
        for _, z in v
        for a, b in zip(z[:-1], z[1:], strict=True)
    )


def test_problems_are_flagged(app, zoned):
    s, sol, res = zoned
    _, plan, _ = _trace(app, s, sol, res, step=0.02, jump_deg=0.05)
    assert plan.counts()["jump"] > 0  # an absurdly small joint step flags jumps
    sim = app["sim"]
    pts = {p.id: (np.array(p.p, float), np.array(p.n, float)) for p in s.targets.points}
    far = [station_pose(s.base, st, s.T_world_base) for st in sol.stations]
    for T in far:
        T[2, 3] += 3.0  # bases 3 m up: the points are out of reach
    plan = trace(sim._checker(), s.part_mesh(), sol, res, pts, far, step=0.05)
    assert plan.counts()["unreachable"] > 0


def test_unzoned_plan_shows_its_reconfigurations(app):
    s, sol, res = _plan(app, zoning=False)
    ch, plan, pts = _trace(app, s, sol, res, step=0.05)
    P = {p: v[0] for p, v in pts.items()}
    seqs = sequences(sol, res, P)
    switches = sum(
        config_label(ch.ik, np.asarray(res[a].q)) != config_label(ch.ik, np.asarray(res[b].q))
        for v in seqs.values()
        for _, z in v
        for a, b in zip(z[:-1], z[1:], strict=True)
    )
    # v6: the path does a stop's points one configuration at a time: one change per extra
    # configuration at a stop, never more than switching along the planner's order would
    lab = {p: config_label(ch.ik, np.asarray(r.q)) for p, r in res.items() if r.q is not None}
    labels = {k: {lab[p] for _, z in v for p in z} for k, v in seqs.items()}
    changes = sum(len(v) - 1 for v in labels.values())
    assert plan.counts()["reconfigure"] == changes <= switches
    split = sequences(sol, res, P, lambda r: config_label(ch.ik, np.asarray(r.q)))
    for k, v in split.items():
        assert len(v) == len(labels[k])
        for _, z in v:  # each group has one configuration
            assert len({config_label(ch.ik, np.asarray(res[p].q)) for p in z}) == 1


def test_path_csv(app, zoned):
    s, sol, res = zoned
    _, plan, _ = _trace(app, s, sol, res, step=0.05)
    rows = path_csv(plan, s.part.T_world_part).strip().splitlines()
    assert rows[0].split(",") == list(CSV_COLUMNS) and len(rows) == len(plan.samples) + 1
    assert all(len(r.split(",")) == len(CSV_COLUMNS) for r in rows[1:])


def test_gui_build_play_scrub_export(app, zoned):
    s, sol, _ = zoned
    sim = app["sim"]
    app["results"].choice.value = next(o for o in app["results"].choice.options if sol.label in o)
    sim.build(block=True)
    assert sim.plan is not None and sim.label == sol.label and "samples" in sim.info.content
    assert sim._handles and not sim.play.disabled and sim.frame.max == len(sim.plan.samples) - 1
    # scrub to the last sample: the carrier sits at its stop, the arm at that sample's joints
    last = sim.plan.samples[-1]
    sim.frame.value = sim.frame.max
    assert np.allclose(s.q, last.q, atol=1e-9)
    c = s.carrier()
    v = app["carrier"].current(c)
    assert np.allclose(s.T_world_base, c.T_world_base(v), atol=1e-9)
    assert v["s"] == pytest.approx(sol.stations[last.station].carrier["s"], abs=1e-6)
    sim.frame.value = 0
    sim.speed.value = 2000.0
    sim.toggle()
    time.sleep(1.0)
    sim.pause()
    time.sleep(0.2)
    assert sim.frame.value > 0  # playback advanced
    path = sim.export_csv()
    assert path.is_file() and path.read_text(encoding="utf-8").startswith("station,")
