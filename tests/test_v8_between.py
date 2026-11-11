"""Moves between groups at a stop: a planned collision-free joint path (default) or via home;
stops still start and end at home (Results -> Paths & simulation -> *Between groups*)."""

import numpy as np
import pytest
import viser

from core.configs import config_label
from core.cycle_time import CycleSettings, path_time
from core.export import station_pose
from core.paths import trace
from tests.helpers import confirm_all


def _runs(samples):
    """Joint-move runs (consecutive kind "home" samples) as lists of samples."""
    out, cur = [], []
    for p in samples:
        if p.kind == "home":
            cur.append(p)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


@pytest.fixture(scope="module")
def rail():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8823, verbose=False)
    app = build_app(server)
    s = app["session"]
    assert app["project"].load_example("rail_long_part")
    stp = app["stations"]
    stp.zoning.value = stp.zoning.options[0]  # best per point: configuration changes at a stop
    confirm_all(stp)
    stp.start(block=True)
    sol = min(s.results, key=lambda r: (len(r.unreached), r.cost))
    yield app, s, sol, s.result_points[sol.label]
    server.stop()


def _trace(app, s, sol, res, between):
    ch = app["sim"]._checker()
    pts = {p.id: (np.array(p.p, float), np.array(p.n, float)) for p in s.targets.points}
    poses = [station_pose(s.base, st, s.T_world_base) for st in sol.stations]
    plan = trace(
        ch, s.part_mesh(), sol, res, pts, poses, step=0.05, home=s.home_q(),
        group_of={p.id: p.group for p in s.targets.points}, between=between,
    )  # fmt: skip
    return ch, plan


def test_trace_rejects_an_unknown_mode(rail):
    app, s, sol, res = rail
    with pytest.raises(ValueError):
        _trace(app, s, sol, res, "teleport")


@pytest.mark.slow
def test_between_groups_direct_or_via_home(rail):
    app, s, sol, res = rail
    home = s.home_q()
    ch, direct = _trace(app, s, sol, res, "direct")
    _, via = _trace(app, s, sol, res, "home")
    assert via.counts()["reconfigure"] > 0  # the plan changes configuration within stops
    n_direct_changes = 0
    for sd, sv in zip(direct.stations, via.stations, strict=True):
        for st in (sd, sv):  # both: every stop starts and ends at home
            assert np.allclose(st.samples[-1].q, home)
            assert st.samples[0].kind == "home" and st.samples[0].a == 0
        inner_v = _runs(sv.samples)[1:-1]
        inner_d = _runs(sd.samples)[1:-1]

        def at_home(r):
            return any(np.allclose(p.q, home, atol=1e-6) for p in r)

        # via home: every move between groups goes to home and out again (the others are
        # surface moves turned into joint moves around a collision: the same in both modes)
        assert len(inner_v) == len(inner_d) and any(at_home(r) for r in inner_v)
        assert all(at_home(rv) or len(rv) == len(rd) for rv, rd in zip(inner_v, inner_d, strict=True))
        # direct: moves between groups go straight from point to point, clear of collisions
        for r in inner_d:
            assert not [p for p in r if p.issue]
            if not at_home(r) and config_label(ch.ik, r[0].q) != config_label(ch.ik, r[-1].q):
                n_direct_changes += 1
        assert sum(map(len, inner_d)) < sum(map(len, inner_v))
    assert n_direct_changes > 0  # configuration changes done without going home
    assert direct.counts()["collision"] == 0
    cs = CycleSettings()
    vlim = ch.robot.model.velocityLimit
    t_direct = path_time(direct, cs, vlim)["total"]
    t_via = path_time(via, cs, vlim)["total"]
    assert t_direct < t_via  # skipping home is faster
    print(f"cycle direct {t_direct:.1f} s, via home {t_via:.1f} s")


def test_sim_panel_has_the_between_groups_choice(rail):
    from ui.settings import SAVED
    from ui.simulate import BETWEEN_LABELS

    sim = rail[0]["sim"]
    assert sim.between.value == "Collision-free joint path"
    assert set(BETWEEN_LABELS.values()) == {"direct", "home"} and "between" in SAVED["sim"]


def test_targets_watcher_during_a_build_keeps_the_new_paths(rail):
    """The 1 s targets watcher ticking while a build finishes (here: during the motion check)
    must not clear the half-built plan ('NoneType' has no attribute 'stations')."""
    app, s, sol, _ = rail
    sim = app["sim"]
    sim.reset("test")
    sim.motion.value = True
    orig = sim._build_progress

    def tick(f):
        orig(f)
        sim.check_targets()

    sim._build_progress = tick
    try:
        sim.build(block=True)
    finally:
        sim._build_progress = orig
        sim.motion.value = False
    assert sim.plan is not None and sim.label == sim.selected().label
    assert "failed" not in sim.info.content
