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
