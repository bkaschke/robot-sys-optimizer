"""v7 round 2, phase C (docs/V7_PLAN.md T13-T15): axis lengths, quick plan, rules-only plan."""

import numpy as np
import pytest
import viser

from core.carrier import Carrier, body_list
from core.carrier_axis import apply_axis_lengths, axis_travel
from core.schema import Base, CarrierJoints, Pose, Primitive
from tests.helpers import confirm_all


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8785, verbose=False)
    app = build_app(server)
    app["server"] = server
    yield app
    server.stop()


def _rail(L=6.6, manual=False, x=(-3.0, 3.0)):
    return Base(
        mode="movable",
        carrier="rail",
        bodies=[
            Primitive(name="track", shape="box", size=(7.0, 0.3, 0.1), attach="fixed", stretch="x"),
            Primitive(name="carriage", shape="box", size=(0.5, 0.5, 0.15)),
        ],
        axis_length_m={"x": L},
        manual_travel=manual,
        joints=CarrierJoints(x=x),
    )


def _extent_x(bodies, name):
    b = next(b for b in bodies if b.name == name)
    v = b.mesh.vertices @ b.T[:3, :3].T + b.T[:3, 3]
    return float(np.ptp(v[:, 0]))


# --- T13 ----------------------------------------------------------------------------------------


def test_rail_length_sets_track_and_travel():
    for L in (4.0, 8.0):
        base = apply_axis_lengths(_rail(L))
        assert base.joints.x == pytest.approx((-(L / 2 - 0.25 - 0.05), L / 2 - 0.25 - 0.05))
        c = Carrier(base, np.eye(4))
        assert _extent_x(c.static_bodies(), "track") == pytest.approx(L)  # visual / standoff geometry
        from opt.sites import CarrierSites

        _, world = CarrierSites(c, None, None).scene_bodies()  # the collision geometry
        mesh = next(m for n, m, *_ in world if n == "track")
        assert float(np.ptp(mesh.vertices[:, 0])) == pytest.approx(L)
    # manual limits override the computed ones (the track still follows the length)
    base = apply_axis_lengths(_rail(5.0, manual=True, x=(-1.0, 0.5)))
    assert base.joints.x == (-1.0, 0.5)
    assert next(b for b in base.bodies if b.name == "track").size[0] == pytest.approx(5.0)
    with pytest.raises(ValueError, match="too short"):
        axis_travel(_rail(0.5))


def test_old_carriers_are_unchanged():
    old = _rail().model_copy(
        update={
            "bodies": [b.model_copy(update={"stretch": ""}) for b in _rail().bodies],
            "axis_length_m": {},
        }
    )
    assert apply_axis_lengths(old) is old
    assert [b.size for b in body_list(old)] == [b.size for b in old.bodies]
    legacy = Base.model_validate(
        {
            "mode": "movable",
            "carrier": "rail",
            "track": {"shape": "box", "size": [7, 0.3, 0.1]},
            "joints": {"x": [-3, 3]},
        }
    )
    assert body_list(legacy)[0].size == (7.0, 0.3, 0.1)


def test_gantry_cylinder_and_bridge_follow_lengths():
    base = Base(
        mode="movable",
        carrier="gantry",
        bodies=[
            Primitive(  # a round X rail lying along x: its height is the length, centre kept
                name="rod",
                shape="cylinder",
                radius=0.05,
                height=2.0,
                pose=Pose(xyz=(-1.0, 1.0, 0.0), rpy=(0.0, np.pi / 2, 0.0)),
                attach="fixed",
                stretch="x",
            ),
            Primitive(name="bridge", shape="box", size=(0.2, 1.0, 0.1), attach="x_axis", stretch="y"),
            Primitive(name="carriage", shape="box", size=(0.4, 0.4, 0.1)),
        ],
        axis_length_m={"x": 4.0, "y": 3.0},
        joints=CarrierJoints(x=(-1, 1), y=(-1, 1)),
    )
    out = apply_axis_lengths(base)
    rod = next(b for b in out.bodies if b.name == "rod")
    assert rod.height == pytest.approx(4.0) and rod.pose.xyz[0] == pytest.approx(-2.0)  # centre at 0
    assert next(b for b in out.bodies if b.name == "bridge").size[1] == pytest.approx(3.0)
    assert out.joints.x == pytest.approx((-(2.0 - 0.2 - 0.05), 2.0 - 0.2 - 0.05))
    assert out.joints.y == pytest.approx((-(1.5 - 0.2 - 0.05), 1.5 - 0.2 - 0.05))


def test_gui_rail_length(app):
    s, panel = app["session"], app["carrier"]
    panel.kind.value = "rail"
    panel.mode.value = "movable"
    from ui.carrier_ui import DEFAULTS

    panel.load(DEFAULTS["rail"])
    assert panel.commit()
    assert panel.len_x.visible and not panel.len_y.visible and panel.tx.disabled
    assert s.base.joints.x == pytest.approx((-3.0, 3.0))
    panel.len_x.value = 5000.0
    assert panel.commit()
    assert s.base.joints.x == pytest.approx((-2.2, 2.2))
    assert next(b for b in s.base.bodies if b.name == "track").size[0] == pytest.approx(5.0)
    assert tuple(panel.tx.value) == pytest.approx((-2200.0, 2200.0))
    panel.manual_travel.value = True
    assert not panel.tx.disabled
    panel.tx.value = (-1000.0, 1500.0)
    assert panel.commit()
    assert s.base.joints.x == pytest.approx((-1.0, 1.5))
    panel.len_x.value = 300.0  # too short
    panel.manual_travel.value = False
    assert not panel.commit()
    assert s.base.joints.x == pytest.approx((-1.0, 1.5))  # the last valid setup is kept


# --- T14 ----------------------------------------------------------------------------------------


@pytest.mark.slow
def test_quick_plan_full_coverage_and_fast(app):
    import time

    s, st = app["session"], app["stations"]
    assert app["project"].load_example("rail_long_part")
    confirm_all(s)
    app["workflow"].refresh(force=True)
    assert not st.quick_run.disabled
    st.start(block=True, quick=True)  # warm up (reachability map, meshes)
    t0 = time.monotonic()
    st.start(block=True, quick=True)
    t_quick = time.monotonic() - t0
    assert s.results and all(r.label.startswith("Quick: ") for r in s.results)
    best = min(s.results, key=lambda r: (len(r.unreached), r.cost))
    assert not best.unreached and len(best.assignment) == len(s.targets.points)
    assert best.label in s.result_points and not best.zones
    assert "not optimized" in st.quick_status.content
    t0 = time.monotonic()
    st.start(block=True)
    t_full = time.monotonic() - t0
    assert not any(r.label.startswith("Quick: ") for r in s.results)
    assert t_quick < t_full / 3 + 0.5, (t_quick, t_full)  # loose: the PC is slow under load


# --- T15 ----------------------------------------------------------------------------------------


def _rules(app, objective, pct=5.0):
    s, st = app["session"], app["stations"]
    st.mode.value = "Rules only"
    st.objective.value = objective
    st.sing_pct.value = pct
    confirm_all(s)
    st.start(block=True)
    assert len(s.results) == 1, st.status.content
    sol = s.results[0]
    return sol, s.result_points[sol.label]


@pytest.mark.slow
def test_rules_only_planner(app):
    from core.feasibility import CheckSettings

    st = app["stations"]
    assert app["project"].load_example("rail_long_part")
    try:
        assert st.costs.folder.visible  # cost mode shows the costs
        st.mode.value = "Rules only"
        assert st.rules_folder.visible and not st.costs.folder.visible
        a, res_a = _rules(app, "Fewest stations")
        assert not a.violations and not a.unreached  # passing = every point reached
        assert all(r.status != "unreachable" for r in res_a.values())
        assert a.planner == "rules: fewest stations" and a.zones
        assert sum(z.wrap_jumps for z in a.zones) == 0
        b, _ = _rules(app, "Fewest groups")
        assert not b.violations and not b.unreached
        assert len(a.stations) <= len(b.stations) and len(b.zones) <= len(a.zones)
        # a high singularity limit breaks rule 6, and the plan says so. The rail example meets
        # even the GUI's highest limit (50 %: its points have poses up to 98 % of the max), so
        # the test sets 95 % directly (above the GUI field's max)
        c, res_c = _rules(app, "Fewest stations", 95.0)
        assert c.violations and any(v.startswith("rule 6:") for v in c.violations), c.violations
        assert any(r.reason == "singular" for r in res_c.values())
        assert "No plan meets every rule" in st.status.content and "rule 6" in st.status.content
        assert CheckSettings().min_manip == 0.0  # cost mode: no singularity rule
    finally:
        st.mode.value = "Cost based"
    assert st.costs.folder.visible and not st.rules_folder.visible
