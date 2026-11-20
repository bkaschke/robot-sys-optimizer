"""Review v8 package K (R13): ``optimizer.margins`` of the project is the one source of the
clearance and joint-limit margin for the Optimize-tab check, the planners and the worker."""

import pytest
import viser

from core.robot import Robot
from core.schema import Margins, Optimizer, Project
from tests.helpers import confirm_all


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8957, verbose=False)
    state = build_app(server)
    assert state["project"].load_example("rail_long_part")
    confirm_all(state["stations"])
    yield state
    server.stop()


def _set(s, clearance_m, joint_deg):
    opt = s.extra.get("optimizer") or Optimizer()
    s.extra["optimizer"] = opt.model_copy(
        update={"margins": Margins(clearance_m=clearance_m, joint_deg=joint_deg)}
    )


def test_planner_checker_uses_project_margins(app):
    from ui.plan_job import build_checker

    s = app["session"]
    _set(s, 0.025, 8.0)
    ck = build_checker(s, Robot(s.robot.spec), s.sites(), 1, lambda *a: None, lambda *a: None)
    assert ck.scene.margins.clearance == pytest.approx(0.025)
    assert ck.settings.joint_margin_deg == 8.0


def test_optimize_tab_checker_uses_project_margins(app):
    s, tab = app["session"], app["optimize"]
    _set(s, 0.025, 8.0)
    ck = tab._build_checker()
    assert ck.scene.margins.clearance == pytest.approx(0.025)
    assert ck.settings.joint_margin_deg == 8.0


def test_fields_write_and_show_the_project_margins(app):
    s, tab = app["session"], app["optimize"]
    tab.clearance.value = 33.0
    tab.joint_margin.value = 7.0
    m = s.extra["optimizer"].margins
    assert (m.clearance_m, m.joint_deg) == (pytest.approx(0.033), 7.0)
    _set(s, 0.02, 9.0)
    tab.load_margins()
    assert (tab.clearance.value, tab.joint_margin.value) == (pytest.approx(20.0), 9.0)


def test_margins_round_trip_and_reach_the_worker(app):
    from ui.plan_job import payload_for, session_from

    s = app["session"]
    _set(s, 0.025, 8.0)
    p = Project.model_validate_json(s.to_project().model_dump_json())
    assert p.optimizer.margins.clearance_m == 0.025
    w = session_from(payload_for(s, {}, False))
    assert w.extra["optimizer"].margins == Margins(clearance_m=0.025, joint_deg=8.0)


def test_old_project_gets_default_margins(app):
    s, tab = app["session"], app["optimize"]
    data = s.to_project().model_dump(mode="json")
    data["optimizer"].pop("margins", None)
    p = Project.model_validate(data)
    assert p.optimizer.margins == Margins()
    s.extra["optimizer"] = p.optimizer
    tab.load_margins()
    assert (tab.clearance.value, tab.joint_margin.value) == (pytest.approx(10.0), 5.0)


def test_margins_label_in_provenance_text():
    from ui.plan_job import margins_label

    assert margins_label(Margins(clearance_m=0.025, joint_deg=8.0)) == "25 mm / 8 deg"
