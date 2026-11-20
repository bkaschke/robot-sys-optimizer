"""Review v8 package N (integration): margins in pool workers, tool optimizer, simulation and
jog checks; staleness of margins / collision geometry; GUI load of margins; deviation warning;
tolerance stamp; load warnings; undo of delete-all."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import viser

from core import collision
from core.robot import Robot
from core.schema import Margins, Optimizer
from tests.helpers import confirm_all
from ui.session import save_project


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8958, verbose=False)
    state = build_app(server)
    assert state["project"].load_example("rail_long_part")
    confirm_all(state["stations"])
    yield state
    server.stop()


def _set(s, clearance_m, joint_deg=8.0):
    opt = s.extra.get("optimizer") or Optimizer()
    s.extra["optimizer"] = opt.model_copy(
        update={"margins": Margins(clearance_m=clearance_m, joint_deg=joint_deg)}
    )


def _spec(s):
    from opt.parallel import CheckerSpec
    from ui.plan_job import checker_config

    sites = s.sites()
    bb, wb = sites.scene_bodies() if sites is not None else ([], [])
    cs = CheckerSpec(
        s.robot.spec,
        s.tool_model,
        s.part_mesh(),
        s.T_world_base,
        s.part.T_world_part,
        bb,
        wb,
        s.obstacle_entries(),
    )
    cs.margins, cs.settings = checker_config(s)
    return cs


# --- 1: pool workers -----------------------------------------------------------------------


def test_pool_built_checker_uses_session_margins(app):
    s = app["session"]
    _set(s, 0.025, 8.0)
    spec = _spec(s)
    ck = spec.build()  # what a pool worker does
    assert ck.scene.margins.clearance == pytest.approx(0.025)
    assert ck.settings.joint_margin_deg == 8.0
    k25 = spec.key()
    _set(s, 0.010, 8.0)
    assert _spec(s).key() != k25  # a margins change builds a new pool


def test_build_checker_hands_margins_to_the_pool_spec(app, monkeypatch):
    import opt.parallel as par
    from ui.plan_job import build_checker

    s = app["session"]
    _set(s, 0.025, 8.0)
    seen = []
    monkeypatch.setattr(par, "pool_for", lambda cs, w: seen.append(cs))
    build_checker(s, Robot(s.robot.spec), s.sites(), 2, lambda *a: None, lambda *a: None)
    assert seen and seen[0].margins.clearance == pytest.approx(0.025)
    assert seen[0].settings.joint_margin_deg == 8.0


# --- 2: tool optimizer ---------------------------------------------------------------------


def test_tool_job_passes_margins(app, monkeypatch):
    import opt.tool_geometry as tg
    from core.schema import Weights
    from ui.plan_job import payload_for, tool_job

    s = app["session"]
    _set(s, 0.025, 8.0)
    got = {}

    def fake(*a, **kw):
        got.update(kw)
        return SimpleNamespace(designs=[], solution=None)

    monkeypatch.setattr(tg, "optimize_tool", fake)
    spec = {"weights": Weights(), "max_stations": 1, "st": tg.GeometrySettings()}
    tool_job(payload_for(s, spec, live=True), lambda *a: None, lambda: False)
    assert got["margins"].clearance == pytest.approx(0.025)
    assert got["settings"].joint_margin_deg == 8.0


def test_tool_evaluator_checker_uses_margins(app):
    from core.feasibility import CheckSettings
    from core.ik import make_ik
    from opt.tool_eval import _Evaluator

    s = app["session"]
    ev = _Evaluator.__new__(_Evaluator)
    ev.robot = Robot(s.robot.spec)
    ev.ik = make_ik(ev.robot)
    ev.part, ev.T_part, ev.T_base0 = s.part_mesh(), s.part.T_world_part, s.T_world_base
    ev.sites, ev.obstacles, ev.cancelled = s.sites(), [], None
    ev.tool = lambda v: s.tool_model
    ev.margins, ev.settings = collision.Margins(clearance=0.025), CheckSettings(joint_margin_deg=8.0)
    ck = ev.checker(None)
    assert ck.scene.margins.clearance == pytest.approx(0.025)
    assert ck.settings.joint_margin_deg == 8.0


# --- 3: simulation and jog checks ----------------------------------------------------------


def test_sim_and_jog_checks_use_margins(app):
    from ui.cell_collision import CellCollision
    from ui.sim_build import path_checker

    s = app["session"]
    _set(s, 0.025, 8.0)
    ck = path_checker(s)
    assert ck.scene.margins.clearance == pytest.approx(0.025)
    assert ck.settings.joint_margin_deg == 8.0
    cc = CellCollision(s)
    assert cc.scene().margins.clearance == pytest.approx(0.025)
    k = cc._setup_key()
    _set(s, 0.010, 8.0)
    assert cc._setup_key() != k  # the scene is rebuilt with the new margin
    assert cc.scene().margins.clearance == pytest.approx(0.010)


# --- 4: staleness --------------------------------------------------------------------------


def test_margin_or_collision_mode_change_marks_plan_stale(app):
    from core.collision_mesh import Geom
    from ui.staleness import changed, stamp_for

    s = app["session"]
    _set(s, 0.010)
    old = stamp_for(s, "plan")
    assert {"collision", "collision_geom", "margins"} <= set(old)
    assert changed(old, stamp_for(s, "plan")) == []
    _set(s, 0.025)
    assert changed(old, stamp_for(s, "plan")) == ["clearance / joint margins"]
    _set(s, 0.010)
    from core import collision_config

    before = collision_config.current()
    try:
        collision_config.update(geom={**before.geom, "part": Geom(mode="low")})
        assert changed(old, stamp_for(s, "plan")) == ["collision geometry"]
    finally:
        collision_config.use(before)


def test_old_stamp_without_new_keys_is_not_flagged(app):
    from ui.staleness import changed, stamp_for

    s = app["session"]
    new = stamp_for(s, "plan")
    old = {k: v for k, v in new.items() if k not in ("collision", "margins")}
    assert changed(old, new) == []


# --- 5: GUI load / settings ----------------------------------------------------------------


def test_loading_a_project_shows_and_keeps_its_margins(app, tmp_path):
    from ui import settings as ui_settings

    s, panel, tab = app["session"], app["project"], app["optimize"]
    for name, m in (("a", 0.010), ("b", 0.025)):
        _set(s, m)
        save_project(s, tmp_path / name)
    assert panel._load(None, tmp_path / "a")
    assert tab.clearance.value == pytest.approx(10)
    assert panel._load(None, tmp_path / "b")
    assert tab.clearance.value == pytest.approx(25)
    s.ui = {"optimize.clearance": 10.0, "optimize.joint_margin": 3.0}  # an older GUI save
    panel.on_loaded()
    ui_settings.apply(app, s.ui)
    assert tab.clearance.value == pytest.approx(25)
    assert s.extra["optimizer"].margins.clearance_m == pytest.approx(0.025)
    saved = ui_settings.collect(app)
    assert "optimize.clearance" not in saved and "optimize.check_from" in saved


# --- 6: deviation warning ------------------------------------------------------------------


def test_deviation_warning_reaches_the_user(app, monkeypatch):
    import core.collision_mesh as cm
    import ui.optimize as uo
    from ui.plan_job import build_checker

    s = app["session"]
    _set(s, 0.025)
    monkeypatch.setattr(cm, "deviation_warning", lambda rep, c: f"dev {c:.3f}")
    emitted = []
    build_checker(s, Robot(s.robot.spec), s.sites(), 1, lambda *a: None, lambda k, d: emitted.append((k, d)))
    assert ("warn", "dev 0.025") in emitted
    seen = []
    monkeypatch.setattr(uo, "log", lambda text, kind="info", *a: seen.append((kind, text)))
    app["optimize"]._build_checker()
    assert ("warn", "dev 0.025") in seen


# --- 7: tolerance stamp --------------------------------------------------------------------


def test_tolerance_result_takes_a_plan_stamp():
    from opt.tolerance import ToleranceResult, ToleranceSettings

    r = ToleranceResult("x", ToleranceSettings())
    assert r.plan_stamp is None
    r.plan_stamp = {"part": "a"}
    assert r.plan_stamp == {"part": "a"}


# --- 8: CSV comment rows -------------------------------------------------------------------


def test_ursim_reads_csv_with_comment_rows(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import ursim_check

    f = tmp_path / "t.csv"
    f.write_text(
        "# OUT OF DATE: Part\n# not checked\n"
        "station,status,q1_deg,q2_deg,q3_deg,q4_deg,q5_deg,q6_deg\n1,ok,1,2,3,4,5,6\n"
    )
    assert ursim_check.read_stops(f) == {1: [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]}


# --- 10: load warnings ---------------------------------------------------------------------


def test_load_warnings_are_notified(app, tmp_path, monkeypatch):
    import ui.project_panel as pp

    s = app["session"]
    save_project(s, tmp_path / "p")
    f = tmp_path / "p" / "project.json"
    d = json.loads(f.read_text())
    d["robot"]["joint_limits_deg"] = [[10.0, 5.0]] * 6
    f.write_text(json.dumps(d))
    got = []
    monkeypatch.setattr(pp, "_notify", lambda e, title, body, ok=False: got.append((title, body)))
    assert app["project"]._load(None, tmp_path / "p")
    assert any(t == "Load warning" and "Joint limits" in b for t, b in got)
    assert got[-1][0] == "Loaded"


# --- 11: undo of delete-all ----------------------------------------------------------------


def test_undo_restores_a_delete_all(app):
    t = app["targets"]
    s = app["session"]
    assert app["project"].load_example("rail_long_part")
    n = len(s.targets.points)
    assert n > 0
    t._delete_all()
    assert not s.targets.points
    assert t.tools.undo_last() == n
    assert len(s.targets.points) == n
    assert "undo_last" not in vars(t.tools)  # no instance override any more


def test_collision_report_names_stay_importable():
    from core.collision import HITS, describe_hits
    from core.collision_hits import describe_hits as d2  # v12 name of the split-out module

    assert describe_hits is d2 and HITS.startswith("collides")
