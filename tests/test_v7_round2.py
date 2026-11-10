"""v7 round 2, phase B (docs/V7_PLAN.md T5-T12): workflow gating and panels."""

import pytest
import viser

from tests.helpers import confirm_all


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8783, verbose=False)
    app = build_app(server)
    app["server"] = server
    assert app["project"].load_example("rail_long_part")
    yield app
    server.stop()


def test_optimizers_need_a_confirmed_setup(app):
    s, wf = app["session"], app["workflow"]
    st, to = app["stations"], app["tool_opt"]
    s.steps.clear()
    wf.refresh(force=True)
    for panel in (st, to):
        assert panel.run.disabled and panel.gate_md.visible
        assert "Confirm **1 Robot & cell**" in panel.gate_md.content
    s.results = []
    st.start(block=True)  # refused: the GUI state cannot be bypassed
    assert not s.results and st._thread is None
    confirm_all(s)
    wf.refresh()
    assert not st.run.disabled and not to.run.disabled and not st.gate_md.visible
    tool = s.tool.model_copy(deep=True)
    tool.name += "_edited"  # edit the tool
    s.tool = tool
    wf.refresh()
    assert st.run.disabled and to.run.disabled
    assert "**3 Tool** changed since you confirmed it" in st.gate_md.content
    assert "Robot & cell" not in st.gate_md.content
    # the feasibility check and the heatmap stay usable
    assert not app["optimize"].run.disabled


def test_delete_buttons_follow_generate(app):
    tab = app["targets"]
    order = [tab.generate, tab.delete, tab.delete_all]
    # viser keeps an insertion order per container: delete comes right after Generate
    ids = [h._impl.props.order for h in order]
    assert ids == sorted(ids) and ids[2] - ids[0] == 2
    assert tab.delete._impl.parent_container_id == tab.generate._impl.parent_container_id
    assert tab.assign._impl.parent_container_id != tab.generate._impl.parent_container_id


def test_jog_panel_docked_below_the_tree(monkeypatch):
    from viser._gui_handles import PanelHandle

    from app import build_app

    calls = []
    for name in ("dock_left", "dock_below"):
        orig = getattr(PanelHandle, name)

        def rec(self, *a, _name=name, _orig=orig):
            calls.append((_name, self, *a))
            return _orig(self, *a)

        monkeypatch.setattr(PanelHandle, name, rec)
    from ui import log_panel

    keep = log_panel._panel  # a second app replaces the module's log panel: put it back after
    server = viser.ViserServer(host="127.0.0.1", port=8784, verbose=False)
    try:
        app = build_app(server)
        objects = app["objects"]
        left = [c[1] for c in calls if c[0] == "dock_left"]
        below = [c for c in calls if c[0] == "dock_below" and c[1] is objects.jog_panel]
        assert below and below[0][2] is left[0]  # docked below the object tree, after it
        assert calls.index(below[0]) > calls.index(next(c for c in calls if c[0] == "dock_left"))
        # the jog widgets and both slots live in the jog panel, not in the tree's tab
        tabs = objects.jog_panel._impl.props._tab_container_ids
        assert objects.jog_folder._impl.parent_container_id in tabs
        assert objects.carrier_slot is not None and objects.arm_slot is not None
    finally:
        server.stop()
        log_panel._panel = keep


def test_log_rows_are_coloured_and_escaped(app):
    from ui import log_panel

    log_panel.clear()
    for level in ("error", "warn", "done", "info"):
        log_panel.log(f"{level} **message** with _note_", level)
    log_panel.log("<script>alert(1)</script> x_y_z", "info")
    body = app["log"].body.content
    for level, (light, _) in log_panel.COLORS.items():
        assert f"color:{light}" in body, level
    assert "<b>message</b>" in body and "<i>note</i>" in body
    assert "<script>" not in body and "&lt;script&gt;" in body
    assert "x_y_z" in body  # snake_case is not italic
    log_panel.set_dark(True)
    try:
        assert log_panel.COLORS["error"][1] in app["log"].body.content
        app["log"].minimize(True)  # the newest row keeps its colour
        assert app["log"].body.content.count("<div") == 1
        assert log_panel.COLORS["info"][1] in app["log"].body.content
    finally:
        app["log"].minimize(False)
        log_panel.set_dark(False)


def test_each_of_three_stayout_zones_can_be_picked_and_moved(app):
    p = app["stayouts"]
    p.s.stayouts = []
    p.refresh()
    for _ in range(3):
        assert p.add_shape("box")
    names = [z.name for z in p.zones]
    assert len({z.pose.xyz for z in p.zones}) == 3  # new zones do not start on top of each other
    p.drag.value = True
    try:
        for k, name in enumerate(names):
            for how in ("dropdown", "click"):
                other = names[(k + 1) % 3]
                p.select(other)  # start from another zone each time
                if how == "dropdown":
                    p.pick.value = name
                else:  # a click on the zone in 3D (the registered handler)
                    h = p._handles[name]
                    assert h._impl.click_cb, "zone takes clicks"
                    h._impl.click_cb[0].callback(None)
                assert p.pick.value == name and p._gizmo_zone == name
                before = {z.name: z.pose for z in p.zones}
                target = (float(k), -1.0 - (how == "click"), 0.0)
                p.gizmo.position = target
                p._from_gizmo(name)
                p._from_gizmo(other)  # a stale event of another zone's gizmo changes nothing
                after = {z.name: z.pose for z in p.zones}
                assert after[name].xyz == pytest.approx(target)
                assert all(after[n] == before[n] for n in names if n != name), (how, name)
        assert not p._handles[p.pick.value]._impl.click_cb  # the zone under the gizmo takes no clicks
    finally:
        p.drag.value = False
        p.s.stayouts = []
        p.refresh()


def test_part_clicks_with_modifiers_and_gizmo_fits_the_part(app):
    from ui.cell import CLICK_MODIFIERS
    from ui.widgets import GizmoStyle

    cell, objects = app["cell"], app["objects"]

    def mods():
        return sorted(str(c.modifier) for c in cell.part_handle._impl.click_cb)

    # plain, Shift and Ctrl clicks select (viser only reports registered combinations); Alt = Move
    assert mods() == sorted([str(m) for m in CLICK_MODIFIERS] + ["alt"])
    objects.gizmo.value = "Part"
    try:  # the selection tools keep working with the part's gizmo on (a browser check found
        assert mods()  # that blocking them made the Targets tab unusable)
        m = app["session"].part_mesh()
        extent = float(max(m.bounds[1] - m.bounds[0]))
        assert GizmoStyle.base_of(cell.part_pose.gizmo) == pytest.approx(min(max(0.6 * extent, 0.25), 2.0))
    finally:
        objects.gizmo.value = "Robot base"
    # only Move mode on the part pauses them, until Confirm / Cancel
    assert objects.start_move("Part")
    assert not cell.part_handle._impl.click_cb
    objects.mover.cancel()
    assert mods() == sorted([str(m) for m in CLICK_MODIFIERS] + ["alt"])


def test_move_mode_cancel_restores_and_confirm_commits(app):
    import numpy as np

    from ui.staleness import step_state

    s, objects, cell = app["session"], app["objects"], app["cell"]
    confirm_all(s)
    objects.rebuild_tree()
    assert "Part" in objects.move_pick.options
    T0 = s.part.T_world_part.copy()
    gizmo_before = objects.gizmo.value
    assert objects.start_move("Part")
    mover = objects.mover
    assert mover.active and objects.gizmo.value == "Part" and mover.panel is not None
    cell.part_pose._apply(
        T0 @ np.diag([1, 1, 1, 1.0]) + np.array([[0, 0, 0, 0.123]] * 3 + [[0] * 4]), "gizmo"
    )
    assert not np.array_equal(s.part.T_world_part, T0)
    mover.cancel()
    assert np.array_equal(s.part.T_world_part, T0)  # bit for bit
    assert objects.gizmo.value == gizmo_before and mover.panel is None
    assert step_state(s, "Cell") == "confirmed"

    assert objects.start_move("Part")
    T1 = T0.copy()
    T1[1, 3] += 0.25
    cell.part_pose._apply(T1, "gizmo")
    mover.confirm()
    assert np.allclose(s.part.T_world_part, T1) and not mover.active
    assert step_state(s, "Cell") == "changed"  # as after any gizmo drag
    cell.part_pose._apply(T0, "code")  # put it back for the other tests

    # a stay-out zone through Alt+click (its registered alt handler)
    p = app["stayouts"]
    p.s.stayouts = []
    p.refresh()
    p.add_shape("box")
    p.add_shape("box")
    first = p.zones[0].name
    pose0 = p.zones[0].pose
    alt = [c for c in p._handles[first]._impl.click_cb if c.modifier == "alt"]
    assert alt
    alt[0].callback(None)
    assert mover.key == f"Stay-out zone/{first}" and p.pick.value == first and p.drag.value
    p.gizmo.position = (3.0, 3.0, 0.0)
    p._from_gizmo(first)
    assert p.zones[0].pose.xyz == pytest.approx((3.0, 3.0, 0.0))
    mover.cancel()
    assert p.zones[0].pose == pose0
    p.s.stayouts = []
    p.refresh()


def test_open_project_popup(app, tmp_path, monkeypatch):
    import ui.session as session_mod
    from ui.open_popup import OpenPopup

    monkeypatch.setattr(session_mod, "LAST_PROJECT", tmp_path / "last_project.txt")
    s, project, gui = app["session"], app["project"], app["server"].gui
    a, b = tmp_path / "a", tmp_path / "b"
    project.save_now(a)
    project.save_now(b)
    session_mod.remember_last(a)
    session_mod.remember_last(b)  # newest first: b, a
    assert session_mod.recent_projects() == [b.resolve(), a.resolve()]

    pop = OpenPopup(gui, project)
    assert pop.reopen_btn.visible and "b  (" in pop.reopen_btn.label
    assert pop.reopen() and pop.closed and s.project_dir.resolve() == b.resolve()

    pop = OpenPopup(gui, project)
    pop.recent_pick.value = pop.recent_pick.options[1]
    assert pop.open_recent() and s.project_dir.resolve() == a.resolve()

    pop = OpenPopup(gui, project)
    pop.example_pick.value = "rail_long_part"
    assert pop.open_example() and pop.closed and s.base.carrier == "rail" and s.targets.points

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "project.json").write_text("{not json", encoding="utf-8")
    session_mod.remember_last(bad)
    pop = OpenPopup(gui, project)
    before = s.project_dir
    assert not pop.reopen() and not pop.closed  # stays open with the error
    assert "Load failed" in pop.error.content and s.project_dir == before

    assert pop.new() and pop.closed
    assert s.robot is not None and s.part is None and s.tool is None and not s.targets.points
    assert s.base.mode == "static" and not s.results and "study_" in s.project_dir.name

    pop = OpenPopup(gui, project)
    assert pop.close() and pop.closed

    # the start-up cell: no autosave until a project is opened, created or saved
    project.untitled = True
    try:
        assert not project.autosave_tick(force=True)
    finally:
        project.untitled = False
    session_mod.LAST_PROJECT.write_text("", encoding="utf-8")
    empty = OpenPopup(gui, project)
    assert not empty.reopen_btn.visible and not empty.recent_pick.visible  # nothing to reopen: hidden
    empty.close()
    assert app["project"].load_example("rail_long_part")  # leave the module's app as it was


# --- phase D -------------------------------------------------------------------------------------


def test_disabled_buttons_say_why(app):
    s, wf, results = app["session"], app["workflow"], app["results"]
    opt, sim, ex = app["optimize"], app["sim"], results.export
    saved = s.targets
    s.results, s.result_points = [], {}
    results.refresh()
    s.targets = s.targets.model_copy(update={"points": []})
    try:
        wf.refresh(force=True)
        assert opt.run.disabled and "Add targets first" in opt.gate_md.content
        assert all(b.disabled for b in (ex.csv, ex.uframes, ex.urdf, ex.report))
        assert "No result to export" in ex.gate_md.content and ex.gate_md.visible
        assert sim.build_btn.disabled and "No plan yet" in sim.gate_md.content
    finally:
        s.targets = saved
    wf.refresh(force=True)
    assert not opt.run.disabled and not opt.gate_md.visible


def test_summary_card_and_inspect_in_results(app):
    from core.feasibility import PointResult
    from core.schema import Solution, Station, Zone
    from ui.summary_card import card

    s = app["session"]
    ids = [p.id for p in s.targets.points]
    sol = Solution(
        label="2 test stations",
        stations=[Station(xyz=(0, 0, 0)), Station(xyz=(1, 0, 0))],
        assignment={i: 0 for i in ids},
        unreached=ids[:1],
        zones=[Zone(station=0, config="S+E+W+", points=ids[1:])],
        violations=["rule 1: 1 point(s) unreached at 2 stations"],
        planner="rules: fewest stations",
        run_s=12.3,
    )
    q = [0.0] * 6
    res = {
        i: PointResult(
            i, "reachable", q=q, clearance=0.02 + 0.001 * k, manipulability=0.3, limit_margin_deg=40.0
        )
        for k, i in enumerate(ids[1:])
    }
    res[ids[0]] = PointResult(ids[0], "unreachable", "singular")
    s.result_points[sol.label] = res
    s.max_manip = 0.6
    md = card(s, sol)
    assert "| Stations | **2** | Groups | **1** |" in md
    assert f"**{len(ids) - 1}/{len(ids)}**" in md and "20 mm" in md and "40°" in md and "50 %" in md
    assert "rules only, fewest stations" in md and "✘ 1 broken" in md and "12 s" in md
    assert card(s, None) == ""

    # T18: Inspect result sits in the Results tab (walk up its containers)
    def ancestors(handle) -> list:
        api, out = handle._impl.gui_api, []
        cid = handle._impl.parent_container_id
        while cid in api._container_handle_from_uuid:
            out.append(cid)
            cid = getattr(
                getattr(api._container_handle_from_uuid[cid], "_impl", None), "parent_container_id", None
            )
        return [*out, cid]

    tabs = app["tabs"]
    assert tabs["Results"]._id in ancestors(app["optimize"].source)
    assert tabs["Optimize"]._id not in ancestors(app["optimize"].source)
