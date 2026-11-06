"""v2 phase 2 (docs/V2_PLAN.md): guided steps, robot + carrier lock, mode-specific fields,
out-of-date results, save optimized tool, object tree and arrow jog."""

from types import SimpleNamespace

import numpy as np
import pytest
import viser

from core.transforms import make_T
from ui.staleness import components, solution_changes, stamp_for, step_state


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8791, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


def test_tabs_are_steps_in_setup_order():
    from app import TAB_TITLES, TABS, UR_NOTICE

    assert TABS[:4] == ("Cell", "External Axis", "Tool Setup", "Targets") and TABS[4:] == (
        "Optimize",
        "Results",
    )
    assert len(TAB_TITLES) == len(TABS)
    assert "© 2023 Universal Robots" in UR_NOTICE and "Â" not in UR_NOTICE  # was double-encoded


# --- steps and the lock --------------------------------------------------------------------------


def test_confirm_step_then_edit_marks_it_changed(app):
    s, wf, cell = app["session"], app["workflow"], app["cell"]
    assert app["project"].load_example("rail_long_part")
    assert step_state(s, "Cell") == "todo"
    wf.confirm("Cell")
    md, ok, _ = wf.headers["Cell"]
    assert step_state(s, "Cell") == "confirmed" and "confirmed" in md.content and ok.disabled
    T = s.part.T_world_part
    cell.part_pose._apply(make_T(T[:3, :3], T[:3, 3] + [0.1, 0, 0]), source="fields")
    wf.refresh()
    assert step_state(s, "Cell") == "changed" and "changed" in md.content and not ok.disabled
    assert "1 Robot & cell" in wf.opt_md.content  # Optimize lists the unconfirmed steps


def test_confirmed_carrier_setup_is_locked(app):
    s, wf, car, cell = app["session"], app["workflow"], app["carrier"], app["cell"]
    assert app["project"].load_example("mobile_amr")
    assert cell.base.xyz.disabled and cell.mount.disabled  # movable: the robot rides the carrier
    assert not car.locked and car.editor.bodies_folder.visible
    wf.confirm("External Axis")
    assert car.locked
    assert not car.axis_folder.visible and not car.editor.bodies_folder.visible and not car.apply.visible
    assert car.dof_folder.visible and car.locked_note.visible  # jogging stays possible
    _, ok, edit = wf.headers["External Axis"]
    assert edit.visible and not ok.visible
    # the assembly moves together
    car.set_values({"x": 0.5, "y": 1.0, "yaw": 0.3})
    c = s.carrier()
    assert np.allclose(s.T_world_base, c.T_world_base(car.current(c)), atol=1e-9)
    wf.edit("External Axis")
    assert not car.locked and car.editor.bodies_folder.visible and ok.visible
    # a static base is free again in the Cell tab
    car.mode.value = "static"
    car.commit()
    wf.refresh()
    assert not cell.base.xyz.disabled and not cell.mount.disabled


def test_fields_follow_mode_and_axis_type(app):
    car = app["carrier"]
    assert app["project"].load_example("rail_long_part")
    assert car.frame_xyz.visible and not car.ty.visible and not car.tyaw.visible and not car.loc.visible
    car.kind.value = "mobile"
    assert not car.frame_xyz.visible and car.ty.visible and car.tyaw.visible and car.loc.visible
    assert "search" in car.tx.label
    car.mode.value = "static"
    assert not car.kind.visible and not car.editor.bodies_folder.visible and not car.dof_folder.visible
    st = app["stations"]
    from ui.stations import ZONING

    st.zoning.value = ZONING[0]
    assert not st.config_cost.visible
    st.zoning.value = ZONING[1]
    assert st.config_cost.visible
    st.zoning.value = ZONING[0]


# --- out-of-date results -----------------------------------------------------------------------


def test_results_go_out_of_date_and_exports_ask(app):
    s, wf, res = app["session"], app["workflow"], app["results"]
    assert app["project"].load_example("rail_long_part")
    app["stations"].start(block=True)
    sol = res.selected()
    assert sol is not None and s.stamps[sol.label]
    wf.refresh()
    assert wf.results_md.content == "" and solution_changes(s, sol) == []
    # change the targets: the plan is out of date and says why
    s.targets = s.targets.model_copy(update={"points": s.targets.points[:-1]})
    wf.refresh()
    assert solution_changes(s, sol) == ["targets"] and "Out of date" in wf.results_md.content
    asked = []
    ex = res.export
    ex._ask = lambda e, fn, changes: asked.append(changes)
    ex._run(SimpleNamespace(client=object()), ex.export_csv)
    assert asked == [["targets"]]  # the export asks first
    # stamps survive save + load
    from ui.session import Session, load_project, save_project

    save_project(s)
    s2 = Session()
    load_project(s2, s.project_dir)
    assert s2.stamps[sol.label] == s.stamps[sol.label]


def test_applying_the_tool_result_does_not_outdate_it(app):
    s = app["session"]
    assert app["project"].load_example("rail_long_part")
    tool_stamp = stamp_for(s, "tool")
    plan_stamp = stamp_for(s, "plan")
    x = s.tool.extension
    nominal = x.nominal.model_copy(update={"L": (x.L[0] + x.L[1]) / 2})
    s.tool = s.tool.model_copy(update={"extension": x.model_copy(update={"nominal": nominal})})
    assert stamp_for(s, "tool") == tool_stamp  # the tool optimizer chooses the nominal
    assert stamp_for(s, "plan")["tool"] != plan_stamp["tool"]  # a plan used the old one


def test_save_optimized_tool_to_library(app):
    from core.tool import TOOLS_DIR, load_tool
    from opt.tool_geometry import GeometryResult

    s, panel = app["session"], app["tool_opt"]
    v = s.tool.extension.nominal.model_copy(update={"L": 0.123})
    panel.result = GeometryResult(True, v, None, {}, [])
    name = "zz_test_opt_tool"
    try:
        path = panel.save_to_library(name)
        saved = load_tool(name)
        assert path.is_file() and saved.extension.nominal.L == pytest.approx(0.123)
        assert s.tool.name != name  # the tool in use is unchanged
    finally:
        (TOOLS_DIR / f"{name}.yaml").unlink(missing_ok=True)


# --- objects and jog ---------------------------------------------------------------------------------


def test_object_visibility_and_gizmo_choice(app):
    ob, cell = app["objects"], app["cell"]
    assert app["project"].load_example("rail_long_part")
    ob.show["Robot"].value = False
    assert cell.view.root.visible is False
    ob.show["Robot"].value = True
    assert cell.view.root.visible is True
    ob.show["Part"].value = False
    assert cell.part_handle.visible is False
    ob.show["Part"].value = True
    ob.gizmo.value = "Part"
    assert cell.part_pose.gizmo.disable_axes is False and cell.base.gizmo.disable_axes is True


def test_jog_rail_clamped_mobile_unlimited_part_and_base(app):
    s, ob, car = app["session"], app["objects"], app["carrier"]
    ob.step.value = 1000.0
    assert app["project"].load_example("rail_long_part")
    hi = car._dof["s"].max
    for _ in range(20):
        ob.jog("x", +1)
    assert car._dof["s"].value == pytest.approx(hi)  # rail stops at the end of its travel
    assert app["project"].load_example("mobile_amr")
    x_max = car._dof["x"].max
    for _ in range(int(x_max / 1000) + 3):
        ob.jog("x", +1)
    c = s.carrier()
    assert car.current(c)["x"] > x_max / 1000  # past the old slider range: no limit
    assert np.allclose(s.T_world_base, c.T_world_base(car.current(c)), atol=1e-9)
    yaw0 = car.current(c)["yaw"]
    ob.angle.value = 90.0
    ob.jog("rz", +1)
    turn = (np.degrees(car.current(c)["yaw"] - yaw0) + 180.0) % 360.0 - 180.0  # yaw wraps
    assert turn == pytest.approx(90.0, abs=1e-6)
    # part
    ob.move.value = "Part"
    p0 = s.part.T_world_part[:3, 3].copy()
    ob.step.value = 100.0
    ob.jog("y", -1)
    assert np.allclose(s.part.T_world_part[:3, 3] - p0, [0, -0.1, 0], atol=1e-9)
    ob.move.value = "Robot / carrier"
    # static base
    car.mode.value = "static"
    car.commit()
    b0 = s.T_world_base[:3, 3].copy()
    ob.jog("z", +1)
    assert np.allclose(s.T_world_base[:3, 3] - b0, [0, 0, 0.1], atol=1e-9)
    assert components(s)["robot"]  # hashes still compute for a static base
