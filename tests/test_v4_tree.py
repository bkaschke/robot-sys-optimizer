"""v3 feedback item 1: the object tree in a panel on the left: every object and its parts
(tool, carrier bodies, targets, selection, stay-out zones, result overlays) can be hidden."""

import pytest
import viser


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8801, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


def test_tree_has_the_cell_hierarchy(app):
    ob = app["objects"]
    for key in (
        "Robot",
        "Robot/Tool",
        "Part/Targets",
        "Part/Selection",
        "Results/Paths",
        "Results/Ghost robots",
    ):
        assert key in ob.show


def test_robot_and_tool_hide_separately(app):
    ob, cell, tool = app["objects"], app["cell"], app["tool"]
    ob.show["Robot/Tool"].value = False
    assert tool.view.root.visible is False and cell.view.root.visible is True
    ob.show["Robot/Tool"].value = True
    assert tool.view.root.visible is True


def test_carrier_bodies_listed_and_hidden_one_by_one(app):
    ob, car = app["objects"], app["carrier"]
    assert app["project"].load_example("mobile_amr")
    ob.apply_visibility()  # (the watcher does this every second in the app)
    names = list(car._body_handles)
    assert names and all(f"External axis/{n}" in ob.show for n in names)
    ob.show[f"External axis/{names[0]}"].value = False
    assert car._body_handles[names[0]].visible is False and car._body_handles[names[1]].visible is True
    ob.show["External axis"].value = False  # the parent hides every body
    assert all(h.visible is False for h in car._body_handles.values())
    ob.show["External axis"].value = True
    ob.show[f"External axis/{names[0]}"].value = True
    assert all(h.visible is True for h in car._body_handles.values())


def test_stayout_zones_in_the_tree(app):
    ob, panel, s = app["objects"], app["stayouts"], app["session"]
    assert panel.add_shape("box") and panel.add_shape("cylinder")
    ob.apply_visibility()
    a, b = (z.name for z in s.stayouts)
    assert f"Stay-out zones/{a}" in ob.show and f"Stay-out zones/{b}" in ob.show
    ob.show[f"Stay-out zones/{a}"].value = False
    assert panel._handles[a].visible is False and panel._handles[b].visible is True
    ob.gizmo.value = "Stay-out zone"
    assert panel.drag.value and panel.gizmo is not None
    ob.gizmo.value = "Robot base"
    panel.pick.value = a
    assert panel.remove_zone()
    ob.apply_visibility()
    assert f"Stay-out zones/{a}" not in ob.show and f"Stay-out zones/{b}" in ob.show
