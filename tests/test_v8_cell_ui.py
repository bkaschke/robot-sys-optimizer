"""v8 follow-ups: a confirmed setup step turns its drag gizmos off; the floor grid fits the cell."""

import numpy as np
import pytest
import viser

from ui.floor import MARGIN, MIN_SIZE, cell_points, grid_layout


def _covers(layout, boxes) -> bool:
    (cx, cy), w, h = layout
    B = np.array(boxes)
    lo, hi = B[:, 0].min(axis=0), B[:, 1].max(axis=0)
    return lo[0] >= cx - w / 2 and hi[0] <= cx + w / 2 and lo[1] >= cy - h / 2 and hi[1] <= cy + h / 2


def test_grid_layout_minimum_and_growth():
    assert grid_layout([]) == ((0.0, 0.0), MIN_SIZE, MIN_SIZE)
    small = [np.array([[-1.0, -1.0], [1.0, 1.0]])]
    assert grid_layout(small) == ((0.0, 0.0), MIN_SIZE, MIN_SIZE)
    big = [np.array([[2.0, -3.0], [26.0, 4.0]])]  # a 24 m long part off to one side
    layout = grid_layout(big)
    (cx, cy), w, h = layout
    assert _covers(layout, big) and w >= 24 + 2 * MARGIN and w == int(w) and cx == round(cx)
    assert h == MIN_SIZE


def test_floor_fits_the_examples_and_gizmos_go_off_on_confirm():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8887, verbose=False)
    try:
        app = build_app(server)
        floor, s = app["floor"], app["session"]
        for ex in ("rail_long_part", "gantry_overhead"):
            assert app["project"].load_example(ex)
            boxes = cell_points(s)
            assert _covers(floor.layout, boxes), ex  # on_loaded fitted it
        s.part.T_world_part = s.part.T_world_part.copy()
        s.part.T_world_part[0, 3] += 30.0  # the part moved far away: the floor follows
        layout = floor.update(s)
        assert _covers(layout, cell_points(s)) and layout[1] > MIN_SIZE

        wf, objects, so, tool = app["workflow"], app["objects"], app["stayouts"], app["tool"]
        s.steps.clear()
        wf.refresh(force=True)
        objects.gizmo.value = "Part"
        so.add_shape("box")
        so.drag.value = True
        assert app["cell"].part_pose.show.value
        wf.confirm("Cell")
        assert objects.gizmo.value == "none" and not so.drag.value
        assert not app["cell"].part_pose.show.value and not app["cell"].base.show.value
        objects.gizmo.value = "Part"  # turned back on deliberately: stays on
        wf.refresh(force=True)
        assert objects.gizmo.value == "Part"
        tool.mount.show.value = True
        wf.confirm("Tool Setup")
        assert not tool.mount.show.value and not tool.tcp_pose.show.value
    finally:
        server.stop()


@pytest.mark.parametrize("step", ["Cell", "Tool Setup"])
def test_loaded_confirmed_steps_start_with_gizmos_off(step):
    from app import build_app
    from ui.staleness import confirm_step

    server = viser.ViserServer(host="127.0.0.1", port=8888, verbose=False)
    try:
        app = build_app(server)
        wf, objects = app["workflow"], app["objects"]
        confirm_step(app["session"], step)
        objects.gizmo.value = "Robot base" if step == "Cell" else "Tool mount"
        wf.forget_states()  # as after a project load
        wf.refresh(force=True)
        assert objects.gizmo.value == "none"
    finally:
        server.stop()
