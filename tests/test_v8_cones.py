"""v8 T15: approach cones (opt.approach_cones) and Start shape: From approach cones."""

import numpy as np
import pytest
import trimesh
import viser

from opt.approach_cones import bend_candidates, cap_directions, free_directions
from tests.helpers import confirm_all


def test_cap_directions_stay_in_the_cone():
    n = np.array([0.3, -0.2, 0.9])
    U = cap_directions(n, 40.0, 50)
    c = U @ (n / np.linalg.norm(n))
    assert np.allclose(np.linalg.norm(U, axis=1), 1) and c.min() >= np.cos(np.radians(40)) - 1e-9
    assert np.allclose(U[0], n / np.linalg.norm(n))


def test_a_point_in_a_slot_is_reached_only_along_the_slot():
    floor = trimesh.creation.box((1.0, 1.0, 0.02), trimesh.transformations.translation_matrix((0, 0, -0.01)))
    walls = [
        trimesh.creation.box((1.0, 0.1, 0.3), trimesh.transformations.translation_matrix((0, y, 0.15)))
        for y in (-0.08, 0.08)  # a 60 mm wide, 300 mm deep slot along x
    ]
    mesh = trimesh.util.concatenate([floor, *walls])
    P, N = np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.0]])
    free = free_directions(mesh, P, N, length=0.5, radius=0.01)[0]
    assert len(free) > 3
    assert np.all(np.abs(free[:, 1]) < 0.2)  # nothing across the slot
    assert np.any(np.abs(free[:, 0]) > 0.6)  # but steep ones along it
    # the same point on an open plate: directions all round
    open_free = free_directions(floor, P, N, length=0.5, radius=0.01)[0]
    assert len(open_free) == 80 and np.abs(open_free[:, 1]).max() > 0.8


def test_bend_candidates_from_cones():
    P = np.zeros((3, 3))
    base = np.array([1.0, 0, 0])  # the arm comes from +x
    straight_up = [np.array([[0, 0, 1.0]])] * 3  # free only straight up: 90 deg from the base side
    c = bend_candidates(straight_up, P, base)
    assert c == [90.0]
    toward = [np.array([[1.0, 0, 0]])] * 3  # free toward the base: no bend needed
    assert bend_candidates(toward, P, base) == [0.0]
    assert bend_candidates([np.zeros((0, 3))], P[:1], base) == []


@pytest.mark.slow
def test_cone_seeds_and_display_on_the_rail():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8873, verbose=False)
    try:
        app = build_app(server)
        assert app["project"].load_example("rail_long_part")
        heat = app["targets"].heat
        heat.cones.value = True
        assert heat._cone_handle is not None and "every point has one" in heat.cones_note.content
        heat.cones.value = False
        assert heat._cone_handle is None
        tool = app["tool_opt"]
        tool.shape_evals.value = 6
        cost = {}
        for start in ("Nominal", "From approach cones"):
            tool.start_shape.value = start
            confirm_all(tool)
            tool.start(block=True)
            cost[start] = tool.result.solution.cost
        assert cost["From approach cones"] <= cost["Nominal"] + 1e-9
        tool.start_shape.value = "Nominal"
    finally:
        server.stop()
