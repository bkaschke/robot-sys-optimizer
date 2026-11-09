"""The shipped example projects load through Project -> Examples and plan end to end:
full coverage, carrier DOFs in range, 3D standoff respected, robot base on the carrier."""

import numpy as np
import pytest
import viser

from core.standoff import distance_3d
from ui.project_panel import list_examples


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8773, verbose=False)
    yield build_app(server)
    server.stop()


def test_examples_are_listed():
    assert {"rail_long_part", "mobile_amr", "gantry_overhead", "floor_xy_rails", "mobile_lift"} <= set(
        list_examples()
    )


@pytest.mark.parametrize(
    "name,kind,n_bodies",
    [
        ("rail_long_part", "rail", 3),
        ("mobile_amr", "mobile", 7),
        ("floor_xy_rails", "gantry", 5),  # v6: X rails + bridge (gantry_overhead: same code path, 96 s)
        ("mobile_lift", "mobile", 8),  # v6: lift height planned per stop
    ],
)
def test_example_loads_and_plans(app, name, kind, n_bodies):
    s, panel = app["session"], app["carrier"]
    assert app["project"].load_example(name)
    assert "workspace" in str(s.project_dir) and name in str(s.project_dir)
    assert s.base.carrier == kind and len(s.base.bodies) == n_bodies
    assert len(s.targets.points) >= 20 and s.tool is not None and s.part is not None
    c = s.carrier()
    # the robot sits on the carrier at the saved pose (sliders follow the base)
    assert np.allclose(s.T_world_base, c.T_world_base(c.clip(panel.values())), atol=1e-6)
    assert panel.editor.pick.options == tuple(b.name for b in s.base.bodies)

    st = app["stations"]  # the panel defaults after opening the example (v1: max stations stayed 1)
    assert st.max_stations.value == 4
    st.start(block=True)
    assert s.results, st.status.content
    best = min((r for r in s.results if not r.unreached), key=lambda r: len(r.stations))
    assert len(best.assignment) == len(s.targets.points)
    part = s.part_mesh()
    for station in best.stations:
        v = {d: station.carrier[d] for d in c.dofs if d in station.carrier}
        for d, (lo, hi) in zip(c.dofs, c.bounds, strict=True):
            if d != "yaw":
                assert lo - 1e-9 <= v[d] <= hi + 1e-9
        bodies = c.moving_bodies(v) + c.static_bodies()
        assert distance_3d(bodies, part, s.part.T_world_part) >= s.base.standoff_m.carrier_part_min - 1e-6
    if "lift" in c.dofs:  # v6: each stop stores its own planned height; the 3 m wall needs it raised
        assert all("lift" in station.carrier for station in best.stations)
        assert max(station.carrier["lift"] for station in best.stations) > 0.25
