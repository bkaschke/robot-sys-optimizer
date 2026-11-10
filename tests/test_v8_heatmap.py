"""v8 T8: Targets -> Display -> Colour part by (ui.part_heatmap)."""

import numpy as np
import pytest
import viser

from tests.helpers import confirm_all
from ui.part_heatmap import BINS, OFF, bins, colour


def test_colours_and_bins_are_monotonic():
    s = np.linspace(-10, 60, 200)
    b = bins(s, 0.0, 50.0)
    assert b.min() == 0 and b.max() == BINS - 1 and np.all(np.diff(b) >= 0)
    c = np.array([colour(f) for f in np.linspace(0, 1, 50)])
    assert np.all(np.diff(c[:, 0]) <= 0) and np.all(np.diff(c[:, 1]) >= 0)  # red fades, green grows
    assert colour(0)[0] > colour(0)[1] and colour(1)[1] > colour(1)[0]


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8857, verbose=False)
    st = build_app(server)
    assert st["project"].load_example("rail_long_part")
    confirm_all(st["stations"])
    st["stations"].start(block=True, quick=True)
    yield st
    server.stop()


def test_heatmap_follows_the_selected_plan_and_switches_off(app):
    heat = app["targets"].heat
    assert heat.choice.value == OFF and not heat._handles  # default: the pre-v8 view
    heat.choice.value = "Joint-limit margin"
    assert heat._handles and "points: lowest" in heat.legend.content
    painted = heat.face_bins[heat.face_bins >= 0]
    assert len(painted) > 0
    heat.choice.value = "Clearance"
    assert heat._handles
    heat.choice.value = "Tolerance reach rate"  # no tolerance check yet
    assert not heat._handles and "tolerance check" in heat.legend.content
    app["tolerance"].samples.value = 3
    app["tolerance"].start(block=True)  # its on_done redraws the part
    assert heat._handles and heat.face_bins.max() == BINS - 1  # every point reached: all green
    heat.choice.value = OFF
    assert not heat._handles and heat.legend.content == ""
