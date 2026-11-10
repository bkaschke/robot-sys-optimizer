"""v8 T4 (method picker, guide search link) and the Phase C method dropdowns."""

import pytest
import viser

from opt.methods import METHODS, by_name, key_of, options


def test_every_slot_keeps_todays_method_first():
    first = {slot: ms[0].key for slot, ms in METHODS.items()}
    assert first == {
        "station_refine": "cmaes",
        "station_objective": "weighted",
        "tool_shape": "cmaes",
        "tool_start": "nominal",
        "tool_loop": "fixed",
        "tool_objective": "weighted",
        "point_order": "pattern",
    }
    for slot, ms in METHODS.items():
        assert len({m.name for m in ms}) == len(ms) and all(m.short for m in ms)
        assert by_name(slot, "no such method") is ms[0]  # unknown saved names: the default
        assert key_of(slot, ms[-1].name) == ms[-1].key == by_name(slot, ms[-1].key).key
        assert options(slot)[0] == ms[0].name


@pytest.fixture(scope="module")
def server():
    s = viser.ViserServer(host="127.0.0.1", port=8853, verbose=False)
    yield s
    s.stop()


def test_picker_shows_the_chosen_methods_description(server):
    from ui.method_picker import MethodPicker

    p = MethodPicker(server.gui, "station_refine", "Refinement")
    assert p.key == "cmaes" and "population" in p.note.content
    p.dropdown.value = "Bayesian (Gaussian process)"  # as a click in the browser
    assert p.key == "gp" and "smooth model" in p.note.content
    p.value = "none"  # restore from saved settings by key or label
    assert p.value == "None" and "fastest" in p.note.content
    p.visible = False
    assert not p.dropdown.visible and not p.note.visible


def test_guide_search_link():
    from ui.guide import link_html

    html = link_html(8081, "stay-out zone")
    assert "?q=stay-out%20zone" in html and "Search the guide for" in html
    assert "?q=" not in link_html(8081)
