"""v8 T21: provenance on every result, and the same plan from the same settings."""

import pytest
import viser

from core.provenance import git_version, settings_hash, stamp
from tests.helpers import confirm_all


def test_settings_hash_is_stable_and_sensitive():
    from core.schema import Weights
    from opt.placement import PlannerSettings

    a = settings_hash(PlannerSettings(), Weights())
    assert a == settings_hash(PlannerSettings(), Weights()) and len(a) == 8
    assert a != settings_hash(PlannerSettings(max_stations=7), Weights())
    assert a != settings_hash(PlannerSettings(), Weights(stations=6.0))
    s = stamp({"refinement": "cmaes"}, PlannerSettings())
    assert s["code"] == git_version() and s["seed"] == "1" and "refinement=cmaes" in s["methods"]


@pytest.mark.slow
def test_same_settings_same_plan_with_provenance():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8878, verbose=False)
    try:
        app = build_app(server)
        assert app["project"].load_example("rail_long_part")
        st, s = app["stations"], app["session"]

        def plan():
            confirm_all(st)
            st.start(block=True)
            return [
                (r.label, round(r.cost, 9), [sorted(x.carrier.items()) for x in r.stations])
                for r in s.results
            ]

        a = plan()
        prov = dict(s.results[0].provenance)
        assert set(prov) == {"code", "methods", "settings", "seed"}
        assert plan() == a  # seeded: the same plan again
        assert s.results[0].provenance == prov
        assert f"settings {prov['settings']}" in app["results"].card.content
    finally:
        server.stop()
