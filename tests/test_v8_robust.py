"""v8 T12: robust objective (opt.robust): plans scored by their worst result under tolerance."""

import pytest
import viser

from opt.robust import aggregate
from opt.tolerance import ToleranceSettings
from tests.helpers import confirm_all


def test_aggregate():
    lost = [0, 0, 3, 1, 0, 0, 0, 0, 5, 0]
    assert aggregate(lost, "Worst case") == 5
    assert aggregate(lost, "CVaR 20 %") == pytest.approx(4.0)  # worst 2 of 10: (5 + 3) / 2
    assert aggregate([], "Worst case") == 0


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8871, verbose=False)
    st = build_app(server)
    assert st["project"].load_example("rail_long_part")
    yield st
    server.stop()


@pytest.mark.slow
def test_robust_plan_survives_tolerance_better(app):
    """Part ±350 mm / 4°, stops ±175 mm (the rail keeps its points far beyond realistic
    offsets): judged by an independent 30-sample check, the robust plan reaches every point in
    more samples than the weighted plan (measured 47 % vs 37 %)."""
    from ui.plan_job import tolerance_job

    st, s = app["stations"], app["session"]
    st.areas.value = False
    st.max_stations.value = 3
    tol = ToleranceSettings(part_mm=350, part_deg=4, station_mm=175, samples=20, seed=11)
    st.tolerance_settings = lambda: tol
    judge = ToleranceSettings(part_mm=350, part_deg=4, station_mm=175, samples=30, seed=99)

    def rate(sol):
        spec = {"sol": sol, "results": s.result_points[sol.label], "settings": judge, "workers": 1}
        return tolerance_job({"session": s, "spec": spec}, lambda k, d: None, lambda: False).all_reached

    confirm_all(st)
    st.start(block=True)
    weighted = rate(s.results[0])
    st.objective_m.value = "Robust (worst case)"
    assert st.robust_measure.visible and st.robust_samples.visible
    st.robust_samples.value = 20
    confirm_all(st)
    st.start(block=True)
    best = s.results[0]
    assert best.label.startswith("Robust: ") and "robust" in best.breakdown
    assert rate(best) > weighted
    st.objective_m.value = "Weighted cost"


@pytest.mark.slow
def test_tool_designs_ranked_by_robust_cost(app):
    tool = app["tool_opt"]
    tool.shape_evals.value = 4
    tool.objective.value = "Robust (worst case)"
    confirm_all(tool)
    tool.start(block=True)
    ds = tool.result.designs
    assert ds and all(d.solution.label.startswith("Robust: ") for d in ds)
    tool.objective.value = "Weighted cost"
