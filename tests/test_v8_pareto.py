"""v8 T11: Pareto fronts (opt.pareto, Station planner -> Objective, Results -> Pareto front)."""

import numpy as np
import pytest
import viser

from opt.pareto import dominates, front, measures, vector
from tests.helpers import confirm_all


def test_front_keeps_only_non_dominated_items():
    def m(st, cyc, clr, un=0):
        return {"stations": st, "cycle": cyc, "clearance": clr, "length": 0, "unreached": un}

    items = {
        "a": m(2, 100, 20),
        "b": m(3, 80, 20),  # more stops, faster: on the front
        "c": m(3, 90, 10),  # worse than b everywhere: dominated
        "d": m(2, 100, 20),  # duplicate of a: one kept
        "e": m(1, 50, 90, un=2),  # misses points: never on the front
    }
    keys = ("stations", "cycle", "clearance")
    f = front(list(items), keys, items.get)
    assert f == ["a", "b"]
    assert dominates(vector(items["b"], keys), vector(items["c"], keys))
    assert not dominates(vector(items["a"], keys), vector(items["b"], keys))


@pytest.mark.slow
def test_pareto_objective_on_the_rail():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8860, verbose=False)
    try:
        app = build_app(server)
        assert app["project"].load_example("rail_long_part")
        st, s = app["stations"], app["session"]
        st.areas.value = False
        confirm_all(st)
        st.start(block=True)  # weighted cost: the reference best plan
        ref = s.results[0]
        ref_m = measures(ref, s.result_points[ref.label])
        st.objective_m.value = "Pareto front (NSGA-II)"
        assert all(cb.visible for cb in st.pareto_keys.values())
        confirm_all(st)
        st.start(block=True)
        plans = s.results
        assert plans and all(p.label.startswith("Pareto ") for p in plans)
        keys = ("stations", "cycle", "clearance")
        vs = [vector(measures(p, s.result_points[p.label]), keys) for p in plans]
        assert not any(dominates(a, b) for a in vs for b in vs)
        v_ref = vector(ref_m, keys)
        assert any(dominates(v, v_ref) or np.allclose(v, v_ref) for v in vs)  # the front covers it
        view = app["pareto"]
        assert view.folder.visible and view.image is not None and "on the front" in view.note.content
        st.objective_m.value = "Weighted cost"
    finally:
        server.stop()


@pytest.mark.slow
def test_tool_optimizer_pareto_designs():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8869, verbose=False)
    try:
        app = build_app(server)
        assert app["project"].load_example("rail_long_part")
        tool, s = app["tool_opt"], app["session"]
        tool.shape_evals.value = 6
        tool.objective.value = "Pareto front (NSGA-II)"
        assert all(cb.visible for cb in tool.pareto_keys.values())
        confirm_all(tool)
        tool.start(block=True)
        ds = tool.result.designs
        assert ds and all(d.solution.label.startswith("Pareto ") for d in ds)
        keys = ("length", "stations", "cycle")
        vs = [
            vector(measures(d.solution, d.point_results, s.tool_model.length(d.values)), keys) for d in ds
        ]
        assert not any(dominates(a, b) for a in vs for b in vs)
        assert any(r.label.startswith("Pareto ") for r in s.results)
        tool.objective.value = "Weighted cost"
    finally:
        server.stop()
