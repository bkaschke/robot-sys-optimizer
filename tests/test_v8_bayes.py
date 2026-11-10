"""v8 T10: Bayesian refinement (opt.refine_bayes) and the Refinement / Shape search dropdowns."""

import numpy as np
import pytest
import viser

from opt.refine_bayes import make_search
from tests.helpers import confirm_all


def _run(method, f, x0, lo, hi, budget, seed=1):
    s = make_search(method, x0, lo, hi, budget, seed)
    xs_all, fs_all = [], []
    while not s.stop():
        xs = s.ask()
        fs = [f(x) for x in xs]
        s.tell(xs, fs)
        xs_all += xs
        fs_all += fs
    return np.array(xs_all), np.array(fs_all)


@pytest.mark.parametrize("method", ["tpe", "gp"])
def test_searches_respect_budget_bounds_and_start_point(method):
    lo, hi = np.array([-1.0, 0.0, 2.0]), np.array([1.0, 0.5, 3.0])
    x0 = np.array([0.2, 0.1, 2.5])
    xs, fs = _run(method, lambda x: float(np.sum(x**2)), x0, lo, hi, 17)
    assert len(fs) == 17
    assert np.all(xs >= lo - 1e-12) and np.all(xs <= hi + 1e-12)
    assert np.allclose(xs[0], x0)  # the current solution is always tried first


def test_gp_finds_a_smooth_minimum_with_few_evaluations():
    target = np.array([0.31, -0.42])
    f = lambda x: float(np.sum((x - target) ** 2))  # noqa: E731
    _, fs = _run("gp", f, np.zeros(2), -np.ones(2), np.ones(2), 25)
    rng = np.random.default_rng(0)
    rand = min(f(x) for x in rng.uniform(-1, 1, (25, 2)))
    assert fs.min() < 0.01 and fs.min() < rand


def test_gp_survives_rejected_candidates():
    """Planner costs are 1e3 / 1e9 where a candidate is rejected: the model must still work."""
    f = lambda x: 1e9 if x[0] > 0.5 else float((x[0] + 0.3) ** 2)  # noqa: E731
    _, fs = _run("gp", f, np.array([0.9]), np.array([-1.0]), np.array([1.0]), 15)
    assert fs.min() < 0.01


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8859, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


def test_old_checkbox_values_map_onto_the_dropdown(app):
    st = app["stations"]
    st.refine.value = False
    assert st.refine.key == "none"
    st.refine.value = True
    assert st.refine.key == "cmaes"


@pytest.mark.slow
@pytest.mark.parametrize("method", ["CMA-ES", "Bayesian (TPE)", "Bayesian (Gaussian process)", "None"])
def test_each_refinement_method_plans_the_rail(app, method):
    st, s = app["stations"], app["session"]
    assert app["project"].load_example("rail_long_part")
    st.refine.value = method
    st.areas.value = False
    st.zoning.value = st.zoning.options[0]
    confirm_all(st)
    st.start(block=True)
    best = s.results[0]
    n = len(s.targets.points)
    assert best.unreached == [] and len(best.assignment) == n
    st.refine.value = "CMA-ES"
