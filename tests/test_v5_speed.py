"""v4 plan item 6: coarse to fine (set cover and refinement on a farthest-point subsample,
map-level repair, exact verification of every point) and ghost robots reused on redraw."""

import numpy as np
import pytest
import viser

from core import targets as tg
from opt.sampling import farthest_points, repair_cover, sample_size


def test_sample_size_and_spread():
    assert sample_size(40, 0.25, 150, 60) == 40  # small sets: all points
    assert sample_size(357, 0.25, 150, 60) == 89 and sample_size(2000, 0.25, 150, 60) == 150
    rng = np.random.default_rng(0)
    P = rng.uniform(0, 1, (500, 3))
    S = farthest_points(P, 50)
    assert len(set(S)) == 50 and np.array_equal(S, farthest_points(P, 50))  # deterministic
    d = np.linalg.norm(P[S][:, None] - P[S][None], axis=2) + np.eye(50) * 9
    r = rng.choice(500, 50, replace=False)
    dr = np.linalg.norm(P[r][:, None] - P[r][None], axis=2) + np.eye(50) * 9
    assert d.min() > 2 * dr.min()  # spread out, unlike a random pick


def test_repair_adds_a_stop_for_missed_targets():
    covered = np.array([[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]], bool)
    mu = covered.astype(float)
    assert repair_cover(covered, mu, [0]) == [0, 2]
    assert repair_cover(covered, mu, [0, 2]) == [0, 2]
    assert repair_cover(np.array([[1, 0]], bool), np.ones((1, 2)), [0]) == [0]  # 2nd: nobody covers


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8807, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


def test_dense_targets_plan_on_a_subsample_and_verify_all(app, monkeypatch):
    import opt.placement as pl

    s, tab = app["session"], app["targets"]
    assert app["project"].load_example("rail_long_part")
    group = s.targets.points[0].group
    s.targets = s.targets.model_copy(update={"points": []})
    tab.select(tg.select_by_normal(tab.mesh(), (0, 0, 1), 5.0), "set")
    tab.method.value = "Surface grid (rows/columns)"
    tab.row.value = tab.col.value = 50.0
    tab.inset.value = 20.0
    tab.gen_group.value = group
    n = tab.run_generate()
    assert n > 300
    sizes = []
    real = pl.set_cover
    monkeypatch.setattr(
        pl, "set_cover", lambda cov, mu, k, *a: sizes.append(cov.shape[1]) or real(cov, mu, k, *a)
    )
    stp = app["stations"]
    stp.max_stations.value = 4
    stp.start(block=True)
    assert sizes and max(sizes) <= 150 and max(sizes) < n  # the coarse search
    best = min(s.results, key=lambda r: (len(r.unreached), r.cost))
    assert not best.unreached and len(best.assignment) == n  # every point verified exactly
    res = s.result_points[best.label]
    assert all(res[p].q is not None for p in best.assignment)
    # ghosts are re-posed, not rebuilt, when the Results tab redraws
    r = app["results"]
    pool = dict(r._pool)
    r.draw()
    assert pool and all(r._pool[k][1] is pool[k][1] for k in pool if k in r._pool)
    r.show_ghosts.value = False
    assert not r._pool
    r.show_ghosts.value = True
