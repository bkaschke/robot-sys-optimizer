"""v8 T5: tolerance check (opt.tolerance, Results -> Tolerance check)."""

import numpy as np
import pytest
import viser

from opt.tolerance import ToleranceSettings, samples
from tests.helpers import confirm_all
from ui import jobs


def test_samples_are_seeded_and_bounded():
    c = np.array([1.0, 2.0, 0.5])
    a = samples(ToleranceSettings(part_mm=3, part_deg=1, station_mm=4, samples=20, seed=7), c, 3)
    b = samples(ToleranceSettings(part_mm=3, part_deg=1, station_mm=4, samples=20, seed=7), c, 3)
    assert len(a) == 20 and all(
        np.allclose(x[0], y[0]) and np.allclose(x[1], y[1]) for x, y in zip(a, b, strict=True)
    )
    assert np.allclose(a[0][0], np.eye(4)) and not a[0][1].any()  # the first sample is nominal
    for M, e in a[1:]:
        R = M[:3, :3]
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        assert ang <= 1 + 1e-9
        assert np.all(np.abs(M[:3, :3] @ c + M[:3, 3] - c) <= 0.003 + 1e-12)  # centre moves <= 3 mm/axis
        assert np.all(np.abs(e[:, :2]) <= 0.004 + 1e-12) and not e[:, 2].any()


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8854, verbose=False)
    st = build_app(server)
    assert st["project"].load_example("rail_long_part")
    confirm_all(st["stations"])
    st["stations"].start(block=True, quick=True)
    yield st
    from opt.parallel import shutdown_all

    jobs.CHECK_WORKERS = 1
    shutdown_all()
    server.stop()


def _run(app, **kw):
    tp = app["tolerance"]
    for k, v in kw.items():
        getattr(tp, k).value = v
    tp.start(block=True)
    return app["session"].tolerance


def test_zero_tolerance_reaches_every_planned_point(app):
    r = _run(app, part_mm=0.0, part_deg=0.0, station_mm=0.0, samples=3)
    assert r.points and all(p.rate == 1.0 and p.fail_mm is None for p in r.points.values())
    assert r.all_reached == 1.0 and "Every point was reached" in app["tolerance"].status.content


def test_large_offsets_lose_points_and_report_them(app):
    """The rail example has a lot of margin (points stay reachable with the part 0.3 m off), so
    this uses offsets far beyond the GUI's limits, through the job itself."""
    from ui.plan_job import tolerance_job
    from ui.tolerance_ui import summary

    s = app["session"]
    sol = app["results"].selected()
    st = ToleranceSettings(part_mm=800.0, part_deg=10.0, station_mm=300.0, samples=12, seed=3)
    spec = {"sol": sol, "results": s.result_points[sol.label], "settings": st, "workers": 1}
    r = tolerance_job({"session": s, "spec": spec}, lambda k, d: None, lambda: False)
    rates = [p.rate for p in r.points.values()]
    assert min(rates) < 1.0 and r.all_reached < 1.0
    worst = r.worst(5)
    assert worst[0].rate == min(rates) and worst[0].fail_mm is not None and worst[0].fail_mm > 0
    assert "Least robust points" in summary(r)


@pytest.mark.slow
def test_pool_gives_the_same_result(app):
    a = _run(app, part_mm=100.0, part_deg=10.0, station_mm=100.0, samples=8, seed=5)
    jobs.CHECK_WORKERS = 2
    try:
        b = _run(app, part_mm=100.0, part_deg=10.0, station_mm=100.0, samples=8, seed=5)
    finally:
        jobs.CHECK_WORKERS = 1
    assert {k: (p.rate, p.fail_mm) for k, p in a.points.items()} == {
        k: (p.rate, p.fail_mm) for k, p in b.points.items()
    }
