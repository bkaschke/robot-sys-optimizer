"""v8 T1: planners in a worker process (ui.jobs, ui.plan_job)."""

import time

import pytest
import viser

from tests.helpers import confirm_all
from ui import jobs


def stubborn(payload, emit, cancelled):
    """A job that ignores Cancel (the runner must end its process)."""
    emit("progress", ("busy", 0.5))
    while True:
        time.sleep(0.05)


def polite(payload, emit, cancelled):
    """A job that stops on Cancel, emitting progress meanwhile."""
    for i in range(400):
        if cancelled():
            return {"stopped_at": i}
        emit("progress", (f"step {i}", i / 400))
        time.sleep(0.01)
    return {"stopped_at": None, "echo": payload}


def broken(payload, emit, cancelled):
    raise ValueError("bad input 42")


def test_worker_returns_results_and_messages():
    r = jobs.Runner()
    try:
        msgs = []
        out = r.run("tests.test_v8_jobs:polite", {"x": 1}, lambda k, d: msgs.append(k))
        assert out == {"stopped_at": None, "echo": {"x": 1}}
        assert msgs.count("progress") == 400
        with pytest.raises(RuntimeError, match="bad input 42"):  # errors come back, worker lives on
            r.run("tests.test_v8_jobs:broken", None, lambda k, d: None)
        pid = r.pid
        assert r.run("tests.test_v8_jobs:polite", 5, lambda k, d: None)["echo"] == 5
        assert r.pid == pid  # the same worker serves the next job
    finally:
        r.stop()


def test_cancel_stops_a_polite_job_and_kills_a_stubborn_one():
    r = jobs.Runner()
    try:
        t0 = time.monotonic()
        out = r.run("tests.test_v8_jobs:polite", None, lambda k, d: None, lambda: time.monotonic() - t0 > 0.5)
        assert out["stopped_at"] is not None  # stopped through the shared event
        proc = r._proc
        t0 = time.monotonic()
        with pytest.raises(jobs.JobCancelledError):
            r.run("tests.test_v8_jobs:stubborn", None, lambda k, d: None, lambda: time.monotonic() - t0 > 0.3)
        assert time.monotonic() - t0 < 0.3 + jobs.CANCEL_GRACE_S + 3
        assert r.pid is None and not proc.is_alive() and proc.exitcode is not None  # gone: memory freed
        assert r.run("tests.test_v8_jobs:polite", 1, lambda k, d: None)["echo"] == 1  # a fresh worker
    finally:
        r.stop()


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8851, verbose=False)
    st = build_app(server)
    yield st
    jobs.USE_WORKER = False
    jobs.runner().stop()
    server.stop()


def _plans(s):
    return [(r.label, round(r.cost, 6), [sorted(st.carrier.items()) for st in r.stations]) for r in s.results]


@pytest.mark.slow
def test_station_plan_in_worker_equals_in_process(app):
    st, s = app["stations"], app["session"]
    assert app["project"].load_example("rail_long_part")
    confirm_all(st)
    st.start(block=True, quick=True)  # in-process (block)
    ref = _plans(s)
    assert ref and all(label.startswith("Quick: ") for label, *_ in ref)
    jobs.USE_WORKER = True
    try:
        t0 = time.monotonic()
        st.start(quick=True)
        assert time.monotonic() - t0 < 1.0  # the GUI callback returns at once
        st.max_stations.value = st.max_stations.value  # GUI updates keep working meanwhile
        st._thread.join(300)
        assert not st._thread.is_alive()
        assert _plans(s) == ref
        assert jobs.runner().pid is not None
        assert s.max_manip > 0 and "Best:" in st.quick_status.content
    finally:
        jobs.USE_WORKER = False


@pytest.mark.slow
def test_tool_optimizer_in_worker(app):
    tool, s = app["tool_opt"], app["session"]
    assert app["project"].load_example("rail_long_part")
    tool.shape_evals.value = 0
    confirm_all(tool)
    jobs.USE_WORKER = True
    try:
        tool.start()
        tool._thread.join(600)
        assert tool.result is not None and tool.result.samples
        assert any(r.planner == "tool" for r in s.results)
    finally:
        jobs.USE_WORKER = False
