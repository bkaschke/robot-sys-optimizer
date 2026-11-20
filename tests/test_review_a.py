"""Review fixes for the planner worker transport (ui.jobs): H2, M2, R4, L6.

The jobs below run in the spawned worker, which imports them as ``tests.test_review_a:<name>``.
"""

import os
import threading

import pytest

from core.feasibility import CancelledError
from ui import jobs

H = "tests.test_review_a"


def unpicklable_result(payload, emit, cancelled):
    return {"lock": threading.Lock()}  # cannot be pickled back to the GUI


def unpicklable_message(payload, emit, cancelled):
    emit("solution", threading.Lock())  # a streamed message that cannot be pickled
    emit("progress", ("after", 1.0))
    return {"ok": True}


def crash(payload, emit, cancelled):
    os._exit(3)  # stands in for a native crash / out-of-memory kill


def raise_cancel(payload, emit, cancelled):
    raise CancelledError("stopped while building the map")


def broken(payload, emit, cancelled):
    raise ValueError("bad input 42")


def echo(payload, emit, cancelled):
    return payload


@pytest.fixture(scope="module")
def runner():
    r = jobs.Runner()
    yield r
    r.stop()


@pytest.mark.timeout(60)
def test_unpicklable_result_is_an_error_not_a_hang(runner):
    with pytest.raises(RuntimeError, match="could not be sent back") as info:
        runner.run(f"{H}:unpicklable_result", None, lambda k, d: None)
    assert "TypeError" in str(info.value).splitlines()[0]
    assert runner.run(f"{H}:echo", 7, lambda k, d: None) == 7  # the worker still serves jobs


@pytest.mark.timeout(60)
def test_unpicklable_message_becomes_a_warning(runner):
    msgs = []
    out = runner.run(f"{H}:unpicklable_message", None, lambda k, d: msgs.append((k, d)))
    assert out == {"ok": True}
    kinds = [k for k, _ in msgs]
    assert kinds == ["warn", "progress"]  # not lost silently, and the job carried on
    assert "solution" in msgs[0][1] and "could not be sent" in msgs[0][1]


@pytest.mark.timeout(60)
def test_unpicklable_payload_raises_in_the_parent(runner):
    with pytest.raises(TypeError, match="payload"):
        runner.run(f"{H}:echo", threading.Lock(), lambda k, d: None)
    assert runner.run(f"{H}:echo", 3, lambda k, d: None) == 3


@pytest.mark.timeout(60)
def test_worker_crash_reports_exit_code(runner):
    with pytest.raises(RuntimeError, match="exit code 3") as info:
        runner.run(f"{H}:crash", None, lambda k, d: None)
    assert "coarser grid" in str(info.value)
    assert runner.run(f"{H}:echo", 1, lambda k, d: None) == 1  # a fresh worker


def test_exit_code_causes():
    assert "access violation" in jobs.exit_cause(3221225477)
    assert "access violation" in jobs.exit_cause(0xC0000005 - (1 << 32))  # signed form
    assert "stack buffer overrun" in jobs.exit_cause(0xC0000409)
    assert "out of memory" in jobs.exit_cause(0xC0000017)
    assert "out of memory" in jobs.exit_cause(-9)
    assert jobs.exit_cause(3) == ""


@pytest.mark.timeout(60)
def test_cancelled_error_crosses_the_process(runner):
    with pytest.raises(CancelledError, match="building the map"):
        runner.run(f"{H}:raise_cancel", None, lambda k, d: None)


@pytest.mark.timeout(60)
def test_error_keeps_traceback_and_has_a_one_line_summary(runner):
    with pytest.raises(RuntimeError) as info:
        runner.run(f"{H}:broken", None, lambda k, d: None)
    err = info.value
    assert str(err).splitlines()[0] == "ValueError: bad input 42"
    assert "Traceback" in err.worker_traceback and "broken" in err.worker_traceback
