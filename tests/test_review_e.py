"""Review v8 package E: results kept across a failed / cancelled / empty plan (H1 / R9), cancel
reported as a cancel (R4), Pareto / robust labels on cancel (R7), compare settings (R12), the
worker's session cache key (L5) and one-line errors in the status line (L6).

The panels are driven with a stubbed job runner (``ui.jobs.call`` / ``station_job``), so no plan
actually runs."""

import os

import pytest
import viser

from core.feasibility import CancelledError
from core.schema import ExtensionValues, Solution, Station
from tests.helpers import confirm_all
from ui import jobs, log_panel

TB = "Traceback (most recent call last):\n  File 'x.py', line 1\nValueError: boom"


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8931, verbose=False)
    state = build_app(server)
    assert state["project"].load_example("rail_long_part")
    confirm_all(state["stations"])
    yield state
    server.stop()


def sol(label: str, k: int, cost: float = 1.0, tool: bool = False) -> Solution:
    return Solution(
        label=label,
        stations=[Station(xyz=(float(i), 0.0, 0.0)) for i in range(k)],
        cost=cost,
        tool_values=ExtensionValues() if tool else None,
    )


def seed(s) -> None:
    """An earlier station plan and a tool-optimizer design, with joint solutions and stamps."""
    from ui.staleness import stamp_for

    old, tool = sol("2 stations: old", 2, 3.0), sol("Tool design #1", 2, 2.0, tool=True)
    s.results = [tool, old]
    s.result_points = {old.label: {}, tool.label: {}}
    s.stamps[old.label] = stamp_for(s, "plan")
    s.stamps[tool.label] = stamp_for(s, "tool")


def labels(s) -> set:
    return {r.label for r in s.results}


def errors_since(n: int) -> list:
    return [e for e in log_panel.entries()[n:] if e[1] == "error"]


def run_stations(app, monkeypatch, fake):
    st, s = app["stations"], app["session"]
    seed(s)
    seen = {}

    def call(target, payload, on_msg, cancelled, in_process):
        seen["during"] = labels(s)  # what autosave would see while the job runs
        return fake(on_msg)

    monkeypatch.setattr(jobs, "call", call)
    n = len(log_panel.entries())
    st.start(block=True)
    return st, s, seen, n


# --- H1 / R9: results survive until the first new plan ------------------------------------------


def test_failed_run_keeps_previous_results_and_shows_one_line(app, monkeypatch):
    def fake(on_msg):
        raise jobs.WorkerError("ValueError", "boom", TB)

    st, s, seen, n = run_stations(app, monkeypatch, fake)
    assert seen["during"] == {"2 stations: old", "Tool design #1"}  # nothing cleared mid-run
    assert labels(s) == {"2 stations: old", "Tool design #1"}
    assert set(s.result_points) >= {"2 stations: old", "Tool design #1"}
    assert "previous results kept" in st.status.content
    # L6: the status line has the summary only; the traceback goes to the log
    assert "ValueError: boom" in st.status.content
    assert "worker traceback" not in st.status.content and "Traceback" not in st.status.content
    assert any("Traceback" in e[2] for e in log_panel.entries()[n:])


def test_cancel_before_any_plan_keeps_results_and_logs_no_error(app, monkeypatch):
    def fake(on_msg):
        on_msg("progress", ("loading reachability map", 0.0))
        raise CancelledError("cancelled")

    st, s, seen, n = run_stations(app, monkeypatch, fake)
    assert labels(s) == {"2 stations: old", "Tool design #1"}
    assert "Cancelled" in st.status.content and "previous results kept" in st.status.content
    assert "reachability map" not in st.status.content  # not the only place a cancel happens now
    assert not errors_since(n)


def test_run_that_finds_nothing_keeps_results(app, monkeypatch):
    st, s, seen, n = run_stations(app, monkeypatch, lambda on_msg: {"best": None, "rules_status": None})
    assert labels(s) == {"2 stations: old", "Tool design #1"}
    assert "previous results kept" in st.status.content


def test_first_new_plan_replaces_station_plans_but_keeps_tool_designs(app, monkeypatch):
    new = sol("3 stations: new", 3, 1.0)

    def fake(on_msg):
        on_msg("solution", (new, {}))
        return {"best": new, "rules_status": None}

    st, s, seen, n = run_stations(app, monkeypatch, fake)
    assert labels(s) == {"3 stations: new", "Tool design #1"}
    assert "2 stations: old" not in s.result_points and "2 stations: old" not in s.stamps
    assert "Tool design #1" in s.result_points and "Tool design #1" in s.stamps
    assert next(r for r in s.results if r.label == "Tool design #1").run_s is None  # not this run's


def test_replace_message_keeps_tool_designs(app, monkeypatch):
    front = [(sol("Pareto 1/2: a", 2, 1.0), {}), (sol("Pareto 2/2: b", 3, 1.5), {})]

    def fake(on_msg):
        on_msg("replace", front)
        return {"best": front[0][0], "rules_status": None}

    st, s, seen, n = run_stations(app, monkeypatch, fake)
    assert labels(s) == {"Pareto 1/2: a", "Pareto 2/2: b", "Tool design #1"}
    assert "Tool design #1" in s.result_points and "2 stations: old" not in s.stamps


# --- R4 / R12: compare keeps finished rows on cancel and uses the planner's settings ------------


def test_compare_keeps_finished_rows_on_cancel_and_passes_collision_settings(app, monkeypatch):
    from core import collision_config
    from ui import plan_job

    cmp_, st = app["compare"], app["stations"]
    for name, cb in cmp_.pick.items():
        cb.value = name in ("ur20", "ur10e")
    st.cable.wrap.value = 270.0
    old = collision_config.current()
    collision_config.update(dress_m=0.023)
    specs, dress = [], []

    def fake_station_job(payload, emit, cancelled):
        specs.append(payload["spec"])
        dress.append(collision_config.current().dress_m)  # v12: one config value per job
        if len(specs) == 2:
            raise CancelledError("cancelled")  # e.g. while the second robot's map is built
        best = sol("2 stations: ur20", 2, 4.0)
        emit("solution", (best, {}))
        return {"best": best, "rules_status": None}

    monkeypatch.setattr(plan_job, "station_job", fake_station_job)
    n = len(log_panel.entries())
    try:
        cmp_.start(block=True)
    finally:
        collision_config.use(old)
    # v10 T17: the project's own setup (ur20) is planned first; the second setup was cancelled
    assert [r["robot"] for r in cmp_.rows] == ["ur20"], cmp_.status.content
    assert "Cancelled" in cmp_.status.content and "| ur20 (project) |" in cmp_.status.content
    assert not errors_since(n)
    sub = specs[0]
    assert dress[0] == pytest.approx(0.023)
    assert sub["settings"].wrap_limit_deg == pytest.approx(270.0)


def test_compare_failure_shows_one_line(app, monkeypatch):
    from ui import plan_job

    cmp_ = app["compare"]

    def boom(payload, emit, cancelled):
        raise jobs.WorkerError("ValueError", "boom", TB)

    monkeypatch.setattr(plan_job, "station_job", boom)
    cmp_.start(block=True)
    assert "ValueError: boom" in cmp_.status.content and "Traceback" not in cmp_.status.content


# --- R4 / L6: tool optimizer and tolerance check ------------------------------------------------


def test_tool_optimizer_cancel_and_error_messages(app, monkeypatch):
    panel = app["tool_opt"]

    def cancel(target, payload, on_msg, cancelled, in_process):
        raise CancelledError("cancelled")

    monkeypatch.setattr(jobs, "call", cancel)
    n = len(log_panel.entries())
    panel.start(block=True)
    assert "Cancelled" in panel.info.content and "reachability map" not in panel.info.content
    assert not errors_since(n)

    def fail(target, payload, on_msg, cancelled, in_process):
        on_msg("warn", "parallel checks off for this run")
        raise jobs.WorkerError("ValueError", "boom", TB)

    monkeypatch.setattr(jobs, "call", fail)
    n = len(log_panel.entries())
    panel.start(block=True)
    assert "ValueError: boom" in panel.info.content and "Traceback" not in panel.info.content
    new = log_panel.entries()[n:]
    assert any(e[1] == "warn" and "parallel checks off" in e[2] for e in new)


def test_tolerance_error_shows_one_line(app, monkeypatch):
    panel, s = app["tolerance"], app["session"]
    plan = sol("2 stations: tol", 2)
    s.result_points[plan.label] = {1: object()}
    monkeypatch.setattr(panel, "selected", lambda: plan)

    def fail(target, payload, on_msg, cancelled, in_process):
        raise jobs.WorkerError("ValueError", "boom", TB)

    monkeypatch.setattr(jobs, "call", fail)
    panel.start(block=True)
    assert "ValueError: boom" in panel.status.content and "Traceback" not in panel.status.content


# --- R7: Pareto / robust step cancelled ----------------------------------------------------------


def _stub_planner(monkeypatch, plans):
    import opt.placement
    import opt.reachability
    from ui import plan_job

    class Map:
        max_manip = 1.0

    monkeypatch.setattr(opt.reachability, "map_for_tcps", lambda *a, **k: Map())
    monkeypatch.setattr(plan_job, "build_checker", lambda *a, **k: type("C", (), {"pool": None})())

    def plan_stations(*a, finisher=None, **k):
        finisher.append((None, None))
        return plans

    monkeypatch.setattr(opt.placement, "plan_stations", plan_stations)


def _station_spec(objective: str) -> dict:
    from opt.cost import Weights
    from opt.placement import PlannerSettings

    return {"settings": PlannerSettings(), "weights": Weights(), "rules": None, "objective": objective}


def test_cancelled_pareto_step_keeps_plain_labels(app, monkeypatch):
    import opt.pareto
    from ui.plan_job import station_job

    a, b = sol("2 stations: a", 2, 1.0), sol("3 stations: b", 3, 2.0)
    a.planner = b.planner = "cost"
    _stub_planner(monkeypatch, [(a, {}), (b, {})])

    def front(*args):
        raise CancelledError()

    monkeypatch.setattr(opt.pareto, "station_front", front)
    out = []
    spec = {**_station_spec("pareto"), "pareto_keys": ("stations", "cycle")}
    station_job({"session": app["session"], "spec": spec}, lambda k, d: out.append((k, d)), lambda: True)
    plans = next(d for k, d in out if k == "replace")
    assert [p.label for p, _ in plans] == [
        "2 stations: a (cancelled before the Pareto step)",
        "3 stations: b (cancelled before the Pareto step)",
    ]
    assert all(p.planner == "cost" for p, _ in plans)


def test_cancelled_robust_step_keeps_labels_and_drops_partial_scores(app, monkeypatch):
    import opt.robust
    from ui.plan_job import station_job

    a, b = sol("2 stations: a", 2, 1.0), sol("3 stations: b", 3, 2.0)
    a.breakdown, b.breakdown = {"unreached": 0.0}, {"unreached": 0.0}
    _stub_planner(monkeypatch, [(a, {}), (b, {})])

    def robust(finish, used, checker, t, plans, *rest):
        sol0 = plans[0][0]
        sol0.breakdown = {**sol0.breakdown, "robust": 5.0}  # scored before the cancel
        raise CancelledError()

    monkeypatch.setattr(opt.robust, "station_robust", robust)
    out = []
    station_job(
        {"session": app["session"], "spec": _station_spec("robust")},
        lambda k, d: out.append((k, d)),
        lambda: True,
    )
    plans = next(d for k, d in out if k == "replace")
    assert [p.label for p, _ in plans] == [
        "2 stations: a (cancelled before the robust step)",
        "3 stations: b (cancelled before the robust step)",
    ]
    assert all("robust" not in p.breakdown for p, _ in plans)
    assert all(p.planner != "robust" for p, _ in plans)


# --- L5: the worker's session cache sees replaced tool files ------------------------------------


def test_session_cache_rebuilds_when_tool_files_change(app):
    from ui import plan_job

    s = app["session"]
    payload = plan_job.payload_for(s, {}, live=False)
    s1 = plan_job.session_from(payload)
    assert plan_job.session_from(payload) is s1  # unchanged: reused
    urdf = s.resolve(s.tool.urdf)
    mesh = urdf.parent / "meshes" / "pad_link.stl"
    for f in (urdf, mesh):
        st = os.stat(f)
        try:
            os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))
            s2 = plan_job.session_from(payload)
            assert s2 is not s1, f"{f.name} changed but the cached session was reused"
            s1 = s2
        finally:
            os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    plan_job._CACHE.clear()


def test_parallel_warn_hook_is_reset_after_a_job(app, monkeypatch):
    import opt.parallel
    from ui import plan_job

    def body(payload, emit, cancelled):
        opt.parallel.on_warn = lambda m: emit("warn", m)
        raise RuntimeError("x")

    monkeypatch.setattr(plan_job, "_station_job", body, raising=False)
    with pytest.raises(RuntimeError):
        plan_job.station_job({"session": app["session"], "spec": {}}, lambda k, d: None, lambda: False)
    assert opt.parallel.on_warn is None
