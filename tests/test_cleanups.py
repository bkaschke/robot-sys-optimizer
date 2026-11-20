"""B6 clean-ups: background display deviation, Station-planner hook, readout throttle, app size."""

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import trimesh

from ui import display_lod


def test_deviation_measured_in_background_with_fallback(monkeypatch):
    big = trimesh.creation.icosphere(subdivisions=5, radius=0.2)
    monkeypatch.setattr(display_lod, "_DEVIATION", display_lod.OrderedDict())
    monkeypatch.setattr(display_lod, "_FALLBACK", display_lod.OrderedDict())
    gate, fired = threading.Event(), []
    monkeypatch.setattr(display_lod, "_measure", lambda full, low: (gate.wait(5), 0.0042)[1])
    monkeypatch.setattr(display_lod, "on_deviation_ready", [lambda: fired.append(1)])
    t0 = time.perf_counter()
    _, simp = display_lod.display_copy(big, limit=3000, target=1000, background=True)
    assert simp and time.perf_counter() - t0 < 4  # did not wait for the (blocked) measurement
    fb = display_lod.display_deviation(big.copy())
    assert fb == pytest.approx(0.005 * 0.4 * 3**0.5) and fb > 0  # conservative fallback while pending
    assert display_lod.overlay_lift(big, 5e-4) == 5e-4 + fb
    gate.set()
    display_lod.wait_deviation()
    assert display_lod.display_deviation(big.copy()) == 0.0042
    assert fired == [1]  # 0.0042 differs materially from the fallback (0.0035)


def test_no_redraw_when_lift_barely_changes(monkeypatch):
    big = trimesh.creation.icosphere(subdivisions=5, radius=0.2)
    monkeypatch.setattr(display_lod, "_DEVIATION", display_lod.OrderedDict())
    fb = display_lod.fallback_deviation(big)
    monkeypatch.setattr(display_lod, "_measure", lambda full, low: fb + 1e-5)
    fired = []
    monkeypatch.setattr(display_lod, "on_deviation_ready", [lambda: fired.append(1)])
    display_lod.display_copy(big, limit=3000, target=1000, background=True)
    display_lod.wait_deviation()
    assert fired == []


def test_stations_blocked_reason_hook():
    from ui.stations import StationsPanel

    seen = []
    p = SimpleNamespace(blocked_reason=lambda: "nope", _thread=None, s=None)
    import ui.stations as st

    monkey = st.notify
    st.notify = lambda e, title, msg, **k: seen.append((title, msg))
    try:
        StationsPanel.start(p)
    finally:
        st.notify = monkey
    assert seen == [("Cannot plan now", "nope")]


def test_readout_throttled_with_trailing_update():
    from ui.cell import CellTab

    cell = CellTab.__new__(CellTab)
    cell.readout = SimpleNamespace(content="")
    writes = []
    cell.readout = type("R", (), {"content": property(lambda s: "", lambda s, v: writes.append(v))})()
    for i in range(50):
        cell._put_readout(f"v{i}")
    assert len(writes) == 1
    time.sleep(CellTab.READOUT_MIN_DT + 0.3)
    assert writes[-1] == "v49" and len(writes) == 2


def test_app_py_does_not_grow():
    root = Path(__file__).resolve().parent.parent
    # upstream v13 app.py has 588 lines (over the 500-line rule); ours must not grow past that
    assert len((root / "app.py").read_text(encoding="utf-8").splitlines()) <= 588
