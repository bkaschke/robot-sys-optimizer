"""core.app_log (the log file) and core.support_bundle (the zip sent with a problem report)."""

import io
import logging
import threading
import time
import zipfile

import pytest

from core import app_log
from core import support_bundle as sb


@pytest.fixture
def data(tmp_path, monkeypatch):
    """A private data folder; the log file is closed again afterwards."""
    monkeypatch.setenv("RSO_DATA", str(tmp_path / "data"))
    yield tmp_path / "data"
    app_log.shutdown()


def _flush():
    for h in logging.getLogger().handlers:
        h.flush()


def test_no_file_without_setup(data):
    from ui import log_panel

    log_panel.log("nothing is written yet", "warn")
    assert app_log.log_files() == []


def test_setup_writes_logging_and_panel_messages(data):
    from ui import log_panel

    path = app_log.setup("1.2.3")
    assert path == data / "logs" / "app.log" and app_log.setup() == path  # the second call changes nothing
    logging.getLogger("ui.something").warning("planner said %s", "hello")
    logging.getLogger("websockets.server").info("chatter")
    log_panel.log("Station planning finished: **2 stations**", "done")
    log_panel.log("Export failed", "error")
    _flush()
    text = path.read_text(encoding="utf-8")
    assert "started (version 1.2.3)" in text
    assert "WARNING ui.something: planner said hello" in text
    assert "INFO    rso.panel: Station planning finished: **2 stations**" in text
    assert "ERROR   rso.panel: Export failed" in text
    assert "chatter" not in text
    assert app_log.log_files() == [path]


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_uncaught_thread_error_reaches_the_file(data):
    path = app_log.setup()

    def boom():
        raise RuntimeError("worker thread fell over")

    t = threading.Thread(target=boom, name="demo-thread")
    t.start()
    t.join()
    _flush()
    text = path.read_text(encoding="utf-8")
    assert "uncaught exception in thread demo-thread" in text and "worker thread fell over" in text


def test_setup_survives_an_unwritable_folder(data, monkeypatch):
    def refuse(*a, **k):
        raise OSError("read-only")

    monkeypatch.setattr(app_log, "RotatingFileHandler", refuse)
    assert app_log.setup() is None


def _names(data_bytes):
    with zipfile.ZipFile(io.BytesIO(data_bytes)) as z:
        return z.namelist(), {n: z.read(n).decode("utf-8", "replace") for n in z.namelist()}


def test_bundle_without_anything(data):
    path, blob = sb.make_bundle()
    assert path.parent == data / "support" and path.read_bytes() == blob
    names, text = _names(blob)
    assert set(names) == {"panel_log.txt", "info.txt"}
    assert "Robot Cell Optimizer" in text["info.txt"] and "Packages:" in text["info.txt"]
    assert "no log file yet" in text["info.txt"]
    assert text["panel_log.txt"] == "(no messages)\n"


def test_bundle_collects_log_project_and_batch_errors_but_no_meshes(data, tmp_path):
    app_log.setup()
    logging.getLogger("rso.test").error("something broke")
    _flush()
    batch = tmp_path / "batch_1"
    job = batch / "job_001"
    (job / "assets" / "parts").mkdir(parents=True)
    (job / "assets" / "parts" / "secret_part.stl").write_bytes(b"solid x")
    (job / "project.json").write_text('{"robot": null}', encoding="utf-8")
    (job / "error.txt").write_text("ValueError: no plan", encoding="utf-8")
    (batch / "batch.json").write_text("{}", encoding="utf-8")
    entries = [(time.time(), "warn", "Setup check: part is tiny"), (time.time(), "done", "Saved")]
    _path, blob = sb.make_bundle(job, entries)
    names, text = _names(blob)
    assert {
        "info.txt",
        "panel_log.txt",
        "logs/app.log",
        "project/project.json",
        "batch/batch.json",
        "batch/job_001/error.txt",
    } <= set(names)
    assert not any(n.endswith(".stl") for n in names)
    assert "something broke" in text["logs/app.log"]
    assert "warn  Setup check: part is tiny" in text["panel_log.txt"]
    assert f"Open study: {job}" in text["info.txt"] and "Not included" not in text["info.txt"]


def test_large_files_are_cut_to_their_tail(data, tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "MAX_FILE", 100)
    proj = tmp_path / "p"
    proj.mkdir()
    (proj / "project.json").write_bytes(b"a" * 500 + b"TAIL")
    _path, blob = sb.make_bundle(proj)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        got = z.read("project/project.json")
    assert len(got) == 100 and got.endswith(b"TAIL")


def test_settings_button_saves_a_bundle(data):
    import viser

    from ui import log_panel
    from ui.app_state import build_settings, save_bundle
    from ui.session import Session

    server = viser.ViserServer(host="127.0.0.1", port=8874, verbose=False)
    try:
        build_settings(server, False, None, Session())  # the folder builds with the button
        info = server.gui.add_markdown("")
        log_panel.clear()
        path = save_bundle(None, Session(), info)
        assert path is not None and path.is_file() and str(path) in info.content
        assert any("Support bundle saved" in m for _t, _lv, m in log_panel.entries())
    finally:
        server.stop()
