"""Review fixes F: 'Open project.json' never overwrites the current project (R3), and
autosave / Save cannot run inside the load window (R10)."""

import json
import threading

import pytest
import viser

from app import build_app


@pytest.fixture
def app(tmp_path):
    server = viser.ViserServer(host="127.0.0.1", port=8931, verbose=False)
    st = build_app(server)
    st["session"].project_dir = tmp_path / "current"
    st["project"].save_now(tmp_path / "current")
    yield st, tmp_path
    server.stop()


def test_upload_opens_in_new_folder_and_keeps_current_file(app):
    st, tmp = app
    proj, s = st["project"], st["session"]
    cur = tmp / "current" / "project.json"
    other = json.loads(cur.read_text(encoding="utf-8"))
    other["end_effector"] = None  # a different project from the one open
    data = json.dumps(other).encode()
    before = cur.read_bytes()
    assert proj.open_upload(data, "my_cell.json")
    assert cur.read_bytes() == before  # byte-identical
    assert s.project_dir != tmp / "current"
    assert s.project_dir.name.startswith("my_cell_") and s.project_dir.parent == proj_parent()
    assert (s.project_dir / "project.json").read_bytes() == data
    assert proj.folder.value.endswith(s.project_dir.name)


def proj_parent():
    from ui.session import WORKSPACE

    return WORKSPACE


def test_unloadable_upload_leaves_current_project(app):
    st, tmp = app
    proj, s = st["project"], st["session"]
    cur = tmp / "current" / "project.json"
    d = json.loads(cur.read_text(encoding="utf-8"))
    d["robot"]["name"] = "no_such_robot"  # schema-valid, cannot be loaded (v12 opens a missing part)
    before = cur.read_bytes()
    assert not proj.open_upload(json.dumps(d).encode(), "broken.json")
    assert cur.read_bytes() == before
    assert s.project_dir == tmp / "current"
    assert not list(proj_parent().glob("broken_*"))  # the new folder was removed
    assert not proj.loading


def test_autosave_tick_inside_load_window_is_skipped(app, monkeypatch):
    st, tmp = app
    proj, s = st["project"], st["session"]
    other = tmp / "other"
    st["cell"].set_q(s.q * 0 + 0.3)
    proj.save_now(other)
    proj.save_now(tmp / "current")
    st["placement"].half.value = 1.9  # old project's widget value, differs from the saved one
    written = other / "project.json"
    before = written.read_bytes()
    ticks = []
    real = st["cell"].refresh

    def tick_in_load():
        real()
        proj._last_save = -1e9  # interval elapsed
        ticks.append((proj.loading, proj.autosave_tick()))
        errs = []

        def save_from_another_thread():  # the manual Save button is a different thread
            try:
                proj.save_now()
            except ValueError as err:
                errs.append(err)

        t = threading.Thread(target=save_from_another_thread)
        t.start()
        t.join()
        assert len(errs) == 1

    monkeypatch.setattr(st["cell"], "refresh", tick_in_load)
    assert proj._load(None, other)
    assert ticks == [(True, False)]
    assert written.read_bytes() == before  # the new project's file was not rewritten
    assert not proj.loading


def test_autosave_tick_inside_new_cell_is_skipped(app, monkeypatch):
    st, tmp = app
    proj = st["project"]
    ticks = []
    real = st["cell"].refresh

    def tick_in_new_cell():
        real()
        proj._last_save = -1e9
        ticks.append((proj.loading, proj.autosave_tick()))

    monkeypatch.setattr(st["cell"], "refresh", tick_in_new_cell)
    dest = proj.new_cell()
    assert ticks == [(True, False)]
    assert (dest / "project.json").is_file()  # new_cell's own save still worked
    assert not proj.loading
