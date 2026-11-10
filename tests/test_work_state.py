"""Reopen where you left off: joint pose, selection, feasibility and plan joint solutions and
panel settings survive save/load; autosave; the app remembers the last project."""

from dataclasses import dataclass

import numpy as np
import pytest
import viser

from core import schema
from core import targets as tg
from tests.helpers import confirm_all
from ui import session as session_mod


@dataclass
class FakeClick:
    ray_origin: tuple
    ray_direction: tuple
    modifier: str | None = None
    client = None


def _app(port: int):
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=port, verbose=False)
    return server, build_app(server)


@pytest.fixture(scope="module")
def worked(tmp_path_factory):
    """An app with real work done in it, saved to a folder."""
    server, st = _app(8775)
    s, t = st["session"], st["targets"]
    folder = tmp_path_factory.mktemp("job")
    s.project_dir = folder
    s.targets = s.targets.model_copy(update={"points": []})
    t.on_part_click(FakeClick((1.0, 0.0, 2.0), (0.0, 0.0, -1.0)))  # select the plate top
    t.method.value = "Projected grid"
    t.spacing.value = 150.0
    t.run_generate()
    st["optimize"].start(block=True)
    st["stations"].max_stations.value = 1
    confirm_all(st["stations"])
    st["stations"].start(block=True)
    st["cell"].set_q(np.radians([10, -80, 70, -80, -90, 15]))
    # settings to restore
    t.tools.row.value, t.tools.radius.value = 12.0, 33.0
    t.color_by.value = "feasibility"
    st["placement"].half.value = 1.7
    st["dark_toggle"].value = True
    path = st["project"].save_now(folder)
    yield st, folder, path
    server.stop()


def test_saved_file_has_the_work_state(worked):
    st, folder, path = worked
    p = schema.load_project(path)
    w = p.work
    assert len(w.q) == 6 and np.allclose(w.q, np.radians([10, -80, 70, -80, -90, 15]))
    assert w.selection and w.selection_mesh == st["session"].part.mesh_path
    assert len(w.feasibility) == len(p.targets.points) > 0
    assert set(w.result_points) == {sol.label for sol in p.results}
    assert w.ui["targets.tools.row"] == 12.0 and w.ui["app.dark"] is True
    assert not list(folder.glob("*.tmp"))  # atomic write leaves no temp file


def test_reopen_restores_where_left_off(worked):
    st, folder, _ = worked
    s0 = st["session"]
    server, st2 = _app(8776)
    try:
        assert st2["project"]._load(None, folder)
        s, t = st2["session"], st2["targets"]
        assert np.allclose(s.q, s0.q)  # joint pose, and the sliders show it
        assert np.allclose([sl.value for sl in st2["cell"].sliders], np.degrees(s0.q))
        assert np.array_equal(t.selection, st["targets"].selection)
        assert {k: r.status for k, r in s.feasibility.items()} == {
            k: r.status for k, r in s0.feasibility.items()
        }
        label = s0.results[0].label
        for pid, r in s0.result_points[label].items():
            r2 = s.result_points[label][pid]
            assert r2.status == r.status and (r.q is None or np.allclose(r2.q, r.q))
        # settings
        assert t.tools.row.value == 12.0 and t.tools.radius.value == 33.0
        assert t.color_by.value == "feasibility" and st2["placement"].half.value == 1.7
        assert st2["dark_toggle"].value is True
        # results come back with ghosts, and the CSV export works without re-planning
        res = st2["results"]
        assert sum(1 for h in res._ghosts if hasattr(h, "update_cfg")) == len(s.results[0].stations)
        from core import export as ex

        rows = ex.tcp_rows(
            s.results[0], s.result_points[label], s.targets, s.robot, s.tool_model, s.base, s.T_world_base
        )
        assert sum(r["status"] != "unreachable" for r in rows) == len(s.results[0].assignment)
    finally:
        server.stop()


def test_autosave(worked, tmp_path):
    server, st = _app(8777)
    try:
        s, proj = st["session"], st["project"]
        s.project_dir = tmp_path
        assert proj.autosave_tick(force=True)  # first time: writes
        assert (tmp_path / "project.json").is_file()
        assert not proj.autosave_tick(force=True)  # nothing changed: no write
        st["cell"].set_q(np.radians([5, -90, 90, -90, -90, 0]))
        assert not proj.autosave_tick()  # changed, but the interval has not passed
        assert proj.autosave_tick(force=True)
        assert np.allclose(schema.load_project(tmp_path / "project.json").work.q, s.q)
        proj.autosave.value = False
        st["cell"].set_q(np.radians([0, -90, 90, -90, -90, 0]))
        assert not proj.autosave_tick(force=True)  # switched off
        assert "Autosaved" in proj.status.content or "Saved" in proj.status.content
    finally:
        server.stop()


def test_remember_last_project(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "LAST_PROJECT", tmp_path / "last_project.txt")
    assert session_mod.last_project() is None
    job = tmp_path / "job"
    session_mod.remember_last(job)
    assert session_mod.last_project() is None  # no project.json there yet
    job.mkdir()
    (job / "project.json").write_text("{}", encoding="utf-8")
    assert session_mod.last_project() == job.resolve()


def test_panel_records_last_project_only_when_asked(tmp_path, monkeypatch):
    from ui.project_panel import ProjectPanel

    monkeypatch.setattr(session_mod, "LAST_PROJECT", tmp_path / "last.txt")
    monkeypatch.setattr("ui.project_panel.remember_last", session_mod.remember_last)
    server, st = _app(8778)
    try:
        st["project"].save_now(tmp_path / "a")
        assert not (tmp_path / "last.txt").exists()  # tests / scripts: off
        p = ProjectPanel(server, st["session"], on_loaded=lambda: None, remember=True)
        p.save_now(tmp_path / "b")
        assert session_mod.last_project() == (tmp_path / "b").resolve()
    finally:
        server.stop()


def test_old_project_without_work_still_loads():
    from pathlib import Path

    p = schema.load_project(Path(__file__).resolve().parents[1] / "examples" / "sander_mobile.json")
    assert p.work == schema.WorkState()


def test_settings_apply_skips_bad_values():
    from ui import settings

    server, st = _app(8779)
    try:
        t = st["targets"]
        done = settings.apply(
            st,
            {
                "targets.method": "no such method",  # not an option
                "targets.spacing": "ten",  # wrong type
                "targets.direction": [1.0, 2.0],  # wrong length
                "targets.tools.col": 40.0,  # fine
                "unknown.key": 1,
            },
        )
        assert done == ["targets.tools.col"] and t.tools.col.value == 40.0
        assert t.method.value != "no such method"
    finally:
        server.stop()


def test_selection_not_restored_on_a_different_mesh(worked):
    st, folder, _ = worked
    server, st2 = _app(8780)
    try:
        s = st2["session"]
        st2["project"]._load(None, folder)
        assert len(st2["targets"].selection) > 0  # restored on the same mesh
        s.selection = ("assets/parts/other.stl", [0, 1, 2])  # saved on a different mesh
        st2["targets"].selection = np.zeros(0, int)
        st2["project"].on_loaded()  # runs the restore step again
        assert len(st2["targets"].selection) == 0  # not applied to the wrong mesh
        s.selection = (s.part.mesh_path, [10**9])  # index past the mesh: ignored, no crash
        st2["project"].on_loaded()
        assert len(st2["targets"].selection) == 0
        assert tg.as_arrays(s.targets)[1].shape[1] == 3
    finally:
        server.stop()
