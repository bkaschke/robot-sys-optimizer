"""A saved and reopened project keeps its confirmed setup steps (-0.0 vs 0.0 in pose hashes)."""

from ui.session import Session, empty_cell, load_project, save_project
from ui.staleness import components, confirm_step


def test_confirmed_cell_survives_save_and_load(tmp_path):
    s = empty_cell()
    confirm_step(s, "Cell")
    save_project(s, tmp_path)
    t = Session()
    load_project(t, tmp_path)
    a, b = components(s), components(t)
    assert [k for k in a if a[k] != b.get(k)] == []
