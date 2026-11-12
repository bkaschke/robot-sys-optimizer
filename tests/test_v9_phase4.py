"""V9 phase 4 (issues #6, #7, #8): compared robots viewable in Results with their plans and
paths (the project's robot kept until a confirmed switch), saved with the project; the report
gains the comparison and drops the per-point table."""

from types import SimpleNamespace

import pytest
import viser

from tests.helpers import confirm_all

E = SimpleNamespace(client=None)


@pytest.fixture(scope="module")
def compared():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8847, verbose=False)
    st = build_app(server)
    assert st["project"].load_example("rail_long_part")
    stations, cmp_ = st["stations"], st["compare"]
    for name, cb in cmp_.pick.items():
        cb.value = name == "ur10e"
    stations.refine.value = "none"  # keep it short
    confirm_all(stations)
    cmp_.start(block=True)
    yield st
    st["builder_window"].stop()
    server.stop()


def test_compared_plans_are_kept(compared):
    s = compared["session"]
    assert set(s.compared) == {"ur10e"}
    c = s.compared["ur10e"]
    assert c["results"] and set(c["points"]) == {sol.label for sol in c["results"]}
    assert c["metrics"]["label"] == c["results"][0].label  # the best plan first, as in Results
    assert all("robot" in st for st in c["stamps"].values())
    assert "plans" not in compared["compare"].rows[0]  # the table rows stay small
    assert compared["compare_view"].folder.visible


def test_view_a_compared_robot_and_build_its_paths(compared):
    s, view, sim, results = (
        compared["session"],
        compared["compare_view"],
        compared["sim"],
        compared["results"],
    )
    own_results = list(s.results)
    view.view("ur10e")
    assert s.robot_name == "ur10e" and s.view_stash["robot_name"] == "ur20"
    assert [r.label for r in s.results] == [r.label for r in s.compared["ur10e"]["results"]]
    assert compared["cell"].robot_choice.value == "ur10e"
    from ui.staleness import solution_changes

    assert solution_changes(s, s.results[0]) == []  # up to date with the robot it was planned with
    p = s.snapshot()  # saving while viewing saves the project's own robot and plans
    assert p.robot.name == "ur20" and [r.label for r in p.results] == [r.label for r in own_results]
    assert [c.robot for c in p.compared] == ["ur10e"]
    results.choice.value = results.choice.options[0]
    sim.build(block=True)
    assert sim.plan is not None and sim.plan.samples
    view.view(None)  # back
    assert s.robot_name == "ur20" and s.view_stash is None
    assert [r.label for r in s.results] == [r.label for r in own_results]


def test_use_a_compared_robot(compared):
    s, view = compared["session"], compared["compare_view"]
    view.view("ur10e")
    view.confirm_use(E)  # no browser client: no dialog
    assert s.robot_name == "ur10e" and s.view_stash is None
    assert s.snapshot().robot.name == "ur10e"
    assert s.results and s.results[0].label in s.compared["ur10e"]["points"]


def test_compared_robots_are_saved(compared, tmp_path):
    s, project = compared["session"], compared["project"]
    project.save_now(tmp_path / "study")
    assert project._load(E, tmp_path / "study")
    assert set(s.compared) == {"ur10e"} and s.compared["ur10e"]["results"]
    pts = s.compared["ur10e"]["points"]
    label = s.compared["ur10e"]["results"][0].label
    assert pts[label] and next(iter(pts[label].values())).q is not None


def test_report_has_the_comparison_and_no_point_table(compared):
    s = compared["session"]
    ex = compared["results"].export
    sol = compared["results"].selected() or s.results[0]
    names = [h for h, _ in ex.sections(sol)]
    assert "Robot comparison" in names
    path, data = ex.export_report(sol)
    html = data.decode("utf-8") if isinstance(data, bytes) else path.read_text(encoding="utf-8")
    assert "Robot comparison" in html and "| ur10e |" not in html and "ur10e" in html
    assert "<h2>Points</h2>" not in html and "point_id" not in html
    assert "break-inside:avoid" in html  # laid out for printing
