"""core.preflight + ui.preflight_run / preflight_ui: setup mistakes found before a plan."""

import numpy as np
import pytest
import trimesh

from core import golden_parts as gp
from core import preflight as pf
from core import targets as tg
from core.schema import TCP, EndEffector, TargetGroup, Targets
from core.transforms import make_T


def _targets(P, N, group="g", tcps=()):
    t = Targets(groups=[TargetGroup(name=group, tcps=list(tcps))])
    return tg.append_points(t, np.asarray(P, float), np.asarray(N, float), group)


def _plate_top(n=5):
    """Points on the top face of the flat plate (1 x 0.6 x 0.02 m, top at its max z)."""
    m = gp.flat_plate()
    z = float(m.bounds[1, 2])
    c = m.bounds.mean(axis=0)
    xs = np.linspace(-0.3, 0.3, n)
    P = np.column_stack([c[0] + xs, np.full(n, c[1]), np.full(n, z)])
    return m, P, np.tile([0.0, 0.0, 1.0], (n, 1))


def _codes(found):
    return [f.code for f in found]


def test_clean_setup_has_no_findings():
    m, P, N = _plate_top()
    t = _targets(P, N)
    assert pf.check_part(m, make_T(p=(1.0, 0.0, 0.5))) == []
    assert pf.check_targets(m, t) == []


@pytest.mark.parametrize("scale, code", [(1e-3, "part_tiny"), (1e3, "part_huge")])
def test_unit_mistake_is_flagged(scale, code):
    m = gp.flat_plate()
    m.apply_scale(scale)
    found = pf.check_part(m, make_T(p=(0.0, 0.0, 600.0)))
    assert _codes(found) == [code] and found[0].level == "warn"


def test_part_below_floor():
    m = gp.flat_plate()
    found = pf.check_part(m, make_T(p=(0.0, 0.0, -0.5)))
    assert _codes(found) == ["part_below_floor"]
    assert "below the floor" in found[0].text


def test_no_targets_is_an_error():
    found = pf.check_targets(gp.flat_plate(), Targets())
    assert _codes(found) == ["no_targets"] and found[0].level == "error"


def test_inward_normals_and_off_surface_points():
    m, P, N = _plate_top(6)
    N[:2] *= -1  # two normals point into the plate
    P[5, 2] += 0.05  # one point floats 50 mm above it
    t = _targets(P, N)
    found = {f.code: f for f in pf.check_targets(m, t)}
    ids = [p.id for p in t.points]
    assert set(found) == {"inward_normal", "off_surface"}
    assert found["inward_normal"].ids == ids[:2]
    assert found["off_surface"].ids == [ids[5]]
    assert "50 mm" in found["off_surface"].text


def test_duplicates_only_with_the_same_normal():
    m, P, N = _plate_top(3)
    P2 = np.vstack([P, P[0], P[1]])
    N2 = np.vstack([N, N[0], -N[1]])  # the second copy faces the other way: not a duplicate
    found = {f.code: f for f in pf.check_targets(m, _targets(P2, N2))}
    assert len(found["duplicates"].ids) == 1


def test_sampling_caps_the_surface_queries(monkeypatch):
    m, P, N = _plate_top(40)
    N[::2] *= -1
    monkeypatch.setattr(pf, "MAX_SAMPLE", 10)
    f = next(f for f in pf.check_targets(m, _targets(P, N)) if f.code == "inward_normal")
    assert "of 10 points sampled" in f.text and 0 < len(f.ids) <= 10


def test_tool_and_tcp_names():
    _m, P, N = _plate_top(2)
    tool = EndEffector(name="t", tcps=[TCP(name="pad")])
    assert _codes(pf.check_tool(None, Targets())) == ["no_tool"]
    assert pf.check_tool(tool, _targets(P, N, tcps=["pad"])) == []
    bad = pf.check_tool(tool, _targets(P, N, tcps=["tip"]))
    assert _codes(bad) == ["unknown_tcp"] and bad[0].level == "error"
    some = pf.check_tool(tool, _targets(P, N, tcps=["pad", "tip"]))
    assert some[0].level == "warn"


def test_robot():
    assert _codes(pf.check_robot(None)) == ["no_robot"]
    assert _codes(pf.check_robot("numeric")) == ["numeric_ik"]
    assert pf.check_robot("ur_analytic") == []


def test_span_and_height_for_a_static_base():
    P = np.array([[0.0, 0, 0.5], [6.0, 0, 0.5]])
    N = np.tile([0.0, 0, 1], (2, 1))
    kw = dict(floor_mount=True, base_z=0.0, shoulder_m=0.2)
    t = _targets(P, N)
    assert _codes(pf.check_span(t, np.eye(4), 1.75, 0.2, static=True, **kw)) == ["span"]
    assert pf.check_span(t, np.eye(4), 1.75, 0.2, static=False, **kw) == []  # a carrier: not judged
    assert pf.check_span(t, np.eye(4), None, 0.2, static=True, **kw) == []  # reach unknown
    near = _targets(P * [0.1, 1, 1], N)
    assert pf.check_span(near, np.eye(4), 1.75, 0.2, static=True, **kw) == []
    high = pf.check_span(near, make_T(p=(0, 0, 3.0)), 1.75, 0.2, static=True, **kw)
    assert _codes(high) == ["too_high"] and len(high[0].ids) == 2
    kw["floor_mount"] = False  # a ceiling mount: the height bound does not apply
    assert pf.check_span(near, make_T(p=(0, 0, 3.0)), 1.75, 0.2, static=True, **kw) == []


def test_points_inside_a_stayout_zone():
    P = np.array([[0.0, 0, 0.1], [2.0, 0, 0.1]])
    t = _targets(P, np.tile([0.0, 0, 1], (2, 1)))
    box = trimesh.creation.box((0.5, 0.5, 0.5))
    found = pf.check_zones(t, np.eye(4), [("fence", box, make_T(p=(2.0, 0, 0.1)))])
    assert _codes(found) == ["in_stayout"] and found[0].ids == [t.points[1].id]
    assert "'fence'" in found[0].text
    assert pf.check_zones(t, np.eye(4), [("far", box, make_T(p=(9.0, 0, 0)))]) == []


def test_a_failing_check_does_not_stop_the_others():
    def boom():
        raise RuntimeError("no mesh\nsecond line")

    found = pf.run_checks([("part", boom), ("robot", lambda: pf.check_robot(None))])
    assert _codes(found) == ["no_robot", "check_failed"]  # errors first
    assert "second line" not in found[1].text
    assert pf.headline(found) == "Setup check: 1 error(s), 1 note(s)."
    assert pf.headline([]) == "Setup check: nothing found."
    assert pf.as_text(found).count("\n") == 2


# --- session adapter, CLI, panel --------------------------------------------------------------


def _session():
    from ui.session import PartState, Session

    s = Session()
    s.part = PartState("assets/parts/flat_plate.stl", "m", make_T(p=(1.0, 0.0, 0.5)))
    s.load_robot_from_registry("ur20")
    s.load_tool_from_library("orbital_sander")
    return s


def test_session_findings():
    from ui.preflight_run import findings_for

    s = _session()
    assert _codes(findings_for(s)) == ["no_targets"]
    _m, P, N = _plate_top(4)
    s.targets = _targets(P, N)
    assert findings_for(s) == []
    s.part.units = "mm"  # the same file read as mm: 1 mm across, and the points are now off it
    assert {"part_tiny", "off_surface"} <= set(_codes(findings_for(s)))
    s.part = None
    assert _codes(findings_for(s)) == ["no_part"]


def test_cli_exit_codes(tmp_path, capsys):
    from ui.preflight_run import main
    from ui.session import save_project

    s = _session()
    save_project(s, tmp_path / "p")
    assert main([str(tmp_path / "p")]) == 1  # no targets: an error
    assert "no target points" in capsys.readouterr().out.lower()
    _m, P, N = _plate_top(4)
    s.targets = _targets(P, N)
    save_project(s, tmp_path / "p")
    assert main([str(tmp_path / "p")]) == 0
    assert "nothing found" in capsys.readouterr().out
    assert main([str(tmp_path / "missing")]) == 2


def test_panel_lists_findings_and_logs_only_problems():
    import viser

    from ui import log_panel
    from ui.preflight_ui import SetupCheckPanel, findings_md

    server = viser.ViserServer(host="127.0.0.1", port=8873, verbose=False)
    try:
        s = _session()
        panel = SetupCheckPanel(server, s)
        log_panel.clear()
        panel.run(block=True)
        assert "no target points" in panel.result.content.lower()
        assert [lv for _t, lv, _m in log_panel.entries()] == ["error"]
        _m, P, N = _plate_top(4)
        s.targets = _targets(P, N)
        log_panel.clear()
        panel.result.content = "kept"
        panel.run(block=True, quiet=True)  # a planner started and nothing is wrong: silent
        assert panel.result.content == "kept" and log_panel.entries() == []
        panel.run(block=True)
        assert panel.result.content == findings_md([]) and "Nothing found" in panel.result.content
    finally:
        server.stop()
