"""v8 T16: two-bend tools, several TCPs in the planner (MultiMap), part family."""

import numpy as np
import pytest
import viser

from core.schema import EndEffector, Extension, ExtensionValues
from core.tool import ToolModel
from tests.helpers import confirm_all


def _bent2(**nominal):
    x = Extension(
        type="bent2",
        L=(0.1, 0.4),
        theta_deg=(-90, 90),
        d=(0.0, 0.2),
        theta2_deg=(-90, 90),
        d2=(0.0, 0.2),
        radius=0.02,
        nominal=ExtensionValues(L=0.2, theta_deg=90, d=0.1, theta2_deg=-90, d2=0.05, **nominal),
    )
    return EndEffector(name="t2", extension=x, tcps=[{"name": "tip"}])


def test_two_bend_geometry():
    tm = ToolModel(_bent2(), None)
    v = tm.values()
    segs = tm.extension_segments(v)
    assert len(segs) == 3  # L, d, d2 legs
    assert tm.length(v) == pytest.approx(0.2 + 0.1 + 0.05)
    p = tm.T_mount_head(v)[:3, 3]
    # up 0.2; +90 about x turns the d leg (local y) up: +0.1 in z; -90 turns back: d2 along y
    assert np.allclose(p, [0.0, 0.05, 0.3], atol=1e-9)
    R = tm.T_mount_head(v)[:3, :3]
    assert np.allclose(R, np.eye(3), atol=1e-9)  # +90 then -90: the head points like the mount
    with pytest.raises(ValueError):  # nominal outside its range
        Extension(type="bent2", d2=(0.0, 0.1), nominal=ExtensionValues(d2=0.2))


def test_two_bend_tool_exports(tmp_path):
    from core.export import envelope_svg, export_tool

    ee = _bent2()
    tm = ToolModel(ee, None)
    out = export_tool(ee, None, None, tmp_path)
    assert out.exists() and "extension" in out.read_text()
    assert envelope_svg(tm).startswith("<svg")


def test_multimap_takes_the_best_tcp():
    from opt.reachability import MultiMap

    class Fake:
        def __init__(self, k, mm):
            self.k, self.max_manip = k, mm

        def lookup(self, p, a, cone):
            n = len(p)
            return np.full(n, self.k), np.full(n, 0.1 * self.k)

    m = MultiMap([Fake(0.2, 1.0), Fake(0.9, 2.0)])
    sc, mu = m.lookup(np.zeros((3, 3)), np.zeros((3, 3)), 10.0)
    assert np.allclose(sc, 0.9) and np.allclose(mu, 0.09) and m.max_manip == 2.0


@pytest.mark.slow
def test_two_tip_tool_plans_with_both_maps_and_part_family(tmp_path):
    from app import build_app
    from ui.session import save_project

    server = viser.ViserServer(host="127.0.0.1", port=8874, verbose=False)
    try:
        app = build_app(server)
        assert app["project"].load_example("rail_long_part")
        s = app["session"]
        family_dir = tmp_path / "other"
        save_project(s, family_dir)  # the same part in another project: a family of two
        tip = s.tool.tcps[0]
        T = tip.T_head_tcp.model_copy(update={"xyz": (tip.T_head_tcp.xyz[0] + 0.25, *tip.T_head_tcp.xyz[1:])})
        tool = s.tool.model_copy(
            update={"tcps": [tip, tip.model_copy(update={"name": "tip2", "T_head_tcp": T})]}
        )
        s.set_tool(tool)
        st = app["stations"]
        confirm_all(st)
        st.start(block=True, quick=True)
        res = s.result_points[s.results[0].label]
        used = {r.tcp for r in res.values() if r.status != "unreachable"}
        assert used <= {tip.name, "tip2"} and used
        # the tool optimizer on the family: the other part's cost is added
        s.set_tool(tool.model_copy(update={"tcps": [tip]}))
        to = app["tool_opt"]
        to.shape_evals.value = 0
        confirm_all(to)
        to.start(block=True)
        alone = to.result.solution.cost
        to.family.value = str(family_dir)
        confirm_all(to)
        to.start(block=True)
        both = to.result.solution
        assert "family" in both.breakdown and both.cost > alone
        to.family.value = ""
    finally:
        server.stop()
