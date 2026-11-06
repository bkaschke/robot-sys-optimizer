"""v2 phase 4 (docs/V2_PLAN.md): pose tools (head pose x y z rx ry rz with ranges), spline
connector, the flange map for any TCP offset, the pose optimizer (best overall cost, no
payload), GUI and exports."""

import numpy as np
import pytest
import viser

from core.ik import best_copies, make_ik
from core.robot import Robot
from core.schema import EndEffector, Extension, ExtensionValues, length_ref
from core.tool import SPLINE_SEGMENTS, ToolModel
from core.transforms import make_T, rpy_to_matrix
from opt.reachability import OffsetMap, PoseOffsetMap, ReachMap


def _pose_tool(**nominal) -> ToolModel:
    v = ExtensionValues(**{"z": 0.3, **nominal})
    x = Extension(
        type="pose",
        x=(-0.2, 0.2),
        y=(-0.2, 0.2),
        z=(0.05, 0.5),
        ry_deg=(-90, 90),
        rx_deg=(-90, 90),
        rz_deg=(-90, 90),
        nominal=v,
    )
    from core.schema import TCP, Pose

    ee = EndEffector(name="p", extension=x, tcps=[TCP(name="tcp", T_head_tcp=Pose(xyz=(0, 0, 0.05)))])
    return ToolModel(ee, None)


# --- schema and geometry --------------------------------------------------------------------------


def test_pose_schema_ranges_and_reference_length():
    with pytest.raises(ValueError, match="outside its range"):
        Extension(type="pose", z=(0.1, 0.2), nominal=ExtensionValues(z=0.3))
    x = Extension(type="pose", x=(0, 0.3), y=(0, 0), z=(0.1, 0.4), nominal=ExtensionValues(z=0.2))
    assert length_ref(x) == pytest.approx(0.5)  # farthest corner (0.3, 0, 0.4)
    assert length_ref(Extension(L=(0.0, 0.7), nominal=ExtensionValues(L=0.2))) == pytest.approx(0.7)
    Extension(type="straight", nominal=ExtensionValues(L=0.2, z=9.0))  # legacy ignores pose values


def test_pose_frames_and_straight_connector():
    tm = _pose_tool(x=0.05, y=-0.02, z=0.3, rx_deg=10, ry_deg=-20, rz_deg=30)
    T = tm.T_flange_head()
    assert np.allclose(T, make_T(rpy_to_matrix(np.radians([10, -20, 30])), (0.05, -0.02, 0.3)), atol=1e-12)
    straight = _pose_tool(z=0.3)
    segs = straight.extension_segments()
    assert len(segs) == SPLINE_SEGMENTS
    for a, b in segs:  # head right above the mount: a straight connector along z
        assert np.allclose((b - a)[:2], 0, atol=1e-12)
    assert straight.length() == pytest.approx(0.3, abs=1e-9)


def test_spline_connector_leaves_and_arrives_along_the_axes():
    tm = _pose_tool(x=0.15, z=0.25, ry_deg=60)
    P = tm.connector_points()
    T_fh = tm.T_flange_head()
    assert np.allclose(P[0], 0) and np.allclose(P[-1], T_fh[:3, 3])
    d0 = (P[1] - P[0]) / np.linalg.norm(P[1] - P[0])
    d1 = (P[-1] - P[-2]) / np.linalg.norm(P[-1] - P[-2])
    assert np.degrees(np.arccos(d0 @ [0, 0, 1])) < 12  # leaves the mount along its z
    assert np.degrees(np.arccos(d1 @ T_fh[:3, 2])) < 12  # arrives along the head z
    assert tm.length() > np.linalg.norm(T_fh[:3, 3])  # a curve, longer than the chord
    tm.ee = tm.ee.model_copy(update={"extension": tm.ee.extension.model_copy(update={"kg_per_m": 2.0})})
    assert tm.mass_properties().mass_kg == pytest.approx(2.0 * tm.length(), rel=1e-9)


# --- map for any TCP offset ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def flange_map():
    return ReachMap.load_or_build(Robot.from_registry("ur20"), np.eye(4))


def test_pose_offset_map_equals_offset_map_for_straight_tcp(flange_map):
    T_ft = make_T(p=(0, 0, 0.3))
    rng = np.random.default_rng(2)
    P = rng.uniform(-1.5, 1.5, (400, 3))
    A = rng.normal(size=(400, 3))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    s1, m1 = OffsetMap(flange_map, 0.3).lookup(P, A, 10.0)
    s2, m2 = PoseOffsetMap(flange_map, T_ft).lookup(P, A, 10.0)
    assert np.allclose(s1, s2) and np.allclose(m1, m2)  # spin does not matter on the axis


def test_pose_offset_map_agrees_with_exact_ik(flange_map):
    robot = Robot.from_registry("ur20")
    ik = make_ik(robot)
    T_ft = make_T(rpy_to_matrix(np.radians([0, 50, 20])), (0.12, -0.05, 0.25))
    pm = PoseOffsetMap(flange_map, T_ft)
    rng = np.random.default_rng(1)
    M = 300
    r, ph = rng.uniform(0.3, 1.9, M), rng.uniform(-np.pi, np.pi, M)
    P = np.column_stack([r * np.cos(ph), r * np.sin(ph), rng.uniform(-0.6, 1.6, M)])
    A = rng.normal(size=(M, 3))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    pred = pm.lookup(P, A, 0.0)[0] >= 0.5
    m = np.radians(5)
    lo, hi, T_tf = robot.lower + m, robot.upper - m, np.linalg.inv(T_ft)
    exact = np.zeros(M, bool)
    for i in range(M):
        a = A[i]
        ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        x = ref - (ref @ a) * a
        x /= np.linalg.norm(x)
        y = np.cross(a, x)
        for s in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            R = np.column_stack([np.cos(s) * x + np.sin(s) * y, -np.sin(s) * x + np.cos(s) * y, a])
            sols = ik.solve(make_T(R, P[i]) @ T_tf)
            if len(sols) and best_copies(sols, lo, hi, robot.lower, robot.upper)[1].any():
                exact[i] = True
                break
    assert (pred == exact).mean() > 0.97  # 0.995 on 3000 samples during development


# --- optimizer, GUI, exports ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8794, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


def _make_pose(app, z=(0.1, 0.5), ry=(-45.0, 45.0)):
    s, tool = app["session"], app["tool"]
    assert app["project"].load_example("rail_long_part")
    tool.ext_type.value = "pose"  # GUI path: legacy -> pose
    x = s.tool.extension
    assert x.type == "pose" and tool.ext_ranges["z"].visible and not tool.ext_ranges["L"].visible
    tool.ext_ranges["z"].value = tuple(v * 1000 for v in z)
    tool.ext_ranges["ry_deg"].value = ry
    x = s.tool.extension
    assert x.z == pytest.approx(z) and x.ry_deg == pytest.approx(ry)
    assert x.z[0] - 1e-9 <= x.nominal.z <= x.z[1] + 1e-9
    assert "ry_deg" in tool._nominal and "x" not in tool._nominal  # sliders only for free values
    return s


def test_pose_optimizer_best_cost_no_payload(app):
    s = _make_pose(app)
    panel = app["tool_opt"]
    panel.shape_evals.value = 6
    panel.start(block=True)
    r = panel.result
    assert r.feasible and r.solution is not None and not r.solution.unreached, r.message
    v = r.values
    x = s.tool.extension
    assert x.z[0] - 1e-9 <= v.z <= x.z[1] + 1e-9 and x.ry_deg[0] - 1e-9 <= v.ry_deg <= x.ry_deg[1] + 1e-9
    assert v.x == x.nominal.x and v.rx_deg == x.nominal.rx_deg  # fixed components untouched
    assert len(r.samples) >= 7 and r.solution.cost == pytest.approx(min(sm.cost for sm in r.samples), abs=0.5)
    assert all(p.torque_util == 0.0 for p in r.point_results.values() if p.q is not None)  # payload off
    assert "connector (mm)" in panel.info.content
    panel.apply_values(v)
    assert s.tool.extension.nominal.z == pytest.approx(v.z)


def test_pose_tool_exports(app, tmp_path):
    from core.export import envelope_svg, tool_urdf

    s = _make_pose(app)
    tm = s.tool_model
    svg = envelope_svg(tm)
    assert "spline" in svg and "head x y z" in svg
    urdf = tool_urdf(s.tool, None, s.resolve(s.tool.urdf) if s.tool.urdf else None)
    assert urdf.count("<cylinder") >= SPLINE_SEGMENTS
