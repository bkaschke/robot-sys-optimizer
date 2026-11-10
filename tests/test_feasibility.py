"""M5 golden scenes. Done-when: reachable counts match hand-verified expectations.

Robot: UR20 at the world origin (floor mount), reach 1.75 m from the shoulder axis.
Tool: sample orbital sander (TCP 0.19 m + L beyond the flange, pad contact link).
Each scene states why the expected answer is obvious from geometry.
"""

from pathlib import Path

import numpy as np
import pytest

from core import golden_parts as gp
from core.collision import CollisionScene
from core.feasibility import FeasibilityChecker, candidate_poses, target_frame
from core.ik import make_ik
from core.robot import Robot
from core.schema import TargetGroup, TargetPoint, Targets
from core.tool import ToolModel, load_tool
from core.transforms import make_T

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def robot():
    return Robot.from_registry("ur20")


@pytest.fixture(scope="module")
def ik(robot):
    return make_ik(robot)


def _checker(robot, ik, part, T_part, **ee_update):
    ee = load_tool("orbital_sander")
    nominal = ee_update.pop("L", None)
    if nominal is not None:
        ext = ee.extension.model_copy(
            update={"nominal": ee.extension.nominal.model_copy(update={"L": nominal})}
        )
        ee = ee.model_copy(update={"extension": ext})
    ee = ee.model_copy(update=ee_update)
    tm = ToolModel(ee, REPO / ee.urdf)
    scene = CollisionScene(robot, tm, part, np.eye(4), T_part)
    return FeasibilityChecker(robot, ik, tm, scene, np.eye(4), T_part)


def _targets(pts, group: TargetGroup) -> Targets:
    return Targets(
        groups=[group],
        points=[TargetPoint(id=i, p=p, n=n, group=group.name) for i, (p, n) in enumerate(pts)],
    )


SANDING = TargetGroup(name="top", tilt_deg=10.0)


def test_scene_plate_in_front_all_reachable(robot, ik):
    """Plate top 0.32 m high, points 0.7-1.1 m in front of the base: well inside reach,
    far from the base and floor -> all 9 reachable."""
    T_part = make_T(p=(0.9, 0.0, 0.3))
    pts = [((x, y, 0.02), (0, 0, 1)) for x in (-0.2, 0, 0.2) for y in (-0.2, 0, 0.2)]
    res = _checker(robot, ik, gp.flat_plate(), T_part).check_all(_targets(pts, SANDING))
    assert [r.status for r in res].count("reachable") == 9
    for r in res:  # the reported solution really puts the TCP on the target, pointing in
        T = robot.fk(r.q) @ load_tool_tcp()
        p_world = T_part[:3, :3] @ np.array(pts[r.id][0]) + T_part[:3, 3]
        assert np.allclose(T[:3, 3], p_world, atol=1e-6)
        assert np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1))) <= 10.0 + 1e-6


def load_tool_tcp() -> np.ndarray:
    ee = load_tool("orbital_sander")
    return ToolModel(ee, REPO / ee.urdf).T_flange_tcp("pad")


def test_scene_long_rail_counts(robot, ik):
    """6 m rail 0.8 m beside the base, top 0.55 m high. Points at |x| <= 1 m are <= 1.28 m
    from the base axis (reachable); points at |x| >= 2 m are >= 2.15 m away, beyond
    reach + tool (1.75 + 0.36) -> exactly 5 reachable, 6 unreachable for lack of IK."""
    xs = [-3, -2.5, -2, -1, -0.5, 0, 0.5, 1, 2, 2.5, 3]
    pts = [((x, 0, 0.15), (0, 0, 1)) for x in xs]
    res = _checker(robot, ik, gp.long_rail(), make_T(p=(0, 0.8, 0.4))).check_all(_targets(pts, SANDING))
    reach = [r.status != "unreachable" for r in res]
    assert reach == [abs(x) <= 1 for x in xs]
    assert all(r.reason == "no_ik" for r in res if r.status == "unreachable")


def test_scene_contact_filtering(robot, ik):
    """Sanding contact: the pad face sits exactly on the surface. With the pad as contact
    link the point is reachable (the motor body keeps 20 mm > 10 mm clearance); without a
    contact link the pad itself violates the clearance -> collision."""
    T_part = make_T(p=(0.9, 0.0, 0.3))
    t = _targets([((0, 0, 0.02), (0, 0, 1))], SANDING)
    ok = _checker(robot, ik, gp.flat_plate(), T_part).check_all(t)[0]
    assert ok.status == "reachable" and np.isclose(ok.clearance, 0.02, atol=1e-4)
    tcp = load_tool("orbital_sander").tcps[0]
    no_contact = [tcp.model_copy(update={"contact_links": []})]
    bad = _checker(robot, ik, gp.flat_plate(), T_part, tcps=no_contact).check_all(t)[0]
    assert bad.status == "unreachable" and bad.reason == "collision"


def test_contact_faces_with_one_ring(robot, ik):
    """Picking only the pad's working face (instead of the whole link) must also work: the
    rim faces touching the part at the pad edge are covered by the one-ring rule."""
    ee = load_tool("orbital_sander")
    pad = ToolModel(ee, REPO / ee.urdf).link("pad_link").mesh
    top = np.flatnonzero(pad.face_normals[:, 2] > 0.99).tolist()
    tcp = ee.tcps[0].model_copy(update={"contact_links": [], "contact_faces": {"pad_link": top}})
    t = _targets([((0, 0, 0.02), (0, 0, 1))], SANDING)
    r = _checker(robot, ik, gp.flat_plate(), make_T(p=(0.9, 0, 0.3)), tcps=[tcp]).check_all(t)[0]
    assert r.status == "reachable"


@pytest.mark.parametrize(("L", "expected"), [(0.05, False), (0.15, False), (0.3, True), (0.6, True)])
def test_scene_pocket_needs_long_tool(robot, ik, L, expected):
    """Pocket floor 0.45 m below the rim, 0.4 m wide. TCP-to-flange = 0.19 + L. For L <= 0.15
    the wrist (~0.15 m dia., offset links) must enter the pocket -> collision; for
    L >= 0.3 the flange stays above the rim -> reachable."""
    t = _targets([((0, 0, 0.05), (0, 0, 1))], TargetGroup(name="floor", tilt_deg=0.0))
    r = _checker(robot, ik, gp.open_box(), make_T(p=(0.9, 0, 0.0)), L=L).check_all(t)[0]
    assert (r.status != "unreachable") == expected
    if not expected:
        assert r.reason == "collision"


def test_scanning_standoff(robot, ik):
    """Contact-free process: TCP held 200 mm off the surface along the normal."""
    from core.process import apply_preset

    ee = apply_preset(load_tool("scanner"), "scanning")
    tm = ToolModel(ee, REPO / ee.urdf)
    T_part = make_T(p=(0.9, 0, 0.3))
    scene = CollisionScene(robot, tm, gp.flat_plate(), np.eye(4), T_part)
    g = TargetGroup(name="scan", tilt_deg=3.0, standoff_m=0.2)
    r = FeasibilityChecker(robot, ik, tm, scene, np.eye(4), T_part).check_point(
        0, np.array([0, 0, 0.02]), np.array([0, 0, 1.0]), g
    )
    assert r.status == "reachable"
    T = robot.fk(r.q) @ tm.T_flange_tcp("sensor")
    assert np.isclose(T[2, 3], 0.3 + 0.02 + 0.2, atol=1e-6)


# --- sampling and statics units ---------------------------------------------------------


def test_candidate_poses_cover_cone_spin_and_standoff():
    g = TargetGroup(name="g", spin_deg="free", tilt_deg=12.0, standoff_m=(0.1, 0.3))
    n = np.array([0.0, 0.6, 0.8])
    poses = list(candidate_poses(np.zeros(3), n, g, ["a"]))
    assert len(poses) == (1 + 12) * 3 * 12  # tilts x standoffs x spins
    tilts = {round(t, 6) for _, _, t, _, _ in poses}
    assert tilts == {0.0, 6.0, 12.0}
    for _, _, tilt, so, T in poses:
        ang = np.degrees(np.arccos(np.clip(-(T[:3, 2] @ n), -1, 1)))
        assert np.isclose(ang, tilt, atol=1e-6)  # tool axis inside the cone
        assert np.allclose(T[:3, 3], so * n)  # standoff along the normal
    assert poses[0][2] == 0.0  # untilted first


def test_fixed_spin_is_centred_first():
    g = TargetGroup(name="g", spin_deg=(80.0, 100.0))
    spins = [s for _, s, _, _, _ in candidate_poses(np.zeros(3), np.array([0, 0, 1.0]), g, ["a"])]
    assert spins[0] == 90.0 and min(spins) >= 80.0 and max(spins) <= 100.0
    T0 = target_frame(np.zeros(3), np.array([0, 0, 1.0]))
    assert np.allclose(T0[:3, 0], [1, 0, 0])  # spin reference: part +x
