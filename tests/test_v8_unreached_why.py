"""Why a point is unreachable: a collision names what the best arm pose hits (result notes),
the scene markers are coloured by reason and labelled, and the log lists what was hit."""

from pathlib import Path

import numpy as np

from core import golden_parts as gp
from core.collision import HITS, CollisionScene, describe_hits
from core.feasibility import FeasibilityChecker, PointResult
from core.ik import make_ik
from core.robot import Robot
from core.schema import TargetGroup, TargetPoint, Targets
from core.tool import ToolModel, load_tool
from core.transforms import make_T

REPO = Path(__file__).resolve().parents[1]


def test_describe_hits_groups_by_obstacle():
    pairs = [
        ("forearm_link_0", "part"),
        ("tool_ext_0", "part"),
        ("upper_arm_link_0", "floor"),
        ("wrist_3_link_0", "stayout_fence"),
        ("carrier_mast", "hose_2"),
        ("tool_pad_link_body", "upper_arm_link_0"),
        ("shoulder_link_0", "wrist_2_link_0"),
    ]
    s = describe_hits(pairs)
    assert s.startswith(HITS)
    assert "part (forearm, tool)" in s and "floor (upper_arm)" in s
    assert "stay-out zone fence (wrist_3)" in s and "carrier mast (hose)" in s
    assert "tool vs arm (upper_arm)" in s and "arm itself (shoulder, wrist_2)" in s
    assert describe_hits([]) == ""


def test_pocket_collision_names_the_part():
    """Short tool into the open box pocket: the wrist must enter it, so the reason is
    collision and the note names the part."""
    robot = Robot.from_registry("ur20")
    ee = load_tool("orbital_sander")
    ext = ee.extension.model_copy(update={"nominal": ee.extension.nominal.model_copy(update={"L": 0.05})})
    ee = ee.model_copy(update={"extension": ext})
    tm = ToolModel(ee, REPO / ee.urdf)
    T_part = make_T(p=(0.9, 0, 0.0))
    ch = FeasibilityChecker(
        robot,
        make_ik(robot),
        tm,
        CollisionScene(robot, tm, gp.open_box(), np.eye(4), T_part),
        np.eye(4),
        T_part,
    )
    g = TargetGroup(name="floor", tilt_deg=0.0)
    t = Targets(groups=[g], points=[TargetPoint(id=0, p=(0, 0, 0.05), n=(0, 0, 1), group="floor")])
    r = ch.check_all(t)[0]
    assert r.status == "unreachable" and r.reason == "collision"
    assert r.notes and r.notes[0].startswith(HITS + "part (")


def test_labels_colours_and_log_say_why():
    from ui.log_panel import unreached_report
    from ui.unreached_view import REASON_COLORS, why

    res = {
        1: PointResult(1, "unreachable", "no_ik"),
        2: PointResult(2, "unreachable", "collision", notes=[HITS + "part (forearm); floor (upper_arm)"]),
        3: PointResult(3, "unreachable", "collision", notes=[HITS + "part (tool)"]),
        4: PointResult(4, "unreachable", "joint_limits"),
    }
    assert why(res[1]) == "out of reach" and why(res[4]) == "joint limits"
    assert why(res[2]) == "collision: part, floor"
    assert why(PointResult(5, "unreachable", "collision")) == "collision"
    assert len({REASON_COLORS[k] for k in ("no_ik", "joint_limits", "singular", "collision")}) == 4
    msg = unreached_report(res, [1, 2, 3, 4], "Plan")
    assert "hitting part (2), floor (1)" in msg


def test_markers_are_coloured_and_labelled():
    from types import SimpleNamespace

    from ui.unreached_view import REASON_COLORS, draw_unreached

    calls = {}

    def rec(kind):
        def f(name, *args, **kw):
            calls[name] = (kind, args, kw)
            return SimpleNamespace(name=name)

        return f

    scene = SimpleNamespace(add_point_cloud=rec("pc"), add_line_segments=rec("ls"), add_label=rec("label"))
    res = {
        7: PointResult(7, "unreachable", "collision", notes=[HITS + "floor (upper_arm)"]),
        8: PointResult(8, "unreachable", "no_ik"),
    }
    P, N = np.zeros((2, 3)), np.tile((0, 0, 1.0), (2, 1))
    out = draw_unreached(SimpleNamespace(scene=scene), "/r", [7, 8], P, N, [7, 8], 0.004, res)
    assert set(out) >= {"unreached", "unreached_x", "unreached_label_7"}
    colors = calls["/r/unreached"][1][1]
    assert tuple(colors[0]) == REASON_COLORS["collision"] and tuple(colors[1]) == REASON_COLORS["no_ik"]
    assert calls["/r/unreached_label_7"][1][0] == "✘ 7 collision: floor"
    assert calls["/r/unreached_label_8"][1][0] == "✘ 8 out of reach"
    seg = calls["/r/unreached_x"][1][1]
    assert seg.shape == (4, 2, 3)


def test_plan_keeps_the_collision_note():
    """verify() keeps the note with the furthest reason it saw (pocket scene, one station)."""
    from opt.station_checks import verify

    robot = Robot.from_registry("ur20")
    ee = load_tool("orbital_sander")
    ext = ee.extension.model_copy(update={"nominal": ee.extension.nominal.model_copy(update={"L": 0.05})})
    ee = ee.model_copy(update={"extension": ext})
    tm = ToolModel(ee, REPO / ee.urdf)
    T_part = make_T(p=(0.9, 0, 0.0))
    ch = FeasibilityChecker(
        robot,
        make_ik(robot),
        tm,
        CollisionScene(robot, tm, gp.open_box(), np.eye(4), T_part),
        np.eye(4),
        T_part,
    )
    from opt.placement import Targets as Arrays

    g = TargetGroup(name="floor", tilt_deg=0.0)
    P, N = np.array([[0, 0, 0.05]]), np.array([[0, 0, 1.0]])
    t = Arrays(np.array([0]), P + T_part[:3, 3], N, P, N, [g], np.zeros(1))
    sites = type("S", (), {"T": staticmethod(lambda v: np.eye(4)), "perturbations": lambda self, v: []})()
    _, results = verify(ch, t, [{}], sites, np.array([0]))
    assert results[0].reason == "collision" and results[0].notes[0].startswith(HITS)
