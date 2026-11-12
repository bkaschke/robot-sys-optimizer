"""M10 checks. Done-when (defined for M10, the plan gives none): the exports reproduce the
plan exactly.
  * CSV: every row's TCP pose, recomputed from its joint solution, lies on its target
    (p + standoff * n) within 1 um, and re-parses with the documented columns;
  * tool URDF: loads in Pinocchio and yourdfpy (with meshes); every tcp_<name> frame equals
    ToolModel.T_flange_tcp for the optimized values;
  * HTML report: self-contained (no external requests) with all sections;
  * all three exports work from the GUI (Results tab -> Export) into <project>/export/.
"""

import csv
import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest
import yourdfpy

from core import export as ex
from core import golden_parts as gp
from core.carrier import Carrier
from core.collision import CollisionScene
from core.feasibility import FeasibilityChecker
from core.ik import make_ik
from core.report import ReportInput, build_report
from core.robot import Robot
from core.schema import Base, CarrierJoints, ExtensionValues, Pose, Primitive, Station, TargetGroup, Weights
from core.schema import Targets as SchemaTargets
from core.targets import append_points
from core.tool import ToolModel, load_tool
from core.transforms import make_T, rpy_to_matrix
from opt.placement import PlannerSettings, Targets, plan_stations
from opt.reachability import map_for_tool
from tests.helpers import confirm_all

REPO = Path(__file__).resolve().parents[1]
V_OPT = ExtensionValues(L=0.3, theta_deg=20.0)


@pytest.fixture(scope="module")
def sander():
    ee = load_tool("orbital_sander")
    return ee, ToolModel(ee, REPO / ee.urdf)


# --- tool URDF -------------------------------------------------------------------------------------


def test_tool_urdf_tcp_frames_match_model(sander, tmp_path):
    ee, tm = sander
    path = ex.export_tool(ee, V_OPT, REPO / ee.urdf, tmp_path / "out")
    model = pin.buildModelFromUrdf(str(path))
    data = model.createData()
    pin.framesForwardKinematics(model, data, pin.neutral(model))
    T_flange = data.oMf[model.getFrameId("flange")].homogeneous
    for t in ee.tcps:
        T = np.linalg.inv(T_flange) @ data.oMf[model.getFrameId(f"tcp_{t.name}")].homogeneous
        assert np.allclose(T, tm.T_flange_tcp(t.name, V_OPT), atol=1e-9)
    # meshes travel with the URDF: yourdfpy resolves them relative to the file
    u = yourdfpy.URDF.load(str(path), load_meshes=True, build_scene_graph=True)
    assert len(u.scene.geometry) >= 2


def test_tool_urdf_straight_and_bent_differ(sander):
    ee, _ = sander
    a = ex.tool_urdf(ee, ExtensionValues(L=0.3), None)
    b = ex.tool_urdf(ee, V_OPT, None)
    assert a != b and ET.fromstring(b).get("name") == "orbital_sander_optimized"
    assert len(ET.fromstring(b).findall("link[@name='extension_0']")) == 1


# --- station poses -----------------------------------------------------------------------------------


def test_station_pose_carrier_and_floor():
    gantry = Base(
        mode="movable",
        carrier="gantry",
        pose=Pose(xyz=(0, 0, 2.5)),
        chassis=Primitive(shape="box", size=(0.4, 0.4, 0.1)),
        T_mount=Pose(rpy=(np.pi, 0, 0)),
        joints=CarrierJoints(x=(0, 3), y=(-1, 1)),
    )
    st = Station(xyz=(1.0, 0.5, 2.5), yaw=0.0, carrier={"x": 1.0, "y": 0.5})
    T = ex.station_pose(gantry, st, np.eye(4))
    assert np.allclose(T, Carrier(gantry).T_world_base({"x": 1.0, "y": 0.5}))
    assert np.allclose(T[:3, 2], [0, 0, -1])  # inverted mount survives (yaw alone would lose it)
    T0 = make_T(rpy_to_matrix((0, 0, 0.3)), (0, 0, 0.1))
    floor = ex.station_pose(Base(), Station(xyz=(1, 2, 0.1), yaw=0.8), T0)
    assert np.allclose(floor[:3, 3], [1, 2, 0.1]) and np.isclose(np.arctan2(floor[1, 0], floor[0, 0]), 0.8)


# --- CSV ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plate_plan(sander):
    ee, tm = sander
    robot = Robot.from_registry("ur20")
    T_part = make_T(p=(0.9, 0.0, 0.3))
    part = gp.flat_plate()
    pts = np.array([(x, y, 0.02) for x in (-0.3, 0.0, 0.3) for y in (-0.2, 0.2)])
    st = SchemaTargets(groups=[TargetGroup(name="top", tilt_deg=10.0)])
    st = append_points(st, pts, np.tile([0, 0, 1.0], (len(pts), 1)), "top")
    t = Targets(
        np.arange(len(pts)),
        pts + T_part[:3, 3],
        np.tile([0, 0, 1.0], (len(pts), 1)),
        pts,
        np.tile([0, 0, 1.0], (len(pts), 1)),
        [st.groups[0]] * len(pts),
        np.full(len(pts), 10.0),
    )
    ch = FeasibilityChecker(
        robot, make_ik(robot), tm, CollisionScene(robot, tm, part, np.eye(4), T_part), np.eye(4), T_part
    )
    rmap = map_for_tool(robot, tm.T_flange_tcp("pad"))
    b = part.bounds
    fp = np.array([b[0, :2] + T_part[:2, 3], b[1, :2] + T_part[:2, 3]])
    sol, res = plan_stations(rmap, ch, t, np.eye(4), fp, Weights(), PlannerSettings(max_stations=1))[0]
    return robot, tm, sol, res, st, T_part


def test_csv_rows_reproduce_targets(plate_plan):
    robot, tm, sol, res, st, T_part = plate_plan
    rows = ex.tcp_rows(sol, res, st, robot, tm, Base(), np.eye(4))
    text = ex.rows_to_csv(rows)
    back = list(csv.DictReader(io.StringIO(text)))
    assert list(back[0]) == ex.CSV_COLUMNS and len(back) == len(st.points)
    for row in back:
        assert row["status"] in ("reachable", "marginal")
        p = next(pt for pt in st.points if pt.id == int(row["point_id"]))
        target = T_part[:3, :3] @ np.array(p.p) + T_part[:3, 3]
        tcp = np.array([float(row[f"tcp_world_{a}_mm"]) for a in "xyz"]) / 1000
        assert np.allclose(tcp, target, atol=1e-6)
        q = np.radians([float(row[f"q{j}_deg"]) for j in range(1, 7)])
        T_base = ex.station_pose(Base(), sol.stations[int(row["station"]) - 1], np.eye(4))
        T = T_base @ robot.fk(q) @ tm.T_flange_tcp(row["tcp"])
        assert np.allclose(T[:3, 3], tcp, atol=2e-6)  # rows round q to 1e-4 deg
        assert np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1))) <= 10.0 + 1e-3  # inside the tilt cone


def test_csv_lists_unreached_points(plate_plan):
    robot, tm, sol, res, st, _ = plate_plan
    sol2 = sol.model_copy(update={"unreached": [99]})
    rows = ex.tcp_rows(sol2, res, st, robot, tm, Base(), np.eye(4))
    assert rows[-1]["point_id"] == 99 and rows[-1]["status"] == "unreachable"


# --- envelope SVG and report ----------------------------------------------------------------------------


def test_envelope_svg(sander):
    _, tm = sander
    svg = ex.envelope_svg(tm, V_OPT)
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "L = 300.0 mm" in svg and "θ = 20.0°" in svg and "TCP pad" in svg


def test_report_is_self_contained(plate_plan, sander):
    robot, tm, sol, res, st, _ = plate_plan
    rows = ex.tcp_rows(sol, res, st, robot, tm, Base(), np.eye(4))
    img = np.zeros((20, 30, 3), np.uint8)
    from core.report import png_data_uri

    html = build_report(
        ReportInput(
            title="t",
            robot="ur20",
            tool="orbital_sander",
            process="sanding",
            base_mode="static",
            solution=sol,
            rows=rows,
            station_poses=[ex.station_pose(Base(), s, np.eye(4)) for s in sol.stations],
            envelope_svg=ex.envelope_svg(tm, V_OPT),
            images=[("view", png_data_uri(img))],
            notes=["n1"],
        )
    )
    assert "<h2>Points</h2>" not in html  # v9 #8: the CSV export has every point
    for section in ("Cost breakdown", "Stations", "Tool geometry", "Notes", "Universal Robots"):
        assert section in html
    refs = re.findall(r"""(?:src|href)\s*=\s*['"]([^'"]+)""", html)
    assert refs and all(r.startswith("data:") for r in refs)  # nothing fetched from the network
    assert "<script" not in html and "<link" not in html


# --- GUI ----------------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_gui_exports(tmp_path):
    import viser

    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8772, verbose=False)
    try:
        state = build_app(server)
        s, cell, tab = state["session"], state["cell"], state["targets"]
        s.project_dir = tmp_path
        cell.part_pose.xyz.value = (900.0, 0.0, 300.0)
        pts = np.array([(x, 0.0, 0.02) for x in (-0.2, 0.0, 0.2)])
        s.targets = s.targets.model_copy(update={"groups": [TargetGroup(name="top", tilt_deg=10.0)]})
        s.targets = append_points(s.targets, pts, np.tile([0, 0, 1.0], (3, 1)), "top")
        tab.refresh()
        confirm_all(state["stations"])
        state["stations"].start(block=True)  # static base: one station
        res = state["results"]
        assert res.selected() is not None and len(res.selected().stations) == 1
        exp = res.export
        for fn, name in (
            (exp.export_csv, "tcp_poses.csv"),
            (exp.export_urdf, "orbital_sander_optimized.zip"),
            (exp.export_report, "report.html"),
        ):
            path, data = fn(res.selected(), None)
            assert path == tmp_path / "export" / name and path.read_bytes() == data and len(data) > 200
        assert (tmp_path / "export" / "orbital_sander_optimized" / "orbital_sander_optimized.urdf").is_file()
        exp._run(type("E", (), {"client": None})(), exp.export_csv)  # the button path, no browser attached
        assert "tcp_poses.csv" in exp.info.content
    finally:
        server.stop()
