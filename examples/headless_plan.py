"""Headless example: plan a sanding job on the flat test plate without the GUI, then export.

    python examples/headless_plan.py [output_folder]

Steps: load robot + tool -> place the part -> make targets -> check feasibility -> plan
one base station -> find the shortest tool -> write CSV, tool URDF and HTML report.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core import export as ex  # noqa: E402
from core import golden_parts as gp  # noqa: E402
from core.collision import CollisionScene  # noqa: E402
from core.feasibility import FeasibilityChecker  # noqa: E402
from core.ik import make_ik  # noqa: E402
from core.report import ReportInput, build_report  # noqa: E402
from core.robot import Robot  # noqa: E402
from core.schema import Base, TargetGroup, Weights  # noqa: E402
from core.schema import Targets as ProjectTargets  # noqa: E402
from core.targets import append_points, projected_grid, select_by_normal  # noqa: E402
from core.tool import ToolModel, load_tool  # noqa: E402
from core.transforms import make_T  # noqa: E402
from opt.placement import PlannerSettings, Targets, plan_stations  # noqa: E402
from opt.reachability import map_for_tool  # noqa: E402
from opt.tool_geometry import GeometrySettings, optimize_tool  # noqa: E402


def main(out: Path) -> None:
    robot = Robot.from_registry("ur20")
    ee = load_tool("orbital_sander")
    tool = ToolModel(ee, ROOT / ee.urdf)

    # part: the flat test plate, 0.9 m in front of the robot, top at z = 0.32 m
    part = gp.flat_plate()
    T_part = make_T(p=(0.9, 0.0, 0.3))

    # targets: 100 mm grid on the upward faces (part frame), tolerances from the sanding preset
    top = select_by_normal(part, (0, 0, 1), 5.0)
    P, N = projected_grid(part, top, 0.10)
    group = TargetGroup(name="top", tilt_deg=ee.tolerances.tilt_deg)
    project_targets = append_points(ProjectTargets(groups=[group]), P, N, "top")
    t = Targets(
        np.arange(len(P)),
        P @ T_part[:3, :3].T + T_part[:3, 3],
        N @ T_part[:3, :3].T,
        P,
        N,
        [group] * len(P),
        np.full(len(P), group.tilt_deg),
    )
    print(f"{len(P)} targets")

    # feasibility from the current base (robot at the origin)
    scene = CollisionScene(robot, tool, part, np.eye(4), T_part)
    checker = FeasibilityChecker(robot, make_ik(robot), tool, scene, np.eye(4), T_part)
    res = checker.check_all(project_targets)
    print(
        "feasibility from the origin:",
        {s: sum(r.status == s for r in res) for s in ("reachable", "marginal", "unreachable")},
    )

    # one base station on the free floor (static base mode), off the part footprint
    b = part.bounds
    footprint = np.array([b[0, :2] + T_part[:2, 3], b[1, :2] + T_part[:2, 3]])
    rmap = map_for_tool(robot, tool.T_flange_tcp("pad"))  # ~1 min the first time, then cached
    solution, point_results = plan_stations(
        rmap, checker, t, np.eye(4), footprint, Weights(), PlannerSettings(max_stations=1)
    )[0]
    print(
        "plan:",
        solution.label,
        f"cost {solution.cost:.2f}",
        "base at",
        tuple(round(v, 3) for v in solution.stations[0].xyz),
    )

    # shortest tool (straight length first, then bend angle within the tool's ranges)
    geo = optimize_tool(
        robot,
        ee,
        ROOT / ee.urdf,
        part,
        T_part,
        np.eye(4),
        t,
        footprint,
        Weights(),
        1,
        GeometrySettings(shape_evals=12),
    )
    print("tool:", geo.message, geo.values)

    # exports
    out.mkdir(parents=True, exist_ok=True)
    sol, results = geo.solution, geo.point_results
    if sol is None:  # no length reaches every target: export the best design found instead
        if not geo.designs:
            print("no tool design to export")
            return
        sol, results = geo.designs[0].solution, geo.designs[0].point_results
    rows = ex.tcp_rows(sol, results, project_targets, robot, tool, Base(), np.eye(4))
    (out / "tcp_poses.csv").write_text(ex.rows_to_csv(rows), encoding="utf-8")
    ex.export_tool(ee, sol.tool_values, ROOT / ee.urdf, out / "tool")
    html = build_report(
        ReportInput(
            title=f"Headless example: {sol.label}",
            robot="ur20",
            tool=ee.name,
            process=ee.process,
            base_mode="static",
            solution=sol,
            rows=rows,
            station_poses=[ex.station_pose(Base(), s, np.eye(4)) for s in sol.stations],
            envelope_svg=ex.envelope_svg(tool, sol.tool_values),
            length_samples=geo.samples,
        )
    )
    (out / "report.html").write_text(html, encoding="utf-8")
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "workspace" / "headless_example")
