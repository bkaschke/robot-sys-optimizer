"""Write the ready-to-load example projects (Project panel -> Examples, or Folder + Load).

    python scripts/make_examples.py

examples/rail_long_part/project.json
    UR20 on a 7th-axis floor rail (track, carriage, riser built from boxes/cylinders) beside
    the 6 m long test rail, a wall keep-out behind the part, 250 mm grid on the top face.
examples/mobile_amr/project.json
    UR20 on a mobile robot designed from simple shapes (4 wheels, chassis, battery box,
    column, robot on the column), tip-over check on, same long part, top face.

Both use the orbital sander and are ready for Optimize -> Station planner -> Plan stations.
Paths in the files are repo-relative (part STL, tool URDF), so the folders can be loaded
in place or copied anywhere inside the repo's workspace/.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core import golden_parts as gp  # noqa: E402
from core.carrier import Carrier  # noqa: E402
from core.schema import Base, CarrierJoints, Keepout, Pose, Primitive, Standoffs, TargetGroup  # noqa: E402
from core.schema import Targets as ProjectTargets  # noqa: E402
from core.targets import append_points, projected_grid, select_by_normal  # noqa: E402
from core.transforms import make_T  # noqa: E402
from ui.session import PartState, Session, save_project  # noqa: E402

EXAMPLES = ROOT / "examples"
PART_POSE = make_T(p=(0.0, 0.8, 0.5))  # long rail, 0.5 m above the floor (top at 0.65 m)
WALL = Keepout(min_xy=(-5.0, 1.3), max_xy=(5.0, 3.0))


def rail_base() -> Base:
    return Base(
        mode="movable",
        carrier="rail",
        pose=Pose(xyz=(0.0, 0.15, 0.0)),
        bodies=[
            Primitive(name="track", shape="box", size=(7.0, 0.3, 0.1), attach="fixed"),
            Primitive(name="carriage", shape="box", size=(0.5, 0.5, 0.15), pose=Pose(xyz=(0, 0, 0.1)), kg=60),
            Primitive(
                name="riser", shape="cylinder", radius=0.15, height=0.45, pose=Pose(xyz=(0, 0, 0.25)), kg=40
            ),
        ],
        T_mount=Pose(xyz=(0.0, 0.0, 0.7)),
        joints=CarrierJoints(x=(-3.0, 3.0)),
        keepouts=[WALL],
        standoff_m=Standoffs(carrier_part_min=0.3, carrier_part_preferred=0.4, carrier_env_min=0.1),
    )


def mobile_base() -> Base:
    wheel = dict(shape="cylinder", radius=0.1, height=0.08, kg=10)
    wheels = [
        Primitive(name=f"wheel_{tag}", pose=Pose(xyz=(x, y, 0.1), rpy=(np.pi / 2, 0, 0)), **wheel)
        for tag, x, y in (("fl", 0.45, 0.39), ("fr", 0.45, -0.31), ("rl", -0.45, 0.39), ("rr", -0.45, -0.31))
    ]
    return Base(
        mode="movable",
        carrier="mobile",
        bodies=[
            *wheels,
            Primitive(name="chassis", shape="box", size=(1.1, 0.7, 0.3), pose=Pose(xyz=(0, 0, 0.1)), kg=220),
            Primitive(
                name="battery", shape="box", size=(0.35, 0.5, 0.2), pose=Pose(xyz=(-0.3, 0, 0.4)), kg=60
            ),
            Primitive(
                name="column", shape="cylinder", radius=0.16, height=0.3, pose=Pose(xyz=(0.2, 0, 0.4)), kg=25
            ),
        ],
        T_mount=Pose(xyz=(0.2, 0.0, 0.7)),
        joints=CarrierJoints(x=(-4.0, 4.0), y=(-3.0, 0.5), yaw_deg=(-180, 180)),
        keepouts=[WALL],
        standoff_m=Standoffs(carrier_part_min=0.3, carrier_part_preferred=0.5, carrier_env_min=0.1),
        localization_m=0.01,
        localization_deg=1.0,
        tip_over=True,
        facing="front",  # v2: the robot side faces the part, one fixed 400 mm standoff (opt.contour)
        fixed_standoff_m=0.4,
    )


def build(name: str, base: Base, start: dict) -> Path:
    s = Session(project_dir=EXAMPLES / name)
    s.load_robot_from_registry("ur20")
    s.load_tool_from_library("orbital_sander")
    s.part = PartState("assets/parts/long_rail.stl", "m", PART_POSE)
    s.base = base
    s.T_world_base = Carrier(base).T_world_base(start)
    part = gp.long_rail()
    top = select_by_normal(part, (0, 0, 1), 5.0)
    P, N = projected_grid(part, top, 0.25)
    group = TargetGroup(name="top", tilt_deg=s.tool.tolerances.tilt_deg)
    s.targets = append_points(ProjectTargets(groups=[group]), P, N, "top")
    path = save_project(s)
    print(f"{path.relative_to(ROOT)}: {len(P)} targets, {len(base.bodies)} carrier bodies")
    return path


def main() -> None:
    build("rail_long_part", rail_base(), {"s": -1.5})
    build("mobile_amr", mobile_base(), {"x": -1.5, "y": -0.2, "yaw": 0.0})


if __name__ == "__main__":
    main()
