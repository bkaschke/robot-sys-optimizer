"""Write the ready-to-load example projects (Project panel -> Examples, or Folder + Load).

    python scripts/make_examples.py

examples/rail_long_part/project.json
    UR20 on a 7th-axis floor rail (track, carriage, riser built from boxes/cylinders) beside
    the 6 m long test rail, a wall keep-out behind the part, 250 mm grid on the top face.
examples/mobile_amr/project.json
    UR20 on a mobile robot designed from simple shapes (4 wheels, chassis, battery box,
    column, robot on the column), tip-over check on, same long part, top face.

v6, around the 3 m tall wall (assets/parts/tall_wall.stl, 2 x 0.3 m, on the floor):
examples/gantry_overhead/project.json
    UR20 hanging from an overhead X-Y gantry (two fixed beams on posts, a bridge moving in x),
    top face and the upper part (from 1.8 m) of both long faces.
examples/floor_xy_rails/project.json
    UR20 on a floor X-Y gantry (X rails, a bridge moving in x, carriage along it in y),
    0.25 m to 1.8 m up one long face
    and of the end face.
examples/mobile_lift/project.json
    UR20 on a mobile base with a vertical lift (0..1 m, planned per stop), one long face from
    0.5 m to the top.

All use the orbital sander and are ready for Optimize -> Station planner -> Plan stations.
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


def build(
    name: str,
    base: Base,
    start: dict,
    part: str = "long_rail",
    pose: np.ndarray = PART_POSE,
    faces: tuple = (("top", (0, 0, 1), -np.inf, np.inf),),
    spacing: float = 0.25,
    home_deg: list[float] | None = None,
) -> Path:
    """``faces``: (group, outward normal, z min, z max) per target group (part frame, m);
    ``home_deg``: the project's home pose (None = the robot's spec home)."""
    s = Session(project_dir=EXAMPLES / name)
    s.load_robot_from_registry("ur20")
    s.load_tool_from_library("orbital_sander")
    s.part = PartState(f"assets/parts/{part}.stl", "m", pose)
    s.base = base
    s.T_world_base = Carrier(base).T_world_base(start)
    s.home_deg = home_deg
    mesh = gp.GOLDEN_PARTS[part]()
    tilt = s.tool.tolerances.tilt_deg
    targets = ProjectTargets(groups=[TargetGroup(name=g, tilt_deg=tilt) for g, *_ in faces])
    n = 0
    for g, normal, z0, z1 in faces:
        P, N = projected_grid(mesh, select_by_normal(mesh, normal, 5.0), spacing)
        keep = (P[:, 2] >= z0) & (P[:, 2] <= z1)
        targets = append_points(targets, P[keep], N[keep], g)
        n += int(keep.sum())
    s.targets = targets
    path = save_project(s)
    print(f"{path.relative_to(ROOT)}: {n} targets, {len(base.bodies)} carrier bodies")
    return path


# --- v6: tall wall examples ---------------------------------------------------------------------

WALL_FACES_GANTRY = (
    ("top", (0, 0, 1), -np.inf, np.inf),
    ("side_+y", (0, 1, 0), 1.8, np.inf),
    ("side_-y", (0, -1, 0), 1.8, np.inf),
)


def gantry_base() -> Base:
    """Overhead X-Y gantry: beams along x at y = +-1.8 m, 4.4 m up on posts, a bridge between
    them that moves in x (attach "x_axis"); the carriage runs along the bridge, hangs below the
    beam level and carries the UR20 upside down."""
    posts = [
        Primitive(
            name=f"post_{i}", shape="box", size=(0.2, 0.2, 4.4), pose=Pose(xyz=(x, y, -4.4)), attach="fixed"
        )
        for i, (x, y) in enumerate(((-2.8, -1.8), (2.8, -1.8), (-2.8, 1.8), (2.8, 1.8)))
    ]
    beams = [
        Primitive(
            name=f"beam_{t}", shape="box", size=(5.8, 0.2, 0.25), pose=Pose(xyz=(0, y, 0.0)), attach="fixed"
        )
        for t, y in (("a", -1.8), ("b", 1.8))
    ]
    return Base(
        mode="movable",
        carrier="gantry",
        pose=Pose(xyz=(0.0, 0.0, 4.4)),  # gantry frame on the beam bottoms
        bodies=[
            *posts,
            *beams,
            Primitive(  # the Y axis: spans the gap between the beams, rides along them in x
                name="bridge",
                shape="box",
                size=(0.3, 3.4, 0.25),
                pose=Pose(xyz=(0, 0, 0.0)),
                attach="x_axis",
                kg=80,
            ),
            Primitive(
                name="carriage", shape="box", size=(0.5, 0.5, 0.25), pose=Pose(xyz=(0, 0, -0.25)), kg=80
            ),
            Primitive(
                name="drop", shape="cylinder", radius=0.12, height=0.6, pose=Pose(xyz=(0, 0, -0.85)), kg=30
            ),
        ],
        T_mount=Pose(xyz=(0.0, 0.0, -0.85), rpy=(np.pi, 0.0, 0.0)),  # inverted under the drop
        joints=CarrierJoints(x=(-2.4, 2.4), y=(-1.4, 1.4)),
        standoff_m=Standoffs(carrier_part_min=0.3, carrier_part_preferred=0.4, carrier_env_min=0.1),
    )


def floor_xy_base() -> Base:
    """A floor X-Y gantry: two fixed X rails, a bridge across them that moves in x (attach
    "x_axis"), and a carriage running along the bridge in y; the UR20 stands on a riser on
    the carriage. Works the -y side of the wall and its end."""
    rails = [
        Primitive(
            name=f"x_rail_{t}", shape="box", size=(6.0, 0.15, 0.08), pose=Pose(xyz=(0, y, 0)), attach="fixed"
        )
        for t, y in (("a", -0.75), ("b", 0.75))
    ]
    return Base(
        mode="movable",
        carrier="gantry",
        pose=Pose(xyz=(0.0, -1.4, 0.0)),  # axis frame on the floor, 1.4 m in front of the wall
        bodies=[
            *rails,
            Primitive(  # the Y axis: spans both rails, rides along them in x
                name="bridge",
                shape="box",
                size=(0.3, 1.8, 0.1),
                pose=Pose(xyz=(0, 0, 0.08)),
                attach="x_axis",
                kg=50,
            ),
            Primitive(
                name="carriage", shape="box", size=(0.4, 0.5, 0.12), pose=Pose(xyz=(0, 0, 0.18)), kg=50
            ),
            Primitive(
                name="riser", shape="cylinder", radius=0.15, height=0.48, pose=Pose(xyz=(0, 0, 0.30)), kg=40
            ),
        ],
        T_mount=Pose(xyz=(0.0, 0.0, 0.78)),
        joints=CarrierJoints(x=(-2.8, 2.8), y=(-0.35, 0.35)),
        standoff_m=Standoffs(carrier_part_min=0.3, carrier_part_preferred=0.4, carrier_env_min=0.1),
    )


def mobile_lift_base() -> Base:
    """The mobile base of mobile_amr with a 1 m vertical lift: a mast on the chassis (moves
    with the base) and a lift stage that rises with the robot."""
    b = mobile_base()
    bodies = [p for p in b.bodies if p.name != "column"]
    bodies += [
        Primitive(name="mast", shape="box", size=(0.25, 0.3, 0.3), pose=Pose(xyz=(0.2, 0, 0.4)), kg=60),
        Primitive(
            name="lift_stage",  # a 1 m column, inside the chassis and mast when lowered
            shape="cylinder",
            radius=0.11,
            height=1.0,
            pose=Pose(xyz=(0.2, 0, -0.3)),
            attach="lift",
            kg=20,
        ),
    ]
    return b.model_copy(
        update={
            "bodies": bodies,
            "keepouts": [],
            "joints": CarrierJoints(x=(-4.0, 4.0), y=(-3.0, 3.0), yaw_deg=(-180, 180), lift=(0.0, 1.0)),
        }
    )


def main() -> None:
    build("rail_long_part", rail_base(), {"s": -1.5})
    build("mobile_amr", mobile_base(), {"x": -1.5, "y": -0.2, "yaw": 0.0})
    wall = dict(part="tall_wall", pose=np.eye(4))
    build(  # hanging upside down, the spec home (arm "up") would point into the wall
        "gantry_overhead",
        gantry_base(),
        {"x": -1.0, "y": 0.0},
        faces=WALL_FACES_GANTRY,
        home_deg=[0.0, -165.0, -150.0, -180.0, -90.0, 0.0],  # folded towards the carriage
        **wall,
    )
    build(
        "floor_xy_rails",
        floor_xy_base(),
        {"x": -1.0, "y": 0.0},
        faces=(("side_-y", (0, -1, 0), 0.25, 1.8), ("end_+x", (1, 0, 0), 0.25, 1.8)),
        **wall,
    )
    build(
        "mobile_lift",
        mobile_lift_base(),
        {"x": -1.0, "y": -1.2, "yaw": np.pi / 2, "lift": 0.0},
        faces=(("side_-y", (0, -1, 0), 0.5, np.inf),),
        **wall,
    )


if __name__ == "__main__":
    main()
