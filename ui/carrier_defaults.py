"""External Axis tab -> *Load carrier defaults*: one ready carrier per axis type."""

import numpy as np

from core.schema import Base, CarrierJoints, Pose, Primitive, Standoffs

DEFAULTS = {
    # v7 round 2 (T13): the track, X rails and bridge follow the axis lengths; the lengths
    # give the same travel as before (rail +-3 m; gantry x +-1.5 m, y +-1 m)
    "rail": Base(
        mode="movable",
        carrier="rail",
        pose=Pose(xyz=(0, -0.8, 0)),
        bodies=[
            Primitive(name="track", shape="box", size=(6.6, 0.3, 0.1), attach="fixed", stretch="x"),
            Primitive(name="chassis", shape="box", size=(0.5, 0.5, 0.15)),
            Primitive(name="riser", shape="cylinder", radius=0.15, height=0.55, pose=Pose(xyz=(0, 0, 0.15))),
        ],
        axis_length_m={"x": 6.6},
        T_mount=Pose(xyz=(0, 0, 0.7)),
        joints=CarrierJoints(x=(-3.0, 3.0)),
        standoff_m=Standoffs(carrier_part_min=0.3, carrier_part_preferred=0.4),
    ),
    "gantry": Base(
        mode="movable",
        carrier="gantry",
        pose=Pose(xyz=(0, 0, 2.5)),
        bodies=[
            *[
                Primitive(
                    name=f"x_rail_{t}",
                    shape="box",
                    size=(3.5, 0.15, 0.15),
                    pose=Pose(xyz=(0, y, 0)),
                    attach="fixed",
                    stretch="x",
                )
                for t, y in (("a", -1.35), ("b", 1.35))
            ],
            Primitive(name="bridge", shape="box", size=(0.25, 2.5, 0.15), attach="x_axis", stretch="y"),
            Primitive(name="chassis", shape="box", size=(0.4, 0.4, 0.15)),
        ],
        axis_length_m={"x": 3.5, "y": 2.5},
        T_mount=Pose(rpy=(np.pi, 0, 0)),
        joints=CarrierJoints(x=(-1.5, 1.5), y=(-1.0, 1.0)),
        standoff_m=Standoffs(carrier_part_min=0.3, carrier_part_preferred=0.4),
    ),
    "mobile": Base(
        mode="movable",
        carrier="mobile",
        chassis=Primitive(shape="box", size=(1.2, 0.8, 0.4)),
        riser=Primitive(shape="cylinder", radius=0.15, height=0.3),
        T_mount=Pose(xyz=(0.2, 0, 0.7)),
        joints=CarrierJoints(x=(-4.0, 4.0), y=(-4.0, 4.0), yaw_deg=(-180, 180)),
        standoff_m=Standoffs(carrier_part_min=0.3, carrier_part_preferred=0.5),
        localization_m=0.01,
        localization_deg=1.0,
    ),
}
