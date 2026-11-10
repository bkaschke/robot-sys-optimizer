"""Project-file models shared by core.schema (the base model class, poses) and the robot base
/ carrier models (Base, Primitive, CarrierJoints, ...), split out of core.schema to keep that
file under 500 lines; core.schema re-exports everything here."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Vec3 = tuple[float, float, float]
Range = tuple[float, float]


# fields of older versions that no longer exist (payload, torque, mass, tip-over and process
# force were removed in v7 to keep the tool robot-agnostic); dropped silently on load
REMOVED_FIELDS = frozenset(
    {
        "payload_spec",
        "payload",
        "tip_over",
        "tip_margin_m",
        "kg",
        "kg_per_m",
        "process_force_N",
        "mass_kg",
        "com_head",
        "torque",
        "torque_util",
    }
)


def drop_removed(data):
    """``data`` without the keys in REMOVED_FIELDS (dict input only)."""
    if isinstance(data, dict) and REMOVED_FIELDS.intersection(data):
        return {k: v for k, v in data.items() if k not in REMOVED_FIELDS}
    return data


class _Model(BaseModel):
    """Base: reject unknown keys so typos in project files fail loudly (removed fields of
    older versions are dropped first)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def _drop_removed(cls, data):
        return drop_removed(data)


def _check_range(r: Range | None, name: str) -> None:
    if r is not None and r[0] > r[1]:
        raise ValueError(f"{name}: lower bound {r[0]} > upper bound {r[1]}")


class Pose(_Model):
    """Rigid transform as translation (m) + roll/pitch/yaw (rad, fixed-axis XYZ)."""

    xyz: Vec3 = (0.0, 0.0, 0.0)
    rpy: Vec3 = (0.0, 0.0, 0.0)

    @field_validator("xyz", "rpy")
    @classmethod
    def _finite(cls, v: Vec3) -> Vec3:
        if not all(math.isfinite(x) for x in v):
            raise ValueError("pose values must be finite")
        return v


# --- Base / carrier ------------------------------------------------------------


class Primitive(_Model):
    """One carrier body: a box, a cylinder, or an STL/OBJ/PLY mesh.

    Box and cylinder frames sit at the centre of the bottom face; a mesh keeps its own
    origin (scaled by ``units``). ``pose`` places the body frame in its parent frame:
    the carriage frame (``attach="carriage"``, moves with the carrier DOFs), the lift stage
    (``attach="lift"``: the carriage frame raised by the lift DOF, e.g. a platform that rises
    with the robot) or the carrier frame ``Base.pose`` (``attach="fixed"``, e.g. a rail track
    or a gantry beam) or the X axis (``attach="x_axis"``: a gantry bridge across the X rails
    that moves in x but not in y; the carriage runs along it. Rails and mobile bases treat it
    as the carriage).
    """

    shape: Literal["box", "cylinder", "mesh"]
    size: Vec3 | None = None  # box: L x W x H (m)
    radius: float | None = Field(default=None, gt=0)
    height: float | None = Field(default=None, gt=0)
    name: str = Field(default="", max_length=64)
    attach: Literal["carriage", "lift", "fixed", "x_axis"] = "carriage"
    pose: Pose = Field(default_factory=Pose)
    mesh: str | None = None  # mesh path, relative to the project folder or the repo
    units: Literal["m", "mm", "in"] = "m"
    # v8 T18 (stay-out zones only): a speed in % makes it a reduced-speed zone (the arm may
    # enter, slowed to this share of its speed: cycle time) instead of a hard stay-out
    speed_pct: float | None = Field(default=None, gt=0, le=100)
    # v7 round 2 (T13): this body follows an axis length (Base.axis_length_m): a box gets that
    # size along the axis, a cylinder lying along it that height; "" = fixed size
    stretch: Literal["", "x", "y"] = ""

    @model_validator(mode="after")
    def _dims(self) -> Primitive:
        if self.shape == "box" and self.size is None:
            raise ValueError("box primitive needs 'size'")
        if self.shape == "cylinder" and (self.radius is None or self.height is None):
            raise ValueError("cylinder primitive needs 'radius' and 'height'")
        if self.shape == "mesh" and not self.mesh:
            raise ValueError("mesh body needs a 'mesh' file")
        if self.size is not None and any(s <= 0 for s in self.size):
            raise ValueError("primitive size must be positive")
        return self


class CarrierJoints(_Model):
    x: Range | None = None
    y: Range | None = None
    z: Range | None = None
    yaw_deg: Range | None = None
    lift: Range | None = None

    @model_validator(mode="after")
    def _ranges(self) -> CarrierJoints:
        for name in ("x", "y", "z", "yaw_deg", "lift"):
            _check_range(getattr(self, name), name)
        return self


class Keepout(_Model):
    """Axis-aligned floor box (world frame) the carrier must stay out of."""

    min_xy: tuple[float, float]
    max_xy: tuple[float, float]


class Standoffs(_Model):
    carrier_part_min: float = Field(default=0.0, ge=0)
    carrier_part_preferred: float | None = Field(default=None, ge=0)
    carrier_env_min: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _order(self) -> Standoffs:
        p = self.carrier_part_preferred
        if p is not None and p < self.carrier_part_min:
            raise ValueError("preferred standoff must be >= minimum standoff")
        return self


class Base(_Model):
    """Robot base: static, or carried by a rail / gantry / mobile base (station mode).

    Frames: ``pose`` is the carrier frame in the world (rail/gantry: the track frame;
    mobile: ignored, the chassis pose is the DOF). Carrier DOFs (``joints`` ranges):
      rail    s   -> x along the track frame's x axis
      gantry  x,y -> carriage position in the gantry frame (mount usually inverted)
      mobile  x,y,yaw -> chassis pose on the floor (world)
    T_world_base = T_world_carriage(dofs) @ Tz(lift) @ T_mount.
    ``bodies`` is the carrier built from simple shapes or meshes (see Primitive). When it is
    empty, the legacy fields are used instead: ``chassis`` is the moving body (bottom at the
    carriage frame origin), ``riser`` sits on top of it, and ``track`` is the static rail
    body, centred on the travel range (core.carrier.body_list converts them).
    """

    mode: Literal["static", "movable"] = "static"
    carrier: Literal["pedestal", "rail", "gantry", "mobile", "positioner"] | None = None
    pose: Pose = Field(default_factory=Pose)
    chassis: Primitive | None = None
    riser: Primitive | None = None
    track: Primitive | None = None
    bodies: list[Primitive] = Field(default_factory=list, max_length=64)
    joints: CarrierJoints = Field(default_factory=CarrierJoints)
    T_mount: Pose = Field(default_factory=Pose)
    keepouts: list[Keepout] = Field(default_factory=list)
    standoff_m: Standoffs = Field(default_factory=Standoffs)
    localization_m: float = Field(default=0.0, ge=0)  # mobile: position uncertainty
    localization_deg: float = Field(default=0.0, ge=0)  # mobile: heading uncertainty
    # mobile, v2: the side of the base that faces the part and ONE fixed gap between the base
    # outline and the part outline (top view). Both set: stops lie on that contour around the
    # part with the heading given (opt.contour); either None: free x, y, yaw search (v1).
    facing: Literal["front", "back", "left", "right"] | None = None
    fixed_standoff_m: float | None = Field(default=None, ge=0)
    # v3 feedback: several robot sides may face the part (the planner picks per stop; empty:
    # ``facing``), the part sides (part frame) the mobile base may work from (empty: all), and
    # a rail / gantry track kept along the part's long axis
    facings: list[Literal["front", "back", "left", "right"]] = Field(default_factory=list)
    part_sides: list[Literal["+x", "-x", "+y", "-y"]] = Field(default_factory=list)
    align_to_part: bool = False
    # v7 round 2 (T13): rail / gantry axis lengths in one setting (keys "x": rail or X rails,
    # "y": gantry bridge). Bodies with ``stretch`` take that length, and unless
    # ``manual_travel`` the travel range follows: +-(L/2 - carriage extent - end_stop_m)
    # (core.carrier_axis.apply_axis_lengths writes it into ``joints``)
    axis_length_m: dict[Literal["x", "y"], float] = Field(default_factory=dict)
    end_stop_m: float = Field(default=0.05, ge=0)
    manual_travel: bool = False

    @field_validator("axis_length_m")
    @classmethod
    def _lengths(cls, v: dict) -> dict:
        if any(not (math.isfinite(x) and x > 0) for x in v.values()):
            raise ValueError("axis lengths must be positive")
        return v

    def robot_sides(self) -> list[str]:
        """Robot sides allowed to face the part (mobile contour mode)."""
        return list(dict.fromkeys(self.facings)) or ([self.facing] if self.facing else [])

    @model_validator(mode="after")
    def _carrier(self) -> Base:
        if self.mode == "movable" and self.carrier in (None, "pedestal"):
            raise ValueError("movable base needs carrier: rail, gantry, mobile or positioner")
        if self.mode == "movable" and self.carrier == "positioner":
            raise ValueError("part positioners are not supported yet")
        j = self.joints
        need = {"rail": ("x",), "gantry": ("x", "y"), "mobile": ("x", "y")}.get(self.carrier or "", ())
        if self.mode == "movable":
            missing = [k for k in need if getattr(j, k) is None]
            if missing:
                raise ValueError(f"{self.carrier} needs travel ranges for {missing}")
        return self
