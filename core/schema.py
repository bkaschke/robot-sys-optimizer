"""Pydantic models for the project file and the tool library.

One JSON project file is the source of truth (see docs/PROJECT_PLAN.md, "Data model").
Internal units are meters and radians; fields ending in ``_deg`` are degrees because
they are user-facing tolerances, and are converted at the point of use.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.schema_work import PointResultData, WorkState, Zone  # noqa: F401 (re-exported)

SCHEMA_VERSION = 1

Vec3 = tuple[float, float, float]
Range = tuple[float, float]


class _Model(BaseModel):
    """Base: reject unknown keys so typos in project files fail loudly."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


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


# --- Robot -------------------------------------------------------------------


class PayloadSpec(_Model):
    """Datasheet limits. ``None`` means unknown: payload checks must refuse to run.

    ``curve_mm_kg`` is the manufacturer's payload curve: allowed payload (kg) against the
    payload COG distance from the flange (mm), as (offset, kg) points with increasing
    offset. Past the last point it is either refused ("none") or extrapolated linearly
    from the last segment ("linear", an estimate) down to 0 kg.
    """

    rated_kg: float | None = Field(default=None, gt=0)
    wrist_moment_Nm: list[float | None] = Field(default_factory=lambda: [None, None, None])
    joint_torque_Nm: list[float | None] = Field(default_factory=list)
    curve_mm_kg: list[tuple[float, float]] | None = None
    curve_extrapolate: Literal["none", "linear"] = "none"
    curve_source: str = ""

    @field_validator("curve_mm_kg")
    @classmethod
    def _curve(cls, v):
        if v is None:
            return v
        if len(v) < 2:
            raise ValueError("payload curve needs at least 2 points")
        xs = [x for x, _ in v]
        if any(b <= a for a, b in zip(xs, xs[1:], strict=False)) or xs[0] < 0:
            raise ValueError("payload curve offsets must start >= 0 and increase")
        if any(kg < 0 for _, kg in v):
            raise ValueError("payload curve masses must be >= 0")
        return v

    def allowed_kg(self, offset_mm: float) -> tuple[float | None, bool]:
        """(allowed payload at this COG offset, whether it is extrapolated). None = unknown."""
        if self.curve_mm_kg is None:
            return None, False
        xs = [x for x, _ in self.curve_mm_kg]
        ys = [y for _, y in self.curve_mm_kg]
        if offset_mm <= xs[-1]:
            import numpy as np

            return float(np.interp(offset_mm, xs, ys)), False
        if self.curve_extrapolate == "none":
            return 0.0, False
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return max(0.0, ys[-1] + slope * (offset_mm - xs[-1])), True

    def is_complete(self) -> bool:
        """Rated payload, every joint torque, and a payload curve or all wrist moments."""
        return (
            self.rated_kg is not None
            and len(self.joint_torque_Nm) > 0
            and all(t is not None for t in self.joint_torque_Nm)
            and (self.curve_mm_kg is not None or all(m is not None for m in self.wrist_moment_Nm))
        )


class Robot(_Model):
    name: str | None = None  # registry name, e.g. "ur20"; preferred over a raw urdf path
    urdf: str | None = None
    flange_link: str = "tool0"
    mount: Literal["floor", "wall", "ceiling", "custom"] = "floor"
    base_pose: Pose = Field(default_factory=Pose)
    ik: Literal["ur_analytic", "opw", "numeric"] = "numeric"
    payload_spec: PayloadSpec = Field(default_factory=PayloadSpec)

    @model_validator(mode="after")
    def _source(self) -> Robot:
        if self.name is None and self.urdf is None:
            raise ValueError("robot needs a registry 'name' or a 'urdf' path")
        return self


# --- Base / carrier ------------------------------------------------------------


class Primitive(_Model):
    """One carrier body: a box, a cylinder, or an STL/OBJ/PLY mesh.

    Box and cylinder frames sit at the centre of the bottom face; a mesh keeps its own
    origin (scaled by ``units``). ``pose`` places the body frame in its parent frame:
    the carriage frame (``attach="carriage"``, moves with the carrier DOFs) or the carrier
    frame ``Base.pose`` (``attach="fixed"``, e.g. a rail track or a gantry beam).
    """

    shape: Literal["box", "cylinder", "mesh"]
    size: Vec3 | None = None  # box: L x W x H (m)
    radius: float | None = Field(default=None, gt=0)
    height: float | None = Field(default=None, gt=0)
    kg: float | None = Field(default=None, ge=0)
    name: str = Field(default="", max_length=64)
    attach: Literal["carriage", "fixed"] = "carriage"
    pose: Pose = Field(default_factory=Pose)
    mesh: str | None = None  # mesh path, relative to the project folder or the repo
    units: Literal["m", "mm", "in"] = "m"

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
    tip_over: bool = False  # mobile: static tip-over check (needs chassis kg)
    tip_margin_m: float = Field(default=0.05, ge=0)
    # mobile, v2: the side of the base that faces the part and ONE fixed gap between the base
    # outline and the part outline (top view). Both set: stops lie on that contour around the
    # part with the heading given (opt.contour); either None: free x, y, yaw search (v1).
    facing: Literal["front", "back", "left", "right"] | None = None
    fixed_standoff_m: float | None = Field(default=None, ge=0)

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
        if self.tip_over:
            moving = [b for b in self.bodies if b.attach == "carriage"] or [
                b for b in (self.chassis, self.riser) if b is not None
            ]
            if not any(b.kg for b in moving):
                raise ValueError("tip-over check needs the chassis mass")
        return self


# --- End effector ---------------------------------------------------------------


class ExtensionValues(_Model):
    """One concrete extension geometry (the rough placement, or an optimizer result).

    Legacy straight / bent tools use L, theta, d, roll. Pose tools (v2) use the head pose
    relative to the mount: x, y, z (m) and rx, ry, rz (deg, roll-pitch-yaw)."""

    L: float = Field(default=0.1, ge=0)
    theta_deg: float = 0.0
    d: float = 0.0
    roll_deg: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.1
    rx_deg: float = 0.0
    ry_deg: float = 0.0
    rz_deg: float = 0.0


POSE_KEYS = ("x", "y", "z", "rx_deg", "ry_deg", "rz_deg")
LEGACY_KEYS = ("L", "theta_deg", "d", "roll_deg")


class Extension(_Model):
    """Parametric arm between the mount adapter and the tool head.

    Geometry (see core/tool.py). ``pose`` (v2): T_mount_head = translation (x, y, z) and
    roll-pitch-yaw (rx, ry, rz), each within its range; the connector is a smooth spline
    from the mount to the head. Legacy ``straight`` / ``bent``: T_mount_head = Rz(roll) Tz(L)
    Rx(theta) Ty(d) (a straight extension forces theta = d = 0).
    Ranges bound the optimizer (min = max fixes a value); ``nominal`` is the rough placement.
    """

    type: Literal["straight", "bent", "pose"] = "straight"
    L: Range = (0.0, 0.5)
    theta_deg: Range = (0.0, 0.0)
    d: Range = (0.0, 0.0)
    roll_deg: Range = (0.0, 0.0)
    x: Range = (0.0, 0.0)
    y: Range = (0.0, 0.0)
    z: Range = (0.05, 0.5)
    rx_deg: Range = (0.0, 0.0)
    ry_deg: Range = (0.0, 0.0)
    rz_deg: Range = (0.0, 0.0)
    radius: float = Field(default=0.04, gt=0)
    kg_per_m: float = Field(default=0.0, ge=0)
    nominal: ExtensionValues = Field(default_factory=ExtensionValues)

    @model_validator(mode="after")
    def _ranges(self) -> Extension:
        for name in (*LEGACY_KEYS, *POSE_KEYS):
            _check_range(getattr(self, name), name)
        if self.L[0] < 0:
            raise ValueError("extension length must be >= 0")
        if self.type == "straight" and (self.theta_deg != (0.0, 0.0) or self.d != (0.0, 0.0)):
            raise ValueError("a straight extension has theta and d fixed at 0")
        n = self.nominal
        for name in POSE_KEYS if self.type == "pose" else LEGACY_KEYS:
            lo, hi = getattr(self, name)
            v = getattr(n, name)
            if not lo - 1e-9 <= v <= hi + 1e-9:
                raise ValueError(f"nominal {name}={v} outside its range [{lo}, {hi}]")
        return self


def length_ref(x: Extension) -> float:
    """Reference length for the cost's tool-length term: the longest extension the ranges
    allow (legacy: L max; pose: the farthest head position)."""
    if x.type != "pose":
        return max(x.L[1], 1e-6)
    corners = [(a, b, c) for a in x.x for b in x.y for c in x.z]
    return max(float(max((a * a + b * b + c * c) ** 0.5 for a, b, c in corners)), 1e-6)


class TCP(_Model):
    name: str
    T_head_tcp: Pose = Field(default_factory=Pose)
    contact_links: list[str] = Field(default_factory=list)  # whole links allowed to touch
    contact_faces: dict[str, list[int]] = Field(default_factory=dict)  # link -> face ids

    @field_validator("contact_faces")
    @classmethod
    def _faces(cls, v: dict[str, list[int]]) -> dict[str, list[int]]:
        for link, faces in v.items():
            if any(f < 0 for f in faces):
                raise ValueError(f"negative face index on {link}")
        return {k: sorted(set(f)) for k, f in v.items()}


ProcessType = Literal["scanning", "sanding", "grinding", "gripping", "spraying", "custom"]


class Tolerances(_Model):
    """Per-target freedom; a process preset supplies defaults for new target groups."""

    spin_deg: Literal["free"] | Annotated[float, Field(ge=0)] | Range = "free"
    tilt_deg: float = Field(default=0.0, ge=0, le=90)
    standoff_m: float | Range = 0.0


class EndEffector(_Model):
    """A tool-library entry (tools/<name>.yaml) and the project's chosen tool."""

    name: str = "tool"
    tool: str | None = None  # path to the tool-library YAML this was loaded from
    process: ProcessType = "custom"
    urdf: str | None = None
    T_flange_mount: Pose = Field(default_factory=Pose)
    extension: Extension = Field(default_factory=Extension)
    tcps: list[TCP] = Field(min_length=1)
    process_force_N: float = Field(default=0.0, ge=0)
    mass_kg: float | None = Field(default=None, ge=0)  # override of the URDF head mass
    com_head: Vec3 | None = None  # override of the URDF head COM (head frame, m)
    tolerances: Tolerances = Field(default_factory=Tolerances)
    line_of_sight: bool = False  # scanning: require a clear ray from sensor to point
    approach_m: float = Field(default=0.0, ge=0)  # gripping: approach/retreat distance

    @field_validator("tcps")
    @classmethod
    def _unique_tcps(cls, v: list[TCP]) -> list[TCP]:
        names = [t.name for t in v]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate TCP names: {names}")
        return v


# --- Part and targets ----------------------------------------------------------


class Part(_Model):
    mesh: str
    units: Literal["m", "mm", "in"] = "m"
    pose: Pose = Field(default_factory=Pose)


SpinTolerance = Literal["free"] | Annotated[float, Field(ge=0)] | Range


class TargetGroup(_Model):
    name: str
    spin_deg: SpinTolerance = "free"  # "free", +-half-width, or (lo, hi)
    tilt_deg: float = Field(default=0.0, ge=0, le=90)
    standoff_m: float | Range = 0.0
    tcps: list[str] = Field(default_factory=list)  # empty = any TCP


class TargetPoint(_Model):
    id: int = Field(ge=0)
    p: Vec3  # position in the part frame (m)
    n: Vec3  # outward surface normal in the part frame
    group: str

    @field_validator("n")
    @classmethod
    def _unit_normal(cls, n: Vec3) -> Vec3:
        norm = math.sqrt(sum(c * c for c in n))
        if not math.isfinite(norm) or norm < 1e-9:
            raise ValueError("normal must be non-zero and finite")
        return (n[0] / norm, n[1] / norm, n[2] / norm)


class Targets(_Model):
    groups: list[TargetGroup] = Field(default_factory=list)
    points: list[TargetPoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def _refs(self) -> Targets:
        names = {g.name for g in self.groups}
        if len(names) != len(self.groups):
            raise ValueError("duplicate target group names")
        ids = [p.id for p in self.points]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate target point ids")
        missing = {p.group for p in self.points} - names
        if missing:
            raise ValueError(f"points reference unknown groups: {sorted(missing)}")
        return self


# --- Optimizer -----------------------------------------------------------------


class Weights(_Model):
    L: float = Field(default=1.0, ge=0)
    unreached: float = Field(default=1000.0, ge=0)
    stations: float = Field(default=5.0, ge=0)
    travel: float = Field(default=0.5, ge=0)
    manip: float = Field(default=0.3, ge=0)
    clearance: float = Field(default=0.3, ge=0)
    limits: float = Field(default=0.2, ge=0)
    torque: float = Field(default=0.5, ge=0)
    standoff: float = Field(default=1.0, ge=0)  # soft: carrier closer than the preferred standoff
    config: float = Field(default=1.0, ge=0)  # per joint-configuration change (zoning)


class Margins(_Model):
    clearance_m: float = Field(default=0.01, ge=0)
    joint_deg: float = Field(default=5.0, ge=0)
    torque_util: float = Field(default=0.8, gt=0, le=1)


FreeVar = Literal["L", "theta", "d", "roll", "base", "stations"]


class Optimizer(_Model):
    free: list[FreeVar] = Field(default_factory=lambda: ["L", "base"])
    max_stations: int = Field(default=4, ge=1)  # movable bases; static mode always plans one base
    weights: Weights = Field(default_factory=Weights)
    margins: Margins = Field(default_factory=Margins)


# --- Results -------------------------------------------------------------------


class Station(_Model):
    """One base stop: base_link position (world) and yaw, plus the carrier DOF values."""

    xyz: Vec3
    yaw: float = 0.0
    carrier: dict[str, float] = Field(default_factory=dict)


class Solution(_Model):
    """A placement result: stations in visiting order and which station serves each point."""

    label: str = ""
    stations: list[Station] = Field(min_length=1)
    assignment: dict[int, int] = Field(default_factory=dict)  # point id -> station index
    unreached: list[int] = Field(default_factory=list)
    cost: float = 0.0
    breakdown: dict[str, float] = Field(default_factory=dict)
    tool_values: ExtensionValues | None = None
    zones: list[Zone] = Field(default_factory=list)  # joint-configuration zones (zoning on)

    @model_validator(mode="after")
    def _station_refs(self) -> Solution:
        k = len(self.stations)
        bad = [i for i, s in self.assignment.items() if not 0 <= s < k]
        if bad:
            raise ValueError(f"points {bad[:5]} assigned to unknown stations")
        return self


# --- Project -------------------------------------------------------------------


class Project(_Model):
    version: int = SCHEMA_VERSION
    units: Literal["m"] = "m"
    robot: Robot
    base: Base = Field(default_factory=Base)
    end_effector: EndEffector | None = None  # set up in the Tool Setup tab (M3)
    part: Part | None = None
    targets: Targets = Field(default_factory=Targets)
    optimizer: Optimizer = Field(default_factory=Optimizer)
    results: list[Solution] = Field(default_factory=list)
    work: WorkState = Field(default_factory=WorkState)

    @field_validator("version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != SCHEMA_VERSION:
            raise ValueError(f"unsupported project version {v}; expected {SCHEMA_VERSION}")
        return v

    @model_validator(mode="after")
    def _tcp_refs(self) -> Project:
        known = {t.name for t in self.end_effector.tcps} if self.end_effector else set()
        for g in self.targets.groups:
            unknown = set(g.tcps) - known
            if unknown:
                raise ValueError(f"group '{g.name}' references unknown TCPs: {sorted(unknown)}")
        return self


def load_project(path: str | Path) -> Project:
    return Project.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_project(project: Project, path: str | Path) -> None:
    data = project.model_dump(mode="json")
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
