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

from pydantic import Field, field_validator, model_validator

from core.schema_base import (  # noqa: F401 (re-exported)
    REMOVED_FIELDS,
    Base,
    CarrierJoints,
    Keepout,
    Pose,
    Primitive,
    Range,
    Standoffs,
    Vec3,
    _check_range,
    _Model,
    drop_removed,
)
from core.schema_work import PointResultData, WorkState, Zone  # noqa: F401 (re-exported)

SCHEMA_VERSION = 1

# --- Robot -------------------------------------------------------------------


class Robot(_Model):
    name: str | None = None  # registry name, e.g. "ur20"; preferred over a raw urdf path
    urdf: str | None = None
    flange_link: str = "tool0"
    mount: Literal["floor", "wall", "ceiling", "custom"] = "floor"
    base_pose: Pose = Field(default_factory=Pose)
    ik: Literal["ur_analytic", "opw", "numeric"] = "numeric"
    # home pose saved with the project (deg, one per joint); None = the robot's spec.yaml home
    home_deg: list[float] | None = None
    # joint limits saved with the project (deg, (min, max) per joint); None = the URDF limits
    joint_limits_deg: list[tuple[float, float]] | None = None

    @model_validator(mode="after")
    def _source(self) -> Robot:
        if self.name is None and self.urdf is None:
            raise ValueError("robot needs a registry 'name' or a 'urdf' path")
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
    standoff: float = Field(default=1.0, ge=0)  # soft: carrier closer than the preferred standoff
    group: float = Field(default=1.0, ge=0)  # per extra joint-configuration group at a stop
    motion: float = Field(default=1.0, ge=0)  # mean largest joint move between neighbouring points / 90 deg
    wrist_flip: float = Field(default=1.0, ge=0)  # per joint 4 / 6 jump > 180 deg inside a group
    cycle: float = Field(default=0.0, ge=0)  # v8 T6: per minute of estimated cycle time

    @model_validator(mode="before")
    @classmethod
    def _legacy_config(cls, data):
        """Projects before v5 had one ``config`` weight for both counts."""
        data = drop_removed(data)
        if isinstance(data, dict) and "config" in data:
            data = dict(data)
            c = data.pop("config")
            data.setdefault("group", c)
            data.setdefault("wrist_flip", c)
        return data


class Margins(_Model):
    clearance_m: float = Field(default=0.01, ge=0)
    joint_deg: float = Field(default=5.0, ge=0)


FreeVar = Literal["L", "theta", "d", "roll", "base", "stations"]


class Optimizer(_Model):
    free: list[FreeVar] = Field(default_factory=lambda: ["L", "base"])
    max_stations: int = Field(default=4, ge=1, le=40)  # movable bases; static mode plans one base
    weights: Weights = Field(default_factory=Weights)
    margins: Margins = Field(default_factory=Margins)
    tool_weights: Weights | None = None  # the tool-geometry optimizer's own costs (None: weights)


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
    split_stops: list[int] = Field(default_factory=list)  # stops whose area is in pieces (opt.areas)
    violations: list[str] = Field(default_factory=list)  # broken "must be met" rules: not ranked
    # v7 round 2: which planner made it ("cost", "quick", "rules: fewest stations", "tool", ...)
    # and how long that run took (s); shown on the Results summary card
    planner: str = ""
    run_s: float | None = None
    cycle_s: float | None = None  # v8 T6: estimated cycle time (core.cycle_time.plan_estimate)

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
    robot: Robot | None = None  # None: an empty study, robot not chosen yet
    base: Base = Field(default_factory=Base)
    # stay-out zones (world poses; core.stayout): the arm, tool and carrier keep clear of them
    stayouts: list[Primitive] = Field(default_factory=list, max_length=64)
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
