"""Robot registry and Pinocchio-backed kinematics.

Robots live in ``assets/robots/<name>/`` (URDF + meshes + ``spec.yaml``). Code looks them
up by name and never hard-codes a model.

All poses are 4x4 homogeneous matrices expressed in the robot's ``base_link`` frame.
Jacobians are 6 x nq, rows (linear; angular), base-aligned (Pinocchio
LOCAL_WORLD_ALIGNED) and taken at a point fixed to the flange.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pinocchio as pin
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.schema import drop_removed

ROBOTS_DIR = Path(__file__).resolve().parents[1] / "assets" / "robots"


class DHParams(BaseModel):
    """Standard DH parameters of a UR-family arm (m). ``frame`` = URDF link at DH frame 0."""

    model_config = ConfigDict(extra="forbid")
    frame: str
    d1: float
    a2: float
    a3: float
    d4: float
    d5: float
    d6: float


class OPWParams(BaseModel):
    """OPW parameters (m, rad) of an industrial 6R arm, as in ROS-Industrial opw_kinematics
    parameter files: q_i = (theta_i + offsets_i) * sign_corrections_i (core.ik.opw)."""

    model_config = ConfigDict(extra="forbid")
    a1: float
    a2: float
    b: float = 0.0
    c1: float
    c2: float
    c3: float
    c4: float
    offsets: tuple[float, float, float, float, float, float] = (0.0,) * 6
    sign_corrections: tuple[int, int, int, int, int, int] = (1,) * 6


class RobotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _drop_removed(cls, data):
        """Older spec.yaml files carry a ``payload`` section (removed in v7)."""
        return drop_removed(data)

    name: str
    urdf: str
    base_link: str
    flange_link: str
    ik: Literal["ur_analytic", "opw", "numeric"]
    reach_m: float | None = Field(default=None, gt=0)
    # how the controller writes a frame (userframe export): "ur" = x y z + rotation vector,
    # "fanuc" = X Y Z W P R, "matrix" = the 4 x 4 transform only (always added as well)
    frame_convention: Literal["ur", "fanuc", "matrix"] = "matrix"
    # joint angles (deg) the arm waits at while the carrier moves between stops; None = all
    # joints at 0 (clipped into the limits)
    home_deg: list[float] | None = None
    # user joint limits (deg, (min, max) per joint) inside the URDF's; None = the URDF limits
    limits_deg: list[tuple[float, float]] | None = None
    dh: DHParams | None = None  # ik: ur_analytic
    opw: OPWParams | None = None  # ik: opw
    folder: Path | None = None  # filled in by load_spec


def list_robots() -> list[str]:
    return sorted(p.parent.name for p in ROBOTS_DIR.glob("*/spec.yaml"))


def load_spec(name: str) -> RobotSpec:
    path = ROBOTS_DIR / name / "spec.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"robot '{name}' not in registry ({ROBOTS_DIR}); have {list_robots()}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec = RobotSpec.model_validate(data)
    spec.folder = path.parent
    return spec


def _se3_to_mat(m: pin.SE3) -> np.ndarray:
    return np.asarray(m.homogeneous)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class Robot:
    """A registry robot loaded into Pinocchio. Not thread-safe (owns one pin.Data)."""

    def __init__(self, spec: RobotSpec):
        self.spec = spec
        urdf_path = spec.folder / spec.urdf
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        if self.model.nq != self.model.nv:
            raise ValueError(
                f"{spec.name}: continuous/floating joints are not supported yet; give them limits"
            )
        self.nq = self.model.nq
        self.joint_names = [n for n in self.model.names[1:]]
        self.urdf_lower = np.array(self.model.lowerPositionLimit, dtype=float)
        self.urdf_upper = np.array(self.model.upperPositionLimit, dtype=float)
        self.lower, self.upper = self.urdf_lower.copy(), self.urdf_upper.copy()
        self.set_limits(spec.limits_deg)
        self.base_frame_id = self._frame_id(spec.base_link)
        self.flange_frame_id = self._frame_id(spec.flange_link)

    @classmethod
    def from_registry(cls, name: str) -> Robot:
        return cls(load_spec(name))

    @classmethod
    def from_urdf(cls, urdf_path: Path, base_link: str, flange_link: str, name: str | None = None) -> Robot:
        """A non-registry robot (e.g. a user upload): numeric IK."""
        urdf_path = Path(urdf_path).resolve()
        spec = RobotSpec(
            name=name or urdf_path.stem,
            urdf=urdf_path.name,
            base_link=base_link,
            flange_link=flange_link,
            ik="numeric",
            folder=urdf_path.parent,
        )
        return cls(spec)

    def _frame_id(self, name: str) -> int:
        if not self.model.existFrame(name):
            raise ValueError(f"frame '{name}' not in URDF of {self.spec.name}")
        return self.model.getFrameId(name)

    def set_limits(self, limits_deg: list | None) -> None:
        """Joint limits (deg, (min, max) per joint), kept inside the URDF limits; None = the
        URDF limits. Stored in ``spec`` so ``Robot(robot.spec)`` copies keep them."""
        lo, hi = self.urdf_lower.copy(), self.urdf_upper.copy()
        if limits_deg is not None:
            L = np.radians(np.asarray(limits_deg, float))
            if L.shape != (self.nq, 2) or not np.all(np.isfinite(L)):
                raise ValueError(f"joint limits need (min, max) for each of the {self.nq} joints")
            lo, hi = np.maximum(lo, L[:, 0]), np.minimum(hi, L[:, 1])
            bad = [self.joint_names[i] for i in np.flatnonzero(hi <= lo)]
            if bad:
                raise ValueError(f"empty joint range (min >= max inside the URDF limits): {', '.join(bad)}")
            limits_deg = [(float(a), float(b)) for a, b in np.asarray(limits_deg, float)]
        self.lower, self.upper = lo, hi
        self.model.lowerPositionLimit, self.model.upperPositionLimit = lo.copy(), hi.copy()
        self.spec = self.spec.model_copy(update={"limits_deg": limits_deg})

    def _check_q(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape != (self.nq,) or not np.all(np.isfinite(q)):
            raise ValueError(f"q must be a finite vector of length {self.nq}")
        return q

    # --- kinematics ---------------------------------------------------------

    def home_q(self) -> np.ndarray:
        """Home joints (spec ``home_deg``, else all 0), clipped into the joint limits."""
        h = self.spec.home_deg
        q = np.radians(np.asarray(h, float)) if h is not None and len(h) == self.nq else np.zeros(self.nq)
        return np.clip(q, self.lower, self.upper)

    def random_q(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lower, self.upper)

    def within_limits(self, q, margin: float = 0.0) -> bool:
        q = np.asarray(q)
        return bool(np.all(q >= self.lower + margin) and np.all(q <= self.upper - margin))

    def frame_pose(self, q, frame: str) -> np.ndarray:
        """Pose of any URDF frame in base_link."""
        q = self._check_q(q)
        pin.framesForwardKinematics(self.model, self.data, q)
        oMb = self.data.oMf[self.base_frame_id]
        return _se3_to_mat(oMb.actInv(self.data.oMf[self._frame_id(frame)]))

    def fk(self, q) -> np.ndarray:
        """T_base_flange."""
        q = self._check_q(q)
        pin.framesForwardKinematics(self.model, self.data, q)
        return _se3_to_mat(self.data.oMf[self.base_frame_id].actInv(self.data.oMf[self.flange_frame_id]))

    def jacobian(self, q, offset_flange=None) -> np.ndarray:
        """6 x nq Jacobian of a point fixed in the flange frame, base-aligned."""
        q = self._check_q(q)
        J = pin.computeFrameJacobian(
            self.model, self.data, q, self.flange_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        ).copy()
        R_bw = self.data.oMf[self.base_frame_id].rotation.T
        J[:3], J[3:] = R_bw @ J[:3], R_bw @ J[3:]
        if offset_flange is not None:
            R_wf = self.data.oMf[self.flange_frame_id].rotation
            r = R_bw @ R_wf @ np.asarray(offset_flange, float)  # flange origin -> point, in base
            J[:3] -= _skew(r) @ J[3:]
        return J

    def manipulability(self, q, offset_flange=None) -> float:
        """Yoshikawa measure sqrt(det(J J^T))."""
        J = self.jacobian(q, offset_flange)
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
