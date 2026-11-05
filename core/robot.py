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
from pydantic import BaseModel, ConfigDict, Field

from core.schema import PayloadSpec

ROBOTS_DIR = Path(__file__).resolve().parents[1] / "assets" / "robots"
GRAVITY = 9.81


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


class RobotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    urdf: str
    base_link: str
    flange_link: str
    ik: Literal["ur_analytic", "opw", "numeric"]
    reach_m: float | None = Field(default=None, gt=0)
    dh: DHParams | None = None
    payload: PayloadSpec = Field(default_factory=PayloadSpec)
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

    def __init__(self, spec: RobotSpec, gravity_base: np.ndarray | None = None):
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
        self.lower = np.array(self.model.lowerPositionLimit, dtype=float)
        self.upper = np.array(self.model.upperPositionLimit, dtype=float)
        self.effort = np.array(self.model.effortLimit, dtype=float)
        self.base_frame_id = self._frame_id(spec.base_link)
        self.flange_frame_id = self._frame_id(spec.flange_link)
        # gravity in the base_link frame: (0, 0, -g) for a floor mount; walls/ceilings differ
        g = np.array([0, 0, -GRAVITY]) if gravity_base is None else np.asarray(gravity_base, float)
        self.model.gravity.linear = self._base_rotation_world() @ g

    @classmethod
    def from_registry(cls, name: str, gravity_base: np.ndarray | None = None) -> Robot:
        return cls(load_spec(name), gravity_base)

    @classmethod
    def from_urdf(cls, urdf_path: Path, base_link: str, flange_link: str, name: str | None = None) -> Robot:
        """A non-registry robot (e.g. a user upload): numeric IK, unknown payload limits."""
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

    def _base_rotation_world(self) -> np.ndarray:
        """Rotation world<-base_link of the URDF's own root (identity for the UR files)."""
        pin.framesForwardKinematics(self.model, self.data, pin.neutral(self.model))
        return np.asarray(self.data.oMf[self.base_frame_id].rotation)

    def _check_q(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape != (self.nq,) or not np.all(np.isfinite(q)):
            raise ValueError(f"q must be a finite vector of length {self.nq}")
        return q

    # --- kinematics ---------------------------------------------------------

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

    # --- statics ------------------------------------------------------------

    def gravity_torques(self, q) -> np.ndarray:
        """Joint torques that hold the bare arm against gravity."""
        return np.array(pin.computeGeneralizedGravity(self.model, self.data, self._check_q(q)))

    def static_torques(
        self,
        q,
        payload_kg: float = 0.0,
        payload_com=(0.0, 0.0, 0.0),
        wrench=None,
        wrench_point=(0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """tau = g_arm(q) + J_com^T (m * -g) + J_p^T F.

        payload_com and wrench_point are in the flange frame. ``wrench`` (fx fy fz mx my mz,
        base-aligned) is what the tool exerts ON the environment, e.g. a 50 N sanding force
        along -target.z; the joints must supply J^T wrench to produce it.
        """
        q = self._check_q(q)
        if payload_kg < 0:
            raise ValueError("payload mass must be >= 0")
        tau = self.gravity_torques(q)
        if payload_kg > 0:
            g_base = self._base_rotation_world().T @ np.asarray(self.model.gravity.linear)
            Jc = self.jacobian(q, payload_com)
            tau -= Jc[:3].T @ (payload_kg * g_base)
        if wrench is not None:
            tau += self.jacobian(q, wrench_point).T @ np.asarray(wrench, float)
        return tau
