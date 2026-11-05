"""In-memory state of one GUI session, and its mapping to/from the project file.

Everything here is SI (m, rad). The GUI converts to mm/deg at its widgets.
Asset paths in the project file are relative to the project folder; paths that do not
exist there are looked up relative to the repo (bundled assets such as golden parts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from core import schema
from core.robot import Robot, list_robots
from core.tool import ToolModel, load_tool
from core.transforms import matrix_to_pose, pose_to_matrix
from core.urdf_import import inspect_urdf

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_DIR = REPO / "workspace" / "default"
MESH_EXTS = (".stl", ".obj", ".ply")
UNIT_SCALE = {"m": 1.0, "mm": 1e-3, "in": 0.0254}
HOME_Q = {"ur20": np.radians([0, -90, 90, -90, -90, 0])}
MOUNT_RPY = {"floor": (0.0, 0.0, 0.0), "wall": (0.0, -np.pi / 2, 0.0), "ceiling": (np.pi, 0.0, 0.0)}


@dataclass
class PartState:
    mesh_path: str  # as stored in the project (relative to project dir or repo)
    units: str = "m"
    T_world_part: np.ndarray = field(default_factory=lambda: np.eye(4))


@dataclass
class Session:
    project_dir: Path = DEFAULT_PROJECT_DIR
    robot_name: str | None = "ur20"  # registry name, or None for an uploaded URDF
    robot_urdf: str | None = None  # project-relative URDF path when robot_name is None
    robot: Robot | None = None
    mount: str = "floor"
    T_world_base: np.ndarray = field(default_factory=lambda: np.eye(4))
    q: np.ndarray | None = None
    part: PartState | None = None
    tool: schema.EndEffector | None = None
    tool_model: ToolModel | None = None
    targets: schema.Targets = field(default_factory=schema.Targets)
    feasibility: dict = field(default_factory=dict)  # point id -> feasibility.PointResult (not saved)
    results: list[schema.Solution] = field(default_factory=list)  # placement solutions, best first
    result_points: dict = field(default_factory=dict)  # solution label -> {id: PointResult} (not saved)
    base: schema.Base = field(default_factory=schema.Base)  # static or carrier (M9)
    extra: dict = field(default_factory=dict)  # untouched project sections (optimizer)

    # --- assets ---------------------------------------------------------------

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        if p.is_absolute():
            return p
        for base in (self.project_dir, REPO):
            if (base / p).exists():
                return (base / p).resolve()
        raise FileNotFoundError(f"asset not found in project or repo: {rel}")

    def relative(self, path: Path) -> str:
        path = path.resolve()
        for base in (self.project_dir.resolve(), REPO):
            if path.is_relative_to(base):
                return path.relative_to(base).as_posix()
        return path.as_posix()

    def load_robot_from_registry(self, name: str) -> None:
        if name not in list_robots():
            raise ValueError(f"unknown robot '{name}'")
        self.robot = Robot.from_registry(name)
        self.robot_name, self.robot_urdf = name, None
        self.q = HOME_Q.get(name, np.zeros(self.robot.nq)).copy()

    def load_robot_from_urdf(self, urdf: Path, flange_link: str | None = None) -> None:
        info = inspect_urdf(urdf)
        self.robot = Robot.from_urdf(urdf, info.root_link, flange_link or info.leaf_link, info.name)
        self.robot_name, self.robot_urdf = None, self.relative(urdf)
        self.q = np.clip(np.zeros(self.robot.nq), self.robot.lower, self.robot.upper)

    def T_world_flange(self) -> np.ndarray:
        return self.T_world_base @ self.robot.fk(self.q)

    def sites(self):
        """Planner search space: a carrier (movable mode) or one free floor base (static)."""
        from core.carrier import Carrier
        from opt.sites import CarrierSites

        if self.base.mode != "movable" or self.part is None:
            return None
        return CarrierSites(Carrier(self.base, self.T_world_base), self.part_mesh(), self.part.T_world_part)

    def set_tool(self, ee: schema.EndEffector | None) -> None:
        """Select a tool and rebuild its geometry model (meshes are loaded once here)."""
        if ee is None:
            self.tool, self.tool_model = None, None
            return
        urdf = self.resolve(ee.urdf) if ee.urdf else None
        self.tool_model = ToolModel(ee, urdf)
        self.tool = ee

    def load_tool_from_library(self, name: str) -> None:
        self.set_tool(load_tool(name))

    @property
    def robot_urdf_path(self) -> Path:
        return self.robot.spec.folder / self.robot.spec.urdf

    def part_mesh(self) -> trimesh.Trimesh:
        if self.part is None:
            raise ValueError("no part loaded")
        return load_mesh(self.resolve(self.part.mesh_path), self.part.units)

    # --- project file -----------------------------------------------------------

    def to_project(self) -> schema.Project:
        r = self.robot
        robot = schema.Robot(
            name=self.robot_name,
            urdf=self.robot_urdf,
            flange_link=r.spec.flange_link,
            mount=self.mount,
            base_pose=matrix_to_pose(self.T_world_base),
            ik=r.spec.ik,
            payload_spec=r.spec.payload,
        )
        part = None
        if self.part is not None:
            part = schema.Part(
                mesh=self.part.mesh_path,
                units=self.part.units,
                pose=matrix_to_pose(self.part.T_world_part),
            )
        return schema.Project(
            robot=robot,
            part=part,
            end_effector=self.tool,
            targets=self.targets,
            results=self.results,
            base=self.base,
            **self.extra,
        )

    def apply_project(self, project: schema.Project, project_dir: Path) -> None:
        self.project_dir = project_dir
        pr = project.robot
        if pr.name is not None:
            self.load_robot_from_registry(pr.name)
        else:
            self.load_robot_from_urdf(self.resolve(pr.urdf), pr.flange_link)
        self.mount = pr.mount
        self.T_world_base = pose_to_matrix(pr.base_pose)
        self.part = None
        if project.part is not None:
            self.part = PartState(project.part.mesh, project.part.units, pose_to_matrix(project.part.pose))
            self.part_mesh()  # fail early if the mesh is missing or broken
        self.extra = {"optimizer": project.optimizer}
        self.base = project.base
        self.results = list(project.results)
        self.targets = project.targets
        self.set_tool(project.end_effector)


def load_mesh(path: Path, units: str = "m") -> trimesh.Trimesh:
    if path.suffix.lower() not in MESH_EXTS:
        raise ValueError(f"unsupported mesh type {path.suffix}; use {', '.join(MESH_EXTS)}")
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"{path.name} contains no triangles")
    mesh.apply_scale(UNIT_SCALE[units])
    return mesh


def save_project(session: Session, project_dir: Path | None = None) -> Path:
    if project_dir is not None:
        session.project_dir = project_dir
    session.project_dir.mkdir(parents=True, exist_ok=True)
    path = session.project_dir / "project.json"
    schema.save_project(session.to_project(), path)
    return path


def load_project(session: Session, project_dir: Path) -> None:
    """Load into a scratch session first so a broken project leaves ``session`` untouched."""
    project = schema.load_project(project_dir / "project.json")
    fresh = Session()
    fresh.apply_project(project, project_dir)
    session.__dict__.update(fresh.__dict__)
