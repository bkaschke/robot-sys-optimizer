"""In-memory state of one GUI session, and its mapping to/from the project file.

Everything here is SI (m, rad). The GUI converts to mm/deg at its widgets.
Asset paths in the project file are relative to the project folder; paths that do not
exist there are looked up relative to the repo (bundled assets such as golden parts).
"""

from __future__ import annotations

import os
import shutil
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
# where projects live; RSO_WORKSPACE moves it (the test suite uses a temporary one so it never
# writes into the user's projects)
WORKSPACE = Path(os.environ.get("RSO_WORKSPACE") or REPO / "workspace")
DEFAULT_PROJECT_DIR = WORKSPACE / "default"
MESH_EXTS = (".stl", ".obj", ".ply")
UNIT_SCALE = {"m": 1.0, "mm": 1e-3, "in": 0.0254}
MOUNT_RPY = {"floor": (0.0, 0.0, 0.0), "wall": (0.0, -np.pi / 2, 0.0), "ceiling": (np.pi, 0.0, 0.0)}


@dataclass
class PartState:
    mesh_path: str  # as stored in the project (relative to project dir or repo)
    units: str = "m"
    T_world_part: np.ndarray = field(default_factory=lambda: np.eye(4))
    collision_keep: list = field(default_factory=list)  # faces kept at full detail (core.collision)
    collision_mesh: str | None = None  # v9: made in the Package builder (same frame and units)


@dataclass
class Session:
    project_dir: Path = DEFAULT_PROJECT_DIR
    robot_name: str | None = "ur20"  # registry name, or None for an uploaded URDF
    robot_urdf: str | None = None  # project-relative URDF path when robot_name is None
    robot: Robot | None = None
    mount: str = "floor"
    T_world_base: np.ndarray = field(default_factory=lambda: np.eye(4))
    q: np.ndarray | None = None
    home_deg: list | None = None  # project home pose (deg); None = the robot's spec home
    part: PartState | None = None
    tool: schema.EndEffector | None = None
    tool_model: ToolModel | None = None
    targets: schema.Targets = field(default_factory=schema.Targets)
    feasibility: dict = field(default_factory=dict)  # point id -> feasibility.PointResult (not saved)
    results: list[schema.Solution] = field(default_factory=list)  # placement solutions, best first
    result_points: dict = field(default_factory=dict)  # solution label -> {id: PointResult} (not saved)
    base: schema.Base = field(default_factory=schema.Base)  # static or carrier (M9)
    extra: dict = field(default_factory=dict)  # untouched project sections (optimizer)
    selection: tuple[str, list[int]] | None = None  # (part mesh path, face ids): saved selection
    load_warnings: list = field(default_factory=list)  # v9 #12: files a loaded project is missing
    # v9 #6: Compare robots: robot -> {"results", "points" {label: {pid: PointResult}}, "stamps", "metrics"}
    compared: dict = field(default_factory=dict)
    # v9 #6: while Results shows a compared robot, the project's own robot and plans (ui.compare_view)
    view_stash: dict | None = None
    bundled: list = field(default_factory=list)  # v9 #12: files the last save copied into the project
    ui: dict = field(default_factory=dict)  # saved panel settings ("panel.widget" -> value)
    stayouts: list = field(default_factory=list)  # schema.Primitive, world poses (core.stayout)
    stamps: dict = field(default_factory=dict)  # result key -> setup fingerprint (ui.staleness)
    steps: dict = field(default_factory=dict)  # confirmed setup step -> fingerprint
    tolerance: object = None  # v8 T5: last opt.tolerance.ToleranceResult (not saved)
    cycle: object = None  # v8 T6: core.cycle_time.CycleSettings (Optimize -> Cycle time)
    before_save: list = field(default_factory=list)  # GUI hooks that copy their state in here

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
        self.q = self.robot.home_q()  # the spec.yaml home pose
        self.home_deg = None

    def load_robot_from_urdf(self, urdf: Path, flange_link: str | None = None) -> None:
        info = inspect_urdf(urdf)
        self.robot = Robot.from_urdf(urdf, info.root_link, flange_link or info.leaf_link, info.name)
        self.robot_name, self.robot_urdf = None, self.relative(urdf)
        self.q = np.clip(np.zeros(self.robot.nq), self.robot.lower, self.robot.upper)
        self.home_deg = None

    def home_q(self) -> np.ndarray | None:
        """Home joints (rad): the project's saved home, else the robot's spec.yaml home."""
        r = self.robot
        if r is None:
            return None
        h = self.home_deg
        if h is not None and len(h) == r.nq:
            return np.clip(np.radians(np.asarray(h, float)), r.lower, r.upper)
        return r.home_q()

    def T_world_flange(self) -> np.ndarray:
        if self.robot is None:  # empty study: nothing hangs below the base yet
            return self.T_world_base.copy()
        return self.T_world_base @ self.robot.fk(self.q)

    def sites(self):
        """Planner search space: a carrier (movable mode) or one free floor base (static)."""
        from opt.sites import CarrierSites

        if self.base.mode != "movable" or self.part is None:
            return None
        b = self.base
        obs = self.obstacles()  # stay-out zones: the carrier keeps clear too
        if b.carrier == "mobile" and b.robot_sides() and b.fixed_standoff_m is not None:
            from opt.contour import ContourSites  # fixed standoff + facing side: 1-D contour

            return ContourSites(self.carrier(), self.part_mesh(), self.part.T_world_part, obstacles=obs)
        return CarrierSites(self.carrier(), self.part_mesh(), self.part.T_world_part, obstacles=obs)

    def obstacles(self):
        """Stay-out zones as core.stayout.Obstacle (meshes resolved like other assets)."""
        from core.stayout import obstacles

        return obstacles(self.stayouts, self.resolve)

    def obstacle_entries(self) -> list:
        """(name, mesh, T_world) of the stay-out zones, for CollisionScene(obstacles=...)."""
        from core.stayout import scene_entries

        return scene_entries(self.obstacles())

    def carrier(self):
        """core.carrier.Carrier for the current base (mesh bodies resolved like other assets)."""
        from core.carrier import Carrier

        return Carrier(self.base, self.T_world_base, resolve=self.resolve)

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

    def snapshot(self) -> schema.Project:
        """Project including GUI state: runs the ``before_save`` hooks, then to_project(). While
        Results shows a compared robot (``view_stash``), the project's own robot and plans are
        saved."""
        for fn in self.before_save:
            fn()
        stash = self.view_stash
        if not stash:
            return self.to_project()
        cur = {k: getattr(self, k) for k in stash}
        try:
            for k, v in stash.items():
                setattr(self, k, v)
            return self.to_project()
        finally:
            for k, v in cur.items():
                setattr(self, k, v)

    def compared_data(self) -> list:
        from core.feasibility import result_to_data

        return [
            schema.ComparedRobot(
                robot=name,
                metrics={
                    k: v
                    for k, v in c.get("metrics", {}).items()
                    if isinstance(v, (int, float, str, type(None)))
                },
                results=list(c.get("results", [])),
                result_points={
                    label: [result_to_data(r) for r in pts.values()]
                    for label, pts in c.get("points", {}).items()
                },
                stamps={k: dict(v) for k, v in c.get("stamps", {}).items()},
            )
            for name, c in self.compared.items()
        ]

    def work_state(self) -> schema.WorkState:
        from core.feasibility import result_to_data

        sel_mesh, sel = self.selection if self.selection else (None, [])
        return schema.WorkState(
            q=None if self.q is None else [float(v) for v in self.q],
            selection_mesh=sel_mesh,
            selection=[int(f) for f in sel],
            feasibility=[result_to_data(r) for r in self.feasibility.values()],
            result_points={
                label: [result_to_data(r) for r in pts.values()]
                for label, pts in self.result_points.items()
                if label in {sol.label for sol in self.results}
            },
            ui=dict(self.ui),
            stamps={
                k: dict(v)
                for k, v in self.stamps.items()
                if k == "feasibility" or k in {sol.label for sol in self.results}
            },
            steps={k: dict(v) for k, v in self.steps.items()},
        )

    def to_project(self) -> schema.Project:
        r = self.robot
        robot = (
            None
            if r is None
            else schema.Robot(
                name=self.robot_name,
                urdf=self.robot_urdf,
                flange_link=r.spec.flange_link,
                mount=self.mount,
                base_pose=matrix_to_pose(self.T_world_base),
                ik=r.spec.ik,
                home_deg=None if self.home_deg is None else [float(v) for v in self.home_deg],
                joint_limits_deg=r.spec.limits_deg,
            )
        )
        part = None
        if self.part is not None:
            part = schema.Part(
                mesh=self.part.mesh_path,
                units=self.part.units,
                pose=matrix_to_pose(self.part.T_world_part),
                collision_keep=[int(f) for f in self.part.collision_keep],
                collision_mesh=self.part.collision_mesh,
            )
        return schema.Project(
            robot=robot,
            part=part,
            end_effector=self.tool,
            targets=self.targets,
            results=self.results,
            base=self.base,
            stayouts=list(self.stayouts),
            work=self.work_state(),
            compared=self.compared_data(),
            **self.extra,
        )

    def apply_project(self, project: schema.Project, project_dir: Path) -> None:
        from ui.project_assets import tolerate_missing

        self.project_dir = project_dir
        project, self.load_warnings = tolerate_missing(self, project)  # v9 #12: open without them
        pr = project.robot
        if pr is None:  # empty study
            self.robot, self.robot_name, self.robot_urdf, self.q = None, None, None, None
        elif pr.name is not None:
            self.load_robot_from_registry(pr.name)
        else:
            self.load_robot_from_urdf(self.resolve(pr.urdf), pr.flange_link)
        if pr is not None:
            self.mount = pr.mount
            self.T_world_base = pose_to_matrix(pr.base_pose)
            ok = pr.home_deg is not None and self.robot is not None and len(pr.home_deg) == self.robot.nq
            self.home_deg = list(pr.home_deg) if ok else None
            if pr.joint_limits_deg is not None and self.robot is not None:
                try:  # limits that do not fit this robot are dropped
                    self.robot.set_limits(pr.joint_limits_deg)
                    self.q = np.clip(self.q, self.robot.lower, self.robot.upper)
                except ValueError:
                    pass
        self.part = None
        if project.part is not None:
            self.part = PartState(
                project.part.mesh,
                project.part.units,
                pose_to_matrix(project.part.pose),
                list(project.part.collision_keep),
                project.part.collision_mesh,
            )
            self.part_mesh()  # fail early if the mesh is missing or broken
        self.extra = {"optimizer": project.optimizer}
        self.base = project.base
        self.stayouts = list(project.stayouts)
        self.results = list(project.results)
        from core.feasibility import result_from_data

        self.view_stash = None
        self.compared = {
            c.robot: {
                "results": list(c.results),
                "points": {
                    label: {d.id: result_from_data(d) for d in pts} for label, pts in c.result_points.items()
                },
                "stamps": {k: dict(v) for k, v in c.stamps.items()},
                "metrics": dict(c.metrics),
            }
            for c in project.compared
        }
        self.targets = project.targets
        self.set_tool(project.end_effector)
        self._apply_work(project.work)

    def _apply_work(self, w: schema.WorkState) -> None:
        from core.feasibility import result_from_data

        if w.q is not None and self.robot is not None and len(w.q) == self.robot.nq:
            self.q = np.clip(np.array(w.q, float), self.robot.lower, self.robot.upper)
        ids = {p.id for p in self.targets.points}
        self.feasibility = {d.id: result_from_data(d) for d in w.feasibility if d.id in ids}
        labels = {sol.label for sol in self.results}
        self.result_points = {
            label: {d.id: result_from_data(d) for d in pts}
            for label, pts in w.result_points.items()
            if label in labels
        }
        self.selection = (w.selection_mesh, list(w.selection)) if w.selection_mesh else None
        self.ui = dict(w.ui)
        self.stamps = {k: dict(v) for k, v in w.stamps.items() if k == "feasibility" or k in labels}
        self.steps = {k: dict(v) for k, v in w.steps.items()}


def load_mesh(path: Path, units: str = "m") -> trimesh.Trimesh:
    if path.suffix.lower() not in MESH_EXTS:
        raise ValueError(f"unsupported mesh type {path.suffix}; use {', '.join(MESH_EXTS)}")
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"{path.name} contains no triangles")
    mesh.apply_scale(UNIT_SCALE[units])
    return mesh


def save_project(session: Session, project_dir: Path | None = None) -> Path:
    """Save to ``project_dir`` (default: the current folder). Saving to another folder
    ("save as") copies the project-local ``assets/`` along, so relative asset paths keep
    resolving."""
    if project_dir is not None:
        old_assets = Path(session.project_dir) / "assets"
        new = Path(project_dir)
        if old_assets.is_dir() and old_assets.resolve() != (new / "assets").resolve():
            shutil.copytree(old_assets, new / "assets", dirs_exist_ok=True)
        session.project_dir = project_dir
    session.project_dir.mkdir(parents=True, exist_ok=True)
    from ui.project_assets import bundle

    session.bundled = bundle(session)  # v9 #12: outside files are copied into the project
    path = session.project_dir / "project.json"
    project = session.snapshot()
    tmp = path.with_suffix(".json.tmp")  # write-then-rename: a crash never leaves half a file
    schema.save_project(project, tmp)
    tmp.replace(path)
    return path


def load_project(session: Session, project_dir: Path) -> None:
    """Load into a scratch session first so a broken project leaves ``session`` untouched."""
    project = schema.load_project(project_dir / "project.json")
    fresh = Session()
    fresh.apply_project(project, project_dir)
    hooks = session.before_save
    session.__dict__.update(fresh.__dict__)
    session.before_save = hooks  # GUI hooks belong to the app, not to the loaded project


LAST_PROJECT = WORKSPACE / "last_project.txt"


MAX_RECENT = 5


def _recorded() -> list[Path]:
    try:
        lines = LAST_PROJECT.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [Path(x.strip()) for x in lines if x.strip()]


def remember_last(project_dir: Path) -> None:
    """Record the project folder (newest first, the last ``MAX_RECENT`` kept; v7 round 2 T9
    extended the one-line file to a list, which old one-line files still read as)."""
    d = Path(project_dir).resolve()
    keep = [d, *[p for p in _recorded() if p != d]]
    try:
        LAST_PROJECT.parent.mkdir(parents=True, exist_ok=True)
        LAST_PROJECT.write_text("\n".join(str(p) for p in keep[: MAX_RECENT * 2]), encoding="utf-8")
    except OSError:
        pass


def recent_projects() -> list[Path]:
    """The last ``MAX_RECENT`` saved / loaded project folders that still have a project.json,
    newest first."""
    out = [p for p in _recorded() if (p / "project.json").is_file()]
    return out[:MAX_RECENT]


def last_project() -> Path | None:
    """The last saved/loaded project folder, if it still has a project.json."""
    recent = recent_projects()
    return recent[0] if recent else None


def empty_cell() -> Session:
    """The start-up cell (v7 round 2 T9): the default robot on a static base, no part, tool,
    targets or plans."""
    s = Session()
    s.load_robot_from_registry("ur20")
    return s
