"""Parametric end effector: mount adapter + extension (L, theta, d, roll) + head + TCPs.

Frames and geometry (all poses 4x4, SI units):

    T_flange_head = T_flange_mount @ Rz(roll) @ Tz(L) @ Rx(theta) @ Ty(d)
    T_flange_tcp  = T_flange_head @ T_head_tcp

The extension is drawn and collision-checked as capsules of ``extension.radius``:
segment 1 runs from the mount origin along mount z for L; segment 2 (only when d != 0)
is the lateral leg from the bend to the head origin. The head is the EE URDF's root link
frame; its links are rigid (any movable joints are held at zero).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy

from core.schema import EndEffector, ExtensionValues
from core.transforms import make_T, pose_to_matrix


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return make_T(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]))


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return make_T(np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]]))


def _tr(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    return make_T(p=(x, y, z))


@dataclass
class HeadLink:
    name: str
    T_head_link: np.ndarray
    mesh: trimesh.Trimesh  # visual geometry in the link frame


@dataclass
class MassProperties:
    mass_kg: float
    com_flange: np.ndarray
    head_mass_known: bool  # False when the URDF has no inertials and no override is set


class ToolModel:
    def __init__(self, ee: EndEffector, urdf_path: Path | None):
        self.ee = ee
        self.links: list[HeadLink] = []
        self._head_mass, self._head_com = 0.0, np.zeros(3)
        if urdf_path is not None:
            self._load_head(Path(urdf_path))

    # --- head ---------------------------------------------------------------------

    def _load_head(self, path: Path) -> None:
        urdf = yourdfpy.URDF.load(
            path,
            build_scene_graph=True,
            load_meshes=True,
            filename_handler=partial(yourdfpy.filename_handler_magic, dir=path.parent),
        )
        root = urdf.base_link
        geom = urdf.scene.geometry
        graph = urdf.scene.graph
        per_link: dict[str, list[trimesh.Trimesh]] = {}
        for node in graph.nodes_geometry:
            T_root_node, gname = graph.get(frame_to=node, frame_from=root)
            # geometry nodes hang off their link: find the owning link by walking parents
            link = node
            while link not in urdf.link_map:
                link = graph.transforms.parents[link]
            T_root_link = urdf.get_transform(link, root)
            m = geom[gname].copy()
            m.apply_transform(np.linalg.inv(T_root_link) @ T_root_node)
            per_link.setdefault(link, []).append(m)
        for name in urdf.link_map:
            meshes = per_link.get(name)
            if meshes:
                mesh = trimesh.util.concatenate(meshes)
                self.links.append(HeadLink(name, urdf.get_transform(name, root), mesh))
        mass, moment = 0.0, np.zeros(3)
        for name, link in urdf.link_map.items():
            inertial = link.inertial
            if inertial is None or not inertial.mass:
                continue
            T = urdf.get_transform(name, root)
            c_link = inertial.origin[:3, 3] if inertial.origin is not None else np.zeros(3)
            mass += inertial.mass
            moment += inertial.mass * (T[:3, :3] @ c_link + T[:3, 3])
        self._head_mass = mass
        self._head_com = moment / mass if mass > 0 else np.zeros(3)

    def link(self, name: str) -> HeadLink:
        for hl in self.links:
            if hl.name == name:
                return hl
        raise KeyError(f"tool has no link '{name}' with geometry")

    # --- geometry -------------------------------------------------------------------

    def values(self, v: ExtensionValues | None = None) -> ExtensionValues:
        v = v or self.ee.extension.nominal
        if self.ee.extension.type == "straight":
            v = v.model_copy(update={"theta_deg": 0.0, "d": 0.0})
        return v

    @property
    def T_flange_mount(self) -> np.ndarray:
        return pose_to_matrix(self.ee.T_flange_mount)

    def T_mount_head(self, v: ExtensionValues | None = None) -> np.ndarray:
        v = self.values(v)
        return _rz(np.radians(v.roll_deg)) @ _tr(z=v.L) @ _rx(np.radians(v.theta_deg)) @ _tr(y=v.d)

    def T_flange_head(self, v: ExtensionValues | None = None) -> np.ndarray:
        return self.T_flange_mount @ self.T_mount_head(v)

    def T_flange_tcp(self, tcp: str, v: ExtensionValues | None = None) -> np.ndarray:
        for t in self.ee.tcps:
            if t.name == tcp:
                return self.T_flange_head(v) @ pose_to_matrix(t.T_head_tcp)
        raise KeyError(f"no TCP named '{tcp}'")

    def extension_segments(self, v: ExtensionValues | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
        """Capsule axis segments (p0, p1) in the flange frame; zero-length legs are dropped."""
        v = self.values(v)
        T_fm = self.T_flange_mount @ _rz(np.radians(v.roll_deg))
        bend = T_fm @ _tr(z=v.L)
        pts = [T_fm[:3, 3], bend[:3, 3], (bend @ _rx(np.radians(v.theta_deg)) @ _tr(y=v.d))[:3, 3]]
        return [(a, b) for a, b in zip(pts[:-1], pts[1:], strict=True) if np.linalg.norm(b - a) > 1e-9]

    def extension_mesh(self, v: ExtensionValues | None = None) -> trimesh.Trimesh | None:
        r = self.ee.extension.radius
        parts = []
        for a, b in self.extension_segments(v):
            length = float(np.linalg.norm(b - a))
            cap = trimesh.creation.capsule(height=length, radius=r, count=[16, 16])
            # trimesh capsule: hemisphere centres at z = +-length/2 -> align with a..b
            T = trimesh.geometry.align_vectors([0, 0, 1], (b - a) / length)
            T[:3, 3] = (a + b) / 2
            cap.apply_transform(T)
            parts.append(cap)
        return trimesh.util.concatenate(parts) if parts else None

    def length(self, v: ExtensionValues | None = None) -> float:
        return float(sum(np.linalg.norm(b - a) for a, b in self.extension_segments(v)))

    # --- mass -------------------------------------------------------------------------

    def mass_properties(self, v: ExtensionValues | None = None) -> MassProperties:
        head_m = self.ee.mass_kg if self.ee.mass_kg is not None else self._head_mass
        head_c = np.asarray(self.ee.com_head) if self.ee.com_head is not None else self._head_com
        known = self.ee.mass_kg is not None or self._head_mass > 0
        T_fh = self.T_flange_head(v)
        total = head_m
        moment = head_m * (T_fh[:3, :3] @ head_c + T_fh[:3, 3])
        for a, b in self.extension_segments(v):
            m = self.ee.extension.kg_per_m * float(np.linalg.norm(b - a))
            total += m
            moment += m * (a + b) / 2
        com = moment / total if total > 0 else np.zeros(3)
        return MassProperties(total, com, known)


# --- tool library (tools/<name>.yaml) ------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO / "tools"
_NAME_OK = __import__("re").compile(r"^[A-Za-z0-9_-]{1,64}$")


def list_tools() -> list[str]:
    return sorted(p.stem for p in TOOLS_DIR.glob("*.yaml"))


def _tool_path(name: str) -> Path:
    if not _NAME_OK.match(name):
        raise ValueError("tool name: 1-64 letters, digits, '_' or '-'")
    return TOOLS_DIR / f"{name}.yaml"


def load_tool(name: str) -> EndEffector:
    import yaml

    path = _tool_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"tool '{name}' not in library; have {list_tools()}")
    ee = EndEffector.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    ref = path.relative_to(REPO) if path.is_relative_to(REPO) else path
    return ee.model_copy(update={"tool": ref.as_posix()})


def save_tool(ee: EndEffector, name: str | None = None) -> Path:
    import yaml

    path = _tool_path(name or ee.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ee.model_copy(update={"tool": None, "name": path.stem}).model_dump(mode="json", exclude_none=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path
