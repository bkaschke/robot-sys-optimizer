"""Parametric end effector: mount adapter + extension + head + TCPs.

Frames and geometry (all poses 4x4, SI units):

    pose tools (v2):    T_mount_head = Trans(x, y, z) @ RPY(rx, ry, rz)
    legacy tools:       T_mount_head = Rz(roll) @ Tz(L) @ Rx(theta) @ Ty(d)
    T_flange_head = T_flange_mount @ T_mount_head
    T_flange_tcp  = T_flange_head @ T_head_tcp

The extension is drawn and collision-checked as capsules of ``extension.radius``. Pose
tools: the connector is a cubic Hermite spline from the mount origin (leaving along mount
z) to the head origin (arriving along head z), tangent lengths = the chord, sampled into
``SPLINE_SEGMENTS`` capsules; a head straight above the mount gives a straight connector.
Legacy tools: segment 1 runs from the mount along mount z for L; segment 2 (only when
d != 0) is the lateral leg from the bend to the head origin. The head is the EE URDF's root
link frame; its links are rigid (any movable joints are held at zero).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy

from core.schema import EndEffector, ExtensionValues
from core.transforms import make_T, pose_to_matrix, rpy_to_matrix

SPLINE_SEGMENTS = 8  # capsules along a pose tool's connector


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return make_T(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]))


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return make_T(np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]]))


def _tr(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    return make_T(p=(x, y, z))


def _collision_meshes(path: Path) -> dict[str, trimesh.Trimesh]:
    """Link name -> its URDF collision geometry (meshes and primitives) in the link frame."""
    try:
        urdf = yourdfpy.URDF.load(
            path,
            build_scene_graph=False,
            build_collision_scene_graph=True,
            load_meshes=False,
            load_collision_meshes=True,
            filename_handler=partial(yourdfpy.filename_handler_magic, dir=path.parent),
        )
    except Exception:  # no or broken collision elements: the visual geometry is used
        return {}
    sc = urdf.collision_scene
    if sc is None:
        return {}
    root = urdf.base_link
    out: dict[str, list] = {}
    for node in sc.graph.nodes_geometry:
        T_root_node, gname = sc.graph.get(frame_to=node, frame_from=root)
        link = node
        while link not in urdf.link_map:
            link = sc.graph.transforms.parents[link]
        m = sc.geometry[gname].copy()
        m.apply_transform(
            np.linalg.inv(urdf.get_transform(link, root, collision_geometry=True)) @ T_root_node
        )
        out.setdefault(link, []).append(m)
    return {k: trimesh.util.concatenate(v) for k, v in out.items()}


@dataclass
class HeadLink:
    name: str
    T_head_link: np.ndarray
    mesh: trimesh.Trimesh  # visual geometry in the link frame
    collision: trimesh.Trimesh | None = None  # the URDF's own collision mesh, when it differs


class ToolModel:
    def __init__(self, ee: EndEffector, urdf_path: Path | None):
        self.ee = ee
        self.links: list[HeadLink] = []
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
        coll = _collision_meshes(path)
        for name in urdf.link_map:
            meshes = per_link.get(name)
            if meshes:
                mesh = trimesh.util.concatenate(meshes)
                c = coll.get(name)
                if (
                    c is not None
                    and np.array_equal(c.faces.shape, mesh.faces.shape)
                    and np.allclose(c.vertices, mesh.vertices)
                ):
                    c = None  # the same geometry as the visual one
                self.links.append(HeadLink(name, urdf.get_transform(name, root), mesh, c))

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
        if self.ee.extension.type == "pose":
            R = rpy_to_matrix(np.radians([v.rx_deg, v.ry_deg, v.rz_deg]))
            return make_T(R, (v.x, v.y, v.z))
        T = _rz(np.radians(v.roll_deg)) @ _tr(z=v.L) @ _rx(np.radians(v.theta_deg)) @ _tr(y=v.d)
        if self.ee.extension.type == "bent2":  # v8 T16: second bend + leg
            T = T @ _rx(np.radians(v.theta2_deg)) @ _tr(y=v.d2)
        return T

    def T_flange_head(self, v: ExtensionValues | None = None) -> np.ndarray:
        return self.T_flange_mount @ self.T_mount_head(v)

    def T_flange_tcp(self, tcp: str, v: ExtensionValues | None = None) -> np.ndarray:
        for t in self.ee.tcps:
            if t.name == tcp:
                return self.T_flange_head(v) @ pose_to_matrix(t.T_head_tcp)
        raise KeyError(f"no TCP named '{tcp}'")

    def connector_points(self, v: ExtensionValues | None = None, n: int = SPLINE_SEGMENTS) -> np.ndarray:
        """Pose tools: n + 1 points (flange frame) along the Hermite spline mount -> head."""
        T_fm = self.T_flange_mount
        T_fh = T_fm @ self.T_mount_head(v)
        p0, p1, t0, t1 = T_fm[:3, 3], T_fh[:3, 3], T_fm[:3, 2], T_fh[:3, 2]
        c = float(np.linalg.norm(p1 - p0))
        s = np.linspace(0.0, 1.0, n + 1)[:, None]
        h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
        h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
        return h00 * p0 + h10 * c * t0 + h01 * p1 + h11 * c * t1

    def extension_segments(self, v: ExtensionValues | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
        """Capsule axis segments (p0, p1) in the flange frame; zero-length legs are dropped."""
        v = self.values(v)
        if self.ee.extension.type == "pose":
            P = self.connector_points(v)
            return [(a, b) for a, b in zip(P[:-1], P[1:], strict=True) if np.linalg.norm(b - a) > 1e-9]
        T_fm = self.T_flange_mount @ _rz(np.radians(v.roll_deg))
        bend = T_fm @ _tr(z=v.L)
        leg = bend @ _rx(np.radians(v.theta_deg)) @ _tr(y=v.d)
        pts = [T_fm[:3, 3], bend[:3, 3], leg[:3, 3]]
        if self.ee.extension.type == "bent2":  # v8 T16
            pts.append((leg @ _rx(np.radians(v.theta2_deg)) @ _tr(y=v.d2))[:3, 3])
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
