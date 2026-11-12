"""Collision/clearance scene: robot links + parametric tool + part + floor, in one
Pinocchio GeometryModel so all pair checks run in C++ (Coal).

Pair categories and the margin each must keep (``Margins``):
    robot_part, tool_part   >= clearance        (tool_part skips the active TCP's contact geometry)
    tool_robot              >= self_clearance   (the flange link excluded: the tool is mounted on it)
    robot_self              >= 0                (adjacent / always-touching pairs excluded)
    floor                   >= 0                (base and shoulder excluded)
    carrier                 >= self_clearance   (arm from upper_arm on + tool vs carrier bodies)
    hose                    part / stay-outs >= clearance, floor >= 0, carrier >= self_clearance
                            (core.hose: the hose pack as capsules on the arm, when ``HOSE`` is set)

Contact geometry: a head link listed in a TCP's ``contact_links`` is skipped whole for
that TCP. Picked ``contact_faces`` are skipped together with every face sharing a vertex
with them (one ring), so the rim faces that meet the part at the contact edge do not
register as a collision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import coal
import numpy as np
import pinocchio as pin
import trimesh

from core.robot import Robot
from core.schema import ExtensionValues
from core.tool import ToolModel


@dataclass
class Margins:
    clearance: float = 0.01
    self_clearance: float = 0.005
    dress: float = field(default_factory=lambda: DRESS_M)  # v8 T18: hose pack on the arm (m)


# v8 T18: extra clearance around the arm from the upper arm on (hose / dress pack), against the
# part and stay-out zones; set from Optimize -> Cable & safety and by the planner jobs
DRESS_M = 0.0
# the hose pack as a shape on the arm (core.hose.Hose), or None; set like DRESS_M
HOSE = None


# 2026-11-11: low-detail collision copies of meshes (core.collision_mesh): component ("robot",
# "tool", "part", "carrier") -> Geom; PART_KEEP = part face ids kept at full detail. Set from the
# left panel's Collision geometry and shipped to the planner / worker processes (geom_state)
GEOM: dict = {}
PART_KEEP: tuple = ()
ARM_PAD = 0.0  # m: the robot arm's collision shapes are this much larger against everything else
ROBOT_AUTO_FACES = 5000  # robot links: "auto" reduces links above this (URDF CAD meshes)


def geom_state() -> tuple:
    return dict(GEOM), tuple(PART_KEEP), float(ARM_PAD)


def set_geom_state(state) -> None:
    global GEOM, PART_KEEP, ARM_PAD
    if state is not None:
        GEOM, PART_KEEP = dict(state[0]), tuple(int(f) for f in state[1])
        ARM_PAD = float(state[2]) if len(state) > 2 else 0.0


def _geom(component: str):
    from core.collision_mesh import Geom

    return GEOM.get(component) or Geom()


def _low(mesh: trimesh.Trimesh, component: str, keep=None, auto_faces: int | None = None) -> trimesh.Trimesh:
    """The collision copy of a mesh of this component (core.collision_mesh)."""
    from core.collision_mesh import collision_mesh

    return collision_mesh(mesh, _geom(component), keep=keep, auto_faces=auto_faces)


_ROBOT_MESHES: dict = {}  # (path, scale) -> trimesh: robot collision meshes read for simplifying / showing


def _robot_mesh(go) -> trimesh.Trimesh | None:
    path = getattr(go, "meshPath", "")
    if not path or not path.lower().endswith((".stl", ".obj", ".dae", ".ply", ".glb", ".gltf")):
        return None
    scale = np.asarray(go.meshScale, float)
    key = (path, tuple(np.round(scale, 9)))
    if key not in _ROBOT_MESHES:
        try:
            m = trimesh.load(path, force="mesh")
        except Exception:
            return None
        m.apply_transform(np.diag([*scale, 1.0]))
        _ROBOT_MESHES[key] = m
    return _ROBOT_MESHES[key]


HITS = "collides with: "  # PointResult.notes prefix of ``describe_hits``


def _mover(name: str) -> str:
    """A moving geometry's short name: tool, hose, or the robot link (``forearm_link_0`` ->
    ``forearm``)."""
    if name.startswith("tool_"):
        return "tool"
    if name.startswith("hose_"):
        return "hose"
    base = re.sub(r"_\d+$", "", name)
    return re.sub(r"_link$", "", base)


def _obstacle(name: str) -> str | None:
    if name in ("part", "floor"):
        return name
    if name.startswith("carrier_"):
        return f"carrier {name[len('carrier_') :]}"
    if name.startswith("stayout_"):
        return f"stay-out zone {name[len('stayout_') :]}"
    return None


def describe_hits(pairs: list[tuple[str, str]]) -> str:
    """``HITS`` + what collides, from ``CollisionScene.colliding_pairs``: "part (tool,
    forearm); floor (upper_arm)", "arm itself (...)", "tool vs arm (...)"; "" for no pairs."""
    hit: dict[str, list[str]] = {}
    for a, b in pairs:
        oa, ob = _obstacle(a), _obstacle(b)
        if oa or ob:
            key, movers = (oa, [b]) if oa else (ob, [a])
        elif a.startswith("tool_") or b.startswith("tool_"):
            key, movers = "tool vs arm", [b if a.startswith("tool_") else a]
        else:
            key, movers = "arm itself", [a, b]
        lst = hit.setdefault(key, [])
        lst += [m for m in map(_mover, movers) if m not in lst]
    if not hit:
        return ""
    return HITS + "; ".join(f"{k} ({', '.join(v)})" for k, v in hit.items())


def _bvh(mesh: trimesh.Trimesh) -> coal.BVHModelOBBRSS:
    m = coal.BVHModelOBBRSS()
    m.beginModel(len(mesh.faces), len(mesh.vertices))
    m.addVertices(np.ascontiguousarray(mesh.vertices, dtype=float))
    m.addTriangles(np.ascontiguousarray(mesh.faces, dtype=np.int64))
    m.endModel()
    return m


def _se3(T: np.ndarray) -> pin.SE3:
    return pin.SE3(np.asarray(T[:3, :3]), np.asarray(T[:3, 3]))


def _one_ring(mesh: trimesh.Trimesh, faces: set[int]) -> set[int]:
    if not faces:
        return set()
    verts = np.unique(mesh.faces[sorted(faces)])
    return set(np.flatnonzero(np.isin(mesh.faces, verts).any(axis=1)).tolist())


@dataclass
class _Pairs:
    robot_part: list[int] = field(default_factory=list)
    tool_part: list[int] = field(default_factory=list)
    tool_robot: list[int] = field(default_factory=list)
    robot_self: list[int] = field(default_factory=list)
    floor: list[int] = field(default_factory=list)
    carrier: list[int] = field(default_factory=list)
    obstacle: list[int] = field(default_factory=list)


class CollisionScene:
    def __init__(
        self,
        robot: Robot,
        tool: ToolModel | None,
        part_mesh: trimesh.Trimesh | None,
        T_world_base: np.ndarray,
        T_world_part: np.ndarray | None,
        margins: Margins | None = None,
        values: ExtensionValues | None = None,
        floor: bool = True,
        base_bodies: list | None = None,
        world_bodies: list | None = None,
        obstacles: list | None = None,
        hose=-1,
    ):
        """``obstacles``: stay-out zones, (name, mesh, T_world_body): world-fixed geometry the arm
        (from the shoulder on) and the tool must keep ``margins.clearance`` from. ``hose``:
        core.hose.Hose capsules on the arm (default: the module's ``HOSE``; None = no hose)."""
        self.robot = robot
        self.margins = margins or Margins()
        model = robot.model
        urdf = str(robot.spec.folder / robot.spec.urdf)
        self.gm = pin.buildGeomFromUrdf(
            model, urdf, pin.GeometryType.COLLISION, None, [str(robot.spec.folder)]
        )
        self.robot_ids = list(range(self.gm.ngeoms))
        self.shapes: dict[int, trimesh.Trimesh] = {}  # gid -> collision mesh (geometry frame), for showing
        self._simplify_robot()
        # the joint each robot shape moves with (0 = fixed base): pairs are chosen by link, not by
        # the shape's position in the list (a URDF link may have several collision shapes)
        joint_of = [g.parentJoint for g in self.gm.geometryObjects]

        # world <-> pinocchio universe (URDF root)
        pin.framesForwardKinematics(model, robot.data, pin.neutral(model))
        self.T_univ_base = np.asarray(robot.data.oMf[robot.base_frame_id].homogeneous)
        self.T_univ_world = self.T_univ_base @ np.linalg.inv(T_world_base)
        self.T_world_part = T_world_part

        # tool geometry, attached to the flange's parent joint
        self.tool_ids: list[int] = []
        self.tool_contact: dict[int, set[str]] = {}  # geom id -> TCPs that may touch with it
        if tool is not None:
            fr = model.frames[robot.flange_frame_id]
            T_joint_flange = np.asarray(fr.placement.homogeneous)
            for k, (a, b) in enumerate(tool.extension_segments(values)):
                length = float(np.linalg.norm(b - a))
                T = trimesh.geometry.align_vectors([0, 0, 1], (b - a) / length)
                T[:3, 3] = (a + b) / 2
                go = pin.GeometryObject(
                    f"tool_ext_{k}",
                    fr.parentJoint,
                    _se3(T_joint_flange @ T),
                    coal.Capsule(tool.ee.extension.radius, length),
                )
                self.tool_ids.append(self.gm.addGeometryObject(go))
                self.tool_contact[self.tool_ids[-1]] = set()
            T_fh = tool.T_flange_head(values)
            for hl in tool.links:
                for sig, piece in self._tool_pieces(tool, hl):
                    go = pin.GeometryObject(
                        f"tool_{hl.name}_{'+'.join(sorted(sig)) or 'body'}",
                        fr.parentJoint,
                        _se3(T_joint_flange @ T_fh @ hl.T_head_link),
                        _bvh(piece),
                    )
                    gid = self.gm.addGeometryObject(go)
                    self.tool_ids.append(gid)
                    self.tool_contact[gid] = set(sig)
                    self.shapes[gid] = piece

        self.hose_ids: list[int] = []
        hose = HOSE if hose == -1 else hose
        if hose is not None:
            from core.hose import segments

            for k, (j, T, L) in enumerate(segments(robot, hose)):
                go = pin.GeometryObject(f"hose_{k}", j, _se3(T), coal.Capsule(hose.radius, L))
                self.hose_ids.append(self.gm.addGeometryObject(go))

        self.part_id = None
        if part_mesh is not None and T_world_part is not None:
            low = _low(part_mesh, "part", keep=PART_KEEP)
            go = pin.GeometryObject("part", 0, _se3(self.T_univ_world @ T_world_part), _bvh(low))
            self.part_id = self.gm.addGeometryObject(go)
            self.shapes[self.part_id] = low
        self.floor_id = None
        if floor:
            go = pin.GeometryObject(
                "floor", 0, _se3(self.T_univ_world), coal.Halfspace(np.array([0, 0, 1.0]), 0.0)
            )
            self.floor_id = self.gm.addGeometryObject(go)
        # carrier bodies: moving ones are rigid in the base frame, static ones (rail track)
        # are world-fixed like the part. Each entry: (name, mesh, T) with T = T_base_body or
        # T_world_body respectively.
        self.carrier_ids: list[int] = []
        self.world_body_T: dict[int, np.ndarray] = {}
        self.follow: dict[int, object] = {}  # gid -> fn(T_world_base) -> T_base_body (lift)
        for name, mesh, T_base_body, *follow in base_bodies or []:
            if follow:
                T_base_body = follow[0](T_world_base)
            mesh = _low(mesh, "carrier")
            go = pin.GeometryObject(f"carrier_{name}", 0, _se3(self.T_univ_base @ T_base_body), _bvh(mesh))
            gid = self.gm.addGeometryObject(go)
            self.carrier_ids.append(gid)
            self.shapes[gid] = mesh
            if follow:
                self.follow[gid] = follow[0]
        for name, mesh, T_world_body in world_bodies or []:
            mesh = _low(mesh, "carrier")
            go = pin.GeometryObject(f"carrier_{name}", 0, _se3(self.T_univ_world @ T_world_body), _bvh(mesh))
            gid = self.gm.addGeometryObject(go)
            self.carrier_ids.append(gid)
            self.shapes[gid] = mesh
            self.world_body_T[gid] = T_world_body

        self.obstacle_ids: list[int] = []
        for name, mesh, T_world_body in obstacles or []:
            go = pin.GeometryObject(f"stayout_{name}", 0, _se3(self.T_univ_world @ T_world_body), _bvh(mesh))
            gid = self.gm.addGeometryObject(go)
            self.obstacle_ids.append(gid)
            self.world_body_T[gid] = T_world_body  # re-placed in set_base like the part

        self.pairs = _Pairs()
        P = self.pairs

        def add(i: int, j: int, bucket: list[int]) -> None:
            self.gm.addCollisionPair(pin.CollisionPair(i, j))
            bucket.append(len(self.gm.collisionPairs) - 1)

        if self.part_id is not None:
            for i in [*self.robot_ids, *self.hose_ids]:
                add(i, self.part_id, P.robot_part)
            for i in self.tool_ids:
                add(i, self.part_id, P.tool_part)
        flange_joint = model.frames[robot.flange_frame_id].parentJoint  # its link carries the tool
        for i in self.tool_ids:
            for j in self.robot_ids:
                if joint_of[j] != flange_joint:
                    add(i, j, P.tool_robot)
        for a in self.robot_ids:  # links at least two joints apart
            for b in self.robot_ids:
                if b > a and abs(joint_of[b] - joint_of[a]) >= 2:
                    add(a, b, P.robot_self)
        shoulder_on = [i for i in self.robot_ids if joint_of[i] >= 1]
        upper_arm_on = [i for i in self.robot_ids if joint_of[i] >= 2]
        if self.floor_id is not None:
            for i in [*upper_arm_on, *self.tool_ids, *self.hose_ids]:
                add(i, self.floor_id, P.floor)
        for o in self.obstacle_ids:  # stay-out zones: arm from the shoulder on, and the tool
            for i in [*shoulder_on, *self.tool_ids, *self.hose_ids]:
                add(i, o, P.obstacle)
        for c in self.carrier_ids:  # arm beyond the shoulder and the tool vs the carrier
            for i in [*upper_arm_on, *self.tool_ids, *self.hose_ids]:
                add(i, c, P.carrier)

        self.gd = pin.GeometryData(self.gm)
        margin_of = {}
        for k in P.robot_part + P.tool_part + P.obstacle:
            margin_of[k] = self.margins.clearance
        if self.margins.dress > 0:  # v8 T18: the hose pack rides on the arm from the upper arm on
            for k in P.robot_part + P.obstacle:
                g = self.gm.collisionPairs[k].first
                if g < len(joint_of) and joint_of[g] >= 2:
                    margin_of[k] = self.margins.clearance + self.margins.dress
        for k in P.tool_robot + P.carrier:
            margin_of[k] = self.margins.self_clearance
        self.arm_pad = float(ARM_PAD)
        if self.arm_pad > 0:  # 2026-11-11: the arm bloated against everything but itself
            robot = set(self.robot_ids)
            for k in P.robot_part + P.obstacle + P.floor + P.carrier + P.tool_robot:
                pair = self.gm.collisionPairs[k]
                if pair.first in robot or pair.second in robot:
                    margin_of[k] = margin_of.get(k, 0.0) + self.arm_pad
        for k, req in enumerate(self.gd.collisionRequests):
            req.security_margin = margin_of.get(k, 0.0)
            req.enable_contact = False
        self._drop_always_colliding()

    def set_base(self, T_world_base: np.ndarray) -> None:
        """Move the robot base: re-place the world-fixed geometry (part, floor)."""
        self.T_univ_world = self.T_univ_base @ np.linalg.inv(T_world_base)
        if self.part_id is not None:
            self.gm.geometryObjects[self.part_id].placement = _se3(self.T_univ_world @ self.T_world_part)
        if self.floor_id is not None:
            self.gm.geometryObjects[self.floor_id].placement = _se3(self.T_univ_world)
        for gid, T in self.world_body_T.items():
            self.gm.geometryObjects[gid].placement = _se3(self.T_univ_world @ T)
        for gid, fn in self.follow.items():  # carriage bodies below a lift
            self.gm.geometryObjects[gid].placement = _se3(self.T_univ_base @ fn(T_world_base))

    def _simplify_robot(self) -> None:
        """Robot link meshes above ``ROBOT_AUTO_FACES`` triangles (or all, for an explicit
        mode) are replaced by their collision copy (``GEOM["robot"]``)."""
        g = _geom("robot")
        if g.mode == "visual":
            return
        for gid in self.robot_ids:
            go = self.gm.geometryObjects[gid]
            n = getattr(go.geometry, "num_tris", 0)
            if not n or (g.mode == "auto" and n <= ROBOT_AUTO_FACES):
                continue
            mesh = _robot_mesh(go)
            if mesh is None:
                continue
            low = _low(mesh, "robot", auto_faces=ROBOT_AUTO_FACES)
            go.geometry = _bvh(low)
            self.shapes[gid] = low

    def shape(self, gid: int) -> trimesh.Trimesh | None:
        """The collision mesh of geometry ``gid`` in its own frame: meshes and capsules (None
        for the floor; stay-out zones are drawn by their own panel)."""
        if gid in self.shapes:
            return self.shapes[gid]
        go = self.gm.geometryObjects[gid]
        G = go.geometry
        m = None
        if isinstance(G, coal.Capsule):
            m = trimesh.creation.capsule(height=2 * G.halfLength, radius=G.radius, count=[12, 12])
        elif gid in self.robot_ids:
            m = _robot_mesh(go)
        if m is not None:
            self.shapes[gid] = m
        return m

    def world_shapes(self, q: np.ndarray) -> list[tuple[int, str, np.ndarray]]:
        """(gid, name, T_world_geom) of every geometry at joints q (the current base)."""
        pin.updateGeometryPlacements(
            self.robot.model, self.robot.data, self.gm, self.gd, np.asarray(q, float)
        )
        T_world_univ = np.linalg.inv(self.T_univ_world)
        return [
            (gid, go.name, T_world_univ @ np.asarray(self.gd.oMg[gid].homogeneous))
            for gid, go in enumerate(self.gm.geometryObjects)
        ]

    def _tool_pieces(self, tool: ToolModel, hl) -> list[tuple[frozenset, trimesh.Trimesh]]:
        """A head link's collision pieces: (TCPs that may touch the part with it, mesh). A link
        with its own URDF collision mesh (e.g. from the Package builder) uses that, unless a
        TCP has picked contact faces on it (face ids of the visual mesh); otherwise the visual
        mesh split by contact, the non-contact piece simplified (``GEOM["tool"]``)."""
        coll = getattr(hl, "collision", None)
        if coll is not None and not any(t.contact_faces.get(hl.name) for t in tool.ee.tcps):
            sig = frozenset(t.name for t in tool.ee.tcps if hl.name in t.contact_links)
            return [(sig, coll)]
        out = []
        for sig, faces in self._split_by_contact(tool, hl):
            piece = hl.mesh.submesh([np.array(sorted(faces))], append=True)
            if not sig:  # contact pieces stay exact: they meet the part at the contact edge
                piece = _low(piece, "tool")
            out.append((sig, piece))
        return out

    @staticmethod
    def _split_by_contact(tool: ToolModel, hl) -> list[tuple[frozenset, set[int]]]:
        """Partition a head link's faces by which TCPs may touch the part with them."""
        n = len(hl.mesh.faces)
        sig = [set() for _ in range(n)]
        for t in tool.ee.tcps:
            if hl.name in t.contact_links:
                faces = set(range(n))
            else:
                faces = _one_ring(hl.mesh, set(t.contact_faces.get(hl.name, [])))
            for f in faces:
                sig[f].add(t.name)
        groups: dict[frozenset, set[int]] = {}
        for f, s in enumerate(sig):
            groups.setdefault(frozenset(s), set()).add(f)
        return list(groups.items())

    def _drop_always_colliding(self) -> None:
        """Disable robot self pairs that touch at the neutral pose (adjacent links)."""
        q0 = pin.neutral(self.robot.model)
        pin.computeCollisions(self.robot.model, self.robot.data, self.gm, self.gd, q0, False)
        for k in self.pairs.robot_self:
            if self.gd.collisionResults[k].isCollision():
                self.gd.deactivateCollisionPair(k)
        self.pairs.robot_self = [k for k in self.pairs.robot_self if self.gd.activeCollisionPairs[k]]

    def _set_tcp(self, tcp: str | None) -> None:
        for k in self.pairs.tool_part:
            gid = self.gm.collisionPairs[k].first
            if tcp is not None and tcp in self.tool_contact.get(gid, set()):
                self.gd.deactivateCollisionPair(k)
            else:
                self.gd.activateCollisionPair(k)

    def collides(self, q: np.ndarray, tcp: str | None = None) -> bool:
        """True if any active pair is closer than its margin."""
        self._set_tcp(tcp)
        return bool(
            pin.computeCollisions(self.robot.model, self.robot.data, self.gm, self.gd, np.asarray(q), True)
        )

    def colliding_pairs(self, q: np.ndarray, tcp: str | None = None) -> list[tuple[str, str]]:
        self._set_tcp(tcp)
        pin.computeCollisions(self.robot.model, self.robot.data, self.gm, self.gd, np.asarray(q), False)
        out = []
        for k, cp in enumerate(self.gm.collisionPairs):
            if self.gd.activeCollisionPairs[k] and self.gd.collisionResults[k].isCollision():
                out.append((self.gm.geometryObjects[cp.first].name, self.gm.geometryObjects[cp.second].name))
        return out

    def part_clearance(self, q: np.ndarray, tcp: str | None = None) -> float:
        """Smallest distance from robot or (non-contact) tool geometry to the part."""
        if self.part_id is None:
            return float("inf")
        self._set_tcp(tcp)
        pin.updateGeometryPlacements(self.robot.model, self.robot.data, self.gm, self.gd, np.asarray(q))
        best = float("inf")
        for k in self.pairs.robot_part + self.pairs.tool_part:
            if self.gd.activeCollisionPairs[k]:
                d = pin.computeDistance(self.gm, self.gd, k).min_distance
                best = min(best, d)
        return best
