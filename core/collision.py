"""Collision/clearance scene: robot links + parametric tool + part + floor, in one
Pinocchio GeometryModel so all pair checks run in C++ (Coal).

Pair categories and the margin each must keep (``Margins``):
    robot_part, tool_part   >= clearance        (tool_part skips the active TCP's contact geometry)
    tool_robot              >= self_clearance   (wrist_3 excluded: the tool is mounted on it)
    robot_self              >= 0                (adjacent / always-touching pairs excluded)
    floor                   >= 0                (base and shoulder excluded)
    carrier                 >= self_clearance   (arm from upper_arm on + tool vs carrier bodies)

Contact geometry: a head link listed in a TCP's ``contact_links`` is skipped whole for
that TCP. Picked ``contact_faces`` are skipped together with every face sharing a vertex
with them (one ring), so the rim faces that meet the part at the contact edge do not
register as a collision.
"""

from __future__ import annotations

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
    ):
        """``obstacles``: stay-out zones, (name, mesh, T_world_body): world-fixed geometry the arm
        (from the shoulder on) and the tool must keep ``margins.clearance`` from."""
        self.robot = robot
        self.margins = margins or Margins()
        model = robot.model
        urdf = str(robot.spec.folder / robot.spec.urdf)
        self.gm = pin.buildGeomFromUrdf(
            model, urdf, pin.GeometryType.COLLISION, None, [str(robot.spec.folder)]
        )
        self.robot_ids = list(range(self.gm.ngeoms))
        names = [g.name for g in self.gm.geometryObjects]

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
                for sig, faces in self._split_by_contact(tool, hl):
                    piece = hl.mesh.submesh([np.array(sorted(faces))], append=True)
                    go = pin.GeometryObject(
                        f"tool_{hl.name}_{'+'.join(sorted(sig)) or 'body'}",
                        fr.parentJoint,
                        _se3(T_joint_flange @ T_fh @ hl.T_head_link),
                        _bvh(piece),
                    )
                    gid = self.gm.addGeometryObject(go)
                    self.tool_ids.append(gid)
                    self.tool_contact[gid] = set(sig)

        self.part_id = None
        if part_mesh is not None and T_world_part is not None:
            go = pin.GeometryObject("part", 0, _se3(self.T_univ_world @ T_world_part), _bvh(part_mesh))
            self.part_id = self.gm.addGeometryObject(go)
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
            go = pin.GeometryObject(f"carrier_{name}", 0, _se3(self.T_univ_base @ T_base_body), _bvh(mesh))
            gid = self.gm.addGeometryObject(go)
            self.carrier_ids.append(gid)
            if follow:
                self.follow[gid] = follow[0]
        for name, mesh, T_world_body in world_bodies or []:
            go = pin.GeometryObject(f"carrier_{name}", 0, _se3(self.T_univ_world @ T_world_body), _bvh(mesh))
            gid = self.gm.addGeometryObject(go)
            self.carrier_ids.append(gid)
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
            for i in self.robot_ids:
                add(i, self.part_id, P.robot_part)
            for i in self.tool_ids:
                add(i, self.part_id, P.tool_part)
        flange_link_geom = names[-1]  # last robot link (wrist_3 for 6R arms) carries the tool
        for i in self.tool_ids:
            for j in self.robot_ids:
                if names[j] != flange_link_geom:
                    add(i, j, P.tool_robot)
        for a in self.robot_ids:
            for b in self.robot_ids:
                if b >= a + 2:
                    add(a, b, P.robot_self)
        if self.floor_id is not None:
            for i in [*self.robot_ids[2:], *self.tool_ids]:
                add(i, self.floor_id, P.floor)
        for o in self.obstacle_ids:  # stay-out zones: arm from the shoulder on, and the tool
            for i in [*self.robot_ids[1:], *self.tool_ids]:
                add(i, o, P.obstacle)
        for c in self.carrier_ids:  # arm beyond the shoulder and the tool vs the carrier
            for i in [*self.robot_ids[2:], *self.tool_ids]:
                add(i, c, P.carrier)

        self.gd = pin.GeometryData(self.gm)
        margin_of = {}
        for k in P.robot_part + P.tool_part + P.obstacle:
            margin_of[k] = self.margins.clearance
        for k in P.tool_robot + P.carrier:
            margin_of[k] = self.margins.self_clearance
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
