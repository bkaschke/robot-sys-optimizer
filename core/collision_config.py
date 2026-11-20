"""The collision settings as one value (v12 #10).

Before v12 these were eight module globals of core.collision (``DRESS_M``, ``HOSE``, ``GEOM``,
``PART_KEEP``, ``ARM_PAD``, ``LINK_PAD``, ``PART_LOW``, ``PART_LOW_SCALE``), set by the GUI and
copied by hand into every job spec and the check pool's ``CheckerSpec``; a setting missed in one of
those places planned with the wrong geometry without a word. Now:

* ``CollisionConfig`` (frozen) holds them all; ``CollisionScene(config=...)`` uses it (None: the
  process's ``current()`` config);
* the GUI changes the current config with ``update(**fields)``;
* a job takes ``current()`` once into its spec (``spec["collision"]``, ``CheckerSpec.collision``) and
  passes that value to every scene it builds; a worker process never relies on a global.

Also here: the low-detail collision copies of meshes that use the config (``part_low``, ``low``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import trimesh

ROBOT_AUTO_FACES = 5000  # robot links: "auto" reduces links above this (URDF CAD meshes)


@dataclass(frozen=True)
class CollisionConfig:
    # v8 T18: extra clearance around the arm from the upper arm on (hose / dress pack), against the
    # part and stay-out zones (Optimize -> Cable & safety)
    dress_m: float = 0.0
    hose: object = None  # the hose pack as a shape on the arm (core.hose.Hose), or None
    # 2026-11-11: low-detail collision copies (core.collision_mesh): component ("robot", "tool",
    # "part", "carrier") -> Geom; part_keep = part face ids kept at full detail
    geom: dict = field(default_factory=dict)
    part_keep: tuple = ()
    arm_pad: float = 0.0  # m: the arm's collision shapes are this much larger against everything else
    link_pad: dict = field(default_factory=dict)  # v10 T19: Pinocchio joint index -> m (else arm_pad)
    # v9: the part's own collision mesh (absolute path, metres after ``part_low_scale``) from the
    # Package builder; when set it replaces the simplified copy of the part mesh
    part_low: str | None = None
    part_low_scale: float = 1.0

    def replace(self, **kw) -> CollisionConfig:
        return dataclasses.replace(self, **kw)

    def key(self) -> str:
        """A stable text of every setting (caches, staleness)."""
        return repr(
            (
                round(float(self.dress_m), 6),
                repr(self.hose),
                sorted((k, repr(v)) for k, v in self.geom.items()),
                tuple(int(f) for f in self.part_keep),
                round(float(self.arm_pad), 6),
                sorted((int(k), round(float(v), 6)) for k, v in self.link_pad.items()),
                self.part_low,
                round(float(self.part_low_scale), 9),
            )
        )


_CURRENT = CollisionConfig()


def current() -> CollisionConfig:
    """The settings of this process (the GUI's, or the ones a job was started with)."""
    return _CURRENT


def use(cfg: CollisionConfig | None) -> None:
    """Make ``cfg`` the current settings (None: the defaults)."""
    global _CURRENT
    _CURRENT = cfg if cfg is not None else CollisionConfig()


def update(**kw) -> CollisionConfig:
    """Change some of the current settings; returns the new config."""
    use(_CURRENT.replace(**kw))
    return _CURRENT


def link_joints(robot) -> dict[str, int]:
    """Robot link name -> the Pinocchio joint that moves it (for ``link_pad``)."""
    import pinocchio as pin

    out = {}
    for f in robot.model.frames:
        if f.type == pin.FrameType.BODY and f.parentJoint > 0:
            out.setdefault(f.name, int(f.parentJoint))
    return out


def geom_of(cfg: CollisionConfig, component: str):
    from core.collision_mesh import Geom

    return cfg.geom.get(component) or Geom()


def low(
    mesh: trimesh.Trimesh, component: str, cfg: CollisionConfig, keep=None, auto_faces: int | None = None
) -> trimesh.Trimesh:
    """The collision copy of a mesh of this component (core.collision_mesh)."""
    from core.collision_mesh import collision_mesh

    return collision_mesh(mesh, geom_of(cfg, component), keep=keep, auto_faces=auto_faces)


_PART_LOW: dict = {}  # (path, mtime, scale) -> mesh


def part_low(part_mesh: trimesh.Trimesh, cfg: CollisionConfig) -> trimesh.Trimesh:
    """The part's collision mesh: its own (``cfg.part_low``) when there is one, else a copy of
    ``part_mesh`` simplified by ``cfg.geom["part"]`` (``cfg.part_keep`` faces exact)."""
    if cfg.part_low:
        p = Path(cfg.part_low)
        try:
            key = (str(p), p.stat().st_mtime, cfg.part_low_scale)
            if key not in _PART_LOW:
                m = trimesh.load(str(p), force="mesh")
                m.apply_scale(cfg.part_low_scale)
                _PART_LOW.clear()
                _PART_LOW[key] = m
            return _PART_LOW[key]
        except (OSError, ValueError):
            pass  # missing / broken: the simplified copy of the visual mesh
    return low(part_mesh, "part", cfg, keep=cfg.part_keep)


# R11 (review fixes, rebased onto v12's config): the same copies plus what they are
# (core.collision_mesh.MeshInfo), for CollisionScene.geometry_report and the report's deviation note.
def exact_info(n_faces: int, mode: str = "visual"):
    """MeshInfo of a mesh used as it is (no simplified copy)."""
    from core.collision_mesh import MeshInfo

    return MeshInfo(mode, "visual", "exact", int(n_faces), int(n_faces))


def low_info(mesh: trimesh.Trimesh, component: str, cfg: CollisionConfig, keep=None, auto_faces=None):
    """(collision copy, MeshInfo) of a mesh of this component (``low`` plus the measured deviation)."""
    from core.collision_mesh import collision_mesh_info

    return collision_mesh_info(mesh, geom_of(cfg, component), keep=keep, auto_faces=auto_faces)


def part_low_info(part_mesh: trimesh.Trimesh, cfg: CollisionConfig):
    """``part_low`` plus its MeshInfo; the package builder's own collision mesh counts as exact
    (``used`` = "package")."""
    if cfg.part_low and Path(cfg.part_low).is_file():
        from core.collision_mesh import MeshInfo

        own = part_low(part_mesh, cfg)  # falls back to the simplified copy when it cannot be read
        if cfg.part_low and (str(Path(cfg.part_low)), cfg.part_low_scale) in {
            (k[0], k[2]) for k in _PART_LOW
        }:
            return own, MeshInfo("package", "package", "exact", len(part_mesh.faces), len(own.faces))
    return low_info(part_mesh, "part", cfg, keep=cfg.part_keep)


def geom_fingerprint(cfg: CollisionConfig | None = None) -> str:
    """The collision-geometry identity for staleness stamps (every setting, ``CollisionConfig.key``)."""
    return f"{(cfg or current()).key()}|robot_auto={ROBOT_AUTO_FACES}"
