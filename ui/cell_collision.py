"""Live collision check of the jogged arm (v6): the robot and tool against the floor, the part,
the external axis (carrier) bodies and the stay-out zones, with the same pairs and margins as
the planner (core.collision). The scene is rebuilt only when the setup changes (robot, tool,
part, carrier, stay-out zones); a carrier move only re-places the base."""

from __future__ import annotations

import threading

import numpy as np


class CellCollision:
    def __init__(self, session) -> None:
        self.s = session
        self._key = None
        self._scene = None
        self._lock = threading.Lock()  # Viser callbacks run concurrently

    def _setup_key(self) -> str | None:
        from ui.staleness import components

        s = self.s
        if s.robot is None:
            return None
        c = components(s, tool_nominal=True)
        return "|".join((c["robot"], c["carrier"], c["tool"], c["part"], c["stayout"], str(id(s.robot))))

    def scene(self):
        key = self._setup_key()
        if key is None:
            return None
        if key != self._key:
            from core.collision import CollisionScene
            from core.robot import Robot

            s = self.s
            sites = s.sites() if s.base.mode == "movable" else None
            bb, wb = sites.scene_bodies() if sites is not None else ([], [])
            part = s.part
            self._scene = CollisionScene(
                Robot(s.robot.spec),  # its own Pinocchio data (callbacks run on a thread pool)
                s.tool_model,
                s.part_mesh() if part is not None else None,
                s.T_world_base,
                part.T_world_part if part is not None else None,
                base_bodies=bb,
                world_bodies=wb,
                obstacles=s.obstacle_entries(),
            )
            self._key = key
        return self._scene

    def hits(self, q=None, tcp: str | None = None) -> list[tuple[str, str]]:
        """Colliding (robot / tool, other) pairs at joints ``q`` (default: the current pose);
        ``tcp``: that TCP's contact faces may touch the part."""
        s = self.s
        with self._lock:
            try:
                sc = self.scene()
            except Exception:  # a broken setup must not break jogging
                return []
            if sc is None or (q is None and s.q is None):
                return []
            sc.set_base(np.asarray(s.T_world_base, float))
            return sc.colliding_pairs(np.asarray(s.q if q is None else q, float), tcp)


def describe(pairs: list[tuple[str, str]]) -> str:
    """'wrist_3_link ↔ floor, tool ↔ carrier_rail_a' (first three)."""
    names = [f"{a.removesuffix('_0')} ↔ {b.removeprefix('carrier_')}" for a, b in pairs[:3]]
    more = f" (+{len(pairs) - 3})" if len(pairs) > 3 else ""
    return ", ".join(names) + more
