"""Live collision check of the jogged arm (v6): the robot and tool against the floor, the part,
the external axis (carrier) bodies and the stay-out zones, with the same pairs and margins as
the planner (core.collision). The scene is rebuilt only when the setup changes (robot, tool,
part, carrier, stay-out zones); a carrier move only re-places the base."""

from __future__ import annotations

import threading
import time

import numpy as np

RETRY_S = 5.0  # a failed scene build is retried this often even if the setup key is unchanged


class CollisionUnavailableError(Exception):
    """The collision scene could not be built; the message is the reason."""


class CellCollision:
    def __init__(self, session) -> None:
        self.s = session
        self._key = None
        self._scene = None
        self._lock = threading.Lock()  # Viser callbacks run concurrently
        self._fail: tuple[str, str, float] | None = None  # (setup key, reason, time) of a failed build
        self._logged: set[str] = set()  # failure reasons already written to the log
        self.error: str | None = None  # why the last check was unavailable (None = it ran)

    def _setup_key(self) -> str | None:
        from ui.staleness import components

        s = self.s
        if s.robot is None:
            return None
        from core.collision_config import current

        c = components(s, tool_nominal=True)
        geom = str(hash(current().key()))  # collision settings (2026-11-11; v12 #10: one value)
        # v10 #5: file contents (c["collision"], the meshes in the others) rebuild it when a
        # collision file changes on disk
        return "|".join(
            (
                c["robot"],
                c["carrier"],
                c["tool"],
                c["part"],
                c["stayout"],
                c["collision"],
                str(id(s.robot)),
                geom,
                c["margins"],  # review M4: the project's clearance / joint margins
            )
        )

    def scene(self):
        key = self._setup_key()
        if key is None:
            return None
        if key != self._key:
            if self._fail is not None and self._fail[0] == key and time.monotonic() - self._fail[2] < RETRY_S:
                raise CollisionUnavailableError(self._fail[1])
            from core.collision import CollisionScene
            from core.robot import Robot
            from ui.plan_job import checker_config

            s = self.s
            margins = checker_config(s)[0]  # the project's optimizer.margins
            sites = s.sites() if s.base.mode == "movable" else None
            bb, wb = sites.scene_bodies() if sites is not None else ([], [])
            part = s.part
            try:
                self._scene = CollisionScene(
                    Robot(s.robot.spec),  # its own Pinocchio data (callbacks run on a thread pool)
                    s.tool_model,
                    s.part_mesh() if part is not None else None,
                    s.T_world_base,
                    part.T_world_part if part is not None else None,
                    margins,
                    base_bodies=bb,
                    world_bodies=wb,
                    obstacles=s.obstacle_entries(),
                )
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                self._scene, self._key = None, None
                self._fail = (key, reason, time.monotonic())
                raise CollisionUnavailableError(reason) from e
            self._key, self._fail = key, None
        return self._scene

    def check(self, q=None, tcp: str | None = None) -> tuple[list[tuple[str, str]], str | None]:
        """(colliding pairs, None), or ([], reason) when the collision scene cannot be built:
        "unavailable" is not "no collision". A failed build is retried when the setup changes
        (and every RETRY_S otherwise); the failure is logged once per distinct reason."""
        s = self.s
        with self._lock:
            try:
                sc = self.scene()
            except CollisionUnavailableError as e:
                reason = str(e)
            except Exception as e:  # the setup key itself failed
                reason = f"{type(e).__name__}: {e}"
            else:
                self.error = None
                if sc is None or (q is None and s.q is None):
                    return [], None
                sc.set_base(np.asarray(s.T_world_base, float))
                return sc.colliding_pairs(np.asarray(s.q if q is None else q, float), tcp), None
            self.error = reason
            if reason not in self._logged:
                self._logged.add(reason)
                from ui.log_panel import log

                log(f"**Collision check unavailable**: {reason}", "error")
            return [], reason

    def hits(self, q=None, tcp: str | None = None) -> list[tuple[str, str]]:
        """Colliding (robot / tool, other) pairs at joints ``q`` (default: the current pose);
        ``tcp``: that TCP's contact faces may touch the part. [] also when the check is
        unavailable: use ``check`` or read ``error`` to tell."""
        return self.check(q, tcp)[0]


def describe(pairs: list[tuple[str, str]]) -> str:
    """'wrist_3_link ↔ floor, tool ↔ carrier_rail_a' (first three)."""
    names = [f"{a.removesuffix('_0')} ↔ {b.removeprefix('carrier_')}" for a, b in pairs[:3]]
    more = f" (+{len(pairs) - 3})" if len(pairs) > 3 else ""
    return ", ".join(names) + more
