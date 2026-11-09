"""Helpers of the path build (ui.simulate): the thread's own collision checker and the carrier
moves between stops (v4 plan item 4; v6: also without a home pose)."""

from __future__ import annotations

from ui.session import Session


def path_checker(s: Session):
    """A FeasibilityChecker on its own Robot (Pinocchio data is not thread-safe), with the
    carrier bodies, the part, the floor and the stay-out zones."""
    from core.collision import CollisionScene
    from core.feasibility import FeasibilityChecker
    from core.ik import make_ik
    from core.robot import Robot

    robot = Robot(s.robot.spec)
    sites = s.sites()
    bb, wb = sites.scene_bodies() if sites is not None else ([], [])
    scene = CollisionScene(
        robot,
        s.tool_model,
        s.part_mesh(),
        s.T_world_base,
        s.part.T_world_part,
        base_bodies=bb,
        world_bodies=wb,
        obstacles=s.obstacle_entries(),  # stay-out zones
    )
    return FeasibilityChecker(robot, make_ik(robot), s.tool_model, scene, s.T_world_base, s.part.T_world_part)


def carrier_path(s: Session, sol, step: float, checker=None):
    """fn(k_from, k_to, arm_q) -> carrier poses [(values, T_world_base, clear)] between two
    stops, or None when the base has no carrier (static: nothing drives).

    The arm is held at ``arm_q`` (home, or backed off the last point) and checked against the
    scene on the way. With a lift, the lift moves at the stop and the carrier drives at one
    height: the lowest one where the way is clear (also for the arm), else the first clear one
    of the two stops' heights and the lift levels."""
    from core.carrier import lift_levels

    sites = s.sites() if s.base.mode == "movable" else None
    if sites is None or not hasattr(sites, "transit") or not all(st.carrier for st in sol.stations):
        return None
    lifted = "lift" in getattr(sites, "dofs", ())

    def clear(moves, arm_q) -> bool:
        if not all(ok for _, _, ok in moves):
            return False
        if checker is None or arm_q is None:
            return True
        for _, T, _ in moves:
            checker.set_base(T)
            if checker.scene.collides(arm_q):
                return False
        return True

    def path(a: int, b: int, arm_q=None):
        A, B = sol.stations[a].carrier, sol.stations[b].carrier
        if not lifted:
            return [(d, sites.T(d), ok) for d, ok in sites.transit(A, B, step)]
        lo, hi = sites.bounds[sites.dofs.index("lift")]
        heights = [lo, *(E["lift"] for E in (A, B) if "lift" in E), *lift_levels(lo, hi)]
        first = None
        for h in dict.fromkeys(round(float(h), 6) for h in heights):
            moves = [(d, sites.T(d), ok) for d, ok in sites.transit_staged(A, B, step, h)]
            if first is None:
                first = moves
            if clear(moves, arm_q):
                return moves
        return first  # nothing clear: the lowest height, its problems flagged

    return path
