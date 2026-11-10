"""v8 T9: UR10e / UR30 in the registry and Optimize -> Compare robots."""

import pytest
import viser

from core.robot import Robot, list_robots
from tests.helpers import confirm_all


def test_new_robots_load_and_solve_ik():
    import numpy as np

    from core.ik import make_ik

    assert {"ur10e", "ur30", "ur20"} <= set(list_robots())
    for name in ("ur10e", "ur30"):
        r = Robot.from_registry(name)
        ik = make_ik(r)
        q = np.radians([10, -70, 60, -80, 70, 20])
        T = r.fk(q)
        sols = ik.solve(T)
        assert len(sols) and min(np.linalg.norm(r.fk(s)[:3, 3] - T[:3, 3]) for s in sols) < 1e-6


@pytest.mark.slow
def test_compare_two_robots_matches_normal_plans():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8858, verbose=False)
    try:
        app = build_app(server)
        assert app["project"].load_example("rail_long_part")
        st, cmp_, s = app["stations"], app["compare"], app["session"]
        for name, cb in cmp_.pick.items():
            cb.value = name in ("ur20", "ur10e")
        st.refine.value = False  # keep it short
        confirm_all(st)
        cmp_.start(block=True)
        rows = {r["robot"]: r for r in cmp_.rows}
        assert set(rows) == {"ur20", "ur10e"} and "| ur10e |" in cmp_.status.content
        assert s.robot_name == "ur20" and not s.results  # the live session is untouched
        st.start(block=True)  # a normal plan with the UR20 gives the same best plan
        assert s.results[0].label == rows["ur20"]["label"]
        assert s.results[0].cost == pytest.approx(rows["ur20"]["cost"])
        cmp_.choice.value = "ur10e"
        cmp_.use_robot("ur10e")
        assert s.robot_name == "ur10e"
        confirm_all(st)
        st.start(block=True)
        assert s.results[0].label == rows["ur10e"]["label"]
        assert s.results[0].cost == pytest.approx(rows["ur10e"]["cost"])
    finally:
        server.stop()
