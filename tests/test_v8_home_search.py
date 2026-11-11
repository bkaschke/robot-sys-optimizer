"""Clear home pose (core.home_search): a home that collides is replaced for the paths by a
folded pose that is clear at every stop."""

from types import SimpleNamespace

import numpy as np

from core.home_search import blocked_stops, choose_home, clear_home
from core.robot import Robot


def test_clear_home_folds_away_from_an_obstacle():
    r = Robot.from_registry("crx30ia")
    home = r.home_q()

    def collides(q) -> bool:  # anything reaching more than 0.6 m from the base collides
        return np.linalg.norm(r.fk(q)[:3, 3]) > 0.6

    assert collides(home)  # the CRX home stands the arm straight up (about 1.4 m)
    q = clear_home(r, collides, home)
    assert q is not None and not collides(q)
    assert np.all(q >= r.lower) and np.all(q <= r.upper)
    assert q[0] == home[0] and q[3] == home[3] and q[5] == home[5]  # only shoulder, elbow, wrist bend
    assert clear_home(r, lambda q: True, home) is None


def test_choose_home_keeps_a_clear_home_and_reports_a_replacement():
    r = Robot.from_registry("crx10ia")
    stops = [np.eye(4), np.eye(4)]
    state = {"T": None}

    def collides(q, tcp=None) -> bool:
        return np.linalg.norm(r.fk(q)[:3, 3]) > 0.5

    checker = SimpleNamespace(
        robot=r,
        set_base=lambda T: state.update(T=T),
        scene=SimpleNamespace(collides=collides),
    )
    folded = np.radians([0.0, 60.0, -120.0, 0.0, 0.0, 0.0])
    if not collides(folded):
        assert choose_home(checker, folded, stops) == (folded, "")
    q, note = choose_home(checker, r.home_q(), stops)
    assert not collides(q) and "clear folded pose" in note
    assert choose_home(checker, None, stops) == (None, "")
    paths = [
        SimpleNamespace(station=0, T_world_base=np.eye(4)),
        SimpleNamespace(station=3, T_world_base=np.eye(4)),
    ]
    assert blocked_stops(checker, r.home_q(), paths) == [0, 3]
    assert blocked_stops(checker, q, paths) == [] and blocked_stops(checker, None, paths) == []
