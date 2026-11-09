"""v4 plan item 3: each stop's points form one connected area on the part (no overlap between
stops), and the part surface is painted by stop / group in the station colours."""

import numpy as np
import pytest
import viser

from opt.areas import _proposals, components, grow, neighbour_graph, split_stops

LINE = np.array([(x, 0.0, 0.0) for x in np.linspace(-3, 3, 25)])


def test_graph_and_pieces():
    adj = neighbour_graph(LINE)
    assert all(len(a) == (1 if i in (0, 24) else 2) for i, a in enumerate(adj))  # a chain
    label = np.array([0] * 11 + [1] * 3 + [2] * 4 + [1] * 4 + [2] * 3)
    assert [len(c) for c in components(list(np.flatnonzero(label == 1)), adj)] == [4, 3]
    assert split_stops(label, adj) == [1, 2]
    prop = _proposals(label, adj)
    assert prop  # pieces propose to join their neighbours


def test_region_growing_gives_contiguous_stops():
    """The mobile example's 3-stop plan, reduced to a line: interleaved assignment, each stop
    reaching an interval; growing from the point nearest each base gives one piece each."""
    adj = neighbour_graph(LINE)
    label0 = np.array([0] * 11 + [1] * 3 + [2] * 4 + [1] * 4 + [2] * 3)
    reach_iv = {0: (0, 11), 1: (10, 22), 2: (13, 24)}  # index ranges each stop reaches
    bases = np.array([(-1.7, -1, 0), (0.99, -1, 0), (1.79, -1, 0)])

    def reach(i, k):
        lo, hi = reach_iv[k]
        return object() if lo <= i <= hi else None

    label = grow(reach, LINE, bases, label0, adj)
    assert split_stops(label, adj) == [] and set(label) == {0, 1, 2}
    assert all(reach(i, int(k)) is not None for i, k in enumerate(label))  # only reachable moves
    assert np.all(np.diff(label) >= 0)  # stops in base order along the part


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8805, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


def _pieces(s, sol):
    P = {p.id: np.array(p.p) for p in s.targets.points}
    ids = sorted(P)
    adj = neighbour_graph(np.array([P[i] for i in ids]))
    where = {p: i for i, p in enumerate(ids)}
    return [
        len(components([where[p] for p, k in sol.assignment.items() if k == j], adj))
        for j in range(len(sol.stations))
    ]


def test_mobile_plans_have_one_area_per_stop(app):
    s = app["session"]
    assert app["project"].load_example("mobile_amr")
    stp = app["stations"]
    assert stp.areas.value
    stp.start(block=True)
    multi = [r for r in s.results if len(r.stations) >= 3]
    assert multi
    for sol in s.results:
        assert not sol.unreached and not sol.split_stops
        assert _pieces(s, sol) == [1] * len(sol.stations)
        res = s.result_points[sol.label]
        assert all(res[p].q is not None for p in sol.assignment)  # every moved point has its joints
    # the colour layer: one painted mesh per stop, in the station colours, on the part
    from ui.results import STATION_COLORS

    r = app["results"]
    sol = multi[0]
    r.choice.value = next(o for o in r.choice.options if sol.label in o)
    r.draw()
    names = {h.name.rsplit("/", 1)[-1] for h in r._areas}
    # v6: an unzoned stop has one group per joint configuration (the main one in its colour)
    from ui.station_areas import config_groups

    groups = config_groups(sol, r._configs(sol))
    want = {f"S{sol.assignment[p] + 1}_G{g + 1}" for p, g in groups.items()}
    assert names == want and {f"S{k + 1}_G1" for k in range(len(sol.stations))} <= names
    for h in r._areas:
        k = int(h.name.rsplit("/S", 1)[-1].split("_")[0]) - 1
        if h.name.endswith("_G1"):
            assert tuple(h.color) == STATION_COLORS[k % len(STATION_COLORS)]
        assert h.name.startswith("/cell/part/areas/")
    objs = app["objects"]
    objs.show["Results/Station areas"].value = False
    objs.apply_visibility()
    assert all(not h.visible for h in r._areas)
    objs.show["Results/Station areas"].value = True
    objs.apply_visibility()
    r.show_areas.value = False
    assert not r._areas
    r.show_areas.value = True
    assert r._areas


def test_areas_off_keeps_the_planner_assignment(app):
    s = app["session"]
    assert app["project"].load_example("mobile_amr")
    app["stations"].areas.value = False
    app["stations"].start(block=True)
    assert max(max(_pieces(s, sol)) for sol in s.results) > 1  # interleaved without it
    app["stations"].areas.value = True
