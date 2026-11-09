"""Joint-configuration zoning: labels, exact zone cover, path order, wrist unwrapping, the
planner on a scene that forces configuration changes, exports and the GUI."""

from dataclasses import dataclass, replace

import numpy as np
import pytest

from core import golden_parts as gp
from core.carrier import Carrier
from core.collision import CollisionScene
from core.configs import WRAP_JOINTS, config_label, min_zones, order_path, unwrap_path
from core.feasibility import FeasibilityChecker
from core.ik import make_ik
from core.ik.ur import solve_batch
from core.robot import Robot
from core.schema import TargetGroup, Weights
from core.tool import ToolModel, load_tool
from core.transforms import make_T
from opt.placement import PlannerSettings, Targets, plan_stations
from opt.reachability import map_for_tool
from opt.sites import CarrierSites
from tests.test_carriers import REPO, TASKS, rail_base

# --- core ---------------------------------------------------------------------------------------


def test_labels_identify_ik_branches():
    robot = Robot.from_registry("ur20")
    ik = make_ik(robot)
    rng = np.random.default_rng(3)
    for _ in range(200):
        Q, V = solve_batch(ik, robot.fk(rng.uniform(-np.pi, np.pi, 6))[None])
        labels = [config_label(ik, Q[0, b]) for b in range(8) if V[0, b]]
        assert len(set(labels)) == len(labels)  # every branch has its own label
        for b in np.flatnonzero(V[0]):
            q2 = Q[0, b].copy()
            q2[[0, 3, 5]] += 2 * np.pi  # a full turn of joints 1, 4, 6 is the same configuration
            assert config_label(ik, q2) == config_label(ik, Q[0, b])


def test_min_zones_is_exact():
    # greedy would take "A" (3 points) and then need B and C; the optimum is {B, C}
    options = {
        1: {"A": 1.0, "B": 0.5},
        2: {"A": 1.0, "B": 0.5},
        3: {"A": 1.0, "C": 0.5},
        4: {"B": 1.0},
        5: {"C": 1.0},
    }
    zones = dict(min_zones(options))
    assert set(zones) == {"B", "C"} and sorted(zones["B"] + zones["C"]) == [1, 2, 3, 4, 5]
    # tie between single configurations: the better total quality wins
    assert min_zones({1: {"A": 1.0, "B": 2.0}, 2: {"A": 1.0, "B": 2.0}})[0][0] == "B"
    assert min_zones({1: {}}) == []


def test_order_path_is_serpentine():
    xs, ys = np.meshgrid(np.arange(5) * 0.1, np.arange(3) * 0.05)
    P = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(15)])
    rng = np.random.default_rng(0)
    perm = rng.permutation(15)
    order = order_path(P[perm])
    path = P[perm][order]
    assert sorted(order.tolist()) == list(range(15))
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    assert np.isclose(steps.max(), 0.1) or np.isclose(steps.max(), 0.05)  # no long jumps back
    assert steps.sum() == pytest.approx(3 * 0.4 + 2 * 0.05)


def test_unwrap_path_avoids_wrist_spins():
    lo, hi = np.full(6, -2 * np.pi), np.full(6, 2 * np.pi)
    q = np.zeros(6)
    a, b = q.copy(), q.copy()
    a[5], b[5] = np.radians(170), np.radians(-175)  # 15 deg apart across the +-180 seam
    path, jumps = unwrap_path([a, b], lo, hi)
    assert jumps == 0 and np.degrees(abs(path[1][5] - path[0][5])) == pytest.approx(15)
    # joint 6 limited to +-180: the seam cannot be crossed, a 345 deg move remains
    hi2 = hi.copy()
    hi2[5], lo2 = np.pi, lo.copy()
    lo2[5] = -np.pi
    _, jumps = unwrap_path([a, b], lo2, hi2)
    assert jumps == 1 and 5 in WRAP_JOINTS


# --- planner ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rail_plans():
    """The M9 wall-side rail task, planned with and without zoning."""
    robot = Robot.from_registry("ur20")
    ee = load_tool("orbital_sander")
    tm = ToolModel(ee, REPO / ee.urdf)
    rmap = map_for_tool(robot, tm.T_flange_tcp("pad"))
    T_part, part = make_T(p=(0, 0.8, 0.5)), gp.long_rail()
    y, z, n = TASKS["wall_side"]
    xs = np.arange(-3.0, 3.001, 0.25)
    Pp = np.column_stack([xs, np.full_like(xs, y), np.full_like(xs, z)])
    Np = np.tile(n, (len(xs), 1))
    g = TargetGroup(name="w", tilt_deg=10.0)
    t = Targets(np.arange(len(xs)), Pp + T_part[:3, 3], Np, Pp, Np, [g] * len(xs), np.full(len(xs), 10.0))
    out = {}
    for zoning in (False, True):
        sites = CarrierSites(Carrier(rail_base()), part, T_part)
        bb, wb = sites.scene_bodies()
        scene = CollisionScene(robot, tm, part, np.eye(4), T_part, base_bodies=bb, world_bodies=wb)
        ch = FeasibilityChecker(robot, make_ik(robot), tm, scene, np.eye(4), T_part)
        st = replace(PlannerSettings(max_stations=4), zoning=zoning)
        plans = plan_stations(rmap, ch, t, np.eye(4), None, Weights(), st, sites=sites)
        out[zoning] = (plans, ch, sites, t)
    return out


def _configs_per_station(sol, res, ik):
    by: dict[int, set] = {}
    for pid, k in sol.assignment.items():
        by.setdefault(k, set()).add(config_label(ik, res[pid].q))
    return by


def test_scene_forces_changes_without_zoning(rail_plans):
    plans, ch, _, _ = rail_plans[False]
    sol, res = plans[0]
    changes = sum(len(v) - 1 for v in _configs_per_station(sol, res, ch.ik).values())
    assert changes >= 1 and not sol.zones and sol.breakdown["wrist_flip"] == 0.0
    # v6: an unzoned plan pays for its configuration changes (one per extra configuration)
    assert sol.breakdown["group"] == pytest.approx(Weights().group * changes)
    assert sol.breakdown["motion"] > 0


def test_zoning_removes_changes_and_stays_exact(rail_plans):
    plans, ch, sites, t = rail_plans[True]
    sol, res = plans[0]
    assert not sol.unreached and len(sol.stations) <= 4  # M9 regression count unchanged
    # every assigned point in exactly one zone of its own station, with that configuration
    seen = [p for z in sol.zones for p in z.points]
    assert sorted(seen) == sorted(sol.assignment)
    index = {int(pid): i for i, pid in enumerate(t.ids)}
    for zi, z in enumerate(sol.zones):
        ch.set_base(sites.T({d: sol.stations[z.station].carrier[d] for d in sites.dofs}))
        for seq, pid in enumerate(z.points):
            r = res[pid]
            assert sol.assignment[pid] == z.station and r.zone == zi and r.seq == seq
            assert config_label(ch.ik, r.q) == z.config == r.config
            assert np.all(r.q >= ch.lo - 1e-9) and np.all(r.q <= ch.hi + 1e-9)
            assert not ch.scene.collides(r.q, r.tcp)
            # the joint solution still puts the TCP on the target (tool z = -normal)
            T = ch.T_base_part @ np.eye(4)
            T_tcp = np.linalg.inv(T) @ ch.robot.fk(r.q) @ ch.T_flange_tcp[r.tcp]
            i = index[pid]
            assert np.linalg.norm(T_tcp[:3, 3] - (t.P_part[i] + r.standoff * t.N_part[i])) < 1e-6
            assert np.degrees(np.arccos(np.clip(-T_tcp[:3, 2] @ t.N_part[i], -1, 1))) <= 10.0 + 1e-6
    # this scene: one configuration per station, no changes at all
    switches = len(sol.zones) - len({z.station for z in sol.zones})
    jumps = sum(z.wrap_jumps for z in sol.zones)
    assert switches + jumps == 0 and sol.breakdown["group"] == sol.breakdown["wrist_flip"] == 0.0


def test_csv_and_report_follow_zones(rail_plans):
    from core import export as ex
    from core.report import ReportInput, build_report
    from core.schema import Targets as ProjectTargets
    from core.targets import append_points

    plans, ch, sites, t = rail_plans[True]
    sol, res = plans[0]
    pt = append_points(ProjectTargets(groups=[t.groups[0]]), t.P_part, t.N_part, "w")
    tm = ch.tool
    rows = ex.tcp_rows(sol, res, pt, ch.robot, tm, sites.c.base, np.eye(4))
    zoned = [r for r in rows if r["zone"] != ""]
    assert len(zoned) == len(sol.assignment)
    keys = [(r["station"], r["zone"], r["seq"]) for r in zoned]
    assert keys == sorted(keys)  # station -> zone -> path order
    assert {r["config"] for r in zoned} == {z.config for z in sol.zones}
    html = build_report(
        ReportInput(
            title="z",
            robot="ur20",
            tool="sander",
            process="sanding",
            base_mode="rail",
            solution=sol,
            rows=rows,
            station_poses=[ex.station_pose(sites.c.base, s, np.eye(4)) for s in sol.stations],
        )
    )
    assert "Joint-configuration zones" in html and sol.zones[0].config in html


# --- GUI ----------------------------------------------------------------------------------------


@dataclass
class FakeClick:
    ray_origin: tuple
    ray_direction: tuple
    modifier: str | None = None
    client = None


def test_gui_zoning_option_results_and_save(tmp_path):
    import viser

    from app import build_app
    from ui.session import Session, load_project, save_project
    from ui.stations import ZONING

    server = viser.ViserServer(host="127.0.0.1", port=8786, verbose=False)
    try:
        st = build_app(server)
        s, t = st["session"], st["targets"]
        s.targets = s.targets.model_copy(update={"points": []})
        t.on_part_click(FakeClick((1.0, 0.0, 2.0), (0.0, 0.0, -1.0)))
        t.method.value = "Projected grid"
        t.spacing.value = 200.0
        t.run_generate()
        stations = st["stations"]
        stations.zoning.value = ZONING[1]
        assert st["tool_opt"].planner_options()["zoning"] is True
        stations.start(block=True)
        sol = s.results[0]
        assert sol.zones and "Zones" in st["results"].summary.content
        res = s.result_points[sol.label]
        assert all(res[p].config and res[p].zone is not None for p in sol.assignment)
        t.color_by.value = "zone"  # colours from the selected solution's zones
        t.color_by.value = "configuration"
        save_project(s, tmp_path)
        s2 = Session()
        load_project(s2, tmp_path)
        assert s2.results[0].zones == sol.zones
        r2 = s2.result_points[sol.label]
        assert all(r2[p].zone == res[p].zone and r2[p].config == res[p].config for p in sol.assignment)
    finally:
        server.stop()
