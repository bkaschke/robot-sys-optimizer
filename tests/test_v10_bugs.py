"""V10 part A (docs/V10_PLAN.md tasks 1-6): bugs found in the v9 review."""

import time
from types import SimpleNamespace

import numpy as np
import pytest
import viser

from core.schema import TCP, Pose
from core.tool_contacts import carry_contacts
from tests.helpers import confirm_all

E = SimpleNamespace(client=None)


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8851, verbose=False)
    st = build_app(server)
    yield st
    st["builder_window"].stop()
    server.stop()


# --- task 1: edited tools keep their contact pad ------------------------------------------------


def test_carry_contacts_reports_what_is_lost():
    t = TCP(name="pad", contact_links=["pad_link", "gone"], contact_faces={"body": [1, 2], "pad_link": [3]})
    out, lost = carry_contacts([t], {"body": 10, "pad_link": 8}, {"pad_link": 8, "body": 12})
    assert out[0].contact_links == ["pad_link"] and out[0].contact_faces == {"pad_link": [3]}
    assert len(lost) == 2 and any("gone" in m for m in lost) and any("body" in m for m in lost)


def test_tool_package_writes_one_link_per_part(tmp_path):
    from core.tool import ToolModel
    from core.urdf_package import Collision, Frame, Package, Part, read_urdf, to_tool, write_package

    p = Package(
        kind="tool",
        name="t",
        parts=[
            Part(
                name="body",
                stage="tool",
                shape="cylinder",
                radius=0.04,
                height=0.15,
                pose=Pose(xyz=(0, 0, 0.1)),
            ),
            Part(name="pad", stage="tool", shape="box", size=(0.1, 0.1, 0.02), link="pad_link"),
        ],
        tcps=[Frame(name="tcp")],
        collision=Collision(pad_mm=1.0),
    )
    urdf = write_package(p, None, tmp_path / "t")
    links = {hl.name: hl for hl in ToolModel(to_tool(p, str(urdf)), urdf).links}
    assert set(links) == {"body", "pad_link"}
    # the collision copy sits where the visual does (yourdfpy node-name clash, found in v10)
    assert abs(links["body"].collision.bounds[1, 2] - 0.251) < 1e-6
    back = read_urdf(urdf, "tool")
    assert {x.link for x in back.parts} == {"body", "pad_link"}


@pytest.mark.slow
def test_edited_sander_keeps_its_pad(app):
    """Done when: the sander edited and used through the builder still reaches its points with the
    pad as its contact geometry."""
    from ui import package_edit

    s, b, opt = app["session"], app["builder"], app["optimize"]
    assert app["project"].load_example("rail_long_part")
    assert s.tool.tcps[0].contact_links == ["pad_link"]
    confirm_all(s)
    app["workflow"].refresh(force=True)
    app["stations"].start(block=True, quick=True)
    sol = app["results"].selected()
    assert sol is not None and not sol.unreached
    assert package_edit.edit_tool(b, E)
    assert {x.link for x in b.pkg.parts} == {"sander_body", "pad_link"}
    assert b._use(E) and not b.contact_lost
    ee = s.tool
    assert "packages" in ee.urdf and ee.tcps[0].contact_links == ["pad_link"]
    assert {hl.name for hl in s.tool_model.links} == {"sander_body", "pad_link"}
    opt.start(block=True)  # the check from the plan's stops
    assert all(r.status != "unreachable" for r in s.feasibility.values()), opt.summary.content
    # the pad really is what touches: without it as a contact the points collide
    s.set_tool(ee.model_copy(update={"tcps": [ee.tcps[0].model_copy(update={"contact_links": []})]}))
    opt.start(block=True)
    assert any(r.reason == "collision" for r in s.feasibility.values())


# --- task 2: saving no longer clears plans ------------------------------------------------------


def _example_copy(tmp_path, name="rail_long_part"):
    import shutil
    from pathlib import Path

    from ui.session import Session, load_project

    src = Path(__file__).resolve().parents[1] / "examples" / name
    shutil.copytree(src, tmp_path / name)
    s = Session()
    load_project(s, tmp_path / name)
    return s


def test_save_that_bundles_the_part_keeps_plans(tmp_path):
    """Done when: a save that bundles an outside part keeps the plans and feasibility results."""
    import shutil

    from ui import staleness
    from ui.session import save_project

    s = _example_copy(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.copy(s.resolve(s.part.mesh_path), outside / "my_part.stl")
    s.part.mesh_path = (outside / "my_part.stl").as_posix()
    from core.schema import Solution, Station

    s.results = [Solution(label="P", stations=[Station(xyz=(0, 0, 0))])]
    s.stamps = {"P": staleness.stamp_for(s, "plan"), "feasibility": staleness.stamp_for(s, "feasibility")}
    fresh: set[str] = set()
    assert not staleness.drop_stale_results(s, fresh) and fresh == {"P"}
    save_project(s)
    assert s.part.mesh_path.startswith("assets/imported/") and s.bundled
    assert staleness.solution_changes(s, s.results[0]) == []
    assert staleness.feasibility_changes(s) == []
    assert not staleness.drop_stale_results(s, fresh) and [r.label for r in s.results] == ["P"]
    # a real change of the file still counts
    m = s.part_mesh().copy()
    m.apply_translation((0.0, 0.0, 0.001))
    m.export(s.resolve(s.part.mesh_path))
    assert staleness.solution_changes(s, s.results[0]) == ["part"]


def test_v9_stamps_migrate(tmp_path):
    from ui import staleness

    s = _example_copy(tmp_path)
    old = staleness.components(s, legacy=True)
    s.stamps = {"P": dict(old)}
    # v9 steps had no "collision" component (v10 #5 adds it on load)
    s.steps = {"Cell": {k: old[k] for k in staleness.STEP_COMPONENTS["Cell"] if k in old}}
    assert staleness.changed(s.stamps["P"], staleness.components(s))  # paths vs contents
    assert staleness.migrate(s) > 0
    assert staleness.changed(s.stamps["P"], staleness.components(s)) == []
    assert staleness.step_state(s, "Cell") == "confirmed"


# --- task 3: no session swap while viewing a compared robot -------------------------------------


def test_snapshot_never_touches_the_session(tmp_path):
    """Done when: snapshot leaves every session field untouched (a concurrent reader)."""
    import threading

    from core.schema import Solution, Station

    s = _example_copy(tmp_path)
    own = [Solution(label="own plan", stations=[Station(xyz=(0, 0, 0))])]
    viewed = [Solution(label="UR10e plan", stations=[Station(xyz=(1, 0, 0))])]
    s.view_stash = {
        "robot": s.robot,
        "robot_name": "ur20",
        "robot_urdf": None,
        "q": s.q,
        "home_deg": None,
        "results": own,
        "result_points": {},
        "stamps": {},
    }
    s.robot_name, s.results = "ur10e", viewed  # what the compare view shows
    before = dict(s.__dict__)
    seen, stop = set(), threading.Event()

    def reader() -> None:
        while not stop.is_set():
            seen.add((s.robot_name, s.results[0].label))

    th = threading.Thread(target=reader)
    th.start()
    try:
        for _ in range(30):
            p = s.snapshot()
            assert p.robot.name == "ur20" and [r.label for r in p.results] == ["own plan"]
    finally:
        stop.set()
        th.join()
    assert seen == {("ur10e", "UR10e plan")}
    assert all(s.__dict__[k] is v for k, v in before.items())


# --- task 4: carrier edits keep "stretch with the axis" and collision meshes ---------------------


def test_rail_edit_round_trip_keeps_stretch(app, tmp_path):
    """Done when: the rail example's track still follows the travel after an edit round trip."""
    import trimesh

    from ui import package_edit

    s, b, cp = app["session"], app["builder"], app["carrier"]
    assert app["project"].load_example("rail_long_part")
    # give the carriage a mesh with its own collision mesh (metres), as the builder writes it
    box = trimesh.creation.box((0.5, 0.5, 0.15))
    box.apply_translation((0, 0, 0.075))
    (s.project_dir / "assets").mkdir(exist_ok=True)
    (box.copy().apply_scale(1000)).export(s.project_dir / "assets" / "carriage.stl")
    hull = trimesh.creation.box((0.6, 0.6, 0.2))
    hull.apply_translation((0, 0, 0.1))
    hull.export(s.project_dir / "assets" / "carriage_low.stl")
    bodies = list(s.base.bodies)
    k = next(i for i, x in enumerate(bodies) if x.name == "carriage")
    bodies[k] = bodies[k].model_copy(
        update={"shape": "mesh", "size": None, "mesh": "assets/carriage.stl", "units": "mm",
                "collision_mesh": "assets/carriage_low.stl"}
    )  # fmt: skip
    cp.load(s.base.model_copy(update={"bodies": bodies}))
    assert cp.commit(E)
    assert package_edit.edit_carrier(b, E)
    part = next(x for x in b.pkg.parts if x.name == "carriage")
    assert part.collision_file and next(x for x in b.pkg.parts if x.name == "track").stretch == "x"
    assert b._use(E)
    track = next(x for x in s.base.bodies if x.name == "track")
    assert track.stretch == "x" and s.base.axis_length_m == {"x": 6.6} and not s.base.manual_travel
    car = next(x for x in s.base.bodies if x.name == "carriage")
    low = s.carrier().collision_mesh(s.base.bodies.index(car))
    assert low is not None and abs(low.bounds[1, 2] - 0.2) < 1e-6  # metres, as before
    # the track and the travel still follow the rail length
    cp.len_x.value = 8000.0
    assert cp.commit(E)
    track = next(x for x in s.base.bodies if x.name == "track")
    assert (
        abs(track.size[0] - 8.0) < 1e-9 and abs(s.base.joints.x[1] - 3.7) < 1e-6
    )  # 4 - carriage 0.25 - end stop 0.05


# --- task 5: collision changes mark plans out of date --------------------------------------------


def test_collision_changes_flag_plans(app):
    """Done when: changing any of them flags plans, and the jog / left-panel overlay show the new
    part collision mesh."""
    import shutil

    import trimesh

    from core.schema import Solution, Station
    from ui import staleness

    s, coll, view = app["session"], app["collision"], app["collision_view"]
    assert app["project"].load_example("rail_long_part")
    assets = s.project_dir / "assets"
    assets.mkdir(exist_ok=True)
    # the part's own collision mesh, the tool in the project folder, a carrier body collision mesh
    low = s.part_mesh().convex_hull
    low.export(assets / "part_low.stl")
    s.part.collision_mesh = "assets/part_low.stl"
    shutil.copytree(s.resolve(s.tool.urdf).parent, assets / "sander")
    s.set_tool(s.tool.model_copy(update={"urdf": "assets/sander/orbital_sander.urdf"}))
    trimesh.creation.box((0.6, 0.6, 0.2)).export(assets / "carriage_low.stl")
    bodies = [
        x.model_copy(update={"collision_mesh": "assets/carriage_low.stl"}) if x.name == "carriage" else x
        for x in s.base.bodies
    ]
    s.base = s.base.model_copy(update={"bodies": bodies})
    coll.sync()
    sol = Solution(label="P", stations=[Station(xyz=(0, 0, 0))])
    s.results = [sol]

    def stamp() -> None:
        s.stamps["P"] = staleness.stamp_for(s, "plan")
        assert staleness.solution_changes(s, sol) == []

    stamp()
    coll.pad_mm.value = 7.0  # arm padding
    assert staleness.solution_changes(s, sol) == ["collision settings"]
    coll.pad_mm.value = 0.0
    assert staleness.solution_changes(s, sol) == []
    trimesh.creation.box((0.7, 0.6, 0.2)).export(assets / "carriage_low.stl")  # carrier
    assert staleness.solution_changes(s, sol) == ["external axis"]
    stamp()
    (assets / "sander" / "meshes" / "pad_link.stl").write_bytes(
        trimesh.creation.cylinder(radius=0.08, height=0.02).export(file_type="stl")
    )
    assert staleness.solution_changes(s, sol) == ["tool"]
    stamp()
    # the part's collision mesh: the plans are flagged, the jog scene and the overlay follow
    view.show["Part"].value = True
    view.check()
    sc = app["cell"].jog_panel.coll.scene()
    before = sc.shape(sc.part_id).bounds.copy()
    bigger = low.copy()
    bigger.apply_scale(1.5)
    bigger.export(assets / "part_low.stl")
    assert staleness.solution_changes(s, sol) == ["collision settings"]
    view.check()
    sc2 = app["cell"].jog_panel.coll.scene()
    assert sc2 is not sc and np.allclose(sc2.shape(sc2.part_id).bounds, before * 1.5)
    gid = sc2.part_id
    # the checkbox's own redraw runs late on viser's thread pool (CLAUDE.md): when check() comes while
    # it runs, ui.widgets.Latest hands the new redraw to that thread and returns at once (seen on a
    # loaded CI runner), so wait until the overlay shows this scene's mesh
    deadline = time.monotonic() + 10
    while not (gid in view._handles and view._drawn.get(gid) == id(sc2.shape(gid))):
        assert time.monotonic() < deadline, "the overlay never showed the new part collision mesh"
        time.sleep(0.05)
    view.show["Part"].value = False


# --- task 6: gantry placement fixes --------------------------------------------------------------


def test_fixed_bodies_against_keepouts_and_stayouts(tmp_path):
    from core.carrier import Carrier
    from core.schema import Keepout, Pose, Primitive
    from core.stayout import obstacles
    from opt.frame_place import fixed_clear

    s = _example_copy(tmp_path, "gantry_overhead")
    c = Carrier(s.base, resolve=s.resolve)
    assert fixed_clear(c) == ""
    # a floor keep-out at a post (post_3 stands at x 2.8, y 1.8)
    ko = s.base.model_copy(update={"keepouts": [Keepout(min_xy=(2.5, 1.5), max_xy=(3.2, 2.2))]})
    assert fixed_clear(Carrier(ko, resolve=s.resolve)) == "fixed bodies in a keep-out"
    # a stay-out zone next to beam_b (y 1.8, top at 4.4 m)
    zone = Primitive(shape="box", size=(0.5, 0.5, 0.5), name="lamp", pose=Pose(xyz=(0.0, 2.0, 4.2)))
    assert fixed_clear(c, obstacles([zone])) == "fixed bodies too close to the stay-out zone lamp"
    far = zone.model_copy(update={"pose": Pose(xyz=(0.0, 0.0, 0.0))})
    assert fixed_clear(c, obstacles([far])) == ""


def test_gantry_placement_fields(app):
    from ui import settings as ui_settings

    s, frame, picker = app["session"], app["frame"], app["picker"]
    assert app["project"].load_example("gantry_overhead")
    frame.refresh()
    assert frame.z.value == (-850.0, -850.0)
    frame.z.value = (-1200.0, -600.0)
    picker.choice.value = "Gantry placement"
    saved = ui_settings.collect(app)
    assert saved["frame.z"] == [-1200.0, -600.0] and saved["picker.choice"] == "Gantry placement"
    # another project: its own mount height
    assert app["project"].load_example("rail_long_part")
    z = s.base.T_mount.xyz[2] * 1000
    assert frame.z.value == (z, z)
    # a carrier change resets the range too; Use this frame (pose / mount only) does not
    frame.z.value = (100.0, 900.0)
    frame.watch()
    s.base = s.base.model_copy(update={"T_mount": s.base.T_mount.model_copy(update={"xyz": (0, 0, 0.5)})})
    frame.watch()
    assert frame.z.value == (100.0, 900.0)
    s.base = s.base.model_copy(update={"end_stop_m": 0.1})
    frame.watch()
    assert frame.z.value == (500.0, 500.0)
    # the saved values come back with the project
    ui_settings.apply(app, saved)
    assert frame.z.value == (-1200.0, -600.0) and picker.choice.value == "Gantry placement"
