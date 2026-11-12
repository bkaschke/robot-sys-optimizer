"""V9 phase 3 (issues #1, #3, #4, #5, #11): the package builder makes all geometry - part packages,
imported collision meshes, Simplify / Revert with a status, editing the project's tool / part /
external axis, and Use in this project."""

from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
import viser

import core.collision_mesh as cm
from core import collision
from core.urdf_package import (
    Collision,
    Package,
    Part,
    open_package,
    part_collision,
    write_package,
    zip_folder,
)

E = SimpleNamespace(client=None)


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "CACHE", tmp_path / "cache")
    cm._MEMO.clear()
    yield
    cm._MEMO.clear()


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8845, verbose=False)
    st = build_app(server)
    yield st
    st["builder_window"].stop()
    server.stop()


def _sphere(folder, name="ball.stl", r=50.0):
    trimesh.creation.icosphere(subdivisions=5, radius=r).export(folder / name)  # mm


def test_part_package_round_trip(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _sphere(src)
    p = Package(
        kind="part",
        name="ball",
        parts=[Part(name="ball", stage="part", shape="mesh", mesh="ball.stl", units="mm")],
    )
    urdf = write_package(p, src, tmp_path / "ball")
    text = urdf.read_text()
    assert '<link name="part">' in text and "<joint" not in text
    pkg, _ = open_package(zip_folder(urdf.parent), tmp_path / "reopen")
    assert pkg.kind == "part" and pkg.parts[0].stage == "part"


def test_imported_collision_mesh(tmp_path):  # #3
    src = tmp_path / "src"
    src.mkdir()
    _sphere(src)
    (src / "collision_src").mkdir()
    trimesh.creation.box((110.0, 110.0, 110.0)).export(
        src / "collision_src" / "ball.stl"
    )  # mm, like the part
    part = Part(
        name="ball",
        stage="tool",
        shape="mesh",
        mesh="ball.stl",
        units="mm",
        collision_file="collision_src/ball.stl",
    )
    p = Package(kind="tool", name="t", parts=[part], tcps=[], collision=Collision(pad_mm=0.0))
    m = part_collision(p, part, src)
    assert len(m.faces) == 12 and np.allclose(m.extents, 0.11)  # the user's mesh, in metres
    from core.urdf_package import Frame

    urdf = write_package(p.model_copy(update={"tcps": [Frame(name="tcp")]}), src, tmp_path / "t")
    pkg, _ = open_package(zip_folder(urdf.parent), tmp_path / "re")
    assert pkg.parts[0].collision_file == "collision_src/ball.stl"  # as authored (package.yaml)
    # a CAD URDF with its own collision mesh: imported as the part's collision file
    cad = tmp_path / "cad"
    (cad / "meshes").mkdir(parents=True)
    _sphere(cad / "meshes", "body.stl")
    trimesh.creation.box((0.1, 0.1, 0.1)).export(cad / "meshes" / "body_low.stl")  # metres
    (cad / "cad.urdf").write_text(
        '<robot name="cad"><link name="mount">'
        '<visual><geometry><mesh filename="meshes/body.stl" scale="0.001 0.001 0.001"/></geometry></visual>'
        '<collision><geometry><mesh filename="meshes/body_low.stl"/></geometry></collision>'
        '</link><link name="tcp_tip"/><joint name="j" type="fixed"><parent link="mount"/>'
        '<child link="tcp_tip"/></joint></robot>'
    )
    from core.urdf_package import read_urdf

    pkg = read_urdf(cad / "cad.urdf", "tool")
    body = pkg.parts[0]
    assert body.collision_file and (cad / "meshes" / body.collision_file).is_file()
    low = trimesh.load(cad / "meshes" / body.collision_file)
    assert np.allclose(low.extents, 100.0)  # converted to the part's units (mm)


def test_part_simplify_status_revert_and_use(app):  # #4, #5, #11
    from ui import package_edit

    assert app["project"].load_example("rail_long_part")
    s, b = app["session"], app["builder"]
    mesh_before = s.part.mesh_path
    assert package_edit.edit_part(b, E) and b.pkg.kind == "part"
    pc = b.collision
    pc.mode.value = cm.MODE_LABELS["hull"]
    assert "not simplified" in pc.info.content
    pc.simplify()
    assert "ready" in pc.info.content and len(pc.done) == 1
    pc.pad_mm.value = 3.0
    assert "out of date" in pc.info.content
    pc.revert_all()
    assert b.pkg.collision.mode == "visual" and b.pkg.collision.pad_mm == 0.0 and not pc.done
    pc.mode.value = cm.MODE_LABELS["hull"]
    pc.simplify()
    assert b._use(E)
    assert s.part.mesh_path == mesh_before  # same geometry: targets stay valid
    assert s.part.collision_mesh and s.part.collision_mesh.endswith("part_collision.stl")
    assert collision.PART_LOW and collision.PART_LOW.endswith("part_collision.stl")
    sc = app["cell"].jog_panel.coll.scene()
    hull = trimesh.load(collision.PART_LOW)
    assert sc.gm.geometryObjects[sc.part_id].geometry.num_tris == len(hull.faces)
    assert s.to_project().part.collision_mesh == s.part.collision_mesh
    # editing it again brings the package back with its settings
    assert package_edit.edit_part(b, E) and b.pkg.collision.mode == "hull"


def test_edit_tool_keeps_its_process(app):  # #5
    from ui import package_edit

    assert app["project"].load_example("rail_long_part")
    s, b = app["session"], app["builder"]
    ee = s.tool
    assert package_edit.edit_tool(b, E) and b.pkg.kind == "tool"
    assert [t.name for t in b.pkg.tcps] == [t.name for t in ee.tcps]
    assert b._use(E)
    new = s.tool
    assert new.urdf != ee.urdf and new.extension == ee.extension and new.name == ee.name
    assert [t.model_dump(exclude={"T_head_tcp"}) for t in new.tcps] == [
        t.model_dump(exclude={"T_head_tcp"}) for t in ee.tcps
    ]


def test_edit_carrier_and_tabs(app):  # #1
    from ui import package_edit
    from ui.package_builder import TABS

    assert app["project"].load_example("rail_long_part")
    s, b = app["session"], app["builder"]
    assert package_edit.edit_carrier(b, E)
    assert b.pkg.kind == "rail" and len(b.pkg.parts) == len(s.base.bodies)
    assert b.pkg.x == s.base.joints.x
    assert set(TABS.values()) == {"tool", "part", "rail", "gantry", "mobile"}
    b._set_kind("part")
    assert b.kind_md.content == "**Part** package"
    assert not b.tcp.visible and not b.mount_md.visible and not b.stage.visible
    assert not b.robot_preview.active  # no robot under a part
    b._set_kind("tool")
    assert b.tcp.visible
