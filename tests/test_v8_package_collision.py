"""Collision geometry in packages (core.urdf_package: meshes/collision + the URDF's <collision>,
used by the tool and carrier collision checks; ui.package_collision) and the robot arm padding
(core.collision.ARM_PAD)."""

import numpy as np
import pytest
import trimesh

import core.collision_mesh as cm
from core import collision
from core.carrier import Carrier
from core.collision import CollisionScene
from core.collision_mesh import Geom
from core.robot import Robot
from core.schema import Base, EndEffector, Pose
from core.tool import ToolModel
from core.urdf_package import (
    Collision,
    Frame,
    Package,
    Part,
    collision_files,
    open_package,
    to_base,
    to_tool,
    write_package,
    zip_folder,
)


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "CACHE", tmp_path / "cache")
    cm._MEMO.clear()
    yield
    cm._MEMO.clear()


def _detailed(folder):
    """A detailed mesh in mm: a fine sphere with a small box (two components)."""
    s = trimesh.creation.icosphere(subdivisions=5, radius=60.0)
    b = trimesh.creation.box((20, 20, 20))
    b.apply_translation((70, 0, 0))
    trimesh.util.concatenate([s, b]).export(folder / "head.stl")


def _tool_pkg(mode="reduce"):
    return Package(
        kind="tool",
        name="spindle",
        parts=[
            Part(name="head", stage="tool", shape="mesh", mesh="head.stl", units="mm"),
            Part(name="flange", stage="tool", shape="cylinder", radius=0.04, height=0.02),
        ],
        tcps=[Frame(name="tip", pose=Pose(xyz=(0, 0, 0.1)))],
        collision=Collision(mode=mode, faces=800, wrap_mm=4.0),
    )


def test_package_writes_collision_meshes_and_reads_back(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _detailed(src)
    urdf = write_package(_tool_pkg(), src, tmp_path / "spindle")
    low = tmp_path / "spindle" / "meshes" / "collision" / "head.stl"
    assert low.is_file() and len(trimesh.load(low).faces) <= 1000
    text = urdf.read_text()
    assert 'filename="meshes/head.stl"' in text and 'filename="meshes/collision/head.stl"' in text
    assert collision_files(_tool_pkg(), tmp_path / "spindle") == {"head": "collision/head.stl"}
    # the tool uses the package's collision mesh; the viewer keeps the visual one
    tm = ToolModel(to_tool(_tool_pkg(), str(urdf)), urdf)
    hl = tm.link("mount")
    assert hl.collision is not None and len(hl.collision.faces) < len(hl.mesh.faces) / 5
    sc = CollisionScene(Robot.from_registry("ur20"), tm, None, np.eye(4), None)
    tris = sum(sc.gm.geometryObjects[g].geometry.num_tris for g in sc.tool_ids)
    assert tris == len(hl.collision.faces)
    # reopened: the full visual mesh as the part, the collision settings as authored
    pkg, _ = open_package(zip_folder(urdf.parent), tmp_path / "reopen")
    head = next(x for x in pkg.parts if x.name == "head")
    assert pkg.collision.mode == "reduce" and pkg.collision.faces == 800
    m = trimesh.load(tmp_path / "reopen" / "spindle" / "meshes" / head.mesh) if head.shape == "mesh" else None
    assert m is not None and len(m.faces) > 20000


def test_visual_mode_and_per_part_override(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _detailed(src)
    p = _tool_pkg("visual")
    urdf = write_package(p, src, tmp_path / "a")
    assert not (tmp_path / "a" / "meshes" / "collision").exists() and "collision/" not in urdf.read_text()
    p = p.model_copy(update={"parts": [p.parts[0].model_copy(update={"collision": "hull"}), p.parts[1]]})
    write_package(p, src, tmp_path / "b")
    hull = trimesh.load(tmp_path / "b" / "meshes" / "collision" / "head.stl")
    assert hull.is_convex


def test_package_padding(tmp_path):
    from core.urdf_package import part_collision

    src = tmp_path / "src"
    src.mkdir()
    _detailed(src)
    p = _tool_pkg("reduce").model_copy(update={"collision": Collision(mode="reduce", faces=800, pad_mm=10.0)})
    box = Part(name="box", stage="tool", shape="box", size=(0.2, 0.1, 0.05))
    m = part_collision(p, box, src)
    assert np.allclose(m.bounds, [[-0.11, -0.06, -0.01], [0.11, 0.06, 0.06]])  # bottom face on z = 0
    head = part_collision(p, p.parts[0], src)
    plain = part_collision(
        p.model_copy(update={"collision": Collision(mode="reduce", faces=800)}), p.parts[0], src
    )
    assert np.all(head.bounds[1] - plain.bounds[1] > 0.009)  # pushed out by the padding
    urdf = write_package(p.model_copy(update={"parts": [*p.parts, box]}), src, tmp_path / "pad")
    assert set(collision_files(p.model_copy(update={"parts": [*p.parts, box]}), tmp_path / "pad")) == {
        "head",
        "box",
        "flange",
    }
    tm = ToolModel(to_tool(p, str(urdf)), urdf)
    hl = tm.link("mount")
    assert np.all(hl.collision.bounds[0] < hl.mesh.bounds[0] - 0.009)  # the whole tool padded, in place
    assert np.all(hl.collision.bounds[1] > hl.mesh.bounds[1] + 0.009)


def test_carrier_package_collision_meshes(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _detailed(src)
    p = Package(
        kind="rail",
        name="rail",
        parts=[
            Part(name="track", stage="fixed", shape="box", size=(3, 0.3, 0.1)),
            Part(name="sled", stage="carriage", shape="mesh", mesh="head.stl", units="mm"),
        ],
        x=(-1.0, 1.0),
        collision=Collision(mode="wrap", faces=600, wrap_mm=5.0),
    )
    dest = tmp_path / "rail"
    write_package(p, src, dest)
    coll = collision_files(p, dest)
    assert set(coll) == {"sled"}
    base = to_base(p, str(dest / "meshes"), Base(), coll)
    sled = next(b for b in base.bodies if b.name == "sled")
    assert sled.collision_mesh.endswith("collision/sled.stl")
    c = Carrier(base)
    body = next(b for b in c._bodies("carriage", np.eye(4)) if b.name == "sled")
    assert body.coll is not None and len(body.coll.faces) < len(body.mesh.faces)
    assert np.all(body.coll.bounds[1] >= body.mesh.bounds[1])  # the wrap is around the visual


def test_arm_padding(monkeypatch):
    r = Robot.from_registry("ur20")
    part = trimesh.creation.box((0.4, 0.4, 0.4))
    T_part = np.eye(4)
    T_part[:3, 3] = (0.9, 0.0, 0.5)
    monkeypatch.setattr(collision, "ARM_PAD", 0.0)
    sc = CollisionScene(r, None, part, np.eye(4), T_part)
    q = np.radians([0.0, -90.0, 60.0, -60.0, -90.0, 0.0])
    if sc.collides(q):
        pytest.skip("pose collides already")
    d = sc.part_clearance(q)
    monkeypatch.setattr(collision, "ARM_PAD", d - sc.margins.clearance + 0.01)
    padded = CollisionScene(r, None, part, np.eye(4), T_part)
    assert padded.collides(q)  # the bloated arm reaches the part
    reqs = padded.gd.collisionRequests
    assert all(reqs[k].security_margin == 0.0 for k in padded.pairs.robot_self)  # not against itself
    state = collision.geom_state()
    assert len(state) == 5 and state[2] == pytest.approx(collision.ARM_PAD)
    collision.set_geom_state(({"robot": Geom("auto")}, ()))  # an older 2-part state: no padding
    assert collision.ARM_PAD == 0.0


def test_builder_collision_folder(tmp_path):
    import viser

    from ui.package_builder import PackageBuilder

    class S:  # the builder only needs a project folder
        project_dir = tmp_path
        robot_name = "ur20"

    server = viser.ViserServer(host="127.0.0.1", port=8833, verbose=False)
    try:
        b = PackageBuilder(server, S())
        (b.work / "meshes").mkdir(parents=True)
        _detailed(b.work / "meshes")
        b._add(Part(name="head", stage="tool", shape="mesh", mesh="head.stl", units="mm"))
        pc = b.collision
        assert pc.part_mode.visible
        pc.mode.value = cm.MODE_LABELS["hull"]
        assert b.pkg.collision.mode == "hull"
        pc.part_mode.value = cm.MODE_LABELS["reduce"]
        assert b.pkg.parts[b.sel].collision == "reduce"
        pc.show.value = False
        pc.show.value = True
        assert not pc._handles  # v9 #4: nothing is computed until Simplify
        pc.simplify()
        assert len(pc._handles) == 1 and "triangles" in pc.info.content
        b.load_widgets()
        assert pc.part_mode.value == cm.MODE_LABELS["reduce"] and pc.mode.value == cm.MODE_LABELS["hull"]
        _ = EndEffector  # (schema import kept for readers: tools come from to_tool)
    finally:
        server.stop()
