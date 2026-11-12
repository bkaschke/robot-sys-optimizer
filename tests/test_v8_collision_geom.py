"""Low-detail collision geometry (core.collision_mesh, core.collision), the left panel's Collision
geometry (ui.collision_view) and the package builder window (ui.builder_window,
ui.package_robot)."""

import time

import numpy as np
import pytest
import trimesh
import viser

import core.collision_mesh as cm
from core import collision
from core.collision import CollisionScene
from core.collision_mesh import Geom, collision_mesh
from core.robot import Robot


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "CACHE", tmp_path / "collision")  # never the PC's cache
    cm._MEMO.clear()
    yield
    cm._MEMO.clear()


def _bumpy() -> trimesh.Trimesh:
    """A detailed mesh of several components: a fine sphere and two small boxes."""
    s = trimesh.creation.icosphere(subdivisions=5, radius=0.1)  # 20480 faces
    a = trimesh.creation.box((0.02, 0.02, 0.02))
    a.apply_translation((0.113, 0, 0))
    b = trimesh.creation.box((0.02, 0.02, 0.02))
    b.apply_translation((-0.113, 0, 0))
    return trimesh.util.concatenate([s, a, b])


def test_modes():
    m = _bumpy()
    assert collision_mesh(m, Geom("visual")) is m
    small = trimesh.creation.box((0.1, 0.1, 0.1))
    assert collision_mesh(small, Geom("auto")) is small  # below the auto threshold
    red = collision_mesh(m, Geom("reduce", faces=2000))
    assert len(red.faces) <= 2500 and np.allclose(red.bounds, m.bounds, atol=0.01)
    assert collision_mesh(m, Geom("auto", faces=3000)).faces.shape[0] <= 3500  # 20.5k > AUTO_FACES
    hull = collision_mesh(m, Geom("hull"))
    assert hull.is_convex and len(hull.faces) < len(m.faces)
    w = collision_mesh(m, Geom("wrap", faces=3000, wrap_mm=5.0))
    big = max(len(c.faces) for c in w.split(only_watertight=False))
    assert w.is_watertight and big > 0.97 * len(w.faces)  # one body around all three (the gaps closed)
    inside = trimesh.proximity.signed_distance(w, m.sample(2000, seed=1))
    assert inside.min() > 0  # every visual point inside the wrap (conservative)
    assert np.all(w.bounds[0] < m.bounds[0]) and np.all(w.bounds[1] > m.bounds[1])


def test_keep_faces_exact_and_cache():
    m = _bumpy()
    keep = np.arange(0, 400)
    out = collision_mesh(m, Geom("hull"), keep=keep)
    kept = m.triangles[keep].reshape(-1, 3)
    assert np.all(np.isclose(out.vertices[:, None, :], kept[None]).all(axis=2).any(axis=0))
    assert any(cm.CACHE.glob("*.npz"))
    cm._MEMO.clear()
    again = collision_mesh(m, Geom("hull"), keep=keep)  # from the disk cache
    assert np.allclose(again.vertices, out.vertices) and np.array_equal(again.faces, out.faces)


def test_scene_uses_the_collision_copies(monkeypatch):
    part = _bumpy()
    T_part = np.array([[1, 0, 0, 0.6], [0, 1, 0, 0], [0, 0, 1, 0.3], [0, 0, 0, 1.0]])
    monkeypatch.setattr(collision, "GEOM", {"part": Geom("hull")})
    monkeypatch.setattr(collision, "PART_KEEP", ())
    sc = CollisionScene(Robot.from_registry("ur20"), None, part, np.eye(4), T_part)
    assert sc.gm.geometryObjects[sc.part_id].geometry.num_tris <= 4000  # the default target
    monkeypatch.setattr(collision, "GEOM", {"part": Geom("visual")})
    sc = CollisionScene(Robot.from_registry("ur20"), None, part, np.eye(4), T_part)
    assert sc.gm.geometryObjects[sc.part_id].geometry.num_tris == len(part.faces)
    state = collision.geom_state()
    collision.set_geom_state(({"robot": Geom("hull")}, [3, 4]))
    assert collision.PART_KEEP == (3, 4) and collision.GEOM["robot"].mode == "hull"
    collision.set_geom_state(state)


def test_abb_robots_have_light_collision_meshes():
    """GoFa / PoWa URDFs use wrapped collision meshes (scripts/make_collision_meshes.py)."""
    for name in ("gofa10", "gofa12", "powa30"):
        r = Robot.from_registry(name)
        sc = CollisionScene(r, None, None, np.eye(4), None)
        tris = [g.geometry.num_tris for g in sc.gm.geometryObjects if hasattr(g.geometry, "num_tris")]
        assert max(tris) < 7000, (name, tris)
        rng = np.random.default_rng(0)
        Q = [r.random_q(rng) for _ in range(60)]
        t0 = time.perf_counter()
        for q in Q:
            sc.collides(q)
        assert (time.perf_counter() - t0) / len(Q) < 0.004  # was 8-16 ms with the CAD meshes


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8831, verbose=False)
    st = build_app(server)
    yield st
    st["builder_window"].stop()
    server.stop()


def test_collision_panel_shows_and_keeps(app):
    assert app["project"].load_example("rail_long_part")
    cv, cs, s = app["collision_view"], app["collision"], app["session"]
    assert not hasattr(cv, "mode") and not hasattr(cv, "pad_mm")  # the left panel only shows / hides
    for box in cv.show.values():
        box.value = True
    cv.redraw()
    sc = cv._scene()
    n_robot = len([g for g in sc.robot_ids if sc.shape(g) is not None])
    assert len(cv._handles) >= n_robot + 1  # robot links and the part at least
    gid = sc.robot_ids[-1]
    T = dict((g, T) for g, _, T in sc.world_shapes(np.asarray(s.q, float)))[gid]
    assert np.allclose(cv._handles[gid].position, T[:3, 3])
    # v9: Robot & cell only has the arm padding; kept faces of projects from before v9 still apply
    assert not hasattr(cs, "mode") and not hasattr(cs, "keep_selection")
    s.part.collision_keep = list(range(10))
    cs.sync()
    assert collision.PART_KEEP == tuple(range(10))
    assert s.to_project().part.collision_keep == list(range(10))
    s.part.collision_keep = []
    cs.pad_mm.value = 8.0  # robot arm padding (Robot & cell tab): checked and drawn bloated
    assert collision.ARM_PAD == pytest.approx(0.008)
    cv.redraw()
    sc = cv._scene()
    assert sc.arm_pad == pytest.approx(0.008)
    big = cv._pads[next(k for k in cv._pads if k[0] == gid)]
    assert np.all(big.bounds[1] > sc.shape(gid).bounds[1])
    cs.pad_mm.value = 0.0
    for box in cv.show.values():
        box.value = False
    cv.redraw()
    assert not cv._handles


def test_builder_has_its_own_window_and_robot(app):
    w, s = app["builder_window"], app["session"]
    assert w.port != app["server"].get_port() if "server" in app else w.port > 0
    b = w.builder
    assert b.server is w.server and b.robot_preview is not None
    tool_before = s.tool.model_dump_json() if s.tool is not None else None
    b._set_kind("tool")
    b._add(b.pkg.parts[0] if b.pkg.parts else __import__("core.urdf_package", fromlist=["Part"]).Part(
        name="box", stage="tool", shape="box", size=(0.1, 0.1, 0.1)))  # fmt: skip
    rp = b.robot_preview
    rp.joints[0].value = 30.0
    b.draw()
    T = rp.robot.fk(rp.q())
    assert np.allclose(b.root.position, T[:3, 3], atol=1e-9)  # the package rides on the flange
    assert (s.tool.model_dump_json() if s.tool is not None else None) == tool_before  # project untouched
    b._set_kind("rail")
    b.draw()
    assert rp.frame.visible
    from core.transforms import pose_to_matrix

    M = b.root_pose() @ pose_to_matrix(b.pkg.mount)
    assert np.allclose(rp.frame.position, M[:3, 3])
    b._set_kind("tool")
