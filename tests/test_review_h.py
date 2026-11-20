"""Review finding R11: the "auto" collision copy of a large mesh (quadric decimation) is not
conservative; its deviation from the real surface is measured once (cached with the mesh),
reported per body (``CollisionScene.geometry_report``) and warned about
(``collision_mesh.deviation_warning``). Collision results themselves are unchanged."""

import numpy as np
import pytest
import trimesh

import core.collision_mesh as cm
from core import collision, collision_config
from core.collision import CollisionScene
from core.collision_mesh import Geom, collision_mesh
from core.robot import Robot


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "CACHE", tmp_path / "collision")  # never the PC's cache
    old = collision_config.current()
    collision_config.use(None)  # default collision settings (no kept faces)
    cm._MEMO.clear()
    yield
    cm._MEMO.clear()
    collision_config.use(old)


def _plate() -> trimesh.Trimesh:
    """review2/r2_decim.py's bossed plate (1 x 0.6 m, 40 x 25 bosses r 8 mm, h 15 mm) with
    6-sided bosses: 24k faces (r2_decim: 128k), above the auto threshold (20k). Decimated to
    4000 faces the bosses flatten: about 15 mm deviation."""
    parts = [trimesh.creation.box((1.0, 0.6, 0.02))]
    for x in np.linspace(-0.45, 0.45, 40):
        for y in np.linspace(-0.25, 0.25, 25):
            c = trimesh.creation.cylinder(radius=0.008, height=0.015, sections=6)
            c.apply_translation((x, y, 0.0175))
            parts.append(c)
    return trimesh.util.concatenate(parts)


def _brute(m: trimesh.Trimesh, low: trimesh.Trimesh) -> float:
    """Reference: exact distance to the collision copy from every vertex and 60k other samples."""
    pts = np.vstack([m.vertices, m.sample(60000, seed=7)])
    _, d, _ = trimesh.proximity.closest_point(low, pts)
    return float(d.max())


def test_decimated_part_deviation_measured_and_cached():
    m = _plate()
    assert len(m.faces) > cm.AUTO_FACES
    low, info = cm.collision_mesh_info(m, Geom())
    assert info.used == "reduce" and info.kind == "approximate" and info.mode == "auto"
    assert info.faces_before == len(m.faces) and info.faces_after == len(low.faces) < len(m.faces)
    ref = _brute(m, low)
    assert info.deviation_m > 0.005  # well over half a 10 mm clearance
    assert info.deviation_m == pytest.approx(ref, rel=0.05, abs=2e-4)
    assert any(cm.CACHE.glob("*.npz"))
    cm._MEMO.clear()  # from the disk cache: same mesh, same deviation, not measured again
    again, info2 = cm.collision_mesh_info(m, Geom())
    assert np.array_equal(again.faces, low.faces) and info2 == info
    assert collision_mesh(m, Geom()) is again  # the plain API is unchanged


def test_old_cache_file_without_deviation_is_measured():
    m = _plate()
    low, info = cm.collision_mesh_info(m, Geom())
    (path,) = cm.CACHE.glob("*.npz")
    np.savez_compressed(path, v=np.asarray(low.vertices, float), f=np.asarray(low.faces, np.int64))
    cm._MEMO.clear()
    _, info2 = cm.collision_mesh_info(m, Geom())
    assert info2.deviation_m == pytest.approx(info.deviation_m)
    with np.load(path) as z:
        assert "dev" in z.files  # written back


def test_exact_and_conservative_report_zero():
    small = trimesh.creation.box((0.1, 0.1, 0.1))
    out, info = cm.collision_mesh_info(small, Geom())
    assert out is small and info.kind == "exact" and info.deviation_m == 0.0
    _, info = cm.collision_mesh_info(_plate(), Geom("visual"))
    assert info.kind == "exact" and info.deviation_m == 0.0
    _, info = cm.collision_mesh_info(_plate(), Geom("hull"))
    assert info.kind == "conservative" and info.deviation_m == 0.0


def test_warning():
    _, info = cm.collision_mesh_info(_plate(), Geom())
    rep = {"part": {"component": "part", **info.as_dict()}}
    msg = cm.deviation_warning(rep, 0.010)
    assert msg is not None
    assert f"{info.deviation_m * 1000:.1f} mm" in msg and "clearance 10 mm" in msg
    assert "part" in msg and "Wrap (offset shell)" in msg and "optimistic" in msg
    assert cm.deviation_warning(rep, 4 * info.deviation_m) is None  # under half the clearance
    exact = {"part": {"component": "part", **cm.collision_mesh_info(_plate(), Geom("hull"))[1].as_dict()}}
    assert cm.deviation_warning(exact, 0.001) is None


T_PART = np.array([[1, 0, 0, 1.1], [0, 1, 0, 0], [0, 0, 1, 0.2], [0, 0, 0, 1.0]])
QS = [
    np.array(q)
    for q in (
        [0.0, -1.2, 1.4, -0.2, 1.57, 0.0],
        [0.0, -0.6, 1.2, -0.6, 1.57, 0.0],
        [0.2, -0.5, 1.0, -0.5, 1.57, 0.0],
        [0.0, -0.3, 0.9, -0.6, 1.57, 0.0],
    )
]


def test_scene_report_and_unchanged_results():
    part = _plate()
    sc = CollisionScene(Robot.from_registry("ur20"), None, part, np.eye(4), T_PART)
    rep = sc.geometry_report()
    p = rep["part"]
    assert p["component"] == "part" and p["used"] == "reduce" and p["deviation_m"] > 0.005
    assert p["faces_before"] == len(part.faces)
    for gid in sc.robot_ids:  # every robot mesh shape is listed
        name = sc.gm.geometryObjects[gid].name
        if hasattr(sc.gm.geometryObjects[gid].geometry, "num_tris"):
            assert rep[name]["component"] == "robot"
    assert cm.deviation_warning(rep, 0.010) is not None
    # the collision copy and the distances are what they were before the report existed: the
    # part's BVH is the plain quadric decimation to 4000 faces
    ref = part.simplify_quadric_decimation(face_count=4000)
    assert np.array_equal(sc.shapes[sc.part_id].faces, ref.faces)
    assert np.allclose(sc.shapes[sc.part_id].vertices, ref.vertices)
    d = [sc.part_clearance(q) for q in QS]
    # recorded with the code before the report existed (same scene, same queries)
    assert d == pytest.approx([0.455031722218453, 0.0, 0.044868377856850966, 0.0], abs=1e-9)
    assert [sc.collides(q) for q in QS] == [False, True, False, True]
    collision_config.update(geom={"part": Geom("reduce", 4000)})  # explicit reduce: the same body
    sc2 = CollisionScene(Robot.from_registry("ur20"), None, part, np.eye(4), T_PART)
    assert d == pytest.approx([sc2.part_clearance(q) for q in QS], abs=1e-12)
    assert [sc.collides(q) for q in QS] == [sc2.collides(q) for q in QS]


def test_geom_fingerprint_has_modes():
    a = collision.geom_fingerprint()
    collision_config.update(geom={"part": Geom("wrap")})
    b = collision.geom_fingerprint()
    assert a != b and "wrap" in b
