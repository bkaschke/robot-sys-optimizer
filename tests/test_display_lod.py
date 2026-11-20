"""R1c: part_mesh cache and display level of detail (the drawing is simplified; picking is not)."""

import collections

import numpy as np
import trimesh
import viser
from viser.infra import _infra

from core import targets as tg
from ui import display_lod
from ui.session import PartState, Session


def _stl(subdiv: int) -> bytes:
    m = trimesh.creation.icosphere(subdivisions=subdiv, radius=0.2)
    return m.export(file_type="stl")


def _session(tmp_path, name="p.stl", subdiv=3):
    (tmp_path / name).write_bytes(_stl(subdiv))
    s = Session(project_dir=tmp_path)
    s.part = PartState(name, "m", np.eye(4))
    return s


def test_part_mesh_cache_equal_isolated_and_invalidated(tmp_path):
    s = _session(tmp_path)
    a, b = s.part_mesh(), s.part_mesh()
    assert a is not b and np.array_equal(a.vertices, b.vertices) and np.array_equal(a.faces, b.faces)
    a.apply_scale(10.0)  # a caller changing its mesh must not touch the cache
    assert np.allclose(s.part_mesh().vertices, b.vertices)
    s.part.units = "mm"  # other units: other key
    assert np.allclose(s.part_mesh().vertices, b.vertices * 1e-3)
    s.part.units = "m"
    n = len(s.part_mesh().faces)
    (tmp_path / "p.stl").write_bytes(_stl(2))  # the file changed: reloaded
    assert len(s.part_mesh().faces) < n
    # at most KEEP entries
    for i in range(4):
        (tmp_path / f"q{i}.stl").write_bytes(_stl(1))
        s.part = PartState(f"q{i}.stl", "m", np.eye(4))
        s.part_mesh()
    assert len(s._mesh_cache._d) <= display_lod.KEEP


def test_display_copy_threshold():
    m = trimesh.creation.icosphere(subdivisions=4)
    same, simp = display_lod.display_copy(m, limit=len(m.faces), target=1000)
    assert same is m and not simp
    out, simp = display_lod.display_copy(m, limit=len(m.faces) - 1, target=1000)
    assert simp and 500 < len(out.faces) < len(m.faces)


def test_big_part_is_drawn_simplified_and_picks_use_the_full_mesh(tmp_path, monkeypatch):
    from app import build_app

    real = display_lod.display_copy
    monkeypatch.setattr(display_lod, "display_copy", lambda m, *a, **k: real(m, limit=3000, target=1500))
    sent = []
    orig = _infra.WebsockMessageHandler.queue_message
    monkeypatch.setattr(
        _infra.WebsockMessageHandler,
        "queue_message",
        lambda self, msg: (sent.append(msg), orig(self, msg))[1],
    )
    server = viser.ViserServer(host="127.0.0.1", port=8996, verbose=False)
    try:
        app = build_app(server)
        cell, tab, s = app["cell"], app["targets"], app["session"]
        cell.upload_units.value = "m"
        content = _stl(4)  # 5120 faces
        cell.upload_part("ball.stl", content)
        meshes = [m for m in sent if type(m).__name__ == "MeshMessage" and "mesh" in m.name]
        assert meshes, "no mesh drawn"
        drawn = meshes[-1].props.faces
        full = s.part_mesh()
        assert len(full.faces) == 5120 and len(drawn) < 2500  # simplified on screen
        assert cell.lod_note.visible and "Display simplified" in cell.lod_note.content
        # picking: the click ray is cast on the full mesh; face ids are full-mesh ids
        tab.click_mode.value = "Fill +"
        s.part.T_world_part = np.eye(4)  # world rays are part rays
        o, d = np.array([0.03, 0.02, 1.0]), np.array([0.0, 0.0, -1.0])
        want = tg.pick_face(full, o, d)
        assert want is not None and tab.mesh() is not None and len(tab.mesh().faces) == 5120
        assert tg.pick_face(tab.mesh(), o, d) == want
        ev = collections.namedtuple("E", "ray_origin ray_direction modifier client")(o, d, None, None)
        tab.on_part_click(ev)
        assert want in set(tab.selection.tolist()) and tab.selection.max() < 5120
        assert np.array_equal(tab.selection, np.unique(tg.grow_region(full, want, tab.surf.angle.value)))
        # a small part is drawn as it is and shows no note
        cell.upload_part("small.stl", _stl(2))
        assert not cell.lod_note.visible
    finally:
        server.stop()
