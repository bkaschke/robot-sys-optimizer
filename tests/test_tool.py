"""M3 checks. Done-when: changing L in the GUI stretches the collision capsule and moves
the TCP correctly."""

from pathlib import Path

import numpy as np
import pytest
import trimesh
import viser

from core import tool as tool_mod
from core.mesh_select import grow_region, pick_face
from core.process import PRESETS, apply_preset
from core.schema import TCP, EndEffector, Extension, ExtensionValues, Pose
from core.tool import ToolModel, list_tools, load_tool, save_tool
from core.transforms import rpy_to_matrix

REPO = Path(__file__).resolve().parents[1]


def _sander() -> ToolModel:
    ee = load_tool("orbital_sander")
    return ToolModel(ee, REPO / ee.urdf)


# --- core geometry --------------------------------------------------------------------


def test_library_has_samples():
    assert {"orbital_sander", "scanner"} <= set(list_tools())


def test_tcp_moves_along_mount_z_with_L():
    tm = _sander()
    # tilt the mount so "along mount z" is not trivially flange z
    tm.ee = tm.ee.model_copy(update={"T_flange_mount": Pose(xyz=(0.01, 0, 0.02), rpy=(0.3, -0.2, 0.5))})
    z_mount = rpy_to_matrix((0.3, -0.2, 0.5))[:, 2]
    a = tm.T_flange_tcp("pad", ExtensionValues(L=0.10))
    b = tm.T_flange_tcp("pad", ExtensionValues(L=0.35))
    assert np.allclose(b[:3, 3] - a[:3, 3], 0.25 * z_mount)
    assert np.allclose(a[:3, :3], b[:3, :3])  # length changes never rotate the TCP


def test_bent_extension_geometry():
    tm = _sander()
    v = ExtensionValues(L=0.2, theta_deg=90.0, d=0.1)
    tm.ee = tm.ee.model_copy(
        update={
            "T_flange_mount": Pose(),
            "extension": tm.ee.extension.model_copy(update={"d": (0.0, 0.2), "theta_deg": (0, 90)}),
        }
    )
    T = tm.T_flange_head(v)
    # bend 90 deg about x at the end of L, then 0.1 m along the bent y (= flange +z)
    assert np.allclose(T[:3, 3], [0, 0, 0.3])
    assert np.allclose(T[:3, 2], [0, -1, 0])  # head z now points along flange -y
    segs = tm.extension_segments(v)
    assert len(segs) == 2 and np.isclose(tm.length(v), 0.3)


def test_straight_extension_ignores_theta_and_d():
    tm = _sander()
    tm.ee = tm.ee.model_copy(update={"extension": Extension(L=(0, 0.5), nominal=ExtensionValues(L=0.2))})
    assert np.allclose(tm.T_flange_head(ExtensionValues(L=0.2, theta_deg=30, d=0.1))[:3, 3], [0, 0, 0.22])


def test_capsule_spans_segment():
    tm = _sander()
    v = ExtensionValues(L=0.4)
    mesh = tm.extension_mesh(v)
    r = tm.ee.extension.radius
    z0 = tm.ee.T_flange_mount.xyz[2]
    assert np.isclose(mesh.bounds[0, 2], z0 - r, atol=1e-6)
    assert np.isclose(mesh.bounds[1, 2], z0 + 0.4 + r, atol=1e-6)
    assert tm.extension_mesh(ExtensionValues(L=0.0)) is None


def test_mass_properties():
    tm = _sander()
    mp = tm.mass_properties(ExtensionValues(L=0.15))
    assert np.isclose(mp.mass_kg, 2.9 + 2.0 * 0.15)
    # head COM: (2.5*0.075 + 0.4*0.16) / 2.9 above the head origin at z = 0.02 + 0.15
    head_com_z = 0.17 + (2.5 * 0.075 + 0.4 * 0.16) / 2.9
    ext_com_z = 0.02 + 0.075
    assert np.isclose(mp.com_flange[2], (2.9 * head_com_z + 0.3 * ext_com_z) / 3.2)
    tm.ee = tm.ee.model_copy(update={"mass_kg": 5.0, "com_head": (0, 0, 0.1)})
    assert np.isclose(tm.mass_properties().mass_kg, 5.0 + 0.3)


def test_nominal_outside_range_rejected():
    with pytest.raises(ValueError):
        Extension(L=(0.1, 0.2), nominal=ExtensionValues(L=0.5))
    with pytest.raises(ValueError):
        Extension(type="straight", theta_deg=(0, 30))


# --- picking / presets / library ----------------------------------------------------------


def test_region_growing_stops_at_sharp_edges():
    cyl = trimesh.creation.cylinder(radius=0.075, height=0.02, sections=48)
    top = pick_face(cyl, (0.01, 0.02, 1.0), (0, 0, -1))
    region = grow_region(cyl, top, 15.0)
    assert np.allclose(cyl.face_normals[region], [0, 0, 1])  # only the flat top
    side = pick_face(cyl, (1.0, 0.0, 0.0), (-1, 0, 0))
    wall = grow_region(cyl, side, 15.0)  # 7.5 deg between facets: whole wall
    assert np.allclose(cyl.face_normals[wall][:, 2], 0)
    assert len(wall) == 2 * 48
    assert pick_face(cyl, (1.0, 1.0, 1.0), (0, 0, 1)) is None


def test_contact_free_preset_clears_contacts():
    ee = load_tool("orbital_sander")
    scan = apply_preset(ee, "scanning")
    assert scan.tcps[0].contact_links == [] and scan.line_of_sight
    assert scan.tolerances.standoff_m == PRESETS["scanning"].tolerances.standoff_m
    sand = apply_preset(ee, "sanding")
    assert sand.process_force_N == 50.0 and sand.tcps[0].contact_links == ["pad_link"]


def test_library_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(tool_mod, "TOOLS_DIR", tmp_path)
    ee = EndEffector(name="t1", tcps=[TCP(name="a", contact_faces={"pad_link": [3, 1, 1]})])
    save_tool(ee)
    back = load_tool("t1")
    assert back.tcps[0].contact_faces == {"pad_link": [1, 3]}
    with pytest.raises(ValueError):
        save_tool(ee, "../escape")


# --- GUI: the M3 done-when check ------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8766, verbose=False)
    state = build_app(server)
    yield state
    server.stop()


def test_gui_L_stretches_capsule_and_moves_tcp(app):
    tool, s = app["tool"], app["session"]
    sl = tool._nominal["L"]
    r = s.tool.extension.radius
    T_wf = s.T_world_flange()
    z_mount_world = T_wf[:3, :3] @ rpy_to_matrix(s.tool.T_flange_mount.rpy)[:, 2]
    before = tool.view.T_world_tcp("pad")
    for L_mm in (100.0, 350.0, 600.0):
        sl.value = L_mm  # same path as the user dragging the slider
        assert np.isclose(s.tool.extension.nominal.L, L_mm / 1000)
        # capsule: its vertices span L + 2r along mount z (scene handle = what the browser draws)
        verts = np.asarray(tool.view.ext_handle.vertices)
        span = np.ptp(verts @ rpy_to_matrix(s.tool.T_flange_mount.rpy)[:, 2])
        assert np.isclose(span, L_mm / 1000 + 2 * r, atol=1e-6)
        # TCP: moved exactly by the L change along mount z, orientation unchanged
        T_tcp = tool.view.T_world_tcp("pad")
        expected = T_wf @ s.tool_model.T_flange_tcp("pad")
        assert np.allclose(T_tcp, expected, atol=1e-9)
    after = tool.view.T_world_tcp("pad")
    assert np.allclose(after[:3, 3] - before[:3, 3], (0.6 - 0.15) * z_mount_world, atol=1e-9)
    assert np.allclose(after[:3, :3], before[:3, :3])


def test_gui_tool_follows_robot(app):
    cell, tool, s = app["cell"], app["tool"], app["session"]
    cell.set_q(np.radians([20, -70, 80, -100, -90, 30]))
    expected = s.T_world_flange() @ s.tool_model.T_flange_tcp("pad")
    assert np.allclose(tool.view.T_world_tcp("pad"), expected, atol=1e-9)


def test_gui_pick_contact_faces(app):
    tool, s = app["tool"], app["session"]
    pad = s.tool_model.link("pad_link")
    face = int(np.argmax(pad.mesh.triangles_center[:, 2]))  # the working face, toward the part
    tool._on_pick("pad_link", face, None)
    faces = s.tool.tcps[0].contact_faces["pad_link"]
    assert np.allclose(pad.mesh.face_normals[faces], [0, 0, 1])  # flat pad face only
    assert "pad_link" in tool.view.overlay_handles or "pad_link" in s.tool.tcps[0].contact_links
    tool._on_pick("pad_link", face, None)  # clicking the same region again removes it
    assert "pad_link" not in s.tool.tcps[0].contact_faces


def test_gui_invalid_range_reverts(app):
    tool, s = app["tool"], app["session"]
    before = s.tool.extension.L
    tool.r_L.value = (400.0, 100.0)  # lo > hi
    assert s.tool.extension.L == before
