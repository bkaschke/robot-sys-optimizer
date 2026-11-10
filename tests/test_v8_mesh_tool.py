"""v8 follow-up: a tool from one STL / STEP file; STEP datums place the TCP and the mount."""

import numpy as np
import pytest
import trimesh
import viser

from core.step_datums import length_unit, read_datums
from core.tool_import import ToolImportError, import_mesh_tool, load_datums
from tests.step_fixture import probe_step


def test_step_datums_and_units(tmp_path):
    text = probe_step(tmp_path).read_text()
    ds = {d.name: d.T for d in read_datums(text)}
    assert set(ds) == {"CSYS_TCP", "CSYS_FLANGE"}
    assert np.allclose(ds["CSYS_TCP"][:3, 3], (0, 0, 0.13)) and np.allclose(ds["CSYS_TCP"][:3, 2], (0, 0, 1))
    assert np.allclose(ds["CSYS_FLANGE"][:3, 2], (0, 0, -1))
    for T in ds.values():  # proper rotations
        assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3)) and np.linalg.det(T[:3, :3]) == pytest.approx(
            1
        )
    assert length_unit("SI_UNIT(.MILLI.,.METRE.)") == 0.001
    assert length_unit("CONVERSION_BASED_UNIT('INCH',#5)") == 0.0254
    assert length_unit("SI_UNIT($,.METRE.)") == 1.0


def test_import_step_and_stl(tmp_path):
    p = probe_step(tmp_path)
    urdf, ds = import_mesh_tool("probe.step", p.read_bytes(), tmp_path / "a", "probe")
    m = trimesh.load(urdf.parent / "head.stl")
    assert np.allclose(m.bounds, [[-0.03, -0.03, -0.01], [0.03, 0.03, 0.13]], atol=1e-4)
    assert [d.name for d in load_datums(urdf)] == [d.name for d in ds] == ["CSYS_TCP", "CSYS_FLANGE"]
    box = trimesh.creation.box((50, 50, 200))  # mm
    data = trimesh.exchange.stl.export_stl(box)
    urdf2, ds2 = import_mesh_tool("box.stl", data, tmp_path / "b", "box", "mm")
    assert not ds2 and not load_datums(urdf2)
    assert np.allclose(trimesh.load(urdf2.parent / "head.stl").extents, (0.05, 0.05, 0.2))
    with pytest.raises(ToolImportError, match="units"):
        import_mesh_tool("box.stl", data, tmp_path / "c", "box", "m")  # 200 m long: a unit mistake
    with pytest.raises(ToolImportError):
        import_mesh_tool("x.txt", b"hi", tmp_path / "d", "x")


def test_gui_mesh_tool_with_datums(tmp_path):
    from app import build_app
    from core.transforms import pose_to_matrix

    server = viser.ViserServer(host="127.0.0.1", port=8890, verbose=False)
    try:
        app = build_app(server)
        s, tool = app["session"], app["tool"]
        panel = tool.mesh_panel
        assert not panel.set_tcp.visible
        p = probe_step(tmp_path)
        assert panel.add_file("probe.step", p.read_bytes())
        assert s.tool.name == "probe" and s.tool.extension.type == "pose"
        assert s.tool_model.length() == pytest.approx(0.0)  # the mesh sits on the mount
        assert panel.set_tcp.visible and set(panel.tcp_datum.options) == {"CSYS_TCP", "CSYS_FLANGE"}
        panel.tcp_datum.value = "CSYS_TCP"
        assert panel.apply_tcp()
        T = pose_to_matrix(s.tool.tcps[0].T_head_tcp)
        assert np.allclose(T[:3, 3], (0, 0, 0.13)) and np.allclose(T[:3, 2], (0, 0, 1))
        tip = s.tool.tcps[0].contact_faces.get("head", [])
        assert tip  # the faces at the probe tip may touch the part
        hm = next(h for h in s.tool_model.links if h.name == "head").mesh
        assert np.all(
            np.asarray(hm.vertices)[np.asarray(hm.faces)[tip]].max(axis=1)[:, 2] >= 0.13 - 0.005 - 1e-9
        )
        panel.tcp_flip.value = True
        assert panel.apply_tcp()
        assert np.allclose(pose_to_matrix(s.tool.tcps[0].T_head_tcp)[:3, 2], (0, 0, -1))
        panel.mount_datum.value = "CSYS_FLANGE"
        panel.mount_flip.value = True  # its z points into the robot: flip it
        assert panel.apply_mount()
        T_fm = pose_to_matrix(s.tool.T_flange_mount)
        assert np.allclose(T_fm[:3, 3], (0, 0, 0.01), atol=1e-9)  # the plate's bottom face on the flange
        assert np.allclose(T_fm[:3, 2], (0, 0, 1))
        panel.tcp_flip.value = False
        assert panel.apply_tcp()
        # the probe tip TCP is 140 mm out from the flange, its z pointing away from the robot
        T_ft = s.tool_model.T_flange_tcp("tcp")
        assert np.allclose(T_ft[:3, 3], (0, 0, 0.14), atol=1e-9) and np.allclose(T_ft[:3, 2], (0, 0, 1))
        assert not panel.add_file("notes.txt", b"x")
    finally:
        server.stop()
