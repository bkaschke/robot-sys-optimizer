"""STEP import for parts and tool-surface samples (core.step_import)."""

import time

import numpy as np
import pytest

cq = pytest.importorskip("cadquery")

from core import step_import  # noqa: E402
from core.step_import import StepImportError, read_step  # noqa: E402

SIZE_MM = (100.0, 80.0, 40.0)


def _solid():
    """100 x 80 x 40 mm block, 20 x 20 x 10 pocket in the top, outer top edges filleted 4 mm."""
    box = cq.Workplane("XY").box(*SIZE_MM, centered=(True, True, False)).faces(">Z").edges().fillet(4)
    return box.faces(">Z").workplane().rect(20, 20).cutBlind(-10)


@pytest.fixture(scope="module")
def step_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("step") / "block.step"
    cq.exporters.export(_solid(), str(p))
    return p


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("RSO_DATA", str(tmp_path / "data"))


def test_size_in_metres_watertight_and_face_ids(step_file):
    t0 = time.perf_counter()
    sm = read_step(step_file, use_cache=False)
    dt = time.perf_counter() - t0
    m = sm.mesh
    print(f"read_step {len(m.faces)} triangles in {dt:.3f} s, tolerance {sm.linear_deflection_mm} mm")
    assert m.is_watertight
    assert np.allclose(m.bounds[1] - m.bounds[0], np.array(SIZE_MM) / 1000.0, atol=1e-6)
    assert m.volume > 0  # outward winding
    assert len(sm.face_ids) == len(m.faces)
    n_cad = len(_solid().val().Faces())
    assert sm.n_cad_faces == n_cad
    assert set(np.unique(sm.face_ids)) == set(range(n_cad))
    assert "plane" in sm.face_types and "cylinder" in sm.face_types
    assert sm.n_solids == 1
    assert sm.linear_deflection_mm <= step_import.MAX_CHORD_MM


def test_top_face_triangles_share_one_id(step_file):
    sm = read_step(step_file)
    m = sm.mesh
    z = m.vertices[m.faces][:, :, 2]
    flat_top = np.all(np.abs(z - 0.040) < 1e-7, axis=1)  # triangles lying in the z = 40 mm plane
    assert flat_top.sum() >= 8
    ids = np.unique(sm.face_ids[flat_top])
    assert len(ids) == 1 and sm.face_types[ids[0]] == "plane"
    assert (sm.face_ids == ids[0]).sum() == flat_top.sum()
    # the top face area: 100 x 80 minus the pocket, minus the fillet strips
    area = m.area_faces[flat_top].sum()
    assert area == pytest.approx((92 * 72 - 20 * 20) * 1e-6, rel=1e-6)


def test_units_output_and_file_unit(step_file, tmp_path):
    mm = read_step(step_file, units="mm", use_cache=False).mesh
    assert np.allclose(mm.extents, SIZE_MM, atol=1e-3)
    # a file that declares metres instead of millimetres (same numbers) is 1000x bigger
    text = step_file.read_text()
    assert "SI_UNIT(.MILLI.,.METRE.)" in text
    big = tmp_path / "metres.step"
    big.write_text(text.replace("SI_UNIT(.MILLI.,.METRE.)", "SI_UNIT($,.METRE.)"))
    m = read_step(big, units="m", use_cache=False).mesh
    assert np.allclose(m.extents, np.array(SIZE_MM), atol=1e-3)  # 100 m, read in metres
    with pytest.raises(StepImportError, match="units"):
        read_step(step_file, units="cm")


def test_cache_hit_is_fast_and_equal(step_file):
    t0 = time.perf_counter()
    a = read_step(step_file)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    b = read_step(step_file)
    warm = time.perf_counter() - t0
    print(f"cold {cold:.3f} s, cached {warm:.4f} s")
    assert not a.from_cache and b.from_cache
    assert np.array_equal(a.mesh.vertices, b.mesh.vertices) and np.array_equal(a.mesh.faces, b.mesh.faces)
    assert np.array_equal(a.face_ids, b.face_ids) and a.face_types == b.face_types
    assert warm < cold
    assert read_step(step_file, linear_deflection=0.05).from_cache is False  # other tolerance, other key


def test_errors_are_one_line(tmp_path):
    bad = tmp_path / "bad.step"
    bad.write_bytes(b"this is not a step file")
    with pytest.raises(StepImportError) as e:
        read_step(bad)
    assert "\n" not in str(e.value) and "bad.step" in str(e.value)
    with pytest.raises(StepImportError, match="not found"):
        read_step(tmp_path / "missing.step")
    empty = tmp_path / "empty.step"
    empty.write_bytes(b"")
    with pytest.raises(StepImportError):
        read_step(empty)


def test_assembly_merges_solids_and_reports_count(tmp_path):
    a = cq.Workplane("XY").box(10, 10, 10)
    b = cq.Workplane("XY").box(10, 10, 10).translate((30, 0, 0))
    p = tmp_path / "two.step"
    cq.exporters.export(cq.Workplane("XY").newObject([cq.Compound.makeCompound([a.val(), b.val()])]), str(p))
    sm = read_step(p)
    assert sm.n_solids == 2 and sm.n_cad_faces == 12
    assert sm.mesh.is_watertight and len(sm.mesh.split(only_watertight=False)) == 2


def test_load_mesh_keeps_face_ids_through_copy(step_file):
    from ui.session import load_cad_face_ids, load_mesh

    m = load_mesh(step_file, "mm")  # part units are ignored for STEP
    assert np.allclose(m.extents, np.array(SIZE_MM) / 1000.0, atol=1e-6)
    c = m.copy()
    assert np.array_equal(c.metadata["cad_face_ids"], m.metadata["cad_face_ids"])
    assert len(c.metadata["cad_face_ids"]) == len(c.faces)
    assert np.array_equal(load_cad_face_ids(step_file, "in"), m.metadata["cad_face_ids"])
    assert load_cad_face_ids(step_file.with_suffix(".stl")) is None


def test_load_sample_two_step_files(tmp_path):
    from core.tool_surface_eval import load_sample

    folder = tmp_path / "sample"
    folder.mkdir()
    block = _solid()
    cq.exporters.export(block, str(folder / "mandrel.step"))
    top = cq.Workplane("XY").newObject([block.faces(">Z").val()])
    cq.exporters.export(top, str(folder / "tool_surface.step"))
    mesh, truth, info = load_sample(folder)
    assert np.allclose(mesh.extents, np.array(SIZE_MM) / 1000.0, atol=1e-6)
    assert truth.any() and not truth.all()
    assert info["truth_faces"] == int(truth.sum())
    flat_top = np.all(np.abs(mesh.vertices[mesh.faces][:, :, 2] - 0.040) < 1e-7, axis=1)
    assert truth[flat_top].all()  # the whole flat top face is labelled
    assert not truth[mesh.face_normals[:, 2] < 0.1].any()  # sides and bottom are not


def _stl_bytes():
    import trimesh

    return trimesh.creation.box((0.2, 0.2, 0.2)).export(file_type="stl")


@pytest.fixture
def gui(tmp_path):
    import viser

    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8797, verbose=False)
    try:
        state = build_app(server)
        state["session"].project_dir = tmp_path / "proj"
        yield state
    finally:
        server.stop()


def test_upload_step_part_and_corrupt_keeps_old(gui, step_file):
    cell, s = gui["cell"], gui["session"]
    cell.upload_part("block.step", step_file.read_bytes(), None)
    path = s.project_dir / "assets" / "parts" / "block.step"
    assert path.is_file() and s.part is not None and s.part.mesh_path.endswith("block.step")
    m = s.part_mesh()
    assert "cad_face_ids" in m.metadata and len(m.metadata["cad_face_ids"]) == len(m.faces)
    assert np.allclose(m.extents, np.array(SIZE_MM) / 1000.0, atol=1e-6)
    good = path.read_bytes()
    cell.upload_part("block.step", b"garbage, not STEP", None)
    assert path.read_bytes() == good
    assert [x.name for x in path.parent.iterdir()] == ["block.step"]
    cell.upload_part("a.dwg", b"x", None)  # still refused
    assert not (path.parent / "a.dwg").exists()


def test_upload_step_size_limit(gui, monkeypatch):
    import ui.cell as cell_mod

    monkeypatch.setattr(cell_mod, "MAX_UPLOAD", 10)
    gui["cell"].upload_part("big.step", b"x" * 100, None)
    assert not (gui["session"].project_dir / "assets" / "parts" / "big.step").exists()
