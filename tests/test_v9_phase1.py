"""V9 phase 1 (GitHub issues #2, #12, #13, #14): STEP import status in the package builder,
project files kept in the project folder, the mobile_amr example without its keep-out, live
standoff / keep-out overlays with the standoff range as a band."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
import viser
from shapely.geometry import Point
from viser._gui_handles import UploadedFile

from core.standoff import footprint

E = SimpleNamespace(client=None)
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8835, verbose=False)
    st = build_app(server)
    yield st
    st["builder_window"].stop()
    server.stop()


def test_mobile_example_has_no_keepout():  # #13
    d = json.loads((REPO / "examples" / "mobile_amr" / "project.json").read_text(encoding="utf-8"))
    assert d["base"]["keepouts"] == []


def _ring_distance(handle, fp) -> float:
    pts = np.asarray(handle.points)[:, 0, :2]
    return float(np.median([fp.exterior.distance(Point(p)) for p in pts]))


def test_standoff_lines_follow_the_fields(app):  # #14
    assert app["project"].load_example("rail_long_part")
    car, s = app["carrier"], app["session"]
    fp = footprint(s.part_mesh(), s.part.T_world_part)
    car.so.value = (250.0, 600.0)  # typed, not applied
    names = {h.name: h for h in car._overlay["Standoff"]}
    assert _ring_distance(names["/cell/carrier_overlay/standoff_min"], fp) == pytest.approx(0.25, abs=0.01)
    assert _ring_distance(names["/cell/carrier_overlay/standoff_pref"], fp) == pytest.approx(0.6, abs=0.01)
    assert s.base.standoff_m.carrier_part_min != 0.25  # nothing applied
    car.keepouts.value = "-500,2000,500,2600"
    assert len(car._overlay["Keep-outs"]) == 1
    objects = app["objects"]
    objects.rebuild_tree()
    assert "External axis/Keep-outs" in objects.show and "External axis/Standoff" in objects.show
    objects.show["External axis/Keep-outs"].value = False
    assert not car._overlay["Keep-outs"][0].visible
    objects.show["External axis/Keep-outs"].value = True
    car.keepouts.value = ""


def test_standoff_range_is_a_band(app):  # #14
    assert app["project"].load_example("mobile_amr")
    car, s = app["carrier"], app["session"]
    next(iter(car.side_boxes.values())).value = True  # a robot side faces the part: contour plan
    car.fixed_so.value, car.max_so.value = 400.0, 700.0
    band = next(h for h in car._overlay["Standoff"] if h.name.endswith("standoff_band"))
    fp = footprint(s.part_mesh(), s.part.T_world_part)
    d = [fp.exterior.distance(Point(v[:2])) for v in np.asarray(band.vertices)]
    assert min(d) == pytest.approx(0.4, abs=0.01) and max(d) == pytest.approx(0.7, abs=0.02)


def test_project_keeps_its_files(app, tmp_path):  # #12
    s, project = app["session"], app["project"]
    assert project.load_example("rail_long_part")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    box = trimesh.creation.box((0.4, 0.3, 0.2))
    box.apply_translation((0, 0, 0.1))
    box.export(outside / "my_part.stl")
    s.part.mesh_path = (outside / "my_part.stl").as_posix()  # loaded from another folder
    dest = tmp_path / "study"
    project.save_now(dest)
    assert Path(s.part.mesh_path).as_posix().startswith("assets/imported/parts/")
    assert (dest / s.part.mesh_path).is_file() and any("my_part.stl" in n for n in s.bundled)
    (outside / "my_part.stl").unlink()  # the original folder is gone: still opens with the part
    assert project._load(E, dest)
    assert s.part is not None and s.load_warnings == []
    # a missing file no longer stops the project from opening
    d = json.loads((dest / "project.json").read_text(encoding="utf-8"))
    d["part"]["mesh"] = "assets/parts/does_not_exist.stl"
    (dest / "project.json").write_text(json.dumps(d), encoding="utf-8")
    assert project._load(E, dest)
    assert s.part is None and any("does_not_exist" in w for w in s.load_warnings)


def test_step_import_runs_in_the_background_and_reports(tmp_path):  # #2
    from ui.package_builder import PackageBuilder

    class S:
        project_dir = tmp_path
        robot_name = "ur20"

    server = viser.ViserServer(host="127.0.0.1", port=8837, verbose=False)
    try:
        b = PackageBuilder(server, S())

        def upload(name: str, data: bytes) -> None:
            b.upload._impl.value = UploadedFile(name, data)
            b._on_mesh(E)
            t_end = time.time() + 60
            while b.import_bar.visible and time.time() < t_end:
                time.sleep(0.1)

        upload("broken.step", b"ISO-10303-21; this is not a STEP file")
        assert "broken.step" in b.import_errors and "not added" in b.import_info.content
        assert not b.pkg.parts
        b._check(E)
        assert "broken.step: import failed" in b.status.content
        mesh = trimesh.creation.box((100.0, 50.0, 20.0))
        upload("block.stl", trimesh.exchange.stl.export_stl(mesh))
        assert [p.name for p in b.pkg.parts] == ["block"] and "Added" in b.import_info.content
    finally:
        server.stop()
