"""Review WP C: uploads validate before they replace anything; part fingerprint sees file content."""

import io
import zipfile

import pytest
import trimesh

from core.safe_io import replace_dir, replace_file
from core.urdf_import import URDFImportError, import_urdf_zip

URDF = """<robot name='arm'><link name='a'/><link name='b'/>
<joint name='j' type='revolute'><parent link='a'/><child link='b'/>
<limit lower='-1' upper='1' effort='1' velocity='1'/><axis xyz='0 0 1'/></joint></robot>"""


def _stl(extents=(1, 1, 1)) -> bytes:
    return trimesh.exchange.stl.export_stl(trimesh.creation.box(extents=extents))


def _zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, v in files.items():
            z.writestr(k, v)
    return buf.getvalue()


def test_broken_zip_keeps_existing_robot_folder(tmp_path):
    d = tmp_path / "assets" / "robots" / "myarm"
    d.mkdir(parents=True)
    (d / "myarm.urdf").write_text("<robot name='x'/>")
    with pytest.raises(URDFImportError):
        import_urdf_zip(b"this is not a zip", d)
    assert (d / "myarm.urdf").read_text() == "<robot name='x'/>"
    assert [p.name for p in d.parent.iterdir()] == ["myarm"]  # no temp folders left


def test_good_zip_replaces_and_returns_final_paths(tmp_path):
    d = tmp_path / "robots" / "myarm"
    d.mkdir(parents=True)
    (d / "old.txt").write_text("old")
    info = import_urdf_zip(_zip({"sub/arm.urdf": URDF}), d)
    assert info.path == d.resolve() / "sub" / "arm.urdf" and info.path.is_file()
    assert not (d / "old.txt").exists()
    assert [p.name for p in d.parent.iterdir()] == ["myarm"]


def test_replace_file_rejects_and_keeps_old(tmp_path):
    p = tmp_path / "part.stl"
    p.write_bytes(_stl())
    good = p.read_bytes()

    def check(t):
        trimesh.load(t, force="mesh").faces[0]

    with pytest.raises(Exception):  # noqa: B017 - trimesh raises many types on bad files
        replace_file(p, b"garbage", check)
    assert p.read_bytes() == good and [x.name for x in tmp_path.iterdir()] == ["part.stl"]
    assert replace_file(p, _stl((2, 2, 2)), check) is True
    assert replace_file(tmp_path / "new.stl", _stl(), check) is False


def test_replace_dir_restores_on_failure(tmp_path):
    d = tmp_path / "pkg"
    d.mkdir()
    (d / "a").write_text("1")

    def bad(t):
        (t / "b").write_text("2")
        raise ValueError("nope")

    with pytest.raises(ValueError):
        replace_dir(d, bad)
    assert (d / "a").read_text() == "1" and [x.name for x in tmp_path.iterdir()] == ["pkg"]


@pytest.fixture
def gui(tmp_path):
    import viser

    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8795, verbose=False)
    try:
        state = build_app(server)
        state["session"].project_dir = tmp_path
        yield state
    finally:
        server.stop()


def test_part_upload_garbage_keeps_old_file(gui, tmp_path):
    cell = gui["cell"]
    cell.upload_part("part.stl", _stl(), None)
    path = tmp_path / "assets" / "parts" / "part.stl"
    good = path.read_bytes()
    cell.upload_part("part.stl", b"garbage", None)
    assert path.read_bytes() == good
    assert [x.name for x in path.parent.iterdir()] == ["part.stl"]


def test_stayout_upload_garbage_keeps_old_file(gui, tmp_path):
    panel = gui["stayouts"]
    assert panel.add_mesh_bytes("fix.stl", _stl())
    path = tmp_path / "assets" / "stayouts" / "fix.stl"
    good = path.read_bytes()
    n = len(gui["session"].stayouts)
    assert not panel.add_mesh_bytes("fix.stl", b"garbage")
    assert path.read_bytes() == good and len(gui["session"].stayouts) == n


def test_carrier_upload_garbage_keeps_old_file(gui, tmp_path):
    from ui.carrier_ui import DEFAULTS

    panel = gui["carrier"]
    panel.load(DEFAULTS["rail"])
    assert panel.commit()
    ed = panel.editor
    assert ed.add_mesh_bytes("cart.stl", _stl())
    path = tmp_path / "assets" / "carriers" / "cart.stl"
    good = path.read_bytes()
    assert not ed.add_mesh_bytes("cart.stl", b"garbage")
    assert path.read_bytes() == good


def test_export_failure_keeps_previous_export(gui, tmp_path):
    from ui import package_actions as pa

    class Fake:
        s = gui["session"]
        work = tmp_path / "work"

        pkg = type("Pkg", (), {"name": "pk"})

    dest = tmp_path / "export"
    dest.mkdir()
    (dest / "old.txt").write_text("keep")
    import core.urdf_package as up

    orig = pa.write_package
    pa.write_package = lambda *a, **k: (_ for _ in ()).throw(ValueError("not ready"))
    try:
        with pytest.raises(ValueError):
            pa.write(Fake, dest)
    finally:
        pa.write_package = orig
    assert (dest / "old.txt").read_text() == "keep" and up


def test_part_fingerprint_sees_replaced_file(gui, tmp_path):
    from ui.session import PartState
    from ui.staleness import components

    s = gui["session"]
    path = tmp_path / "assets" / "parts" / "p.stl"
    path.parent.mkdir(parents=True)
    path.write_bytes(_stl())
    s.part = PartState("assets/parts/p.stl", "m", s.T_world_base.copy())
    before = components(s)["part"]
    assert components(s)["part"] == before  # stable
    path.write_bytes(_stl((3, 3, 3)))  # same name, new content
    assert components(s)["part"] != before
    # an old-format stamp just differs once (plans show stale); it must not crash
    from ui.staleness import changed

    assert changed({"part": "oldformat0000"}, components(s)) == ["part"]
