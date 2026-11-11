"""Package builder: URDF packages for tools, rails, gantries and mobile bases (core.urdf_package,
ui.package_builder) - write, read back (also CAD-style URDFs), use in the project."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
import viser
from viser._gui_handles import UploadedFile

from core.carrier import Carrier
from core.schema import Base, Pose
from core.transforms import pose_to_matrix
from core.urdf_package import (
    Frame,
    Package,
    Part,
    check,
    open_package,
    read_urdf,
    to_base,
    write_package,
    zip_folder,
)

E = SimpleNamespace(client=None)


def _same(a: Pose, b: Pose) -> bool:
    return np.allclose(pose_to_matrix(a), pose_to_matrix(b), atol=1e-7)


def test_gantry_round_trip(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    trimesh.creation.box((200, 100, 50)).export(src / "block.stl")  # mm
    p = Package(
        kind="gantry",
        name="my gantry",
        parts=[
            Part(
                name="rail_a", stage="fixed", shape="box", size=(3, 0.15, 0.15), pose=Pose(xyz=(0, -1.3, 0))
            ),
            Part(name="bridge", stage="bridge", shape="box", size=(0.25, 2.5, 0.15)),
            Part(name="carriage", stage="carriage", shape="cylinder", radius=0.2, height=0.15),
            Part(
                name="z",
                stage="lift",
                shape="mesh",
                mesh="block.stl",
                units="mm",
                pose=Pose(xyz=(0, 0, -0.5), rpy=(0.1, 0, 0.2)),
            ),
        ],
        mount=Pose(xyz=(0, 0, -0.6), rpy=(np.pi, 0, 0)),
        x=(-1.4, 1.4),
        y=(-1.0, 1.0),
        lift=(-0.5, 0.0),
    )
    urdf = write_package(p, src, tmp_path / "pkg" / "g")
    assert "axis_y" in urdf.read_text() and (urdf.parent / "meshes" / "block.stl").is_file()
    q, _ = open_package(zip_folder(urdf.parent), tmp_path / "open")
    assert (q.kind, q.name, q.x, q.y, q.lift) == ("gantry", "my gantry", p.x, p.y, p.lift)
    assert [(x.name, x.stage, x.shape) for x in q.parts] == [(x.name, x.stage, x.shape) for x in p.parts]
    assert all(_same(a.pose, b.pose) for a, b in zip(p.parts, q.parts, strict=True)) and _same(
        p.mount, q.mount
    )
    b = to_base(q, (tmp_path / "open").as_posix() + "/g/meshes", Base())
    assert [x.attach for x in b.bodies] == ["fixed", "x_axis", "carriage", "lift"]
    c = Carrier(b)
    v = {"x": 0.3, "y": -0.2, "lift": -0.1}
    T = c.T_world_base(v)  # the robot base follows the URDF chain at the same travel
    assert np.allclose(T[:3, 3], (0.3, -0.2, -0.7)) and T[2, 2] < -0.99
    assert len(c.moving_bodies(v)) == 3


def test_cad_style_urdf_is_read_by_its_kinematics(tmp_path):
    """Other link names, joint origins and a mesh in mm on a rail: roles from the joint axes."""
    (tmp_path / "m").mkdir()
    trimesh.creation.box((500, 400, 100)).export(tmp_path / "m" / "Carriage.STL")
    (tmp_path / "rail.urdf").write_text(
        "\n".join(
            [
                '<robot name="MyRail">',
                '<link name="Track"><collision><origin xyz="0 0 0.05"/>',
                '  <geometry><box size="4 0.3 0.1"/></geometry></collision></link>',
                '<link name="Carriage_Link"><visual><origin xyz="0 0 0.05"/><geometry>',
                '  <mesh filename="m/Carriage.STL" scale="0.001 0.001 0.001"/></geometry></visual></link>',
                '<joint name="slide" type="prismatic"><parent link="Track"/><child link="Carriage_Link"/>',
                '  <origin xyz="0 0 0.1"/><axis xyz="1 0 0"/>',
                '  <limit lower="-1.8" upper="1.8" effort="1" velocity="1"/></joint>',
                '<link name="robot_mount"/>',
                '<joint name="m" type="fixed"><parent link="Carriage_Link"/><child link="robot_mount"/>',
                '  <origin xyz="0 0 0.1"/></joint>',
                "</robot>",
            ]
        ),
        encoding="utf-8",
    )
    p = read_urdf(tmp_path / "rail.urdf")
    assert p.kind == "rail" and p.x == (-1.8, 1.8) and not check(p)
    track, car = p.parts
    assert track.stage == "fixed" and np.allclose(track.pose.xyz, (0, 0, 0))  # bottom on the floor
    assert car.stage == "carriage" and car.units == "mm" and np.allclose(car.pose.xyz, (0, 0, 0.15))
    assert np.allclose(p.mount.xyz, (0, 0, 0.2))
    (tmp_path / "bad.urdf").write_text(
        '<robot name="b"><link name="a"/><link name="c"/><joint name="j" type="revolute"><parent link="a"/>'
        '<child link="c"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>'
        "</joint></robot>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rotary"):
        read_urdf(tmp_path / "bad.urdf")


def test_check_lists_problems():
    p = Package(kind="rail", name="r")
    msgs = check(p)
    assert any("part" in m for m in msgs) and any("X travel" in m for m in msgs)
    t = Package(kind="tool", name="t", parts=[Part(name="a", stage="carriage", shape="box", size=(1, 1, 1))])
    assert any("no stage" in m for m in check(t)) and any("TCP" in m for m in check(t))


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8826, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


def test_builder_tool_and_rail_into_the_project(app):
    b, s = app["builder"], app["session"]
    b._reset()
    b._set_kind("tool")
    b._add(Part(name="body", stage="tool", shape="cylinder", radius=0.05, height=0.15))
    b._set_tcp(Frame(name="pad", pose=Pose(xyz=(0, 0, 0.15))))
    b._use(E)
    assert s.tool is not None and [t.name for t in s.tool.tcps] == ["pad"]
    assert np.allclose(s.tool_model.T_flange_tcp("pad")[:3, 3], (0, 0, 0.15))
    b._set_kind("rail")
    assert b.pkg.parts[0].stage in ("fixed", "carriage") and b.tx.visible and not b.ty.visible
    b._add(Part(name="track", stage="fixed", shape="box", size=(4, 0.3, 0.1)))
    b._mount_moved(pose_to_matrix(Pose(xyz=(0, 0, 0.4))))
    b._export(E)
    zp = s.project_dir / "export" / "packages" / f"{b.pkg.name}.zip"
    assert zp.is_file()
    b._use(E)
    assert s.base.mode == "movable" and s.base.carrier == "rail" and len(s.base.bodies) == 2
    assert np.allclose(s.base.T_mount.xyz, (0, 0, 0.4)) and s.base.joints.x == (-1.5, 1.5)
    b._reset()
    b.open._impl.value = UploadedFile(zp.name, zp.read_bytes())  # as a browser upload sets it
    b._open(E)
    assert b.pkg.kind == "rail" and {x.name for x in b.pkg.parts} == {"body", "track"}
    assert Path(b.work / "meshes").is_dir()


def test_tool_tab_upload_reads_tcp_frames(app, tmp_path):
    p = Package(
        kind="tool",
        name="probe",
        parts=[Part(name="body", stage="tool", shape="cylinder", radius=0.02, height=0.2)],
        tcps=[
            Frame(name="tip", pose=Pose(xyz=(0, 0, 0.2))),
            Frame(name="side", pose=Pose(xyz=(0.02, 0, 0.1))),
        ],
    )
    urdf = write_package(p, None, tmp_path / "probe")
    tool = app["tool"]
    tool.upload._impl.value = UploadedFile("probe.zip", zip_folder(urdf.parent))
    tool._on_upload(E)
    s = app["session"]
    assert [t.name for t in s.tool.tcps] == ["tip", "side"]
    assert np.allclose(s.tool.tcps[1].T_head_tcp.xyz, (0.02, 0, 0.1))  # relative to the head (mount)
