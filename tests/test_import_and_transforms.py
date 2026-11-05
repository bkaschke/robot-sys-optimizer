import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from core.robot import ROBOTS_DIR, Robot
from core.transforms import (
    matrix_to_pose,
    matrix_to_rpy,
    matrix_to_wxyz,
    pose_to_matrix,
    rpy_to_matrix,
    wxyz_to_matrix,
)
from core.urdf_import import URDFImportError, import_urdf_zip


def test_rpy_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(500):
        rpy = rng.uniform([-np.pi, -np.pi / 2 + 1e-3, -np.pi], [np.pi, np.pi / 2 - 1e-3, np.pi])
        R = rpy_to_matrix(rpy)
        assert np.allclose(R @ R.T, np.eye(3))
        assert np.allclose(rpy_to_matrix(matrix_to_rpy(R)), R)
        assert np.allclose(wxyz_to_matrix(matrix_to_wxyz(R)), R)


def test_rpy_gimbal_lock():
    R = rpy_to_matrix((0.3, np.pi / 2, 0.2))
    assert np.allclose(rpy_to_matrix(matrix_to_rpy(R)), R)


def test_rpy_matches_urdf_convention():
    # URDF: fixed-axis roll about x, then pitch about y, then yaw about z
    R = rpy_to_matrix((np.pi / 2, 0, np.pi / 2))
    assert np.allclose(R @ [0, 0, 1], [1, 0, 0])


def test_pose_round_trip():
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix((0.1, -0.4, 2.0))
    T[:3, 3] = (1.0, 2.0, 3.0)
    assert np.allclose(pose_to_matrix(matrix_to_pose(T)), T)


def _ur20_zip(package_uris: bool = True) -> bytes:
    """The registry UR20 re-packed like a ROS package: package:// mesh URIs, nested folders."""
    src = ROBOTS_DIR / "ur20"
    urdf = (src / "ur20.urdf").read_text(encoding="utf-8")
    if package_uris:
        urdf = urdf.replace('filename="meshes/', 'filename="package://ur_description/meshes/ur20/')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("ur_description/urdf/ur20.urdf", urdf)
        for f in (src / "meshes").rglob("*"):
            if f.is_file():
                rel = f.relative_to(src / "meshes").as_posix()
                z.write(f, f"ur_description/meshes/ur20/{rel}")
    return buf.getvalue()


def test_import_zip_resolves_package_uris(tmp_path):
    info = import_urdf_zip(_ur20_zip(), tmp_path / "up")
    text = info.path.read_text(encoding="utf-8")
    assert "package://" not in text
    assert info.root_link == "world" and info.leaf_link == "tool0"
    assert len(info.actuated_joints) == 6
    up = Robot.from_urdf(info.path, "base_link", "tool0")
    ref = Robot.from_registry("ur20")
    q = np.array([0.3, -1.0, 1.2, -0.5, 0.8, 0.1])
    assert np.allclose(up.fk(q), ref.fk(q))
    # meshes resolved to real files
    for rel in __import__("re").findall(r'filename="([^"]+)"', text):
        assert (info.path.parent / rel).is_file()


@pytest.mark.parametrize("bad", ["../evil.txt", "/abs.txt", "C:/win.txt", "a/../../b.txt"])
def test_zip_slip_rejected(tmp_path, bad):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(bad, "x")
    with pytest.raises(URDFImportError):
        import_urdf_zip(buf.getvalue(), tmp_path / "up")
    assert not (tmp_path / "evil.txt").exists()


def test_zip_without_urdf(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("robot.urdf.xacro", "<robot/>")
    with pytest.raises(URDFImportError, match="xacro"):
        import_urdf_zip(buf.getvalue(), tmp_path / "up")


def test_not_a_zip(tmp_path):
    with pytest.raises(URDFImportError):
        import_urdf_zip(b"hello", Path(tmp_path) / "up")
