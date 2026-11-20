"""ui.handover: every export of a plan in one zip with a README."""

import io
import json
import zipfile

import pytest

from ui import handover as ho


def _writer(tmp_path, name, data=b"x" * 10):
    def fn():
        p = tmp_path / name
        p.write_bytes(data)
        return p, data

    return fn


def _fail():
    raise OSError("disk full\nmore detail")


def test_zip_holds_the_files_and_a_readme(tmp_path):
    steps = [
        ("TCP poses (CSV)", _writer(tmp_path, "tcp_poses.csv", b"a,b\n1,2\n")),
        ("HTML report", _fail),
        ("Tool URDF", _writer(tmp_path, "sander_optimized.zip")),
    ]
    path, data, failed = ho.build(tmp_path, "study 1/x", steps, ["Plan: 2 stations"])
    assert path == tmp_path / "study_1_x_handover.zip" and path.read_bytes() == data
    assert failed == [("HTML report", "disk full")]
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.namelist() == ["README.txt", "tcp_poses.csv", "sander_optimized.zip"]
        assert z.read("tcp_poses.csv") == b"a,b\n1,2\n"
        text = z.read("README.txt").decode()
    assert "Plan: 2 stations" in text
    assert "tcp_poses.csv: one row per point" in text
    assert "sander_optimized.zip: the tool as a URDF" in text
    assert "Not included" in text and "HTML report: disk full" in text and "more detail" not in text


def test_nothing_exported_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="nothing could be exported: A \\(disk full\\)"):
        ho.build(tmp_path, "s", [("A", _fail)])
    assert not list(tmp_path.glob("*.zip"))


def test_readme_without_failures_has_no_missing_section(tmp_path):
    _p, data, failed = ho.build(tmp_path, "", [("x", _writer(tmp_path, "unknown.bin"))])
    assert failed == []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        text = z.read("README.txt").decode()
    assert "Not included" not in text and "\nunknown.bin\n" in text
    assert (tmp_path / "plan_handover.zip").is_file()


def test_plan_lines_and_project_copy(tmp_path):
    import numpy as np

    from core.schema import Solution, Station, TargetGroup, Targets
    from core.targets import append_points
    from core.transforms import make_T
    from ui.session import PartState, Session

    s = Session(project_dir=tmp_path / "proj")
    s.part = PartState("assets/parts/flat_plate.stl", "m", make_T(p=(1.0, 0.0, 0.5)))
    s.load_robot_from_registry("ur20")
    s.load_tool_from_library("orbital_sander")
    t = Targets(groups=[TargetGroup(name="g")])
    s.targets = append_points(t, np.zeros((3, 3)), np.tile([0, 0, 1.0], (3, 1)), "g")
    sol = Solution(label="1 station", stations=[Station(xyz=(0, 0, 0))], unreached=[s.targets.points[0].id])
    lines = ho.plan_lines(s, sol, stale=["part"])
    assert lines[0].startswith("OUT OF DATE") and "part" in lines[0]
    assert "Plan: 1 station" in lines and "points reached: 2 of 3" in "\n".join(lines)
    assert any("Not every point" in x for x in lines) and any("ur20" in x for x in lines)
    assert not any("OUT OF DATE" in x for x in ho.plan_lines(s, sol))
    path, data = ho.project_step(s, tmp_path / "export")
    assert path == tmp_path / "export" / "project.json"
    assert json.loads(data)["robot"]["name"] == "ur20"
    assert not (tmp_path / "proj" / "project.json").exists()  # the project's own file is not written
