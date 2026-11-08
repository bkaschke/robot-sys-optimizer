"""v4 plan item 7: userframe export (robot base -> part frame per stop, robot-dependent CSV)."""

import csv
import io

import numpy as np
import pytest
import viser
from scipy.spatial.transform import Rotation

from core.transforms import make_T, rpy_to_matrix
from core.userframe import MATRIX_COLUMNS, base_to_part, orientation, userframe_csv


def test_conventions_rebuild_the_rotation():
    R = rpy_to_matrix(np.radians([20.0, -35.0, 110.0]))
    rv = orientation(R, "ur")
    assert np.allclose(Rotation.from_rotvec(rv).as_matrix(), R, atol=1e-12)
    w, p, r = orientation(R, "fanuc")
    Rf = (
        rpy_to_matrix(np.radians([0, 0, r]))
        @ rpy_to_matrix(np.radians([0, p, 0]))
        @ rpy_to_matrix(np.radians([w, 0, 0]))
    )
    assert np.allclose(Rf, R, atol=1e-12)  # FANUC: Rz(R) Ry(P) Rx(W)
    assert orientation(R, "matrix") == ()


def test_base_to_part():
    T_wb = make_T(rpy_to_matrix((0, 0, np.pi / 2)), (1.0, 2.0, 0.0))
    T_wp = make_T(np.eye(3), (1.0, 3.0, 0.5))
    T = base_to_part(T_wb, T_wp)
    assert np.allclose(T[:3, 3], (1.0, 0.0, 0.5)) and np.allclose(T_wb @ T, T_wp)


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8802, verbose=False)
    st = build_app(server)
    yield st
    server.stop()


def test_userframes_put_the_targets_where_the_robot_reaches(app):
    s = app["session"]
    assert app["project"].load_example("rail_long_part")
    app["stations"].start(block=True)
    sol = app["results"].selected()
    ex = app["results"].export
    path, data = ex.export_userframes(sol)
    rows = list(csv.DictReader(io.StringIO(data.decode())))
    assert path.name == "userframes.csv" and len(rows) == len(sol.stations)
    assert {"rx_rad", "ry_rad", "rz_rad", *MATRIX_COLUMNS} <= set(rows[0])  # UR20: UR pose columns
    res = s.result_points[sol.label]
    pts = {p.id: p for p in s.targets.points}
    tm = s.tool_model
    for pid, k in sol.assignment.items():
        r = rows[k]
        R = Rotation.from_rotvec([float(r["rx_rad"]), float(r["ry_rad"]), float(r["rz_rad"])]).as_matrix()
        M = np.array([float(r[c]) for c in MATRIX_COLUMNS]).reshape(3, 3)
        assert np.allclose(R, M, atol=1e-6)
        t = np.array([float(r["x_mm"]), float(r["y_mm"]), float(r["z_mm"])]) / 1000
        p, n = np.array(pts[pid].p), np.array(pts[pid].n)
        target_in_base = R @ (p + res[pid].standoff * n) + t  # the userframe applied to the point
        tcp = (s.robot.fk(res[pid].q) @ tm.T_flange_tcp(res[pid].tcp))[:3, 3]
        assert np.linalg.norm(target_in_base - tcp) < 1e-5  # where the robot's own FK puts the TCP
    text = userframe_csv([], "fanuc").splitlines()[0]
    assert "w_deg" in text and "rx_rad" not in text
