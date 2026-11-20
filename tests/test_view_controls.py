"""R2: lighting presets, shadows, camera fit and view buttons (ui/view_controls.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import viser

from core.transforms import make_T
from ui import settings as ui_settings
from ui import view_controls as vc
from ui.session import PartState, Session


def _session() -> Session:
    s = Session()
    s.part = PartState("assets/parts/flat_plate.stl", "m", make_T(p=(1.0, 0.0, 0.5)))
    s.load_robot_from_registry("ur20")
    return s


@pytest.fixture(scope="module")
def ctl():
    server = viser.ViserServer(host="127.0.0.1", port=8771, verbose=False)
    s = _session()
    c = vc.ViewControls(server, s)
    yield server, s, c
    server.stop()


def test_bounds_cover_part_and_robot(ctl):
    _, s, _ = ctl
    box = vc.cell_bounds(s)
    # flat plate 1 x 0.6 x 0.02 m centred at (1, 0, 0.5)
    assert box[0][0] <= 0.5 + 1e-6 and box[1][0] >= 1.5 - 1e-6
    assert box[0][1] <= -0.3 + 1e-6 and box[1][1] >= 0.3 - 1e-6
    assert box[0][2] <= 0.49 + 1e-6 and box[1][2] >= 0.51 - 1e-6
    # the robot base is at the origin
    assert np.all(box[0] <= 0.0 + vc.ROBOT_PAD + 1e-9) and box[0][0] <= -0.0
    pure_part = vc.cell_points(Session(part=s.part))[0]
    assert pure_part.min(axis=0)[0] == pytest.approx(0.5, abs=1e-6)
    assert pure_part.max(axis=0)[0] == pytest.approx(1.5, abs=1e-6)


def test_fit_looks_at_centre_from_fit_distance(ctl):
    server, s, c = ctl
    assert c.fit()
    box = vc.cell_bounds(s)
    centre, dist = c.frame
    assert np.allclose(centre, (box[0] + box[1]) / 2)
    assert dist == pytest.approx(vc.fit_distance(box))
    init = server.initial_camera
    pos, look, _ = vc.view_pose(centre, dist, "Iso")
    assert np.allclose(init.look_at, look)
    assert np.linalg.norm(look - centre) == pytest.approx(vc.PANEL_SHIFT * dist)  # shifted, not far
    assert np.linalg.norm(np.asarray(init.position) - look) == pytest.approx(dist)
    # the bounding sphere fits the view
    radius = 0.5 * np.linalg.norm(box[1] - box[0])
    assert dist * np.sin(np.radians(vc.FOV_DEG) / 2) > radius


@pytest.mark.parametrize("name", list(vc.VIEWS))
def test_view_buttons_set_direction_and_keep_distance(ctl, name):
    server, _, c = ctl
    c.fit()
    centre, dist = c.frame
    c.set_view(name)
    d, up = vc.VIEWS[name]
    d = np.asarray(d) / np.linalg.norm(d)
    init = server.initial_camera
    off = np.asarray(init.position) - np.asarray(init.look_at)
    assert np.linalg.norm(off) == pytest.approx(dist)
    assert np.allclose(off / dist, d)
    assert np.linalg.norm(np.asarray(init.look_at) - centre) == pytest.approx(vc.PANEL_SHIFT * dist)
    assert tuple(init.up_direction) == up


def test_view_pose_directions():
    for view, expect in {"Top": (0, 0, 1), "Front": (0, -1, 0), "Side": (1, 0, 0)}.items():
        pos, look, _ = vc.view_pose((1, 2, 3), 5.0, view)
        assert np.allclose(pos - look, 5.0 * np.asarray(expect))


def test_empty_study_does_not_fit():
    assert vc.cell_bounds(Session(robot_name=None)) is None


def test_lighting_calls_bundled_preset(ctl, monkeypatch):
    server, _, c = ctl
    calls = []
    monkeypatch.setattr(
        server.scene, "configure_environment_map", lambda *a, **k: calls.append((a, k)), raising=False
    )
    c.lighting.value = "Off"
    c.lighting.value = "Studio"
    args, kw = calls[-1]
    assert args[0] == vc.HDRI
    assert (Path(viser.__file__).parent / "_assets" / "hdri").is_dir()
    from viser._scene_api import _HDRI_FILENAMES

    assert (Path(viser.__file__).parent / "_assets" / "hdri" / _HDRI_FILENAMES[vc.HDRI]).is_file()
    assert kw["background"] is False  # the theme background stays
    assert kw["environment_intensity"] <= 0.6
    assert c.key.visible and not c.ambient.visible
    c.lighting.value = "Flat"
    assert c.ambient.visible and not c.key.visible
    c.lighting.value = "Studio"


def test_shadows_toggle_flips_cast_shadow(ctl):
    _, _, c = ctl
    c.lighting.value = "Studio"
    assert c.shadows.value is False and c.key.cast_shadow is False
    c.shadows.value = True
    assert c.key.cast_shadow is True
    c.shadows.value = False
    assert c.key.cast_shadow is False


def test_settings_saved_and_restored(ctl):
    _, _, c = ctl
    state = {"view": c}
    c.lighting.value, c.shadows.value = "Flat", True
    saved = ui_settings.collect(state)
    assert saved["view.lighting"] == "Flat" and saved["view.shadows"] is True
    c.lighting.value, c.shadows.value = "Studio", False
    done = ui_settings.apply(state, saved)
    assert {"view.lighting", "view.shadows"} <= set(done)
    assert c.lighting.value == "Flat" and c.shadows.value is True
    assert c.key.cast_shadow is True
    c.lighting.value, c.shadows.value = "Studio", False
    ui_settings.apply(state, {"view.lighting": "Nonsense"})  # unknown option is skipped
    assert c.lighting.value == "Studio"


def test_refit_only_for_a_different_part(ctl):
    server, s, c = ctl
    c.fit()
    init = server.initial_camera

    def cam():
        return np.asarray(init.position).copy(), np.asarray(init.look_at).copy()

    p0, l0 = cam()
    s.part.T_world_part = make_T(p=(2.0, 1.0, 0.5))  # pose edit
    c.on_part_changed()
    s.part.mesh_path = "assets/parts/flat_plate_refined_5mm.stl"  # tessellate: same part
    c.on_part_changed()
    assert np.allclose(cam()[0], p0) and np.allclose(cam()[1], l0)
    s.part.mesh_path = "assets/parts/curved_panel.stl"  # a different part
    s.part.T_world_part = make_T(p=(5.0, 5.0, 0.5))
    c.on_part_changed()
    assert not np.allclose(cam()[1], l0)
    assert c.fit_part == "assets/parts/curved_panel.stl"
