"""T2: tool-surface scoring, pick-and-grow, confidence, secondary candidates, sanding checks."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from core.mandrel_cases import SYNTHETIC_CASES, iter_cases
from core.mandrels_hard import steep_pocket
from core.tool_surface_eval import score
from core.tool_surface_pick import DEFAULT_PARAMS, detect, detect_tool_surface

FAST = (
    "plate_a",
    "male_flange",
    "female_steep70",
    "male_decoy_flange",
    "male_ambiguous_flange",
    "plate_joggle_step",
)


def _check_case(case, mesh, truth):
    g = detect_tool_surface(mesh)
    s = score(g.mask, truth, mesh)
    if case.expect.get("ambiguous"):
        assert g.confidence < 0.5, (case.name, g.confidence)
    else:
        assert s["iou"] >= case.expect["min_iou"], (case.name, s)
        assert g.confidence >= 0.5, (case.name, g.confidence, g.confidence_parts)
    return g, s


@pytest.mark.parametrize("name", FAST)
def test_fast_cases_meet_expectations(name):
    for case, mesh, truth, _ in iter_cases(names=(name,)):
        _check_case(case, mesh, truth)


def test_all_cases_meet_expectations_and_means():
    easy, hard = [], []
    for case, mesh, truth, _ in iter_cases():
        g, s = _check_case(case, mesh, truth)
        if not case.hard:
            easy.append(s["iou"])
        elif not case.expect.get("ambiguous"):
            hard.append(s["iou"])
    assert len(easy) + len(hard) + 1 == len(SYNTHETIC_CASES)
    assert np.mean(easy) >= 0.98
    assert np.mean(hard) >= 0.9


def test_deterministic():
    case, mesh, truth, _ = next(iter_cases(names=("female_steep85",)))
    a = detect_tool_surface(mesh)
    b = detect_tool_surface(mesh.copy())
    assert np.array_equal(a.faces, b.faces)
    assert a.confidence == b.confidence
    assert [r.reason for r in a.regions] == [r.reason for r in b.regions]


def test_male_flange_return_is_secondary():
    case, mesh, truth, _ = next(iter_cases(names=("male_flange",)))
    g = detect_tool_surface(mesh)
    assert g.secondary, "the flange return must be offered"
    c = g.secondary[0]
    assert c.kind == "steep_convex" and "convex break" in c.reason
    assert not g.mask[c.faces].any()  # not in the main selection
    assert c.area_m2 > 0.1 * g.area_m2


def test_reasons_summary_and_components():
    case, mesh, truth, _ = next(iter_cases(names=("plate_joggle_step",)))
    g = detect_tool_surface(mesh)
    assert g.regions and all(r.reason.strip() for r in g.regions)
    assert all(c.reason.strip() for c in g.secondary)
    assert {"area", "up", "vis"} <= set(g.regions[0].components)
    roles = {r.label: r.role for r in g.regions}
    assert sorted(lb for lb, r in roles.items() if r in ("seed", "selected")) == list(g.selected_labels)
    assert g.summary.startswith("Tool surface: 3 regions") and "confidence" in g.summary
    assert len(g.selected_labels) == 3


def test_detect_returns_mask():
    case, mesh, truth, _ = next(iter_cases(names=("plate_a",)))
    m = detect(mesh)
    assert m.dtype == bool and m.shape == (len(mesh.faces),)


def test_unknown_param_raises():
    case, mesh, truth, _ = next(iter_cases(names=("plate_a",)))
    with pytest.raises(TypeError, match="unknown"):
        detect_tool_surface(mesh, crease_max=40)
    assert "crease_max_deg" in DEFAULT_PARAMS


def _fillet_trough(radius=0.005, steps=9, length=0.3, ny=30):
    """Open sheet: floor, a concave fillet of ``radius`` (``steps`` facets), then a 45 deg slope."""
    xs = [-0.2, -0.1]
    zs = [0.0, 0.0]
    for k in range(steps + 1):  # arc from straight down (floor) to 45 deg
        a = np.radians(45.0) * k / steps
        xs.append(radius * np.sin(a))
        zs.append(radius * (1 - np.cos(a)))
    x1, z1 = xs[-1], zs[-1]
    xs += [x1 + 0.1, x1 + 0.2]
    zs += [z1 + 0.1, z1 + 0.2]
    px, pz = np.array(xs), np.array(zs)
    ys = np.linspace(0, length, ny + 1)
    V = np.array([(x, y, z) for y in ys for x, z in zip(px, pz, strict=True)])
    n = len(px)
    F = []
    for j in range(ny):
        for i in range(n - 1):
            a, b, c, d = j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i
            F += [(a, b, c), (a, c, d)]
    m = trimesh.Trimesh(V, np.array(F), process=False)
    if m.face_normals[:, 2].mean() < 0:
        m.invert()
    return m


def test_small_concave_fillet_warns_with_big_pad():
    mesh = _fillet_trough(radius=0.005)
    g = detect_tool_surface(mesh, pad_radius_m=0.025)
    assert g.mask[np.flatnonzero(mesh.triangles_center[:, 0] > -0.1)].mean() > 0.9  # the fillet is selected
    warn = [w for w in g.warnings if "smaller than the pad radius 25 mm" in w]
    assert warn, g.warnings
    rad_mm = float(warn[0].split("~")[1].split(" mm")[0])
    assert 3 <= rad_mm <= 7  # the estimate is close to the 5 mm fillet
    assert not [w for w in detect_tool_surface(mesh, pad_radius_m=0.002).warnings if "pad radius" in w]
    assert not [w for w in detect_tool_surface(mesh).warnings if "pad radius" in w]


def test_steep_pocket_warns_steep_and_sharp_corner():
    mesh, truth, _ = steep_pocket(wall_deg=70.0, seed=1)
    g = detect_tool_surface(mesh, pad_radius_m=0.02)
    assert any(w.startswith("steep:") for w in g.warnings), g.warnings
    assert any("cannot be sanded with this pad" in w for w in g.warnings), g.warnings
    flat = detect_tool_surface(next(iter_cases(names=("plate_a",)))[1], pad_radius_m=0.02)
    assert not flat.warnings, flat.warnings
