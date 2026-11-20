"""Synthetic mandrel generator, metrics and the two-file sample format (T0)."""

import numpy as np
import pytest
import trimesh

from core.mandrel_cases import SYNTHETIC_CASES, build_case
from core.mandrels import (
    faceplate_eggcrate,
    thin_shell,
    vary,
)
from core.tool_surface_eval import (
    largest_smooth,
    load_sample,
    normal_up,
    save_sample,
    score,
)


@pytest.fixture(scope="module")
def plate():
    return faceplate_eggcrate(seed=1, resolution=0.04)


@pytest.fixture(scope="module")
def male():
    return thin_shell("male", seed=1, resolution=0.02)


@pytest.fixture(scope="module")
def female():
    return thin_shell("female", seed=1, resolution=0.02)


def _mean_nz(mesh, truth):
    a = mesh.area_faces[truth]
    return float((mesh.face_normals[truth, 2] * a).sum() / a.sum())


def test_generators_have_truth_facing_up(plate, male, female):
    for mesh, truth, info in (plate, male, female):
        assert truth.shape == (len(mesh.faces),) and truth.any()
        assert _mean_nz(mesh, truth) > 0.5
        assert info["winding_consistent"] and info["watertight"]
        assert mesh.volume > 0  # outward normals


def test_surfaces_have_convex_and_concave_regions(plate, male, female):
    for mesh, truth, _ in (plate, male, female):
        top = mesh.submesh([np.flatnonzero(truth)], append=True)
        v = top.vertices
        # sign of the second difference along x on a vertex sample: both signs must occur
        order = np.argsort(v[:, 0])
        z = v[order, 2]
        curv = np.gradient(np.gradient(np.convolve(z, np.ones(25) / 25, "same")))
        assert (curv > 1e-9).any() and (curv < -1e-9).any()


def test_flange_is_not_truth():
    m, t, info = thin_shell("male", flange=True, resolution=0.02)
    m2, t2, _ = thin_shell("male", flange=False, resolution=0.02)
    assert t.sum() == t2.sum()
    assert len(m.faces) > len(m2.faces) and info["flange"]


def test_baselines_fail_where_expected(female):
    mesh, truth, _ = female
    s = score(normal_up(mesh), truth, mesh)
    assert s["recall"] > 0.95  # the trough faces up
    assert s["precision"] < 1.0  # but the flange lip also scores


def test_deterministic():
    a = build_case(SYNTHETIC_CASES[0])
    b = build_case(SYNTHETIC_CASES[0])
    assert np.array_equal(a[0].vertices, b[0].vertices) and np.array_equal(a[1], b[1])


@pytest.mark.parametrize("remesh", ["coarse", "fine", "irregular"])
def test_vary_keeps_labels_aligned(female, remesh):
    mesh, truth, _ = female
    m2, t2, info = vary(mesh, truth, remesh=remesh, seed=3)
    assert len(t2) == len(m2.faces) and t2.any()
    # re-derive labels from the original truth faces by proximity (the sample-format rule)
    sub = trimesh.Trimesh(mesh.vertices, mesh.faces[truth], process=False)
    from core.tool_surface_eval import labels_from_proximity

    ref = labels_from_proximity(m2, sub, tol=0.0008)
    assert score(t2, ref, m2)["iou"] >= 0.95


def test_vary_holes_noise_tilt(male):
    mesh, truth, _ = male
    m2, t2, info = vary(mesh, truth, noise_m=0.0002, tilt_deg=8.0, holes=3, seed=1)
    assert len(m2.faces) < len(mesh.faces) and len(t2) == len(m2.faces)
    assert not info["watertight"]
    assert _mean_nz(m2, t2) > 0.5
    assert abs(np.degrees(np.arccos(np.clip(m2.face_normals[t2, 2].mean(), -1, 1))) - 0) < 30


def test_score_simple():
    m = trimesh.creation.box((1, 1, 1))
    truth = m.face_normals[:, 2] > 0.5
    assert score(truth, truth, m) == {"iou": 1.0, "precision": 1.0, "recall": 1.0}
    s = score(np.ones(len(m.faces), bool), truth, m)
    assert s["recall"] == 1.0 and s["precision"] == pytest.approx(1 / 6)
    assert s["iou"] == pytest.approx(1 / 6)
    s = score(~truth, truth, m)
    assert s["iou"] == 0.0 and s["recall"] == 0.0
    with pytest.raises(ValueError):
        score(truth[:-1], truth, m)


def test_load_sample_recovers_truth(tmp_path, male):
    mesh, truth, _ = male
    save_sample(tmp_path / "s1", mesh, truth)
    m2, t2, info = load_sample(tmp_path / "s1")
    assert len(m2.faces) == len(mesh.faces)
    assert score(t2, truth, m2)["iou"] > 0.98
    assert info["truth_faces"] == int(t2.sum())


def test_load_sample_errors(tmp_path):
    (tmp_path / "a").mkdir()
    with pytest.raises(FileNotFoundError):
        load_sample(tmp_path / "a")
    box = trimesh.creation.box((1, 1, 1))
    (tmp_path / "b").mkdir()
    box.export(tmp_path / "b" / "mandrel.stl")
    (tmp_path / "b" / "tool_surface.step").write_text("x")
    with pytest.raises(ValueError, match="not a readable STEP file"):
        load_sample(tmp_path / "b")


def test_largest_smooth_runs(plate):
    mesh, truth, _ = plate
    pred = largest_smooth(mesh)
    assert pred.any() and pred.shape == truth.shape


# ---------------------------------------------------------------- T0b: the hard set
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

HARD = [c for c in SYNTHETIC_CASES if c.hard]


def _patches(mesh, mask, deg=30.0):
    """Number of smooth (<= deg) patches among the masked faces."""
    adj = mesh.face_adjacency
    ok = (mesh.face_adjacency_angles <= np.radians(deg)) & mask[adj[:, 0]] & mask[adj[:, 1]]
    a = adj[ok]
    n = len(mesh.faces)
    _, lab = connected_components(
        coo_matrix((np.ones(len(a)), (a[:, 0], a[:, 1])), shape=(n, n)), directed=False
    )
    return len(np.unique(lab[mask]))


@pytest.fixture(scope="module")
def built():
    t0 = time.perf_counter()
    out = {c.name: build_case(c) for c in SYNTHETIC_CASES}
    out["_seconds"] = time.perf_counter() - t0
    return out


def test_set_size_tags_and_build_time(built):
    assert len(SYNTHETIC_CASES) >= 30 and len(HARD) >= 14
    assert len({c.name for c in SYNTHETIC_CASES}) == len(SYNTHETIC_CASES)
    assert built["_seconds"] < 30
    for c in SYNTHETIC_CASES:
        m, t, info = built[c.name]
        assert t.any() and t.shape == (len(m.faces),)
        assert len(m.faces) <= 100_000
        assert info["tags"] and info["expect"] and info["name"] == c.name
        assert ("hard" in info["tags"]) == c.hard
        assert info["winding_consistent"]
    assert sum("ambiguous" in c.tags for c in SYNTHETIC_CASES) >= 1


def test_hard_cases_deterministic():
    for c in (HARD[0], HARD[3], HARD[-1]):
        a, b = build_case(c), build_case(c)
        assert np.array_equal(a[0].vertices, b[0].vertices) and np.array_equal(a[1], b[1])


def test_example_files_still_match_defaults():
    ex = Path(__file__).parents[1] / "examples" / "mandrels"
    for name in ("plate_coarse", "male_flange", "female_coarse"):
        m, t, _ = build_case(next(c for c in SYNTHETIC_CASES if c.name == name))
        assert len(np.load(ex / f"{name}_truth.npy")) == len(m.faces)
        assert np.array_equal(np.load(ex / f"{name}_truth.npy"), t)


@pytest.mark.parametrize(
    "name,patches", [("plate_joggle_step", 3), ("plate_ridge", 2), ("plate_ridge_valley", 4)]
)
def test_creased_truth_splits_into_patches(built, name, patches):
    m, t, _ = built[name]
    assert _patches(m, t) == patches
    assert _mean_nz(m, t) > 0.8


@pytest.mark.parametrize(
    "name,wall", [("female_steep70", 70), ("female_steep85", 85), ("female_steep88_noise", 88)]
)
def test_steep_truth_includes_walls_without_undercut(built, name, wall):
    m, t, info = built[name]
    nz = m.face_normals[t, 2]
    assert info["wall_deg"] == wall
    assert nz.min() > 0.0  # no undercut
    a = m.area_faces[t]
    assert a[nz < 0.5].sum() > 0.3 * a.sum()  # walls are a large share of the truth area
    wall_nz = np.cos(np.radians(wall))
    assert np.isclose(np.sort(nz)[: max(10, int(0.02 * len(nz)))].mean(), wall_nz, atol=0.03)


def test_decoys_are_bigger_than_truth(built):
    for name in ("male_decoy_flange", "male_ambiguous_flange"):
        m, t, info = built[name]
        up = (m.face_normals[:, 2] > 0.95) & ~t
        assert m.area_faces[up].sum() > 3 * m.area_faces[t].sum()
    assert built["male_ambiguous_flange"][2]["expect"] == {"ambiguous": True}
    m, t, _ = built["male_decoy_flange"]
    assert m.triangles_center[t, 2].min() > m.triangles_center[~t & (m.face_normals[:, 2] > 0.95), 2].max()
    m, t, info = built["plate_plinth_decoy"]
    pad = (m.face_normals[:, 2] > 0.99) & ~t & (np.abs(m.triangles_center[:, 2] - 0.05) < 1e-6)
    assert m.area_faces[pad].sum() > m.area_faces[t].sum()


def test_pins_and_perforation(built):
    m, t, info = built["plate_pins"]
    assert info["pins"] == 4
    xy = np.array(info["pin_xy"])
    c = m.triangles_center
    near = np.hypot(c[:, None, 0] - xy[:, 0], c[:, None, 1] - xy[:, 1]).min(1) < 0.012
    assert near.sum() > 50 and not (t & near).any()  # pin faces and the surface under them are not truth
    m, t, info = built["plate_perforated"]
    assert info["perforated"] and info["watertight"] and m.volume > 0
    m0, t0, _ = built["plate_a"]
    assert t.sum() < t0.sum()


def test_tilt_cases_are_tilted(built):
    for name, deg in (("plate_tilt25", 25), ("female_tilt30", 30)):
        m, t, _ = built[name]
        a = m.area_faces[t]
        mean_n = (m.face_normals[t] * a[:, None]).sum(0)
        tilt = np.degrees(np.arccos(mean_n[2] / np.linalg.norm(mean_n)))
        assert abs(tilt - deg) < 6
