"""Tool-surface segmentation and region features (T1)."""

import numpy as np
import pytest
import trimesh

from core.mandrels import _slab, faceplate_eggcrate, thin_shell
from core.tool_surface import analyze, segment
from core.tool_surface_features import region_features, region_graph, view_directions

try:  # T0b moved the case list into core.mandrel_cases
    from core.mandrel_cases import SYNTHETIC_CASES, build_case
except ImportError:  # pragma: no cover - older tree
    from core.mandrels import SYNTHETIC_CASES, build_case


def oracle(features, truth, mesh):
    """Best IoU reachable by choosing whole regions: every region mostly (> 50 %) in the truth.

    Returns (iou, n_regions_chosen, purity of the chosen union).
    """
    a = mesh.area_faces
    truth = np.asarray(truth, bool)
    pred = np.zeros(len(truth), bool)
    n = 0
    for f in features:
        if a[f.faces][truth[f.faces]].sum() > 0.5 * f.area:
            pred[f.faces] = True
            n += 1
    union = a[pred | truth].sum()
    iou = float(a[pred & truth].sum() / union) if union > 0 else 1.0
    purity = float(a[pred & truth].sum() / a[pred].sum()) if pred.any() else 0.0
    return iou, n, purity


def _cases(hard):
    out = []
    for c in SYNTHETIC_CASES:
        is_hard = "hard" in getattr(c, "tags", ())
        if is_hard == hard:
            out.append(c)
    return out


def _region_of(features, mesh, mask):
    """The region holding most of ``mask``'s area."""
    a = mesh.area_faces
    return max(features, key=lambda f: a[f.faces][mask[f.faces]].sum())


@pytest.fixture(scope="module")
def plate():
    m, t, _ = faceplate_eggcrate(seed=1, resolution=0.04)
    lab, feats = analyze(m)
    return m, t, lab, feats


@pytest.mark.parametrize("case", _cases(False), ids=lambda c: c.name)
def test_easy_truth_is_a_few_whole_regions(case):
    m, t, _ = build_case(case)
    feats = region_features(m, segment(m), compute_visibility=False)
    iou, n, purity = oracle(feats, t, m)
    assert 1 <= n <= 3, f"{case.name}: truth spread over {n} regions"
    assert purity >= 0.95, f"{case.name}: purity {purity:.3f}"
    assert iou >= 0.95, f"{case.name}: oracle IoU {iou:.3f}"


@pytest.mark.parametrize("case", _cases(True), ids=lambda c: c.name)
def test_hard_oracle(case):
    """Hard cases: whole regions reach the case's IoU target, except deliberately ambiguous ones."""
    m, t, _ = build_case(case)
    feats = region_features(m, segment(m), compute_visibility=False)
    iou, n, purity = oracle(feats, t, m)
    print(f"{case.name}: oracle IoU {iou:.3f}, {n} regions, purity {purity:.3f}")
    if case.expect.get("ambiguous"):
        return  # e.g. male_ambiguous_flange: the crease fades out, truth is not a union of regions
    assert iou >= case.expect.get("min_iou", 0.9), f"{case.name}: oracle IoU {iou:.3f}"


def test_faceplate_visibility(plate):
    m, t, lab, feats = plate
    top = _region_of(feats, m, t)
    assert top.visible_frac > 0.98 and top.visible_up_frac > 0.98
    under = _region_of(feats, m, (m.face_normals[:, 2] < -0.9) & (m.triangles_center[:, 2] > 0.2))
    assert under.mean_up < -0.9
    assert under.visible_frac < 0.05
    # rib tops end inside the plate: hidden from every direction
    up_regions = [f for f in feats if f.label != top.label and f.mean_up > 0.9]
    ribs = [f for f in up_regions if f.height_min > top.height_min - 0.1]
    assert len(ribs) >= 10
    assert max(f.visible_frac for f in ribs) == 0.0
    # frame rails / bosses low down: hidden from straight up, but the egg-crate is open at the
    # sides, so slanted rays from outside the plate edge reach them
    low = [f for f in up_regions if f not in ribs]
    a = sum(f.area for f in low)
    assert sum(f.area * f.visible_up_frac for f in low) / a < 0.05


def test_female_trough_visible_outer_skin_hidden():
    m, t, _ = thin_shell("female", seed=1, resolution=0.02)
    lab, feats = analyze(m)
    inner = _region_of(feats, m, t)
    assert inner.visible_frac > 0.95
    assert inner.visible_up_frac > 0.9
    skin = _region_of(feats, m, m.face_normals[:, 2] < -0.5)
    assert skin.visible_frac < 0.05


def _pocket(res=0.01, depth=0.05, wall=0.004):
    """Flat-bottomed pocket in a slab (height field), walls ~85 deg steep."""
    base = np.arange(-0.2, 0.2001, res)
    br = [-0.1 - wall, -0.1, 0.1, 0.1 + wall]
    xs = np.unique(np.round(np.concatenate([base[np.min(np.abs(base[:, None] - br), 1) > 0.3 * res], br]), 9))
    X, Y = np.meshgrid(xs, xs)
    inside = (np.abs(X) <= 0.1) & (np.abs(Y) <= 0.1)
    z = np.where(inside, 0.1 - depth, 0.1)
    v, f, ntop = _slab(xs, xs, z, 0.02)
    m = trimesh.Trimesh(v, f, process=False)
    if m.volume < 0:
        m = trimesh.Trimesh(v, f[:, ::-1], process=False)
    c = m.triangles_center
    floor = (np.abs(c[:, 0]) < 0.1) & (np.abs(c[:, 1]) < 0.1) & (np.abs(c[:, 2] - (0.1 - depth)) < 1e-6)
    rim = (m.face_normals[:, 2] > 0.99) & (np.abs(c[:, 2] - 0.1) < 1e-6)
    return m, floor, rim


def test_boundary_kinds():
    m, t, _ = faceplate_eggcrate(seed=1, resolution=0.04)
    feats = region_features(m, segment(m), compute_visibility=False)
    top = _region_of(feats, m, t)
    assert len(top.neighbors) == 4  # the four side strips
    assert all(b.kind == "convex" and b.angle_deg > 60 for b in top.neighbors.values())
    assert top.sharp_frac > 0.99 and top.open_frac == 0

    pm, floor, rim = _pocket()
    feats = region_features(pm, segment(pm))
    fl, rm = _region_of(feats, pm, floor), _region_of(feats, pm, rim)
    assert fl.label != rm.label
    walls = list(fl.neighbors)
    assert walls and all(fl.neighbors[w].kind == "concave" for w in walls)
    # each wall meets the rim across a convex break
    for w in walls:
        b = feats[w].neighbors.get(rm.label)
        assert b is not None and b.kind == "convex"
    g = region_graph(feats)
    assert all(g[a][b].length == pytest.approx(g[b][a].length) for a in g for b in g[a])
    assert fl.visible_frac > 0.95  # the pocket floor is open to the cone


def test_non_watertight_soup_and_shells(plate):
    m, t, lab, feats = plate
    rng = np.random.default_rng(0)
    keep = rng.random(len(m.faces)) > 0.02
    holey = trimesh.Trimesh(m.vertices, m.faces[keep], process=False)
    lab2, f2 = analyze(holey)
    top = _region_of(f2, holey, t[keep])
    assert top.open_frac > 0.1 and top.n_holes > 10
    # triangle soup (no shared vertices): same segmentation as the indexed mesh
    soup = trimesh.Trimesh(
        m.triangles.reshape(-1, 3), np.arange(3 * len(m.faces)).reshape(-1, 3), process=False
    )
    assert len(soup.face_adjacency) == 0
    assert np.array_equal(segment(soup), lab)
    # degenerate inputs do not crash
    tri = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], process=False)
    lab1, f1 = analyze(tri)
    assert lab1.tolist() == [0] and f1[0].open_frac == 1.0
    deg = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], [[0, 1, 2], [0, 1, 3]], process=False)
    analyze(deg)
    with pytest.raises(ValueError):
        region_features(m, lab[:-1])


def test_cad_face_ids_merge_tangent_only():
    box = trimesh.creation.box((1, 1, 1))
    ids = np.arange(len(box.faces))  # every triangle its own "CAD face"
    lab = segment(box, cad_face_ids=ids)
    assert len(np.unique(lab)) == 6
    cyl = trimesh.creation.cylinder(radius=0.1, height=0.3, sections=64)
    c = cyl.triangles_center
    side = np.abs(cyl.face_normals[:, 2]) < 0.5
    ids = np.where(side, np.where(c[:, 0] > 0, 1, 2), np.where(c[:, 2] > 0, 3, 4))
    lab = segment(cyl, cad_face_ids=ids)
    assert len(np.unique(lab)) == 3  # the two side halves meet tangentially; caps stay apart
    assert len(np.unique(lab[side])) == 1
    with pytest.raises(ValueError):
        segment(cyl, cad_face_ids=ids[:-1])


def test_fragments_merge_but_sharp_strips_stay():
    """Tessellation noise is merged; small regions behind sharp edges are not."""
    m, t, _ = thin_shell("male", seed=4, resolution=0.02)
    rng = np.random.default_rng(1)
    noisy = m.copy()
    noisy.vertices = noisy.vertices + rng.normal(0, 0.0015, noisy.vertices.shape)
    raw = segment(noisy, min_region_frac=0.0)
    merged = segment(noisy)
    assert len(np.unique(merged)) < len(np.unique(raw))
    feats = region_features(noisy, merged, compute_visibility=False)
    iou, n, purity = oracle(feats, t, noisy)
    assert purity > 0.95 and iou > 0.9
    # the faceplate's thin side strips (well under 0.2 % of the area) survive as regions
    p, pt, _ = faceplate_eggcrate(seed=1, resolution=0.04)
    pf = region_features(p, segment(p), compute_visibility=False)
    assert oracle(pf, pt, p)[2] > 0.999


def test_view_directions_and_rtree_fallback():
    d = view_directions((0, 0, 1))
    assert len(d) == 15 and np.allclose(np.linalg.norm(d, axis=1), 1)
    assert np.allclose(d[0], [0, 0, 1]) and np.degrees(np.arccos(d[:, 2].min())) == pytest.approx(55)
    box = trimesh.creation.box((1, 1, 0.2))
    box.ray = trimesh.ray.ray_triangle.RayMeshIntersector(box)
    feats = region_features(box, segment(box))
    top = _region_of(feats, box, box.face_normals[:, 2] > 0.9)
    bot = _region_of(feats, box, box.face_normals[:, 2] < -0.9)
    assert top.visible_up_frac == 1.0 and bot.visible_frac == 0.0
    assert top.n_vis_samples <= 300


def test_tilted_up_vector():
    m, t, _ = faceplate_eggcrate(seed=1, resolution=0.04)
    R = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
    m.apply_transform(R)
    up = R[:3, :3] @ [0, 0, 1]
    feats = region_features(m, segment(m), up=up)
    top = _region_of(feats, m, t)
    assert top.mean_up > 0.95 and top.visible_up_frac > 0.98 and top.height_mean > 0.8
