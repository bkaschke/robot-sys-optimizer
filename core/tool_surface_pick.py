"""Tool-surface detection, part 2: score regions, pick and grow, confidence, sanding checks (T2).

``detect_tool_surface(mesh)`` segments the mesh and describes each region (T1,
``core.tool_surface.analyze``), then:

1. **Seeds**: every up-facing, visible region (``frac_up``/``mean_up`` >= ``up_min``, visible from
   straight up) is a seed. Its region score is ``area_frac * up * vis``.
2. **Grow** each seed across region boundaries (``_grow``):
   - tangent joints into up-facing visible regions (run-off, smooth transitions);
   - creases (convex or concave, mean dihedral <= ``crease_max_deg``) into up-facing visible
     regions (joggles, ridges, valleys);
   - concave or tangent joints into steep walls that face *into* the selection (a female
     cavity: the walls surround the floor). Walls around a raised island face away and are
     not added, so a base flange never swallows the male form that stands on it;
   - never across convex breaks into steep regions (flange returns, faceplate sides): those
     become ``secondary`` candidates.
3. **Pick** the grown set with the best set score ``sum(area * quality) * reach``, where
   ``reach = min(1, top_height / reach_ref) ** 2`` and ``top_height`` is the set's highest
   point relative to the part (0 = bottom, 1 = top). The tool surface reaches the top of the
   part; a base flange or a machined pad at the bottom does not.
4. **Confidence** from the margin to the best alternative set, the share of near-threshold
   area next to the selection, and "open creases" (internal edges sharper than the feature
   angle: the segmentation could not close a crease, so the boundary is unclear).
5. **Sanding checks** (warnings only): concave curvature radius below the pad radius, and
   steep selected regions.

No viser / ui imports; SI units (m, m^2; degrees only in reasons and warnings).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from core.tool_surface import analyze, edge_table

__all__ = ["DEFAULT_PARAMS", "RegionScore", "Candidate", "ToolSurfaceGuess", "detect_tool_surface", "detect"]

# Every threshold the decision uses. Rationale in the comments; tuned on the synthetic set.
DEFAULT_PARAMS = dict(
    up_min=0.5,  # up-facing: >= 50 % of area with n.up > 0.3 and mean n.up >= 0.5 (<= 60 deg slope)
    vis_min=0.5,  # visible: >= 50 % of samples reachable straight from above
    crease_max_deg=50.0,  # creases up to this dihedral join up-facing patches (joggle 40, flange 57+)
    wall_vis_min=0.5,  # a cavity wall must be visible from some cone direction (88 deg walls: ~0.8)
    wall_up_min=-0.05,  # ... and not face down (the outer skin of a wall faces down)
    inward_min=0.3,  # ... and face into the selection (cavity), not away from it (island)
    reach_ref=0.5,  # a set whose top is below half the part height is penalised (base, plinth)
    island_min=0.05,  # a raised form >= 5 % of a set's area standing on it makes the set a base
    base_factor=0.02,  # ... and scales its score: a base must be 50x larger to win
    max_seeds=8,  # seeds grown (by region score); the rest cannot win
    open_crease_deg=30.0,  # internal edges sharper than this inside a region: an open crease
    steep_deg=60.0,  # sanding check: selected area steeper than this
    secondary_min_frac=0.02,  # secondary candidates smaller than this share of the selection are dropped
)


@dataclass
class RegionScore:
    label: int
    area_frac: float
    score: float  # region score area_frac * up * vis (seed ranking)
    components: dict  # area, up, vis, height, ...
    role: str  # "seed" | "selected" | "secondary" | "rejected"
    reason: str


@dataclass
class Candidate:
    labels: tuple
    faces: np.ndarray = field(repr=False)
    area_m2: float = 0.0
    kind: str = ""  # "steep_convex" | "below_threshold"
    reason: str = ""


@dataclass
class ToolSurfaceGuess:
    faces: np.ndarray = field(repr=False)
    mask: np.ndarray = field(repr=False)
    confidence: float = 0.0
    regions: list = field(default_factory=list)  # RegionScore, best first
    secondary: list = field(default_factory=list)  # Candidate
    warnings: list = field(default_factory=list)
    summary: str = ""
    selected_labels: tuple = ()
    labels: np.ndarray = field(default=None, repr=False)  # region per face
    area_m2: float = 0.0
    visible_frac: float = 0.0
    confidence_parts: dict = field(default_factory=dict)


# ------------------------------------------------------------------ helpers
def _frame(up):
    u = np.asarray(up, float)
    u = u / np.linalg.norm(u)
    return u


def _region_geometry(mesh, labels, k, u):
    """Per region: area-weighted centroid [k,3]; per face: horizontal normal [n,3]."""
    af = np.asarray(mesh.area_faces, float)
    c = np.asarray(mesh.triangles_center, float)
    a = np.maximum(np.bincount(labels, weights=af, minlength=k), 1e-300)
    cen = (
        np.column_stack([np.bincount(labels, weights=af * c[:, i], minlength=k) for i in range(3)])
        / a[:, None]
    )
    fn = np.asarray(mesh.face_normals, float)
    nh = fn - np.outer(fn @ u, u)
    return cen, c, nh, af


def _inward(faces, point, centers, nh, af, u):
    """Area-weighted mean of n_h . unit(point - p)_h over ``faces``: +1 facing ``point``."""
    d = point[None, :] - centers[faces]
    d = d - np.outer(d @ u, u)
    dn = np.linalg.norm(d, axis=1)
    ok = dn > 1e-12
    if not ok.any():
        return 0.0
    v = np.einsum("ij,ij->i", nh[faces][ok], d[ok] / dn[ok, None])
    w = af[faces][ok]
    return float((v * w).sum() / max(w.sum(), 1e-300))


def _is_up(f, p):
    return f.frac_up >= p["up_min"] and f.mean_up >= p["up_min"] and f.visible_up_frac >= p["vis_min"]


def _slope_deg(f):
    return float(np.degrees(np.arccos(np.clip(f.mean_up, -1.0, 1.0))))


def _vis(f):
    return 0.5 * (f.visible_up_frac + f.visible_mean)


def _grow(seed, feats, geo, u, p):
    """Grow from ``seed``.

    Returns (members {label: reason}, secondary {label: (kind, reason)}, island walls): island
    walls are steep walls joined concavely that face *away* from the surface (a raised form
    standing on it).
    """
    cen, centers, nh, af = geo
    members = {seed: "seed"}
    secondary = {}
    island = set()
    queue = [seed]
    while queue:
        m = queue.pop(0)
        for nb, b in sorted(feats[m].neighbors.items()):
            if nb in members:
                continue
            f = feats[nb]
            up = _is_up(f, p)
            why = None
            if up and (b.kind == "tangent" or b.angle_deg <= p["crease_max_deg"]):
                what = "tangent joint" if b.kind == "tangent" else f"{b.angle_deg:.0f} deg {b.kind} crease"
                why = f"joined by a {what}, faces up, {f.visible_up_frac:.0%} visible from above"
            elif (
                b.kind in ("concave", "tangent")
                and f.mean_up >= p["wall_up_min"]
                and f.visible_frac >= p["wall_vis_min"]
            ):
                ups = [r for r in members if feats[r].mean_up >= p["up_min"]] or [seed]
                w = np.array([feats[r].area for r in ups])
                floor = (cen[ups] * w[:, None]).sum(0) / w.sum()
                inward = _inward(f.faces, floor, centers, nh, af, u)
                if inward >= p["inward_min"]:
                    why = (
                        f"steep wall ({_slope_deg(f):.0f} deg) joined {b.kind}ly, facing into the"
                        f" surface (cavity), {f.visible_frac:.0%} visible"
                    )
                elif inward <= -p["inward_min"]:
                    island.add(nb)
            if why is not None:
                members[nb] = why
                secondary.pop(nb, None)
                queue.append(nb)
                continue
            if nb in secondary or f.visible_frac < 0.3:
                continue
            if b.kind == "convex" and -0.2 < f.mean_up < 0.8:  # falls away: a return or a side
                secondary[nb] = (
                    "steep_convex",
                    f"steep ({_slope_deg(f):.0f} deg) beyond a {b.angle_deg:.0f} deg convex break"
                    " (flange return or side)",
                )
            elif f.mean_up > 0.3 and f.visible_up_frac > 0.3:
                secondary[nb] = (
                    "below_threshold",
                    f"faces up ({_slope_deg(f):.0f} deg) but joined by a {b.angle_deg:.0f} deg"
                    f" {b.kind} break or only {f.visible_up_frac:.0%} visible",
                )
    return members, secondary, island - set(members)


def _island_area(members, island, feats, p):
    """Area of raised forms standing on the set: island walls + the up-facing tops beyond them."""
    top_h = max(feats[r].height_mean for r in members)
    tops = {
        nb
        for w in island
        for nb, b in feats[w].neighbors.items()
        if nb not in members and b.kind == "convex" and _is_up(feats[nb], p) and feats[nb].height_mean > top_h
    }
    if not tops:
        return 0.0
    return sum(feats[r].area for r in island | tops)


def _quality(f, why):
    """Weight of a member's area: up-facing patches by mean n.up x visibility (a flat patch 1,
    a 60 deg slope 0.5); cavity walls ride with their floor at their visible share."""
    if why.startswith("steep wall"):
        return f.visible_frac
    return max(f.mean_up, 0.0) * _vis(f)


def _set_score(members, island, feats, p):
    """(score, top height, area_frac, is_base) of a grown set."""
    tot = sum(feats[r].area_frac * _quality(feats[r], members[r]) for r in members)
    top = max(feats[r].height_max for r in members)
    reach = min(1.0, top / p["reach_ref"]) ** 2
    area = sum(feats[r].area for r in members)
    base = _island_area(members, island, feats, p) >= p["island_min"] * area
    return (
        tot * reach * (p["base_factor"] if base else 1.0),
        top,
        sum(feats[r].area_frac for r in members),
        base,
    )


def _open_crease(mesh, labels, sel_labels, feats, deg):
    """Length of internal edges sharper than ``deg`` inside the given (up-facing) regions / sqrt(area).

    Walls are left out by the caller: tessellation corners there are not a fading crease."""
    et = edge_table(mesh)
    la, lb = labels[et.pairs[:, 0]], labels[et.pairs[:, 1]]
    inside = (la == lb) & np.isin(la, list(sel_labels)) & (et.angle > np.radians(deg))
    area = sum(feats[r].area for r in sel_labels)
    return float(et.length[inside].sum() / np.sqrt(max(area, 1e-300)))


# ------------------------------------------------------------------ sanding checks
def concave_radius_check(mesh, mask, pad_radius_m, min_deg=2.0, sharp_deg=20.0):
    """Warnings for concave curvature tighter than the pad inside the selection.

    Per concave edge between two selected faces: radius ~ (mean width of the two triangles
    across the edge, 2 * area / edge length) / dihedral. That is the radius of a faceted fillet,
    and about the tessellation size for a sharp concave corner. Edges
    flatter than ``min_deg`` are ignored (noise). Tight edges are grouped into connected runs;
    a run warns when it is longer than five mean edges (isolated noisy edges do not). A run
    whose median dihedral is above ``sharp_deg`` is a sharp edge (radius ~0: the estimate would
    only be the mesh resolution); such edges are tight whatever the pad.
    """
    if pad_radius_m is None or pad_radius_m <= 0:
        return []
    et = edge_table(mesh)
    a, b = et.pairs[:, 0], et.pairs[:, 1]
    af = np.asarray(mesh.area_faces, float)
    sel = mask[a] & mask[b] & ~et.convex & (et.angle > np.radians(min_deg)) & (et.length > 0)
    if not sel.any():
        return []
    width = (af[a[sel]] + af[b[sel]]) / et.length[sel]  # mean of 2 * area / length
    r = width / et.angle[sel]
    tight = (r < pad_radius_m) | (et.angle[sel] >= np.radians(sharp_deg))  # a sharp edge has radius ~0
    if not tight.any():
        return []
    idx = np.flatnonzero(sel)[tight]
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    # runs: tight edges joined through shared vertices, and through edges between two faces that
    # both touch a tight edge (so the parallel facet lines of one fillet form one run)
    on = np.zeros(len(mesh.faces), bool)
    on[a[idx]] = True
    on[b[idx]] = True
    vv = np.vstack([et.verts[idx], et.verts[on[a] & on[b]]])
    nv = et.n_vertices
    g = coo_matrix((np.ones(len(vv)), (vv[:, 0], vv[:, 1])), shape=(nv, nv))
    comp = connected_components(g, directed=False)[1][et.verts[idx, 0]]
    rr, ln = r[tight], et.length[idx]
    ang = np.degrees(et.angle[idx])
    min_len = 5.0 * et.edge_scale
    groups: dict = {}  # identical messages (symmetric corners) are counted, not repeated
    for k in np.unique(comp):
        s = comp == k
        length = float(ln[s].sum())
        if length < min_len:
            continue
        rad = float(np.median(rr[s])) * 1000
        sharp = float(np.median(ang[s])) >= sharp_deg  # one sharp edge: the radius is below the mesh size
        what = f"~0 mm (sharp {np.median(ang[s]):.0f} deg edge)" if sharp else f"~{rad:.0f} mm"
        key = (what, round(length * 1000))
        groups[key] = groups.get(key, 0) + 1
    out = []
    pad = pad_radius_m * 1000
    for (what, length), count in sorted(groups.items(), key=lambda kv: -kv[0][1])[:5]:
        where = f"{count} places, {length} mm of edge each" if count > 1 else f"{length} mm of edge"
        out.append(
            f"concave radius {what} smaller than the pad radius {pad:.0f} mm:"
            f" cannot be sanded with this pad ({where})"
        )
    if len(groups) > 5:
        out.append(f"... and {len(groups) - 5} more concave spots tighter than the pad")
    return out


def steep_check(mesh, mask, up, steep_deg, min_frac=0.02):
    """Warning when more than ``min_frac`` of the selected area is steeper than ``steep_deg``."""
    af = np.asarray(mesh.area_faces, float)
    nup = np.asarray(mesh.face_normals, float) @ _frame(up)
    tot = af[mask].sum()
    steep = mask & (nup < np.cos(np.radians(steep_deg)))
    share = af[steep].sum() / max(tot, 1e-300)
    if share < min_frac:
        return []
    worst = float(np.degrees(np.arccos(np.clip(nup[steep].min(), -1, 1))))
    return [
        f"steep: {share:.0%} of the surface is steeper than {steep_deg:.0f} deg (up to {worst:.0f} deg):"
        " check the approach angle"
    ]


# ------------------------------------------------------------------ main entry
def detect_tool_surface(
    mesh: trimesh.Trimesh,
    up=(0.0, 0.0, 1.0),
    cad_face_ids=None,
    pad_radius_m=None,
    labels=None,
    features=None,
    **params,
) -> ToolSurfaceGuess:
    """Guess the tool surface of a mandrel; see the module docstring.

    ``labels``/``features`` may be passed from an earlier ``analyze`` call to skip it.
    Unknown ``params`` keys raise (typos must not silently fall back to defaults).
    """
    bad = set(params) - set(DEFAULT_PARAMS)
    if bad:
        raise TypeError(f"unknown tool-surface parameters: {sorted(bad)}")
    p = {**DEFAULT_PARAMS, **params}
    u = _frame(up)
    if labels is None or features is None:
        labels, features = analyze(mesh, up=u, cad_face_ids=cad_face_ids)
    labels = np.asarray(labels, np.int64)
    feats = features
    k = len(feats)
    n = len(mesh.faces)
    geo = _region_geometry(mesh, labels, k, u)

    rscore = np.array([f.area_frac * f.frac_up * _vis(f) for f in feats])
    seeds = [r for r in np.argsort(-rscore, kind="stable") if _is_up(feats[r], p)][: p["max_seeds"]]
    warnings: list = []
    if not seeds:  # nothing faces up and is visible: fall back to the best region, flagged
        seeds = [int(np.argmax(rscore))] if k else []
        warnings.append("no up-facing region visible from above: is the up direction right?")
    grown = []
    for s in seeds:
        members, sec, island = _grow(int(s), feats, geo, u, p)
        sc, top, af_, base = _set_score(members, island, feats, p)
        grown.append(
            dict(seed=int(s), members=members, secondary=sec, score=sc, top=top, area_frac=af_, base=base)
        )
    grown.sort(key=lambda g: -g["score"])
    best = grown[0]
    sel = best["members"]
    mask = np.isin(labels, list(sel))
    faces = np.flatnonzero(mask)
    af = geo[3]
    sel_area = float(af[mask].sum())

    # best alternative explanation: another seed's set that overlaps little with the choice and
    # is not one of its own returns (a flange return is explained by the chosen surface)
    alt = 0.0
    alt_name = ""
    for g in grown[1:]:
        if best["secondary"].get(g["seed"], ("",))[0] == "steep_convex":
            continue
        ov = sum(feats[r].area_frac for r in g["members"] if r in sel) / max(g["area_frac"], 1e-300)
        if ov < 0.5:
            alt, alt_name = g["score"], f"r{g['seed']}"
            break
    margin = 1.0 - alt / max(best["score"], 1e-300)
    # near-threshold area next to the selection: up-facing regions just outside the rules
    amb = 0.0
    for r, (kind, _) in best["secondary"].items():
        f = feats[r]
        if kind == "below_threshold" or (kind == "steep_convex" and f.mean_up > p["up_min"] - 0.1):
            nb = max((feats[m].neighbors[r].angle_deg for m in sel if r in feats[m].neighbors), default=90.0)
            if nb < p["crease_max_deg"] + 15:
                amb += f.area
    amb_share = amb / max(sel_area + amb, 1e-300)
    up_sel = [r for r, why in sel.items() if not why.startswith("steep wall")]
    open_c = _open_crease(mesh, labels, up_sel, feats, p["open_crease_deg"])
    seg_amb = float(np.clip((open_c - 0.4) / 0.4, 0.0, 1.0))
    c_margin = float(np.clip(margin / 0.5, 0.0, 1.0))
    confidence = float(c_margin * (1.0 - 0.5 * amb_share) * (1.0 - 0.7 * seg_amb))
    if seg_amb > 0:
        warnings.append(
            f"open crease inside the surface ({open_c:.2f} x sqrt(area) of edges sharper than"
            f" {p['open_crease_deg']:.0f} deg): the boundary is unclear, check the edges of the selection"
        )
    if margin < 0.5 and alt_name:
        warnings.append(
            f"another surface ({alt_name}) scores {alt / max(best['score'], 1e-300):.0%} of the choice"
        )

    # secondary candidates, biggest first
    secondary = []
    for r, (kind, why) in sorted(best["secondary"].items(), key=lambda kv: -feats[kv[0]].area):
        if feats[r].area < p["secondary_min_frac"] * sel_area:
            continue
        secondary.append(Candidate((int(r),), feats[r].faces, float(feats[r].area), kind, why))

    # per-region scores and reasons
    regions = []
    for r in np.argsort(-rscore, kind="stable"):
        f = feats[r]
        comp = dict(
            area=f.area_frac,
            up=f.frac_up,
            mean_up=f.mean_up,
            vis=_vis(f),
            height_max=f.height_max,
        )
        if r in sel:
            role, why = ("seed" if r == best["seed"] else "selected"), sel[r]
            if r == best["seed"]:
                why = (
                    f"best set: {best['area_frac']:.1%} of the part area,"
                    f" {f.visible_up_frac:.0%} visible from above"
                )
        elif r in best["secondary"]:
            role, why = "secondary", best["secondary"][r][1]
        elif f.mean_up < 0:
            role, why = "rejected", f"faces down (mean n.up {f.mean_up:+.2f})"
        elif f.visible_frac < p["wall_vis_min"]:
            role, why = "rejected", f"hidden from above ({f.visible_frac:.0%} visible)"
        elif _is_up(f, p):
            g = next((g for g in grown if g["seed"] == r), None)
            if g is not None and g["base"]:
                why = "base or flange: a raised form stands on it (its walls face away)"
            elif g is not None and g["top"] < p["reach_ref"]:
                why = f"low: top at {g['top']:.0%} of the part height (base, flange or pad)"
            else:
                why = "separate up-facing patch, not joined to the chosen surface"
            role = "rejected"
        else:
            role, why = "rejected", f"steep ({_slope_deg(f):.0f} deg), not a cavity wall of the surface"
        regions.append(RegionScore(int(r), f.area_frac, float(rscore[r]), comp, role, why))

    warnings += steep_check(mesh, mask, u, p["steep_deg"])
    warnings += concave_radius_check(mesh, mask, pad_radius_m)
    vis = sum(feats[r].visible_frac * feats[r].area for r in sel) / max(sel_area, 1e-300)
    summary = (
        f"Tool surface: {len(sel)} region{'s' if len(sel) != 1 else ''}, {sel_area:.2f} m²,"
        f" {vis:.0%} visible from above (confidence {confidence:.2f})"
    )
    return ToolSurfaceGuess(
        faces=faces,
        mask=mask,
        confidence=confidence,
        regions=regions,
        secondary=secondary,
        warnings=warnings,
        summary=summary,
        selected_labels=tuple(sorted(int(r) for r in sel)),
        labels=labels,
        area_m2=sel_area,
        visible_frac=float(vis),
        confidence_parts=dict(margin=margin, ambiguous_share=amb_share, open_crease=open_c, n_faces=n),
    )


def detect(mesh: trimesh.Trimesh) -> np.ndarray:
    """Bool mask per face (the eval script's detector interface)."""
    return detect_tool_surface(mesh).mask
