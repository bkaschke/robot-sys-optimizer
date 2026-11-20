"""The synthetic mandrel set: EASY cases (smooth single-patch surfaces) and HARD cases.

Every case records ``tags`` and ``expect`` in its info dict. ``expect`` is what a good
detector should reach: ``{"min_iou": 0.85}`` or ``{"ambiguous": True}`` (a later detector
should report low confidence there).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.mandrels import faceplate_eggcrate, thin_shell, vary
from core.mandrels_hard import decoy_shell, steep_pocket


@dataclass(frozen=True)
class MandrelCase:
    name: str
    generator: str  # faceplate | male | female | steep | decoy
    gen_kwargs: dict = field(default_factory=dict)
    vary_kwargs: dict = field(default_factory=dict)
    note: str = ""
    tags: tuple = ("easy",)
    expect: dict = field(default_factory=lambda: {"min_iou": 0.9})

    @property
    def hard(self) -> bool:
        return "hard" in self.tags


def _joggle_step(x, y):
    """A 40 degree up-ramp over 0.12 m (a joggle step): three smooth patches in x."""
    k = np.tan(np.radians(40.0))
    return k * (np.clip(x + 0.30, 0, 0.12))


def _ridge(x, y):
    """Convex crease along x = 0.1, 40 degree dihedral: two patches."""
    return -np.tan(np.radians(20.0)) * np.abs(x - 0.1)


def _ridge_valley(x, y):
    """Convex 40 degree crease along x = -0.1 and a concave 36 degree crease along y = 0.1: four patches."""
    return -np.tan(np.radians(20.0)) * np.abs(x + 0.1) + np.tan(np.radians(18.0)) * np.abs(y - 0.1)


_FLAT = dict(saddle=0.015, bump=0.02)  # keep the underlying relief gentle so creases dominate
_JOG_BREAKS = {
    "step": (([-0.30, -0.18]), []),
    "ridge": (([0.1]), []),
    "cross": (([-0.1]), ([0.1])),
}
_H = ("hard",)
_BAD = {"min_iou": 0.85}

SYNTHETIC_CASES: tuple[MandrelCase, ...] = (
    MandrelCase("plate_a", "faceplate", dict(seed=1), note="faceplate on egg-crate"),
    MandrelCase("plate_b_tilt5", "faceplate", dict(seed=2, saddle=0.07, tilt=5.0)),
    MandrelCase("plate_c_tilt10_noise", "faceplate", dict(seed=3, bump=0.08, tilt=10.0, noise=0.0003)),
    MandrelCase("plate_coarse", "faceplate", dict(seed=4, resolution=0.04), note="coarse grid"),
    MandrelCase("plate_holes", "faceplate", dict(seed=5), dict(holes=4, seed=11), "holes in the surface"),
    MandrelCase("male_flange", "male", dict(seed=1, flange=True)),
    MandrelCase("male_noflange", "male", dict(seed=2, flange=False)),
    MandrelCase("male_tilt5", "male", dict(seed=3, tilt=5.0, thickness=0.006)),
    MandrelCase("male_irregular", "male", dict(seed=4), dict(remesh="irregular", seed=3), "irregular mesh"),
    MandrelCase("female_flange", "female", dict(seed=1, flange=True)),
    MandrelCase("female_noflange", "female", dict(seed=2, flange=False)),
    MandrelCase("female_tilt10_noise", "female", dict(seed=3, tilt=10.0, noise=0.0002)),
    MandrelCase("female_coarse", "female", dict(seed=4), dict(remesh="coarse", seed=5), "decimated"),
    MandrelCase("female_fine_holes", "female", dict(seed=5), dict(remesh="fine", holes=3, seed=6)),
    # ---- HARD ----
    MandrelCase(
        "plate_joggle_step",
        "faceplate",
        dict(seed=6, relief="step", **_FLAT),
        note="40 deg joggle: 3 patches",
        tags=_H + ("creases",),
        expect=_BAD,
    ),
    MandrelCase(
        "plate_ridge",
        "faceplate",
        dict(seed=7, relief="ridge", **_FLAT),
        note="40 deg convex crease: 2 patches",
        tags=_H + ("creases",),
        expect=_BAD,
    ),
    MandrelCase(
        "plate_ridge_valley",
        "faceplate",
        dict(seed=8, relief="cross", **_FLAT),
        note="convex + concave creases: 4 patches",
        tags=_H + ("creases",),
        expect=_BAD,
    ),
    MandrelCase(
        "female_steep70",
        "steep",
        dict(wall_deg=70.0, seed=1),
        note="walls 70 deg in truth",
        tags=_H + ("steep",),
        expect=_BAD,
    ),
    MandrelCase(
        "female_steep85",
        "steep",
        dict(wall_deg=85.0, seed=2),
        note="walls 85 deg in truth",
        tags=_H + ("steep",),
        expect=_BAD,
    ),
    MandrelCase(
        "female_steep88_noise",
        "steep",
        dict(wall_deg=88.0, seed=3, noise=0.0002),
        note="walls 88 deg (limit)",
        tags=_H + ("steep",),
        expect=_BAD,
    ),
    MandrelCase(
        "male_decoy_flange",
        "decoy",
        dict(seed=1),
        note="wide lower flat flange is a decoy",
        tags=_H + ("decoy",),
        expect=_BAD,
    ),
    MandrelCase(
        "male_ambiguous_flange",
        "decoy",
        dict(seed=2, ambiguous=True),
        note="flange at the same height, crease only",
        tags=_H + ("decoy", "ambiguous"),
        expect={"ambiguous": True},
    ),
    MandrelCase(
        "plate_plinth_decoy",
        "faceplate",
        dict(seed=9, size=(0.8, 0.6), plinth=(1.0, 0.25)),
        note="large flat machined pad beside the plate",
        tags=_H + ("decoy",),
        expect=_BAD,
    ),
    MandrelCase(
        "plate_tilt25",
        "faceplate",
        dict(seed=10, tilt=25.0),
        note="tilt 25 deg",
        tags=_H + ("tilt",),
        expect=_BAD,
    ),
    MandrelCase(
        "plate_tilt30_noise",
        "faceplate",
        dict(seed=11, tilt=30.0, noise=0.0003),
        note="tilt 30 deg",
        tags=_H + ("tilt",),
        expect=_BAD,
    ),
    MandrelCase(
        "female_tilt25",
        "female",
        dict(seed=6, tilt=25.0),
        note="tilt 25 deg",
        tags=_H + ("tilt",),
        expect=_BAD,
    ),
    MandrelCase(
        "female_tilt30",
        "female",
        dict(seed=7, tilt=30.0, noise=0.0002),
        note="tilt 30 deg",
        tags=_H + ("tilt",),
        expect=_BAD,
    ),
    MandrelCase(
        "plate_perforated",
        "faceplate",
        dict(seed=12, resolution=0.02, perforate=3),
        note="square holes through the plate",
        tags=_H + ("holes",),
        expect=_BAD,
    ),
    MandrelCase(
        "plate_perforated_ridge",
        "faceplate",
        dict(seed=13, perforate=3, relief="ridge", **_FLAT),
        note="perforated + crease",
        tags=_H + ("holes", "creases"),
        expect=_BAD,
    ),
    MandrelCase(
        "plate_pins",
        "faceplate",
        dict(seed=14, pins=4),
        note="raised tooling pins; truth excludes them",
        tags=_H + ("pins",),
        expect=_BAD,
    ),
)

_RELIEF = {"step": _joggle_step, "ridge": _ridge, "cross": _ridge_valley}


def build_case(case: MandrelCase):
    """Deterministically build one case: ``(mesh, truth, info)``."""
    kw = dict(case.gen_kwargs)
    if case.generator == "faceplate":
        if "relief" in kw:
            key = kw.pop("relief")
            kw.update(relief=_RELIEF[key], breaks=_JOG_BREAKS[key])
        m, t, info = faceplate_eggcrate(**kw)
    elif case.generator in ("male", "female"):
        m, t, info = thin_shell(kind=case.generator, **kw)
    elif case.generator == "steep":
        m, t, info = steep_pocket(**kw)
    elif case.generator == "decoy":
        m, t, info = decoy_shell(**kw)
    else:
        raise ValueError(f"unknown generator {case.generator!r} in case {case.name!r}")
    if case.vary_kwargs:
        m, t, info2 = vary(m, t, **case.vary_kwargs)
        info = {**info, **info2}
    info.update(name=case.name, tags=list(case.tags), expect=dict(case.expect), note=case.note)
    return m, t, info


def iter_cases(names=None, tag=None):
    """Yield ``(case, mesh, truth, info)``; optionally a subset by name or tag ("hard", "easy")."""
    for c in SYNTHETIC_CASES:
        if (names is None or c.name in names) and (tag is None or tag in c.tags):
            m, t, info = build_case(c)
            yield c, m, t, info
