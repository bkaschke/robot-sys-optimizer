"""Optimization methods the user can choose per slot (v8 T4).

Each slot lists its methods; the FIRST is the method the app used before v8 and stays the
default, so old projects plan exactly as before. ``short`` is the one-line description shown
under the dropdown (the user guide's "Optimization methods" section has the long version).
The planners look the chosen method up by name (``key``): GUI labels can change, keys don't.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Method:
    key: str
    name: str  # dropdown label
    short: str  # one line under the dropdown


CMAES = Method("cmaes", "CMA-ES", "evolves a population of candidates; robust, needs many evaluations")
TPE = Method(
    "tpe",
    "Bayesian (TPE)",
    "learns which regions give good results and samples there; good with a small budget",
)
GP = Method(
    "gp",
    "Bayesian (Gaussian process)",
    "fits a smooth model of the cost and tries the most promising point next; the most "
    "sample-efficient for 1-4 smooth variables",
)
NONE = Method("none", "None", "keeps the set-cover result without refinement (fastest)")
WEIGHTED = Method("weighted", "Weighted cost", "one number from the cost weights; returns the best plan")
PARETO = Method(
    "pareto",
    "Pareto front (NSGA-II)",
    "keeps the trade-off plans no other plan beats on every measure; you pick one",
)
ROBUST = Method(
    "robust",
    "Robust (worst case)",
    "scores each plan by its worst result under part / station tolerance instead of the nominal one",
)

METHODS: dict[str, tuple[Method, ...]] = {
    "station_refine": (CMAES, TPE, GP, NONE),
    "station_objective": (WEIGHTED, PARETO, ROBUST),
    "tool_shape": (CMAES, TPE, GP),
    "tool_start": (
        Method("nominal", "Nominal", "starts the shape search from the tool's nominal values"),
        Method(
            "cones",
            "From approach cones",
            "starts from bend angles / offsets derived from each point's collision-free approach directions",
        ),
    ),
    "tool_loop": (
        Method(
            "fixed",
            "Stations fixed after length search",
            "plans the stations once at the shortest length, then tunes the shape for them",
        ),
        Method(
            "alternate",
            "Alternate",
            "re-plans the stations after each shape step, until neither improves (slower, often better)",
        ),
    ),
    "tool_objective": (WEIGHTED, PARETO, ROBUST),
    "point_order": (
        Method("pattern", "Pattern order", "the path pattern per group (raster, serpentine, ...)"),
        Method(
            "gtsp",
            "Least joint motion (GTSP)",
            "chooses the point order and joint solution per point together to minimize joint motion "
            "and wrist flips",
        ),
    ),
}


def options(slot: str) -> tuple[str, ...]:
    return tuple(m.name for m in METHODS[slot])


def by_name(slot: str, name: str | None) -> Method:
    """The method with this label (or key); unknown names fall back to the slot's default."""
    for m in METHODS[slot]:
        if name in (m.name, m.key):
            return m
    return METHODS[slot][0]


def key_of(slot: str, name: str | None) -> str:
    return by_name(slot, name).key
