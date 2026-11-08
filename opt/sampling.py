"""Coarse to fine (v4 plan item 6): the station search works on a spread-out subsample of the
targets, the winners are verified on every point.

``farthest_points`` is farthest-point sampling (deterministic: starts at the point nearest
the centroid). ``repair_cover`` adds stations (by the reachability map) for targets the
chosen set does not cover although some candidate does, so the subsample cannot hide
points that need an extra stop.
"""

from __future__ import annotations

import numpy as np


def sample_size(n: int, fraction: float, at_most: int, at_least: int) -> int:
    """How many targets the coarse search uses (all of them for small sets)."""
    if n <= at_least:
        return n
    return int(min(n, max(at_least, min(at_most, round(fraction * n)))))


def farthest_points(P: np.ndarray, m: int) -> np.ndarray:
    """Indices of m points of P spread as evenly as possible (sorted)."""
    P = np.asarray(P, float)
    n = len(P)
    if m >= n:
        return np.arange(n)
    if m <= 0:
        return np.zeros(0, int)
    first = int(np.argmin(np.linalg.norm(P - P.mean(0), axis=1)))
    chosen = [first]
    d = np.linalg.norm(P - P[first], axis=1)
    for _ in range(m - 1):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    return np.sort(np.array(chosen, int))


def repair_cover(covered: np.ndarray, mu: np.ndarray, chosen: list[int]) -> list[int]:
    """Greedily add candidates (rows of ``covered``) until every target some candidate covers
    is covered by the chosen set; ties by summed manipulability."""
    chosen = list(chosen)
    coverable = covered.any(axis=0)
    have = covered[chosen].any(axis=0) if chosen else np.zeros(covered.shape[1], bool)
    while True:
        missing = coverable & ~have
        if not missing.any():
            return chosen
        gain = covered[:, missing].sum(axis=1) + 1e-3 * (mu[:, missing] * covered[:, missing]).sum(axis=1)
        gain[chosen] = -1
        best = int(np.argmax(gain))
        if gain[best] < 1:
            return chosen
        chosen.append(best)
        have |= covered[best]
