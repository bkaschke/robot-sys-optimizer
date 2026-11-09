"""Tool geometry for pose tools (v2 phase 4): where the head sits relative to the flange.

The head pose relative to the mount is x, y, z, rx, ry, rz, each within its range (equal
ends fix it). The search is CMA-ES over the active components in normalized [0, 1]
coordinates. Every candidate is planned with the station planner's quick plan (no
refinement), and the objective is the full cost J (user decision: best overall cost:
unreached points, stations, travel, manipulability, clearance, limits and the connector
length as one weighted term). Payload is not considered (torque / payload checks off).
The best candidate is planned once more with refinement.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from core.feasibility import CancelledError
from core.schema import POSE_KEYS, ExtensionValues

SCALE = {"x": 1000.0, "y": 1000.0, "z": 1000.0}  # labels: m -> mm


def pose_label(v: ExtensionValues, keys=POSE_KEYS) -> str:
    return ", ".join(
        f"{k.split('_')[0]} {getattr(v, k) * SCALE.get(k, 1.0):.0f}{' mm' if k in SCALE else '°'}"
        for k in keys
    )


def optimize_pose(
    ev, st, samples: list, say: Callable[[str, float], None], cancelled=None, cands: list | None = None
):
    """Returns (feasible, values, solution, point results, message). ``ev`` is the tool
    geometry evaluator (opt.tool_geometry._Evaluator); ``samples`` gets one LSample per
    evaluated candidate (length = connector arc length); ``cands`` gets (values, solution,
    results, feasible) per candidate (the refined winner replaces its quick plan)."""
    cands = [] if cands is None else cands
    import cma

    from opt.tool_geometry import LSample

    x = ev.ee.extension
    n_targets = len(ev.t.ids)
    active = [k for k in POSE_KEYS if getattr(x, k)[1] - getattr(x, k)[0] > 1e-9]
    bounds = np.array([getattr(x, k) for k in active], float).reshape(-1, 2)
    v0 = x.nominal
    best: dict = {"key": (2, np.inf)}

    def values(u: np.ndarray) -> ExtensionValues:
        u = np.clip(np.asarray(u, float)[: len(active)], 0.0, 1.0)
        return v0.model_copy(
            update={k: float(lo + w * (hi - lo)) for k, (lo, hi), w in zip(active, bounds, u, strict=True)}
        )

    def evaluate(v: ExtensionValues, label: str, f: float) -> float:
        say(label, f)
        ok, sol, res = ev.plan(v, cancelled, refine=False)
        reached = n_targets - len(sol.unreached) if sol else 0
        cost = sol.cost if sol else float("inf")
        samples.append(
            LSample(ev.tm.length(v), ok, reached, n_targets, len(sol.stations) if sol else 0, cost)
        )
        key = (0 if ok else 1, cost)
        cands.append((v, sol, res, ok))
        if key < best["key"]:
            best.update(key=key, v=v, sol=sol, res=res, ok=ok)
        return cost if np.isfinite(cost) else 1e9

    try:
        evaluate(v0, "nominal pose", 0.05)
        if active and st.shape_evals > 0:
            u0 = (np.array([getattr(v0, k) for k in active]) - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
            if len(u0) == 1:  # CMA-ES needs >= 2 dims: pad with a coordinate values() ignores
                u0 = np.append(u0, 0.5)
            opts = {"bounds": [0, 1], "maxfevals": st.shape_evals, "verbose": -9, "seed": 3, "popsize": 6}
            es = cma.CMAEvolutionStrategy(np.clip(u0, 0, 1), 0.25, opts)
            k = 0
            while not es.stop():
                us = es.ask()
                fs = []
                for u in us:
                    k += 1
                    f = 0.05 + 0.8 * min(1.0, k / st.shape_evals)  # CMA-ES may finish its last batch
                    fs.append(evaluate(values(u), f"pose {k} (budget {st.shape_evals})", f))
                es.tell(us, fs)
        if best.get("sol") is not None:  # final: the winner with the exact refinement
            say("refine the best pose", 0.9)
            ok, sol, res = ev.plan(best["v"], cancelled, refine=True)
            if sol is not None and (0 if ok else 1, sol.cost) <= best["key"]:
                best.update(sol=sol, res=res, ok=ok)
                cands[:] = [c for c in cands if c[0] is not best["v"]] + [(best["v"], sol, res, ok)]
    except CancelledError:
        pass
    stopped = cancelled is not None and cancelled()
    if best.get("sol") is None:
        msg = "cancelled before any pose was planned" if stopped else "no pose could be planned"
        return False, v0, None, {}, msg
    v = best["v"]
    msg = (
        f"best pose ({'reaches every target' if best['ok'] else 'NOT every target'}): {pose_label(v)}; "
        f"cost {best['sol'].cost:.2f}" + (" (cancelled: best found so far)" if stopped else "")
    )
    sol = best["sol"].model_copy(
        update={"tool_values": v, "label": f"{pose_label(v, active)}: {best['sol'].label}"}
    )
    return best["ok"], v, sol, best["res"], msg
