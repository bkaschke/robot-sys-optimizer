"""Alternate tool shape and stations (v8 T13, Tool optimizer -> Tool and stations: Alternate).

The default ("Stations fixed after length search") plans the stations once at the shortest
tool length and tunes the shape for those stations. Alternate then repeats: plan the stations
again for the best shape so far, tune the shape for the new stations, and so on, until a round
improves the cost by less than ``MIN_GAIN`` (1 %) or after ``ROUNDS`` rounds. The best design
seen is kept, so the result is never worse than the fixed-stations one.

Pose tools re-plan the stations for every candidate already (opt.tool_pose): nothing to do.
"""

from __future__ import annotations

from core.feasibility import CancelledError

ROUNDS = 5
MIN_GAIN = 0.01


def alternate(ev, st, best: list, active: list, values_for, say, cancelled=None) -> int:
    """``best`` = [cost, solution, results] (updated in place); ``values_for(v)`` -> the
    values(u) mapping of the shape search around tool values ``v``. Returns the rounds run."""
    from opt.tool_geometry import _shape_search

    rounds = 0
    for r in range(ROUNDS):
        if cancelled is not None and cancelled():
            break
        before = best[0]
        v = best[1].tool_values
        say(f"alternate round {r + 1}: re-plan the stations", 0.95)
        ok, sol, res = ev.plan(v, cancelled)
        if sol is None:
            break
        sol = sol.model_copy(update={"tool_values": v})
        if (len(sol.unreached), sol.cost) < (len(best[1].unreached), best[0]):
            best[:] = [sol.cost, sol, res]
        say(f"alternate round {r + 1}: tune the shape", 0.97)
        try:
            _shape_search(ev, st, v, sol, active, values_for(v), say, best)
        except CancelledError:
            break
        rounds += 1
        if before - best[0] < MIN_GAIN * abs(before):
            break
    return rounds
