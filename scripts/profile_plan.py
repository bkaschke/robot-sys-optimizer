"""Profile one station plan of a shipped example (docs/V7_PLAN.md, phase A).

    python scripts/profile_plan.py rail_long_part
    python scripts/profile_plan.py rail_long_part --record   # v8 T21: no profile, append the time

``--record`` times the plan without the profiler (cProfile slows it down) and appends a row to
docs/perf_history.csv: date, code version (core.provenance), example, seconds, results, best
cost, so speed changes show up over time.

Loads the example in a headless app, runs Plan stations under cProfile and prints the
cumulative and own-time tables. Uses a temporary workspace (never workspace/).
"""

import cProfile
import os
import pstats
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RSO_WORKSPACE", tempfile.mkdtemp())

import viser  # noqa: E402

from app import build_app  # noqa: E402


def record(name: str, seconds: float, app) -> None:
    import csv
    from datetime import datetime

    from core.provenance import git_version

    path = ROOT / "docs" / "perf_history.csv"
    new = not path.exists()
    res = app["session"].results
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "code", "example", "seconds", "results", "best_cost"])
        w.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                git_version(),
                name,
                f"{seconds:.1f}",
                len(res),
                f"{res[0].cost:.3f}" if res else "",
            ]
        )
    print(f"recorded {name}: {seconds:.1f} s -> {path.relative_to(ROOT)}")


def main(name: str, rec: bool = False) -> None:
    server = viser.ViserServer(host="127.0.0.1", port=8791, verbose=False)
    try:
        app = build_app(server)
        assert app["project"].load_example(name), f"no example {name}"
        from ui.staleness import STEP_COMPONENTS, confirm_step

        for step in STEP_COMPONENTS:  # the planner needs a confirmed setup (v7 round 2, T5)
            confirm_step(app["session"], step)
        if rec:  # v8 T21: plain timing, appended to docs/perf_history.csv
            t0 = time.time()
            app["stations"].start(block=True)
            record(name, time.time() - t0, app)
            return
        pr = cProfile.Profile()
        t0 = time.time()
        pr.enable()
        app["stations"].start(block=True)
        pr.disable()
        print(name, "plan", round(time.time() - t0, 1), "s,", len(app["session"].results), "results")
        stats = pstats.Stats(pr)
        stats.sort_stats("cumulative").print_stats(35)
        stats.sort_stats("tottime").print_stats(20)
    finally:
        server.stop()


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(names[0] if names else "rail_long_part", "--record" in sys.argv)
