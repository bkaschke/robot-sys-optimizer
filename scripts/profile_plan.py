"""Profile one station plan of a shipped example (docs/V7_PLAN.md, phase A).

    python scripts/profile_plan.py rail_long_part

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


def main(name: str) -> None:
    server = viser.ViserServer(host="127.0.0.1", port=8791, verbose=False)
    try:
        app = build_app(server)
        assert app["project"].load_example(name), f"no example {name}"
        from ui.staleness import STEP_COMPONENTS, confirm_step

        for step in STEP_COMPONENTS:  # the planner needs a confirmed setup (v7 round 2, T5)
            confirm_step(app["session"], step)
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
    main(sys.argv[1] if len(sys.argv) > 1 else "rail_long_part")
