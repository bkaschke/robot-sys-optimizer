"""Run the station optimizer on a folder of part packages without the GUI (v13 #8).

    python scripts/batch_run.py <project folder> <parts folder> <output folder>
                                [--parts a,b] [--frame-search] [--workers N]
    python scripts/batch_run.py <batch folder> [--frame-search]

The second form runs the pending jobs of a batch made in Optimize -> Several parts (a folder
with batch.json: queued part packages, part files and selections; ``ui.batch_job``).

The project folder holds the project.json whose robot, end effector, external axis, stay-outs
and planner settings every part uses (save the project in the app first). The parts folder holds
part packages made in the package builder's Part tab (one folder per part). The output folder
gets a folder per part (a project with its plans, report.pdf) and summary.pdf / .csv / .json;
open it in the app under Results -> Batch results. Same code as Optimize -> Several parts.

Prints a line per part as it goes; the exit code is 1 when a part failed, 2 for bad arguments.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    from core import batch as bt
    from core import part_package as pp

    args = list(sys.argv[1:] if argv is None else argv)
    pos = [x for x in args if not x.startswith("-")]
    if len(pos) == 1 and (Path(pos[0]) / bt.MANIFEST_NAME).is_file():
        from ui.batch_job import main as run_queue

        return run_queue(args)
    from ui.batch_run import plan_workers, run_batch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", type=Path, help="project folder (with project.json)")
    ap.add_argument("parts", type=Path, help="folder of part packages")
    ap.add_argument("out", type=Path, help="output folder")
    ap.add_argument("--parts", dest="only", default="", help="only these package folders (comma separated)")
    ap.add_argument("--frame-search", action="store_true", help="rail / gantry: search the frame per part")
    ap.add_argument(
        "--workers", type=int, default=None, help="parts at once (default: from cores and memory)"
    )
    a = ap.parse_args(args)

    if not (a.project / "project.json").is_file():
        print(f"error: no project.json in {a.project}", file=sys.stderr)
        return 2
    try:
        ok, bad = pp.scan_folder(a.parts)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    for d, why in bad:
        print(f"skipped {d.name}: {why}")
    parts = [(d.name, d) for d, _ in ok]
    if a.only:
        want = [x.strip() for x in a.only.split(",") if x.strip()]
        unknown = sorted(set(want) - {n for n, _ in parts})
        if unknown:
            print(f"error: not in the parts folder: {', '.join(unknown)}", file=sys.stderr)
            return 2
        parts = [p for p in parts if p[0] in want]
    if not parts:
        print("error: no part packages to run", file=sys.stderr)
        return 2
    workers, why = (a.workers, f"{a.workers} at once (set)") if a.workers else plan_workers(len(parts))
    print(f"{len(parts)} part(s), {why}", flush=True)
    t0 = time.monotonic()

    def progress(name: str, stage: str, frac: float) -> None:
        print(f"[{time.monotonic() - t0:7.1f} s] {name}: {stage}", flush=True)

    rows = run_batch(
        a.project,
        parts,
        a.out,
        frame_search=a.frame_search,
        workers=workers,
        on_progress=progress,
        parts_folder=str(a.parts),
    )
    for r in rows:
        if r["status"] == "done":
            reached = f"{r['reached']}/{r['points']} reached"
            print(f"{r['part']}: {reached}, {r['stops']} stop(s), cycle {r['cycle_s']} s")
        else:
            print(f"{r['part']}: {r['status']}: {r.get('reason', '')}")
    print(f"summary: {a.out / 'summary.pdf'}")
    return 0 if all(r["status"] == "done" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
