"""Run the test suite in batches, one pytest process per batch (low peak memory).

    python scripts/test_batches.py [--size 4] [pytest args...]

The suite starts a full app in many test files; in one process its memory keeps growing,
which the development PC could not always afford. Each batch is a fresh process, so memory
is returned between batches. Prints one line per batch and a total; exit code 1 if any
batch failed.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    args = sys.argv[1:]
    size, first = 4, ""
    while args[:1] in (["--size"], ["--from"]):
        if args[0] == "--size":
            size = int(args[1])
        else:
            first = args[1]  # resume: start at this test file (e.g. test_tool.py)
        args = args[2:]
    files = sorted(str(p.relative_to(ROOT).as_posix()) for p in (ROOT / "tests").glob("test_*.py"))
    files = [f for f in files if Path(f).name >= first]
    batches = [files[i : i + size] for i in range(0, len(files), size)]
    total = {"passed": 0, "failed": 0, "error": 0}
    failed_batches = []
    t0 = time.perf_counter()
    for k, batch in enumerate(batches, 1):
        t = time.perf_counter()
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:warnings",
                "-p",
                "no:cacheprovider",
                *args,
                *batch,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        out = r.stdout + r.stderr
        counts = {key: int(n) for n, key in re.findall(r"(\d+) (passed|failed|error)", out)}
        for key in total:
            total[key] += counts.get(key, 0)
        status = "ok" if r.returncode == 0 else "FAILED"
        print(
            f"[{k}/{len(batches)}] {status} {counts} {time.perf_counter() - t:.0f} s  {' '.join(batch)}",
            flush=True,
        )
        if r.returncode != 0:
            failed_batches.append(batch)
            for line in out.splitlines():
                if line.startswith(("FAILED", "ERROR", "E ")):
                    print("    " + line, flush=True)
    print(f"TOTAL {total} in {time.perf_counter() - t0:.0f} s", flush=True)
    return 1 if failed_batches else 0


if __name__ == "__main__":
    sys.exit(main())
