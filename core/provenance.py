"""Where a result came from (v8 T21): code version, methods, settings and seed.

Every plan and tool result gets ``Solution.provenance``: the git commit of the code (with
"+dirty" when the working tree had changes), the methods used, a short hash of all planner
settings and weights, and the seed (the planners' searches are seeded: the same settings on
the same code give the same plan). The Results card shows it; the project file keeps it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 1  # the planners' fixed seed (CMA-ES, TPE, GP, NSGA-II, tolerance samples default)


@lru_cache(maxsize=1)
def git_version() -> str:
    """Short commit hash (+dirty), or "unknown" (no git / not a checkout, e.g. the portable zip
    built from ``git archive``: then the zip name carries the commit)."""
    try:
        h = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5
        ).stdout.strip()
        if not h:
            return "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def settings_hash(*parts) -> str:
    """Short stable hash of settings objects (dataclasses, pydantic models, dicts, values)."""

    def plain(x):
        if hasattr(x, "model_dump"):
            return x.model_dump(mode="json")
        if hasattr(x, "__dataclass_fields__"):
            return {k: plain(getattr(x, k)) for k in x.__dataclass_fields__}
        if isinstance(x, dict):
            return {str(k): plain(v) for k, v in x.items()}
        if isinstance(x, list | tuple):
            return [plain(v) for v in x]
        if isinstance(x, float | int | str | bool) or x is None:
            return x
        return repr(type(x).__name__)

    blob = json.dumps([plain(p) for p in parts], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def stamp(methods: dict, *settings) -> dict[str, str]:
    return {
        "code": git_version(),
        "methods": ", ".join(f"{k}={v}" for k, v in methods.items()),
        "settings": settings_hash(*settings),
        "seed": str(SEED),
    }
