"""Where the app reads bundled files and where it writes user data.

* ``resources()``: read-only root holding ``assets/``, ``docs/``, ``examples/``, ``tools/`` and the
  optional prebuilt ``cache/``. The repo folder in a source checkout; the unpack folder
  (``sys._MEIPASS``) or the executable's folder in a frozen build.
* ``data_dir()``: writable root (projects, caches, user tools). The repo folder in a source checkout,
  so nothing changes for developers and tests. Frozen: ``%LOCALAPPDATA%/RobotCellOptimizer`` on
  Windows, ``$XDG_DATA_HOME`` (or ``~/.local/share``)``/robot-cell-optimizer`` elsewhere.
  ``RSO_DATA`` overrides it in both modes; ``RSO_WORKSPACE`` overrides only the workspace.

Nothing is cached at import time, so tests can change ``sys.frozen`` and the environment.
Never write under ``resources()``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
CACHE_KINDS = ("reachability", "collision", "numba")


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resources() -> Path:
    """Read-only root with the bundled assets, docs and examples."""
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        return Path(meipass) if meipass else Path(sys.executable).resolve().parent
    return _SOURCE_ROOT


def _default_user_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "RobotCellOptimizer"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "robot-cell-optimizer"


def data_dir() -> Path:
    """Writable root; created on first use."""
    override = os.environ.get("RSO_DATA")
    root = Path(override) if override else (_default_user_dir() if is_frozen() else _SOURCE_ROOT)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise OSError(
            f"cannot create the data folder {root}: {err}. Set RSO_DATA to a writable folder."
        ) from err
    return root


def workspace_dir() -> Path:
    override = os.environ.get("RSO_WORKSPACE")
    return Path(override) if override else data_dir() / "workspace"


def cache_dir(kind: str) -> Path:
    """Writable cache folder (not created here; writers create it)."""
    if kind not in CACHE_KINDS:
        raise ValueError(f"unknown cache kind '{kind}'; use one of {CACHE_KINDS}")
    return data_dir() / "cache" / kind


def cache_read_dirs(kind: str) -> list[Path]:
    """Folders to look in for a cached file: user data first, then the bundled seed."""
    first = cache_dir(kind)
    seed = resources() / "cache" / kind
    return [first] if seed == first else [first, seed]


def find_cached(kind: str, filename: str) -> Path | None:
    """Read-through lookup: the first existing file in data, then in the bundled seed."""
    for d in cache_read_dirs(kind):
        p = d / filename
        if p.is_file():
            return p
    return None


def last_project_file() -> Path:
    return workspace_dir() / "last_project.txt"


def tools_dir() -> Path:
    """Writable tool library (``<name>.yaml``); bundled tools live in ``resources()/tools``."""
    return data_dir() / "tools"


def asset_bases() -> list[Path]:
    """Roots a relative asset path (``assets/...``) is looked up in: user data, then bundled."""
    d, r = data_dir(), resources()
    return [d] if d == r else [d, r]


def configure_numba_cache() -> None:
    """Point numba's on-disk cache at the data folder when frozen (or RSO_DATA is set).
    Call before numba is first imported. A source checkout keeps numba's default."""
    if (is_frozen() or os.environ.get("RSO_DATA")) and "NUMBA_CACHE_DIR" not in os.environ:
        d = cache_dir("numba")
        d.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(d)
