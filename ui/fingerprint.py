"""File-content fingerprints for the setup hashes (v10 #2).

Plans are stamped with hashes of the setup (ui.staleness). Hashing a mesh by its *path* made a
save that copies an outside file into the project (ui.project_assets.bundle) look like a new part
and drop the plans. These hashes follow the file content instead; each file's hash is cached by
(path, mtime, size), so the watcher's once-a-second checks only stat the files.
"""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

_CACHE: dict[str, tuple[tuple[int, int], str]] = {}
_LOCK = threading.Lock()
SETTLE_NS = 2_000_000_000  # newer files are hashed again on every call (see file_hash)
MAX_FOLDER_FILES = 2000  # a URDF folder larger than this: only the URDF file itself counts


def file_hash(path: Path) -> str:
    """sha1 (12 hex digits) of a file's bytes, cached by path + mtime + size."""
    st = path.stat()
    sig = (st.st_mtime_ns, st.st_size)
    key = str(path)
    with _LOCK:
        hit = _CACHE.get(key)
    if hit is not None and hit[0] == sig:
        return hit[1]
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    out = h.hexdigest()[:12]
    # review R2: a file written in the last 2 s may be rewritten within the same mtime tick (coarse
    # clocks on Windows): same size + mtime, new content. Cache it only once it has settled.
    if time.time_ns() - st.st_mtime_ns > SETTLE_NS:
        with _LOCK:
            _CACHE[key] = (sig, out)
    return out


def folder_hash(folder: Path) -> str:
    """Hash of every file under ``folder`` (relative names + contents)."""
    files = sorted(f for f in folder.rglob("*") if f.is_file())
    if len(files) > MAX_FOLDER_FILES:
        raise ValueError("folder too large")
    h = hashlib.sha1()
    for f in files:
        h.update(f.relative_to(folder).as_posix().encode() + b"\0" + file_hash(f).encode())
    return h.hexdigest()[:12]


def content(s, rel: str | None, folder: bool = False) -> str | None:
    """The content hash of a project file (``folder``: its whole folder, for URDFs and their
    meshes); None stays None, a missing file hashes its name."""
    if not rel:
        return rel
    try:
        p = s.resolve(rel)
        if folder:
            try:
                return "d:" + folder_hash(p.parent)
            except ValueError:
                pass
        return "f:" + file_hash(p)
    except OSError:
        return f"missing:{rel}"
