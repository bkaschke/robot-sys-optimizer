"""Replace a file or folder only after the new content has been built and validated, so a
rejected upload never destroys what was there before."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path


def replace_file(dest: Path, data: bytes, validate: Callable[[Path], object] | None = None) -> bool:
    """Write ``data`` next to ``dest`` as a temp file (same suffix), call ``validate(temp)``
    (it raises to reject), then ``os.replace`` it onto ``dest``. On any failure the temp file
    is deleted, ``dest`` is untouched and the error propagates. True if a file was replaced."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=dest.parent, prefix=".upload-", suffix=dest.suffix)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        if validate is not None:
            validate(tmp)
        existed = dest.exists()
        os.replace(tmp, dest)
        return existed
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def replace_dir(dest: Path, build: Callable[[Path], object]) -> bool:
    """Build a folder with ``build(temp_dir)`` (a fresh sibling of ``dest``), then swap it in:
    the old folder is renamed aside, the new one renamed into place, the old one deleted. On
    any failure the temp folder is removed and the old ``dest`` is kept (restored if the swap
    had started). True if an existing folder was replaced."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}.new-"))
    old: Path | None = None
    try:
        build(tmp)
        existed = dest.exists()
        if existed:
            old = Path(tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}.old-"))
            old.rmdir()
            os.replace(dest, old)
        try:
            os.replace(tmp, dest)
        except BaseException:
            if old is not None:
                os.replace(old, dest)
                old = None
            raise
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
        return existed
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
