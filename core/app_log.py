"""The program's log file: ``<data folder>/logs/app.log`` (rotating, 1 MB x 4 files).

``setup()`` is called once by the real app (``app.main``); tests and scripts do not call it, so
they write no file. It records Python's ``logging`` output from INFO up, every message of the
GUI's log panel (logger ``rso.panel``) and uncaught exceptions of the main thread and of other
threads. Worker processes do not write here: their errors come back to the main process and
are logged there. The file is what *Save support bundle* collects (``core.support_bundle``).
"""

from __future__ import annotations

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

from core import data_paths

FILE_NAME = "app.log"
MAX_BYTES = 1_000_000
BACKUPS = 3
PANEL_LOGGER = "rso.panel"
PANEL_LEVELS = {"error": logging.ERROR, "warn": logging.WARNING, "done": logging.INFO, "info": logging.INFO}
NOISY = ("websockets", "asyncio", "matplotlib", "numba", "PIL")

_handler: RotatingFileHandler | None = None
_hooked = False  # the exception hooks are installed once per process
logging.getLogger(PANEL_LOGGER).addHandler(logging.NullHandler())  # silent until setup() adds the file


def log_dir() -> Path:
    return data_paths.data_dir() / "logs"


def log_files() -> list[Path]:
    """The current log file and its rotated copies, newest first (those that exist)."""
    d = log_dir()
    names = [FILE_NAME, *(f"{FILE_NAME}.{i}" for i in range(1, BACKUPS + 1))]
    return [d / n for n in names if (d / n).is_file()]


def panel(msg: str, level: str = "info") -> None:
    """Mirror a log-panel message into the log file (a no-op before ``setup``)."""
    logging.getLogger(PANEL_LOGGER).log(PANEL_LEVELS.get(level, logging.INFO), "%s", msg)


def _hooks() -> None:
    global _hooked
    if _hooked:
        return
    _hooked = True
    log = logging.getLogger("rso.crash")
    prev_sys, prev_thread = sys.excepthook, threading.excepthook

    def on_error(kind, value, tb) -> None:
        if not issubclass(kind, KeyboardInterrupt):
            log.error("uncaught exception", exc_info=(kind, value, tb))
        prev_sys(kind, value, tb)

    def on_thread_error(args) -> None:
        if args.exc_type is not SystemExit:
            name = args.thread.name if args.thread else "?"
            log.error(
                "uncaught exception in thread %s",
                name,
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
        prev_thread(args)

    sys.excepthook, threading.excepthook = on_error, on_thread_error


def setup(version: str = "") -> Path | None:
    """Start logging to the file (once). Returns its path, or None when the folder cannot be
    written (the app then runs without a log file). Never raises."""
    global _handler
    if _handler is not None:
        return Path(_handler.baseFilename)
    try:
        d = log_dir()
        d.mkdir(parents=True, exist_ok=True)
        h = RotatingFileHandler(d / FILE_NAME, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8")
    except OSError:
        return None
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(h)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    for name in NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
    _handler = h
    _hooks()
    logging.getLogger("rso").info("started%s", f" (version {version})" if version else "")
    return Path(h.baseFilename)


def shutdown() -> None:
    """Stop writing the file (tests; the hooks stay, they only log)."""
    global _handler
    if _handler is not None:
        logging.getLogger().removeHandler(_handler)
        _handler.close()
        _handler = None
