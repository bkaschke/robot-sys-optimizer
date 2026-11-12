"""Project panel: project folder on disk, save/load, download/upload of project.json,
example projects, and autosave.

Autosave (on by default, every 2 min) writes the current project folder's project.json
when anything changed since the last save, including the work state (joint pose,
selection, solutions, panel settings). With ``remember=True`` (the running app, not tests)
every save, load and autosave records the folder in workspace/last_project.txt, which the
app reopens on the next start.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import viser
from pydantic import ValidationError

from core import schema
from ui.session import REPO, WORKSPACE, Session, load_project, remember_last, save_project

MAX_JSON = 20 * 1024 * 1024
EXAMPLES = REPO / "examples"
NO_EXAMPLE = "(choose)"


def list_examples() -> list[str]:
    """Example projects shipped in examples/<name>/project.json."""
    return sorted(p.parent.name for p in EXAMPLES.glob("*/project.json"))


def _notify(event, title: str, body: str, ok: bool = False) -> None:
    from ui.log_panel import log

    log(f"**{title}**: {body}", "done" if ok else "error")
    if getattr(event, "client", None) is not None:
        event.client.add_notification(title, body, color="green" if ok else "red", auto_close_seconds=6)


def _dump(project: schema.Project) -> str:
    """The exact text core.schema.save_project writes (for 'changed since last save')."""
    return json.dumps(project.model_dump(mode="json"), indent=2) + "\n"


def _example_copy(name: str) -> Path:
    """Working folder for example ``name``: workspace/examples/<name>, <name>_2, ... The
    first one that does not exist or is still an unchanged copy (same project.json as the
    template) is used; a copy you changed is never reused or replaced."""
    shipped = (EXAMPLES / name / "project.json").read_bytes()
    base = WORKSPACE / "examples" / name
    k = 1
    while True:
        d = base if k == 1 else base.with_name(f"{name}_{k}")
        f = d / "project.json"
        if not d.exists() or (f.is_file() and f.read_bytes() == shipped):
            return d
        k += 1


def _project_dir(text: str) -> Path:
    p = Path(text.strip()).expanduser()
    return p if p.is_absolute() else (REPO / p)


class ProjectPanel:
    def __init__(
        self,
        server: viser.ViserServer,
        session: Session,
        on_loaded: Callable[[], None],
        remember: bool = False,
        autosave_thread: bool = False,
    ):
        self.s = session
        self.on_loaded = on_loaded
        self.remember = remember
        # v7 round 2 (T9): the app starts on an empty cell that belongs to no folder yet;
        # autosave waits until a project is opened, created or saved (never writes the
        # empty start-up cell over workspace/default)
        self.untitled = False
        self._saved_json: str | None = None
        self._last_save = time.monotonic()
        self._lock = threading.Lock()
        with server.gui.add_folder("Project", expand_by_default=False):
            rel = (
                session.project_dir.relative_to(REPO)
                if session.project_dir.is_relative_to(REPO)
                else session.project_dir
            )
            self.folder = server.gui.add_text(
                "Folder", str(Path(rel).as_posix()), hint="relative to the repo, or absolute"
            )
            self.new_part = server.gui.add_button(
                "New part, same cell",
                hint="keeps robot, tool, carrier, stay-out zones; clears part, targets, plans, paths",
            )
            self.new_empty = server.gui.add_button(
                "New empty study", hint="nothing at all: start from scratch"
            )
            self.save = server.gui.add_button("Save", icon=None)
            self.load = server.gui.add_button("Load")
            self.download = server.gui.add_button("Download project.json")
            self.upload = server.gui.add_upload_button("Open project.json", mime_type="application/json")
            self.example = server.gui.add_dropdown("Examples", (NO_EXAMPLE, *list_examples()))
            self.open_example = server.gui.add_button("Open example")
            self.example_name = server.gui.add_text(
                "New example name", "", hint="letters, digits, - and _ (examples/<name>/)"
            )
            self.example_overwrite = server.gui.add_checkbox("Overwrite existing example", False)
            self.save_example = server.gui.add_button("Save as example")
            self.autosave = server.gui.add_checkbox(
                "Autosave", True, hint="save to the project folder when changed"
            )
            self.autosave_min = server.gui.add_number(
                "Autosave every (min)", 2.0, min=0.5, max=60.0, step=0.5
            )
            self.status = server.gui.add_markdown("")

        @self.save.on_click
        def _(e) -> None:
            try:
                path = self.save_now(_project_dir(self.folder.value))
            except (OSError, ValidationError, ValueError) as err:
                return _notify(e, "Save failed", str(err))
            _notify(e, "Saved", str(path), ok=True)

        @self.load.on_click
        def _(e) -> None:
            self._load(e, _project_dir(self.folder.value))

        @self.download.on_click
        def _(e) -> None:
            data = json.dumps(self.s.snapshot().model_dump(mode="json"), indent=2).encode()
            e.client.send_file_download("project.json", data)

        @self.upload.on_upload
        def _(e) -> None:
            f = self.upload.value
            if len(f.content) > MAX_JSON:
                return _notify(e, "Too large", "project.json over 20 MB")
            try:
                schema.Project.model_validate_json(f.content)
            except ValidationError as err:
                return _notify(e, "Invalid project", str(err)[:500])
            # assets resolve relative to the folder field, so store the upload there
            d = _project_dir(self.folder.value)
            d.mkdir(parents=True, exist_ok=True)
            (d / "project.json").write_bytes(f.content)
            self._load(e, d)

        self.new_part.on_click(lambda e: self.new_study(keep_cell=True, e=e))
        self.new_empty.on_click(lambda e: self.new_study(keep_cell=False, e=e))

        @self.open_example.on_click
        def _(e) -> None:
            self.load_example(self.example.value, e)

        @self.save_example.on_click
        def _(e) -> None:
            try:
                path = self.save_as_example(self.example_name.value, self.example_overwrite.value)
            except (OSError, ValidationError, ValueError) as err:
                return _notify(e, "Save as example failed", str(err))
            _notify(e, "Example saved", str(path), ok=True)

        if autosave_thread:
            threading.Thread(target=self._autosave_loop, daemon=True).start()

    def load_example(self, name: str, e=None) -> bool:
        """Examples are read-only templates: copy examples/<name> to a NEW working folder
        workspace/examples/<name>[_2, _3, ...] and open that. Saving and autosave write the
        copy; earlier copies (your edits) are never replaced."""
        if name not in list_examples():
            _notify(e, "No example", "Choose an example first")
            return False
        dest = _example_copy(name)
        shutil.copytree(EXAMPLES / name, dest, dirs_exist_ok=True)
        self.folder.value = self._rel(dest)  # the workspace may live outside the repo (RSO_WORKSPACE)
        return self._load(e, dest)

    def save_as_example(self, name: str, overwrite: bool = False) -> Path:
        """Write the current project to examples/<name>/ (with its project-local assets) so
        it shows up under Examples. The only way the GUI writes into examples/."""
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise ValueError("example name: 1-64 letters, digits, - or _")
        dest = EXAMPLES / name
        if (dest / "project.json").exists() and not overwrite:
            raise ValueError(f"example '{name}' exists: tick 'Overwrite existing example'")
        project = self.s.snapshot()
        dest.mkdir(parents=True, exist_ok=True)
        src_assets = Path(self.s.project_dir) / "assets"
        if src_assets.is_dir() and src_assets.resolve() != (dest / "assets").resolve():
            shutil.copytree(src_assets, dest / "assets", dirs_exist_ok=True)
        tmp = dest / "project.json.tmp"
        schema.save_project(project, tmp)
        tmp.replace(dest / "project.json")
        self.example.options = (NO_EXAMPLE, *list_examples())
        return dest / "project.json"

    def new_study(self, keep_cell: bool, e=None) -> Path:
        """Start a new study in a new folder workspace/study_<date-time>; the current one is
        saved first and stays as it is. ``keep_cell``: keep robot, tool, carrier and stay-out
        zones and clear the part, its targets, plans, checks and paths (a new part in the same
        cell). Otherwise everything starts empty, without even a robot."""
        try:
            self.save_now()  # keep the current work (not possible inside examples/: a template)
        except (OSError, ValidationError, ValueError):
            pass
        dest = WORKSPACE / f"study_{datetime.now():%Y%m%d_%H%M%S}"
        s = self.s
        if keep_cell:
            s.part, s.selection = None, None
            s.targets = s.targets.model_copy(update={"points": []})  # keep the tolerance groups
            s.results, s.result_points, s.feasibility, s.stamps = [], {}, {}, {}
            s.steps.pop("Targets", None)
        else:
            from ui.session import Session

            hooks = s.before_save
            fresh = Session()  # no robot, tool, part, carrier, targets or results
            fresh.robot_name = None  # (the dataclass default names the UR20 without loading it)
            fresh.project_dir = s.project_dir
            s.__dict__.update(fresh.__dict__)
            s.before_save = hooks
        self.on_loaded()
        path = self.save_now(dest)  # save_project carries project-local assets along
        what = "new part in the same cell" if keep_cell else "new empty study"
        self.status.content = f"_Started a {what} in `{self.folder.value}`._"
        _notify(e, "New study", f"{what}: {path.parent}", ok=True)
        return dest

    def new_cell(self, e=None) -> Path:
        """A new empty cell in a new folder workspace/study_<date-time>: the default robot on a
        static base, no part, tool, targets or plans (the start-up popup's *New empty cell*).
        The current project, if it has a folder, is saved first."""
        from ui.session import empty_cell

        if not self.untitled:
            try:
                self.save_now()
            except (OSError, ValidationError, ValueError):
                pass
        dest = WORKSPACE / f"study_{datetime.now():%Y%m%d_%H%M%S}"
        hooks = self.s.before_save
        self.s.__dict__.update(empty_cell().__dict__)
        self.s.before_save = hooks
        self.s.project_dir = dest
        self.on_loaded()
        path = self.save_now(dest)
        self.status.content = f"_Started a new empty cell in `{self.folder.value}`._"
        _notify(e, "New cell", str(path.parent), ok=True)
        return dest

    # --- saving -------------------------------------------------------------------------------

    def save_now(self, project_dir: Path | None = None) -> Path:
        target = Path(project_dir if project_dir is not None else self.s.project_dir).resolve()
        if target.is_relative_to(EXAMPLES.resolve()):
            raise ValueError("examples/ holds read-only templates: use 'Save as example' to change one")
        with self._lock:
            path = save_project(self.s, project_dir)
            self.untitled = False
            self._saved_json = path.read_text(encoding="utf-8")
            self._last_save = time.monotonic()
        if self.remember:
            remember_last(self.s.project_dir)
        self.folder.value = self._rel(self.s.project_dir)
        self.status.content = f"_Saved {datetime.now():%H:%M:%S} to `{self.folder.value}`._"
        return path

    def autosave_tick(self, force: bool = False) -> bool:
        """Save if autosave is on, the interval has passed (or ``force``), and the project
        changed since the last save. Returns True when it wrote the file."""
        if not self.autosave.value or self.untitled:
            return False
        if not force and time.monotonic() - self._last_save < self.autosave_min.value * 60:
            return False
        try:
            current = _dump(self.s.snapshot())
        except (ValidationError, ValueError):
            return False  # mid-edit state that does not validate: try again next tick
        if current == self._saved_json:
            self._last_save = time.monotonic()
            return False
        try:
            self.save_now()
        except (OSError, ValidationError, ValueError) as err:
            self.status.content = f"_Autosave failed: {err}_"
            return False
        self.status.content = f"_Autosaved {datetime.now():%H:%M:%S} to `{self.folder.value}`._"
        return True

    def _autosave_loop(self) -> None:
        while True:
            time.sleep(5.0)
            try:
                self.autosave_tick()
            except Exception as err:  # never let the background thread die silently
                self.status.content = f"_Autosave error: {err}_"

    @staticmethod
    def _rel(d: Path) -> str:
        d = Path(d)
        return d.relative_to(REPO).as_posix() if d.is_relative_to(REPO) else d.as_posix()

    # --- loading ------------------------------------------------------------------------------

    def _load(self, e, project_dir: Path) -> bool:
        d = Path(project_dir).resolve()
        if d.is_relative_to(EXAMPLES.resolve()) and d.parent == EXAMPLES.resolve():
            return self.load_example(d.name, e)  # never work inside the shipped templates
        try:
            load_project(self.s, project_dir)
        except (OSError, ValueError, ValidationError) as err:  # session untouched on failure
            _notify(e, "Load failed", str(err)[:500])
            return False
        self.untitled = False
        self.on_loaded()
        if self.s.load_warnings:  # v9 #12: opened without the files that are missing
            from ui.log_panel import log

            for w in self.s.load_warnings:
                log(f"Missing file: {w}", "warn")
            _notify(e, "Opened with missing files", "; ".join(self.s.load_warnings)[:500])
        with self._lock:
            self._saved_json = None  # compare against the next snapshot, not the old project
            self._last_save = time.monotonic()
        try:
            self._saved_json = _dump(self.s.snapshot())
        except (ValidationError, ValueError):
            pass
        if self.remember:
            remember_last(project_dir)
        self.folder.value = self._rel(project_dir)
        self.status.content = f"_Opened `{self.folder.value}` at {datetime.now():%H:%M:%S}._"
        _notify(e, "Loaded", str(project_dir / "project.json"), ok=True)
        return True
