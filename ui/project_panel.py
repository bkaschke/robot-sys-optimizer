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
import shutil
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import viser
from pydantic import ValidationError

from core import schema
from ui.session import REPO, Session, load_project, remember_last, save_project

MAX_JSON = 20 * 1024 * 1024
EXAMPLES = REPO / "examples"
NO_EXAMPLE = "(choose)"


def list_examples() -> list[str]:
    """Example projects shipped in examples/<name>/project.json."""
    return sorted(p.parent.name for p in EXAMPLES.glob("*/project.json"))


def _notify(event, title: str, body: str, ok: bool = False) -> None:
    if getattr(event, "client", None) is not None:
        event.client.add_notification(title, body, color="green" if ok else "red", auto_close_seconds=6)


def _dump(project: schema.Project) -> str:
    """The exact text core.schema.save_project writes (for 'changed since last save')."""
    return json.dumps(project.model_dump(mode="json"), indent=2) + "\n"


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
            self.save = server.gui.add_button("Save", icon=None)
            self.load = server.gui.add_button("Load")
            self.download = server.gui.add_button("Download project.json")
            self.upload = server.gui.add_upload_button("Open project.json", mime_type="application/json")
            self.example = server.gui.add_dropdown("Examples", (NO_EXAMPLE, *list_examples()))
            self.open_example = server.gui.add_button("Open example")
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

        @self.open_example.on_click
        def _(e) -> None:
            self.load_example(self.example.value, e)

        if autosave_thread:
            threading.Thread(target=self._autosave_loop, daemon=True).start()

    def load_example(self, name: str, e=None) -> bool:
        """Copy examples/<name> to workspace/examples/<name> (so saving never edits the
        shipped copy) and load it. An existing workspace copy is replaced."""
        if name not in list_examples():
            _notify(e, "No example", "Choose an example first")
            return False
        dest = REPO / "workspace" / "examples" / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(EXAMPLES / name, dest)
        self.folder.value = dest.relative_to(REPO).as_posix()
        return self._load(e, dest)

    # --- saving -------------------------------------------------------------------------------

    def save_now(self, project_dir: Path | None = None) -> Path:
        with self._lock:
            path = save_project(self.s, project_dir)
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
        if not self.autosave.value:
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
        try:
            load_project(self.s, project_dir)
        except (OSError, ValueError, ValidationError) as err:  # session untouched on failure
            _notify(e, "Load failed", str(err)[:500])
            return False
        self.on_loaded()
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
