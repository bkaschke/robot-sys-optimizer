"""Project panel: project folder on disk, save/load, download/upload of project.json."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import viser
from pydantic import ValidationError

from core import schema
from ui.session import REPO, Session, load_project, save_project

MAX_JSON = 20 * 1024 * 1024


def _notify(event, title: str, body: str, ok: bool = False) -> None:
    if getattr(event, "client", None) is not None:
        event.client.add_notification(title, body, color="green" if ok else "red", auto_close_seconds=6)


def _project_dir(text: str) -> Path:
    p = Path(text.strip()).expanduser()
    return p if p.is_absolute() else (REPO / p)


class ProjectPanel:
    def __init__(self, server: viser.ViserServer, session: Session, on_loaded: Callable[[], None]):
        self.s = session
        self.on_loaded = on_loaded
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

        @self.save.on_click
        def _(e) -> None:
            try:
                path = save_project(self.s, _project_dir(self.folder.value))
            except (OSError, ValidationError) as err:
                return _notify(e, "Save failed", str(err))
            _notify(e, "Saved", str(path), ok=True)

        @self.load.on_click
        def _(e) -> None:
            self._load(e, _project_dir(self.folder.value))

        @self.download.on_click
        def _(e) -> None:
            data = json.dumps(self.s.to_project().model_dump(mode="json"), indent=2).encode()
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

    def _load(self, e, project_dir: Path) -> None:
        try:
            load_project(self.s, project_dir)
        except (OSError, ValueError, ValidationError) as err:  # session untouched on failure
            return _notify(e, "Load failed", str(err)[:500])
        self.on_loaded()
        _notify(e, "Loaded", str(project_dir / "project.json"), ok=True)
