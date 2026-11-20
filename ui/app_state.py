"""Pieces of ``app.build_app`` that stand alone: the Settings & help folder and the two
project hooks that copy GUI state to and from the session."""

from __future__ import annotations

import numpy as np
import viser

from core.schema import Optimizer
from ui import settings as ui_settings
from ui.guide import add_guide_panel


def build_settings(server: viser.ViserServer, dark: bool, guide_port: int | None, session=None):
    """The "Settings & help" folder. Returns (dark checkbox, worker checkbox, guide handle)."""
    with server.gui.add_folder("Settings & help", expand_by_default=False):  # v7: out of the way
        dark_toggle = server.gui.add_checkbox("Dark mode", dark)
        from ui import jobs

        worker_box = server.gui.add_checkbox(
            "Run planners in a separate process",
            jobs.USE_WORKER,
            hint="keeps the panel responsive; Cancel always ends the run; "
            "the worker's memory is freed after 5 idle minutes",
        )

        def set_worker(_e=None) -> None:
            jobs.USE_WORKER = bool(worker_box.value)

        worker_box.on_update(set_worker)
        workers_num = server.gui.add_number(
            "Check worker processes",
            jobs.CHECK_WORKERS,
            min=1,
            max=16,
            step=1,
            hint="exact checks of the station planner in parallel (1 = off); each holds a robot + scene "
            "(~300 MB)",
        )

        def set_workers(_e=None) -> None:
            jobs.CHECK_WORKERS = int(workers_num.value)

        workers_num.on_update(set_workers)
        guide = add_guide_panel(server, guide_port)
        from core.version import app_version

        server.gui.add_markdown(f"Version: {app_version()}")
        bundle_btn = server.gui.add_button(
            "Save support bundle (zip)",
            hint="version, system info, the log and the open study's project.json (no part or tool "
            "files) in one zip to send along with a problem report; nothing is sent anywhere",
        )
        bundle_md = server.gui.add_markdown("")
        bundle_btn.on_click(lambda e: save_bundle(e, session, bundle_md))
    return dark_toggle, worker_box, guide


def save_bundle(event, session, info=None):
    """Write the support bundle (core.support_bundle), offer it as a download and say where it is.
    Returns its path, or None when it could not be written."""
    from core.support_bundle import make_bundle
    from ui import log_panel
    from ui.widgets import notify

    try:
        path, data = make_bundle(
            getattr(session, "project_dir", None) if session is not None else None, log_panel.entries()
        )
    except OSError as err:
        notify(event, "Support bundle", f"could not be written: {err}")
        return None
    if info is not None:
        info.content = f"Saved `{path}`"
    client = getattr(event, "client", None)
    if client is not None:
        client.send_file_download(path.name, data)
    notify(event, "Support bundle saved", str(path), ok=True)
    return path


def restore_work(state: dict, session) -> None:
    """Bring back where the user left off: selection, then panel settings."""
    t = state["targets"]
    sel = session.selection
    if sel and session.part is not None and sel[0] == session.part.mesh_path:
        m = t.mesh()
        faces = np.asarray(sel[1], dtype=int)
        if m is not None and (len(faces) == 0 or faces.max() < len(m.faces)):
            t.selection = faces
    # the project's station limit first; saved panel settings (if any) override it
    n = (session.extra.get("optimizer") or Optimizer()).max_stations
    for panel in (state["stations"], state["tool_opt"]):
        panel.max_stations.value = float(min(max(n, 1), 40))
    state["stations"].costs.load()  # the project's cost weights
    state["tool_opt"].costs.load()  # and the tool optimizer's
    if "frame" in state:
        state["frame"].reset_range()  # v10 #6: this project's mount height, unless saved
    state["optimize"].load_margins()  # the project's clearance / joint margins (not a GUI setting)
    ui_settings.apply(state, session.ui)
    state["collision"].sync()
    t.refresh()


def collect_work(state: dict, session) -> None:
    """before_save hook: copy GUI-side state into the session."""
    t = state["targets"]
    part = session.part
    session.selection = (part.mesh_path, [int(f) for f in t.selection]) if part is not None else None
    session.ui = ui_settings.collect(state)
    opt = session.extra.get("optimizer") or Optimizer()
    session.extra["optimizer"] = opt.model_copy(
        update={"max_stations": int(state["stations"].max_stations.value)}
    )
