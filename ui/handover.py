"""Handover package: every export of one plan in a single zip, with a README (no viser).

``build`` runs the export steps it is given (each returns the file it wrote), zips the files
that were made and lists in ``README.txt`` what is inside, what could not be made and which plan
and program version it came from. One failed export never stops the others.
"""

from __future__ import annotations

import io
import logging
import zipfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

ABOUT = {  # file name (or suffix) -> what it is, for the README
    "tcp_poses.csv": "one row per point: station, TCP pose (world and robot base), joints. "
    "Lines starting with # are comments.",
    "userframes.csv": "the part frame in the robot base frame at every stop (teach the points in it).",
    "report.html": "the plan report; open it in a browser (it needs no internet) or print it to PDF.",
    "report.pdf": "the report as PDF.",
    "paths.csv": "the built paths, sample by sample (a preview of the motion, not a robot program).",
    "project.json": "the study itself; open it with Project: Open project.json (part and tool meshes "
    "are not in this zip).",
    "_optimized.zip": "the tool as a URDF with its meshes.",
    ".step": "the tool extension and flange plate for CAD.",
    "rso_path": "the robot program of the built paths (URScript .script, FANUC TP .ls or ABB RAPID .mod;"
    " Results -> Export -> Program format). Check it on the controller before running it.",
}


def about(name: str) -> str:
    hit = (text for key, text in ABOUT.items() if name == key or name.endswith(key) or name.startswith(key))
    return next(hit, "")


def readme(title: str, made: list[str], failed: list[tuple[str, str]], lines: list[str]) -> str:
    out = [title, "=" * len(title), "", *lines, "", "Files", "-----"]
    out += [f"{name}: {about(name)}".rstrip(": ") for name in made]
    if failed:
        out += ["", "Not included", "------------"]
        out += [f"{what}: {why}" for what, why in failed]
    return "\n".join(out) + "\n"


def build(
    out_dir: Path,
    stem: str,
    steps: list[tuple[str, Callable[[], tuple[Path, bytes]]]],
    lines: list[str] | None = None,
    title: str = "Robot cell plan: handover package",
) -> tuple[Path, bytes, list[tuple[str, str]]]:
    """Run ``steps`` = [(what, fn -> (path, bytes))] and zip their files into
    ``out_dir/<stem>_handover.zip``. Returns (zip path, zip bytes, [(what, why it failed)]).
    Raises ValueError when no step produced a file."""
    out_dir = Path(out_dir)
    made: list[tuple[str, bytes]] = []
    failed: list[tuple[str, str]] = []
    for what, fn in steps:
        try:
            path, data = fn()
            made.append((Path(path).name, data))
        except Exception as err:  # noqa: BLE001 - the other files are still worth handing over
            log.warning("handover: %s not made: %s", what, err)
            first = str(err).splitlines()[0] if str(err) else type(err).__name__
            failed.append((what, first))
    if not made:
        raise ValueError("nothing could be exported: " + "; ".join(f"{w} ({why})" for w, why in failed))
    stamp = f"Made {datetime.now():%Y-%m-%d %H:%M}."
    text = readme(title, [n for n, _ in made], failed, [*(lines or []), stamp])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", text)
        for name, data in made:
            z.writestr(name, data)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem) or "plan"
    zpath = out_dir / f"{safe}_handover.zip"
    zpath.write_bytes(buf.getvalue())
    return zpath, buf.getvalue(), failed


def plan_lines(session, sol, stale=None) -> list[str]:
    """The README's header lines for a plan of a session."""
    from core.version import app_version

    total = len(session.targets.points)
    rows = [
        f"Plan: {sol.label}",
        f"Stations: {len(sol.stations)}; points reached: {total - len(sol.unreached)} of {total}.",
        f"Robot: {session.robot.spec.name if session.robot else '-'}; "
        f"tool: {session.tool.name if session.tool else '-'}.",
        f"Program version: {app_version()}.",
    ]
    if stale:
        rows.insert(0, f"OUT OF DATE: the setup changed after this plan was made ({', '.join(stale)}).")
    if sol.unreached:
        rows.append("Not every point is reached: see 'Unreached points' in the report.")
    rows.append("The report's 'Not checked' section lists what this plan does not guarantee.")
    return rows


def project_step(session, out_dir: Path) -> tuple[Path, bytes]:
    """The study as ``export/project.json`` (a copy: the project's own file is not touched)."""
    import json

    data = (json.dumps(session.snapshot().model_dump(mode="json"), indent=2) + "\n").encode("utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / "project.json"
    path.write_bytes(data)
    return path, data
