"""Setup check for a session or a project folder (no viser): ``core.preflight`` fed from a Session.

    python -m ui.preflight_run <project folder>

prints the findings; the exit code is 1 when there is an error, else 0.
"""

from __future__ import annotations

from pathlib import Path

from core import preflight as pf


def _shoulder(robot) -> float | None:
    try:
        from core.ik import make_ik

        return float(make_ik(robot).base_height)
    except Exception:  # noqa: BLE001 - the default shoulder height is used
        return None


def findings_for(session) -> list[pf.Finding]:
    """Every check that applies to this session, errors first."""
    s = session
    robot = s.robot
    checks = [
        ("robot", lambda: pf.check_robot(None if robot is None else robot.spec.ik)),
        ("tool", lambda: pf.check_tool(s.tool, s.targets)),
    ]
    if s.part is None:
        checks.append(
            (
                "part",
                lambda: [
                    pf.Finding("error", "no_part", "No part is loaded.", "Load one (Robot & cell tab).")
                ],
            )
        )
        return pf.run_checks(checks)
    box: dict = {}

    def mesh():
        if "m" not in box:
            box["m"] = s.part_mesh()
        return box["m"]

    T = s.part.T_world_part
    checks.append(("part", lambda: pf.check_part(mesh(), T)))
    checks.append(("targets", lambda: pf.check_targets(mesh(), s.targets)))
    checks.append(("stay-out zone", lambda: pf.check_zones(s.targets, T, s.obstacle_entries())))
    if robot is not None:
        tm = s.tool_model

        def span():
            return pf.check_span(
                s.targets,
                T,
                robot.spec.reach_m,
                tm.length() if tm is not None else 0.0,
                static=s.base.mode != "movable",
                floor_mount=s.mount == "floor",
                base_z=float(s.T_world_base[2, 3]),
                shoulder_m=_shoulder(robot),
            )

        checks.append(("reach", span))
    return pf.run_checks(checks)


SETUP = pf.BATCH_PREFIX


def batch_warnings(session, old: list[str]) -> list[str]:
    """A batch job's warnings with this session's setup findings (errors and warnings, one line
    each, prefixed ``setup: ``) replacing those of an earlier run. Never raises."""
    keep = [w for w in old if not w.startswith(SETUP)]
    try:
        found = [f for f in findings_for(session) if f.level in ("error", "warn")]
    except Exception:  # noqa: BLE001 - advice must not fail a job
        return keep
    return keep + [f"{SETUP}{f.text}" for f in found]


def check_folder(project_dir) -> list[pf.Finding]:
    """The findings of ``<project_dir>/project.json`` (raises when it cannot be loaded)."""
    from ui.session import Session, load_project

    s = Session()
    load_project(s, Path(project_dir))
    return findings_for(s)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Check a project's setup for common mistakes.")
    ap.add_argument("project_dir", type=Path)
    a = ap.parse_args(argv)
    try:
        found = check_folder(a.project_dir)
    except (OSError, ValueError) as err:
        print(f"cannot open {a.project_dir}: {str(err).splitlines()[0]}", flush=True)
        return 2
    print(pf.as_text(found).encode("ascii", "replace").decode(), flush=True)
    return 1 if pf.counts(found)["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
