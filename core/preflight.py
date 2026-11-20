"""Setup check: common setup mistakes found before a long plan (no viser, no ui).

Each check takes plain data (mesh, poses, targets) and returns ``Finding`` rows; ``check_all``
runs them in order. A finding is advice with a level:

* ``error``: planning cannot give a useful result (no targets, no tool, an unknown TCP);
* ``warn``: probably a mistake (part 1000 times too big or small, normals pointing into the
  part, points off the surface, points inside a stay-out zone, more than one base can cover);
* ``info``: worth knowing (a large point set).

Nothing here is exact feasibility: that is the feasibility check and the planners. The checks
are cheap (the surface queries use at most ``MAX_SAMPLE`` points) and never raise: a check that
fails internally is reported as an ``info`` finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LEVELS = ("error", "warn", "info")
BATCH_PREFIX = "setup: "  # prefix of setup-check lines among a batch job's warnings
NOUNS = ("error(s)", "warning(s)", "note(s)")
ICONS = {"error": "✘", "warn": "⚠", "info": "•"}
SMALL_PART_M = 0.01  # a part under 10 mm across: metres loaded as mm?
LARGE_PART_M = 30.0  # a part over 30 m across: mm loaded as metres?
FLOOR_TOL_M = 0.001
OFF_SURFACE_M = 0.005  # a target further than this from the part surface
INWARD_DOT = -0.2  # target normal against the surface normal
DUPLICATE_M = 1e-4
MAX_SAMPLE = 3000  # targets used for the surface queries (evenly spread over the list)
MANY_POINTS = 5000
DEFAULT_SHOULDER_M = 0.3  # shoulder height when the IK solver does not give one


@dataclass
class Finding:
    level: str
    code: str
    text: str
    fix: str = ""
    ids: list[int] = field(default_factory=list)  # target point ids it is about (may be empty)

    def line(self) -> str:
        return f"{ICONS.get(self.level, '•')} {self.text}" + (f" {self.fix}" if self.fix else "")


def _ids(ids, limit: int = 8) -> str:
    ids = [int(i) for i in ids]
    head = ", ".join(str(i) for i in ids[:limit])
    return f"{head}, ..." if len(ids) > limit else head


def _arrays(targets) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = targets.points
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, int)
    P = np.array([p.p for p in pts], float)
    N = np.array([p.n for p in pts], float)
    return P, N, np.array([p.id for p in pts], int)


def _world(T, P: np.ndarray) -> np.ndarray:
    T = np.asarray(T, float)
    return P @ T[:3, :3].T + T[:3, 3]


def check_part(mesh, T_world_part) -> list[Finding]:
    """Size (a unit mistake) and position against the floor."""
    out = []
    diag = float(np.linalg.norm(mesh.extents))
    if diag < SMALL_PART_M:
        out.append(
            Finding(
                "warn",
                "part_tiny",
                f"The part is only {diag * 1000:.2f} mm across.",
                "Wrong units? A file in metres loaded as mm is 1000 times too small (Robot & cell: Part).",
            )
        )
    elif diag > LARGE_PART_M:
        out.append(
            Finding(
                "warn",
                "part_huge",
                f"The part is {diag:.0f} m across.",
                "Wrong units? A file in mm loaded as metres is 1000 times too big (Robot & cell: Part).",
            )
        )
    z_min = float(_world(T_world_part, np.asarray(mesh.vertices, float))[:, 2].min())
    if z_min < -FLOOR_TOL_M:
        out.append(
            Finding(
                "warn",
                "part_below_floor",
                f"The part reaches {-z_min * 1000:.0f} mm below the floor (z = 0).",
                "The arm and tool stay above the floor, so points down there cannot be reached: "
                "raise the part.",
            )
        )
    return out


def _sample(n: int) -> np.ndarray:
    return np.arange(n) if n <= MAX_SAMPLE else np.unique(np.linspace(0, n - 1, MAX_SAMPLE).astype(int))


def check_targets(mesh, targets) -> list[Finding]:
    """No points, points off the surface, normals pointing into the part, duplicates."""
    P, N, ids = _arrays(targets)
    if len(P) == 0:
        return [
            Finding(
                "error", "no_targets", "There are no target points.", "Generate or import some (Targets tab)."
            )
        ]
    out = []
    if len(P) > MANY_POINTS:
        out.append(
            Finding(
                "info",
                "many_points",
                f"{len(P)} target points: a full plan can take a long time.",
                "Try Quick plan first, or a wider point spacing.",
            )
        )
    pick = _sample(len(P))
    part = "" if len(pick) == len(P) else f" (of {len(pick)} points sampled)"
    import trimesh

    _, dist, tri = trimesh.proximity.closest_point(mesh, P[pick])
    off = dist > OFF_SURFACE_M
    if off.any():
        out.append(
            Finding(
                "warn",
                "off_surface",
                f"{int(off.sum())} point(s){part} are more than {OFF_SURFACE_M * 1000:.0f} mm off the part "
                f"surface (worst {float(dist.max()) * 1000:.0f} mm; ids {_ids(ids[pick][off])}).",
                "Imported with the wrong units, or made for another part?",
                [int(i) for i in ids[pick][off]],
            )
        )
    dots = np.einsum("ij,ij->i", N[pick], np.asarray(mesh.face_normals, float)[tri])
    inward = (dots < INWARD_DOT) & ~off
    if inward.any():
        out.append(
            Finding(
                "warn",
                "inward_normal",
                f"{int(inward.sum())} point(s){part} have a normal pointing into the part "
                f"(ids {_ids(ids[pick][inward])}).",
                "The tool would have to come from inside: flip them (Targets: Edit points).",
                [int(i) for i in ids[pick][inward]],
            )
        )
    from scipy.spatial import cKDTree

    pairs = cKDTree(P).query_pairs(DUPLICATE_M, output_type="ndarray")
    if len(pairs):
        same = np.einsum("ij,ij->i", N[pairs[:, 0]], N[pairs[:, 1]]) > 0.99
        dup = np.unique(pairs[same, 1])
        if len(dup):
            out.append(
                Finding(
                    "warn",
                    "duplicates",
                    f"{len(dup)} point(s) sit on top of another point (ids {_ids(ids[dup])}).",
                    "They would be worked twice: remove them (Targets: Remove point).",
                    [int(i) for i in ids[dup]],
                )
            )
    return out


def check_tool(tool, targets) -> list[Finding]:
    """A tool exists and every group's allowed TCPs are TCPs of that tool."""
    if tool is None:
        return [Finding("error", "no_tool", "No tool is set up.", "Choose or upload one (Tool tab).")]
    names = {t.name for t in tool.tcps}
    used = {p.group for p in targets.points}
    out = []
    for g in targets.groups:
        missing = [t for t in g.tcps if t not in names]
        if missing and (g.name in used or not targets.points):
            every = len(missing) == len(g.tcps)
            out.append(
                Finding(
                    "error" if every else "warn",
                    "unknown_tcp",
                    f"Group '{g.name}' allows TCP {', '.join(repr(m) for m in missing)}, which tool "
                    f"'{tool.name}' does not have"
                    + (": none of its points can be reached." if every else "."),
                    "Fix the group's TCP list (Targets: Groups & tolerances).",
                )
            )
    return out


def check_robot(ik: str | None) -> list[Finding]:
    if ik is None:
        return [Finding("error", "no_robot", "No robot is chosen.", "Choose one (Robot & cell tab).")]
    if ik == "numeric":
        return [
            Finding(
                "warn",
                "numeric_ik",
                "This robot uses numeric IK: it can be jogged and feasibility-checked, but the station "
                "planner and the tool optimizer need an analytic solver.",
                "See the guide: Adding a robot.",
            )
        ]
    return []


def check_span(
    targets, T_world_part, reach_m, tool_len_m, *, static: bool, floor_mount: bool, base_z: float, shoulder_m
) -> list[Finding]:
    """Static base only: can one base position cover the points at all? Uses the arm's reach plus
    the tool length as a generous bound (so it never flags a point that could be reached)."""
    P, _N, ids = _arrays(targets)
    if not static or reach_m is None or len(P) == 0:
        return []
    W = _world(T_world_part, P)
    r = float(reach_m) + float(tool_len_m)
    out = []
    centre = (W.min(axis=0) + W.max(axis=0)) / 2
    far = np.linalg.norm(W - centre, axis=1)
    if float(far.max()) > r:  # no sphere of radius r holds them all (the box centre is the best case)
        span = float(np.linalg.norm(W.max(axis=0) - W.min(axis=0)))
        out.append(
            Finding(
                "warn",
                "span",
                f"The points span {span:.2f} m, more than one fixed base can cover (reach {reach_m:.2f} m + "
                f"tool {tool_len_m:.2f} m).",
                "A static base will leave points unreached: use a movable carrier (External axis tab).",
            )
        )
    if floor_mount:
        top = base_z + (DEFAULT_SHOULDER_M if shoulder_m is None else float(shoulder_m)) + r
        high = W[:, 2] > top
        if high.any():
            out.append(
                Finding(
                    "warn",
                    "too_high",
                    f"{int(high.sum())} point(s) are higher than the arm can reach from the floor "
                    f"(above z = {top:.2f} m; ids {_ids(ids[high])}).",
                    "Raise the robot (a pedestal or a lift axis) or lower the part.",
                    [int(i) for i in ids[high]],
                )
            )
    return out


def check_zones(targets, T_world_part, zones) -> list[Finding]:
    """Targets inside a stay-out zone. ``zones`` = [(name, mesh in the zone frame, T_world_zone)]."""
    P, _N, ids = _arrays(targets)
    if len(P) == 0:
        return []
    W = _world(T_world_part, P)
    out = []
    for name, mesh, T in zones:
        if not mesh.is_watertight:
            continue
        inside = mesh.contains(_world(np.linalg.inv(np.asarray(T, float)), W))
        if inside.any():
            out.append(
                Finding(
                    "warn",
                    "in_stayout",
                    f"{int(inside.sum())} point(s) lie inside stay-out zone '{name}' "
                    f"(ids {_ids(ids[inside])}).",
                    "They cannot be reached: move the zone or remove the points.",
                    [int(i) for i in ids[inside]],
                )
            )
    return out


def run_checks(checks) -> list[Finding]:
    """Run ``[(name, callable)]``; a check that raises becomes an info finding, the rest still
    run. Errors first, then warnings, then info (stable within a level)."""
    out: list[Finding] = []
    for name, fn in checks:
        try:
            out += fn()
        except Exception as err:  # noqa: BLE001 - advice must never stop the user
            first = str(err).splitlines()[0] if str(err) else ""
            out.append(
                Finding(
                    "info", "check_failed", f"The {name} check could not run ({type(err).__name__}: {first})."
                )
            )
    return sorted(out, key=lambda f: LEVELS.index(f.level) if f.level in LEVELS else len(LEVELS))


def counts(findings) -> dict[str, int]:
    return {lv: sum(f.level == lv for f in findings) for lv in LEVELS}


def headline(findings) -> str:
    c = counts(findings)
    if not findings:
        return "Setup check: nothing found."
    parts = [f"{c[lv]} {name}" for lv, name in zip(LEVELS, NOUNS, strict=True) if c[lv]]
    return "Setup check: " + ", ".join(parts) + "."


def as_text(findings) -> str:
    """Plain text: the headline, then one line per finding."""
    return "\n".join([headline(findings), *(f.line() for f in findings)])
