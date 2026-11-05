# robot-sys-optimizer — Claude Code guide

Browser-based cell-layout tool for a robotic process (scanning, sanding, gripping). It answers:
1. **Where should the robot base (or its carrier) stand?**
2. **What is the shortest end effector that still reaches every target without collision or overload?**

Full spec: `docs/PROJECT_PLAN.md`. It is the source of truth for scope and milestones. This file holds the conventions every session must follow, plus the current status. **Update this file when a milestone, decision or command changes.**

## Status

- **Current milestone:** M1 Robot kinematics (next). M0 is done on branch `m0-skeleton`.
- **Blocker for M1:** Pinocchio (`pin`) and Coal have **no Windows wheels on PyPI**. Choose conda-forge (Miniforge) or WSL before starting M1.
- **Branch policy:** one branch per milestone (`m0-skeleton`, `m1-kinematics`, …); merge to `main` when its "done when" check passes.
- **Last updated:** 2026-11-04

| Milestone | State | Done when |
| --- | --- | --- |
| M0 Skeleton | **done** 2026-11-04 | `python app.py` serves the page; `pytest` passes |
| M1 Robot kinematics | todo | IK round-trip (q → FK → IK → FK) within 1e-6 m on 10,000 samples, all branches found |
| M2 Viewer | todo | Viewer robot pose matches Pinocchio FK for the same q |
| M3 Tool Setup page | todo | Changing L stretches the collision capsule and moves the TCP correctly |
| M4 Targets | todo | 10 mm grid on the curved panel has normals within 1° of analytic |
| M5 Feasibility checker | todo | Golden-scene reachable counts match hand-verified values |
| M6 Reachability maps | todo | Heatmap peak within 50 mm of brute-force best base (flat plate) |
| M7 Placement optimizer | todo | Two-station scene → finds 2 stations, full coverage |
| M8 Tool-geometry optimizer | todo | Pocket scene → minimal L within 5 mm |
| M9 Carriers | todo | Long rail: 1-axis rail respects 300 mm standoff, fewer stations than mobile base |
| M10 Exports and report | todo | CSV, updated EE URDF, HTML summary |

## Architecture rules

- **Everything is Python.** The GUI is Viser (Three.js served from Python). No hand-written HTML/JS, no REST API, no TypeScript type generation (the plan's data-model section mentions it, but that applies only to a later custom frontend).
- `core/` and `opt/` must **never import viser** or anything in `ui/`. The math must stay usable without the viewer.
- Robots are looked up by name from `assets/robots/<name>/` (URDF + meshes + `spec.yaml` + IK solver name). **Never hard-code the UR20.**
- The end effector is **mount + parametric extension (L, θ, d) + head + TCPs**. It is never a static URDF alone.
- The optimizer has three nested layers: tool geometry (outer) → base/station placement (middle) → per-point IK (inner). The reachability map depends only on robot + tool geometry, and is cached on disk keyed by a hash of those inputs.
- Long optimizer runs go in a background thread or process. The GUI must stay responsive.
- Payload checks **refuse to run** (and say so) when the `spec.yaml` limits are missing. Never guess datasheet values.

## Frames and units (do not deviate)

- Internally: **meters and radians**. GUI shows mm and degrees, and converts at the UI boundary only.
- `world`: part/cell origin, **z up**.
- `base`: robot base link. Pose = carrier pose × mount offset.
- `flange`: `tool0` from the robot URDF.
- `tcp_i`: one per contact region. **z points out of the tool, toward the part.**
- `target_j`: one per surface point. **z is the outward surface normal.**
- A target is reached when origins coincide (within tolerance) and **`tcp.z = -target.z`**, within that target's tolerances:
  - spin about the normal: free, or a range
  - tilt cone: max angle between tool axis and -normal
  - standoff: offset along the normal (0 = contact)
- Transforms are named `T_a_b` = pose of frame b expressed in frame a (e.g. `T_flange_mount`, `T_head_tcp`).
- Target points are stored in the **part frame**.

## Cost function

All terms are normalized to roughly 0–1. The weights are shown in the GUI.

```
J = w_L·L/L_ref + w_U·N_unreached + w_S·(k−1) + w_T·D_travel/D_ref
  + w_M·mean(1/(1+μ)) + w_C·mean(max(0, 1 − c/c_ref)) + w_J·mean(s_limit) + w_P·max(τ_util)
```

L = tool length; N_unreached = unreachable points (huge weight); k = stations; D_travel = carrier travel; μ = Yoshikawa manipulability; c = clearance; s_limit = closeness to joint limits; τ_util = worst static joint-torque utilization, τ = g(q) + Jᵀ·F.

**Hard constraints:** IK exists within joint limits minus a 5° margin; no collision with a 10 mm clearance (contact faces excepted); torque utilization ≤ 80% and EE inside the payload diagram; carrier inside its travel limits, outside keep-outs, and beyond the minimum standoff.

## Tech stack

Viser (+ `yourdfpy`), Pinocchio (FK/Jacobian/RNEA), UR analytic IK (OPW later), multi-start DLS numeric IK, Coal for distance queries, trimesh (+ CoACD), shapely for the 2D standoff pre-filter, `cma` / SciPy / OR-Tools, Pydantic for the schema, pytest.

## Layout

```
app.py            Viser entry point (build_app(server) is testable without blocking)
docs/             PROJECT_PLAN.md (the spec)
scripts/          make_golden_parts.py → assets/parts/*.stl
ui/               Viser tabs: tool_setup, targets_ui, cell, optimize, results
core/             schema, golden_parts, robot, ik/{ur,opw,numeric}, collision, tool,
                  process, carrier, standoff, targets, payload
opt/              reachability, placement, tool_geometry, cost
tools/            tool library (one YAML per EE)
assets/robots/    robot registry (ur20/ in v1)
assets/tools/     EE URDFs + meshes
assets/parts/     test parts
examples/         ready-to-run project files
tests/            pytest + golden scenes
```

**Golden parts** (`core/golden_parts.py`, built procedurally so tests can compare against analytic geometry), used in every milestone's tests:
- `flat_plate`: 1 × 0.6 × 0.02 m.
- `curved_panel`: cylinder R = 1 m, axis x. Analytic helpers: `panel_outer_point` and `panel_outer_normal`.
- `open_box`: 0.6 × 0.6 × 0.5 m with a 0.45 m deep pocket (forces a long tool). It is five concatenated boxes, so it is not watertight.
- `long_rail`: 6 m (forces stations).

The STLs in `assets/parts/` are generated from these; regenerate them if the builders change.

## Environment

- Windows 10, Git Bash or PowerShell. Python **3.12** venv in `.venv/`. The system default 3.14 is too new for the robotics wheels.
- `pyproject.toml` holds the ruff and pytest config only. Dependencies are in `requirements*.txt`. Packages (`core`, `opt`, `ui`) are imported from the repo root, not installed.
- Pinocchio and Coal are not pip-installable on Windows (see Status). Decide on the environment at the start of M1 and record it here.

## Commands

```
py -3.12 -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt   # setup
.venv/Scripts/python app.py [--port 8080]       # serve the GUI → http://localhost:8080
.venv/Scripts/python -m pytest -q               # tests
.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .   # lint + format
.venv/Scripts/python scripts/make_golden_parts.py   # regenerate assets/parts/*.stl
```

## Working conventions

- Write tests first for M1, M5 and M6. Frame and sign errors are the most common bugs.
- Add a debug view per math layer (frames, voxels, candidate bases).
- Keep files under 500 lines. Validate input at boundaries (URDF import, project JSON, CSV import).
- No `Co-Authored-By` trailers on commits.

## Decisions log

- 2026-11-04: Viser chosen over a custom Three.js frontend (plan § Architecture).
- 2026-11-04: Repo layout follows the plan (packages at the repo root, not `src/`). The plan moved to `docs/PROJECT_PLAN.md`.
- 2026-11-04: No TypeScript type generation and no custom WebSocket layer. Viser handles UI transport.
- 2026-11-04: Python 3.12 venv.
- 2026-11-04: Ruflo is used lightly (memory and routing) for M0–M2. Revisit agent swarms at M5+.
- 2026-11-04: Git: commit locally; **do not push** to `origin` unless the user asks.
- 2026-11-04: The user approved fetching the UR20 description from Universal Robots' ROS 2 repo (BSD-3) in M1. Convert xacro → URDF with the pip `xacro` package.
- 2026-11-04: The schema rejects unknown keys (`extra="forbid"`). Target normals are auto-normalized; zero normals are rejected. `Robot` takes a registry `name` or a raw `urdf`.

## Open decisions (from plan; decide before M5)

- [ ] Viser picking + region growing good enough on real meshes, or PyVista for selection? (prototype in M4)
- [ ] Tool geometry v1: straight only, or straight + one bend?
- [ ] Single tool length for all stations, or a tool swap per station?
- [ ] Gantry: station mode only in v1, or coordinated mode too?
- [ ] Designer output: envelope sketch, URDF, or STEP stub?
