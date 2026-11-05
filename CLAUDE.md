# robot-sys-optimizer — Claude Code guide

Browser-based cell-layout tool for a robotic process (scanning, sanding, gripping). It answers:
1. **Where should the robot base (or its carrier) stand?**
2. **What is the shortest end effector that still reaches every target without collision or overload?**

Full spec: `docs/PROJECT_PLAN.md`. It is the source of truth for scope and milestones. This file holds the conventions every session must follow, plus the current status. **Update this file when a milestone, decision or command changes.**

## Status

- **Current milestone:** M3 Tool Setup page (next). M0–M2 are done and merged to `main`.
- **Known perf item for M6:** UR analytic IK takes ~0.8 ms per pose in pure Python. Vectorize it (batch numpy) or JIT it before building reachability maps.
- **Branch policy:** one branch per milestone (`m0-skeleton`, `m1-kinematics`, …); merge to `main` when its "done when" check passes.
- **Last updated:** 2026-11-04

| Milestone | State | Done when |
| --- | --- | --- |
| M0 Skeleton | **done** 2026-11-04 | `python app.py` serves the page; `pytest` passes |
| M1 Robot kinematics | **done** 2026-11-04 (worst 4.9e-10 m) | IK round-trip (q → FK → IK → FK) within 1e-6 m on 10,000 samples, all branches found |
| M2 Viewer | **done** 2026-11-04 (25 random base+q, 4 links, ≤1e-6 m) | Viewer robot pose matches Pinocchio FK for the same q |
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
environment.yml   conda env `rso` (all dependencies)
docs/             PROJECT_PLAN.md (the spec)
scripts/          make_golden_parts.py → assets/parts/*.stl; import_ur_description.py
ui/               session (state ↔ project file), cell, robot_view, project_panel;
                  later: tool_setup, targets_ui, optimize, results
core/             schema, golden_parts, robot, ik/{ur,opw,numeric}, transforms, urdf_import,
                  collision, tool,
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

- Windows 10, Git Bash or PowerShell. **Miniforge** is at `C:\Users\Ben\miniforge3`, not on PATH. The conda env **`rso`** (Python 3.12) is defined in `environment.yml`, the single source of dependencies.
- Pinocchio and Coal come from conda-forge; there are no Windows wheels on PyPI. Pip-only packages (viser, yourdfpy, pydantic, trimesh, xacro) go in the `pip:` section of `environment.yml`.
- **BLAS must be OpenBLAS.** The conda-forge MKL build kills the process (0xc06d007f) on any numpy matmul on this machine. `environment.yml` pins `libblas=*=*openblas`.
- The old `.venv/` from M0 is obsolete (gitignored); do not use it.
- `pyproject.toml` holds the ruff and pytest config only. Packages (`core`, `opt`, `ui`) are imported from the repo root, not installed.

## Commands

Git Bash paths (in PowerShell use `$env:USERPROFILE\miniforge3\...`):

```
PY=/c/Users/Ben/miniforge3/envs/rso/python.exe
RUFF=/c/Users/Ben/miniforge3/envs/rso/Library/bin/ruff.exe
/c/Users/Ben/miniforge3/Scripts/conda.exe env update -n rso -f environment.yml --prune   # setup / sync
$PY app.py [--port 8080]                  # serve the GUI → http://localhost:8080
$PY -m pytest -q                          # tests (~35 s; 10k-sample IK round trip dominates)
$RUFF check . && $RUFF format --check .   # lint + format
$PY scripts/make_golden_parts.py          # regenerate assets/parts/*.stl
$PY scripts/import_ur_description.py <clone> ur20   # re-import a UR model (see the script docstring)
```

## Kinematics API (M1)

- `core.robot.Robot.from_registry("ur20")` provides `fk(q)` → T_base_flange (4×4, in `base_link`), `frame_pose(q, link)`, `jacobian(q, offset_flange)` (6×n, rows lin/ang, base-aligned, at a point fixed to the flange), `manipulability`, `gravity_torques(q)`, and `static_torques(q, payload_kg, payload_com, wrench, wrench_point)`.
- Sign convention: `wrench` is what the tool exerts **on** the environment (base-aligned), so τ = g(q) + J_comᵀ(−m·g) + Jᵀ·wrench.
- Gravity defaults to −z in `base_link` (floor mount). Pass `gravity_base` for wall or ceiling mounts.
- `core.ik.make_ik(robot)` dispatches on `spec.ik`. Every solver has `solve(T) → (k, 6)`. Analytic results are wrapped to (−π, π] with one row per branch (≤ 8). `expand_to_limits()` adds the ±2π copies a joint range allows.
- `URAnalyticIK` measures the URDF↔DH frame offsets at construction; nothing UR-specific is hard-coded beyond the DH table in `spec.yaml`. `NumericIK` is multi-start DLS on Pinocchio and works for any chain.
- The URDF rounds π/2 to 9 digits, so it differs from exact DH by about 1e-10. Joint-space comparisons near singularities therefore need about 1e-4 rad tolerance, while pose comparisons hold to about 1e-9.

## Viewer (M2)

- **State flow:** widgets → `ui.session.Session` (SI units, source of truth) → scene. A project file maps to and from `Session` via `to_project()` / `apply_project()`. Loading goes into a fresh `Session` first, so a broken project leaves the current one untouched.
- **Scene tree:** `/cell/base` is a transform gizmo at the robot `base_link` pose. Under it, `/cell/base/urdf_root` holds the ViserUrdf link frames. `/cell/part` is the part gizmo, with `/cell/part/mesh` under it.
- **Viser fires `on_update` when code assigns `.value`.** CellTab guards its callbacks with `self._busy` during refreshes. `PoseWidgets` uses `_syncing` for the gizmo↔field sync. Never round widget values for display: the next edit reads the rounded value back into the pose.
- `RobotView.link_pose_in_parent()` composes the pose from the scene handles (what the browser gets). It relies on ViserUrdf's private `_joint_frames` and `_joint_map_values`; `tests/test_app.py` breaks if a Viser upgrade changes them.
- **Assets:** project paths are relative to the project folder (default `workspace/default/`, gitignored), with the repo as fallback (golden parts). Uploads go to `<project>/assets/robots/<name>/` and `<project>/assets/parts/`.
- **URDF upload:** a zip goes through `core.urdf_import`, which guards against zip-slip and archive bombs, resolves `package://` and `file://`, and rejects xacro-only zips. An uploaded robot uses numeric IK, has unknown payload limits, uses the root link as base and the deepest leaf as flange. Continuous joints are not supported yet.
- Mount presets (floor, wall, ceiling) keep the base yaw. Any other rotation switches the mount to `custom`.
- Visual smoke test with no GPU: run the app, then `msedge --headless=new --use-angle=swiftshader --enable-unsafe-swiftshader --virtual-time-budget=40000 --screenshot=out.png http://127.0.0.1:<port>/`.

## Robot registry: UR20

- `assets/robots/ur20/` contains `ur20.urdf` (generated from UR's ROS 2 description with xacro, mesh paths made relative), meshes, `spec.yaml` (DH, rated 20 kg, joint torques 738/738/433/107/107/107 N·m) and both licenses.
- Wrist-moment limits are **null** on purpose (not in the sources we have), so `payload.is_complete()` is False and payload-diagram checks must refuse until they are filled in from the datasheet.
- **Mesh license:** UR's "Terms and Conditions for Use of Graphical Documentation" (`meshes/LICENSE.txt`). It allows simulation and visualization use. Redistribution must include that file, and public displays should carry "© 2023 Universal Robots A/S. Use hereof is subject to Universal Robots A/S' Terms and Conditions for Use of Graphical Documentation." The URDF and xacro source are BSD-3.

## Working conventions

- Write tests first for M1, M5 and M6. Frame and sign errors are the most common bugs.
- Add a debug view per math layer (frames, voxels, candidate bases).
- Keep files under 500 lines. Validate input at boundaries (URDF import, project JSON, CSV import).
- No `Co-Authored-By` trailers on commits.

## Decisions log

- 2026-11-04: Viser chosen over a custom Three.js frontend (plan § Architecture).
- 2026-11-04: Repo layout follows the plan (packages at the repo root, not `src/`). The plan moved to `docs/PROJECT_PLAN.md`.
- 2026-11-04: No TypeScript type generation and no custom WebSocket layer. Viser handles UI transport.
- 2026-11-04: Python 3.12. M1 switched from a venv to the Miniforge conda env `rso` (the user's choice over WSL), with OpenBLAS instead of MKL.
- 2026-11-04: Ruflo is used lightly (memory and routing) for M0–M2. Revisit agent swarms at M5+.
- 2026-11-04: Git: commit locally; **do not push** to `origin` unless the user asks.
- 2026-11-04: The user approved fetching the UR20 description from Universal Robots' ROS 2 repo (BSD-3) in M1. Convert xacro → URDF with the pip `xacro` package.
- 2026-11-04: The schema rejects unknown keys (`extra="forbid"`). Target normals are auto-normalized; zero normals are rejected. `Robot` takes a registry `name` or a raw `urdf`.
- 2026-11-04 (M2): `Project.end_effector` and `Project.part` are optional until the Tool Setup (M3) and part steps fill them in. Projects are saved as `<folder>/project.json`. The UI shows UR's required copyright notice.

## Open decisions (from plan; decide before M5)

- [ ] Viser picking + region growing good enough on real meshes, or PyVista for selection? (prototype in M4)
- [ ] Tool geometry v1: straight only, or straight + one bend?
- [ ] Single tool length for all stations, or a tool swap per station?
- [ ] Gantry: station mode only in v1, or coordinated mode too?
- [ ] Designer output: envelope sketch, URDF, or STEP stub?
