# robot-sys-optimizer — Claude Code guide

Browser-based cell-layout tool for a robotic process (scanning, sanding, gripping). It answers:
1. **Where should the robot base (or its carrier) stand?**
2. **What is the shortest end effector that still reaches every target without collision?**

Full spec: `docs/PROJECT_PLAN.md` (scope and the v1 milestones). This file holds the conventions
every session must follow and the current state. **How each part works and why it was built that
way is in `docs/HISTORY.md`, by release (M0-M10, v2 ... v13): read the section of the area you change
before changing it.** The user guide's chapter 10 (*For software engineers*) is the architecture
overview. **Update this file when a convention, command or the current state changes; add the
per-task notes to `docs/HISTORY.md`.**

## Current state

- **Branches:** v1 is on `main`; every later round has its own branch (`v2` ... `v13`, each from the
  previous one), pushed only when the user says so and not merged into `main` (the user's choice).
  The plan and status of a round are in `docs/V<n>_PLAN.md`; v12 (`docs/V12_PLAN.md`) is the
  follow-up of the code review `docs/V11_REVIEW.md`; v13 (`docs/V13_PLAN.md`) its leftovers and the
  next feedback round.
- **Releases:** portable zips (`scripts/make_portable.py`) go to GitHub **draft** releases, only at
  the user's request.
- **Still open with the user:** their own part mesh in the browser (Viser picking vs PyVista);
  features not tried in a real browser (each round's plan lists them in its Status table).
- **Speed facts:** IK in bulk = `solve_batch` (all branches, a few us per pose); the exact checker
  (`core/feasibility.py`) is the cost of planning (collision checks); reach maps are cached on disk
  (`cache/reachability/`); robot textures are downscaled before viser sends them
  (`ui.robot_view.light_urdf`, v13: PNG encoding of the 2048 px UR atlases was most of the app start).
- **Our line on v13 (branch `rebase/v13`, 2026-11-18):** the v8 review fixes, the batch queue,
  rendering, Windows exe / installer / release workflow (GitHub and GitLab), tool-surface detection,
  STEP parts and the v9 usability round, rebased onto `v13` (51 commits, not pushed). How each was
  ported: `docs/HISTORY.md`, "Our line rebased onto v13".
- **Batch (one system since 2026-11-19, branch `feature/batch-merge`):** the queue (`core/batch.py`,
  `batch.json`; jobs from the current part + selection, part files with the tool-surface rule, or
  v13 part packages) turns each job into a part package (`ui.batch_materialize.make_package`), and
  v13's runner plans them (`ui/batch_run.py`: one spawned process per part driving a browserless
  app, `ui/batch_part.py`; several at once). Every job uses the cell's settings (`_setup/`, the
  open project at Run); per job only *Quick plan*. Cancel sends unfinished jobs back to pending.
  GUI: Optimize -> Several parts (batch) (`ui/batch_ui.py` + `ui/batch_queue_ui.py`), Results ->
  Batch results (`<batch>/results`); CLI: `scripts/batch_run.py` (a batch folder, or project +
  parts folder). Keep v13's runner files close to upstream (small hooks only: part extras,
  `on_row`) so later upstream rounds rebase easily.

## Working notes for this repo

- In Git Bash or PowerShell, heredocs and here-strings mangle `\n` escapes, quotes and non-ASCII
  (`°`, `±`, `×`): write patch scripts with the Write tool into the scratchpad and run them with
  `$PY -I script.py <repo>`. PowerShell `Get-Content` / `Set-Content` round-trips break UTF-8 (they
  turned `×` into mojibake in v12): read and write files with Python (`encoding="utf-8"`).
- Ruff reformats long calls, so string-replace patches must match the *formatted* text; read the
  file first. Imports are sorted by ruff (`I`): `ruff check --fix --select I` after adding one.
- Files stay under 500 lines: put new code in new modules and leave one-line hooks.
- Tests never write to the user's `workspace/` (`tests/conftest.py` sets `RSO_WORKSPACE`); each new
  test file that starts an app takes its own viser port (v12 used 8983-8985).
- Memory is tight on this PC: run tests in batches (`scripts/test_batches.py`), the full fast suite
  once at the end of a round.
- Viser runs widget callbacks on a thread pool, also for values set from code: guard callbacks
  (`_busy`) and expect them late. Background writers of the session take `ui.session_lock.LOCK`
  (v12, guide 10.4a).
- Plans are keyed by `Solution.key` (`ui/plan_ids.py`), never by their label (v12).
- Collision settings are one `core.collision_config.CollisionConfig` (v12): change them with
  `collision_config.update(...)`, never with module globals; jobs carry `current()`.
- Commits: local, one per task, no `Co-Authored-By` trailer; push / merge / release only when the
  user asks. Leave `examples/mobile_test/` and `v1_feedback.txt` untracked.
- Ruflo memory: keys `robot-sys-optimizer-*` in namespace `patterns` hold per-milestone lessons.

## Architecture rules

- **Everything is Python.** The GUI is Viser (Three.js served from Python). No hand-written HTML/JS, no REST API, no TypeScript type generation (the plan's data-model section mentions it, but that applies only to a later custom frontend).
- `core/` and `opt/` must **never import viser** or anything in `ui/`. The math must stay usable without the viewer. `tests/test_v12_layers.py` enforces it.
- Robots are looked up by name from `assets/robots/<name>/` (URDF + meshes + `spec.yaml` + IK solver name). **Never hard-code the UR20.**
- The end effector is **mount + parametric extension (L, θ, d) + head + TCPs**. It is never a static URDF alone.
- The optimizer has three nested layers: tool geometry (outer) → base/station placement (middle) → per-point IK (inner). The reachability map depends only on robot + tool geometry, and is cached on disk (`data_paths.cache_dir('reachability')`, read-through to the bundled maps) keyed by a hash of those inputs.
- Long optimizer runs go in a background thread or process. The GUI must stay responsive.
- **Robot-agnostic (v7):** no payload, torque, mass or tip-over checks (removed at the user's request). Code never asks the IK solver for UR internals: use the solver interface in `core/ik/__init__.py` (`solve`, `solve_batch`, `manipulability`, `config_label`, `n_branches`, `batch`, `base_height`).

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
  + w_M·mean(1/(1+μ)) + w_C·mean(max(0, 1 − c/c_ref)) + w_J·mean(s_limit)
```

L = tool length; N_unreached = unreachable points (huge weight); k = stations; D_travel = carrier travel; μ = Yoshikawa manipulability; c = clearance; s_limit = closeness to joint limits. (Plus standoff, group, wrist-flip and motion terms, see v5/v6 notes; the torque term was removed in v7.)

**Hard constraints:** IK exists within joint limits minus a 5° margin; no collision with a 10 mm clearance (contact faces excepted); carrier inside its travel limits, outside keep-outs, and beyond the minimum standoff.

## Tech stack

Viser (+ `yourdfpy`), Pinocchio (FK/Jacobian), UR analytic IK, OPW analytic IK (v7), multi-start DLS numeric IK, Coal for distance queries, trimesh (+ CoACD), shapely for the 2D standoff pre-filter, `cma` / SciPy / OR-Tools, Pydantic for the schema, pytest.

## Layout

The v1 layout below still holds for the top level; `core/`, `opt/` and `ui/` have grown to about 175 modules since (one per feature; the user guide's section 10.2 lists the packages, `docs/HISTORY.md` each module's release).

```
app.py            Viser entry point (build_app(server, dark=...) is testable without blocking)
README.md         landing page -> docs/USER_GUIDE.md (user guide)
CHANGELOG.md      user-facing changes per development round since main (newest first); add to it with each round
packaging/        Windows exe: rso.spec (PyInstaller onedir), launch.py (entry point + --selftest + offline guard),
                  RobotCellOptimizer.iss (Inno Setup, per-user), ci_serve_check.ps1, README.md
.github/workflows tests.yml (lint on Linux; guide check + fast tests on Windows + ubuntu-22.04), release.yml (tag -> exe, zip, setup, draft release)
.gitlab-ci.yml    GitLab twin of both workflows: variables (runner tags, images) + workflow rules; includes .gitlab/ci/setup.yml
                  (conda env, caches), tests.yml, release.yml. Change a check on one host, change it on the other.
environment.yml   conda env `rso` (all dependencies)
docs/             PROJECT_PLAN.md (the spec)
scripts/          run_app.bat (Windows launcher); make_examples.py → examples/{rail_long_part,mobile_amr}/project.json;
                  make_golden_parts.py → assets/parts/*.stl; import_ur_description.py;
                  make_sample_tools.py → assets/tools/{orbital_sander,scanner} + tools/*.yaml;
                  build_exe.py (PyInstaller build, selftest, zip, installer, SHA256SUMS); tool_surface_eval.py (detector
                  accuracy on the synthetic set or a folder of labelled samples); run_app.sh (Linux)
ui/               session (state ↔ project file), widgets (PoseWidgets, notify), cell, robot_view,
                  tool_setup, tool_view, targets_ui, optimize (feasibility run/inspect),
                  placement (heatmap), stations (planner panel), tool_opt (tool-geometry panel),
                  results (Results tab), carrier_ui + carrier_bodies (External Axis tab), target_tools_ui
                  (brush / single points / tessellate), project_panel (+ Examples)
                  batch_ui + batch_queue_ui (Optimizer → Several parts (batch) panel: run / queue edits), batch_job (queue runner
                  over v13's batch_run + CLI `python -m ui.batch_job`, no viser), batch_materialize (job → part package, auto tool
                  surface rule, review pictures),
                  report_make (report / CSV / userframes without the GUI), report_text (viser-free texts), report_views (framed Iso/Top
                  3D pictures for the report), view_controls (Lighting, Shadows, Fit view, Top/Front/Side/Iso), display_lod (simplified
                  display copy above 300k triangles, overlay lift, mesh cache), colour_mesh (vertex-coloured GLB overlays),
                  targets_draw (Targets scene drawing, updated in place), tool_surface_ui (Targets → Detect tool surface),
                  preflight_run (setup check of a session, CLI) + preflight_ui (Optimize -> Setup check), handover (Everything (zip)),
                  app_state (Settings & help folder + project hooks, support bundle button), plan_assets (asset stamps for the worker), sim_status (playback status text)
core/             schema, golden_parts, robot, ik/{ur,opw,numeric}, transforms, urdf_import,
                  tool (parametric EE + library I/O), process (presets), mesh_select (pick + region grow),
                  targets (selection, normals, generators, CSV, target-set edits), target_tools (lattice,
                  brush dab, dedupe, refine_mesh),
                  collision (Coal via Pinocchio), feasibility (per-point check),
                  carrier (rail/gantry/mobile from primitives, URDF), standoff (shapely zones),
                  data_paths (resources() read-only vs data_dir()/workspace_dir()/cache_dir() writable; frozen-aware),
                  batch (batch.json manifest, status, summary; no viser), target_gen (GenSettings, generate_targets without the GUI),
                  step_import (STEP → mesh + CAD face id per triangle, OCP, cached), tool_surface (smooth-region segmentation),
                  tool_surface_features (per-region features, ray visibility), tool_surface_pick (scoring, pick-and-grow, confidence,
                  DEFAULT_PARAMS), tool_surface_eval (IoU metrics, labelled-sample loader, baselines), mandrels / mandrel_cases /
                  mandrels_hard (synthetic mandrels with ground truth; EASY and HARD sets), safe_io (validate, then replace),
                  preflight (setup-check findings), app_log (log file),
                  support_bundle (zip for problem reports, no meshes)
opt/              reachability (map, offset map, lookup, voting, coverage matrix), placement (stations),
                  sites (floor / carrier search spaces), tool_geometry (L bisection + shape CMA-ES), cost
tools/            tool library (one YAML per EE)
assets/robots/    robot registry (ur20/ in v1)
assets/tools/     EE URDFs + meshes
assets/parts/     test parts
examples/         ready-to-run project files
tests/            pytest + golden scenes
cache/            reachability maps (*.npz, gitignored, keyed by robot+TCP+settings hash); source checkout only, see core/data_paths
```

**Golden parts** (`core/golden_parts.py`, built procedurally so tests can compare against analytic geometry), used in every milestone's tests:
- `flat_plate`: 1 × 0.6 × 0.02 m.
- `curved_panel`: cylinder R = 1 m, axis x. Analytic helpers: `panel_outer_point` and `panel_outer_normal`.
- `open_box`: 0.6 × 0.6 × 0.5 m with a 0.45 m deep pocket (forces a long tool). It is five concatenated boxes, so it is not watertight.
- `long_rail`: 6 m (forces stations).
- `deep_hole`: 0.6 m block with a 0.1 m square hole, 0.3 m deep. The hole is narrower than the UR20 wrist_3 (r 55.7 mm), so a straight probe has the analytic L_min = depth − standoff + clearance (M8 done-when).

The STLs in `assets/parts/` are generated from these; regenerate them if the builders change.

## Environment

- Windows 10/11, Git Bash or PowerShell (Linux is tested in CI). Miniforge/conda may not be on PATH: call its `conda.exe` by path. The conda env **`rso`** (Python 3.12) is created from `environment.yml`, the single source of dependencies. **`$PY` below = that env's `python.exe`** (e.g. `<miniforge>/envs/rso/python.exe`; find it with `conda env list`). Builds and tests must use this env, not a venv.
- Pinocchio and Coal come from conda-forge; there are no Windows wheels on PyPI. Pip-only packages (viser, yourdfpy, pydantic, trimesh, xacro) go in the `pip:` section of `environment.yml`.
- **BLAS must be OpenBLAS.** The conda-forge MKL build kills the process (0xc06d007f) on any numpy matmul on this machine. `environment.yml` pins `libblas=*=*openblas`.
- The old `.venv/` from M0 is obsolete (gitignored); do not use it.
- **Portable bundle** (user's offline PC, 2026-11-05): `scripts/make_portable.py` = `conda-pack -p sys.prefix` (conda-pack lives in the base env; `-n` fails because conda is not on PATH) + `git archive HEAD`, merged into one zip under `robot-sys-optimizer/` with `env/` and a root `Start Robot Optimizer.bat`. `run_app.bat` prefers `..\env\python.exe` and runs `env\Scripts\conda-unpack.exe` once (marker `env\.unpacked`). `.gitattributes` forces CRLF for `*.bat` (cmd `goto` labels misbehave with LF).
- `pyproject.toml` holds the ruff and pytest config only. Packages (`core`, `opt`, `ui`) are imported from the repo root, not installed.

## Commands

Set `PY` and `RUFF` to your `rso` env (Git Bash style paths; in PowerShell use backslashes):

```
PY=<rso env>/python.exe
RUFF=<rso env>/Library/bin/ruff.exe
<miniforge>/Scripts/conda.exe env update -n rso -f environment.yml --prune   # setup / sync (create: env create)
$PY app.py [--port 8080]                  # serve the GUI → http://localhost:8080
$PY -m pytest -q                          # all tests in one process (~13 min; can run out of memory)
$PY scripts/test_batches.py               # same tests, 4 files per process (low memory, ~60 min)
$PY scripts/test_batches.py --fast        # skip the tests marked slow (plans, tool optimizer, paths; ~14 min, v13)
                                          # --jobs N: N batches at once (CI uses 3; keep 1 on this PC)
# tests/conftest.py points RSO_WORKSPACE at a temporary folder: tests never write the user's workspace/
$RUFF check . && $RUFF format --check .   # lint + format
$PY scripts/make_portable.py              # dist/*-portable-<sha>.zip: last commit + conda-pack'd env (offline PCs; guide 2.6; the installer is the recommended route, guide 2.1)
$PY scripts/build_exe.py --zip            # PyInstaller onedir + selftest + zip (--maps --installer for a release; packaging/README.md;
                                          # --map-jobs N builds N maps at once, CI uses 4)
$PY scripts/tool_surface_eval.py --synthetic --detector core.tool_surface_pick:detect   # IoU table of the detector
$PY scripts/batch_run.py <batch dir>       # run the pending jobs of a batch without the GUI (or <project> <parts> <out>)
$PY -m ui.preflight_run <project dir>     # setup check of a saved project (exit 1 on an error)
$PY scripts/make_golden_parts.py          # regenerate assets/parts/*.stl
$PY scripts/import_ur_description.py <clone> ur20   # re-import a UR model (see the script docstring)
```

## Working conventions

- Write tests first for M1, M5 and M6. Frame and sign errors are the most common bugs.
- Add a debug view per math layer (frames, voxels, candidate bases).
- Keep files under 500 lines. Validate input at boundaries (URDF import, project JSON, CSV import).
- No `Co-Authored-By` trailers on commits.

## Robot registry: UR20

- `assets/robots/ur20/` contains `ur20.urdf` (generated from UR's ROS 2 description with xacro, mesh paths made relative), meshes, `spec.yaml` (DH, rated 20 kg, joint torques 738/738/433/107/107/107 N·m) and both licenses.
- Payload, torque and wrist-moment checks were removed in v7 (the user wants the tool robot-agnostic); old `spec.yaml` payload fields are ignored.
- **Mesh license:** UR's "Terms and Conditions for Use of Graphical Documentation" (`meshes/LICENSE.txt`). It allows simulation and visualization use. Redistribution must include that file, and public displays should carry "© 2023 Universal Robots A/S. Use hereof is subject to Universal Robots A/S' Terms and Conditions for Use of Graphical Documentation." The URDF and xacro source are BSD-3.

## Decisions log

- 2026-11-04: Viser chosen over a custom Three.js frontend (plan § Architecture).
- 2026-11-04: Repo layout follows the plan (packages at the repo root, not `src/`). The plan moved to `docs/PROJECT_PLAN.md`.
- 2026-11-04: No TypeScript type generation and no custom WebSocket layer. Viser handles UI transport.
- 2026-11-04: Python 3.12. M1 switched from a venv to the Miniforge conda env `rso` (the user's choice over WSL), with OpenBLAS instead of MKL.
- 2026-11-04: Ruflo is used lightly (memory and routing) for M0–M2. Revisit agent swarms at M5+.
- 2026-11-04: Git: commit locally; **do not push** to `origin` unless the user asks.
- 2026-11-04: The user approved fetching the UR20 description from Universal Robots' ROS 2 repo (BSD-3) in M1. Convert xacro → URDF with the pip `xacro` package.
- 2026-11-04: The schema rejects unknown keys (`extra="forbid"`). Target normals are auto-normalized; zero normals are rejected. `Robot` takes a registry `name` or a raw `urdf`.
- 2026-11-04 (M9): Static mode = one base (plan semantics). The done-when uses the wall-side face scene (see Carriers). Added `shapely`. The part footprint is a convex hull.
- 2026-11-04 (M8): Straight tools reuse one flange map via `OffsetMap`. The L search is a coarse scan plus bisection (not CMA-ES on L), and shape tuning keeps the stations fixed.
- 2026-11-04 (M7): Placement alternatives are the best solution for each station count (k_min..max) rather than near-duplicate no-good cuts. Added `ortools` and `cma` (pip) to environment.yml.
- 2026-11-04 (M6): Base voting scores coverage plus the *worst* target's manipulability (maximin). Map defaults: 50 mm, 600 directions, 4 spins.
- 2026-11-04 (M5): The feasibility check takes the first feasible pose in tilt order rather than optimizing per point, and the marginal clearance band is margin + 5 mm (a contact tool's body always sits close to the part).
- 2026-11-04 (M4): Added a separate Targets tab. Click picking plus region growing in Viser works on the curved test panel. Whether it works on real part meshes stays open until the user supplies one; PyVista remains the fallback.
- 2026-11-04 (M3): The extension model is fixed as Rz(roll)·Tz(L)·Rx(θ)·Ty(d) with capsule collision. This answers the plan's open question "straight or one bend": both are supported, and `type` selects which.
- 2026-11-04 (M2): `Project.end_effector` and `Project.part` are optional until the Tool Setup (M3) and part steps fill them in. Projects are saved as `<folder>/project.json`. The UI shows UR's required copyright notice.

- 2026-11-05 (post-v1): The user asked for a GUI carrier builder (shapes + STL, own tab), mount on top of a body with gizmo fine-tuning, rail + mobile examples, then single points, a row/column brush and mesh tessellation; pushed on the new branch `external-axis-builder`. The "(M7–M8) not built yet" placeholder in the Optimize tab was stale and has been removed (on main).

## Open decisions (from plan; decide before M5)

- [ ] Viser picking + region growing good enough on real meshes, or PyVista for selection? It works on the curved test panel (M4); confirm on a real part mesh.
- [x] Tool geometry v1: straight + one bend (with lateral offset d), selectable per tool (decided in M3).
- [ ] Single tool length for all stations, or a tool swap per station?
- [ ] Gantry: station mode only in v1, or coordinated mode too?
- [x] Designer output: URDF + CSV + HTML with a dimensioned SVG envelope; STEP deferred (user, 2026-11-04).

