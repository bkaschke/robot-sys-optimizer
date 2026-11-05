# robot-sys-optimizer — Claude Code guide

Browser-based cell-layout tool for a robotic process (scanning, sanding, gripping). It answers:
1. **Where should the robot base (or its carrier) stand?**
2. **What is the shortest end effector that still reaches every target without collision or overload?**

Full spec: `docs/PROJECT_PLAN.md`. It is the source of truth for scope and milestones. This file holds the conventions every session must follow, plus the current status. **Update this file when a milestone, decision or command changes.**

## Status

- **Current milestone:** M8 Tool-geometry optimizer (next). M0–M7 are done and merged to `main`.
- **IK speed:** use `core.ik.ur.solve_batch` (about 7 µs per pose, all 8 branches) for anything bulk. `URAnalyticIK.solve` (about 0.8 ms) is for one-off calls.
- **Branch policy:** one branch per milestone (`m0-skeleton`, `m1-kinematics`, …); merge to `main` when its "done when" check passes.
- **Last updated:** 2026-11-04

| Milestone | State | Done when |
| --- | --- | --- |
| M0 Skeleton | **done** 2026-11-04 | `python app.py` serves the page; `pytest` passes |
| M1 Robot kinematics | **done** 2026-11-04 (worst 4.9e-10 m) | IK round-trip (q → FK → IK → FK) within 1e-6 m on 10,000 samples, all branches found |
| M2 Viewer | **done** 2026-11-04 (25 random base+q, 4 links, ≤1e-6 m) | Viewer robot pose matches Pinocchio FK for the same q |
| M3 Tool Setup page | **done** 2026-11-04 (GUI L slider → capsule span L+2r, TCP Δ = ΔL·z_mount, 1e-9) | Changing L stretches the collision capsule and moves the TCP correctly |
| M4 Targets | **done** 2026-11-04 (click-select → 10 mm grid on rotated panel, worst normal < 1°) | 10 mm grid on the curved panel has normals within 1° of analytic |
| M5 Feasibility checker | **done** 2026-11-04 (6 golden scenes in tests/test_feasibility.py match hand expectations) | Golden-scene reachable counts match hand-verified values |
| M6 Reachability maps | **done** 2026-11-04 (tilted-plate scene: map peak = brute-force peak, 0 mm) | Heatmap peak within 50 mm of brute-force best base (flat plate) |
| M7 Placement optimizer | **done** 2026-11-04 (long rail, 21 pts: 2 stations, 21/21 exact-verified; 1 station proven insufficient) | Two-station scene → finds 2 stations, full coverage |
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
scripts/          make_golden_parts.py → assets/parts/*.stl; import_ur_description.py;
                  make_sample_tools.py → assets/tools/{orbital_sander,scanner} + tools/*.yaml
ui/               session (state ↔ project file), widgets (PoseWidgets, notify), cell, robot_view,
                  tool_setup, tool_view, targets_ui, optimize (feasibility run/inspect),
                  placement (heatmap), stations (planner panel), results (Results tab), project_panel
core/             schema, golden_parts, robot, ik/{ur,opw,numeric}, transforms, urdf_import,
                  tool (parametric EE + library I/O), process (presets), mesh_select (pick + region grow),
                  targets (selection, normals, generators, CSV, target-set edits),
                  collision (Coal via Pinocchio), payload (static torque), feasibility (per-point check);
                  later: carrier, standoff
opt/              reachability (map, lookup, voting, coverage matrix), placement (stations), cost;
                  later: tool_geometry
tools/            tool library (one YAML per EE)
assets/robots/    robot registry (ur20/ in v1)
assets/tools/     EE URDFs + meshes
assets/parts/     test parts
examples/         ready-to-run project files
tests/            pytest + golden scenes
cache/            reachability maps (*.npz, gitignored, keyed by robot+TCP+settings hash)
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

## Tool Setup (M3)

- **EE geometry (core/tool.py):** `T_flange_head = T_flange_mount · Rz(roll) · Tz(L) · Rx(θ) · Ty(d)` and `T_flange_tcp = T_flange_head · T_head_tcp`. The extension collision is capsules of `radius`: segment 1 runs from the mount origin along mount z for L; segment 2 (only when d ≠ 0) is the lateral leg to the head origin. A `straight` extension forces θ = d = 0. The head frame is the EE URDF root link, and head links are rigid.
- `Extension` holds **ranges** (optimizer bounds) plus `nominal` (the user's rough placement, validated to lie inside the ranges). Changing a range in the GUI clamps the nominal value into it.
- **Contacts per TCP:** `contact_links` (whole links) and `contact_faces` (link → face indices on that link's *visual* mesh as loaded by yourdfpy). Collision filtering in M5 must use those same meshes and indices. Contact-free presets (scanning, spraying) clear all contacts.
- **Mass:** URDF inertials (head) + `kg_per_m` × extension length at segment midpoints. `mass_kg` and `com_head` override the head. `head_mass_known=False` when the URDF has no inertials and there is no override.
- **Tool library:** `tools/<name>.yaml` (EndEffector dump; name restricted to `[A-Za-z0-9_-]`). Saving an uploaded tool copies its URDF folder into `assets/tools/<name>/` so the entry is self-contained. The sample tools are stand-in geometry from `scripts/make_sample_tools.py`.
- **GUI edits** go through `ToolSetupTab._commit(fn)`: dump to a dict, apply the change, re-validate the whole `EndEffector`, and on error notify and revert the widgets. Meshes load only in `Session.set_tool`; parameter edits swap `tool_model.ee` without reloading.
- **Face picking:** a click ray (world) goes into the link frame, then `pick_face`, then `grow_region` (dihedral ≤ slider angle). Clicking an already-picked face removes its region. Picking is off until the checkbox is ticked.
- `CellTab.listeners` get `T_world_flange` after every base or joint change; the tool view follows the flange this way.

## Targets (M4)

- **The GUI has 5 tabs:** Tool Setup, Cell, **Targets**, Optimize, Results. Targets got its own tab instead of living in Tool Setup as the plan suggested.
- **Session state:** `session.targets` (schema.Targets) holds points in the **part frame** (m) with outward unit normals. Groups carry tolerances; a new group copies the current tool's process tolerances. The scene draws points, normals and the selection under `/cell/part`, so they follow the part gizmo.
- **Selection:** an array of face indices on `Session.part_mesh()` (scaled to m), cached per (path, units) and cleared when the mesh changes. Click selects a dihedral region; shift-click adds; ctrl/meta-click removes. You can also select by normal direction, or by box: `on_rect_select` projects face centres through `client.camera`, assuming an OpenCV camera frame. Box select ignores occlusion and has not been checked in a real browser yet.
- **Normals:** the default is barycentric interpolation of area-weighted vertex normals computed **from the selected faces only**, so normals never blend across the selection border. On the curved panel this gives < 0.05° interior error and ≤ ½ facet angle at the border. `normals="face"` gives flat normals.
- **Generators (`core/targets.py`):**
  - `projected_grid`: rays against the mean normal or a given direction; keeps only first hits on selected, front-facing faces.
  - `poisson_disk`: dart throwing on a hash grid; 0.7 s for 0.48 m² at 10 mm.
  - `raster`: `mesh_plane` sections, chained, resampled by arc length, serpentine. Note that `mesh_multiplane` returns **2D** coordinates, which caused a bug here.
  - `edge_paths`: the selection's boundary loops.
  - `parse_csv`: `x,y,z,nx,ny,nz[,group]`, optional header, mm or m.
- **Edits** apply to points lying on the selected surface: assign group, flip, delete. "Preview tool at point" draws a ghost of the tool posed with tcp.z = −n, at origin p + standoff·n (`targets_ui.target_tcp_pose`).

## Feasibility (M5)

- **Collision (`core/collision.py`):** a single Pinocchio `GeometryModel` holds the robot collision meshes (from the URDF), the tool extension capsules, the head link meshes split by contact signature, the part (BVH) and the floor (half-space at world z = 0). Pair categories and margins:
  - robot–part and tool–part: ≥ clearance.
  - tool–robot: ≥ 5 mm, with the last link (wrist_3) excluded.
  - robot self-pairs: |i − j| ≥ 2, minus any that touch at the neutral pose.
  - floor: links from upper_arm on, plus the tool.

  `collides(q, tcp)` uses Coal `security_margin`, so it is a clearance test, not just contact. A collision check takes about 0.7 ms.
- **Contact filtering is per TCP.** A link in `contact_links` is skipped whole. Picked `contact_faces` are skipped **plus their one-ring** (faces sharing a vertex), otherwise the rim faces at the contact edge always register as a collision.
- **Target frame / spin reference:** z = n; x = part +x projected onto the tangent plane (+y when n ∥ x). The TCP pose is `T0 · Rz(az) · Rx(tilt) · Rz(spin − az)` with `T0 = target · Rx(π)`, plus `standoff · n`.
- **Sampling:**
  - Spin: free = 30° steps; a range starts at its centre and works outward, ends included.
  - Tilt: 0, then rings at cone/2 and cone with 6 azimuths each.
  - Standoff: mid, lo, hi.

  Poses are tried with increasing tilt, and the **first pose with a feasible branch wins**; its best branch is chosen by manipulability + 0.1 × limit margin. Full cost-based choice comes in M7.
- **Checks, in order:** IK → joint limits − margin (the best 2π copy is kept) → static torque (τ = g + J_comᵀ(−mg) + Jᵀ F with F = −force·n; utilization ≤ limit; mass ≤ rated) → collision. An unreachable point reports the first stage nothing got past (`no_ik`, `joint_limits`, `torque`, `collision`). `marginal` means clearance < margin + 5 mm, torque > 90 % of its limit, or a joint within 15° of its limit.
- **Payload:** joint torque limits come from spec.yaml, with URDF effort as fallback. Payload-diagram and wrist-moment checks are reported as *skipped* until spec.yaml has them. `Robot.set_mount(R_world_base)` rotates gravity for wall and ceiling mounts.
- **GUI (Optimize tab):** the check runs in a thread on a **separate `Robot` instance**, because Pinocchio data is not thread-safe. Results go in `session.feasibility` (not saved, and stale after edits until re-run). Targets switch to "Color points by: feasibility". The inspector moves the robot to the chosen solution.
- **Speed:** reachable points take about 10 ms; unreachable points try every candidate (up to 13 tilts × 12 spins × 3 standoffs). The checker still uses the scalar IK; switch it to `solve_batch` when M7 needs throughput.

## Reachability (M6)

- **Batch IK:** `solve_batch(ik, T[N,4,4]) → q[N,8,6], valid[N,8]`, with branch index 4·i1 + 2·i5 + i3. Wrist-singular poses are marked invalid. `ur_manipulability` is the closed form |a2·a3·s3·s5·(a2·c2 + a3·c23 + d5·s234)| = |det J|, the same for any TCP offset and frame (verified against Pinocchio).
- **Map (`opt/reachability.py`):** IK + joint limits (with margin) only; no collision or torque. It relies on the **symmetry about base z**, which is valid when joint 1 turns a full revolution (every UR). It is stored on a 2D (r, z) grid × 600 Fibonacci tool-axis directions in the radial frame, taking any of 4 spins. `reach` is a boolean and `manip` the best |det J|. Defaults: 50 mm cells, about 70 s to build for UR20 + sander, then cached in `cache/reachability/<hash>.npz`. The hash covers the URDF bytes, DH, limits, `T_flange_tcp` and the settings, so **every tool length L gets its own map**. Non-UR robots raise `NotImplementedError`.
- **Lookup** is bilinear in (r, z). The tilt cone is applied **around the actual target axis** at query time; if the cone holds no direction, the nearest one is used. Dilating per direction bin instead (cone + half a bin) made the map optimistic and moved the optimum by 175 mm, and 200 directions were too few for a 10° cone.
- **Voting:** `vote_bases(map, P, N, cones, xs, ys, base_z, R_world_base, quality="min")`. Score = coverage + the worst target's normalized manipulability. **Base yaw has no effect**, by the symmetry. The "mean" quality is nearly flat, so its peak position is poorly defined; "min" (maximin) gives a sharp optimum. For the done-when the scene must not be mirror-symmetric, otherwise there are two equal optima.
- **GUI:** the "Base placement" panel in the Optimize tab draws the heatmap as an RGBA `add_image` on the floor and marks the best base. "Move base to best" goes through `CellTab.set_base`. **Viser image orientation:** columns run along +x, and **row 0 is at −y** (checked with a 2×2 test image), so the (ny, nx) vote grid is drawn unflipped.

## Station planner (M7)

- **Pipeline (`opt/placement.py`):**
  1. Candidate bases on a floor grid (100 mm) around the targets, minus the part's world-xy AABB grown by 0.25 m.
  2. Map coverage matrix (B × T), with "covered" meaning map score ≥ 0.5. Pruned to about 400 candidates: the best overall plus the 5 best per target.
  3. **CP-SAT** for each k from k_min to max_stations: maximize coverage ≫ Σ assigned manipulability ≫ fewer bases. The greedy cover is the fallback and sets k_min.
  4. Open-TSP order: brute force for k ≤ 7, otherwise nearest neighbour.
  5. **CMA-ES** per station (≤ 24 evaluations, σ 80 mm, within 0.3 m), scored by the exact checker on that station's targets with a 40-pose screening budget (`check_point(max_poses=...)`).
  6. `verify`: each target is checked exactly from its preferred station, then from every other one.
  7. Cost via `opt/cost.py` (the plan's J; μ normalized by the map max; c_ref 50 mm; D_ref 1 m; limits relative to 30°).

  Solutions are streamed through `on_solution` and returned sorted by cost.
- `FeasibilityChecker.set_base` / `CollisionScene.set_base` re-place the part and floor without rebuilding the geometry, which keeps refinement cheap (verified identical to a fresh build).
- **Results** are typed (`schema.Station`, `schema.Solution`; assignment = point id → station index) and saved in `project.results`. The per-point joint solutions (`session.result_points`, used for ghost robots) are kept only in memory.
- Travel is the straight-line distance between stations; obstacles are ignored in v1, so stations may sit on opposite sides of the part. Base yaw is kept from the current base.
- **GUI:** the Optimize tab has a "Station planner" panel (background thread, progress, cancel). The **Results** tab ranks solutions, shows a text cost-breakdown bar, colored station frames with labels, lines to the points each station serves, semi-transparent ghost UR20s (ViserUrdf with `mesh_color_override` RGBA) at each station's first solution, and "Move base to station".
- **Planning time** for the rail (21–25 points, k = 2..4) is about 15–30 s including refinement, with the map already cached.

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
- 2026-11-04 (M7): Placement alternatives are the best solution for each station count (k_min..max) rather than near-duplicate no-good cuts. Added `ortools` and `cma` (pip) to environment.yml.
- 2026-11-04 (M6): Base voting scores coverage plus the *worst* target's manipulability (maximin). Map defaults: 50 mm, 600 directions, 4 spins.
- 2026-11-04 (M5): The feasibility check takes the first feasible pose in tilt order rather than optimizing per point, and the marginal clearance band is margin + 5 mm (a contact tool's body always sits close to the part).
- 2026-11-04 (M4): Added a separate Targets tab. Click picking plus region growing in Viser works on the curved test panel. Whether it works on real part meshes stays open until the user supplies one; PyVista remains the fallback.
- 2026-11-04 (M3): The extension model is fixed as Rz(roll)·Tz(L)·Rx(θ)·Ty(d) with capsule collision. This answers the plan's open question "straight or one bend": both are supported, and `type` selects which.
- 2026-11-04 (M2): `Project.end_effector` and `Project.part` are optional until the Tool Setup (M3) and part steps fill them in. Projects are saved as `<folder>/project.json`. The UI shows UR's required copyright notice.

## Open decisions (from plan; decide before M5)

- [ ] Viser picking + region growing good enough on real meshes, or PyVista for selection? It works on the curved test panel (M4); confirm on a real part mesh.
- [x] Tool geometry v1: straight + one bend (with lateral offset d), selectable per tool (decided in M3).
- [ ] Single tool length for all stations, or a tool swap per station?
- [ ] Gantry: station mode only in v1, or coordinated mode too?
- [ ] Designer output: envelope sketch, URDF, or STEP stub?
