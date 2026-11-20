# Changelog

Our changes from `main` (a572f3f, 2026-11-06) up to `feature/v9-usability`, newest first, and since
2026-11-18 the same work rebased onto upstream `v13` (`rebase/v13`), then the batch merge
(`feature/batch-merge`). No release has been tagged yet. The
sections below are the development rounds, each a branch built on the one before:

| Round | Branch | Dates |
| --- | --- | --- |
| One batch system | `feature/batch-merge` | 2026-11-19 |
| Rebased onto upstream v13 | `rebase/v13` | 2026-11-18 |
| v9 usability | `feature/v9-usability` | 2026-11-16 |
| Batch, rendering, packaging, tool surface | `feature/v8-batch-render-exe` | 2026-11-13 .. 15 |
| v8 review fixes | `fix/v8-review` | 2026-11-13 |
| v8 | `v8` | 2026-11-09 .. 12 |
| v7 and v7 round 2 | `v7` | 2026-11-09 |
| v6 | `v6` | 2026-11-08 .. 09 |
| v4 plan round (v5) | `v5` | 2026-11-08 |
| v3 feedback round (v4) | `v4` | 2026-11-06 |

Details per feature are in the user guide (`docs/USER_GUIDE.md`) and in `CLAUDE.md`.

**Compatibility notes** (also listed in their sections):
- Old project files still load. Removed settings (payload, torque, mass, tip-over) are dropped silently.
- Plans made before the review fixes show as out of date once, because the part fingerprint now includes
  the mesh content.
- Tool-optimizer costs now include carrier travel and the standoff penalty, so they differ from earlier
  versions.
- `batch.json` is version 2 from v9 on and version 3 from the batch merge. Older batches open; an older
  program refuses a newer file.
- Batch jobs no longer keep their own planner settings: every job of a batch plans with the cell's
  settings (only *Quick plan* is per job), and a cancelled job runs again from the start.

---

## One batch system (2026-11-19)

Upstream's *Several parts (batch)* and our *Batch queue* are one tool now: **Optimize → Optimizer:
Several parts (batch)**.

- **Add jobs** three ways: the current part with its face selection; a folder of **part packages**
  (from the package builder); a folder of **part files** (STL / OBJ / PLY / STEP) with the tool surface
  detected automatically (below *Min confidence* the part waits for review). One **Add folder** button
  takes packages and part files from the same folder.
- **Planned like the GUI:** each job runs in its own process in a browserless app, several at once
  when the PC has the cores and memory: stations, optional axis frame search per part, paths and cycle
  time, a saved project and a PDF report per part.
- **Kept from the queue:** statuses, remove / reorder / reset, resume (only pending jobs run, also
  after a restart), *Open for review*, the review page with pictures, setup warnings per part, the
  time-left estimate.
- **Results → Batch results** opens the finished parts (it opens by itself when the batch ends);
  *Open result* in the panel is gone.
- **Run batch** uses the open project as it is (no need to save); a batch made in another project keeps
  the cell it was made with.
- **Command line:** `scripts/batch_run.py <batch folder>` runs a batch's pending jobs;
  `scripts/batch_run.py <project> <parts folder> <output>` still runs a parts folder directly.
- **Removed:** the *Batch queue* entry, per-job planner settings, the *partial* status (a cancelled job
  goes back to pending), the PDF checkbox (a PDF is always printed when Edge or Chrome is found), the
  per-job CSV / userframes exports (open the part's project to export them).
- **CI, first runs on GitHub** (on a fork): the tests passed; the release build failed its offline
  browser check (headless Edge did not exit) and took 38 minutes. Now: the browser check caps Edge's time
  and prints Edge's own errors; the release builds 4 reachability maps at once (`build_exe.py
  --map-jobs`); the conda env is cached; test batches run 3 at a time (`test_batches.py --jobs`); tests
  no longer run twice on a tag; the actions are on Node 24 (GitHub's deprecation warning). GitLab gets
  the same commands.

## Rebased onto upstream v13 (2026-11-18)

Upstream's own rounds v9 to v13 (one Optimizer dropdown, package builder geometry, gantry placement,
trajectories, robot programs, compare setups, part-package batches, ...) are described in
`docs/HISTORY.md` and `docs/V9_PLAN.md` .. `docs/V13_PLAN.md`; this section lists only what changed
in our work when it moved onto them.

### Changed
- **Two batch systems.** Upstream's *Several parts (batch)* (part packages) stays; ours is
  **Optimizer → Batch queue** (meshes with a selection or the tool-surface rule, review page, PDF and
  time left).
- **Setup check** is above the Optimizer dropdown.
- **Compare setups** (upstream's successor of Compare robots) keeps finished rows on Cancel and uses
  the Station planner's exact settings.
- **Everything (zip)** also holds the robot program (URScript / TP / RAPID) when the plan's paths
  are built.
- GitLab CI lints in its own Linux job, like upstream's GitHub workflow.

### Fixed (in upstream v13 code)
- The project's clearance and joint margins now reach the parallel check workers.
- A part file rewritten within the same second is noticed (out-of-date tracking).
- Changing the collision-geometry mode (reduce / hull / wrap) marks plans out of date.
- With scipy 1.14 the app and many tests failed ("buffer source array is read-only").

---

## v9 usability (2026-11-16)

### Added
- **Setup check** (Optimize → Setup check): finds common mistakes before a plan, as advice only. Checks:
  - a part under 10 mm or over 30 m across (usually the wrong units), or reaching below the floor;
  - normals pointing into the part, points more than 5 mm off the surface, duplicate points;
  - points inside a stay-out zone, and groups that allow TCPs the tool does not have;
  - for a static base, points spanning more than one base position can cover.

  It also runs quietly when a planner starts. Command line: `python -m ui.preflight_run <project>`.
- **Everything (zip)** (Results → Export): one handover zip with the TCP and userframe CSVs, the report,
  the tool URDF, the built paths and `project.json`, plus a README that explains each file and lists
  what could not be made.
- **Log file** `logs\app.log` in the data folder (rotating, 4 × 1 MB), and **Settings & help → Save
  support bundle (zip)**: version, system info, logs, `project.json` and batch errors. It never contains
  part or tool meshes, and nothing is sent anywhere.
- **Batch queue:**
  - optional **PDF report per job** (checkbox, or `--pdf` on the command line);
  - a **time-left estimate** in the progress line;
  - setup-check findings listed as job warnings on the review page.
- **GitLab CI** (`.gitlab-ci.yml`, `.gitlab/ci/`): the same tests and release build as the GitHub
  workflows. GitLab has no draft releases, so its release job waits for a manual click. It has not been
  run on GitLab yet.

### Changed
- `batch.json` is version 2 (it stores the PDF setting).
- `scripts/release_notes.py --changes FILE` appends a change list (used by GitLab). Without the option
  the output is unchanged.
- `scripts/make_portable.py` checks that every runtime package imports in the env before it writes the
  portable zip, and names any that are missing.

---

## Batch, rendering, packaging, tool surface (2026-11-13 .. 15)

### Added
- **Batch queue** (Optimize → Batch queue):
  - queue many parts against one cell, each in its own study folder;
  - a fresh worker process per job, cancel and resume;
  - exports and `summary.csv` / `summary.html` per batch;
  - command line `python -m ui.batch_job <batch dir>`;
  - STEP parts;
  - a self-contained `review.html` with offline software renders of each part and its tool surface.
- **Detect tool surface** (Targets tab): proposes the surface the tool works on, with confidence,
  warnings and secondary candidates to add. In a batch, the rule "auto: tool surface" sends jobs with
  low confidence to *needs review*. Validated on synthetic mandrels only. An optional tool pad radius
  feeds a concave-radius warning.
- **STEP / STP parts** (OCP import, CAD face ids per triangle, cached tessellation).
- **Windows installer and zip:**
  - PyInstaller onedir build, per-user Inno Setup installer (no admin rights);
  - `--selftest` that runs offline;
  - per-user data folder in `%LOCALAPPDATA%\RobotCellOptimizer`;
  - `scripts/build_exe.py`.
- **Release workflow** (GitHub, tag or manual run): builds the installer and zip, tests the exe offline
  behind a firewall rule, install-tests it and drafts a release with user-facing notes. It has not been
  run on GitHub yet.
- **App version** shown in the panel, the ready line, `--version` and the installer.
- **Linux:** CI test matrix on ubuntu-22.04, `scripts/run_app.sh`, Linux browsers for PDF reports.
- **View controls:** Studio / Flat lighting with an offline HDRI, optional shadows, Fit / Top / Front /
  Side / Iso views, refit on load.
- **Report 3D views:** framed Iso and Top pictures with helpers hidden.
- Report, CSV, userframe and target generation can be run without the GUI (`report_make`, `target_gen`).

### Changed
- **Faster drawing:**
  - selection, point cloud and normals are updated in place;
  - heatmap and station areas are one vertex-coloured mesh with cached modes;
  - parts over 300k triangles are shown as a simplified copy (picking and planning still use the full
    mesh);
  - playback updates are throttled.
- Writable data is kept apart from the bundled read-only files (`core/data_paths`). Source checkouts
  behave as before.

### Fixed
- After a save and reopen, the Cell step no longer shows as changed because of a `-0.0` in a pose.
- An untagged build no longer breaks the installer's file version (a bare commit hash becomes 0.0.0.0).

---

## v8 review fixes (2026-11-13)

### Fixed
- **Worker jobs:** results that cannot be sent between processes give an error instead of a hang. A
  crashed worker reports its exit code. Errors have a one-line summary, with the traceback in the log.
- **Check pool:** a broken pool falls back to serial checks with a warning.
- **Reachability lookups** use bounded memory and give identical results.
- **Uploads** are validated before they replace a mesh or URDF; a rejected upload keeps the old file.
- **Planning flow:**
  - previous results are kept until the first new plan arrives, and are restored after a failed run;
  - a station run no longer drops tool-optimizer designs;
  - cancel handling is fixed for Pareto, robust and Compare runs.
- **Open project.json** loads into a new folder, never over the current project. Autosave and Save wait
  until the load has finished.
- **Jog** blocks steps when the collision check is unavailable (unless *Allow collisions* is on).
- **Margins:** `optimizer.margins` is the single source for clearance and joint margin in every
  planner and check. Changing it marks plans out of date.
- **Reports and CSV:**
  - a "Not checked" section;
  - collision geometry per body;
  - an OUT OF DATE banner on "Export anyway";
  - clearer PDF errors;
  - the UR notice only for UR meshes;
  - CSV comment rows, and an empty cell for clearance that was not measured.
- **Other fixes:**
  - multi-body carriages in contour mode;
  - deterministic numeric IK;
  - a CSV with a BOM keeps its first row;
  - Delete all points asks first and can be undone.

### Changed
- Tool-optimizer costs include carrier travel and the standoff penalty (values differ from before).
- Plans show as out of date once after the upgrade (the part fingerprint now includes the mesh content).

---

## v8 (2026-11-09 .. 12)

### Added
- **Speed:** the station planner and tool optimizer run in a separate worker process, and exact checks
  run in parallel in a process pool. Plans are identical to serial runs.
- **Planning methods** chosen in dropdowns:
  - Bayesian refinement;
  - Pareto fronts, with a clickable chart that selects a plan;
  - robust objective, and alternating tool and stations;
  - least-joint-motion point order, including over every arm configuration (GTSP).
- **Tolerance check** (Results): Monte Carlo of part and stop offsets.
- **Cycle time:** estimate, cost term and path cycle time. Reduced-speed zones count in it.
- Collision checks between path samples; part coloured by a score of the selected plan.
- **Robots:** UR10e, UR30, FANUC CRX-10iA / 20iA/L / 30iA, ABB GoFa 10 / GoFa 12 / PoWa 30; *Compare
  robots*.
- **Tool design:**
  - approach cones as a start shape;
  - two-bend tools;
  - several TCPs in the planner;
  - part families;
  - standard tube sizes and bend-angle steps;
  - STEP export of the tool;
  - tools from one STL / OBJ / PLY / STEP file, whose STEP datums place the TCP and mount.
- **Safety:** cable-wrap cost, hose pack (clearance margin or a shape on the arm), reduced-speed zones,
  reach envelope.
- **Package builder** (own window with a robot preview): URDF packages for tools, rails, gantries and
  mobile bases, with collision geometry per part.
- **Collision geometry:** low-detail collision copies (reduce / hull / wrap) per component, arm padding,
  settings in Robot & cell and show / hide in the left panel.
- **Jog:** TCP and carrier drag gizmos, a carrier joystick, mobile moves without limits.
- **Planner:**
  - one shoulder side for arms that cannot swing over the top;
  - a home pose that reaches every stop;
  - unwinding at joint limits;
  - collision-free joint moves between groups.
- **Mobile contour standoff range** (planned gap). Unreachable points say why and what they hit.
- **Reports:** customer report sections and PDF export; `scripts/ursim_check.py`.
- **Quality:**
  - browser end-to-end tests in headless Edge;
  - GitHub CI workflow;
  - result provenance (code version, methods, settings hash, seed);
  - speed history (`profile_plan.py --record`);
  - searchable user guide.
- Confirmed setup steps turn their gizmos off; the floor grid is sized to the cell.

### Fixed
- Build paths no longer fails with "'NoneType' has no attribute 'stations'" when the targets watcher
  ticks mid-build.
- A standoff "up to" at or below "from" is one fixed standoff (Apply no longer fails).

### Not done
- A convex part model (CoACD) measured slower than the existing BVH and was dropped.

---

## v7 and v7 round 2 (2026-11-09)

### Added
- **Robot-agnostic IK:** OPW solver for FANUC / ABB / KUKA / Yaskawa-style arms, and a common solver
  interface; `scripts/check_robot.py`.
- **Opposite sides:** *Each side separately* generates targets per side of the part. Thin plates keep
  both faces.
- **Planners:** *Quick station plan*, and a *Rules only* mode (rules 1-6, fewest stations or fewest
  groups, minimum manipulability).
- **Start-up window:** reopen last, recent projects, examples, new empty cell.
- **Move mode:** Alt+click or Objects → Move… with Confirm / Cancel.
- **Axis lengths** for rails and gantries (bodies stretch, travel follows).
- Results summary card, HTML log rows, Jog in its own panel, *Minimize log*.
- `scripts/profile_plan.py` (station-plan profiler).

### Changed
- **Removed payload, torque, mass and tip-over checks** (user decision). Old files load with those
  fields dropped.
- Plan stations, Quick plan and Optimize tool stay disabled until steps 1-4 are confirmed.
- **Faster planning:** rail example 52 s → 8 s, mobile example ~58 s → 13 s (checker cache, fewer
  collision calls).
- Panel cleanup: Settings & help folder, collapsed folders.

### Fixed
- Collision pairs are chosen by link rather than by shape index (needed for robots with several
  collision shapes per link).
- Heatmap errors are shown instead of lost.
- Unreachable markers no longer use an out-of-date feasibility check.
- `examples/headless_plan.py` no longer crashes when no tool length reaches every point.
- Shift / Ctrl clicks on the part arrive (click modifiers are registered).

---

## v6 (2026-11-08 .. 09)

### Added
- **Lift height planned per stop**, lift-stage bodies, lift staged at the stops.
- **Examples:** tall wall with an overhead gantry, floor X-Y rails, mobile with lift.
- **Gantry bridge axis** (bodies on the X axis).
- **Collision-free moves:**
  - home and reconfiguration moves;
  - detours around collisions;
  - retreat and park poses;
  - a *Drive carrier between stops* toggle.
- **Paths:** configuration groups in paths and areas; each point sanded once; *Back to start*; *Clear
  paths*.
- **Joint limits** and a **home pose** per project, TCP lock, jogging in world / base / part / tool
  frames, collision-checked jogging (*Allow collisions*).
- **Log panel**, unreachable-point markers, the top 3 tool designs.

### Changed
- Configuration and motion costs; fewer configuration changes on gantries.
- Carrier sliders and arm jog moved to the left Jog panel.

---

## v4 plan round, v5 (2026-11-08)

### Added
- **Targets:**
  - one *Tool* dropdown (fill, brush handle, box, by normal, full surface, single points);
  - a rows / columns surface grid.
- **Paths:** sand-optimizer patterns per stop and group, lifts over crossings, home and carrier moves
  between stops.
- **Station areas:** connected areas per stop, a colour layer on the part.
- **Costs:** editable costs (per stop, per group, per wrist flip), must-meet rules, up to 40 stops.
- **Userframe export:** robot base → part frame per stop, in the robot's frame convention.
- **Mobile carrier** routes around stay-out zones while keeping its standoff; saved robot home pose.

### Changed
- **Faster station search:** coarse-to-fine, and ghost robots are reused on redraw. Rail top at 50 mm:
  43 s → 18 s.

---

## v3 feedback round, v4 (2026-11-06)

### Added
- **New part, same cell / New empty study** (the robot is optional).
- **3D stay-out zones** (arm, tool and carrier keep clear) with a gizmo.
- **Mobile robot sides** (several), the part sides the base may work from, and rail / gantry tracks
  aligned with the part.
- **Left object tree** (bodies, zones, result overlays, visibility).
- Tests run in a private workspace; a batched, resumable test runner (`scripts/test_batches.py`).

### Changed
- Paths are cleared on new runs and target changes, and skip deleted points.
- Uploads accept mm, meshes are double-sided, and gizmos have an adjustable size and draw on top.
