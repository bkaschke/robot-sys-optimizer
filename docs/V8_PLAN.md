# V8 plan: task list (branch `v8`, from `v7`)

The user approved this list on 2026-11-09 (see "User decisions"). It adds reach-study
features, new optimization methods next to the current ones, generative tool design, and
work that makes the app faster and easier to test. **Start a new session with: "Implement
docs/V8_PLAN.md on branch v8, task by task, in the order given."**

## Ground rules (from CLAUDE.md / memory, repeated so a new session has them)

- Python: `C:/Users/Ben/miniforge3/envs/rso/python.exe`; ruff: `.../envs/rso/Library/bin/ruff.exe`.
  Lint with `ruff check . && ruff format .`.
- Run the full suite with `scripts/test_batches.py` (about 60 min, memory is tight; `--fast`
  skips `@pytest.mark.slow`). While working, run only the affected test files. Tests set
  `RSO_WORKSPACE` (conftest) and never write to `workspace/`. Don't restart a background run
  that was killed for memory unless the user asks.
- Keep files under 500 lines: split before a file grows past that.
- Keep `docs/USER_GUIDE.md` in sync with every GUI label, and rebuild `docs/USER_GUIDE.html`
  with `scripts/build_guide.py`. Add a "v8" section to `CLAUDE.md`.
- Commit / push only when the user asks. No `Co-Authored-By` trailer; end commits with the
  `Claude-Session:` line only. Leave `docs/ur20_data_sheet.pdf`, `examples/mobile_test/` and
  `v1_feedback.txt` untracked.
- When the user asks for a release: a draft release only, with the zip from
  `scripts/make_portable.py` (new dependencies must be in that zip).
- Patch with the Edit tool or Python scripts that use `encoding="utf-8"`; sed / heredocs
  mangle backslashes, and cp1252 writes break files with ⚠ / − in them.
- viser 1.1.1 only reports click modifiers a handler registered (see `ui/cell.CLICK_MODIFIERS`).
- Things only a browser can show must be listed in the final report as "not verified in a
  browser", unless T20 covers them or the user tried them.

## User decisions (2026-11-09)

| Question | Decision |
|---|---|
| Scope | everything proposed, **except items that need a payload model** (process-force deflection, tool mass / CoG into the torque check) |
| Current optimizers | **keep them all**, unchanged and the default; the new methods are added beside them |
| Choosing a method | **dropdown menus**, each with a short description of what the selected method does |
| Guide | updated for every new option, and **searchable** |
| Bayesian GP | **own numpy / scipy GP** (no PyTorch); TPE and NSGA-II from Optuna |
| CadQuery | **wanted**: installed in `rso`, in `environment.yml` and **in the portable zip** (STEP export is a normal feature, not optional) |
| Run mode | **run every task to the end unattended**; commit v7 first, branch `v8` from it, commit locally after each task, **no push** (so T21's workflow is written and linted but its GitHub run is not verified) |
| UR10e / UR30 | download from Universal Robots' ROS 2 description repo with `scripts/import_ur_description.py`, licences included, like the UR20 |
| URSim | not available: T19 writes the script and dry-runs it only |

## Dependencies (ask the user before installing; add to `environment.yml` and the portable zip)

| Package | For | Notes |
|---|---|---|
| `optuna` 5.0 | T10 Bayesian (TPE), T11 NSGA-II | pure-Python wheel + SQLAlchemy, alembic, Mako, MarkupSafe (small; checked with `pip install --dry-run` 2026-11-09). Optuna's **GP sampler needs PyTorch** (~1 GB unpacked): the GP method is a small numpy / scipy GP in `opt/refine_bayes.py` instead, unless the user wants torch |
| `coacd` 1.0.14 | T3 convex decomposition | binary wheel `cp39-abi3-win_amd64`, no other dependencies; optional, falls back to the current mesh |
| `cadquery` 2.8 | T17 STEP export | **heavy**: pulls `cadquery-ocp` (OpenCascade), VTK, matplotlib, numba, trame and more. Optional, kept **out of the portable zip** unless the user decides otherwise; without it the tool exports STL / URDF as today |

Not needed: cycle time uses a built-in trapezoidal profile (no `toppra`); the PDF report is
printed by Edge headless (`--print-to-pdf`, no `reportlab`); browser tests reuse CDP over
`websockets` (no Playwright). OR-Tools is already installed (CP-SAT, routing for T13).

---

## Phase A: groundwork (every later task runs more plans)

### T1. Planning in a worker process
- New `opt/jobs.py`: a `Job` runs a planner call (station plan, quick plan, tool optimizer,
  feasibility check, the new studies) in a `multiprocessing` worker (spawn). The worker gets
  a picklable snapshot of the session (project model + settings), streams progress / log
  rows / intermediate results back through a queue, and returns the `Solution`s.
- The GUI keeps today's progress bars, Cancel and log; Cancel terminates the worker if it
  doesn't stop within 2 s. Only one job per kind at a time; a queue line in the status shows
  waiting jobs.
- `block=True` (tests, `profile_plan.py`) still runs in-process, so tests keep their speed.
- **Done when:**
  - a station plan from the GUI gives the same solutions as `block=True` (test);
  - Cancel ends the worker and frees its memory (test checks the process is gone);
  - the GUI answers (a button callback runs) while a plan runs (test).

### T2. Parallel exact checks
- `FeasibilityChecker.check_points(indices, workers=N)`: split the points over a process
  pool (each worker builds its checker once from the snapshot). Used by `verify`, zoning,
  *Check all targets* and T4 / T5. `CheckMemo` results come back to the parent.
- Setting: *Settings → Worker processes* (default: CPU count − 1, max 8; 1 = today's path).
- **Done when:** results equal the serial path on the rail example (test); the rail plan's
  zoning step is at least 2× faster with 4 workers (measured, loose bound, `slow`).

### T3. Faster collision and clearance
- Convex decomposition of the part with CoACD (cached on disk by mesh hash) for
  `collides`; a precomputed signed distance field (voxel grid, 5 mm default) for
  `part_clearance`, with trilinear lookup and an exact fallback below 2 cells.
- Setting: *Settings → Collision model*: **Exact mesh** (today, default) / **Convex parts
  (CoACD)**. Clearance via SDF is always on once built, and exact within 1 mm (test).
- **Done when:** same reach result on all 5 examples with both models (test, `slow`); the
  profile for `rail_long_part` is re-recorded in Status.

### T4. Method picker and searchable guide (UI groundwork for Phase C)
- New `opt/methods.py`: a registry `METHODS[slot] = [Method(name, short, run)]`. `short` is
  one line (the dropdown hint and the text under it). The first entry of each slot is
  today's method.
- New `ui/method_picker.py`: `picker(gui, slot, label)` makes the dropdown plus a muted
  markdown line under it that shows the selected method's description and updates on
  change. Saved in settings by name; an unknown saved name falls back to the default.
- **Searchable guide** (`scripts/build_guide.py`):
  - a search box at the top of the sidebar (`/` focuses it);
  - as you type, the sidebar is filtered to matching sections, matches in the page are
    highlighted, and a result count is shown; Enter / Shift+Enter jump to the next / previous
    match, Esc clears;
  - `USER_GUIDE.html?q=word` runs a search on load, so the app can link to it;
  - self-contained inline JS, works offline, keeps the theme toggle and print styles.
- In the app: a *Search the guide* text box next to the guide link opens the guide with `?q=`.
- **Done when:** `test_guide.py` checks the search box, the `?q=` handling and that the
  build is up to date; a headless Edge check (T20 harness) shows a search finds "CMA-ES".

---

## Phase B: reach study

### T5. Tolerance check (Monte Carlo)
- New `opt/tolerance.py`: perturb the part pose (± mm, ± deg) and optionally the station
  poses, N samples (default 50, seeded), re-run the exact check at the plan's stations.
  Per point: reach rate (%), and margin = smallest of distance to the workspace boundary
  (from the reachability map), joint-limit margin and clearance.
- GUI: Optimize → *Tolerance check* folder: *Part offset (mm)*, *Part tilt (deg)*,
  *Station offset (mm)*, *Samples*, *Seed*, **Run tolerance check**; result in the Inspect
  view ("worst 10 points") and as a heatmap source (T8).
- **Done when:** with zero tolerance every reached point scores 100 % (test); a point placed
  at the edge of reach scores clearly below 100 % at ±5 mm (test).

### T6. Cycle time
- New `core/cycle_time.py`: per move a synchronized trapezoidal joint profile from the
  robot's joint speed / acceleration limits (in `assets/robots/*/spec.yaml`; UR20 values from
  its data sheet) scaled by the speeds the export uses; linear moves (approach / retract,
  process paths) from TCP speed and length; carrier moves from the carrier axis speeds
  (new fields with defaults); a fixed time per stop for settling.
- Shown in the summary card and the Results table (*Cycle time (s)*), per stop and in total;
  available as cost term **w_cycle** (default 0, so current results don't change).
- **Done when:** a single joint move matches the hand-computed trapezoid (test); the total
  equals the sum of its parts on the rail example (test).

### T7. Collision checks between points
- Check the motion between consecutive points of the built paths: joint-space interpolation
  for joint moves, Cartesian interpolation (with IK continuity) for linear moves, sampled
  every 2° / 5 mm, against the part, the stay-out zones and the carrier.
- Built paths get a *Motion check* column; failing moves are listed and drawn red in the
  path view. *Build paths* gets a checkbox *Check moves between points* (default on).
- **Done when:** a move that passes through the part between two clear points is flagged
  (test, synthetic part); all example paths pass or their failures are listed in Status.

### T8. Reach heatmap on the part
- Colour the part's faces by a chosen score: *Reached / unreached* (today), *Margin*,
  *Manipulability*, *Configurations found*, *Tolerance reach rate* (T5).
- GUI: Targets → *Colour part by* dropdown with a colour legend; the per-point values come
  from the last check or plan, interpolated to faces nearest their points.
- **Done when:** face colours are monotonic in the score (test); switching back to
  *Reached / unreached* restores today's view.

### T9. Robot and scenario comparison
- Add UR10e and UR30 to `assets/robots` (public DH, joint limits and speeds).
- New `opt/compare.py` + Optimize → *Compare scenarios*: pick robots and / or saved carrier
  setups; for each, run the station plan (T1 worker, one after another) and collect
  stations, reached, cycle time, worst margin, cost, run time. Shown as a table; a row
  can be opened as the current setup.
- **Done when:** comparing UR20 vs UR10e on an example gives two rows and each matches a
  normal plan with that robot (test, `slow`).

---

## Phase C: optimization methods (added beside the current ones)

Every slot keeps today's method as the first, default entry; old projects load and plan
exactly as before. Each new dropdown uses `ui/method_picker.py` (T4).

| Slot (where) | Methods (first = today's) |
|---|---|
| Station planner → *Refinement* | **CMA-ES** · Bayesian (TPE) · Bayesian (Gaussian process) · None |
| Station planner → *Objective* (Cost-based mode) | **Weighted cost** · Pareto front (NSGA-II) · Robust (worst case) |
| Tool optimizer → *Shape search* | **CMA-ES** · Bayesian (TPE) · Bayesian (Gaussian process) |
| Tool optimizer → *Start shape* | **Nominal** · From approach cones (T14) |
| Tool optimizer → *Tool and stations* | **Stations fixed after length search** · Alternate |
| Tool optimizer → *Objective* | **Weighted cost** · Pareto front (NSGA-II) · Robust (worst case) |
| Build paths → *Point order* | **Pattern order** · Least joint motion (GTSP) |

Short descriptions (the dropdown hint and the line under it; the guide gets a longer
paragraph per method):

- **CMA-ES**: evolves a population of station / shape candidates; robust, needs many
  evaluations.
- **Bayesian (TPE)**: learns which regions give good results and samples there; good with a
  small budget and mixed settings.
- **Bayesian (Gaussian process)**: fits a smooth model of the cost and picks the most
  promising next try; the most sample-efficient for 1–4 smooth variables.
- **None**: keeps the set-cover result without refinement (fastest).
- **Weighted cost**: one number from the cost weights; returns the best plan.
- **Pareto front (NSGA-II)**: keeps the trade-off plans (stations, cycle time, margin, tool
  length) that no other plan beats on every measure; you pick one.
- **Robust (worst case)**: scores each plan by its worst tolerance result (T5) instead of the
  nominal one; plans survive fixture error.
- **Alternate**: re-plans the stations after each shape step, until neither improves.
- **Pattern order**: the current path patterns per group.
- **Least joint motion (GTSP)**: picks the point order and IK branch per point together to
  minimize joint motion and wrist flips.

### T10. Bayesian refinement (TPE and GP)
- `opt/refine_bayes.py`: the same variables, bounds and budget as the CMA-ES refinement
  (`refine_evals`) and the tool's `shape_evals`, through an Optuna study (TPE or GP
  sampler, seeded). The *Refine with exact check (CMA-ES)* checkbox becomes the
  *Refinement* dropdown; a saved `refine: false` loads as **None**, `true` as **CMA-ES**.
- **Done when:** each method returns a valid plan on the rail example (test); with the same
  budget, GP is no worse than CMA-ES on the tool example (measured and reported, not asserted).

### T11. Pareto front (NSGA-II)
- `opt/pareto.py`: Optuna NSGA-II over the same variables; objectives chosen by checkboxes
  (stations, cycle time, worst margin, tool length; default the first three). Results: all
  non-dominated plans in the Results list (labelled "Pareto 1/n …") and a 2-D scatter
  (pick the axes) in Results; clicking a dot selects that plan.
- **Done when:** no returned plan dominates another (test); the weighted-cost best plan is
  dominated by or on the front (test).

### T12. Robust objective
- Cost = worst (or CVaR 20 %, dropdown *Worst case* / *CVaR 20 %*) over the T5 samples of
  the plan's cost, with fewer samples during the search (default 10) and the full T5 check
  on the result.
- **Done when:** on a synthetic edge-of-reach case the robust plan has a better tolerance
  reach rate than the weighted plan (test).

### T13. Alternate tool and stations
- `opt/tool_geometry.py`: after the length search, loop {shape step (chosen method) → re-plan
  stations with that shape} until the cost improves by < 1 % or 5 rounds. File is near the
  limit: put the loop in `opt/tool_alternate.py`.
- **Done when:** the result is never worse than **Stations fixed** with the same seed (test).

### T14. Least joint motion order (GTSP)
- `core/path_order_gtsp.py`: per group, a layered graph (point × IK branch × spin sample);
  edge cost = joint move time (T6) + wrist-flip penalty; solve with OR-Tools routing (with
  disjunctions per point) and fall back to dynamic programming over a fixed order for > 200
  points. Lifts and crossings are re-computed for the new order.
- **Done when:** on every example the joint motion is ≤ the pattern order's (test); paths
  still pass T7.

---

## Phase D: generative tool design

### T15. Approach cones
- `opt/approach_cones.py`: per target, sample tool directions within its tilt tolerance and
  keep those whose tool capsule is collision-free with the part (fast, no IK); cluster the
  free directions (k-means on the sphere) and derive bend angle / offset candidates.
- *Start shape → From approach cones* seeds the shape search with the best candidates;
  the cones can be shown on the part (Targets → *Show approach cones*).
- **Done when:** a point in a slot gets cones only along the slot (test); seeding never ends
  worse than *Nominal* with the same budget (test).

### T16. Richer tool shapes
- Extension types: **Two bends** (θ1, d1, θ2, d2, roll) and **Spline** (3–4 control points,
  minimum bend radius); **Two tips** (a second TCP on the same tool: the set cover may
  assign each point to either tip, recorded per point and exported as two TCPs).
- **Part family**: the tool optimizer can take several parts (saved targets from other
  projects); the cost is the sum (or worst) over the parts.
- **Done when:** each new type builds, checks and exports (test per type); a two-tip tool
  reaches a set that neither tip reaches alone (synthetic test).

### T17. Manufacturable output
- Snap options: *Tube size* (standard OD list), *Bend angles* (free / 5° / 15° steps), ISO
  9409 flange pattern on the export. Snapping happens inside the search (a constraint), not
  after.
- STEP export through CadQuery when installed (else the button explains it and STL / URDF
  export works as today).
- **Done when:** snapped designs only use allowed values (test); STEP export round-trips the
  tube length within 0.1 mm (test, skipped without CadQuery).

---

## Phase E: safety, output, quality

### T18. Cable dress and safety
- Cost term **w_wrap**: joint-6 total wrap per stop beyond a limit (setting, default ±360°),
  and a hose-pack zone (a capsule along the upper arm and wrist) checked against the part.
- Safety: the robot's maximum reach envelope per stop drawn as a translucent shell; a
  *Guarding* section that lists the envelope's bounding box and suggests a fence line at a
  set distance (setting); stay-out zones can be marked *Reduced speed*, which T6 uses.
- **Done when:** a path with > 360° wrap gets the cost (test); a reduced-speed zone raises
  the cycle time of moves inside it (test).

### T19. Customer report and URSim check
- *Export → Report*: an HTML report (summary card, setup, stations, cycle time, tolerance,
  unreached points with reasons, screenshots of the cell taken by the client) printed to
  PDF by Edge headless; the HTML is kept too.
- URSim check (optional, user-run): `scripts/ursim_check.py` loads the exported program into
  a running URSim (docker, address setting) and reports whether it runs to the end.
- **Done when:** the report builds from an example (test checks sections, not looks); the
  URSim script is documented and dry-runs without URSim.

### T20. Browser end-to-end tests
- `tests/e2e/` (marked `e2e`, skipped when Edge is missing): the CDP harness from the v7
  click fix as a fixture. Tests: surface selection (plain / Shift / Ctrl), gizmo drag moves
  the part, Alt+click opens Move and Cancel restores, stay-out zone click and drag, dark-mode
  log colours (computed style), guide search (T4).
- **Done when:** all pass locally; `scripts/test_batches.py --e2e` runs them.

### T21. CI and reproducibility
- `.github/workflows/tests.yml`: lint + `-m "not slow and not e2e"` on Windows with the
  conda env (cache the env and the reachability maps).
- Each `Solution` stores `seed`, the method names, the settings hash and the git hash; the
  summary card shows them.
- `scripts/profile_plan.py --record` appends timings to `docs/perf_history.csv`.
- **Done when:** a dry run of the workflow passes (`act` or a pushed branch, when the user
  allows a push); two runs with the same seed give equal plans (test).

---

## Guide updates (part of every task)

- A new section **Optimization methods**: a table of every slot and method (as above), then a
  paragraph per method (what it does, when to use it, typical run time, what it needs).
- New sections or entries for: worker processes, collision model, tolerance check, cycle
  time, motion check, *Colour part by*, scenario comparison, approach cones, new extension
  types, snapping / STEP, wrap and safety, report, guide search.
- Every new GUI label appears in the guide (`test_guide.py` checks labels as today).

## Order

T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8 → T9 → T10 → T11 → T12 → T13 → T14 → T15 → T16 →
T17 → T18 → T19 → T20 → T21. After each phase: lint, run the affected tests, update the
guide and CLAUDE.md, tick the task in Status. At the end: the full suite and a report,
including what was not verified in a browser.

## Status

| Task | State | Notes |
|---|---|---|
| T1 | todo | |
| T2 | todo | |
| T3 | todo | |
| T4 | todo | |
| T5 | todo | |
| T6 | todo | |
| T7 | todo | |
| T8 | todo | |
| T9 | todo | |
| T10 | todo | |
| T11 | todo | |
| T12 | todo | |
| T13 | todo | |
| T14 | todo | |
| T15 | todo | |
| T16 | todo | |
| T17 | todo | |
| T18 | todo | |
| T19 | todo | |
| T20 | todo | |
| T21 | todo | |
