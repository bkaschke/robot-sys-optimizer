# V7 plan, round 2: task list (branch `v7`, after 66541c2)

The user's requests came on 2026-11-09 and their decisions are recorded below. The speed and
usability items are findings from a profile and the round-1 review. **Start a new session with:
"Implement docs/V7_PLAN.md on branch v7, task by task, in the order given."**

## Ground rules (from CLAUDE.md / memory, repeated so a new session has them)

- Python: `C:/Users/Ben/miniforge3/envs/rso/python.exe`; ruff: `.../envs/rso/Library/bin/ruff.exe`.
  Lint with `ruff check . && ruff format .`.
- Run the full suite with `scripts/test_batches.py` (about 60 min, memory is tight). While
  working, run only the affected test files. Tests set `RSO_WORKSPACE` (conftest) and never
  write to `workspace/`.
- Keep files under 500 lines: split before growing a file past that. `core/schema.py` is
  already 532 lines.
- Keep `docs/USER_GUIDE.md` in sync with every GUI label, and rebuild `docs/USER_GUIDE.html`
  (see `tests/test_guide.py` for how). Add a "v7 round 2" section to `CLAUDE.md`.
- Commit / push only when the user asks. No `Co-Authored-By` trailer; end commits with the
  `Claude-Session:` line only. Leave `docs/ur20_data_sheet.pdf`, `examples/mobile_test/` and
  `v1_feedback.txt` untracked.
- When the user asks for a release: a draft release only, with the zip from
  `scripts/make_portable.py`.
- Patch with the Edit tool or Python scripts; sed / heredocs mangle backslashes.
- Things only a browser can show (gizmo grabbing, modals, docking, colours) must be listed in
  the final report as "not verified in a browser", unless the user tried them.

## User decisions (2026-11-09)

| Question | Decision |
|---|---|
| Jog position | its own panel at the bottom of the left column |
| Feasibility check / heatmap before setup is confirmed | stay usable |
| Singularity limit | configurable percentage of the robot's max manipulability (default 5 %) |
| Rules planner objective | the user picks: **Fewest stations** or **Fewest groups** |
| Startup | the app opens empty, with a project popup (details in T9) |

---

## Phase A: speed first (makes testing everything after it faster)

Profile, `rail_long_part` station plan (UR20, 86 points, zones on), **51 s total**:

- zoning (`opt/zoning.py: zone_solution` → `FeasibilityChecker.check_point_configs`): 45 s
  - `CollisionScene.collides`: 21.5 s, 26 278 calls
  - `CollisionScene.part_clearance`: 20.7 s, 14 753 calls
- set cover, CMA-ES refinement, IK: about 5 s

To reprofile, run `python scripts/profile_plan.py <example>` (cProfile around
`app["stations"].start(block=True)`).

### T1. Skip branches of configurations already found
- `core/feasibility.py`: `_feasible(..., skip=None)`. Before the collision check, drop branches
  for which `skip(q)` is true. `check_point_configs` passes
  `skip=lambda q: config_label(ik, q) in out`.
- **Done when:**
  - `check_point_configs` returns the same dict as before on the rail example's points (a test
    compares old and new: configurations, q, clearance);
  - `collides` calls drop by more than 50 %.

### T2. Clearance only for kept solutions
- In `check_point_configs`, call `_feasible(..., clearance=False)` and compute
  `scene.part_clearance` only for each configuration's first hit, before returning. Status
  "marginal" is recomputed from that clearance.
- **Done when:**
  - results are identical to T1's (status, clearance);
  - `part_clearance` calls ≤ 8 per point.

### T3. Cache exact checks across station counts
- `opt/station_checks.py`: a per-plan memo keyed by (rounded station values to 0.1 mm /
  0.01°, point id) for `check_point`, `_robust` and `check_point_configs`. It is created in
  `plan_stations` and dropped at the end, so it never outlives one plan (the setup can change
  between plans).
- **Done when:** the plans are unchanged (the test_examples assertions plus the same station
  count and coverage).

### T4. Measure
- Rerun the profile on `rail_long_part` and `mobile_amr`.
- Targets: rail < 15 s (from 51 s), mobile_amr < 20 s (from ~58 s).
- Record the before/after numbers in the Status section below.
- Tighten `tests/test_v5_speed.py` limits only if the margin is large (the PC is slow under
  load).

---

## Phase B: workflow and panels

### T5. Optimizers need a confirmed setup
- `ui/workflow.py refresh()`: when any of steps 1–4 is not "confirmed", set `disabled=True` on:
  - *Plan stations*, the tool optimizer's run button and the quick planner (T14), and later
    rules mode, which shares the station planner's button;
  - a markdown line under each one: "Confirm **2 External axis** first", or "**3 Tool** changed
    since you confirmed it — confirm it again".
- Each `start()` checks the same condition and calls `notify` instead of running, so tests and
  scripts can't bypass it.
- **Stays usable:** the feasibility check and the base-placement heatmap (user decision).
- Add `tests/helpers.confirm_all(session)` (or a conftest fixture); `scripts/profile_plan.py` and
  `examples/headless_plan.py` must confirm too (or the shipped examples are saved confirmed). Update every test that
  plans without confirming; grep for `.start(block=True)`.
- **Done when:** a test shows run disabled → confirm all → enabled → edit the tool → disabled,
  with the reason naming step 3.

### T6. Jog panel at the bottom of the left column
- `ui/objects.py`: move the *Jog* folder (Move / Step / arrows, `carrier_slot`, `arm_slot`)
  into a new panel: `jog = server.gui.add_panel()` with tab "Jog", then
  `jog.dock_below(tree)`. viser 1.1.1: the anchor must be docked first, and `tree.dock_left()`
  already is.
- `app.py` passes `objects.arm_slot` / `carrier_slot` as now.
- **Done when:**
  - the jog widgets exist in the new panel, and `tests/test_app.py` jog tests still pass;
  - a test checks the panel is docked below the tree, if viser exposes it; otherwise list it
    as browser-only.

### T7. Coloured log
- `ui/log_panel.py`: render the rows with `gui.add_html` instead of `add_markdown`.
  - Colours: error #d32f2f, warn #b26a00, done #2e7d32, info grey.
  - Dark-mode variants: read the app's dark flag; `app.py` toggles it.
  - Escape the messages with `html.escape`, then convert `**bold**` and `_italic_` (the only
    markdown the log uses) to tags.
- The minimized view keeps the newest row's colour.
- **Done when:** a test renders one message of each level and finds its colour, and an
  injected `<script>` text is escaped.

### T8. Targets tab: delete options under point generation (user request)
- `ui/targets_ui.py`: move *Delete points on selected surface* and *Delete all points* out of
  *Edit points* into the *Generate points* folder, right under *Generate on selection* (or into
  a "Delete points" folder placed directly below *Generate points*, expanded). *Assign* and
  *Flip normals* stay in *Edit points*.
- Update the user guide and `ui/settings.py` if it stores folder states.
- **Done when:** in the folder order, the delete buttons follow Generate, and
  `tests/test_targets_gui.py` passes.

### T9. Open empty, with a project popup
- `app.py __main__`: no longer reopens the last project automatically. The cell starts empty:
  the robot shown, no part, no targets, no tool, with the default static base. Keep `--fresh`
  as an alias that does nothing, so old shortcuts don't break, and add `--open <folder>` to skip
  the popup.
- On each client connect (first connect only, per session), open `client.gui.add_modal(
  "Open a project")` with:
  - **Reopen last project** (folder name and date; hidden if none);
  - **Recent projects** (dropdown + Open; the last 5, stored where `last_project()` keeps its
    record; extend it to a list);
  - **Examples** (dropdown + Open: `ProjectPanel.load_example`);
  - **New empty cell** and **Close**.
- The popup closes after a successful load, on New and on Close. A failed load shows the
  error in the popup and in the log, and the popup stays open.
- `scripts/run_app.bat` and the portable launcher: check they don't pass flags that change
  meaning.
- **Done when:** tests call the popup's handlers directly (build the modal content in a
  function that takes a `gui` object; the server-level `gui` works in tests) for reopen,
  example, new and close, and the session state matches each.

### T10. Gizmos hard to grab in the middle of a model
- `GizmoStyle.on_top` (depth_test off) is already the default, so this is not occlusion.
  Likely causes, check them in this order:
  1. Clickable meshes win the pointer. The part, carrier bodies and stay-outs have `on_click`
     handlers (face selection, picking), and viser raycasts clickable meshes before the
     transform controls. Fix: while a gizmo is shown, turn the clickable handlers off on the
     object under it, e.g. `remove_click_callback`, or a guard flag plus no handler registered.
  2. The gizmo is smaller than the body: its handles sit inside the mesh silhouette, where the
     orbit controls take the drag. Fix: scale the gizmo to the object's bounding box
     (`scale = 0.6 × max extent`, clamped), with the *Gizmo size* slider as a multiplier.
- **Ask the user to retry in the browser** after each fix: this can't be verified headlessly.
  Note in the report which cause was fixed.

### T11. Alt+click → Move, with Confirm / Cancel
- viser 1.1.1 does not report right-clicks: `on_click` only has modifiers (`alt`,
  `cmd/ctrl`, `shift`, ...). So use **Alt+click** on an object (part, robot base, carrier,
  stay-out zone, carrier body), and add a **Move…** button per object in the Objects tree, for
  discoverability.
- **Flow:**
  1. Alt+click.
  2. A small floating panel appears near the click (`panel.float(x, y)` from `screen_pos`):
     "Move <object>", with **Confirm** and **Cancel**.
  3. That object's gizmo turns on; any other gizmo turns off.
  4. The pose is saved before the move.
  5. **Confirm** keeps the new pose and commits it the way the existing gizmo commit does
     (setup step goes "changed").
  6. **Cancel** (or Esc, if viser can catch it; else only the button) restores the saved pose
     exactly and redraws.
- One shared helper, `ui/move_mode.py`: `start(obj_key, get_pose, set_pose)`,
  `confirm()`, `cancel()`. The existing per-panel gizmo checkboxes call it too, so there is
  one gizmo code path.
- If Alt+click turns out to clash with viser's camera controls in the browser, use
  Ctrl+click.
- **Done when:** tests drive `start` → move → `cancel` (the pose is restored bit for bit),
  and `start` → move → `confirm` (saved, step marked changed).

### T12. Stay-out zones: the first zone can't be selected after adding another
- Reproduce first (headless):
  1. add box, add box;
  2. set `pick.value` to the first;
  3. drag the gizmo;
  4. check which zone moved.

  Suspects:
  - `refresh(pick)` / `sync()` re-entrancy (`_busy`) leaves `pick.value` pointing at the
    newest zone;
  - the gizmo keeps the newest zone's pose because `place_gizmo` only moves an existing
    gizmo when `drag` is on;
  - two zones with the same default name: `_unique` then `index()` finds the first by name.
- Fix, and also:
  - click on a zone in 3D selects it (mesh `on_click` → `pick.value`);
  - zones are listed in the Objects tree, each with **Move…** (T11).
- **Done when:** a test with three zones selects each in turn (dropdown and simulated click)
  and moves it, and only that zone changes.

---

## Phase C: rails and planners

### T13. Rail / gantry length in one setting
- Move `Base`, `Primitive`, `CarrierJoints` and related models from `core/schema.py` to a new
  `core/schema_base.py`, re-exported from `core.schema`. This gets schema.py under 500 lines.
- **Schema:**
  - `Primitive.stretch: Literal["", "x", "y"] = ""` marks the bodies that follow an axis
    length;
  - `Base.axis_length_m: dict[str, float] = {}` (keys `x`, `y`);
  - `Base.end_stop_m: float = 0.05`;
  - `Base.manual_travel: bool = False`.
- `core/carrier.py body_list`: a stretched box gets size[axis] = length.
  - Its pose along the axis stays centred.
  - Cylinders along the axis use `height`.
- **Travel limits**, unless `manual_travel`:
  `lo, hi = ±(L/2 − carriage half-extent along the axis − end_stop)`.
  - The carriage extent is the max over `attach="carriage"` bodies (rail) or `x_axis` bodies
    (gantry y).
  - A validation error if that range is empty.
- **GUI**, new file `ui/carrier_axis.py` (`carrier_ui.py` is at 486 lines):
  - *Rail length (mm)* (rail), *X rail length* + *Bridge length* (gantry);
  - an **Advanced** folder (collapsed) with *End stop (mm)* and *Set travel limits manually*,
    which shows the existing Travel fields; otherwise they are read-only and show the computed
    limits.
  - Mobile bases don't show these.
- `scripts/make_examples.py` and *Load carrier defaults*: set `stretch` on the track, X rails
  and bridge, and `axis_length_m`. Regenerate the example projects.
- Old projects (no `stretch`) behave exactly as before: lengths are not shown, and travel stays
  manual.
- **Done when:**
  - a test changes the rail length: the track's visual and collision geometry change size;
  - the travel limits follow, and manual limits override;
  - an old project loads unchanged;
  - `test_examples` passes.

### T14. Quick station planner (3rd optimizer)
- A new folder in Optimize, "Quick station plan", with *Max stations* (shared with the station
  planner), **Quick plan** and a status line.
- It runs `plan_stations` with `refine=False, zoning=False, areas=False, patience=1,
  quick_first=True`.
- Results are labelled "Quick: …". The status says "not optimized; run Plan stations to
  optimize".
- Same gating (T5), Cancel and progress as the planner. Logs the run time.
- **Done when:** on `rail_long_part` it gives a full-coverage plan, and takes less than a third
  of the full planner's time (measured in the test, with a loose bound).

### T15. Rules-only station planner (a mode of the station planner)
- *Station planner → Mode*: **Cost based** (today) / **Rules only**. In rules mode the Costs
  panel is hidden; these show instead:
  - *Objective*: **Fewest stations** / **Fewest groups** (user decision);
  - *Singularity limit (% of max manipulability)*: default 5, range 0–50 (user decision:
    configurable);
  - a fixed list of the rules, as text.
- **Rules** (all hard):
  1. every target reached;
  2. tool orientation within each group's spin / tilt tolerance (already hard in
     `candidate_poses`);
  3. joint limits with the margin (already hard);
  4. no collision with the clearance (already hard);
  5. one joint configuration per group, with no wrist jump > 180° (zoning on; a zone with
     wrap jumps fails);
  6. **new:** manipulability ≥ limit × robot max.
- **Rule 6 implementation:**
  - `CheckSettings.min_manip: float = 0.0` (absolute; the GUI converts the % using
    `rmap.max_manip`, or a robot-level max from the reachability map).
  - In `_feasible`, branches below it are skipped.
  - A new stage "singular" between joint_limits and collision in `STAGES`, with a
    `REASONS["singular"]` text: "only reachable close to a singularity".
  - Default 0, so cost mode and old results are unchanged.
- **Search** (`opt/rules_plan.py`, new):
  - **Fewest stations:** for k = k_min … Max stations, run the quick plan (no refinement); if
    it passes, try up to 3 alternative set covers at that k (exclude one chosen candidate at a
    time), zone each, and keep the one with the fewest groups. Stop at the first k with a
    passing plan.
  - **Fewest groups:** go through k as above, but keep going while the group count drops,
    stopping at Max stations or after `patience` counts without fewer groups. Report the plan
    with the fewest groups, then fewer stations.
  - No CMA-ES. Refinement optimizes cost, which rules mode has none of.
- **No plan passes:** show the closest plan (fewest broken rules, then fewest broken points),
  marked ✘. The status line and the log list each rule with its count, e.g. "rule 6: 12 points
  only reachable close to a singularity at 4 stations".
- `Solution.violations` already exists; add per-rule counts to it or to `breakdown`.
- **Done when:** tests check that
  - rules mode never returns a plan with unreached points marked as passing;
  - Fewest stations ≤ Fewest groups in station count, and Fewest groups ≤ Fewest stations in
    group count, on the same example;
  - a high singularity limit (e.g. 50 %) makes the plan fail with rule 6 named;
  - cost mode is unchanged (the existing tests).

---

## Phase D: usability (smaller, after A–C)

- T16. Disabled buttons say why, everywhere, not just the optimizers: Export (no result),
  Build paths (no plan), Check feasibility (no targets).
- T17. A **Results summary card** at the top of Results for the selected plan:
  - stations, groups, points reached;
  - worst clearance, closest joint limit, lowest manipulability %;
  - the planner and mode, rules pass / fail, run time.
- T18. Move *Inspect result* from Optimize to Results.
- T19. `scripts/test_batches.py --fast`: skip the tests marked `slow` (full plans, examples);
  mark them. The fast run should take under 10 min.

## Order

T1 → T2 → T3 → T4 → T5 → T8 → T6 → T7 → T12 → T10 → T11 → T9 → T13 → T14 → T15 → T16–T19.

After each phase:
- lint;
- run the affected tests;
- update the user guide and CLAUDE.md;
- tick the task in Status.

After all of them, run the full suite and report to the user, including what was not verified
in a browser.

## Status

| Task | State | Notes |
|---|---|---|
| T1–T19 | not started | |
