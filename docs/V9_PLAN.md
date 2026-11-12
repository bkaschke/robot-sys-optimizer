# V9 plan: GitHub issues #1-#14 (branch `v9`, from `v8`)

Reviewed with the user on 2026-11-12. Work goes phase by phase in the order below; commit locally
after each phase; push / release only when the user asks. **Start a new session with:
"Implement docs/V9_PLAN.md on branch v9, phase by phase, in the order given."**

## Ground rules

- Python `C:/Users/Ben/miniforge3/envs/rso/python.exe`; ruff `.../envs/rso/Library/bin/ruff.exe`
  (`ruff check . && ruff format .`); gh `C:/Users/Ben/miniforge3/Library/bin/gh.exe`.
- Run only the affected test files while working; the full suite in batches (memory is tight; don't
  restart a run that was killed for memory). Tests never write to `workspace/`.
- Files under 500 lines; split before growing past that.
- Keep `docs/USER_GUIDE.md` in sync with every GUI label; rebuild the HTML with `scripts/build_guide.py`;
  add a v9 section to `CLAUDE.md`.
- Commits: no `Co-Authored-By` trailer (no `attribution.commit` in `.claude/settings.json`); end with the
  `Claude-Session:` line. Leave `examples/mobile_test/` and `v1_feedback.txt` untracked.
- Releases are drafts with the `scripts/make_portable.py` zip; the user publishes them.
- Close an issue only when the user asks (reference it in the commit message: "#12").

## User decisions (2026-11-12)

| Question | Decision |
|---|---|
| Robot arm padding | stays in **Robot & cell** (the robot is not a package) |
| Loading a part | Robot & cell keeps a button that opens the package builder's **Part** tab |
| Gantry placement | the gantry / rail frame is placed with **standoff distances** from the part and a **z range**; other options welcome |
| Compare robots in Results | **view** each compared robot at its stations and build its paths **without** changing the project; a separate button switches the project's robot, **after a confirmation** |
| #13 | delete the floor keep-out from the mobile_amr example |
| Order | the most efficient one: bugs, GUI layout, package builder, compare / report, gantry placement |

## Phase 1: bugs

- **#2 STEP import fails silently.** The package builder meshes STEP in the upload callback (about 20 s
  even for a small file, no feedback). Run it in a thread with a progress bar and a status line; on
  failure show the error, keep the package, and make **Check** list the failed part.
- **#12 project files.** On every save copy each referenced asset that is outside the project folder and
  not shipped with the app (part mesh, tool URDF + meshes, carrier meshes, uploaded robot) into
  `<project>/assets/...` and store the new relative path. Opening a project with a missing file still
  opens it: warn in the log and ask for the file again (part: the Part upload; tool / carrier: their
  tabs) instead of failing the whole load.
- **#13** remove the keep-out from `examples/mobile_amr`; floor keep-outs get their own entry in the left
  panel's show / hide tree.
- **#14** standoff lines follow the fields as they are typed (no *Apply base settings* needed for the
  preview); the standoff range (*Standoff from* .. *up to*) is drawn as a filled band, min / preferred
  as lines.

## Phase 2: GUI layout (#10)

- **Optimize tab:** one **Optimizer** dropdown (Feasibility check, Base placement map, Station planner,
  Quick plan, Rules only, Tool geometry, Compare robots, Gantry placement (phase 5)), each with a short
  description; only the chosen method's settings are visible; one **Run** / **Cancel** and one progress
  bar. Shared settings in one **Settings** folder: requirements & costs (one copy for the planner and the
  tool optimizer), cycle time, cable & safety.
- **Hide what does not apply:** External axis shows only the chosen carrier type's fields (mobile-only,
  rail / gantry-only); the Tool tab only its tool type's fields; rarely used fields under **Advanced**.
- **Consolidate duplicates:** one copy of max stations / costs / rules; one "move robot to this solution"
  (Results); one report export with a format choice (HTML / PDF); Results' *Clear from scene* folded into
  the left panel's toggles; the home pose in one place.
- **Other:** a status strip (each workflow step's state and what is out of date), a summary card after
  every run, consistent units and hints. Propose further changes to the user as found.

## Phase 3: package builder for all geometry (#1, #3, #4, #5, #11)

- Builder tabs: **End effector**, **Part**, **Rail**, **Gantry**, **Mobile robot**. The Part tab takes the
  workpiece (mesh / STEP, units, origin), its collision simplification and the part surfaces kept exact.
- Collision configuration only in the builder (method, triangles, wrap offset, padding) with a
  **Simplify** button, a progress bar, a per-part status (not simplified / simplified N triangles /
  out of date) and **Revert to original**. Remove the Robot & cell collision folder except **Robot arm
  padding**; the left panel keeps only show / hide.
- **#3** each part takes a visual mesh and, optionally, its own collision mesh (upload next to it); URDF
  packages with both are read as they are.
- **#5** *Use in this project* asks for confirmation, loads, and closes the builder window; **Edit
  current tool / part / carrier** loads the project's into the builder.

## Phase 4: compare robots and report (#6, #7, #8)

- Compare keeps every robot's plans (solutions + joints), saved with the project. Results gets a
  **Robot** selector: view that robot at its stations, build its paths (a robot model for viewing and
  paths that is separate from the project's robot); **Use this robot in the project** switches it, after
  a confirmation.
- PDF: the comparison table and each robot's best plan; no per-point tables; summary cards, a per-stop
  table, cycle time, unreached counts by reason, images, page breaks for reading.

## Phase 5: gantry / rail placement (#9)

- A **Gantry placement** optimizer: the carrier frame (x, y, yaw; height within a **z range**) is placed
  relative to the part keeping **standoff distances** (min / max from the part outline) before the stops
  are planned: candidate frames scored quickly with the reach maps (coverage, then stops), the best few
  planned in full. Rails too (their track along / beside the part).
