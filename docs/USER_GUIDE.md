# Robot Cell & End-Effector Optimizer: User Guide

A browser-based tool that answers two questions for a robotic process (sanding, scanning, grinding, gripping, spraying):

1. **Where should the robot base (or its carrier) stand?**
2. **What is the shortest end effector that still reaches every target without collision or overload?**

You load a robot, a tool and a part, mark the surfaces to work on, and the optimizer reports base stations, the tool geometry, a per-point feasibility report and export files. The design rationale is in `docs/PROJECT_PLAN.md`; developer conventions are in `CLAUDE.md`.

---

## Contents

1. [Quickstart (5 minutes)](#1-quickstart-5-minutes)
2. [Set up on a new computer](#2-set-up-on-a-new-computer)
3. [How it works: concepts and conventions](#3-how-it-works-concepts-and-conventions)
4. [Tools and features, tab by tab](#4-tools-and-features-tab-by-tab)
5. [Worked examples](#5-worked-examples)
6. [Files, folders and formats](#6-files-folders-and-formats)
7. [Using it from Python (headless)](#7-using-it-from-python-headless)
8. [Troubleshooting and limitations](#8-troubleshooting-and-limitations)
9. [For developers](#9-for-developers)

---

## 1. Quickstart (5 minutes)

On a computer that is already set up (see section 2 otherwise):

```bat
scripts\run_app.bat            :: or: scripts\run_app.bat --dark
```

or from any shell:

```bash
C:/Users/<you>/miniforge3/envs/rso/python.exe app.py      # add --dark for dark mode, --port N to change port
```

Open **http://localhost:8080** in Chrome, Edge or Firefox. You will see a UR20 with an orbital sander and a flat test plate.

This guide is built into the tool: **User guide** (top of the panel) → **📖 Open user guide** opens it in a new tab (served at http://localhost:8081, the app's port + 1).

Then try a full job on the test plate:

1. **Targets tab** → *Tool* = `Fill +` → click the top of the plate (it turns green).
2. *Generate points*: set *Row spacing* and *Column spacing* to `100` → **Generate on selection**. Points with normals appear.
3. **Optimize tab** → **Check all targets**. The points turn green, amber or red (reachable, marginal, unreachable).
4. **Optimize tab** → *Base placement* → **Show base heatmap** (the first run builds a reachability map, about 1 min) → **Move base to best**.
5. **Optimize tab** → *Station planner* → **Plan stations**. The result appears in the **Results** tab.
6. **Optimize tab** → *Tool geometry* → **Optimize tool** → **Apply to tool**.
7. **Results tab** → *Export* → **TCP poses (CSV)**, **Optimized tool URDF (zip)**, **HTML report**. Files are downloaded and also saved to `workspace/default/export/`.
8. **Project** (top of the panel) → **Save** stores everything in `workspace/default/project.json`. Autosave also does this every 2 minutes. At the next start, the **Open a project** window offers **Reopen last project**.

Tick **Dark mode** under **Settings & help** at the top of the panel at any time.

**Ready-made examples:** **Project** → *Examples* → `rail_long_part` (UR20 on a 7th-axis rail), `mobile_amr` (UR20 on a mobile robot built from simple shapes), or, around a 3 m tall wall, `gantry_overhead` (UR20 upside down on an overhead X-Y gantry with a bridge), `floor_xy_rails` (UR20 on a floor X-Y gantry: X rails, a bridge, a carriage on the bridge) and `mobile_lift` (mobile robot with a 1 m vertical lift) → **Open example** → **Optimize** → **Plan stations**. Examples are read-only templates: opening one makes a working copy in `workspace/examples/<name>/` (`<name>_2`, … when an earlier copy has your changes, which is never replaced), and saving or autosave only ever writes that copy.

---

## 2. Set up on a new computer

Tested on **Windows 10/11**. Linux and macOS should work with the same conda environment but have not been tested. **No conda or internet on that PC?** Use the portable copy in [2.5](#25-portable-copy-for-a-pc-without-conda-or-internet).

### 2.1 Prerequisites

| What | Why | Get it |
| --- | --- | --- |
| Git | clone the repo | https://git-scm.com |
| Miniforge (conda) | Pinocchio and Coal (kinematics and collision) are only packaged for Windows on conda-forge, not pip | https://github.com/conda-forge/miniforge |
| A modern browser | the GUI runs in the browser | Chrome / Edge / Firefox |

Install Miniforge **for your user** (default location `C:\Users\<you>\miniforge3`). You do not need to add it to PATH.

### 2.2 Clone and create the environment

```bash
git clone https://github.com/bkaschke/robot-sys-optimizer.git
cd robot-sys-optimizer
C:/Users/<you>/miniforge3/Scripts/conda.exe env create -f environment.yml      # creates env "rso" (~5 min)
```

To update an existing environment after pulling changes:

```bash
C:/Users/<you>/miniforge3/Scripts/conda.exe env update -n rso -f environment.yml --prune
```

`environment.yml` pins **OpenBLAS**. The default MKL build crashed numpy on the development machine, so keep that line.

### 2.3 Check the installation

```bash
PY=C:/Users/<you>/miniforge3/envs/rso/python.exe
$PY -c "import pinocchio, coal, viser; print('ok')"
$PY -m pytest -q                        # ~160 tests, ~8 min; the first run also builds reachability maps
$PY examples/headless_plan.py           # end-to-end plan + exports without the GUI (~1 min)
```

### 2.4 First launch

```bash
scripts\run_app.bat        # Windows double-click / cmd
# or
$PY app.py --port 8080 [--dark]
```

The first station plan or heatmap for a tool builds a **reachability map** (about 50–70 s), which is cached in `cache/reachability/`. Later runs reuse it in seconds. The cache is not in git; each computer builds its own.

### 2.5 Portable copy for a PC without conda or internet

For a PC where you cannot install Miniforge or download packages, build one zip on a set-up computer and copy it over. The zip contains the app, its complete Python environment (Pinocchio and Coal included) and the reachability maps already built on the set-up computer (so the first plan on the target PC does not spend a minute building them; leave them out with `--no-maps`). Nothing is installed on the target PC: no admin rights, conda, Git or network are needed.

**On the set-up computer** (commit your changes first: the zip holds the last commit):

```bash
C:/Users/<you>/miniforge3/Scripts/conda.exe install -n base -c conda-forge conda-pack   # once
$PY scripts/make_portable.py          # -> dist/robot-sys-optimizer-portable-<commit>.zip (a few minutes)
```

**Get the zip from GitHub instead of a USB stick.** The zip is too large for the repository itself (GitHub's limit is 100 MB per file), so it is attached to a **GitHub Release**. A *draft* release is visible only to you (the repo owner), and you can download from it like from a published one:

1. On any computer or phone with GitHub access, sign in to GitHub as the repo owner and open **https://github.com/bkaschke/robot-sys-optimizer/releases**.
2. Find the release named **Portable Windows build (\<commit\>)**. A draft shows a grey **Draft** label and is listed only while you are signed in. Pick the newest one, or the one matching the commit you want.
3. Under **Assets**, click **`robot-sys-optimizer-portable-<commit>.zip`** (about 370 MB) to download it.
   - If the target PC itself can reach github.com, download it there directly.
   - Otherwise download it on another computer and copy it over with a USB stick or a network share.
4. Optional check that the file is complete: in PowerShell on the target PC, `Get-FileHash .\robot-sys-optimizer-portable-<commit>.zip` should show the same SHA-256 the release page lists next to the asset.

The draft link only works while you are signed in as the owner. To let colleagues download without signing in, open the draft → **Edit** → create a tag (e.g. `portable-<commit>`) → **Publish release**. The repository is public, so a published release can be downloaded by anyone.

**To put a new zip on GitHub** (on the set-up computer, after `make_portable.py`): upload it through **Releases → Draft a new release** (drag the zip into the assets box, then **Save draft**), or with the GitHub CLI:

```bash
gh release create portable-<commit> dist/robot-sys-optimizer-portable-<commit>.zip --draft --target <full commit sha> --title "Portable Windows build (<commit>)"
```

`--target` needs the **full** commit SHA (`git rev-parse HEAD`); the short form is rejected.

**On the target PC:**

1. Unzip the zip into a folder you can write to, for example `C:\Users\<you>\robot-sys-optimizer`. Avoid `Program Files`: the app saves projects and caches inside its own folder. (Right-click → **Extract All…** works; it takes a few minutes for the 55,000 files.) If Windows marks the files as downloaded from the internet and blocks the `.bat`, right-click the zip → **Properties** → tick **Unblock** before extracting.
2. Double-click **`Start Robot Optimizer.bat`**. The first start prepares the bundled environment once (about a minute), then the app starts.
3. Open **http://localhost:8080** in the browser. The browser only talks to this PC; the app makes no internet requests.

The target PC must be 64-bit Windows 10/11 like the build computer. To update, build a new zip and unzip it into a new folder; copy `workspace/` (your projects) and optionally `cache/` (reachability maps) over from the old folder.

---

## 3. How it works: concepts and conventions

### 3.1 Frames and units

| Frame | Meaning |
| --- | --- |
| `world` | cell origin, **z up** |
| `base` | robot `base_link` (= carrier pose × mount) |
| `flange` | robot `tool0` |
| `tcp` | one per contact region on the tool; **z points out of the tool, toward the part** |
| `target` | one per surface point; **z = outward surface normal** |

A target is reached when the TCP origin is on the point (plus any standoff along the normal) and **tcp.z = −target.z**, within the target's tolerances. Internally everything is metres and radians; the GUI shows **mm and degrees**.

### 3.2 Target tolerances (per group)

- **Spin** about the normal: `free`, a half-width (e.g. `30`), or a range (e.g. `80,100`).
- **Tilt cone**: the largest angle between the tool axis and the normal.
- **Standoff**: the distance along the normal (0 = contact; a range for scanners).

These come from the tool's process preset and can be edited per group.

### 3.3 The parametric end effector

The tool is **mount + extension + head**: `T_flange_head = T_flange_mount · Rz(roll) · Tz(L) · Rx(θ) · Ty(d)`.

- **L**: straight extension length; **θ**: one bend; **d**: lateral offset after the bend; **roll**: about the mount axis.
- The extension is a capsule of a given radius for collision.
- Each value has a **range** (the optimizer's bounds) and a **rough placement** (the value shown and used until optimized).

### 3.4 What "feasible" means

For each target, poses inside its tolerances are tried, starting with the least tilt. A pose is feasible when **all** of these pass:

1. **IK** exists.
2. **Joint limits**, with a margin (default 5°).
3. **Collision** with 10 mm clearance: robot, tool, part, floor and carrier. The tool's **contact areas** may touch the part.

Status: **reachable**; **marginal** (feasible but tight: clearance < margin + 5 mm or a joint within 15° of its limit); or **unreachable**, with the first check that failed.

### 3.5 The optimizer in three layers

1. **Tool geometry (outer):** the shortest L by bisection, then θ / d / roll by CMA-ES.
2. **Placement (middle):** a cached **reachability map** scores base positions, an exact **set cover** picks the fewest stations, stations are ordered for short travel, then refined against the exact checks (CMA-ES by default; see *Optimization methods* in section 4 for the alternatives).
3. **Per point (inner):** the feasibility check above.

The cost J weighs tool length, unreached points, extra stations, travel, manipulability, clearance, joint-limit proximity and the carrier's preferred-standoff penalty. The Results tab shows the breakdown.

---

## 4. Tools and features, tab by tab

### Guided steps

The tabs are numbered in setup order: **1 Robot & cell → 2 External axis → 3 Tool → 4 Targets → 5 Optimize → 6 Results**. Each setup tab starts with its status (*not confirmed yet*, *confirmed*, or *changed since you confirmed it*) and a **Confirm step** button (**Confirm setup** on the External axis tab, which also locks the carrier). You can open any tab at any time; the Optimize tab lists the steps that are not confirmed. **Confirming a step turns its drag gizmos off**, so a confirmed object can't be moved by accident: step 1 the robot base, part, stay-out and reduced-speed zone gizmos, step 2 the carrier body and mount gizmos, step 3 the tool mount and TCP gizmos (also when a project opens with those steps confirmed). To move one again on purpose, pick it in Objects → *Drag gizmo on* (or use **Move…** / Alt+click); the step then shows *changed* until you confirm it again.

**Out-of-date results.** Every check, plan and tool result remembers the setup it was computed for. When you then change the robot or mount, the external axis, the tool, the part (mesh or position) or the targets, the Results tab shows **Out of date: … changed** for that result, the Optimize tab flags an out-of-date feasibility check, and exports ask before writing (**Export anyway** / **Cancel**). Re-run to clear it. **Plans that no longer apply are removed:** when the **part moves** (or its mesh changes), the **tool** changes or **targets** are added, removed or edited, the station plans and tool-optimizer results made for the old setup are cleared from the Results list together with their paths (the Log says which and why; applying a tool design's values does not clear that design). A plan that was already out of date when its project was opened is only flagged, not removed. (Moving the robot or carrier between stops does not make a plan out of date; the feasibility check, which runs from the current base, does go out of date.) Results saved before v2 carry no record and are not flagged.

**Log** (window at the bottom centre of the 3D view; drag it elsewhere, or **Minimize log** to keep only the newest message). Its top line lists the setup steps that are **not confirmed** yet. Below, newest first: every error and notification, each finished run (station plan, tool optimization, feasibility check, path build) with its result and how long it took, and **why points could not be reached**: grouped by reason (out of reach, beyond the joint limits, collisions, not covered by any stop) with the point ids. **Clear log** empties it.

### Top of the panel

**Settings & help** (closed by default):

- **Dark mode**: switches the GUI and the floor grid. The `--dark` flag starts in dark mode.
- **Floor grid**: sized to the cell. It covers the part, the robot's reach around its base and around every planned stop, the carrier (fixed bodies and the moving ones at both ends of their travel) and the stay-out / reduced-speed zones, plus 1 m all round (at least 10 × 10 m, in whole metres). It follows when you move or add things (checked every second).
- **Run planners in a separate process** (on by default): *Plan stations*, *Quick plan* and *Optimize tool* run in a worker process, so the panel stays responsive, **Cancel** always ends the run (a run that does not stop within 2 s is ended with its process, which frees its memory; the plans already listed stay), and a second run waits until the first has finished ("waiting for the running planner job"). The worker keeps the robot, maps and meshes loaded between runs and is ended after 5 idle minutes. Untick it to run in the panel's own thread as before.
- **Check worker processes** (default: the number of CPUs − 1, at most 4): the station planner's exact checks (verification, localization re-checks, zones, refinement, station areas) run in this many processes at once; **1** = one after the other. Each process holds its own robot and collision scene (about 300 MB). Starting them takes about 10 s; they are kept for the next plan of the same setup. The plans are identical either way; it pays off on large point sets (the overhead gantry example: 114 s → 78 s), not on the 25-point examples. If the processes cannot start, the plan runs without them and the log says so.
- **User guide**: **📖 Open user guide** opens this guide in a new browser tab. It is served by the app on its port + 1 (`--guide-port N` to change it) and also works from another PC on the network. **Search the guide**: type a word and click the **🔎 Search the guide for …** link under it; the guide opens with that search done. **Download user guide (HTML)** saves the same page as a file.
- **Searching this guide**: the box at the top of the guide's sidebar (or press **/**) finds a word or phrase anywhere in the guide: matches are highlighted, the sidebar keeps only the sections that contain one, and the count shows where you are. **Enter** jumps to the next match, **Shift+Enter** to the previous, **Esc** clears. A link ending in `?q=word` opens the guide with that search.
- **Objects** (its own panel on the **left** of the 3D view): *Show / hide* is a tree of the cell: **Robot** (and its *Tool*), **External axis** (and each carrier body), **Part** (*Targets*, *Selection*), **Stay-out zones** (each zone), **Reduced-speed zones** (each zone) and **Results** (*Stops and lines*, *Ghost robots*, *Station areas*, *Paths*); a child is shown only when its parent is ticked too. *Gizmos*: *Drag gizmo on* chooses which object has drag handles (robot base, part, tool mount, carrier body, stay-out zone, reduced-speed zone); *Gizmo size* scales all of them and *Gizmos always on top* draws them through bodies, so one inside a body can still be grabbed. The part and stay-out gizmos are sized to their object (its handles reach outside it; *Gizmo size* multiplies that), The Targets selection tools keep working while the part's gizmo is on; only while **Move…** moves the part do clicks on it go to its gizmo. **Move object** + **Move…** (or **Alt+click** the part or a stay-out zone in 3D): a small *Move* window appears with **Confirm** and **Cancel**, and only that object's gizmo is on. Drag it, then **Confirm** keeps the new pose (a confirmed setup step shows *changed*, as after any drag) or **Cancel** puts the object back exactly where it was (Esc does not work here, use the button). The robot base, the carrier (moved with the Jog panel while the window is open) and the carrier bodies (while the External axis setup is not confirmed) are in the *Move object* list too.
- **Jog** (its own panel at the **bottom of the left column**, below Objects): *Move* the robot / carrier or the part with the arrow buttons (−X/+X, −Y/+Y, −Z/+Z, −Rz/+Rz) by *Step (mm)* / *Turn step (deg)*. With a carrier the arrows drive its axes: a rail stops at the end of its travel, a gantry within its x / y range, a mobile base moves and turns without limits (Z = the lift axis, if any). Below them, **Carrier position** holds the carrier's axis sliders (from the External Axis tab) and **Robot arm** the arm jog (frames, joint steps; described under Robot & cell → *Jog*), so everything that moves the robot is in one place.
- **Project → New part, same cell**: keeps the robot, tool, external axis and stay-out zones, clears the part, its targets, plans, checks and paths. **Project → New empty study**: nothing at all, not even a robot (choose one in tab 1). Both first save the current study and start in a new folder `workspace/study_<date>_<time>`, so the previous study stays as it was.
- **Project**: *Folder* (relative to the repo, or absolute), **Save** / **Load** (`<folder>/project.json`), **Download project.json**, **Open project.json** (upload). Loading never half-applies: a broken project leaves the current session untouched.
  - **Autosave** (on by default) saves to the current project folder every *Autosave every (min)* minutes (default 2) when something changed. The line under it shows the last save time.
  - **Open a project at start**: the app opens on an **empty cell** (the robot on a static base, no part, tool or targets) and the first browser that connects gets the **Open a project** window: **Reopen last project** (folder and date), **Recent projects** (the last five you saved, loaded or autosaved; *Open recent*), **Examples** (*Open example*), **New empty cell** (a new folder `workspace/study_<date-time>`) and **Close** (keep the empty cell). A project that fails to load shows the error in the window, which stays open. The empty start-up cell is not autosaved until you open, create or save a project. Start with `--open <folder>` (e.g. `scripts\run_app.bat --open workspace\examples\rail_long_part`) to open a project directly without the window; `--fresh` is still accepted but does nothing now.
  - A project brings back the whole cell **and** where you were: the arm's joint pose, the surface selection, feasibility colours, every plan's joint solutions (ghost robots, CSV export without re-planning), the selected solution, and the panel settings (spacing, brush, check and planner settings, display options, dark mode). *Examples* + **Open example** opens a working copy of a shipped example (see Quickstart). To add or update an example, type a *New example name* and click **Save as example** (tick *Overwrite existing example* to replace one); this is the only way the app writes into `examples/`. Saving a project to another folder takes its uploaded meshes (`assets/`) along.

### 1 Robot & cell tab

| Section | What it does |
| --- | --- |
| **Robot** | Choose a robot from the registry (the UR20, UR10e, UR30, or `generic_6r`, a made-up industrial arm for trying non-UR kinematics; see [Adding a robot](#adding-a-robot)), or **Upload URDF (.zip)** for another robot (numeric IK: display, jogging and the feasibility check; the planners need an analytic solver). The mount presets are floor, wall and ceiling. Move the base with the gizmo or the xyz/rpy fields (static base only: with a carrier the base rides on it and these fields are read-only, so the robot can never be dragged off the carrier). *Joints (deg)* sliders jog the arm; the readout shows the flange pose computed by Pinocchio. **Go to home** jogs the arm to its home pose. *Home position* (folder): jog the arm with the sliders and press **Set home to current pose**, or type one value per joint in *Home joints (deg)* (comma separated) and press **Apply typed values**; values are clipped to the joint limits. The home is saved with the project; **Use robot default** goes back to `home_deg` in the robot's `spec.yaml` (UR20: 0, −90, 0, −90, 0, 0). The paths use this pose between stops. *Joint limits* (folder): a *min / max* (deg) per joint; **Apply joint limits** narrows the robot's range (kept inside the URDF limits, a joint with min ≥ max is refused) and **Use URDF limits** goes back. The limits are saved with the project and every check uses them: IK solutions, the station planner, the tool optimizer, the paths, the joint sliders and the home pose (re-run the planner after changing them; Results are flagged out of date). The quick reach pre-check keeps joint 1's full turn and the exact checks apply its limit. *Jog* (left column → Jog panel → **Robot arm**): step the robot like a teach pendant. *Frame* = **Joint** (*Joint − / +* J1..J6 by *Joint step*), or **World**, **Robot base**, **Part** or **Tool**: *Move − / +* X, Y, Z move the *Jog point* (the flange or a TCP of the tool) by *Step (mm)* along that frame's axes, and *Turn − / +* Rx, Ry, Rz turn it by *Turn step (deg)* about those axes through the jog point (the point stays put while it turns). The joints come from IK nearest the current pose. A step that leaves the reach or the joint limits, or would need a configuration change (a joint moving more than 30°, e.g. through a singularity), is refused with a message. The readout shows the jog point's pose in the chosen frame (Tool: in the world). Every jog step is checked against the floor, the part, the external axis bodies and the stay-out zones (the planner's pairs and clearances): a step that collides is refused and names what it hits; tick *Allow collisions* to jog anyway. After any move (also with the joint sliders, which are not refused) the readout warns **⚠ Collision** with the colliding pairs. |
| **Part** | A test part (flat plate, curved panel, open box, long rail, deep hole) or **Upload mesh (.stl/.obj/.ply)**: set *Upload units* first (**mm** by default); *Units* is the current part's unit. Meshes are drawn double-sided (never see-through from one side). Place it with the gizmo or the fields. |

| **Stay-out zones** | Volumes the **arm, the tool and the carrier** keep clear of, in the station planner, the feasibility check, the tool optimizer and the paths: **Add box**, **Add cylinder** or **Add STL/OBJ/PLY zone** (*Mesh units*, mm by default). Pick a zone (in the *Zone* list, or **click it in 3D**) to edit its *Name*, size, *Position* and *Rotation*, or tick *Drag with gizmo*; **Remove zone**. New zones start next to each other, not on top of each other. The zone under the gizmo ignores clicks, so the gizmo is easy to grab. The arm and tool keep the part clearance (10 mm by default) from them, the carrier the keep-out clearance. Drawn as see-through red volumes, saved in the project; changing them makes plans out of date. (The red floor areas of the External axis tab are different: floor keep-outs for the carrier only.) |
| **Reduced-speed zones** | Volumes where the robot only **moves slower**: they never block the arm, the tool or the carrier. The same editor as the stay-out zones (**Add box**, **Add cylinder**, **Add STL/OBJ/PLY zone**, *Name*, size, *Position*, *Rotation*, *Drag with gizmo*, **Remove zone**) plus **Speed (%)** (default 25 %): the share of the programmed speed inside the zone. A move whose **TCP** ends inside a zone takes 100 / speed times as long (overlapping zones: the slowest counts). This counts in the station planner's and the tool optimizer's **cycle-time estimate** (so *Cost per minute of cycle time* and the Pareto front's cycle time see it, and plans may move work out of slow zones) and in the cycle time of built paths (shown as *reduced*). Drawn yellow, saved in the project (projects where a stay-out zone had a *Reduced speed (%)* open with that zone here). Only the TCP position is tested, not the arm's links. |

### 2 External axis tab

Build the carrier (external axis) the robot rides on, mount the robot on it, and set where it may go. Only the settings of the chosen *Mode* and *Carrier* are shown (for example *Yaw range* and *Localization* only for a mobile base; for a mobile base *Travel* x / y only bounds the planner's search, jogging has no limits).

**Confirm setup** locks the assembly: the editing sections are hidden, the robot rides on the carrier, and the two move together (the *Carrier position* sliders or the jog arrows, left column → Jog panel). **Edit setup** unlocks it to change bodies, mount or settings.

| Section | What it does |
| --- | --- |
| **Axis type** | *Mode* **static** (one fixed base; the planner places one station) or **movable** (a carrier; the planner chooses its stops). *Carrier*: **rail** (travel *s* along the track frame's x axis), **gantry** (x/y carriage, robot usually upside down) or **mobile** (x, y, yaw on the floor). **Load carrier defaults** fills in a ready-made carrier of that kind. *Track/gantry frame*, *Travel*, *Yaw range* and *Lift range* (optional vertical axis) set the motion. With a lift, the planner chooses the **lift height of every stop** (like the travel position; seeded at 2 to 6 heights over the range) and the carrier move between two stops changes the height on the way. **Rail length (mm)** (rail) or **X rail length** and **Bridge length** (gantry) size the track, X rails and bridge in one setting: every body whose *Follows axis length* is set takes that length (a box along that axis, a cylinder lying along it as its height, centred where it was), and the travel follows: the carriage stops *End stop* (default 50 mm) before each end. The *Travel* fields then only show the computed limits. *Advanced* (folded): *End stop (mm)* and **Set travel limits manually** (the Travel fields become editable again; the bodies still follow the lengths). A length too short for the carriage and the end stops is refused with a message. The length fields appear only when some body follows a length, so older projects look and behave as before; mobile bases have none. |
| **Bodies** | The carrier's shapes. **Add box**, **Add cylinder**, or **Add STL/OBJ/PLY body** (choose *Mesh units* first; the file is stored in `<project>/assets/carriers/`). Pick a body in *Body* to edit its *Name*, size, *Position* and *Rotation*. *Follows axis length*: **x (rail / X rails)** or **y (gantry bridge)** makes the body take that axis length (see Axis type). *Attached to*: **carriage (moves)** for everything that travels with the axis (chassis, wheels, a lift mast), **lift stage (rises with the robot)** for the part of a lift that goes up and down with the robot (a platform or the moving column), **fixed (track / frame)** for parts that stay put (rail track, gantry beam), **X axis / gantry bridge (moves in x only)** for a gantry's bridge: it spans the X rails and travels along them, while the carriage runs along it in y (on a rail or a mobile base it simply moves with the carriage). With a lift, carriage bodies stay on the floor while the robot rises, also in the collision check. Tick **Drag body with gizmo** to place it in 3D. **Duplicate body** / **Remove body**. The selected body is drawn in blue. |
| **Robot mount** | *Mount on body* + **Snap robot to top of body** puts the robot base on the top centre of that body (keeps the yaw). Fine-tune with *Mount xyz / rpy* (robot base in the carriage frame) or **Drag mount with gizmo**. |
| **Standoff and safety** | *Part standoff min / preferred*: carrier-to-part distance; the minimum is hard, closer than preferred adds a cost. It is measured in 3D, so a carrier may sit partly under a raised part: model fixtures and stands as part of the mesh or as keep-outs. *Keep-outs*: floor rectangles `x0,y0,x1,y1; ...` in mm. *Localization* (mobile): position/heading error every stop must tolerate. **Mobile base: robot sides, fixed standoff, part sides.** Tick the **robot sides that may face the part**: *Front*, *Side 1 (left)*, *Side 2 (right)*, *Rear* (several allowed: the planner picks per stop; *Side 1 / Side 2* = the base runs parallel to the part). With any side ticked, every stop keeps one *Fixed standoff (mm)*, and you can restrict the **part sides it may work from** (*Part ±X / ±Y side*, in the part's own frame; none ticked = all). No side ticked = the free x / y / heading search with *Part standoff min / preferred*. **Rail / gantry: *Align track with the part*** turns the track along the part's long axis and centres it on the part (keeping its side and distance); it follows the part when you move it. (Older wording: — pick which side of the base turns towards the part (front = carriage +x, left = +y) and one gap between the base outline and the part outline (top view). Every stop then sits exactly at that gap with that side facing the part: the stops lie on one contour around the part (like a rail wrapped around it), which keeps one safety distance everywhere and makes planning much faster. Choose *free (search the heading)* for the older free x / y / heading search with *Part standoff min / preferred*.) |
| **Apply base settings** | Commits the axis-type and safety fields. Body and mount edits apply immediately. |
| **Carrier position** | In the left column → Jog panel → **Carrier position**: sliders move the carrier, and the robot with it. The readout gives the footprint-to-part distance; it and the red/amber standoff outlines follow the part when you move it. |

Every body collides with the arm and tool (from the upper arm on), and counts for standoff and keep-outs. Drawing: red line = minimum standoff, amber = preferred, red boxes = keep-outs.

### 3 Tool tab

| Section | What it does |
| --- | --- |
| **Library** | *Tool* picks a saved tool (`tools/*.yaml`). **Upload tool URDF (.zip)** imports a URDF plus meshes; `package://` paths are resolved. **Save to library** stores the tool and copies its geometry into `assets/tools/<name>/`. |
| **Mesh / STEP tool** | A tool from **one file**: **Upload tool mesh (STL/OBJ/PLY/STEP)** (*Mesh units* for STL / OBJ / PLY, mm by default; a STEP file carries its own unit and is meshed with CadQuery). The whole file becomes the tool head, sitting directly on the flange mount (a *pose* tool with every extension range at 0; widen the ranges if the tool optimizer may add a connector). A tool more than 10 m across is refused (a unit mistake). **STEP coordinate systems:** the model's **named coordinate systems** (datum CSYS features, e.g. *CSYS_TCP*) are listed. *TCP datum* + **Set TCP to datum** puts the TCP chosen in the TCPs folder on it (*Flip TCP z* turns it 180° about x: the TCP z must point **out of the tool, toward the part**); faces within *Contact faces within (mm)* of the TCP (default 5; 0 = none, for a non-contact tool), grown over the flat or smooth surface around them, become that TCP's **contact faces** (the probe's end face may touch the part). *Flange datum* + **Mount datum on flange** makes that coordinate system coincide with the robot flange (*Flip flange z* when its z points into the robot). Only named coordinate systems are listed; for a datum inside a sub-part of an assembly the sub-part's own placement is not applied (export single-part tools). |
| **Process** | *Type* applies a preset: scanning, sanding, grinding, gripping, spraying or custom. You can then edit the force, tilt cone, standoff and spin. Contact-free processes (scanning, spraying) clear all contact areas. |
| **Flange mount** | Places the tool on the flange with a gizmo or typed values. |
| **Extension** | *Type* **pose** (default for new tools): where the head sits relative to the flange mount, as **x, y, z** (mm) and **rx, ry, rz** (deg, roll-pitch-yaw), each with a *range* the optimizer may use (equal ends fix a value). The connector between mount and head is drawn and collision-checked as a **smooth curve** (it leaves the mount along its axis and arrives along the head's axis), not a straight cylinder; its *capsule radius* applies along the curve. Legacy *straight* / *bent* tools keep the old L, θ, d, roll ranges (θ and d for bent only); the sample tools are legacy. **Switching a legacy tool to pose keeps its shape**: its head pose becomes x y z / rx ry rz with every range fixed at that value; widen the ones the optimizer may change. rx / ry / rz are the W / P / R angles. **Rough placement** sliders set the current values of the free ones. **bent2** (v8): two bends: up *L*, turn by *θ*, a leg *d*, turn again by *θ2* and a second leg *d2* (an S- or U-shaped probe; *roll* turns it all about the flange axis), each with its range; it uses the shared flange map like pose tools. The **pose** type is the free-form (spline) shape: its connector is a smooth curve from the mount to wherever the head is placed. |
| **TCPs** | Add, rename or remove TCPs and place each with a gizmo. **Lock TCP position** is ticked by default: the TCP's xyz / rpy fields are read-only and its gizmo has no handles, so it cannot be moved by accident. Untick it to move or type the TCP pose. **Pick contact faces (click tool)**: tick it, then click the tool; one click selects the whole smooth face (*Region angle* sets how far it spreads), and clicking again removes it. **Whole-link contact** marks entire links. **Two tips / several TCPs** (v8): a tool can carry more than one TCP (e.g. a pad on each end); the station planner and the tool optimizer then use the reachability maps of all of them (a point counts as covered when any tip covers it) and the exact check chooses the tip per point among the TCPs its group allows (*Groups & tolerances*). The CSV export names each point's TCP. |

Sample tools: **orbital_sander** (bent extension range, pad as the contact link) and **scanner** (contact-free). Both are simple stand-in geometry. Switch a tool to *Type* **pose** to let the optimizer move and tilt its head.

### 4 Targets tab

| Section | What it does |
| --- | --- |
| **Select surface & points** | Pick a **Tool**; only its settings are shown. **Fill +** / **Fill −**: click a face to add / remove its smooth region (neighbouring faces may turn by at most the *Fill angle limit*). **Brush +** / **Brush −**: an orange-ringed **handle** appears on the part; drag it and every face within the *Brush radius* of its path is added / removed (it follows the surface even if you drag above it, and skips faces turned more than *Brush max surface angle* from the surface under it, so it stays off the back of thin parts). A click paints one dab and moves the handle there. **Box +** / **Box −**: drag a rectangle in the 3D view (the camera does not orbit while a Box tool is chosen). **By normal**: every face within an angle of a direction. **Full surface**: click a body of the part to add all of its faces. **Single point**: one target exactly where you click (normal from the surface). **Remove point**: deletes the target nearest the click, within the brush radius. **Point brush**: each click adds a dab of the *Part-anchored lattice* inside the brush radius. **Ctrl-click** turns + into − (and back) for the click tools. **Select all**, **Clear** and **Invert** act on the whole part; the selected face count and area are shown below. |
| **Generate points** | Pick a **Generator** (only its settings are shown); it runs on the selection. **Surface grid (rows/columns)** (as in the sand-optimizer): the grid lies across the selection's mean facing direction, rows run along the part X axis turned by *Row direction (deg)*, points along a row are *Column spacing* apart and rows *Row spacing* apart, centred on the selection; points closer than *Edge inset* to the selection's edge are left out. A strongly curved selection gets a warning (select one side at a time). **Each side separately** (on by default; grid, lattice, projected grid and raster lines with an automatic direction): a selection that covers several sides of the part, e.g. the top **and** bottom of a plate or a whole closed part, is split into sides that each face one way, and every side gets its own points (the info line says how many sides). Sides smaller than one grid cell (thin edge strips) are skipped. Points on opposite faces of a thin part are never treated as duplicates. **Part-anchored lattice**: rows / columns anchored at the part origin (the Point brush grid), rows along *Row direction (part frame)*. **Projected grid**, **Poisson (even)**, **Raster lines** and **Selection edges** as before (spacing in mm). Normals are *interpolated* (smooth) or *face*. Points that already exist are skipped, and **Undo last add** removes the last generate, click or dab. **Delete points on selected surface** and **Delete all points** sit right under *Generate on selection*. |
| **Refine part mesh** | **Tessellate part mesh** splits the part's triangles until no edge is longer than *Max edge length*. The shape does not change, but clicks, region growing, brushes and interpolated normals work on a finer mesh (useful for coarse CAD exports with long thin triangles). The refined mesh is saved as `<project>/assets/parts/<name>_refined_<N>mm.stl` and becomes the part; the selection and existing points are kept. |
| **Groups & tolerances** | Each point belongs to a group, and each group has spin, tilt, standoff and an allowed TCP list. **Apply to group** saves them. |
| **Edit points** | Assign the points on the selected surface to a group, or flip their normals. **Import CSV** reads `x,y,z,nx,ny,nz[,group]` in mm or m. |
| **Display** | Colour points by *group*, *feasibility*, or the selected solution's *zone* / *configuration*; show normals; **Highlight unreachable points** (on by default) marks every point the selected plan (or, without one, the feasibility check) cannot reach with a large red dot, a red cross and its id (the Log says why); **Preview tool at point** draws a see-through tool aimed into the surface at any point. **Colour part by** (v8) paints the part's surface around the points of the **selected plan** (Results; without a plan: the feasibility check) by a score, red (poor) → yellow → green (good), with a legend and the lowest / median value: **Clearance** (0 to 50 mm), **Manipulability** (% of the robot's best), **Joint-limit margin** (0 to 30°), **Tolerance reach rate** (the last Results → *Tolerance check*). **Reached / unreached (points)** (the default) shows no surface colour, as before. The colours follow the selected plan. |

Points are stored **in the part frame**, so moving the part moves them with it.

### 5 Optimize tab

**The optimizers need a confirmed setup.** *Plan stations*, *Quick plan* and *Optimize tool* stay greyed out until steps 1 to 4 are confirmed; a line under each button says what is missing, e.g. "Confirm **2 External axis** first" or "**3 Tool** changed since you confirmed it — confirm it again". The feasibility check and the base heatmap work at any time.

| Section | What it does |
| --- | --- |
| **Feasibility check** | **Check all targets** with clearance and joint margin as settings. *Check from* = **stops of the selected plan** (default): each point is checked from the stop the plan in Results assigns it to, with the carrier there (points the plan did not reach: every stop is tried); or **current robot position**: everything from where the robot stands now (without a plan this is used automatically). The carrier's own bodies count as obstacles. The summary gives counts and reasons. The button is greyed out while there is nothing to check, and a line under it says why (no robot, part, tool or targets yet). Inspecting the result is now under **6 Results → Inspect result**. |
| **Base placement (reachability)** | **Show base heatmap** draws where the base could stand on the floor (bright = all points reachable with the best worst-case manipulability) and marks the best spot. **Move base to best** moves it there. Ignores collisions: confirm with the feasibility check. |
| **Station planner** | *Mode*: **Cost based** (the default, below) or **Rules only** (see the next row). **Plan stations** finds the fewest stations that reach every point, within *Max stations* (up to 40; static mode = 1). It plans one station count after the other and stops once two more stops in a row gave no better plan, so a large *Max stations* does not mean that many full plans. With many points the stop search works on a spread-out sample (a quarter of the points, 60 to 150; all of them below 60), adds a stop where the sample missed points that need one, and then checks **every** point exactly, so the result is still exact (357 points on the rail: 43 s → 18 s, same plan). Each solution goes to the Results tab as soon as it finishes. *Candidate grid* sets the search resolution; *Refinement* chooses how the stops are fine-tuned against the exact checks (**CMA-ES**, the default, **Bayesian (TPE)**, **Bayesian (Gaussian process)** or **None**), and *Objective* what the planner optimizes (**Weighted cost**, **Pareto front (NSGA-II)** or **Robust (worst case)**); see [Optimization methods](#optimization-methods). **Connected station areas** (on by default) reassigns points after planning so each stop's points form **one connected area** on the part and no stop's points sit among another's: every stop grows outward from the point nearest its base, taking only points it reaches exactly, then stray points join the stop around them. It is kept only when it leaves fewer pieces than the planner's own split. Plans appear quickly: every station count is first verified at its first-guess stops (a few seconds each), then the most promising one or two are refined and replace their quick version when better. **Cancel** stops within about a second and keeps the plans already finished. *Joint configuration* = **zones (minimize changes)** (the default) groups each stop's points into zones that keep one arm configuration and puts each zone in path order (see *Joint-configuration zones* below; since v7 round 2 the zone search reuses checks, so the rail example plans in about 8 s and the mobile example in about 13 s, down from 52 s and 58 s). **best per point** lets each point take its own best configuration, which can mix all of them at a stop (the overhead gantry example had four per stop); **Costs & requirements** (folder): every cost weight can be edited: **Cost per stop** (each base / carrier position beyond the first, default 5), **Cost per group** (each extra joint configuration at a stop, with or without zones), **Cost of arm motion** (the mean of the largest joint move between neighbouring points of a stop, per 90°: plans whose arm swings a lot from point to point cost more), **Cost per wrist flip** (a joint 4 / joint 6 jump over 180° inside a group's path), per unreached point, carrier travel per m, tool length, manipulability, clearance, joint limits and preferred standoff. They are saved with the project; **Default costs** restores them. **Must be met**: **All points reachable** and **No configuration change within a group** (turns zoning on; a stop may hold several groups, each in one configuration, and a wrist flip inside a group breaks the rule). With a rule ticked the planner keeps adding stops until a plan meets it (up to *Max stations*); plans that break a ticked rule stay in the Results list marked **✗**, below the valid plans, and the summary names the rule. |
| **Station planner → Mode: Rules only** | No costs: the *Costs & requirements* folder is hidden and every rule must hold: 1. every target reached; 2. the tool orientation within each group's spin / tilt tolerance; 3. joint limits (with the margin); 4. no collision (with the clearance); 5. one joint configuration per group, with no wrist jump over 180° (zones are always on); 6. **manipulability at least the *Singularity limit*** (% of the robot's maximum, default 5 %, 0 to 50): arm poses closer to a singularity do not count. *Objective*: **Fewest stations** tries 1, 2, … stops and stops at the first count with a plan that meets every rule; there it also tries up to three other stop sets (leaving out one chosen stop each time) and keeps the one with the fewest groups. **Fewest groups** goes on with more stops while that gives fewer joint-configuration groups (up to *Max stations*, or until two more counts give nothing better). No fine-tuning (no cost to optimize). When no plan meets every rule, the closest one is listed, marked **✗**, and the status line and the log name each broken rule with its count, e.g. "rule 6: 12 point(s) only reachable close to a singularity at 4 stations". |
| **Quick station plan** | **Quick plan**: a first look in seconds with the Station planner's *Max stations*: no fine-tuning, no zones, no station areas, and it stops after the first station count that gives nothing better. Its plans are labelled **Quick:** in Results; the status says they are not optimized (run **Plan stations** for that). Same Cancel, progress and confirmed-setup rule as the planner; the log gives the run time. |
| **Compare robots** (v8) | Tick robots (the UR20, UR10e and UR30 by default; every robot in the registry is listed) and click **Compare robots**: the study is planned once per robot with everything else unchanged (tool, part, targets, carrier, stay-out zones, the station planner's settings and costs; each robot uses its own joint limits and home). The table lists per robot: stations, points reached, the estimated **cycle time**, the worst clearance, the lowest manipulability (% of that robot's best), the cost (✗ = a must-meet rule broken) and the run time. The current cell is not changed; **Use robot** switches the cell to the robot picked under *Robot* (then plan again). The first plan with a robot builds its reachability map (about a minute). |
| **Tool geometry** | **Optimize tool**. **Pose tools:** tries head poses within the ranges (CMA-ES, *Shape / pose evaluations* candidates), plans stations for each (quick plan) and keeps the pose with the **best overall cost** (reach, stops, travel, manipulability, clearance, joint limits, and connector length as one term); the winner is planned once more with refinement. The table lists every candidate (connector length, full coverage, reached, stops, cost). **Legacy straight / bent tools:** the shortest L that still reaches everything within the station limit, then θ / d / roll (coverage-vs-length table). *Length tolerance* (mm) is up to you (no upper limit). **Cancel** stops within about a second and keeps the best result so far. The **best 3 designs** are offered under *Design* (✔ = reaches every target, then fewest missed points, then cost), **also when no design reaches every target**; each one is added to the Results list so you can inspect it or build its paths. **Costs & requirements** (in the Tool geometry folder) are the tool optimizer's **own** cost weights, separate from the station planner's (they start as a copy of them and are saved with the project): raise *Tool length* to prefer short tools, *Cost per stop* to prefer designs needing fewer stops, and so on. *Must be met → All points reachable* offers only designs that reach every target; *No configuration change within a group* plans each design with zoning and rejects plans with a configuration change inside a group. **Manufacturable designs** (v8): *Tube OD (mm)* uses a standard round tube (12 to 80 mm) as the extension instead of the tool's own capsule radius (its collision radius is then half the OD), and *Bend angle steps* (**free**, **5°**, **15°**) lets the optimizer use only bend angles (θ, θ2, roll; for pose tools rx, ry, rz) in those steps: every candidate is snapped while searching, so the result is a design a tube bender can make (rounding afterwards could lose points). **Part family** (v8): *Part family (folders)* takes other project folders (separated by `;`); every tool candidate is then also planned on each of those projects' parts, targets and cells, and the cost is the **sum** or the **worst** (*Family cost*) over all the parts; a point missed on another part counts as unreached. Use it to design one tool for several part variants. *Shape search* (**CMA-ES**, **Bayesian (TPE)**, **Bayesian (Gaussian process)**), *Tool and stations* (**Stations fixed after length search** or **Alternate**) and *Objective* (**Weighted cost**, **Pareto front (NSGA-II)**, **Robust (worst case)**) choose how the tool is searched; see [Optimization methods](#optimization-methods). **Apply to tool** writes the chosen design into the Tool tab; the tool step then counts as changed, and **Confirm tool** (shown under it while step 3 is not confirmed) confirms step **3 Tool** right here, so the planners can run with the new tool. *Tool extension* (folder at the top of Tool geometry, closed) is a **copy of 3 Tool → Extension**: *Type*, the ranges the optimizer searches, *Capsule radius* and the *Rough placement* sliders. Both copies edit the same tool and always show the same values. An edit in the copy shows its own **Confirm tool** button under the fields, so you can confirm step 3 without leaving the Optimize tab. **Save optimized tool to library** stores the tool with the chosen design's values under *Save as tool* (default `<tool>_opt`) without changing the tool in use. |

**Cycle time** (folder in the Optimize tab, closed): the speeds of the robot program, used to estimate how long one part takes. *Joint speed* / *Joint acceleration* (joint moves; each joint is also capped by the robot's own speed limit, so the UR20's base and shoulder never go faster than 120°/s), *Process speed* (along the surface), *Approach speed* (lifts, approach and retract), *Linear acceleration*, *Carrier speed* / *Carrier acceleration* (rail, gantry axes, mobile base), *Lift speed*, and *Settle time per stop* (the carrier settles or localizes before the arm starts). Every move is a trapezoid (accelerate, cruise, brake; a triangle when it is too short to reach full speed); a joint move lasts as long as its slowest joint. Each plan gets an **estimated cycle time** (Results → summary card): the joint moves between neighbouring points of every stop, the carrier moves between stops and the settle time per stop. **Build paths** gives the cycle time of the actual paths, split into process, approach, joint, carrier and settle time. *Costs & requirements → Cost per minute of cycle time* (default 0, so plans cost what they did before) makes the planners prefer faster plans.

**Cable & safety** (folder in the Optimize tab, closed, v8): *Hose pack* models a hose or dress pack strapped to the arm. **none** (default); **clearance margin**: *Hose pack clearance (mm)* adds clearance around the arm from the upper arm on (upper arm, forearm, wrist) against the part and the stay-out zones; **hose along the arm**: the hose as a **shape** — a chain of round segments (*Hose diameter (mm)*, default 60) running from the upper arm to the wrist, *Hose offset from the arm (mm)* (default 150) from the line through the joints, on the *Hose side* (**+** or **−**) you choose. Each segment moves with its arm link; the hose must keep the part clearance from the part and the stay-out zones, stay above the floor and clear of the carrier (it is not checked against the arm or the tool it rides on). With *Show hose on the robot* it is drawn dark grey on the robot, so you can check its side and offset against your real dress pack. Unlike the margin, it only blocks poses where the hose itself would hit something, on its side of the arm. It is a layout model (straight hose per arm segment); the slack loop at the wrist is not modelled. **Joint 6 turn limit (deg)** (default 360°): at each stop, how far joint 6 may turn across the stop's points before the cables / hoses through the wrist wind up; beyond it the cost term *Cost per 90° of joint-6 wrap* (Costs & requirements, default 1) applies per 90°. **Reduced-speed zones** are under Robot & cell → *Reduced-speed zones* (see there).

### Optimization methods

Every optimizer step can use more than one method. Each dropdown below shows a one-line description of the chosen method under it; the **first** entry of each is the method the app used before v8 and stays the default, so old projects plan exactly as before.

| Where | Dropdown | Methods (first = default) |
| --- | --- | --- |
| Optimize → Station planner | *Refinement* | **CMA-ES** · Bayesian (TPE) · Bayesian (Gaussian process) · None |
| Optimize → Station planner (Mode: Cost based) | *Objective* | **Weighted cost** · Pareto front (NSGA-II) · Robust (worst case) |
| Optimize → Tool geometry | *Shape search* | **CMA-ES** · Bayesian (TPE) · Bayesian (Gaussian process) |
| Optimize → Tool geometry | *Start shape* | **Nominal** · From approach cones |
| Optimize → Tool geometry | *Tool and stations* | **Stations fixed after length search** · Alternate |
| Optimize → Tool geometry | *Objective* | **Weighted cost** · Pareto front (NSGA-II) · Robust (worst case) |
| Results → Paths & simulation | *Point order* | **Pattern order** · Least joint motion (GTSP) |

**CMA-ES** (Covariance Matrix Adaptation Evolution Strategy). Tries a small population of candidates (stop positions, or tool angles and offsets) around the current best, learns from which directions improved and moves and reshapes its search cloud. Robust and needs no tuning, but it uses many evaluations and finishes its last population even past the budget. Each evaluation is an exact check of the points, so this is what costs time.

**Bayesian (TPE)** (Tree-structured Parzen Estimator, from Optuna). Splits the candidates tried so far into good and bad ones, models where each kind lies and samples where good ones are likelier. Works with small budgets and does not mind rejected candidates. On the rail example's tool it found the same design as CMA-ES in 29 s instead of 39 s.

**Bayesian (Gaussian process)**. Fits a smooth model of the cost over the variables (with an estimate of how sure it is), then tries the point with the largest *expected improvement*: where the model is good or still unsure. The most sample-efficient method for one to four smooth variables (a station's position, a bend angle and offset); less suited to many variables. Our own implementation (no extra packages); candidates that are rejected outright (huge costs) are capped so they mark a bad region without spoiling the model.

**None** (Refinement). Keeps the stops the set cover and reachability map chose, verified exactly but not fine-tuned. The fastest; the plan is usually a little more expensive.

**Weighted cost** (Objective). The cost J from *Costs & requirements*: one number per plan, and the cheapest plan wins.

**Pareto front (NSGA-II)** (Objective). Instead of one best plan, the plans that **no other plan beats on every measure** you tick: *Pareto: Stations*, *Cycle time (s)*, *Worst clearance (mm)* and, for the tool optimizer, *Tool length (mm)* (defaults: stations, cycle time and worst clearance; tool: length, stations, cycle time). Example: a 2-stop plan that is slow and a 3-stop plan that is fast are both on the front, because each wins on one measure. The station planner plans as usual, then NSGA-II (a genetic algorithm for several objectives, from Optuna) moves the stops of each station count's plan within the refinement radius, and every candidate is verified exactly. The Results list then holds only the front, labelled **Pareto 1/n**, **Pareto 2/n**, …, and **Results → Pareto front** draws them as numbered dots over two measures of your choice (*X axis*, *Y axis*); **click a dot** to select that plan (or a *Select plan* button under the chart); the selected plan's dot is larger and orange. Plans that miss more points than the best one are never on the front. The tool optimizer offers the front of every design it evaluated.

**Robust (worst case)** (Objective). Scores each plan by how it does when the part or the stops are off: robust cost = cost + *Cost per unreached point* × the points lost under tolerance. The offsets are those of **Results → Tolerance check**; *Robust samples* (default 10) samples are checked per candidate, and *Robust measure* chooses **Worst case** (the most points lost in any sample) or **CVaR 20 %** (the average of the worst fifth of the samples, less sensitive to one unlucky sample). The station planner then searches stop positions that lower the robust cost (TPE) and lists the best plan of each station count, labelled **Robust:** (the cost breakdown shows the *robust* term). Much slower than weighted cost (each candidate is checked once per sample: about 2 min on the rail example instead of 8 s). It helps when the points sit near the edge of reach: on the rail with the part off by up to ±350 mm / 4°, the robust plan reached every point in 47 % of independent samples against 37 % for the weighted plan. It cannot help much when the offsets are so large that no stop position copes. The tool optimizer re-ranks its six cheapest designs by robust cost.

**Nominal** (Start shape). The shape search starts from the tool's nominal values (Tool tab).

**From approach cones** (Start shape). First finds, for every target, the directions from which a straight tool of the current length and radius can come in at all (its *approach cone*: rays along the tool's centre line and sides must not hit the part; within 75° of the surface normal). A target in a slot only has directions along the slot; one behind a wall only those leaning away from it. From the cones it derives a few bend angles: per target, how far the head must turn from the direction back toward the robot base to point along its free cone, clustered into up to three angles. The tool optimizer tries those bends (both signs, within the range; for pose tools on the head turn *rx* or *ry* with the widest range) before searching and starts the search from the best one. **Targets → Display → Show approach cones** draws each target's free directions as short blue lines and names any point with none (no straight tool reaches it).

**Stations fixed after length search** (Tool and stations). The tool optimizer finds the shortest length, plans the stations once for it, then tunes the shape (angle, offset, roll) for those stations.

**Alternate** (Tool and stations). After that, plans the stations again for the best shape, tunes the shape for the new stations, and repeats until a round improves the cost by less than 1 % (at most 5 rounds). Slower; never worse, since the best design seen is kept. Pose tools already plan the stations for every candidate, so this applies to straight and bent tools.

**Pattern order** (Point order). The path pattern of *Paths & simulation* (raster, serpentine, …) orders the points along the surface.

**Least joint motion (GTSP)** (Point order). Chooses the order of the points **and the arm configuration at each point** together so the arm moves least: every point's TCP pose is solved again for all IK branches (inside the joint limits, not singular, not colliding), and the route picks one solution per point so the sum of the joint move times between neighbours (as for the cycle time) is smallest; switching configuration between two points also costs the back-off and approach time. The route starts where the pattern starts, so groups still chain corner to corner. **Zoned plans keep their zones** (each zone's points stay in the zone's configuration, so only the order changes); plans made with *Joint configuration = best per point* are solved per stop over all its points and may change configurations, then split into one group per configuration. Solved as a generalized travelling-salesman path with OR-Tools (up to 1200 point solutions per problem) and kept only when it beats the planner's order and solutions. Moves between points that are not surface neighbours lift off the surface. Measured on unzoned plans: rail example cycle time 128 s → 83 s (reconfigurations 4 → 0), mobile example 126 s → 114 s (2 → 0), overhead gantry 467 s → 339 s (19 → 4, path problems 53 → 1).

### Joint-configuration zones

A 6-axis arm (UR or industrial OPW type) can reach most poses in up to 8 ways: **shoulder** (S), **elbow** (E) and **wrist** (W) each have two branches, written as a label like `S-E+W-`. Moving from one branch to another needs a big reconfiguration motion, which you rarely want in the middle of a sanding pass. With *Joint configuration* = **zones (minimize changes)**, the planner:

1. checks, for every point, which configurations can do it from its stop;
2. picks the **fewest configurations** that cover the stop's points (exact), each point using its best solution in one of them; a stop can therefore have more than one **zone**;
3. orders each zone's points as a serpentine **path** and picks joint values (±360° copies of the wrist joints) that keep the path continuous;
4. counts **configuration changes** = extra zones at a stop + remaining jumps of joint 4 or 6 above 180° between neighbouring points, and adds *Cost per config change* × changes to the cost J. Because the planner also offers plans with more stops, a plan with one more stop and no changes can win.

It also re-runs the localization checks for the chosen solutions. It used to be 3 to 6× slower than *best per point*; it now skips arm poses of configurations already found and reuses checks within a plan, so it costs little extra. When the points a stop does in its most common configuration leave only a few others, and those can join it, only those few are searched again. The Results tab lists the zones (`S1·Z1 S-E+W- 6 pts`), draws each zone's path as a thick coloured line, and **Targets → Display → Color points by** `zone` or `configuration` colours the points of the selected solution. The CSV and the HTML report list zone, path position and configuration per point. The Tool geometry optimizer uses the same setting.

### 6 Results tab

- **Summary card** (top): the selected plan at a glance: stations, joint-configuration groups, points reached; the worst clearance, the closest joint limit and the lowest manipulability (in % of the robot's maximum after a plan in this session; after reopening a project, as |det J|); which planner made it (cost based, quick, rules only and its objective, tool optimizer), whether the rules are met, and the run time.
- **Inspect result** (folder, moved here from Optimize): *Show* = the feasibility check, or (the default) the **plan selected above**; it switches to each new plan or tool result automatically. **Move robot to this solution** moves the carrier (or base) to the point's station and the arm to its solution; **◀ previous point / next point ▶** steps through the points and moves the robot to each one at once.
- **Solution**: the ranked list (best cost first); **◀ previous / next ▶** switches between the plans (with *Move robot with the plan* ticked the robot goes to the new plan's first stop), and a table compares all plans (stops, points reached, cost; ⚠ = out of date). The feasibility check, *Inspect result*, *Move robot to station* and the paths all use the selected plan; paths built for another plan are cleared. It shows a summary (with the zones and configuration changes when zoning is on), the cost breakdown bars and the station list.
- **Scene**: coloured station markers, **lines to served points**, **ghost robots** posed at a real solution, and the **station areas on the part**: the part surface painted in each stop's colour (the colour of its lines), a stop's second and third groups in lighter shades. Without zones, each joint configuration a stop uses is its own group (the most used configuration in the stop's colour, the others lighter), so you can see where the arm has to change configuration. A coarse part mesh is subdivided for this display only. If a stop's area is still in pieces after planning, the summary says so (⚠).
- **Move robot to station**: moves the carrier to the station's axis values (the robot base rides on it; static: the base itself) and the arm to the first point that station serves.
- **Paths & simulation** (folder, open, above Export; the planner and the tool optimizer point to it when they finish): **Build paths** (greyed out, with the reason below it, until there is a plan) connects the selected solution's points **stop by stop**: every point of a stop is done before the carrier moves on. Within a stop each group (zone, one joint configuration) is ordered by the **Pattern** (as in the sand-optimizer): **Zigzag** (rows back and forth), **Unidirectional** (every row the same way), **Cross-hatch** (zigzag, then again at +90°), **Concentric** (rings from the border, *Outside → in* or inside out), **Spiral**, **Hilbert** (space-filling curve), **Shortest path (TSP)**, or **Planner order** (the planner's own order). Rows run at *Raster angle* from the part X axis and start at the *Start corner* (`min-min` = lowest row, low end); *Row tolerance* (0 = half the point spacing) decides which points share a row or ring. Only the chosen pattern's settings are shown. Groups follow each other nearest first, and each later group starts at the corner nearest where the previous one ended. *Point order*: **Pattern order** (the pattern above) or **Least joint motion (GTSP)** (each group re-ordered so the arm moves least; see [Optimization methods](#optimization-methods)). **Prevent crossing paths**: a move that would cross an earlier move of the same pass, or **run over another target point** (within 0.4 × the point spacing; such a point would be sanded twice), lifts off the surface by *Lift height* instead, so each point is sanded once. **Home between stops** (on by default): each stop starts with a joint move from the robot's home pose and ends with one back to home. **Drive carrier between stops** (on by default, independent of home): the mobile base or external axis move between stops is planned and played back; with home on the arm rides at home, with home off the tool first backs 150 mm straight off its last point and the arm rides in that pose, then a planned joint move takes it to the next stop's first point. If that backed-off pose would hit something on the way, the arm goes to its home pose for the drive instead, and a direct joint move that cannot get around an obstacle goes via home when that is clear. When no way is found to a point, the planner also tries the same arm pose with joints a full turn apart (e.g. the shoulder at +14° instead of −346°), which can avoid a long swing through the carrier. Untick it to plan the stops without carrier moves. The carrier moves along a rail along its travel, a mobile base along its contour path when both stops use the same robot side, otherwise in a straight line), and every carrier pose on the way is checked against keep-outs, the part standoff and stay-out zones. When that move is blocked, a **mobile base drives around** the stay-out zones, keep-outs and the part (a route planned on the floor, turning where needed) and keeps its **standoff distance** from the stay-out zones on the way (the *Fixed standoff* on a contour, else the minimum part standoff, at least the keep-out clearance; close to a stop the keep-out clearance the stop was planned with is enough). Rails and gantries cannot drive around: a blocked move is flagged `carrier_blocked`. With a **lift**, the lift moves only while the carrier stands: at the stop it lowers to the travel height (the lowest height, or the first of the two stops' heights and the lift levels at which the drive, the lift moves and the arm at home are all clear), the carrier drives at that height, and at the next stop the lift rises to that stop's height before the arm leaves home. The home pose is the one set under Robot & cell → *Home position*. **Moves to and from home are collision-checked and planned** against the part, the floor, stay-out zones and the carrier's own bodies: each stop starts with the joint values nearest home (the same arm pose, without a joint turning the long way round). A straight joint move, or one joint group at a time (arm, base, wrist), is used when it is clear. Otherwise the tool first backs straight off the surface (150 mm along its axis, same configuration) and the arm finds a way around. Only when no way is found is the straight move shown, with its collisions flagged. If the **home pose itself collides** at a stop (e.g. a robot hanging upside down from a gantry whose default home points down into the part), the summary says so: set a home that is clear there (the *gantry_overhead* example saves a folded home, 0, −165, −150, −180, −90, 0). **A move along the surface where the arm or tool would collide** (typically a long move across the part that swings the upper arm or forearm into it, the carrier or a stay-out zone) **is replaced by a planned joint move**: the tool backs off, the arm goes around, and comes in again at the next point (grey, like the other joint moves). Otherwise, between two points on the surface the TCP follows the part (sampled every *Path sample step*, projected onto the part, standoff kept) and turns smoothly from one planned tool orientation to the next; every sample is solved for the arm in the same joint configuration, closest to the previous sample. The summary lists flagged samples (red dots in the scene):
  - **unreachable** (no solution within the joint limits), **config flip** (only another shoulder / elbow / wrist configuration reaches it), **jump** (a joint moves more than *Max joint step* between neighbouring samples), **collision**;
  - **reconfigure**: the plan itself switches configuration there. The arm does not flip next to the part: it goes back to home and comes in again in the new configuration, with the same collision-free planned moves (directly, without home, when *Home between stops* is off). Without zones, the path does all of a stop's points in one configuration before the next (most used first), so a stop changes configuration once per extra configuration, not each time neighbouring points differ. Each **target group** (face) is a path of its own too: the tool is not dragged over an edge of the part from one face to the next (a planned joint move joins them), and a zone's faces are done one after the other before the next zone. **Fewer changes:** set *Joint configuration* = **zones (minimize changes)** in the Station planner (the default; it picks the fewest configurations per stop that still reach every point, and *Cost per group* makes the planner prefer stops that need fewer), or tick *No configuration change within a group* under *Costs & requirements* to require it.
  - **carrier blocked**: a carrier pose between two stops hits a keep-out, the part standoff or a stay-out zone.
  - **motion** (with *Check moves between points*, on by default): both ends of a move are clear, but the arm hits something on the way between them (it sweeps through the part, a stay-out zone or the carrier). Every move is checked every 2° of joint motion; the flag sits on the move's end sample. The surface moves are straight TCP lines in the robot program; between samples a few millimetres apart the check stands for them too. Carrier moves are checked when they are planned (carrier blocked).

  The paths are drawn as coloured lines (one colour per zone); lifts, home moves and carrier moves are thin grey lines. **▶ Play** / **⏸ Pause** moves the arm along the samples and the carrier between the stops at *Speed (mm/s)*; *Sample* scrubs; **⏮ Back to start** stops and returns to the first sample (Play at the end also starts again from the beginning). **Clear paths** removes the built paths (lines, issue dots and playback) until you build them again. **Export path CSV** writes `export/paths.csv` (stop, kind = surface / lift / home / carrier, the carrier values, zone, from / to point (0 = home), TCP pose in the world in mm + quaternion, joints in degrees, issue per sample). It is a preview of the motion, not a timed robot program.
- **Pareto front** (folder, shown when the list holds a Pareto front): the plans as numbered dots over two measures (*X axis*, *Y axis*: stations, cycle time, worst clearance, tool length); the numbers are the **Pareto i/n** plans in the list. **Click a dot** (or a *Select plan* button) to select that plan. See [Optimization methods](#optimization-methods).
- **Tolerance check** (folder, closed): does the selected plan survive fixture and positioning error? It moves the part by a random offset (up to *Part offset ± (mm)* in each axis) and turn (up to *Part tilt up to (deg)*, about the part's centre), and each stop by up to *Stop offset ± (mm)* in x and y (carrier positioning error), *Samples* times (the first sample is the nominal setup; *Seed*: the same seed gives the same samples), and checks every planned point again, exactly, from its own stop. The points move with the part, as when the program is taught in a user frame on the part: the question is whether the arm can still *reach* them. **Run tolerance check** (greyed out until a plan is selected; runs in the worker process and the check workers like the planner; **Cancel**) reports the share of samples in which **every** point was reached and lists the **least robust points**: how often each was reached, **fails from**: how far that point had moved in the smallest sample where it failed, its clearance and its joint-limit margin. Points the plan itself does not reach are not tested. The result can colour the part (*Colour part by → Tolerance reach rate*, Targets tab). Typical values: ±1 to ±5 mm and up to 1° for a fixture, ±10 mm for a mobile base (the planner's localization check already tests the mobile tolerance set on the External axis tab).
- **Safety** (folder, closed, v8): **Show reach envelope** draws, for every stop of the selected plan, a see-through sphere around the robot's shoulder with the robot's reach plus the tool as radius: everything the arm and tool could touch from that stop. Below it a **suggested fence**: the envelopes' floor area grown by *Fence distance (mm)* (default 500), drawn on the floor in orange, with its size and corner coordinates. It is a layout aid, not a safety assessment: the risk assessment (ISO 10218-2, ISO 13855 separation distances) decides the real guarding.
- **Export** (greyed out, with the reason, until there is a result):
  - **TCP poses (CSV)**
  - **Userframes (CSV)**: one row per stop: its carrier values and the **user frame**, the pose of the part frame in the robot's base frame at that stop (teach the points in it and they stay valid). The columns follow the robot model: UR arms `x_mm, y_mm, z_mm` + rotation vector `rx_rad, ry_rad, rz_rad` (as in a UR pose `p[...]`, with x y z converted to m); FANUC-style arms `X Y Z` (mm) + `W P R` (deg); every robot also gets the rotation matrix `r11 … r33`. The format comes from `frame_convention` in the robot's `spec.yaml` (`ur`, `fanuc` or `matrix`).
  - **Optimized tool URDF (zip)**
  - **HTML report**: self-contained, printable to PDF, with an optional 3D snapshot. Since v8 it opens with the customer sections: the **Summary** card, the **Cycle time** (the plan's estimate, and the built paths' when they belong to this plan), the **Tolerance check** of this plan (when one was run) and the **Unreached points** with their reasons.
  - **Report (PDF)** (v8): the same report printed to `export/report.pdf` by Edge headless (Chrome works too); without a browser the HTML report is still written.
  - **Tool (STEP)** (v8): the selected result's tool (its optimized values, else the nominal) as a STEP file for CAD: the extension as round tube legs (the *Tube OD* chosen under Tool geometry, else the tool's radius) with a ball at each bend, plus an **ISO 9409-1 flange plate** at the mount (*Flange*: 50-4-M6 for UR arms, 63-4-M6, 80-6-M8; 8 mm thick, with the bolt holes and the centring bore). The head is not in it (use its own CAD). Made with CadQuery, which is part of the environment and the portable copy.

  Files go to `<project folder>/export/` and also download in the browser.

---

## 5. Worked examples

### Example A: sand a flat plate from one base (static)

1. Cell → *Part* `flat_plate`, xyz `900, 0, 300` mm.
2. Targets → *Tool* `Fill +` → click the plate top → *Row* / *Column spacing* `100` → **Generate on selection**.
3. Optimize → **Check all targets**. Everything should be reachable from the origin.
4. Optimize → **Plan stations** → Results: "1 floor station: N/N reached".
5. Optimize → **Optimize tool**. On this plate the shortest straight tool is the bottom of the range (50 mm), and the bend stage may lower the cost further.
6. Results → **HTML report**.

The same job without the GUI is in `examples/headless_plan.py` (section 7).

### Example B: a long rail part with a 7th-axis rail carrier

Shortcut: **Project** → *Examples* `rail_long_part` → **Open example**, then step 4.

1. Cell → *Part* `long_rail`, xyz `0, 800, 500`.
2. Targets → *Select by normal* `(0,0,1)` within `5` → *Spacing* `250` → **Generate**.
3. External Axis → Carrier `rail` → **Load carrier defaults**. Set *Track/gantry frame xyz* to `0, 150, 0`, so the 0.5 m carriage sits exactly 300 mm from the part. Optionally add a wall keep-out `-5000,1300,5000,3000`. Click **Apply base settings**.
4. Optimize → *Max stations* `5` → **Plan stations**. Expect 2 rail stations for the top face; the Results tab shows the carriage positions *s*.
5. Try Carrier `mobile` with its defaults on the same targets and compare. Both need 2 stops on the top face. On the wall-side face (generate on the face pointing to the wall) the rail needs 4 and the mobile base 5.

### Example B2: design a mobile robot from simple shapes

Shortcut: **Project** → *Examples* `mobile_amr` → **Open example** (four wheels, chassis, battery box and a column with the UR20 on top).

1. External Axis → Carrier `mobile` → **Load carrier defaults** (a chassis and a riser).
2. Select *Body* `chassis`, set *Box L W H* to `1100, 700, 300` and *Position* to `0, 0, 100` (raised for wheels).
3. **Add cylinder** → *Name* `wheel_fl`, radius/height `100, 80`, *Rotation* `90, 0, 0` (axle along y), *Position* `450, 390, 100`. **Duplicate body** and move the copies to `450,-310,100`, `-450,390,100`, `-450,-310,100`.
4. **Add box** → a battery or cabinet, or **Add STL/OBJ/PLY body** for a CAD model of your AMR (set *Mesh units* first).
5. *Mount on body* `riser` (or your column) → **Snap robot to top of body**.
6. **Apply base settings**, then Optimize → **Plan stations**.

### Example B3: avoid joint-configuration changes on the rail

1. **Project** → *Examples* `rail_long_part` → **Open example**. Targets: select the face pointing to the wall (y+) and generate points with *Spacing* `250`.
2. Optimize → *Station planner*: *Joint configuration* `best per point` → **Plan stations**. Targets → *Color points by* `configuration`: stops can mix two configurations, meaning the arm would reconfigure mid-pass (in the test version of this task every stop did).
3. Switch to `zones (minimize changes)` → **Plan stations** again. The Results summary now lists the zones and the number of configuration changes (0 in the test version of this task, at almost the same cost), and the zone paths are drawn in the scene.

### Example B4: tall wall with a gantry, floor X-Y axes or a mobile lift

The 3 m tall wall (`tall_wall`, 2 × 0.3 m) is out of reach from one fixed robot height. Three examples work it with different carriers (orbital sander, 250 mm grid):

* `gantry_overhead`: the UR20 hangs upside down from a carriage on an overhead X-Y gantry: two fixed beams 4.4 m up on four posts, a bridge between them that moves in x, and the carriage running along the bridge in y. Targets: the top face and both long faces from 1.8 m up; the gantry's y axis takes the robot over to either side.
* `floor_xy_rails`: a floor X-Y gantry: two fixed X rails along the wall, a bridge across them that moves in x, and a carriage running along the bridge in y (towards the wall); the UR20 stands on a riser on the carriage. Targets: the front face and the end face from 0.25 m to 1.8 m (the sander cannot work right at the floor).
* `mobile_lift`: the mobile robot of `mobile_amr` with a 1 m lift (a *mast* on the chassis and a *lift_stage* that rises with the robot). Targets: the front face from 0.5 m to the top; every stop gets its own lift height, so the robot works the bottom low and the top raised.

Open one, then Optimize → **Plan stations**; Results lists each stop's carrier values (with the lift height for `mobile_lift`).

### Example C: the shortest probe for a deep hole

1. Cell → *Part* `deep_hole`, xyz `800, 0, 0` (100 mm hole, 300 mm deep).
2. Tool Setup → upload or define a slim straight tool. The test suite uses a probe with capsule radius 20 mm and no head; see `tests/test_tool_geometry.py::_probe`.
3. Targets → *Tool* `Single point` → click the hole bottom; group standoff `50` mm, tilt `0`.
4. Optimize → **Optimize tool**: L ≈ **260 mm** = depth − standoff + clearance (the wrist cannot enter the hole).

### Example D: your own robot cell

1. **Part**: Cell → **Upload mesh**, choose *Units* `mm` if your CAD exports in mm.
2. **Tool**: Tool Setup → **Upload tool URDF (.zip)** (URDF + meshes), or **Upload tool mesh (STL/OBJ/PLY/STEP)** (one file; a STEP file's coordinate systems can place the TCP and the mount). Set the Process, place the TCP(s), pick the contact faces, set the extension ranges → **Save to library**.
3. **Targets**: if the STL has long thin triangles, **Tessellate part mesh** first. Then click-select the faces to process and generate, brush the areas you want with *brush points*, add single points, or **Import CSV** from a CAD/CAM tool (`x,y,z,nx,ny,nz[,group]`).
4. **Carrier** (optional): External Axis → build the rail or mobile robot from shapes or an STL and snap the robot onto it.
5. Then the same steps as Example A. Save the project so it can be reloaded later.

### Adding a robot

Robots live in `assets/robots/<name>/`: the URDF with its meshes and a `spec.yaml`. Every robot in that folder appears in **Robot & cell → Robot**. To add one:

1. Make a folder `assets/robots/<name>/` with the URDF (meshes next to it, relative paths) and a `spec.yaml`. Copy `assets/robots/generic_6r/spec.yaml` as a start.
2. Fill in `spec.yaml`: `name`, `urdf`, `base_link`, `flange_link` (z out of the flange), `reach_m`, `home_deg` (the arm pose while the carrier moves), optionally `limits_deg` and `frame_convention` (`ur`, `fanuc` or `matrix`, for the userframe export).
3. Choose the IK solver with `ik`:
   - `ur_analytic`: Universal Robots arms; add the `dh:` table (d1, a2, a3, d4, d5, d6, as in UR's published DH parameters).
   - `opw`: most industrial 6-axis arms with a spherical wrist (FANUC, ABB, KUKA, Yaskawa, Stäubli...); add the `opw:` parameters `a1, a2, b, c1, c2, c3, c4`, `offsets` and `sign_corrections`. They are the same numbers as ROS-Industrial's `opw_kinematics` parameter files, which exist for many arms.
   - `numeric`: any chain. Display, jogging and the feasibility check work; the station planner, tool optimizer and heatmap do not (they need the fast batch solver).
4. Run `python scripts/check_robot.py <name>`. It loads the files, checks the IK against the URDF on 500 random poses (wrong OPW parameters fail here with the size of the error), and checks the base joint range, reach and home pose. Fix anything marked FAIL.

The first plan with a new robot builds its reachability map (about a minute per tool), which is then cached.

---

### Checking a plan in URSim

`scripts/ursim_check.py` (v8) runs an exported plan in **URSim**, Universal Robots' simulator (e.g. the `universalrobots/ursim_e-series` Docker image with ports 29999 and 30002 published). Export **TCP poses (CSV)**, then:

```
python scripts/ursim_check.py <project>/export/tcp_poses.csv --host 127.0.0.1
python scripts/ursim_check.py <project>/export/tcp_poses.csv --dry-run      # only write the programs
```

It writes one URScript program per stop (`export/ursim/stop_<k>.script`: a `movej` to every reached point in the planned order, with the planned joint values), sends each to URSim, waits until it has stopped and reads the safety status: a protective stop (joint limit, URSim's own self-collision model, a singularity on the way) fails that stop. `--stop k` runs one stop; `--speed` / `--acc` set the `movej` speed and acceleration. Carrier moves between stops are not part of it: place URSim's base per stop yourself. It needs URSim running; the dry run needs nothing.

## 6. Files, folders and formats

| Path | Contents |
| --- | --- |
| `app.py` | GUI entry point (`--port`, `--host`, `--dark`, `--guide-port`, `--open <folder>`; `--fresh` is accepted and ignored) |
| `scripts/run_app.bat` | Windows launcher using the `rso` env |
| `core/` | kinematics, IK, collision, tool model, targets, feasibility, carriers, exports (no GUI code) |
| `opt/` | reachability maps, station planner, tool-geometry optimizer, cost |
| `ui/` | the Viser GUI tabs |
| `assets/robots/ur20/` | UR20 URDF, meshes, `spec.yaml` (DH parameters, home, reach), licences |
| `assets/robots/generic_6r/` | a made-up industrial arm (OPW kinematics), generated by `scripts/make_generic_robot.py`; a template for adding robots |
| `assets/tools/`, `tools/*.yaml` | sample tool geometry and the tool library |
| `assets/parts/` | test parts (STL) |
| `examples/` | example projects (`rail_long_part/`, `mobile_amr/`, `gantry_overhead/`, `floor_xy_rails/`, `mobile_lift/`, regenerate with `scripts/make_examples.py`), `sander_mobile.json` (format sample) and `headless_plan.py` |
| `workspace/` (not in git) | your project folders: `project.json`, uploads, `export/` |
| `cache/` (not in git) | reachability maps |
| `docs/USER_GUIDE.md` / `.html` | this guide; the HTML is generated from the Markdown with `scripts/build_guide.py` |
| `docs/PROJECT_PLAN.md` | the design plan |

**Project file** (`project.json`): the robot and base pose, base mode and carrier (its `bodies`: shape, size or mesh file, pose, attachment), tool, part, targets and groups, optimizer settings, and the saved solutions. Older projects with `chassis` / `riser` / `track` still load; the External Axis tab converts them to bodies. The `work` section holds where you left off: joint pose, selection, per-point joint solutions of the feasibility check and of each plan, and panel settings. Saves are atomic (written to a temporary file, then renamed). `workspace/last_project.txt` records the recent project folders (newest first) for the start-up window.

**TCP poses CSV** has one row per point, in station order:
- `point_id, group, status, reason, station, zone, seq, config, carrier` (zone / seq = path position / config label when zoning is on; rows then follow station → zone → path order);
- base pose (`base_x/y/z_mm, base_yaw_deg`);
- TCP in **world** (`tcp_world_x/y/z_mm`, quaternion `qw,qx,qy,qz`) and in the **robot base frame** (`tcp_base_*_mm`, `roll/pitch/yaw_deg`): use the base-frame values to program the robot at that station;
- joints `q1..q6_deg`;
- `spin_deg, tilt_deg, standoff_mm, clearance_mm, manipulability, limit_margin_deg`.

Unreached points come last.

**Optimized tool URDF** (zip): `flange → mount → extension_k → head links → tcp_<name>`. The root link `flange` attaches to the robot's `tool0`. The extension is a cylinder with sphere caps. Meshes are included.

**HTML report**: KPIs, cost breakdown, stations, a dimensioned tool sketch, coverage vs tool length (when it came from the tool optimizer), the point table, settings and notes. It is a single file with no internet access, and the browser can print it to PDF.

---

## 7. Using it from Python (headless)

Every computation lives in `core/` and `opt/` and works without the GUI. `examples/headless_plan.py` is a complete, tested script:

```bash
$PY examples/headless_plan.py workspace/headless_example
# 77 targets
# feasibility from the origin: {'reachable': 77, 'marginal': 0, 'unreachable': 0}
# plan: 1 floor station: 77/77 reached ...
# tool: minimal L = 50.0 mm ...
# wrote ['report.html', 'tcp_poses.csv', 'tool']
```

The essential calls:

```python
robot = Robot.from_registry("ur20")
ee = load_tool("orbital_sander")
tool = ToolModel(ee, ROOT / ee.urdf)
scene = CollisionScene(robot, tool, part_mesh, T_world_base, T_world_part)
checker = FeasibilityChecker(robot, make_ik(robot), tool, scene, T_world_base, T_world_part)
results = checker.check_all(project_targets)  # per-point feasibility
rmap = map_for_tool(robot, tool.T_flange_tcp("pad"))  # cached reachability map
plans = plan_stations(
    rmap, checker, targets, T_world_base, footprint, Weights(), PlannerSettings(max_stations=3)
)
geo = optimize_tool(robot, ee, urdf, part_mesh, T_world_part, T_world_base, targets, footprint, Weights(), 1)
```

For carriers, pass `sites=CarrierSites(Carrier(base), part_mesh, T_world_part)` to `plan_stations` (a carrier with mesh bodies resolves relative paths against the repo; pass `Carrier(base, resolve=fn)` to use another folder). `scripts/make_examples.py` shows how to build a carrier from bodies in code. Build the collision scene with the carrier bodies: `bb, wb = sites.scene_bodies()`; `CollisionScene(..., base_bodies=bb, world_bodies=wb)`.

---

## 8. Troubleshooting and limitations

| Symptom | Fix |
| --- | --- |
| `conda env create` fails or `import pinocchio` errors | Use Miniforge (conda-forge), not pip. Keep the OpenBLAS line in `environment.yml`. |
| Python crashes silently on matrix maths | The MKL BLAS problem: run `conda install -n rso "libblas=*=*openblas"`. |
| The first heatmap or plan takes ~1 min | The reachability map is being built (once per tool shape); later runs are fast. |
| "Software WebGL rendering" banner | The browser has no GPU acceleration; enable hardware acceleration in its settings. |
| Port already in use | `app.py --port 8090`. Viser may also pick the next free port by itself; check the console output. |
| **Open user guide** does nothing or shows an error page | The guide server uses the app's port + 1; the console prints its address at start-up. If that port is blocked (firewall when opening from another PC), use **Download user guide (HTML)** or open `docs/USER_GUIDE.html` directly. |
| The app opens an empty cell at start | That is the start-up state: pick a project in the **Open a project** window (also *Project → Load*), or start with `--open <folder>`. |
| The planner puts a mobile base under a raised part | Standoff is 3D; add the part's stand to the mesh or draw a keep-out. |

**Known limitations (v1)**
- Planning features (heatmap, station planner, tool optimizer) need an **analytic IK solver** (`ur_analytic` or `opw`) and a full-turn base joint. Robots with numeric IK (uploads) can be displayed, jogged and feasibility-checked. See [Adding a robot](#adding-a-robot).
- No payload, torque or tip-over checks (removed in v7 to keep the tool robot-agnostic): check the tool mass and the mobile base's stability with the manufacturer's tools. Older projects that still carry those settings load; the fields are dropped.
- Collisions are checked at each point only; the motion **between** points and stations is not checked (zones give an ordered path with continuous joints, but the path itself is not collision-checked). Travel distance ignores obstacles.
- Zoning works per stop with the stops the planner chose; it does not move points between stops to save a configuration change. The plans with other stop counts are the trade-off it offers.
- The part footprint for standoffs is a convex hull, which is conservative.
- The **brush** works click by click (one dab per click): Viser reports clicks and box drags, not freehand mouse strokes. For big areas, select them and use **Fill selection with rows/columns**.
- Mesh bodies on a carrier are used as-is for collision; an STL with millions of triangles slows the checks. Simplify it in CAD first.
- **Not yet confirmed in a real browser** (only in automated and headless tests): gizmo dragging (parts, carrier bodies, robot mount), contact-face clicking, shift/ctrl modifiers, box select, brush and single-point clicks on the part, file upload (meshes, carrier STLs) and download. Please report anything odd.

---

## 9. For developers

```bash
PY=C:/Users/<you>/miniforge3/envs/rso/python.exe
RUFF=C:/Users/<you>/miniforge3/envs/rso/Library/bin/ruff.exe
$PY -m pytest -q                         # all tests (one process; can run out of memory)
$PY scripts/test_batches.py              # the same, 4 files per process (~60 min)
$PY scripts/test_batches.py --fast       # skip the slow tests (plans, tool optimizer, paths)
$PY scripts/test_batches.py --e2e        # browser tests in headless Edge (tests/e2e, ~1 min)
$RUFF check . && $RUFF format --check .   # lint + format
```

- **CI** (v8): `.github/workflows/tests.yml` runs lint, the guide check and `scripts/test_batches.py --fast` on Windows for every push and pull request (conda env from `environment.yml`, reachability maps cached). The slow and browser tests run locally.
- **Reproducible results** (v8): every plan and tool result records where it came from: the code version (git commit, `+dirty` with uncommitted changes), the methods, a hash of all planner settings and weights, and the seed (all searches are seeded, so the same settings on the same code give the same plan). The Results summary card shows it on its last line; the project file keeps it.
- **Speed history** (v8): `$PY scripts/profile_plan.py <example> --record` times a plan and appends a row to `docs/perf_history.csv` (date, code version, example, seconds, results, best cost); without `--record` it prints a profile.
- **Browser tests** (v8, `tests/e2e/`, marker `e2e`): the app runs in-process and headless Edge clicks into it through the DevTools protocol (`tests/e2e/cdp.py`): face selection with plain / Shift / Ctrl clicks, Alt+click → Move and Cancel, dragging the part's gizmo, picking a stay-out zone in 3D, the dark-mode log colours and the guide's search. They are skipped when Edge is not installed.

- `CLAUDE.md` holds the conventions (frames, units, cost), the milestone status with each done-when check, and design notes per module.
- `core/` and `opt/` must not import the GUI.
- **Edit `docs/USER_GUIDE.md`, then run `$PY scripts/build_guide.py`** to regenerate the HTML. `tests/test_guide.py` fails if the HTML is out of date.
- Each milestone's acceptance test lives in `tests/` (for example `test_kinematics.py`, `test_feasibility.py`, `test_reachability.py`, `test_placement.py`, `test_tool_geometry.py`, `test_carriers.py`, `test_export.py`). The carrier builder is covered by `test_carrier_builder.py` and `test_examples.py`, the brush / single points / tessellation by `test_point_tools.py`.

UR robot models © 2023 Universal Robots A/S. Use hereof is subject to Universal Robots A/S' Terms and Conditions for Use of Graphical Documentation (see `assets/robots/ur20/meshes/LICENSE.txt`).
