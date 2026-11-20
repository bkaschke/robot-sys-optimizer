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
10. [For software engineers](#10-for-software-engineers)

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
7. **Results tab** → *Export* → **TCP poses (CSV)**, **Optimized tool URDF (zip)**, **Export report** (*Report format* PDF or HTML). Files are downloaded and also saved to `workspace/default/export/`.
8. **Project** (top of the panel) → **Save** stores everything in `workspace/default/project.json`. Autosave also does this every 2 minutes. At the next start, the **Open a project** window offers **Reopen last project**.

Tick **Dark mode** under **Settings & help** at the top of the panel at any time.

**Ready-made examples:** **Project** → *Examples* → `rail_long_part` (UR20 on a 7th-axis rail), `mobile_amr` (UR20 on a mobile robot built from simple shapes), or, around a 3 m tall wall, `gantry_overhead` (UR20 upside down on an overhead X-Y gantry with a bridge), `floor_xy_rails` (UR20 on a floor X-Y gantry: X rails, a bridge, a carriage on the bridge) and `mobile_lift` (mobile robot with a 1 m vertical lift) → **Open example** → **Optimize** → **Plan stations**. Examples are read-only templates: opening one makes a working copy in `workspace/examples/<name>/` (`<name>_2`, … when an earlier copy has your changes, which is never replaced), and saving or autosave only ever writes that copy.

---

## 2. Set up on a new computer

Tested on **Windows 10/11**. Linux and macOS should work with the same conda environment but have not been tested. **Just want to run it?** Use the installer or zip in [2.1](#21-download-and-run-windows-installer-or-zip) (recommended: no conda, Git or internet). **No installer allowed?** The portable copy in [2.6](#26-portable-copy-for-a-pc-without-conda-or-internet) also works offline.

### 2.1 Download and run (Windows installer or zip)

**The recommended route for a Windows PC.** The release contains everything (Python, the libraries, the robot models, prebuilt reachability maps, the guide). It makes **no internet connection**, at install or at run time, and needs **no administrator rights**. Get the files from the project's **Releases** page on GitHub or GitLab (or from a colleague): `RobotCellOptimizer-<version>-setup.exe` (installer), `RobotCellOptimizer-<version>-win64.zip` (the same folder as a zip) and `SHA256SUMS.txt` (to check a download: `Get-FileHash <file>` in PowerShell shows the same SHA-256).

- **Installer.** Double-click the setup exe and follow the wizard. It installs **for your user only** into `%LOCALAPPDATA%\Programs\RobotCellOptimizer`, adds a Start-menu entry (**Robot Cell Optimizer**), optionally a desktop shortcut, and an uninstaller. Start the program from the Start menu: a console window opens with the app's messages, and after a few seconds the browser opens the GUI at `http://localhost:8080`. Closing the console window stops the app.
- **Zip.** For a PC where installers are blocked: unzip anywhere you can write to (not `Program Files`) and run `RobotCellOptimizer\RobotCellOptimizer.exe`. If Windows marked the zip as downloaded from the internet, right-click it, **Properties**, tick **Unblock** before extracting.
- **Your data** (projects, the tool library, caches) lives in `%LOCALAPPDATA%\RobotCellOptimizer` (`workspace\`, `tools\`, `cache\`), separate from the program. Set the environment variable `RSO_DATA` to use another folder. Uninstalling does **not** delete this folder.
- **Updates.** There is no automatic update and no update check. Run the new version's setup exe (or unzip the new zip into a new folder): the program is replaced and your data is kept.
- **SmartScreen.** The program is not code-signed, so Windows may show **Windows protected your PC** on first run: click **More info**, then **Run anyway**. On a managed PC that blocks unsigned programs, IT has to allow the file (by path or by the SHA-256 in `SHA256SUMS.txt`).
- **Version.** The version is shown under **Settings & help** (line "Version: ..."), at the top of the side panel, and by `RobotCellOptimizer.exe --version`. The release page explains which file to download and the first-run warnings step by step.

Sections [2.2](#22-prerequisites) to [2.5](#25-first-launch) set up from source with conda (for developers, or PCs that can use Miniforge); [2.6](#26-portable-copy-for-a-pc-without-conda-or-internet) is the older conda-pack zip.

### 2.2 Prerequisites

| What | Why | Get it |
| --- | --- | --- |
| Git | clone the repo | https://git-scm.com |
| Miniforge (conda) | Pinocchio and Coal (kinematics and collision) are only packaged for Windows on conda-forge, not pip | https://github.com/conda-forge/miniforge |
| A modern browser | the GUI runs in the browser | Chrome / Edge / Firefox |

Install Miniforge **for your user** (default location `C:\Users\<you>\miniforge3`). You do not need to add it to PATH.

### 2.3 Clone and create the environment

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

### 2.4 Check the installation

```bash
PY=C:/Users/<you>/miniforge3/envs/rso/python.exe
$PY -c "import pinocchio, coal, viser; print('ok')"
$PY -m pytest -q                        # ~160 tests, ~8 min; the first run also builds reachability maps
$PY examples/headless_plan.py           # end-to-end plan + exports without the GUI (~1 min)
```

### 2.5 First launch

```bash
scripts\run_app.bat        # Windows double-click / cmd
# or
$PY app.py --port 8080 [--dark]
```

The first station plan or heatmap for a tool builds a **reachability map** (about 50–70 s), which is cached in `cache/reachability/`. Later runs reuse it in seconds. The cache is not in git; each computer builds its own.

### 2.6 Portable copy for a PC without conda or internet

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

The tabs are numbered in setup order: **1 Robot & cell → 2 External axis → 3 Tool → 4 Targets → 5 Optimize → 6 Results**. A **status strip** above the tabs shows every setup step at a glance (✔ confirmed, ✎ changed since you confirmed it, ○ not confirmed yet) and whether the selected plan and the feasibility check are up to date (⚠ with what changed). Each setup tab starts with its status (*not confirmed yet*, *confirmed*, or *changed since you confirmed it*) and a **Confirm step** button (**Confirm setup** on the External axis tab, which also locks the carrier). You can open any tab at any time; the Optimize tab lists the steps that are not confirmed. **Confirming a step turns its drag gizmos off**, so a confirmed object can't be moved by accident: step 1 the robot base, part, stay-out and reduced-speed zone gizmos, step 2 the carrier body and mount gizmos, step 3 the tool mount and TCP gizmos (also when a project opens with those steps confirmed). To move one again on purpose, pick it in Objects → *Drag gizmo on* (or use **Move…** / Alt+click); the step then shows *changed* until you confirm it again.

**Out-of-date results.** Every check, plan and tool result remembers the setup it was computed for. When you then change the robot or mount, the external axis, the tool, the part (mesh or position), the targets or the **collision geometry** (the arm padding, the part's, tool's or external axis bodies' collision meshes, also when such a file is rewritten on disk), the Results tab shows **Out of date: … changed** for that result, the Optimize tab flags an out-of-date feasibility check, and exports ask before writing (**Export anyway** / **Cancel**). Re-run to clear it. **Plans that no longer apply are removed:** when the **part moves** (or its mesh changes), the **tool** changes or **targets** are added, removed or edited, the station plans and tool-optimizer results made for the old setup are cleared from the Results list together with their paths (the Log says which and why; applying a tool design's values does not clear that design). A plan that was already out of date when its project was opened is only flagged, not removed. (Moving the robot or carrier between stops does not make a plan out of date; the feasibility check, which runs from the current base, does go out of date.) Results saved before v2 carry no record and are not flagged.

**Log** (window at the bottom centre of the 3D view; drag it elsewhere, or **Minimize log** to keep only the newest message). Its top line lists the setup steps that are **not confirmed** yet. Below, newest first: every error and notification, each finished run (station plan, tool optimization, feasibility check, path build) with its result and how long it took, and **why points could not be reached**: grouped by reason (out of reach, beyond the joint limits, collisions, not covered by any stop) with the point ids. **Clear log** empties it.

### Top of the panel

**Settings & help** (closed by default):

- **Dark mode**: switches the GUI and the floor grid. The `--dark` flag starts in dark mode.
- **Floor grid**: sized to the cell. It covers the part, the robot's reach around its base and around every planned stop, the carrier (fixed bodies and the moving ones at both ends of their travel) and the stay-out / reduced-speed zones, plus 1 m all round (at least 10 × 10 m, in whole metres). It follows when you move or add things (checked every second).
- **Run planners in a separate process** (on by default): *Plan stations*, *Quick plan* and *Optimize tool* run in a worker process, so the panel stays responsive, **Cancel** always ends the run (a run that does not stop within 2 s is ended with its process, which frees its memory; the plans already listed stay), and a second run waits until the first has finished ("waiting for the running planner job"). The worker keeps the robot, maps and meshes loaded between runs and is ended after 5 idle minutes. Untick it to run in the panel's own thread as before.
- **Check worker processes** (default: the number of CPUs − 1, at most 4): the station planner's exact checks (verification, localization re-checks, zones, refinement, station areas) run in this many processes at once; **1** = one after the other. Each process holds its own robot and collision scene (about 300 MB). Starting them takes about 10 s; they are kept for the next plan of the same setup. The plans are identical either way; it pays off on large point sets (the overhead gantry example: 114 s → 78 s), not on the 25-point examples. If the processes cannot start, the plan runs without them and the log says so.
- **User guide**: **📖 Open user guide** opens this guide in a new browser tab. It is served by the app on its port + 1 (`--guide-port N` to change it) and also works from another PC on the network. **Search the guide**: type a word and click the **🔎 Search the guide for …** link under it; the guide opens with that search done. **Download user guide (HTML)** saves the same page as a file.
- **Save support bundle (zip)**: when something goes wrong, click it and send the zip along with your description. It holds the program version, facts about the PC (Windows version, memory, package versions, the data folders), the messages of the Log window, the program's log file and the open study's `project.json`; for a batch job also `batch.json` and each job's `error.txt`. **It never contains part, tool or carrier files**, and nothing is sent anywhere: the zip is saved to `support\` in the data folder (the line under the button shows the path) and offered as a download. The log file itself is `logs\app.log` in the data folder (four files of 1 MB at most; the newest messages are in `app.log`): every Log-window message, warnings and errors of the program, and errors that would otherwise be lost. It is written by the running program only, not by scripts or tests.
- **Searching this guide**: the box at the top of the guide's sidebar (or press **/**) finds a word or phrase anywhere in the guide: matches are highlighted, the sidebar keeps only the sections that contain one, and the count shows where you are. **Enter** jumps to the next match, **Shift+Enter** to the previous, **Esc** clears. A link ending in `?q=word` opens the guide with that search.
- **Objects** (its own panel on the **left** of the 3D view): *Show / hide* is a tree of the cell: **Robot** (and its *Tool*), **External axis** (and each carrier body), **Part** (*Targets*, *Selection*), **Stay-out zones** (each zone), **Reduced-speed zones** (each zone) and **Results** (*Stops and lines*, *Ghost robots*, *Station areas*, *Paths*); a child is shown only when its parent is ticked too. *Collision geometry*: **Show robot / tool / part / external axis** draws that component's collision shapes in translucent orange, at the current pose, exactly as the planner, the paths and the jog check them (the robot arm bloated by its padding). How they are made is set in the Package builder (see [Collision geometry](#collision-geometry)); the robot arm padding in Robot & cell. *Gizmos*: *Drag gizmo on* chooses which object has drag handles (robot base, part, tool mount, carrier body, stay-out zone, reduced-speed zone); *Gizmo size* scales all of them and *Gizmos always on top* draws them through bodies, so one inside a body can still be grabbed. The part and stay-out gizmos are sized to their object (its handles reach outside it; *Gizmo size* multiplies that), The Targets selection tools keep working while the part's gizmo is on; only while **Move…** moves the part do clicks on it go to its gizmo. **Move object** + **Move…** (or **Alt+click** the part or a stay-out zone in 3D): a small *Move* window appears with **Confirm** and **Cancel**, and only that object's gizmo is on. Drag it, then **Confirm** keeps the new pose (a confirmed setup step shows *changed*, as after any drag) or **Cancel** puts the object back exactly where it was (Esc does not work here, use the button). The robot base, the carrier (moved with the Jog panel while the window is open) and the carrier bodies (while the External axis setup is not confirmed) are in the *Move object* list too.
- **View** (top of the **Objects** panel on the left): **Lighting** = *Studio* (the default: soft environment light plus a key light; the environment picture is bundled, nothing is loaded from the internet), *Flat* (even light) or *Off* (the plain viewer lights). **Shadows** (off by default) lets the key light cast shadows on the floor; it needs graphics power and can be slow without a graphics card. **Fit view** frames the robot, part, carrier and zones. **Direction** *Top / Front / Side / Iso* looks at the cell from that side, at the distance of the last fit. The camera also fits by itself when a project opens or a different part is loaded (not when you only move or tessellate the part).
- **Jog** (its own panel at the **bottom of the left column**, below Objects): *Move* the robot / carrier or the part with the arrow buttons (−X/+X, −Y/+Y, −Z/+Z, −Rz/+Rz) by *Step (mm)* / *Turn step (deg)*. With a carrier the arrows drive its axes: a rail stops at the end of its travel, a gantry within its x / y range, a mobile base moves and turns without limits (Z = the lift axis, if any). Below them, **Carrier position** holds the carrier's axis sliders (from the External Axis tab) and **Robot arm** the arm jog (frames, joint steps; described under Robot & cell → *Jog*), so everything that moves the robot is in one place. A mobile base's *x* / *y* are number fields without limits (type any position; the arrows, gizmo and joystick drive past the planner's search range too). **Carrier gizmo & joystick** (folder): *Show carrier gizmo* puts a drag gizmo on the carriage (robot mount) with only the carrier's own axes: a rail along its track, a gantry in x / y, a mobile base in x / y plus a turn about z, and z for a lift. Drag it and the carrier (and the robot) follows; rails and gantries stop at their travel. The **joystick** pad drives the carrier while you hold and drag the knob (further from the centre = faster, up to *Max speed* / *Max turn*; release stops it). Mobile base: *Mobile drive* = **Drive and turn** (up / down drives forward / back along the heading, sideways turns) or **Strafe** (up / down along the heading, sideways across it). Rail: sideways moves along the rail, up / down the lift. Gantry: sideways = x, up / down = y of the gantry frame. The *Joystick* text below the pad is the value the pad sends (you can ignore it).
- **Project → New part, same cell**: keeps the robot, tool, external axis and stay-out zones, clears the part, its targets, plans, checks and paths. **Project → New empty study**: nothing at all, not even a robot (choose one in tab 1). Both first save the current study and start in a new folder `workspace/study_<date>_<time>`, so the previous study stays as it was.
- **Project**: *Folder* (relative to the repo, or absolute), **Save** / **Load** (`<folder>/project.json`), **Download project.json**, **Open project.json** (upload). Loading never half-applies: a broken project leaves the current session untouched.
  - **Autosave** (on by default) saves to the current project folder every *Autosave every (min)* minutes (default 2) when something changed. The line under it shows the last save time.
  - **Open a project at start**: the app opens on an **empty cell** (the robot on a static base, no part, tool or targets) and the first browser that connects gets the **Open a project** window: **Reopen last project** (folder and date), **Recent projects** (the last five you saved, loaded or autosaved; *Open recent*), **Examples** (*Open example*), **New empty cell** (a new folder `workspace/study_<date-time>`) and **Close** (keep the empty cell). A project that fails to load shows the error in the window, which stays open. The empty start-up cell is not autosaved until you open, create or save a project. Start with `--open <folder>` (e.g. `scripts\run_app.bat --open workspace\examples\rail_long_part`) to open a project directly without the window; `--fresh` is still accepted but does nothing now.
  - **Open project.json** (the file picker) loads into a **new folder** (`workspace/<stem>_<time>`) and never overwrites the current project; autosave and **Save** are blocked while it loads.
  - **Uploads** (meshes, URDF zips) are checked before they replace anything: a rejected upload keeps the old file, and replacing a file shows a "replaced existing" notice. A part's fingerprint includes its mesh content, so plans made with an older version show as out of date once after the upgrade.
  - A project brings back the whole cell **and** where you were: the arm's joint pose, the surface selection, feasibility colours, every plan's joint solutions (ghost robots, CSV export without re-planning), the selected solution, and the panel settings (spacing, brush, check and planner settings, display options, dark mode). *Examples* + **Open example** opens a working copy of a shipped example (see Quickstart). To add or update an example, type a *New example name* and click **Save as example** (tick *Overwrite existing example* to replace one); this is the only way the app writes into `examples/`. Saving a project to another folder takes its uploaded meshes (`assets/`) along. Every save also copies any file the project uses from outside its folder (a part, tool, carrier or stay-out mesh, an uploaded robot, loaded from another project or from anywhere on the PC) into `<project>/assets/imported/` and refers to the copy, so the project keeps working when that folder is moved or deleted. Plans and check results stay up to date across that copy: the app compares files by their contents, not by where they are. Files shipped with the app (`assets/`, `examples/`) are not copied. Opening a project whose file is missing no longer fails: the project opens without that item (the part, the tool, a carrier body or stay-out zone; a missing uploaded robot falls back to the UR20), and the log and a message say what to load again. Before leaving an item out, the app looks for a file of the same name anywhere in the project folder (for a project copied to another PC with its files put next to it) and uses it. A missing part, carrier body or stay-out mesh then opens the **Locate missing files** dialog: give each file with its *Locate …* button and the project is saved and opened again with it (the file is copied into `<project>/assets/imported/`), or **Close** to go on without them. Until then, saving keeps the left-out items in `project.json` with their old paths, so an autosave does not drop them.

### 1 Robot & cell tab

| Section | What it does |
| --- | --- |
| **Robot** | Choose a robot from the registry (the UR20, UR10e, UR30, the FANUC CRX-10iA, CRX-20iA/L and CRX-30iA (`crx10ia`, `crx20ia_l`, `crx30ia`), the ABB GoFa 10, GoFa 12 and PoWa 30 (`gofa10`, `gofa12`, `powa30`), or `generic_6r`, a made-up industrial arm for trying non-UR kinematics; see [Adding a robot](#adding-a-robot)), or **Upload URDF (.zip)** for another robot (numeric IK: display, jogging and the feasibility check; the planners need an analytic solver). The mount presets are floor, wall and ceiling. Move the base with the gizmo or the xyz/rpy fields (static base only: with a carrier the base rides on it and these fields are read-only, so the robot can never be dragged off the carrier). *Joints (deg)* sliders jog the arm; the readout shows the flange pose computed by Pinocchio. *Home position* (folder, everything about the home): **Go to home** jogs the arm to its home pose; jog the arm with the sliders and press **Set home to current pose**, or type one value per joint in *Home joints (deg)* (comma separated) and press **Apply typed values**; values are clipped to the joint limits. The home is saved with the project; **Use robot default** goes back to `home_deg` in the robot's `spec.yaml` (UR20: 0, −90, 0, −90, 0, 0). The paths use this pose between stops. When the home collides at a stop (for example a default home that stands the arm straight up, on a robot hanging from a gantry), is on the other shoulder side from the plan (FANUC CRX, ABB GoFa / PoWa: see [Joint-configuration zones](#joint-configuration-zones)), or has no clear joint path to a few planned poses at every stop, the paths use a clear folded pose that has (if one does) and say which one (Paths info and the log); *Home position* then shows it with **Use the paths' home**, which keeps it as the project's home. *Joint limits* (folder): a *min / max* (deg) per joint; **Apply joint limits** narrows the robot's range (kept inside the URDF limits, a joint with min ≥ max is refused) and **Use URDF limits** goes back. The limits are saved with the project and every check uses them: IK solutions, the station planner, the tool optimizer, the paths, the joint sliders and the home pose (re-run the planner after changing them; Results are flagged out of date). The quick reach pre-check keeps joint 1's full turn and the exact checks apply its limit. *Jog* (left column → Jog panel → **Robot arm**): step the robot like a teach pendant. v11: the **arrows** (◀ axis ▶ for Move X Y Z, Turn Rx Ry Rz and each joint) jog one step on a click and **keep jogging while held** (one step every 0.12 s after a short delay); the pre-v11 step buttons are in the *Single steps* folder. *Frame* = **Joint** (*Joint − / +* J1..J6 by *Joint step*), or **World**, **Robot base**, **Part** or **Tool**: *Move − / +* X, Y, Z move the *Jog point* (the flange or a TCP of the tool) by *Step (mm)* along that frame's axes, and *Turn − / +* Rx, Ry, Rz turn it by *Turn step (deg)* about those axes through the jog point (the point stays put while it turns). The joints come from IK nearest the current pose. A step that leaves the reach or the joint limits, or would need a configuration change (a joint moving more than 30°, e.g. through a singularity), is refused with a message. The readout shows the jog point's pose in the chosen frame (Tool: in the world). Every jog step is checked against the floor, the part, the external axis bodies and the stay-out zones (the planner's pairs and clearances): a step that collides is refused and names what it hits; tick *Allow collisions* to jog anyway. *Show TCP gizmo* puts a drag gizmo on the jog point with the axes of the chosen *Frame* (World, Robot base, Part or Tool; Joint uses the world axes): drag an arrow to move the point, a ring to turn it about the point. The same rules apply as for the buttons (a pose out of reach, past a configuration change or into a collision is not taken: the arm stays at the last good pose and the gizmo jumps back onto it when you let go). After any move (also with the joint sliders, which are not refused) the readout warns **⚠ Collision** with the colliding pairs. |
| **Part** | A test part (flat plate, curved panel, open box, long rail, deep hole) or **Upload part (.stl/.obj/.ply/.step)** (also `.stp`; a STEP file's units come from the file itself, so *Upload units* applies only to STL / OBJ / PLY): set *Upload units* first (**mm** by default); *Units* is the current part's unit. A STEP part is meshed on import (offline, with the bundled CadQuery; a big file takes a while the first time, then the mesh is cached) and each triangle remembers its CAD face, which *Detect tool surface* uses. **A part over about 300,000 triangles is drawn as a simplified copy** (a note under *Units* says so); planning, collision and face picking still use the full mesh. Meshes are drawn double-sided (never see-through from one side). Place it with the gizmo or the fields. |

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
| **Standoff and safety** | *Part standoff min / preferred*: carrier-to-part distance; the minimum is hard, closer than preferred adds a cost. It is measured in 3D, so a carrier may sit partly under a raised part: model fixtures and stands as part of the mesh or as keep-outs. *Keep-outs*: floor rectangles `x0,y0,x1,y1; ...` in mm. *Localization* (mobile): position/heading error every stop must tolerate. **Mobile base: robot sides, standoff range, part sides.** Tick the **robot sides that may face the part**: *Front*, *Side 1 (left)*, *Side 2 (right)*, *Rear* (several allowed: the planner picks per stop; *Side 1 / Side 2* = the base runs parallel to the part). With any side ticked, every stop keeps a standoff between **Standoff from (mm)** and **Standoff up to (mm)**: the same value in both = one fixed standoff on one contour; a range (e.g. 400 to 700 mm) lets the planner choose the distance per stop (it moves the base straight out from the part, keeping the side that faces it), and you can restrict the **part sides it may work from** (*Part ±X / ±Y side*, in the part's own frame; none ticked = all). No side ticked = the free x / y / heading search with *Part standoff min / preferred*. **Rail / gantry: *Align track with the part*** turns the track along the part's long axis and centres it on the part (keeping its side and distance); it follows the part when you move it. (Older wording: — pick which side of the base turns towards the part (front = carriage +x, left = +y) and one gap between the base outline and the part outline (top view). Every stop then sits exactly at that gap with that side facing the part: the stops lie on one contour around the part (like a rail wrapped around it), which keeps one safety distance everywhere and makes planning much faster. Choose *free (search the heading)* for the older free x / y / heading search with *Part standoff min / preferred*.) |
| **Apply base settings** | Commits the axis-type and safety fields. Body and mount edits apply immediately. |
| **Carrier position** | In the left column → Jog panel → **Carrier position**: sliders move the carrier, and the robot with it. The readout gives the footprint-to-part distance; it and the red/amber standoff outlines follow the part when you move it. |

Every body collides with the arm and tool (from the upper arm on), and counts for standoff and keep-outs. Drawing: red line = minimum standoff, amber = preferred, red boxes = keep-outs; with robot sides ticked (a contour plan) the **standoff range** from *Standoff from* to *Standoff up to* is a blue band around the part. The lines, the band and the keep-outs follow the fields **as you type** (a preview: **Apply base settings** still applies them), and the left panel lists them under External axis as **Standoff** and **Keep-outs**, so they can be hidden on their own.

### 3 Tool tab

| Section | What it does |
| --- | --- |
| **Tool type** | Where the tool comes from; only that type's fields are shown: **Library or URDF** (the *Library* folder below), **Mesh / STEP file** (the *Mesh / STEP tool* folder), or **Package builder** (build it from parts there, then **Use in this project**). Loading a tool selects its type. For a contact-free process (scanning, spraying) the contact-face fields under TCPs are hidden. |
| **Library** | *Tool* picks a saved tool (`tools/*.yaml`). **Upload tool URDF (.zip)** imports a URDF plus meshes; `package://` paths are resolved. **Save to library** stores the tool and copies its geometry into `assets/tools/<name>/`. |
| **Mesh / STEP tool** | A tool from **one file**: **Upload tool mesh (STL/OBJ/PLY/STEP)** (*Mesh units* for STL / OBJ / PLY, mm by default; a STEP file carries its own unit and is meshed with CadQuery). The whole file becomes the tool head, sitting directly on the flange mount (a *pose* tool with every extension range at 0; widen the ranges if the tool optimizer may add a connector). A tool more than 10 m across is refused (a unit mistake). **STEP coordinate systems:** the model's **named coordinate systems** (datum CSYS features, e.g. *CSYS_TCP*) are listed. *TCP datum* + **Set TCP to datum** puts the TCP chosen in the TCPs folder on it (*Flip TCP z* turns it 180° about x: the TCP z must point **out of the tool, toward the part**); faces within *Contact faces within (mm)* of the TCP (default 5; 0 = none, for a non-contact tool), grown over the flat or smooth surface around them, become that TCP's **contact faces** (the probe's end face may touch the part). *Flange datum* + **Mount datum on flange** makes that coordinate system coincide with the robot flange (*Flip flange z* when its z points into the robot). Only named coordinate systems are listed; for a datum inside a sub-part of an assembly the sub-part's own placement is not applied (export single-part tools). |
| **Process** | *Type* applies a preset: scanning, sanding, grinding, gripping, spraying or custom. You can then edit the force, tilt cone, standoff and spin. Contact-free processes (scanning, spraying) clear all contact areas. |
| **Flange mount** | Places the tool on the flange with a gizmo or typed values. |
| **Extension** | *Type* **pose** (default for new tools): where the head sits relative to the flange mount, as **x, y, z** (mm) and **rx, ry, rz** (deg, roll-pitch-yaw), each with a *range* the optimizer may use (equal ends fix a value). The connector between mount and head is drawn and collision-checked as a **smooth curve** (it leaves the mount along its axis and arrives along the head's axis), not a straight cylinder; its *capsule radius* applies along the curve. Legacy *straight* / *bent* tools keep the old L, θ, d, roll ranges (θ and d for bent only); the sample tools are legacy. **Switching a legacy tool to pose keeps its shape**: its head pose becomes x y z / rx ry rz with every range fixed at that value; widen the ones the optimizer may change. rx / ry / rz are the W / P / R angles. **Rough placement** sliders set the current values of the free ones. **bent2** (v8): two bends: up *L*, turn by *θ*, a leg *d*, turn again by *θ2* and a second leg *d2* (an S- or U-shaped probe; *roll* turns it all about the flange axis), each with its range; it uses the shared flange map like pose tools. The **pose** type is the free-form (spline) shape: its connector is a smooth curve from the mount to wherever the head is placed. |
| **TCPs** | Add, rename or remove TCPs and place each with a gizmo. **Lock TCP position** is ticked by default: the TCP's xyz / rpy fields are read-only and its gizmo has no handles, so it cannot be moved by accident. Untick it to move or type the TCP pose. **Pick contact faces (click tool)**: tick it, then click the tool; one click selects the whole smooth face (*Region angle* sets how far it spreads), and clicking again removes it. **Whole-link contact** marks entire links. **Two tips / several TCPs** (v8): a tool can carry more than one TCP (e.g. a pad on each end); the station planner and the tool optimizer then use the reachability maps of all of them (a point counts as covered when any tip covers it) and the exact check chooses the tip per point among the TCPs its group allows (*Groups & tolerances*). The CSV export names each point's TCP. **Travel direction** (v11, per TCP): how the tool may move over the surface. *free* (the default: a sander, the tool may spin about the normal), *along TCP x* / *along TCP y* (a line scanner sweeping along that axis) or *TCP x across travel* (a drum sander: the drum axis is x). A directional TCP's spin is set at every path sample so that axis lies along (or across) the travel direction; the paths pick the way round (the tool is symmetric) for the least joint motion. With a row pattern (Paths: *Zigzag*, *Unidirectional*, *Cross-hatch*) the tool keeps its orientation on a row change and steps sideways, so it moves along its other axis only at the end of a row. **Scan width (mm)** and **Overlap (%)** give the row spacing that covers the surface (shown under them; **Use for the target rows** writes it into Targets → Generate points: *Row spacing* and *Raster step-over*). **Overrun past edges (mm)** runs every row that far past the part at both ends and makes the row changes out there, off the part, so the scan covers the edges. Path samples where the tool cannot keep its axis on the travel direction (no arm pose at the required spin) are flagged **direction**. The station planner does not know the travel direction yet: if a plan's points cannot be reached at the travel spin, widen the group's spin tolerance or replan. The package builder has the same fields (*What may touch the part*). **Minimum surface radius (mm)** (v10, per TCP; 0 = any): the tightest curve this TCP can work on (a flat pad cannot sit in a fillet tighter than itself); **Radius limit applies to** *concave (inside curves)* (default: fillets, pockets, the inside of a tube), *convex (outside curves)* or *both*. At each target the surface is measured over the TCP's contact footprint (its contact parts / faces around the TCP; without contacts, the minimum radius itself): points whose radius is below the minimum are **not compatible with the tool**. They are marked on the part with a brown cross and `✘ id radius r mm concave < R`, can colour the part (*Colour part by → Surface radius (tool fit)*, % of the minimum: red below it), are listed in the Log when a plan starts, on the Results card (**Tool fit**) and in the report (**Tool fit** section). Optimize → Settings → **Leave out points the tool cannot fit** leaves them out of planning (the card then counts them as not reached); otherwise they are planned and only reported. Packages set the same fields in the package builder (*What may touch the part*). |

Sample tools: **orbital_sander** (bent extension range, pad as the contact link) and **scanner** (contact-free). Both are simple stand-in geometry. Switch a tool to *Type* **pose** to let the optimizer move and tilt its head.

### 4 Targets tab

| Section | What it does |
| --- | --- |
| **Select surface & points** | Pick a **Tool**; only its settings are shown. **Fill +** / **Fill −**: click a face to add / remove its smooth region (neighbouring faces may turn by at most the *Fill angle limit*). **Brush +** / **Brush −**: an orange-ringed **handle** appears on the part; drag it and every face within the *Brush radius* of its path is added / removed (it follows the surface even if you drag above it, and skips faces turned more than *Brush max surface angle* from the surface under it, so it stays off the back of thin parts). A click paints one dab and moves the handle there. **Box +** / **Box −**: drag a rectangle in the 3D view (the camera does not orbit while a Box tool is chosen). **By normal**: every face within an angle of a direction. **Full surface**: click a body of the part to add all of its faces. **Single point**: one target exactly where you click (normal from the surface). **Remove point**: deletes the target nearest the click, within the brush radius. **Point brush**: each click adds a dab of the *Part-anchored lattice* inside the brush radius. **Ctrl-click** turns + into − (and back) for the click tools. **Select all**, **Clear** and **Invert** act on the whole part; the selected face count and area are shown below. |
| **Generate points** | Pick a **Generator** (only its settings are shown); it runs on the selection. **Surface grid (rows/columns)** (as in the sand-optimizer): the grid lies across the selection's mean facing direction, rows run along the part X axis turned by *Row direction (deg)*, points along a row are *Column spacing* apart and rows *Row spacing* apart, centred on the selection; points closer than *Edge inset* to the selection's edge are left out. A strongly curved selection gets a warning (select one side at a time). **Each side separately** (on by default; grid, lattice, projected grid and raster lines with an automatic direction): a selection that covers several sides of the part, e.g. the top **and** bottom of a plate or a whole closed part, is split into sides that each face one way, and every side gets its own points (the info line says how many sides). Sides smaller than one grid cell (thin edge strips) are skipped. Points on opposite faces of a thin part are never treated as duplicates. **Part-anchored lattice**: rows / columns anchored at the part origin (the Point brush grid), rows along *Row direction (part frame)*. **Projected grid**, **Poisson (even)**, **Raster lines** and **Selection edges** as before (spacing in mm). Normals are *interpolated* (smooth) or *face*. Points that already exist are skipped, and **Undo last add** removes the last generate, click or dab. **Delete points on selected surface** and **Delete all points** sit right under *Generate on selection*. |
| **Refine part mesh** | **Tessellate part mesh** splits the part's triangles until no edge is longer than *Max edge length*. The shape does not change, but clicks, region growing, brushes and interpolated normals work on a finer mesh (useful for coarse CAD exports with long thin triangles). The refined mesh is saved as `<project>/assets/parts/<name>_refined_<N>mm.stl` and becomes the part; the selection and existing points are kept. |
| **Detect tool surface** (folder, closed) | For a sanding mandrel (the surface the laminate is laid on; male or female, faceplate on egg-crate or thin shell): **Detect tool surface** suggests which faces to work on. It looks from world **+Z** (a rotated part still looks from above), so put the part in its working pose first. The suggestion is a magenta **preview**, not yet the selection. The panel shows a one-line reason (the area and the share visible from above) and a **confidence** (0 to 100 %); under 70 % it adds **Low confidence: review before accepting**. It may list **Candidates** it left out (typically steep flange returns or sides of the faceplate that fall away across a convex break); **Add** puts one into the preview. **Warnings** are advice only and change nothing: a concave area tighter than the tool's pad radius, a steep selected area, nothing visible from above. Then **Replace selection** (the preview becomes the selection), **Add to selection** (adds it to what is selected) or **Discard**; edit the result with the usual Fill / Brush tools. **Limits:** it is a suggestion to review, not a result to trust blind. It was tuned and tested on synthetic mandrels only (no real parts yet), so expect mistakes on real shapes; ambiguous shapes (several similar up-facing areas, creases it cannot close) get a low confidence on purpose. It assumes the tool surface faces up, and that steep walls inside a female cavity belong to the tool surface while walls that fall away outside it do not. |
| **Groups & tolerances** | Each point belongs to a group, and each group has spin, tilt, standoff and an allowed TCP list. **Path pattern** (v10): the group's own pattern for the paths (any Paths pattern); *(Paths setting)* uses the one chosen in Results → Paths. **Apply to group** saves them. |
| **Edit points** | Assign the points on the selected surface to a group, or flip their normals. **Import CSV** reads `x,y,z,nx,ny,nz[,group]` in mm or m. |
| **Display** | Colour points by *group*, *feasibility*, or the selected solution's *zone* / *configuration*; show normals; **Highlight unreachable points** (on by default) marks every point the selected plan (or, without one, the feasibility check) cannot reach with a large dot, a cross and a label with its id and **why**, coloured by reason: **red** out of reach (no arm pose gets there at any allowed tool angle), **orange** only beyond the joint limits, **violet** only close to a singularity, **magenta** collision, with what the best arm pose runs into (e.g. `✘ 12 collision: part, floor`; the Log lists the same per reason, e.g. "3 × every arm pose collides, hitting part (2), floor (1)"); **Preview tool at point** draws a see-through tool aimed into the surface at any point. **Colour part by** (v8) paints the part's surface around the points of the **selected plan** (Results; without a plan: the feasibility check) by a score, red (poor) → yellow → green (good), with a legend and the lowest / median value: **Clearance** (0 to 50 mm), **Manipulability** (% of the robot's best), **Joint-limit margin** (0 to 30°), **Tolerance reach rate** (the last Results → *Tolerance check*), **Edge check** (v10: the last Results → *Edge check* of the selected plan, red where a checked point is a gap, green where it is reached). **Reached / unreached (points)** (the default) shows no surface colour, as before. The colours follow the selected plan. |

Points are stored **in the part frame**, so moving the part moves them with it.

**Delete all points** asks for confirmation, and **Undo** brings the points back. A points CSV saved with a byte-order mark (BOM, e.g. from Excel) keeps its first row.

### 5 Optimize tab

**The optimizers need a confirmed setup.** *Plan stations*, *Quick plan* and *Optimize tool* stay greyed out until steps 1 to 4 are confirmed; a line under each button says what is missing, e.g. "Confirm **2 External axis** first" or "**3 Tool** changed since you confirmed it — confirm it again". The feasibility check and the base heatmap work at any time.

**Layout (v9).** The **Optimizer** dropdown at the top picks one optimizer: *Station planner*, *Quick station plan*, *Feasibility check*, *Base placement map*, *Tool geometry*, *Compare setups*, *Gantry placement* or *Several parts (batch)* (v13). A line under it says what that optimizer does, and only its section is shown, with one **Run** button (labelled for the optimizer), its **Cancel** and its progress bar. The settings they share are in one **Settings** folder at the bottom of the tab: **Max stations** (every optimizer that plans stations uses it), **Leave out points the tool cannot fit** (v10: points below the tool's minimum surface radius, Tool tab), **Costs & requirements** (station planner, quick plan, compare), **Costs & requirements (tool optimizer)**, **Cycle time** and **Cable & safety**. In the Station planner, *Candidate grid*, *Refinement* and *Objective* (with the fields that depend on them) are under **Advanced**. The rows below describe each optimizer and those settings. v11: **Confirm all setup steps** (top of the tab) confirms steps 1-4 at once after a dialog that lists the tabs not confirmed yet (and why: not confirmed, or changed since); and the Optimizer list only offers what fits the workspace's external axis (*Gantry placement* only with a rail or a gantry). Gantry placement also refuses a frame whose bridge would travel through the part (a floor gantry whose rails straddle it). v13: and one whose carriage (with its drop or lift stage) cannot get past the part anywhere along the bridge (it hits the part at every Y), since the bridge could not carry it across.

| Section | What it does |
| --- | --- |
| **Setup check** (folder, closed) | **Check setup** looks for common mistakes before you spend time on a plan, and lists what it finds: ✘ errors (no robot, part, tool or target points; a group that only allows TCPs the tool does not have), ⚠ warnings and • notes. The warnings: the **part is under 10 mm or over 30 m across** (almost always the wrong *Units*: a file in metres loaded as mm, or the other way round); the part **reaches below the floor**; points whose **normal points into the part** (the tool would have to come from inside: flip them under *Edit points*); points **more than 5 mm off the part surface** (a CSV in the wrong units, or made for another part); points **on top of each other**; points **inside a stay-out zone**; and, for a static base, points that **span more than one base position can cover** or lie **higher than the arm can reach from the floor** (reach plus tool length, a generous bound). Each finding names up to eight point ids. It is advice: nothing is changed and nothing is blocked. It does not test reach or collisions point by point: that is the feasibility check below. On a very large point set the surface tests use 3,000 points spread over the list and say so. The same check runs quietly whenever **Plan stations**, **Quick plan** or **Optimize tool** starts and writes only its errors and warnings to the Log; the batch runs it per job (see [Several parts (batch)](#several-parts-batch)). Without the GUI: `$PY -m ui.preflight_run <project folder>` (exit code 1 when there is an error). |
| **Feasibility check** | **Check all targets** with clearance and joint margin as settings. *Check from* = **stops of the selected plan** (default): each point is checked from the stop the plan in Results assigns it to, with the carrier there (points the plan did not reach: every stop is tried); or **current robot position**: everything from where the robot stands now (without a plan this is used automatically). The carrier's own bodies count as obstacles. The summary gives counts and reasons. The button is greyed out while there is nothing to check, and a line under it says why (no robot, part, tool or targets yet). Inspecting the result is now under **6 Results → Inspect result**. |
| **Base placement (reachability)** | **Show base heatmap** draws where the base could stand on the floor (bright = all points reachable with the best worst-case manipulability) and marks the best spot. **Move base to best** moves it there. Ignores collisions: confirm with the feasibility check. |
| **Station planner** | *Mode*: **Cost based** (the default, below) or **Rules only** (see the next row). **Plan stations** finds the fewest stations that reach every point, within *Max stations* (up to 40; static mode = 1). It plans one station count after the other and stops once two more stops in a row gave no better plan, so a large *Max stations* does not mean that many full plans. With many points the stop search works on a spread-out sample (a quarter of the points, 60 to 150; all of them below 60), adds a stop where the sample missed points that need one, and then checks **every** point exactly, so the result is still exact (357 points on the rail: 43 s → 18 s, same plan). Each solution goes to the Results tab as soon as it finishes. *Candidate grid* sets the search resolution; *Refinement* chooses how the stops are fine-tuned against the exact checks (**CMA-ES**, the default, **Bayesian (TPE)**, **Bayesian (Gaussian process)** or **None**), and *Objective* what the planner optimizes (**Weighted cost**, **Pareto front (NSGA-II)** or **Robust (worst case)**); see [Optimization methods](#optimization-methods). **Connected station areas** (on by default) reassigns points after planning so each stop's points form **one connected area** on the part and no stop's points sit among another's: every stop grows outward from the point nearest its base, taking only points it reaches exactly, then stray points join the stop around them. It is kept only when it leaves fewer pieces than the planner's own split. Plans appear quickly: every station count is first verified at its first-guess stops (a few seconds each), then the most promising one or two are refined and replace their quick version when better. **Cancel** stops within about a second and keeps the plans already finished. *Joint configuration* = **zones (minimize changes)** (the default) groups each stop's points into zones that keep one arm configuration and puts each zone in path order (see *Joint-configuration zones* below; since v7 round 2 the zone search reuses checks, so the rail example plans in about 8 s and the mobile example in about 13 s, down from 52 s and 58 s). **best per point** lets each point take its own best configuration, which can mix all of them at a stop (the overhead gantry example had four per stop); **Costs & requirements** (folder): every cost weight can be edited: **Cost per stop** (each base / carrier position beyond the first, default 5), **Cost per group** (each extra joint configuration at a stop, with or without zones), **Cost of arm motion** (the mean of the largest joint move between neighbouring points of a stop, per 90°: plans whose arm swings a lot from point to point cost more), **Cost per wrist flip** (a joint 4 / joint 6 jump over 180° inside a group's path), per unreached point, carrier travel per m, tool length, manipulability, clearance, joint limits and preferred standoff. They are saved with the project; **Default costs** restores them. **Must be met**: **All points reachable** and **No configuration change within a group** (turns zoning on; a stop may hold several groups, each in one configuration, and a wrist flip inside a group breaks the rule). With a rule ticked the planner keeps adding stops until a plan meets it (up to *Max stations*); plans that break a ticked rule stay in the Results list marked **✗**, below the valid plans, and the summary names the rule. |
| **Station planner → Mode: Rules only** | No costs: the *Costs & requirements* folder is hidden and every rule must hold: 1. every target reached; 2. the tool orientation within each group's spin / tilt tolerance; 3. joint limits (with the margin); 4. no collision (with the clearance); 5. one joint configuration per group, with no wrist jump over 180° (zones are always on); 6. **manipulability at least the *Singularity limit*** (% of the robot's maximum, default 5 %, 0 to 50): arm poses closer to a singularity do not count. *Objective*: **Fewest stations** tries 1, 2, … stops and stops at the first count with a plan that meets every rule; there it also tries up to three other stop sets (leaving out one chosen stop each time) and keeps the one with the fewest groups. **Fewest groups** goes on with more stops while that gives fewer joint-configuration groups (up to *Max stations*, or until two more counts give nothing better). No fine-tuning (no cost to optimize). When no plan meets every rule, the closest one is listed, marked **✗**, and the status line and the log name each broken rule with its count, e.g. "rule 6: 12 point(s) only reachable close to a singularity at 4 stations". |
| **Quick station plan** | **Quick plan**: a first look in seconds with *Max stations* (Settings): no fine-tuning, no zones, no station areas, and it stops after the first station count that gives nothing better. Its plans are labelled **Quick:** in Results; the status says they are not optimized (run **Plan stations** for that). Same Cancel, progress and confirmed-setup rule as the planner; the log gives the run time. |
| **Compare setups** (v8 robots, v10 tools) | Tick robots under *Robots* (the UR20, UR10e and UR30 by default; every robot in the registry is listed, including the FANUC CRX and ABB GoFa / PoWa cobots) and / or tools under *Tools* (the library's tools and the project's tool packages from the package builder) and click **Compare setups**: the study is planned once per **setup** = every ticked robot with every ticked tool (no robot ticked: the project's robot; no tool ticked: the project's tool), with everything else unchanged (part, targets, carrier, stay-out zones, the station planner's settings and costs; each robot uses its own joint limits and home, each tool its own TCPs, contacts and ranges). The project's own setup is always planned too and listed first, marked *(project)*. **External axes** (v11) adds the axis type as a third dimension: *Static arm (one base)* (the station planner places one floor base), *Rail*, *Floor XY gantry*, *Overhead gantry*, *Mobile base*, *Mobile base with lift* (the shipped examples' carriers) and the project's rail / gantry / mobile packages from the package builder. A rail or a gantry is first placed for this part with the Gantry placement scoring (the best grid frame; an overhead gantry also tries lower beam heights, down to 1.2 m above the part, its legs stretching), so each type gets a fair frame; the setup's name gets the axis (`ur20 · Rail`). Viewing such a setup in Results swaps its carrier in (and back), and saving keeps the project's own. Setups are named by the robot, plus the tool when it is not the project's (`ur10e`, `ur20 · scanner`). The table lists per setup: stations, points reached, the estimated **cycle time**, the worst clearance, the lowest manipulability (% of that robot's best), the cost (✗ = a must-meet rule broken) and the run time. The current cell is not changed. Every setup's plans are kept (with their joint solutions and its tool) and saved with the project: pick one under *Setup* and **View in Results** to see it at its stops with its plans (Results → *Compared robots*). The first plan with a robot builds its reachability map (about a minute). |
| **Gantry placement** (v9) | For a **rail** or a **gantry** (External axis tab): places the whole carrier frame relative to the part, then plans its stops. Candidate frames: a rail's track along the part's long axis, on either side, its line at every *Track standoff* from the part outline (from .. to, in *Standoff step*); a gantry frame along or across the part's long axis, centred on the part and moved up to *Frame shift* both ways (in 300 mm steps) (over the part: overhead gantries); v12: also gantry frames **beside the part** (floor gantries such as the XY rails), on every side, with the X rails along that side or across it, the gantry's nearest body anywhere along its travel (rails, legs, and the bridge and carriage at the ends of their travel) at every *Track standoff* from the part outline, moved along the side up to *Frame shift*; and always the **frame as you placed it**, scored like the others, marked *(your frame)* and listed even when it is not among the best; each with every *Robot mount height* (the robot mount's z on the carriage, *Height steps* values from .. to; equal values keep the current height): how high the robot sits on a rail, how low it hangs from a gantry. A frame is skipped when its fixed bodies (track, legs, beams) come closer to the part than the part standoff minimum, stand in a floor keep-out or come closer to a stay-out zone than the keep-out clearance, or no stop of it is allowed (standoff, keep-outs, travel). The *Robot mount height* range starts at the project's mount height and resets when another project opens or the carrier changes. The rest are scored in seconds with the reach maps (targets the stops cover, the fewest stops that cover them), and the best *Plan the best* frames are planned in full with the station planner's settings and costs. v10: **Refine the best frame** (on) then searches around the best grid frame without the grid (steps along and across the frame and in the mount height, halved until no step helps; marked *(refined)* in the list); frames that score the same as one already chosen and stand within 0.6 m of it with the same heading are skipped, so the planned frames differ. **Beam height (mm)** (gantry; equal values = keep it) with **Beam height steps** also tries the whole frame at other heights: the legs (fixed bodies standing on the floor) stretch to reach it, the beams, bridge and carriage move with it. The table lists each planned frame (position, heading, mount height, stations, points reached, cycle time, cost); pick one under *Frame* and **Use this frame**: the project's carrier frame and robot mount height move there (*Align track with the part* is switched off; v11: the setup steps you had confirmed stay confirmed, so **Plan stations** is usable straight away, and the bridge / carriage and robot are shown on the moved frame at the plan's first stop) and that frame's plans appear in Results (when Results already holds plans, a dialog asks first, since they are replaced). The chosen *Optimizer* and these fields are saved with the project. The same confirmed-setup rule as the planner. |
| **Tool geometry** | **Optimize tool**. **Pose tools:** tries head poses within the ranges (CMA-ES, *Shape / pose evaluations* candidates), plans stations for each (quick plan) and keeps the pose with the **best overall cost** (reach, stops, travel, manipulability, clearance, joint limits, and connector length as one term); the winner is planned once more with refinement. The table lists every candidate (connector length, full coverage, reached, stops, cost). **Legacy straight / bent tools:** the shortest L that still reaches everything within the station limit, then θ / d / roll (coverage-vs-length table). *Length tolerance* (mm) is up to you (no upper limit). **Cancel** stops within about a second and keeps the best result so far. The **best 3 designs** are offered under *Design* (✔ = reaches every target, then fewest missed points, then cost), **also when no design reaches every target**; each one is added to the Results list so you can inspect it or build its paths. **Costs & requirements** (in the Tool geometry folder) are the tool optimizer's **own** cost weights, separate from the station planner's (they start as a copy of them and are saved with the project): raise *Tool length* to prefer short tools, *Cost per stop* to prefer designs needing fewer stops, and so on. *Must be met → All points reachable* offers only designs that reach every target; *No configuration change within a group* plans each design with zoning and rejects plans with a configuration change inside a group. **Manufacturable designs** (v8): *Tube OD (mm)* uses a standard round tube (12 to 80 mm) as the extension instead of the tool's own capsule radius (its collision radius is then half the OD), and *Bend angle steps* (**free**, **5°**, **15°**) lets the optimizer use only bend angles (θ, θ2, roll; for pose tools rx, ry, rz) in those steps: every candidate is snapped while searching, so the result is a design a tube bender can make (rounding afterwards could lose points). **Part family** (v8): *Part family (folders)* takes other project folders (separated by `;`); every tool candidate is then also planned on each of those projects' parts, targets and cells, and the cost is the **sum** or the **worst** (*Family cost*) over all the parts; a point missed on another part counts as unreached. Use it to design one tool for several part variants. *Shape search* (**CMA-ES**, **Bayesian (TPE)**, **Bayesian (Gaussian process)**), *Tool and stations* (**Stations fixed after length search** or **Alternate**) and *Objective* (**Weighted cost**, **Pareto front (NSGA-II)**, **Robust (worst case)**) choose how the tool is searched; see [Optimization methods](#optimization-methods). **Apply to tool** writes the chosen design into the Tool tab; the tool step then counts as changed, and **Confirm tool** (shown under it while step 3 is not confirmed) confirms step **3 Tool** right here, so the planners can run with the new tool. *Tool extension* (folder at the top of Tool geometry, closed) is a **copy of 3 Tool → Extension**: *Type*, the ranges the optimizer searches, *Capsule radius* and the *Rough placement* sliders. Both copies edit the same tool and always show the same values. An edit in the copy shows its own **Confirm tool** button under the fields, so you can confirm step 3 without leaving the Optimize tab. **Save optimized tool to library** stores the tool with the chosen design's values under *Save as tool* (default `<tool>_opt`) without changing the tool in use. |

**Cycle time** (folder in the Optimize tab, closed): the speeds of the robot program, used to estimate how long one part takes. *Joint speed* / *Joint acceleration* (joint moves; each joint is also capped by the robot's own speed limit, so the UR20's base and shoulder never go faster than 120°/s), *Process speed* (along the surface), *Approach speed* (lifts, approach and retract), *Linear acceleration*, *Carrier speed* / *Carrier acceleration* (rail, gantry axes, mobile base), *Lift speed*, and *Settle time per stop* (the carrier settles or localizes before the arm starts). Every move is a trapezoid (accelerate, cruise, brake; a triangle when it is too short to reach full speed); a joint move lasts as long as its slowest joint. Each plan gets an **estimated cycle time** (Results → summary card): the joint moves between neighbouring points of every stop, the carrier moves between stops and the settle time per stop. **Build paths** gives the cycle time of the actual paths, split into process, approach, joint, carrier and settle time. **Path timing** (v10) decides how the built paths are timed: **Time-optimal (speed, acceleration, jerk limits)** (default) times each run of samples along its actual joint path, as fast as the joint speed, acceleration and **Joint jerk (°/s³)** allow (0 = no jerk limit) and, on the surface and on approach moves, at most the process / approach speed of the TCP; a joint path that bends must slow down at each bend. **Trapezoid per move (pre-v10)** treats each run as one move at the slowest joint's trapezoid, as before (faster but optimistic: the CRX-30 overhead-gantry example takes 459 s that way and 522 s time-optimal, mostly its joint moves, 84 s → 133 s). The plans' estimated cycle time is always per move. *Costs & requirements → Cost per minute of cycle time* (default 0, so plans cost what they did before) makes the planners prefer faster plans.

**Cable & safety** (folder in the Optimize tab, closed, v8): *Hose pack* models a hose or dress pack strapped to the arm. **none** (default); **clearance margin**: *Hose pack clearance (mm)* adds clearance around the arm from the upper arm on (upper arm, forearm, wrist) against the part and the stay-out zones; **hose along the arm**: the hose as a **shape** — a chain of round segments (*Hose diameter (mm)*, default 60) running from the upper arm to the wrist, *Hose offset from the arm (mm)* (default 150) from the line through the joints, on the *Hose side* (**+** or **−**) you choose. Each segment moves with its arm link; the hose must keep the part clearance from the part and the stay-out zones, stay above the floor and clear of the carrier (it is not checked against the arm or the tool it rides on). With *Show hose on the robot* it is drawn dark grey on the robot, so you can check its side and offset against your real dress pack. Unlike the margin, it only blocks poses where the hose itself would hit something, on its side of the arm. It is a layout model (straight hose per arm segment); the slack loop at the wrist is not modelled. **Joint 6 turn limit (deg)** (default 360°): at each stop, how far joint 6 may turn across the stop's points before the cables / hoses through the wrist wind up; beyond it the cost term *Cost per 90° of joint-6 wrap* (Costs & requirements, default 1) applies per 90°. **Reduced-speed zones** are under Robot & cell → *Reduced-speed zones* (see there).

**Several parts (batch).** Plans a queue of parts with this cell, several at once: see [Several parts (batch)](#several-parts-batch).
**Margins and re-runs:** the *clearance* and *joint margin* on the Optimize tab are the project's single setting (`optimizer.margins`): the feasibility check, every planner, the tool optimizer and the paths all use it, and it is saved in the report's provenance. Changing it marks the existing plans **out of date**. **Plan stations** and **Quick plan** keep showing the previous results until the first new plan arrives, and a failed run restores them; a station run never removes designs from the tool optimizer. A cancelled Pareto or robust step keeps its plain label with "(cancelled before the ... step)". **Compare setups** keeps the rows that finished when you cancel, and uses exactly the station planner's settings. Errors show as one line in the status text; the full traceback goes to the log. If the separate planner process crashes, the message gives its exit code and the likely cause; if the parallel check pool breaks, the run continues with serial checks and a warning.

### Several parts (batch)

**Optimize tab -> Optimizer: Several parts (batch)** plans many parts with this project's robot, end effector, external axis, stay-outs and settings: "queue now, plan later". Each part gets its own plan, its paths, a saved project and a PDF report; the tool is not changed. (Before this round there were two batch tools, v13's *Several parts* for part packages and our *Batch queue* for part meshes; they are one now.)

**1. Queue the parts.** The first job creates a batch folder; **Batch** + **Open batch** reopens an earlier one, **New batch** starts another. Under **Add jobs**:

- **Add current part**: the part as loaded, with the faces selected in the Targets tab and the point-generator settings there (a selection that gives 0 targets is refused).
- **Parts folder** + **Add folder** queues everything in that folder: every **part package** (a folder with `part.json`, made in the package builder's *Part* tab: part, its place in the cell and its targets; see *Package builder tab*) and every **part file** (STL / OBJ / PLY / STEP) as an **auto: tool surface** job: the faces come from *Detect tool surface* when the batch runs, with the current generator settings and part position. **Min confidence** (default 0.70): a part file whose detection is less sure than that is not planned but marked **needs_review**. Packages that cannot be used (unreadable `part.json`, missing mesh, no targets) and unsupported files are listed as skipped, with the reason.
- **Mode** (*Plan stations*, or *Quick plan*: the quick station plan) is stored with each new job.

**2. Edit the queue.** The **queue table** lists each job: part, faces (a number, *auto (confidence)* or *package*), targets, mode, status, reached / total, cost and cycle time, error. Pick a job in **Job** to **Remove** it, **Move up** / **Move down**, **Reset to pending** (failed and needs_review jobs) or **Open for review** (a needs_review job: its part is loaded into the Targets tab and *Detect tool surface* runs; check the suggestion, then **Add current part** and remove the old job).

**3. Run batch.** The open project is used as it is now (it does not need saving: a copy goes to `<batch>/_setup`). A batch made in another project keeps the cell it was made with, and the status line says so. **Search the axis frame per part** (rail / gantry, off by default) first runs *Gantry placement* for each part and plans the stops of its best frame; off, the project's frame is kept (one real cell, many parts). Each job becomes a part package, then the parts are planned in **fresh processes** (a big part cannot starve the next one, and a crash fails only that job), **several at once** when the PC has room: half the logical cores and 80 % of the free memory (about 1.5 GB per part) decide, and the line above the button says how many and why. The status line shows each running part's stage (loading, planning, paths, saving, report) and, once a part has finished, roughly **how long the rest will take** ("about 12 min left"). For each part the batch plans the stations (the Station planner's settings and costs), builds the paths of the best plan (cycle time), saves the part's project and prints its report. When it ends, **Results -> Batch results** opens the batch's results.

**Statuses:** *pending*, *running*, *done*, *failed* (a one-line error in the table; the traceback is in `<batch>/<job>/error.txt`) and *needs_review*. A failed part does not stop the others. A job whose part file changed after it was queued, or whose faces no longer exist, fails and says why. **Cancel** stops the parts not started and ends the running ones; the jobs that did not finish go back to *pending*, and the next **Run batch** plans only the pending jobs (finished jobs are kept), also after the app was closed.

**Where it goes.** `workspace/batch_<date>_<time>/` (installed: under `%LOCALAPPDATA%\RobotCellOptimizer`) holds `batch.json` (the queue), `parts/` and `packages/` (the queued files and the package of each job), `_setup/` (the cell), `results/` (one project folder per part: `project.json` + `assets/`, opens with Project -> Open, plans included; `report.pdf`, or `export/report.html` when no Edge / Chrome is found to print it; and `summary.pdf` / `.html` / `.csv` / `.json`, written after every part), plus `summary.csv` / `summary.html` (one row per job with the queue's columns: confidence, stops, reached / total, cost, cycle time, run time, error) and `review.html`.

**Setup warnings per job.** Every part file job gets the [Setup check](#5-optimize-tab) of its part. What it finds is listed with the job's warnings on the review page, starting with `setup:`, for example "setup: The part is only 0.60 mm across." for a file queued with the wrong units. Such a job usually fails with "no targets generated" or "smaller than one grid cell"; the warning next to the error says why.

**Review page.** `review.html` in the batch folder is ONE self-contained file (pictures inside, no network): one card per job with the iso and top picture (the selected tool surface in magenta; for a *needs_review* job the detected faces, with the secondary candidates in orange), status, confidence (auto) or manual selection, the detector's warnings and the job's error, reached / total, stations, cost, run time and links to the report and the project folder. Jobs that need a person come first (needs review, failed), then done jobs from the least confident up, then pending; the buttons at the top filter All / Needs review / Failed / Done. It is written as the jobs are prepared and at the end. **Open review page** writes it for the current batch, also for batches run before this feature (missing pictures are rendered then; the status line shows the file path: open that file in a browser). The pictures are approximate renders, not the 3D view: use them to spot a wrong selection, then open the part in *Batch results*.

**Without the GUI:** `python scripts/batch_run.py <batch folder> [--frame-search]` runs the pending jobs of a batch and prints a line per job (Ctrl+C cancels; unfinished jobs stay pending); the exit code is 0 only if every job is done. `$PY -m ui.batch_job <batch folder> --review-only` rebuilds the review page without running jobs. A folder of part packages runs without a queue: `python scripts/batch_run.py <project folder> <parts folder> <output folder> [--parts a,b] [--frame-search] [--workers N]` (the project folder is a saved project; exit code 1 when a part failed).

**While a batch runs, the Station planner is blocked** (its plan buttons answer "Cannot plan now: A batch is running"), and so are adding jobs and editing jobs that are not pending. Wait for it or click **Cancel**.

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

A 6-axis arm (UR or industrial OPW type) can reach most poses in up to 8 ways (cobots with an offset wrist, such as the FANUC CRX and ABB GoFa / PoWa, in up to 16; two of them can share a label): **shoulder** (S), **elbow** (E) and **wrist** (W) each have two branches, written as a label like `S-E+W-`. Moving from one branch to another needs a big reconfiguration motion, which you rarely want in the middle of a sanding pass. With *Joint configuration* = **zones (minimize changes)**, the planner:

1. checks, for every point, which configurations can do it from its stop;
2. picks the **fewest configurations** that cover the stop's points (exact), each point using its best solution in one of them; a stop can therefore have more than one **zone**;
3. orders each zone's points as a serpentine **path** and picks joint values (±360° copies of the wrist joints) that keep the path continuous;
4. counts **configuration changes** = extra zones at a stop + remaining jumps of joint 4 or 6 above 180° between neighbouring points, and adds *Cost per config change* × changes to the cost J. Because the planner also offers plans with more stops, a plan with one more stop and no changes can win.

**Shoulder side (FANUC CRX, ABB GoFa / PoWa).** A UR shoulder turns ±360°, so the arm can swing over the top from one shoulder side (`S+`) to the other either way round. These cobots' shoulders stop at ±180°: the only way across is through shoulder = 0, the arm pointing straight along the base's z. On the floor that is straight up and usually clear; hung from a gantry it points straight down into the part. For these robots the planner tests, at every stop that uses both sides, whether the arm can swing between them (a collision-free joint path). Where it cannot, the plan keeps **one shoulder side** (the one most points can be done on; the paths' home is on it too): each point takes its best solution on that side, a point that has none there moves to another stop that can do it, and otherwise it is reported unreachable with the reason *shoulder side*. Because unreached points cost, the planner then prefers a plan with one more stop. This applies with either *Joint configuration* setting.

It also re-runs the localization checks for the chosen solutions. It used to be 3 to 6× slower than *best per point*; it now skips arm poses of configurations already found and reuses checks within a plan, so it costs little extra. When the points a stop does in its most common configuration leave only a few others, and those can join it, only those few are searched again. The Results tab lists the zones (`S1·Z1 S-E+W- 6 pts`), draws each zone's path as a thick coloured line, and **Targets → Display → Color points by** `zone` or `configuration` colours the points of the selected solution. The CSV and the HTML report list zone, path position and configuration per point. The Tool geometry optimizer uses the same setting.

### 6 Results tab

- **Compared robots** (v9; shown after Optimize → Compare setups): *Robot* lists the project's setup and every compared one. Picking a compared setup puts its robot and its tool in the cell (3D view, jog, Tool tab, paths) and lists **its** plans: select one, look at the robot at each stop, build its paths and play them. The project is not changed: a banner says which setup is shown and which robot the project uses, and saving saves the project's own robot, tool and plans. **Back to the project's robot** brings them back. **Use this robot in the project** switches the project to the shown robot and its plans, after a confirmation (the project's previous plans are dropped).
- **Summary card** (top): the selected plan at a glance: stations, joint-configuration groups, points reached; the worst clearance, the closest joint limit and the lowest manipulability (in % of the robot's maximum after a plan in this session; after reopening a project, as |det J|); which planner made it (cost based, quick, rules only and its objective, tool optimizer), whether the rules are met, and the run time.
- **Inspect result** (folder, right under the plan list since v11; moved here from Optimize): *Show* = the feasibility check, or (the default) the **plan selected above**; it switches to each new plan or tool result automatically. **Move robot to this solution** moves the carrier (or base) to the point's station and the arm to its solution; **◀ previous point / next point ▶** steps through the points and moves the robot to each one at once.
- **Solution**: the ranked list (best cost first); **◀ previous / next ▶** switches between the plans (with *Move robot with the plan* ticked the robot goes to the new plan's first stop), and a table compares all plans (stops, points reached, cost; ⚠ = out of date). The feasibility check, *Inspect result*, *Move robot to station* and the paths all use the selected plan; paths built for another plan are cleared. It shows a summary (with the zones and configuration changes when zoning is on), the cost breakdown bars and the station list.
- **Scene**: coloured station markers, **lines to served points**, **ghost robots** posed at a real solution, and the **station areas on the part**: the part surface painted in each stop's colour (the colour of its lines), a stop's second and third groups in lighter shades. Without zones, each joint configuration a stop uses is its own group (the most used configuration in the stop's colour, the others lighter), so you can see where the arm has to change configuration. A coarse part mesh is subdivided for this display only. If a stop's area is still in pieces after planning, the summary says so (⚠).
- **Move robot to station**: moves the carrier to the station's axis values (the robot base rides on it; static: the base itself) and the arm to the first point that station serves.
- **Paths & simulation** (folder, open, above Export; the planner and the tool optimizer point to it when they finish): **Build paths** (greyed out, with the reason below it, until there is a plan) connects the selected solution's points **stop by stop**: every point of a stop is done before the carrier moves on. Within a stop each group (zone, one joint configuration) is ordered by the **Pattern** (as in the sand-optimizer): **Zigzag** (rows back and forth), **Unidirectional** (every row the same way), **Cross-hatch** (zigzag, then again at +90°), **Concentric** (rings from the border, *Outside → in* or inside out), **Spiral**, **Hilbert** (space-filling curve), **Shortest path (TSP)**, **Contour-parallel** and **Spiral (along the surface)** (v10: passes at equal distance from the group's border measured along the surface, so they also work on faces that wrap around, such as a cylinder side or a bent panel, where *Concentric* / *Spiral* work in a flat projection and jump between rings; contour-parallel closes each pass and steps in, the spiral joins the passes into one), or **Planner order** (the planner's own order). A target group can have its own pattern (Targets → *Groups & tolerances* → *Path pattern*). **Lift on wide turns** (v10, row patterns): the move from one row to the next lifts off when it is longer than the tool's footprint diameter (the first TCP's contact parts / faces; 50 mm without). Rows run at *Raster angle* from the part X axis and start at the *Start corner* (`min-min` = lowest row, low end); *Row tolerance* (0 = half the point spacing) decides which points share a row or ring. Only the chosen pattern's settings are shown. Groups follow each other nearest first, and each later group starts at the corner nearest where the previous one ended. *Point order*: **Pattern order** (the pattern above) or **Least joint motion (GTSP)** (each group re-ordered so the arm moves least; see [Optimization methods](#optimization-methods)). **Prevent crossing paths**: a move that would cross an earlier move of the same pass, or **run over another target point** (within 0.4 × the point spacing; such a point would be sanded twice), lifts off the surface by *Lift height* instead, so each point is sanded once. A move along the surface that would take a joint past its limit (for example a CRX wrist, joint 4 only turns ±190°) becomes a joint move instead: the tool backs off, the wrist unwinds and the tool comes back, so every point is still touched (before, the arm stopped at the limit and those samples were flagged unreachable). **Home between stops** (on by default): each stop starts with a joint move from the robot's home pose and ends with one back to home. **Between groups**: how the arm gets from one group (zone, face or joint configuration) to the next at the same stop. **Collision-free joint path** (the default) plans a direct joint move around the part, floor, stay-out zones and carrier (the tool backs off first when needed), also for a change of joint configuration, and goes via home only when no clear way is found; **Via home** always goes back to home in between. Stops start and end at home either way when *Home between stops* is on. **Drive carrier between stops** (on by default, independent of home): the mobile base or external axis move between stops is planned and played back; with home on the arm rides at home, with home off the tool first backs 150 mm straight off its last point and the arm rides in that pose, then a planned joint move takes it to the next stop's first point. If that backed-off pose would hit something on the way, the arm goes to its home pose for the drive instead, and a direct joint move that cannot get around an obstacle goes via home when that is clear. When no way is found to a point, the planner also tries the same arm pose with joints a full turn apart (e.g. the shoulder at +14° instead of −346°), which can avoid a long swing through the carrier. Untick it to plan the stops without carrier moves. The carrier moves along a rail along its travel, a mobile base along its contour path when both stops use the same robot side, otherwise in a straight line), and every carrier pose on the way is checked against keep-outs, the part standoff and stay-out zones. When that move is blocked, a **mobile base drives around** the stay-out zones, keep-outs and the part (a route planned on the floor, turning where needed) and keeps its **standoff distance** from the stay-out zones on the way (the *Fixed standoff* on a contour, else the minimum part standoff, at least the keep-out clearance; close to a stop the keep-out clearance the stop was planned with is enough). Rails and gantries cannot drive around: a blocked move is flagged `carrier_blocked`. With a **lift**, the lift moves only while the carrier stands: at the stop it lowers to the travel height (the lowest height, or the first of the two stops' heights and the lift levels at which the drive, the lift moves and the arm at home are all clear), the carrier drives at that height, and at the next stop the lift rises to that stop's height before the arm leaves home. The home pose is the one set under Robot & cell → *Home position*. **Moves to and from home are collision-checked and planned** against the part, the floor, stay-out zones and the carrier's own bodies: each stop starts with the joint values nearest home (the same arm pose, without a joint turning the long way round). A straight joint move, or one joint group at a time (arm, base, wrist), is used when it is clear. Otherwise the tool first backs straight off the surface (150 mm along its axis, same configuration) and the arm finds a way around. Only when no way is found is the straight move shown, with its collisions flagged. If the **home pose itself collides** at a stop (e.g. a robot hanging upside down from a gantry whose default home points down into the part), the summary says so: set a home that is clear there (the *gantry_overhead* example saves a folded home, 0, −165, −150, −180, −90, 0). **A move along the surface where the arm or tool would collide** (typically a long move across the part that swings the upper arm or forearm into it, the carrier or a stay-out zone) **is replaced by a planned joint move**: the tool backs off, the arm goes around, and comes in again at the next point (grey, like the other joint moves). Otherwise, between two points on the surface the TCP follows the part (sampled every *Path sample step*, projected onto the part, standoff kept) and turns smoothly from one planned tool orientation to the next; every sample is solved for the arm in the same joint configuration, closest to the previous sample. The summary lists flagged samples (red dots in the scene):
  - **unreachable** (no solution within the joint limits), **config flip** (only another shoulder / elbow / wrist configuration reaches it), **jump** (a joint moves more than *Max joint step* between neighbouring samples), **collision**;
  - **reconfigure**: the plan itself switches configuration there. With *Between groups* = **Collision-free joint path** the arm changes configuration by a planned joint move that keeps clear of the part (backing off first); with **Via home** it goes back to home and comes in again in the new configuration. Without zones, the path does all of a stop's points in one configuration before the next (most used first), so a stop changes configuration once per extra configuration, not each time neighbouring points differ. Each **target group** (face) is a path of its own too: the tool is not dragged over an edge of the part from one face to the next (a planned joint move joins them), and a zone's faces are done one after the other before the next zone. **Fewer changes:** set *Joint configuration* = **zones (minimize changes)** in the Station planner (the default; it picks the fewest configurations per stop that still reach every point, and *Cost per group* makes the planner prefer stops that need fewer), or tick *No configuration change within a group* under *Costs & requirements* to require it.
  - **carrier blocked**: a carrier pose between two stops hits a keep-out, the part standoff or a stay-out zone.
  - **motion** (with *Check moves between points*, on by default): both ends of a move are clear, but the arm hits something on the way between them (it sweeps through the part, a stay-out zone or the carrier). Every move is checked every 2° of joint motion; the flag sits on the move's end sample. The surface moves are straight TCP lines in the robot program; between samples a few millimetres apart the check stands for them too. Carrier moves are checked when they are planned (carrier blocked).
  - *Optimize tool spin* (on by default): where a group's spin is free (or a range), the paths also try choosing the tool spin at every sample (in 15° steps, least joint motion including the moves to and from home, away from the joint limits) and keep whichever version has fewer flagged samples, then the shorter cycle time; the cycle-time line says which (*Tool spin: optimized along the path* / *as planned*). Building takes about three times as long with it on. The planner's points keep their planned spins in the CSV export of the plan; the path CSV has the spins actually used.
  - *Smooth joint moves* (v10, on by default): the planned joint moves between points (to and from home, between groups, configuration changes) are optimised after they are found: fewer zig-zags, a shorter joint path and at least 50 mm from the part where possible (a CHOMP-style optimisation started from the found path). A move is changed only when the result is better, collision-free and no closer to the part; otherwise the found path is kept. On the CRX-30 overhead-gantry example 5 of 14 moves changed, 24 % less joint travel on those (cycle 459 s → 457 s); building the paths takes longer (2 min → 3.5 min there): untick it for faster builds.
  - **Carrier motion** (v10; a movable rail or gantry only): **Uncoordinated (stop and go)** (default: the carrier drives to a stop, the arm works, as before) or **Coordinated (the carrier follows the tool)**: the carrier moves while the arm works, so a long part is worked in continuous passes. Each target group (face) is one pass over all its reached points from every stop, in the face's most common joint configuration (one pass per configuration only when that has fewer problems), rows along the rail by default (the *Pattern* above applies). In a pass the carrier keeps the arm at the offset along the axis it had at the plan's stops and the arm does the rest; a gantry follows along the axis the part extends most and holds its other axis where the stops had it. Between passes the arm goes home (*Home between stops*), the carrier drives and the arm comes back. Every sample is solved and collision checked from its moving base; playback moves the carrier along; the cycle time counts the carrier as one more axis. The report's *Cycle time* section compares both when both were built for the plan. rail_long_part: **72 s coordinated vs 83 s stop and go** (no drives between stops, one settle), both clean. Not every cell gains: on gantry_overhead (a tall wall worked from above on three faces) coordinated took 789 s vs 428 s and had collision samples (the moving arm poses are not the ones the planner checked), so keep stop and go there.

  The paths are drawn as coloured lines (one colour per zone); lifts, home moves and carrier moves are thin grey lines. **▶ Play** / **⏸ Pause** moves the arm along the samples and the carrier between the stops: with **Play at program timing** (v10, on) as the cycle time has it (real time), else at *Speed (mm/s)*; *Sample* scrubs; **⏮ Back to start** stops and returns to the first sample (Play at the end also starts again from the beginning). **Clear paths** removes the built paths (lines, issue dots and playback) until you build them again. **Export path CSV** writes `export/paths.csv` (stop, kind = surface / lift / home / carrier, the carrier values, zone, from / to point (0 = home), TCP pose in the world in mm + quaternion, joints in degrees, issue per sample). It is a preview of the motion, not a timed robot program.
- **Pareto front** (folder, shown when the list holds a Pareto front): the plans as numbered dots over two measures (*X axis*, *Y axis*: stations, cycle time, worst clearance, tool length); the numbers are the **Pareto i/n** plans in the list. **Click a dot** (or a *Select plan* button) to select that plan. See [Optimization methods](#optimization-methods).
- **Edge check** (v10, folder, closed): the planner checks only the target points and assumes the arm also reaches the surface between them. That fails first where something changes, so the edge check puts extra verification points there and checks each one exactly: **between stops** (halfway between neighbouring targets served by different stops, checked from either stop), **between zones** (between neighbouring targets in different joint-configuration zones of a stop, in either zone's configuration), **part edge** (along the edges of the targeted surface and the creases inside it, one point per side of a crease, only within reach of the target points) and **low margin** (a ring at half the point spacing around each point that was only just reached: marginal, or a joint within 15° of its limit). Targets on opposite sides of a thin wall get no points between them. Work points inside a zone stay sparse. *Check after planning* (on by default) runs it on the best plan after each **Plan stations** run; **Run edge check** runs it on the selected plan (greyed out until a plan is selected; worker process + check workers; **Cancel**). It reports how many points are not reached, by kind and stop, in the panel and the Log, marks the gaps on the part with cyan dots and crosses (with *Highlight unreachable points*) and colours the part with *Colour part by → Edge check*. **Densify edges** adds the gaps as target points (each in its nearest target's group; Targets → *Undo last add* removes them again); plan again to include them. Examples: rail_long_part and mobile_amr, all 101 edge points reached in under 0.5 s; gantry_overhead (UR20), 19 of 330 not reached (13 low margin, 4 between zones, 2 part edge) in 8 s with 4 check workers.
- **Tolerance check** (folder, closed): does the selected plan survive fixture and positioning error? It moves the part by a random offset (up to *Part offset ± (mm)* in each axis) and turn (up to *Part tilt up to (deg)*, about the part's centre), and each stop by up to *Stop offset ± (mm)* in x and y (carrier positioning error), *Samples* times (the first sample is the nominal setup; *Seed*: the same seed gives the same samples), and checks every planned point again, exactly, from its own stop. The points move with the part, as when the program is taught in a user frame on the part: the question is whether the arm can still *reach* them. **Run tolerance check** (greyed out until a plan is selected; runs in the worker process and the check workers like the planner; **Cancel**) reports the share of samples in which **every** point was reached and lists the **least robust points**: how often each was reached, **fails from**: how far that point had moved in the smallest sample where it failed, its clearance and its joint-limit margin. Points the plan itself does not reach are not tested. The result can colour the part (*Colour part by → Tolerance reach rate*, Targets tab). Typical values: ±1 to ±5 mm and up to 1° for a fixture, ±10 mm for a mobile base (the planner's localization check already tests the mobile tolerance set on the External axis tab).
- **Safety** (folder, closed, v8): **Show reach envelope** draws, for every stop of the selected plan, a see-through sphere around the robot's shoulder with the robot's reach plus the tool as radius: everything the arm and tool could touch from that stop. Below it a **suggested fence**: the envelopes' floor area grown by *Fence distance (mm)* (default 500), drawn on the floor in orange, with its size and corner coordinates. It is a layout aid, not a safety assessment: the risk assessment (ISO 10218-2, ISO 13855 separation distances) decides the real guarding.
- **Export** (greyed out, with the reason, until there is a result):
  - **TCP poses (CSV)**
  - **Userframes (CSV)**: one row per stop: its carrier values and the **user frame**, the pose of the part frame in the robot's base frame at that stop (teach the points in it and they stay valid). The columns follow the robot model: UR arms `x_mm, y_mm, z_mm` + rotation vector `rx_rad, ry_rad, rz_rad` (as in a UR pose `p[...]`, with x y z converted to m); FANUC-style arms `X Y Z` (mm) + `W P R` (deg); every robot also gets the rotation matrix `r11 … r33`. The format comes from `frame_convention` in the robot's `spec.yaml` (`ur`, `fanuc` or `matrix`).
  - **Robot program** (v10) with *Program format* (**Auto** = the robot's own language: UR → URScript, FANUC → TP / .ls, ABB GoFa / PoWa → RAPID; or pick one. For a robot whose language is not known, such as an uploaded URDF, **Auto** refuses and asks you to pick one): writes `export/rso_path.script` / `.ls` / `.mod` from the selected plan's **built paths** (build them first). Surface samples are linear moves of the TCP in a **user frame on the part** (one per stop: the part seen from that stop's base), joint moves (home, retreats, moves around the part) are joint moves with the planned joints, and the tool is defined from the TCP. Between stops the program calls a `rso_carrier` routine (and says where the carrier goes) for the integrator to fill in. **Coordinated** paths use one user frame and give each point the carrier values as extended axes (FANUC `E1`.., RAPID `extax`, mm); URScript has no external axes, so there each move is a base-frame pose preceded by `rso_carrier(...)`. Configuration data comes from the planned joints (RAPID also turns `ConfL` / `ConfJ` off). Path samples without a joint solution are left out with a `skipped` comment; paths that use more than one TCP are refused (one program per TCP is not supported yet). Speeds come from Optimize → Cycle time. **Check every program in the robot maker's simulator (URSim, ROBOGUIDE, RobotStudio) before running it on a robot**: they have not been run on a controller.
  - **Optimized tool URDF (zip)**
  - **Export report** with *Report format* = **HTML** (v9: one button, the format chosen above it): self-contained, printable to PDF, with an optional 3D snapshot. Since v8 it opens with the customer sections: the **Summary** card, the **Cycle time** (the plan's estimate, and the built paths' when they belong to this plan), the **Tolerance check** of this plan (when one was run), the **Setup comparison** with a picture per setup, **Gantry placement** and **Carrier motion** (v10, when run / built), the **Tool fit** (v10: points below the tool's minimum surface radius) and the **Unreached points** with their reasons.
  - *Report format* = **PDF** (the default): the same report printed to `export/report.pdf` by Edge headless (Chrome works too); without a browser the HTML report is still written.
  - **Report 3D views**: **Include 3D views in report** (on by default) and the checkboxes **Report view: Iso / Top / Front / Side** (Iso and Top ticked) choose the pictures put into the HTML report. Each is rendered by your browser from that side, framed on the cell, with the grid, gizmos and selection hidden; your camera is put back afterwards. They need the GUI open in a browser, so a report made by the batch has none.
  - **Tool (STEP)** (v8): the selected result's tool (its optimized values, else the nominal) as a STEP file for CAD: the extension as round tube legs (the *Tube OD* chosen under Tool geometry, else the tool's radius) with a ball at each bend, plus an **ISO 9409-1 flange plate** at the mount (*Flange*: 50-4-M6 for UR arms, 63-4-M6, 80-6-M8; 8 mm thick, with the bolt holes and the centring bore). The head is not in it (use its own CAD). Made with CadQuery, which is part of the environment and the portable copy.
  - **Everything (zip)**: the handover package. One click writes the TCP poses, the userframes, the HTML report (with the 3D views), the tool URDF, the path CSV (when paths are built for this plan) and a copy of `project.json` and puts them in `export/<project folder name>_handover.zip`, with a `README.txt` that says what each file is, which plan and program version it came from, and what could not be made (one failed export does not stop the others). Part and tool meshes are not in it, apart from the tool URDF's own.

  Files go to `<project folder>/export/` and also download in the browser.

**Batch results (v13).** *Batch results* reads the results of a batch run (`<batch>/results`, Optimize ->
Several parts; it opens by itself when a batch ends, and the last one comes back at the next start).
It shows the summary table (reached points, stops, cycle time, cost per part; failed parts with the
reason). Pick a **Part** and press **Open this part**: its solved project replaces the open one
(asks first when the open project has unsaved changes), so its plans are in the list above to view,
inspect, check and simulate; with **Build paths when opened** its paths are built straight away
(paths are not saved in a project). **Download report** sends the part's PDF report.
  **What the exports say (v8 review):** the HTML report ends with a **Not checked** section: payload / torque / dynamics, transit between stops, kinematic-only motion, sampled spin / tilt and collision geometry are not guaranteed by the plan. The TCP CSV starts with `#` comment rows (provenance and warnings): **skip lines starting with `#` in your own CSV readers**. An unmeasured clearance is an empty cell. **Export anyway** on an out-of-date plan writes files with an **OUT OF DATE** banner. The Universal Robots graphics notice appears only for robots with `mesh_license: ur` in their spec.yaml. If the PDF fails, the message says whether the browser was missing or failed.

### Collision geometry

Every collision check (planner, tool optimizer, paths, jog) uses a **collision copy** of each mesh; the 3D view always shows the full visual meshes. Large CAD meshes (a detailed end effector, a part with many small features, a rail with bolts and cable chains) make every check slower, so they are simplified for collision. **Since v9 all of it is made in the [Package builder](#package-builder-tab)**: for the end effector, the part, a rail, a gantry or a mobile robot, choose how its meshes are simplified, its padding, or import your own collision mesh, press **Simplify**, check the result, and **Use in this project**. A mesh that has no collision copy of its own is reduced automatically when it is large (*Auto*). The simplification modes:

| Mode | Collision copy |
|---|---|
| **Auto (reduce large meshes)** (default) | Meshes above 20 000 triangles (robot links: 5 000) are reduced to *Triangles*; smaller ones are used as they are. |
| **Same as visual** | The full mesh. |
| **Reduce triangles** | Quadric decimation to about *Triangles* (the shape stays within a few mm). |
| **Convex hull** | One convex hull around the whole mesh (at most about *Triangles*): fast and conservative, but fills hollows and openings. |
| **Wrap (offset shell)** | A closed shell *Wrap offset* (mm) outside the surface, like an alpha wrap: many small components, holes and gaps become one simple body, slightly larger than the visual one (never inside it). For complex assemblies. On a large mesh the offset is at least about 1/100 of its size. |

**Robot & cell → Collision geometry** keeps the robot's own setting: **Padding per link (mm)** (v10) pads chosen links on their own instead, e.g. `forearm_link 20, wrist_1_link 10` (the robot's link names; a link not listed uses the arm padding; a line under the field lists what was not understood). **Robot arm padding (mm)** makes the robot arm's collision shapes this much larger against the part, the floor, the carrier bodies, the stay-out zones and the tool (not against the arm itself), on top of the clearances; for example 10 mm to allow for cable ties, covers or model error. It applies to the planner, the paths and the jog, and the left panel's **Show robot** draws the arm bloated by the same amount. Below it, a line says whether the part has its own collision mesh (made in the builder), and **Simplify the part in the package builder ↗** opens the builder.

**Left panel → Collision geometry**: **Show robot / tool / part / external axis** draws each component's collision shapes in orange, exactly as they are checked.

Simplified meshes are cached in `cache/collision` (per PC), so a large mesh is only simplified once. The ABB GoFa 10, GoFa 12 and PoWa 30 ship with wrapped collision meshes (their URDFs used the CAD meshes, which made each check about 10× slower than for a UR or CRX).

For a simplified (decimated) mesh the largest deviation from the visual mesh is measured and cached; a warning appears when it exceeds half the clearance. The report lists the collision geometry used for each body. In the Jog panel, if the collision scene cannot be built, steps are blocked unless **Allow collisions** is on, and the readout shows a warning.

### Package builder tab

Builds a **URDF package** (a zip: URDF + meshes + `package.yaml`) for an **end effector**, the **part**, a **rail**, a **gantry** or a **mobile robot**, from boxes, cylinders and meshes, with its **collision geometry**, and loads it into the project. Since v9 this is where all the project's geometry is made. Step by step: [Building URDF packages](#building-urdf-packages).

| Control | What it does |
|---|---|
| **Mobile robot** (v11) | **Add mobile base** puts a chassis box on the floor; **Add vertical stage** puts a mast (*Mast height (mm)*) on the chassis and a plate that rises on it (*Stage travel (mm)*: sets *Lift travel*), with the **robot mount on the plate**, so the URDF chain is floor → x → y → yaw → chassis → lift → robot mount and the planner uses the lift as a DOF. Both are ordinary parts afterwards: resize them, move them or replace them with meshes (e.g. your AGV's STL on the chassis stage). |
| **Package** (tabs) / **Name** | **End effector**, **Part**, **Rail**, **Gantry**, **Mobile robot**: a tab bar at the top, the open type highlighted (and named in the line under it). Changing the type keeps the parts (stages that do not exist in the new type are moved). A **Part** package is the workpiece: its meshes, in the part's own frame, and its collision geometry (no TCPs, no travel, no robot preview). |
| **From the project** | **Edit the project's tool / part / external axis** loads what the project uses into the builder: the tool's URDF and TCPs, the part mesh (or the part package made here before, with its settings), the carrier's bodies, travel and robot mount. **Use in this project** then replaces it; a tool edited this way keeps its process and extension, and its contact settings come back into *What may touch the part*. Each part of a tool is written to its own URDF link (named as in the tool's URDF), so contact parts such as a sanding pad stay contact parts; contact parts or faces that no longer exist (a renamed part, a changed mesh) are listed in the builder status and the log as **Contact settings not carried over**. An external axis edited this way keeps its bodies' collision meshes and which bodies follow the axis length (*Follows axis length*), and the project's axis lengths stay in charge of the travel. |
| **Parts** | **Add box**, **Add cylinder**, **Add mesh (STL/OBJ/PLY/STEP)** (*Mesh units* for STL / OBJ / PLY; STEP files are meshed with CadQuery in metres). v12: **Load relative to** (end-effector / tool packages only; a tool package's frame is the robot flange, z out of the flange face): **Robot flange** (the default) puts the file's own origin and axes on the flange; **STEP coordinate system** puts a named coordinate system of the STEP file there instead (the mounting face): one in the file is used, several open a dialog to pick it (or keep the file origin), none keeps the file origin with a note. STL / OBJ / PLY files have no coordinate systems (the flange is used; move the part with its pose fields). The import line says what was used. v10: *Mesh units* = **auto (from its size)** (the default) reads the units from the mesh's largest side: 100 to 5000 (file units) = mm, 0.005 to 5 = m; over 5000 (a part over 5 m long) and in between (a 12 m frame or a 12 mm piece? 30 mm or 30 inches?) the part is added with the likelier one and a dialog asks (mm / m / in); the status line says what was used. **Units of this mesh** changes a mesh part's units afterwards. **Cancel import** (STEP) stops the meshing, which runs in its own process; nothing is added. The import runs in the background with a progress bar and a status line under the button (a STEP file takes from seconds to minutes); a file that cannot be read is reported there, nothing is added, and **Check** lists it until a good file is added. Pick a *Part* to edit its name, **Stage** (which moving stage it rides on: *tool*; *fixed* track, *bridge* moving in X, *carriage*; the mobile *chassis*; *lift stage*), size and pose (fields in mm / deg, or drag its gizmo). Boxes and cylinders sit on their bottom face, centred. **Remove part**. |
| **Frames and travel** | Every mm / deg field in the app takes up to four decimal places (v12; the arrows keep their steps). **Frame from STEP** (v11): the named coordinate systems (CAD datum CSYS features) of the STEP parts imported here, as `part / name`; **Assign to** *TCP (selected)*, *Robot mount*, *Vertical stage mount* (the lift stage's parts move so the first one sits on the datum, the robot mount rides along) or *Package origin* (the datum becomes the package frame: the flange of a tool, the floor frame of a carrier; every part, the mount and the TCPs are re-expressed in it), then **Assign**. They are kept with the part (package.yaml). Only placements with a name are read, and in an assembly a sub-part's own placement is not applied to its datums. Tool: the **TCPs** (**Add TCP**, name, pose; z out of the tool toward the part). **What may touch the part** (v10, per TCP chosen above): tick **Selected part may touch** for a whole part (a pad, a brush, a nozzle tip; Parts → *Part* chooses which), and / or **Pick faces** on the selected mesh part in the 3D view: **Click +/−** adds / removes a smooth surface (it grows across edges up to *Surface angle*), **Brush +/−** a dab of *Brush radius* around the click, **Box +/−** every face inside a dragged rectangle; **Clear picked faces of this part**. Contact parts are drawn green in the preview, picked faces as a green layer; the line under the fields says what may touch (e.g. "TCP **pad** may touch with: 48 faces of **pad_link**"). **Minimum surface radius (mm)** and **Radius limit applies to** set this TCP's minimum radius (as in the Tool tab). **Travel direction**, **Scan width (mm)** and **Overrun past edges (mm)** (v11) as in the Tool tab. Everything not picked still counts as a collision with the part (the faces right next to picked ones are tolerated, as in the Tool tab). The picks are saved in package.yaml and become the tool's contact parts / faces on **Use in this project**; renaming a part keeps them, removing it drops them. Carriers: the **Robot mount** pose (where the robot's base sits; a gantry's is usually turned 180° about x so the robot hangs), **X travel** (rail, gantry), **Y travel** (gantry) and **Lift travel** (any carrier; 0, 0 = no lift), in mm. |
| **Open package builder ↗** | The builder opens in **its own browser window** (the app's port + 1, e.g. `http://localhost:8081`), so its parts, frames, gizmos and robot never appear in the project's 3D view, and nothing stays behind when you close it. The controls below are in that window. |
| **Preview** | *Show preview* draws the package: parts coloured by stage (tool contact parts green), the TCP or robot-mount frames, the package frame's axes. A rail, gantry or mobile base stands at *Preview at*. |
| **Collision geometry** | How the package's parts are simplified for collision: *Simplify* (*Auto*, *Same as visual*, *Reduce triangles*, *Convex hull*, *Wrap*), *Triangles*, *Wrap offset (mm)* and **Padding (mm)** (every part's collision shape that much larger: boxes and cylinders exactly, a wrap by a larger offset, other meshes pushed out along their surface). **This part** gives the selected part its own mode. **Collision mesh for this part** imports your own low-detail mesh (STL / OBJ / PLY, the part's units and origin) instead (**Remove its collision mesh** undoes it); a URDF opened here with its own collision meshes brings them in the same way. Part packages: **Keep the project's selected surfaces exact** (after *Edit the project's part*) keeps the faces selected in Targets at full detail; on a part made of several meshes (v10) the selected faces are found on each mesh by their position, so the right faces of each mesh stay exact. Editing a setting computes nothing: **Simplify** makes the collision copies in the background with a progress bar (**Cancel**, v10: stops after the part being simplified; the finished parts stay ready), and the status lists each part: *not simplified*, *ready, N → M triangles*, or *out of date* when a setting changed since. **Show collision shapes** draws them (orange). **Revert to original** drops the simplification and the padding (the visual meshes; imported collision meshes stay). Export and **Use in this project** write the copies to `meshes/collision/` and the URDF's `<collision>` elements (the `<visual>` keeps the full mesh); a part package gives the project's part its collision mesh. Boxes and cylinders without padding are used as they are. The settings are saved in `package.yaml` and come back with **Open package**. |
| **Preview on robot** | *Show on a robot* puts the package on a robot from the library (*Robot*, default the project's): a tool on the flange, or the robot on a carrier's *Robot mount*. The *J1..J6* sliders pose the arm and **Home pose** resets it. Only the builder window shows this robot: the project's robot, tool and external axis do not change until **Use in this project**. |
| **Check** | Lists what still stops an export (no parts, a box without size, a tool without a TCP, a missing travel range...). |
| **Export package (.zip)** | Writes `<project>/export/packages/<name>/` and `<name>.zip`, and downloads the zip. |
| **Use in this project** | Asks first (it replaces the project's tool, part or external axis), then writes `<project>/assets/packages/<name>/` and loads it, and the builder window closes (if the browser does not allow that, close it yourself). A tool becomes the current tool (Tool tab: set the process and extension ranges, then **Save to library**); a part package becomes the part, keeping the part's mesh file when its geometry did not change (so targets and the surface selection stay valid) and giving it the collision mesh made here; a rail, gantry or mobile robot becomes the external axis (External axis tab: its bodies, travel ranges and robot mount are set; the standoffs, keep-outs, carrier frame and the mobile base's floor area stay as they were). |
| **Open package (.zip)** | Reads a package made here, by hand or exported from CAD (see the conventions below) back into the builder to edit or use. |
| **New package** | Starts an empty package of the current type. |

**Part packages for batch runs (v13).** In the **Part** tab, the folder *Part package: world
position and targets* makes the packages that Optimize -> *Several parts (batch)* runs on. A part
package is a folder `<parts folder>/<name>/` with `part.json` (the part's units, its pose in the
world frame and its target groups and points) and the mesh (and the collision mesh made here).

1. Add the part's mesh in **Parts** (or leave the builder empty to use the project's part as it is),
   and give the package a **Name**.
2. Set **Position in world (mm)** and **Rotation in world (deg)** (*Take the project part's
   position* copies the current one) and press **Place in the project**: the part goes into the
   project at that pose.
3. In the main window's **Targets** tab, make its targets as usual (groups, tolerances, patterns).
   A note there says which package is being edited.
4. Type the **Parts folder** and press **Save part package** (here, or the same button in the
   Targets tab). It asks first when the part has no targets or the name is already in the folder.

**List the folder** and **Open part package** bring a saved package back into the project (part,
pose and targets) to change it; save it again under the same name to replace it. The robot, tool,
external axis, stay-outs and planner settings are not in a package: a batch takes them from the
project it runs in.

---

## 5. Worked examples

### Example A: sand a flat plate from one base (static)

1. Cell → *Part* `flat_plate`, xyz `900, 0, 300` mm.
2. Targets → *Tool* `Fill +` → click the plate top → *Row* / *Column spacing* `100` → **Generate on selection**.
3. Optimize → **Check all targets**. Everything should be reachable from the origin.
4. Optimize → **Plan stations** → Results: "1 floor station: N/N reached".
5. Optimize → **Optimize tool**. On this plate the shortest straight tool is the bottom of the range (50 mm), and the bend stage may lower the cost further.
6. Results → **Export report**.

The same job without the GUI is in `examples/headless_plan.py` (section 7).

### Example B: a long rail part with a 7th-axis rail carrier

Shortcut: **Project** → *Examples* `rail_long_part` → **Open example**, then step 4.

1. Cell → *Part* `long_rail`, xyz `0, 800, 500`.
2. Targets → *Select by normal* `(0,0,1)` within `5` → *Spacing* `250` → **Generate**.
3. External Axis → Carrier `rail` → **Load carrier defaults**. Set *Track/gantry frame xyz* to `0, 150, 0`, so the 0.5 m carriage sits exactly 300 mm from the part. Optionally add a wall keep-out `-5000,1300,5000,3000`. Click **Apply base settings**.
4. Optimize → Settings → *Max stations* `5` → **Plan stations**. Expect 2 rail stations for the top face; the Results tab shows the carriage positions *s*.
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

### Example B5: many parts in one batch

1. **Project -> Examples -> rail_long_part** (UR20 on a rail, orbital sander).
2. Optimize -> Optimizer **Several parts (batch)**. Parts folder: `examples/batch_parts` (three
   packages made by `scripts/make_batch_example.py`: a flat plate, a curved panel and an open box,
   each beside the rail with targets on its upward faces). **Add folder**: three jobs, *package*.
3. **Run batch** (the batch folder is in the workspace): the
   three parts run at once on a PC with 6+ cores (about 45 s). The plate and the panel are reached
   everywhere; the box's pocket floor (0.45 m deep) is out of reach of the sander, so 8 of its 16
   points stay unreached (the report says why).
4. **Results -> Batch results**: pick *open_box*, **Open this part**; the plan and its paths are in
   the Cell. **Download report** for its PDF; `results/summary.pdf` in the batch folder lists all three.

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
   - `offset_wrist`: cobots whose wrist is not spherical (FANUC CRX, ABB GoFa / PoWa): joint 1 vertical, joint 2 horizontal, joint 3 parallel to joint 2 and the joint 4 and joint 5 axes intersecting; no parameters (everything is measured from the URDF). It sweeps joint 6 and solves the rest in closed form (up to 16 solutions; compiled with numba). Poses right at a singularity can be missed.
   - `numeric`: any chain. Display, jogging and the feasibility check work; the station planner, tool optimizer and heatmap do not (they need the fast batch solver).
4. Run `python scripts/check_robot.py <name>`. It loads the files, checks the IK against the URDF on 500 random poses (wrong OPW parameters fail here with the size of the error), and checks the base joint range, reach and home pose. Fix anything marked FAIL.

The first plan with a new robot builds its reachability map (about a minute per tool), which is then cached.

### Building URDF packages

A package describes an end effector or an external axis (rail, gantry, mobile base) as a URDF with its meshes, in a form this app reads back (and other URDF software understands). You can make one in the **Package builder** tab, or write / export it yourself and open it there.

**In the Package builder (from CAD meshes):**

1. In CAD, export each body that moves on its own (track, carriage, lift column, tool housing...) as an STL or STEP file, **all in one common frame**: for a tool, the robot flange (origin at the flange centre, z out of the flange); for a rail or gantry, the carrier frame on the floor (x along the travel); for a mobile base, the base centre on the floor (x forward). Then every part's pose in the builder is zero. Export them **at zero travel** (carriage at the middle of the rail, lift down).
2. Package builder → *Package type* and *Name*. **Add mesh** for each file (*Mesh units* for STL: usually mm), or **Add box / Add cylinder** for simple shapes. Give each part its **Stage**: what it moves with.
3. Tool: add the **TCP**(s), z pointing out of the tool toward the part. Carrier: place the **Robot mount** (the robot's base frame: z up, or down for a hanging gantry robot) and set the **travel** ranges (mm, relative to the zero position you exported in).
4. *Show preview*, check it from all sides, **Check**, then **Export package (.zip)** to keep or share it, and **Use in this project** to load it.
5. Finish in the normal tabs: Tool tab (process, contact faces, extension ranges, **Save to library**) or External axis tab (standoffs, keep-outs, the carrier's place in the cell).

**Writing or exporting the URDF yourself** (e.g. with a SolidWorks or Fusion URDF exporter): zip the folder with the `.urdf` and its meshes (relative paths or `package://` URIs) and open it in the builder. It reads any link names; it needs:

| Package | Structure |
|---|---|
| Tool | No moving joints. The root link is the **mount**: it sits on the robot flange (z out of the flange). Fixed frames whose names start with `tcp` (`tcp_pad`, `tcp`) become the TCPs (z out of the tool). The Tool tab's **Upload tool URDF (.zip)** reads these TCP frames too. |
| Rail | A **prismatic** joint along **+x** of the root (the track frame). Links above it move (carriage); links not above it are fixed. |
| Gantry | Prismatic joints along **+x** (the bridge) and **+y** (the carriage, below the bridge in the tree). |
| Mobile base | No x / y joints; the root is the chassis frame (origin on the floor, x forward). The planner drives it on the floor. |
| Any carrier | Optionally a prismatic joint along **+z**: the lift (links above it rise). A frame named **`robot_mount`**: where the robot's base goes. |

Joint axes are judged in the root frame at zero travel (an exporter's joint origins are fine); rotary joints are refused. Box and cylinder geometry, meshes (STL / OBJ / PLY; other formats such as DAE are converted to STL) and their `scale` (1, 0.001 for mm, 0.0254 for inches; others are converted) are read from the collision elements (the visual ones when a link has none). An optional `package.yaml` next to the URDF sets `kind: tool | rail | gantry | mobile`, `name` and `description`; without it the kind follows from the joints. The builder writes exactly this structure: `mount` + `tcp_<name>` (tool); `base_link` --`axis_x`--> `carriage` (rail); `base_link` --`axis_x`--> `bridge` --`axis_y`--> `carriage` (gantry); `base_link` (mobile); then `lift` --> `lift_stage` and `robot_mount`.

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
| `assets/robots/crx10ia/`, `crx20ia_l/`, `crx30ia/`, `gofa10/`, `gofa12/`, `powa30/` | FANUC CRX (FANUC's description package, Apache-2.0) and ABB GoFa / PoWa cobots (meshes from the work-r-labs robot library, MIT), generated by `scripts/import_cobots.py` |
| `assets/robots/generic_6r/` | a made-up industrial arm (OPW kinematics), generated by `scripts/make_generic_robot.py`; a template for adding robots |
| `assets/tools/`, `tools/*.yaml` | sample tool geometry and the tool library |
| `assets/parts/` | test parts (STL) |
| `examples/` | example projects (`rail_long_part/`, `mobile_amr/`, `gantry_overhead/`, `floor_xy_rails/`, `mobile_lift/`, regenerate with `scripts/make_examples.py`), `sander_mobile.json` (format sample) and `headless_plan.py` |
| `workspace/` (not in git) | your project folders: `project.json`, uploads, `export/`, and the `batch_*/` folders of *Several parts (batch)* |
| `cache/` (not in git) | reachability maps |
| `packaging/`, `scripts/build_exe.py` | the Windows build: PyInstaller spec, launcher, Inno Setup script (see below) |
| `docs/USER_GUIDE.md` / `.html` | this guide; the HTML is generated from the Markdown with `scripts/build_guide.py` |
| `docs/PROJECT_PLAN.md` | the design plan |

**Batch folder** (`workspace/batch_<date>_<time>/`): `batch.json` (the manifest: the cell study it starts from and one entry per job with part, faces or rule, generator and planner settings, status, error, run time, summary), `parts/` (the queued part files), `summary.csv`, `summary.html`, and one study folder per job (`job_001/`, ...) with `project.json`, `assets/` and `export/` (`tcp_poses.csv`, `userframes.csv`, `report.html`); a failed job also has `error.txt`.

**Data folders of the installed program.** The program files are read-only in the install folder (`%LOCALAPPDATA%\Programs\RobotCellOptimizer`, or where you unzipped). Everything the program writes goes to `%LOCALAPPDATA%\RobotCellOptimizer` (or `RSO_DATA`): `workspace\` (projects, batches, `last_project.txt`), `tools\` (your tool library; the bundled tools are read from the install folder) and `cache\` (`reachability`, `collision`, `numba`; the prebuilt reachability maps in the install folder are used until a map is built here). It also holds `logs\` (the program's log file `app.log` and up to three older copies) and `support\` (the zips made by **Save support bundle**). From source, these are the repo's own `workspace/`, `tools/`, `cache/`, `logs/` and `support/`.

**Build outputs** (developers, `$PY scripts/build_exe.py --maps --zip --installer`): `dist/RobotCellOptimizer/` (the program folder), `dist/RobotCellOptimizer-<version>-win64.zip`, `dist/RobotCellOptimizer-<version>-setup.exe` (needs Inno Setup) and `dist/SHA256SUMS.txt`. `RobotCellOptimizer.exe --selftest` checks a build headless and offline. Details: `packaging/README.md`. A GitHub Actions workflow builds and tests these on a version tag and drafts a release (first run on a fork on 2026-11-19); `.gitlab-ci.yml` does the same on GitLab, where the release waits for a click instead of being a draft (not run yet either).

**Project file** (`project.json`): the robot and base pose, base mode and carrier (its `bodies`: shape, size or mesh file, pose, attachment), tool, part, targets and groups, optimizer settings, and the saved solutions. Older projects with `chassis` / `riser` / `track` still load; the External Axis tab converts them to bodies. The `work` section holds where you left off: joint pose, selection, per-point joint solutions of the feasibility check and of each plan, and panel settings. Saves are atomic (written to a temporary file, then renamed). `workspace/last_project.txt` records the recent project folders (newest first) for the start-up window.

**TCP poses CSV** has one row per point, in station order:
- `point_id, group, status, reason, station, zone, seq, config, carrier` (zone / seq = path position / config label when zoning is on; rows then follow station → zone → path order);
- base pose (`base_x/y/z_mm, base_yaw_deg`);
- TCP in **world** (`tcp_world_x/y/z_mm`, quaternion `qw,qx,qy,qz`) and in the **robot base frame** (`tcp_base_*_mm`, `roll/pitch/yaw_deg`): use the base-frame values to program the robot at that station;
- joints `q1..q6_deg`;
- `spin_deg, tilt_deg, standoff_mm, clearance_mm, manipulability, limit_margin_deg`.

Unreached points come last.

**Optimized tool URDF** (zip): `flange → mount → extension_k → head links → tcp_<name>`. The root link `flange` attaches to the robot's `tool0`. The extension is a cylinder with sphere caps. Meshes are included.

**HTML report**: KPIs (including the estimated cycle time), the summary, cycle time, the **setup comparison** (v10; when Compare setups was run: each setup's best plan side by side, the project's own setup always first, and a plan-view picture per setup with the part's floor outline and its stops in visiting order), the **gantry placement** results (v10; when Optimize → Gantry placement was run: the planned frames, cheapest first) and the **carrier motion** (v10; when the paths were built both coordinated and stop and go: both cycle times), tolerance check, unreached points by reason, cost breakdown, stations, joint-configuration zones (counts, no point lists), a dimensioned tool sketch, coverage vs tool length (when it came from the tool optimizer), settings and notes. Since v9 it has no per-point table (the TCP poses CSV has every point), and it is laid out for printing: sections, tables and pictures are not split across pages. It is a single file with no internet access, and the browser can print it to PDF.

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

`$PY -m ui.preflight_run <project folder>` prints the [Setup check](#5-optimize-tab) of a saved project (exit code 1 when it finds an error); in code, `core.preflight` has the single checks (`check_part`, `check_targets`, `check_zones`, ...) and `ui.preflight_run.findings_for(session)` runs them all.

For carriers, pass `sites=CarrierSites(Carrier(base), part_mesh, T_world_part)` to `plan_stations` (a carrier with mesh bodies resolves relative paths against the repo; pass `Carrier(base, resolve=fn)` to use another folder). `scripts/make_examples.py` shows how to build a carrier from bodies in code. Build the collision scene with the carrier bodies: `bb, wb = sites.scene_bodies()`; `CollisionScene(..., base_bodies=bb, world_bodies=wb)`.

---

## 8. Troubleshooting and limitations

| Symptom | Fix |
| --- | --- |
| `conda env create` fails or `import pinocchio` errors | Use Miniforge (conda-forge), not pip. Keep the OpenBLAS line in `environment.yml`. |
| Python crashes silently on matrix maths | The MKL BLAS problem: run `conda install -n rso "libblas=*=*openblas"`. |
| The first heatmap or plan takes ~1 min | The reachability map is being built (once per tool shape); later runs are fast. |
| A plan shows "out of date" after upgrading, or after changing the margins | Expected: the part fingerprint now includes the mesh content, and the margins are part of the plan. Plan again. |
| "Software WebGL rendering" banner | The browser has no GPU acceleration; enable hardware acceleration in its settings. |
| Port already in use | `app.py --port 8090`. Viser may also pick the next free port by itself; check the console output. |
| **Open user guide** does nothing or shows an error page | The guide server uses the app's port + 1; the console prints its address at start-up. If that port is blocked (firewall when opening from another PC), use **Download user guide (HTML)** or open `docs/USER_GUIDE.html` directly. |
| The app opens an empty cell at start | That is the start-up state: pick a project in the **Open a project** window (also *Project → Load*), or start with `--open <folder>`. |
| A batch job says **needs_review** | The tool-surface detection was less confident than *Min confidence*, so the job was not planned. Select it, **Open for review**, check the suggestion in the Targets tab, then **Add current part** with the checked selection (or **Reset to pending** to run it again). |
| Planning says **A batch is running** (or adding a job says **Batch running**) | The batch uses the planner, so the Station planner and queue edits wait. Let it finish or click **Cancel** in the Several parts (batch) folder. |
| A note says **Display simplified: ...k → ...k triangles** | The part has over about 300,000 triangles, so the 3D view shows a simplified copy to keep the browser fast. Planning, collision and face picking use the full mesh; only the picture is coarser. |
| Windows shows **Windows protected your PC** when starting the installer or program | The build is not code-signed. Click **More info**, then **Run anyway**; on a managed PC ask IT to allow the file (see [2.1](#21-download-and-run-windows-installer-or-zip)). |
| The 3D view is slow, or the shadows stutter | Untick **Shadows** (Objects panel, View), or set **Lighting** to *Flat*. |
| A batch report has no 3D pictures | Expected: the pictures come from the browser. Export the report from Results with the study open in the GUI. |
| Something fails and you need help | **Settings & help → Save support bundle (zip)** and send the zip with a description of what you did. It has the version, the log and the study's `project.json`, no part or tool files. |
| A plan leaves many points unreached, or a batch job fails with "no targets generated" | Run **Optimize → Setup check**: it finds a part loaded with the wrong units, normals pointing into the part, points off the surface or inside a stay-out zone. |
| The planner puts a mobile base under a raised part | Standoff is 3D; add the part's stand to the mesh or draw a keep-out. |

**Known limitations (v1)**
- Tool-optimizer costs now include carrier travel and the standoff penalty, so values differ from earlier versions; numeric IK (uploaded robots) is deterministic; contour mode supports multi-body carriages. Reachability lookups are memory-budgeted, so huge cases run in blocks rather than failing.
- Planning features (heatmap, station planner, tool optimizer) need an **analytic IK solver** (`ur_analytic`, `opw` or `offset_wrist`) and a full-turn base joint. Robots with numeric IK (uploads) can be displayed, jogged and feasibility-checked. See [Adding a robot](#adding-a-robot).
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

- **CI** (v8): `.github/workflows/tests.yml` runs lint, the guide check and `scripts/test_batches.py --fast` on Windows for every push and pull request (conda env from `environment.yml`, reachability maps cached). `.gitlab-ci.yml` runs the same checks on GitLab. The slow and browser tests run locally.
- **Reproducible results** (v8): every plan and tool result records where it came from: the code version (git commit, `+dirty` with uncommitted changes), the methods, a hash of all planner settings and weights, and the seed (all searches are seeded, so the same settings on the same code give the same plan). The Results summary card shows it on its last line; the project file keeps it.
- **Speed history** (v8): `$PY scripts/profile_plan.py <example> --record` times a plan and appends a row to `docs/perf_history.csv` (date, code version, example, seconds, results, best cost); without `--record` it prints a profile.
- **Browser tests** (v8, `tests/e2e/`, marker `e2e`): the app runs in-process and headless Edge clicks into it through the DevTools protocol (`tests/e2e/cdp.py`): face selection with plain / Shift / Ctrl clicks, Alt+click → Move and Cancel, dragging the part's gizmo, picking a stay-out zone in 3D, the dark-mode log colours and the guide's search. They are skipped when Edge is not installed.

- `CLAUDE.md` holds the conventions (frames, units, cost), the milestone status with each done-when check, and design notes per module.
- `core/` and `opt/` must not import the GUI.
- **Edit `docs/USER_GUIDE.md`, then run `$PY scripts/build_guide.py`** to regenerate the HTML. `tests/test_guide.py` fails if the HTML is out of date.
- Each milestone's acceptance test lives in `tests/` (for example `test_kinematics.py`, `test_feasibility.py`, `test_reachability.py`, `test_placement.py`, `test_tool_geometry.py`, `test_carriers.py`, `test_export.py`). The carrier builder is covered by `test_carrier_builder.py` and `test_examples.py`, the brush / single points / tessellation by `test_point_tools.py`.

## 10. For software engineers

A short map of the code for someone who has to review or change it. History and the decision log
are in `CLAUDE.md`; a critical review of the current state is in `docs/V11_REVIEW.md`.

### 10.1 Languages and how it runs

- **Python 3.12 only.** No compiled extension of our own (the offset-wrist IK is compiled at first
  use by numba), no JavaScript build, no web framework. HTML / JavaScript appear only as small
  inline strings: the guide's search box, the report, and three hidden-input "bridges" that let an
  HTML widget send clicks back to Python (`ui/joystick.py`, `ui/package_tabs.py`, `ui/jog_arrows.py`).
- **The GUI is Viser**: a Python server that streams a Three.js scene and a widget panel to the
  browser over a websocket. Every widget callback runs in Python, on viser's thread pool.
- `app.py` builds the whole app (`build_app(server)`), `scripts/run_app.bat` starts it, the
  portable zip (`scripts/make_portable.py`) is the repo plus a conda-packed environment.

### 10.2 Packages and what each one is for

| Package | Used for |
|---|---|
| viser, yourdfpy | the browser GUI and 3D scene; loading URDF robots and tools for display |
| pinocchio | robot kinematics: forward kinematics, Jacobians, joint limits, the geometry model for collisions |
| coal (ex hpp-fcl) | collision and distance queries between meshes (through pinocchio, and directly for carriers) |
| numpy, scipy | everything numeric; scipy for KD-trees, rotations, sparse graphs, interpolation |
| numba | the compiled offset-wrist IK of the FANUC CRX / ABB GoFa / PoWa |
| trimesh, rtree | meshes: loading, ray casting, closest points, submeshes |
| fast_simplification, manifold3d | low-detail collision copies of large meshes (reduce / wrap) |
| shapely | 2D floor footprints, standoff zones, keep-outs |
| ortools (CP-SAT) | the station set cover, the GTSP point order |
| cma, optuna | CMA-ES refinement; TPE refinement and NSGA-II Pareto fronts |
| pydantic | the project file schema (`core/schema*.py`), validated on load |
| cadquery | STEP import (tessellation) and STEP export of designed tools |
| pillow, markdown-it-py | report images; this guide's HTML |
| pytest, ruff | tests; lint and formatting |

### 10.3 Layers

```
app.py      wires the panels together (one function, build_app)
ui/         one module per panel or tab: widgets, scene drawing, jobs started from the GUI
opt/        optimizers: reach maps, station planner, zoning, areas, tool geometry, gantry placement
core/       math and data: schema, robot / IK, tool, collision, feasibility, paths, timing, exports
```

`core/` and `opt/` never import viser or `ui/`, so everything below the GUI runs headless
(`examples/headless_plan.py`) and in worker processes. The state of an open study is one
`ui.session.Session` object (SI units); `Session.to_project()` / `apply_project()` convert it to and
from the project file (`core.schema.Project`).

### 10.4 How the main parts work

- **Kinematics and IK** (`core/robot.py`, `core/ik/`): robots come from `assets/robots/<name>/`
  (URDF + `spec.yaml`). IK is analytic where the arm allows it (UR, OPW, offset wrist) and numeric
  otherwise; every solver has the same interface (`solve`, `solve_batch`, `config_label`).
- **Tool** (`core/tool.py`): mount + a parametric extension (straight, bent, pose) + the head URDF +
  TCPs. TCPs carry their tolerances, contact parts / faces, minimum surface radius and travel rule.
- **Collision** (`core/collision.py`, `core/collision_mesh.py`): one pinocchio geometry model with
  the robot, tool pieces, part, floor, carrier bodies and stay-out zones; margins per pair; large
  meshes replaced by low-detail copies for checking only.
- **Feasibility** (`core/feasibility.py`): for one target, try tool poses in preference order
  (spin, tilt, standoff), solve IK, keep in-limit branches, check collision, return the best.
- **Reach maps** (`opt/reachability.py`): IK-only reachability on an (r, z) grid x tool directions,
  cached on disk; they make base placement fast. Exact checks always confirm the result.
- **Station planner** (`opt/placement.py` and friends): candidate stops from the map -> CP-SAT set
  cover per station count -> route order -> refinement (CMA-ES / TPE) -> exact verification ->
  zoning by joint configuration -> station areas. Every step is verified by the exact checker.
- **Paths** (`core/paths.py`, `core/path_*.py`, `core/coordinated.py`): order the planned points by
  a pattern, sample between them on the surface, solve IK per sample in the zone's configuration,
  plan joint moves around obstacles (RRT-connect + shortcut, then CHOMP-style smoothing), choose the
  tool spin per sample by dynamic programming, flag what fails.
- **Timing** (`core/cycle_time.py`, `core/trajectory.py`): per run of samples, a time-optimal
  forward / backward pass with joint speed / acceleration limits, then a jerk filter.
- **Exports** (`core/export.py`, `core/report.py`, `core/robot_program.py`, `core/userframe.py`):
  CSV, tool URDF / STEP, HTML / PDF report, URScript / TP / RAPID programs.
- **Package builder** (`core/urdf_package*.py`, `ui/package_*.py`): tools, parts and carriers built
  from primitives and meshes, written as URDF packages (package.yaml keeps what URDF cannot).
- **Jobs** (`ui/jobs.py`, `opt/parallel.py`): long runs go to a worker process that gets a snapshot
  of the project; exact checks are spread over a pool of check processes. The GUI only shows
  progress and results.
- **Staleness** (`ui/staleness.py`, `ui/fingerprint.py`): every plan stores hashes of what it was
  planned from; Results flags plans whose inputs changed, and the setup steps use the same hashes.
- **Plan ids** (`ui/plan_ids.py`): a plan's joint solutions, stamp, paths and checks are keyed by
  `Solution.key` (its `id`, given when it enters the results; the label for plans saved before
  v12). The label is only for people.
- **Batch runs** (`core/part_package.py`, `ui/batch_*.py`, `ui/package_part.py`): part packages
  (part + world pose + targets) in a folder each. A batch runs every part in a fresh spawned process
  (`ui.batch_run`, a process pool with one task per process); there `ui.batch_part` builds the app
  without a browser, opens the part's project and drives the same panels a user would (plan, build
  paths, save, report), so batches and the GUI cannot drift apart.

### 10.4a Threads and the session lock

Several threads touch the one `Session`. `ui/session_lock.LOCK` (one re-entrant lock per process)
is held for short changes, never around a long computation or while waiting for another thread:

| Thread | What it does to the session | Holds the lock |
|---|---|---|
| viser callback pool | widget and scene callbacks (clicks, edits) | drags (`ui.widgets.Latest`), placing a frame |
| watcher (`ui/workflow.py`, 1 s) | step status, banners, stale plans dropped, hooks | each hook |
| autosave (`ui/project_panel.py`, 5 s) | snapshot to `project.json` | the snapshot |
| planner / check threads (`ui/stations.py`, `tool_opt`, `compare`, `edge_ui`, `tolerance_ui`, `optimize`, `simulate`) | work on a snapshot; then write results, stamps, paths | the write of the results |
| project load / save (`ui/session.py`) | swap the whole session / bundle files and snapshot | the swap / bundle + snapshot |

Playback and the joystick move the robot through the scene only; they do not take the lock.

### 10.5 Run, test, lint, ship

See section 9 for the commands. In short: `scripts/run_app.bat`, `scripts/test_batches.py --fast`
(~30 min, tests marked `slow` skipped), `ruff check . && ruff format --check .`,
`scripts/build_guide.py` after editing this guide, `scripts/make_portable.py` for the offline zip.
Tests never write to the user's workspace (`tests/conftest.py`). CI runs lint, the guide check and
the fast suite on Windows for every push.

### 10.6 Where to start reading

1. `CLAUDE.md`, sections "Architecture rules" and "Frames and units".
2. `core/schema.py` (what a project is), then `ui/session.py` (how it lives in memory).
3. `core/feasibility.py` (the exact check everything rests on).
4. `opt/placement.py` (the planner pipeline) and `core/paths.py` (`trace`).
5. `app.py` (how the panels are put together).
6. `docs/HISTORY.md`: how each part was built and why, by release (the section of the area you
   change).

UR robot models © 2023 Universal Robots A/S. Use hereof is subject to Universal Robots A/S' Terms and Conditions for Use of Graphical Documentation (see `assets/robots/ur20/meshes/LICENSE.txt`).
