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
8. **Project** (top of the panel) → **Save** stores everything in `workspace/default/project.json`. Autosave also does this every 2 minutes, and the next start reopens the project where you left off.

Tick **Dark mode** at the top of the panel at any time.

**Ready-made examples:** **Project** → *Examples* → `rail_long_part` (UR20 on a 7th-axis rail), `mobile_amr` (UR20 on a mobile robot built from simple shapes), or, around a 3 m tall wall, `gantry_overhead` (UR20 upside down on an overhead X-Y gantry), `floor_xy_rails` (UR20 on two crossed floor axes) and `mobile_lift` (mobile robot with a 1 m vertical lift) → **Open example** → **Optimize** → **Plan stations**. Examples are read-only templates: opening one makes a working copy in `workspace/examples/<name>/` (`<name>_2`, … when an earlier copy has your changes, which is never replaced), and saving or autosave only ever writes that copy.

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
- The extension is a capsule of a given radius for collision, and has mass per metre.
- Each value has a **range** (the optimizer's bounds) and a **rough placement** (the value shown and used until optimized).

### 3.4 What "feasible" means

For each target, poses inside its tolerances are tried, starting with the least tilt. A pose is feasible when **all** of these pass:

1. **IK** exists.
2. **Joint limits**, with a margin (default 5°).
3. **Static torque**: gravity + tool + process force, at most 80 % of each joint limit.
4. **Payload curve**: tool mass against the distance of its centre of gravity from the flange, from the UR20 datasheet.
5. **Collision** with 10 mm clearance: robot, tool, part, floor and carrier. The tool's **contact areas** may touch the part.

Status: **reachable**; **marginal** (feasible but tight: clearance < margin + 5 mm, torque > 90 % of its limit, or a joint within 15° of its limit); or **unreachable**, with the first check that failed.

### 3.5 The optimizer in three layers

1. **Tool geometry (outer):** the shortest L by bisection, then θ / d / roll by CMA-ES.
2. **Placement (middle):** a cached **reachability map** scores base positions, an exact **set cover** picks the fewest stations, stations are ordered for short travel, then refined by CMA-ES against the exact checks.
3. **Per point (inner):** the feasibility check above.

The cost J weighs tool length, unreached points, extra stations, travel, manipulability, clearance, joint-limit proximity, torque and the carrier's preferred-standoff penalty. The Results tab shows the breakdown.

---

## 4. Tools and features, tab by tab

### Guided steps

The tabs are numbered in setup order: **1 Robot & cell → 2 External axis → 3 Tool → 4 Targets → 5 Optimize → 6 Results**. Each setup tab starts with its status (*not confirmed yet*, *confirmed*, or *changed since you confirmed it*) and a **Confirm step** button (**Confirm setup** on the External axis tab, which also locks the carrier). You can open any tab at any time; the Optimize tab lists the steps that are not confirmed.

**Out-of-date results.** Every check, plan and tool result remembers the setup it was computed for. When you then change the robot or mount, the external axis, the tool, the part (mesh or position) or the targets, the Results tab shows **Out of date: … changed** for that result, the Optimize tab flags an out-of-date feasibility check, and exports ask before writing (**Export anyway** / **Cancel**). Re-run to clear it. (Moving the robot or carrier between stops does not make a plan out of date; the feasibility check, which runs from the current base, does go out of date.) Results saved before v2 carry no record and are not flagged.

### Top of the panel

- **Dark mode**: switches the GUI and the floor grid. The `--dark` flag starts in dark mode.
- **User guide**: **📖 Open user guide** opens this guide in a new browser tab. It is served by the app on its port + 1 (`--guide-port N` to change it) and also works from another PC on the network. **Download user guide (HTML)** saves the same page as a file.
- **Objects** (its own panel on the **left** of the 3D view): *Show / hide* is a tree of the cell: **Robot** (and its *Tool*), **External axis** (and each carrier body), **Part** (*Targets*, *Selection*), **Stay-out zones** (each zone) and **Results** (*Stops and lines*, *Ghost robots*, *Station areas*, *Paths*); a child is shown only when its parent is ticked too. *Gizmos*: *Drag gizmo on* chooses which object has drag handles (robot base, part, tool mount, carrier body, stay-out zone); *Gizmo size* scales all of them and *Gizmos always on top* draws them through bodies, so one inside a body can still be grabbed. **Jog**: *Move* the robot / carrier or the part with the arrow buttons (−X/+X, −Y/+Y, −Z/+Z, −Rz/+Rz) by *Step (mm)* / *Turn step (deg)*. With a carrier the arrows drive its axes: a rail stops at the end of its travel, a gantry within its x / y range, a mobile base moves and turns without limits (Z = the lift axis, if any).
- **Project → New part, same cell**: keeps the robot, tool, external axis and stay-out zones, clears the part, its targets, plans, checks and paths. **Project → New empty study**: nothing at all, not even a robot (choose one in tab 1). Both first save the current study and start in a new folder `workspace/study_<date>_<time>`, so the previous study stays as it was.
- **Project**: *Folder* (relative to the repo, or absolute), **Save** / **Load** (`<folder>/project.json`), **Download project.json**, **Open project.json** (upload). Loading never half-applies: a broken project leaves the current session untouched.
  - **Autosave** (on by default) saves to the current project folder every *Autosave every (min)* minutes (default 2) when something changed. The line under it shows the last save time.
  - **Reopen where you left off**: the app remembers the last project you saved, loaded or autosaved, and opens it at the next start. Start with `--fresh` (e.g. `scripts\run_app.bat --fresh`) for the default cell instead.
  - A project brings back the whole cell **and** where you were: the arm's joint pose, the surface selection, feasibility colours, every plan's joint solutions (ghost robots, CSV export without re-planning), the selected solution, and the panel settings (spacing, brush, check and planner settings, display options, dark mode). *Examples* + **Open example** opens a working copy of a shipped example (see Quickstart). To add or update an example, type a *New example name* and click **Save as example** (tick *Overwrite existing example* to replace one); this is the only way the app writes into `examples/`. Saving a project to another folder takes its uploaded meshes (`assets/`) along.

### 1 Robot & cell tab

| Section | What it does |
| --- | --- |
| **Robot** | Choose the UR20 from the registry, or **Upload URDF (.zip)** for another robot (numeric IK; planning features need a UR-type arm). The mount presets are floor, wall and ceiling. Move the base with the gizmo or the xyz/rpy fields (static base only: with a carrier the base rides on it and these fields are read-only, so the robot can never be dragged off the carrier). *Joints (deg)* sliders jog the arm; the readout shows the flange pose computed by Pinocchio. **Go to home** jogs the arm to its home pose. *Home position* (folder): jog the arm with the sliders and press **Set home to current pose**, or type one value per joint in *Home joints (deg)* (comma separated) and press **Apply typed values**; values are clipped to the joint limits. The home is saved with the project; **Use robot default** goes back to `home_deg` in the robot's `spec.yaml` (UR20: 0, −90, 0, −90, 0, 0). The paths use this pose between stops. |
| **Part** | A test part (flat plate, curved panel, open box, long rail, deep hole) or **Upload mesh (.stl/.obj/.ply)**: set *Upload units* first (**mm** by default); *Units* is the current part's unit. Meshes are drawn double-sided (never see-through from one side). Place it with the gizmo or the fields. |

| **Stay-out zones** | Volumes the **arm, the tool and the carrier** keep clear of, in the station planner, the feasibility check, the tool optimizer and the paths: **Add box**, **Add cylinder** or **Add STL/OBJ/PLY zone** (*Mesh units*, mm by default). Pick a zone to edit its *Name*, size, *Position* and *Rotation*, or tick *Drag with gizmo*; **Remove zone**. The arm and tool keep the part clearance (10 mm by default) from them, the carrier the keep-out clearance. Drawn as see-through red volumes, saved in the project; changing them makes plans out of date. (The red floor areas of the External axis tab are different: floor keep-outs for the carrier only.) |

### 2 External axis tab

Build the carrier (external axis) the robot rides on, mount the robot on it, and set where it may go. Only the settings of the chosen *Mode* and *Carrier* are shown (for example *Yaw range* and *Localization* only for a mobile base; for a mobile base *Travel* x / y only bounds the planner's search, jogging has no limits).

**Confirm setup** locks the assembly: the editing sections are hidden, the robot rides on the carrier, and the two move together (carrier position sliders or the jog arrows under **Objects**). **Edit setup** unlocks it to change bodies, mount or settings.

| Section | What it does |
| --- | --- |
| **Axis type** | *Mode* **static** (one fixed base; the planner places one station) or **movable** (a carrier; the planner chooses its stops). *Carrier*: **rail** (travel *s* along the track frame's x axis), **gantry** (x/y carriage, robot usually upside down) or **mobile** (x, y, yaw on the floor). **Load carrier defaults** fills in a ready-made carrier of that kind. *Track/gantry frame*, *Travel*, *Yaw range* and *Lift range* (optional vertical axis) set the motion. With a lift, the planner chooses the **lift height of every stop** (like the travel position; seeded at 2 to 6 heights over the range) and the carrier move between two stops changes the height on the way. |
| **Bodies** | The carrier's shapes. **Add box**, **Add cylinder**, or **Add STL/OBJ/PLY body** (choose *Mesh units* first; the file is stored in `<project>/assets/carriers/`). Pick a body in *Body* to edit its *Name*, size, *Position*, *Rotation* and *Mass*. *Attached to*: **carriage (moves)** for everything that travels with the axis (chassis, wheels, a lift mast), **lift stage (rises with the robot)** for the part of a lift that goes up and down with the robot (a platform or the moving column), **fixed (track / frame)** for parts that stay put (rail track, gantry beam). With a lift, carriage bodies stay on the floor while the robot rises, also in the collision check. Tick **Drag body with gizmo** to place it in 3D. **Duplicate body** / **Remove body**. The selected body is drawn in blue. |
| **Robot mount** | *Mount on body* + **Snap robot to top of body** puts the robot base on the top centre of that body (keeps the yaw). Fine-tune with *Mount xyz / rpy* (robot base in the carriage frame) or **Drag mount with gizmo**. |
| **Standoff and safety** | *Part standoff min / preferred*: carrier-to-part distance; the minimum is hard, closer than preferred adds a cost. It is measured in 3D, so a carrier may sit partly under a raised part: model fixtures and stands as part of the mesh or as keep-outs. *Keep-outs*: floor rectangles `x0,y0,x1,y1; ...` in mm. *Localization* (mobile): position/heading error every stop must tolerate. **Mobile base: robot sides, fixed standoff, part sides.** Tick the **robot sides that may face the part**: *Front*, *Side 1 (left)*, *Side 2 (right)*, *Rear* (several allowed: the planner picks per stop; *Side 1 / Side 2* = the base runs parallel to the part). With any side ticked, every stop keeps one *Fixed standoff (mm)*, and you can restrict the **part sides it may work from** (*Part ±X / ±Y side*, in the part's own frame; none ticked = all). No side ticked = the free x / y / heading search with *Part standoff min / preferred*. **Rail / gantry: *Align track with the part*** turns the track along the part's long axis and centres it on the part (keeping its side and distance); it follows the part when you move it. (Older wording: — pick which side of the base turns towards the part (front = carriage +x, left = +y) and one gap between the base outline and the part outline (top view). Every stop then sits exactly at that gap with that side facing the part: the stops lie on one contour around the part (like a rail wrapped around it), which keeps one safety distance everywhere and makes planning much faster. Choose *free (search the heading)* for the older free x / y / heading search with *Part standoff min / preferred*.) *Tip-over check* (mobile, needs a mass on at least one body). |
| **Apply base settings** | Commits the axis-type and safety fields. Body and mount edits apply immediately. |
| **Carrier position** | Sliders move the carrier, and the robot with it. The readout gives the footprint-to-part distance; it and the red/amber standoff outlines follow the part when you move it. |

Every body collides with the arm and tool (from the upper arm on), and counts for standoff and keep-outs. Bodies touching the floor form the **support polygon** for the tip-over check (e.g. the four wheels), and every body's mass counts. Drawing: red line = minimum standoff, amber = preferred, red boxes = keep-outs.

### 3 Tool tab

| Section | What it does |
| --- | --- |
| **Library** | *Tool* picks a saved tool (`tools/*.yaml`). **Upload tool URDF (.zip)** imports a URDF plus meshes; `package://` paths are resolved. **Save to library** stores the tool and copies its geometry into `assets/tools/<name>/`. |
| **Process** | *Type* applies a preset: scanning, sanding, grinding, gripping, spraying or custom. You can then edit the force, tilt cone, standoff and spin. Contact-free processes (scanning, spraying) clear all contact areas. |
| **Flange mount** | Places the tool on the flange with a gizmo or typed values. |
| **Extension** | *Type* **pose** (default for new tools): where the head sits relative to the flange mount, as **x, y, z** (mm) and **rx, ry, rz** (deg, roll-pitch-yaw), each with a *range* the optimizer may use (equal ends fix a value). The connector between mount and head is drawn and collision-checked as a **smooth curve** (it leaves the mount along its axis and arrives along the head's axis), not a straight cylinder; its *capsule radius* and *mass per metre* apply along the curve. Legacy *straight* / *bent* tools keep the old L, θ, d, roll ranges (θ and d for bent only); the sample tools are legacy. **Switching a legacy tool to pose keeps its shape**: its head pose becomes x y z / rx ry rz with every range fixed at that value; widen the ones the optimizer may change. rx / ry / rz are the W / P / R angles. **Rough placement** sliders set the current values of the free ones. |
| **TCPs** | Add, rename or remove TCPs and place each with a gizmo. **Pick contact faces (click tool)**: tick it, then click the tool; one click selects the whole smooth face (*Region angle* sets how far it spreads), and clicking again removes it. **Whole-link contact** marks entire links. |
| **Mass** | Shows the total mass and COG relative to the flange. It can override the head mass and COG when the URDF has no inertials. |

Sample tools: **orbital_sander** (bent extension range, pad as the contact link) and **scanner** (contact-free). Both are simple stand-in geometry. Switch a tool to *Type* **pose** to let the optimizer move and tilt its head.

### 4 Targets tab

| Section | What it does |
| --- | --- |
| **Select surface & points** | Pick a **Tool**; only its settings are shown. **Fill +** / **Fill −**: click a face to add / remove its smooth region (neighbouring faces may turn by at most the *Fill angle limit*). **Brush +** / **Brush −**: an orange-ringed **handle** appears on the part; drag it and every face within the *Brush radius* of its path is added / removed (it follows the surface even if you drag above it, and skips faces turned more than *Brush max surface angle* from the surface under it, so it stays off the back of thin parts). A click paints one dab and moves the handle there. **Box +** / **Box −**: drag a rectangle in the 3D view (the camera does not orbit while a Box tool is chosen). **By normal**: every face within an angle of a direction. **Full surface**: click a body of the part to add all of its faces. **Single point**: one target exactly where you click (normal from the surface). **Remove point**: deletes the target nearest the click, within the brush radius. **Point brush**: each click adds a dab of the *Part-anchored lattice* inside the brush radius. **Ctrl-click** turns + into − (and back) for the click tools. **Select all**, **Clear** and **Invert** act on the whole part; the selected face count and area are shown below. |
| **Generate points** | Pick a **Generator** (only its settings are shown); it runs on the selection. **Surface grid (rows/columns)** (as in the sand-optimizer): the grid lies across the selection's mean facing direction, rows run along the part X axis turned by *Row direction (deg)*, points along a row are *Column spacing* apart and rows *Row spacing* apart, centred on the selection; points closer than *Edge inset* to the selection's edge are left out. A strongly curved selection gets a warning (select one side at a time). **Part-anchored lattice**: rows / columns anchored at the part origin (the Point brush grid), rows along *Row direction (part frame)*. **Projected grid**, **Poisson (even)**, **Raster lines** and **Selection edges** as before (spacing in mm). Normals are *interpolated* (smooth) or *face*. Points that already exist are skipped, and **Undo last add** removes the last generate, click or dab. |
| **Refine part mesh** | **Tessellate part mesh** splits the part's triangles until no edge is longer than *Max edge length*. The shape does not change, but clicks, region growing, brushes and interpolated normals work on a finer mesh (useful for coarse CAD exports with long thin triangles). The refined mesh is saved as `<project>/assets/parts/<name>_refined_<N>mm.stl` and becomes the part; the selection and existing points are kept. |
| **Groups & tolerances** | Each point belongs to a group, and each group has spin, tilt, standoff and an allowed TCP list. **Apply to group** saves them. |
| **Edit points** | Assign to group, flip normals, or delete the points on the selected surface; delete all. **Import CSV** reads `x,y,z,nx,ny,nz[,group]` in mm or m. |
| **Display** | Colour points by *group*, *feasibility*, or the selected solution's *zone* / *configuration*; show normals; **Preview tool at point** draws a see-through tool aimed into the surface at any point. |

Points are stored **in the part frame**, so moving the part moves them with it.

### 5 Optimize tab

| Section | What it does |
| --- | --- |
| **Feasibility check** | **Check all targets** with clearance, joint margin and torque limit as settings. *Check from* = **stops of the selected plan** (default): each point is checked from the stop the plan in Results assigns it to, with the carrier there (points the plan did not reach: every stop is tried); or **current robot position**: everything from where the robot stands now (without a plan this is used automatically). The carrier's own bodies count as obstacles. The summary gives counts and reasons. *Inspect result* shows either the feasibility check or, by default, the **plan selected in the Results tab** (*Show*); it switches to each new plan or tool result automatically. **Move robot to this solution** moves the carrier (or base) to the point's station and the arm to its solution; **◀ previous point / next point ▶** steps through the points and moves the robot to each one at once. |
| **Base placement (reachability)** | **Show base heatmap** draws where the base could stand on the floor (bright = all points reachable with the best worst-case manipulability) and marks the best spot. **Move base to best** moves it there. Ignores collisions: confirm with the feasibility check. |
| **Station planner** | **Plan stations** finds the fewest stations that reach every point, within *Max stations* (up to 40; static mode = 1). It plans one station count after the other and stops once two more stops in a row gave no better plan, so a large *Max stations* does not mean that many full plans. With many points the stop search works on a spread-out sample (a quarter of the points, 60 to 150; all of them below 60), adds a stop where the sample missed points that need one, and then checks **every** point exactly, so the result is still exact (357 points on the rail: 43 s → 18 s, same plan). Each solution goes to the Results tab as soon as it finishes. *Candidate grid* sets the search resolution; *Refine* turns the exact fine-tuning on or off. **Connected station areas** (on by default) reassigns points after planning so each stop's points form **one connected area** on the part and no stop's points sit among another's: every stop grows outward from the point nearest its base, taking only points it reaches exactly, then stray points join the stop around them. It is kept only when it leaves fewer pieces than the planner's own split. Plans appear quickly: every station count is first verified at its first-guess stops (a few seconds each), then the most promising one or two are refined and replace their quick version when better. **Cancel** stops within about a second and keeps the plans already finished. *Joint configuration* = **zones (minimize changes)** groups each stop's points into zones that keep one arm configuration and puts each zone in path order (see *Joint-configuration zones* below); **Costs & requirements** (folder): every cost weight can be edited: **Cost per stop** (each base / carrier position beyond the first, default 5), **Cost per group** (each extra joint-configuration group at a stop), **Cost per wrist flip** (a joint 4 / joint 6 jump over 180° inside a group's path), per unreached point, carrier travel per m, tool length, manipulability, clearance, joint limits, torque and preferred standoff. They are saved with the project; **Default costs** restores them. **Must be met**: **All points reachable** and **No configuration change within a group** (turns zoning on; a stop may hold several groups, each in one configuration, and a wrist flip inside a group breaks the rule). With a rule ticked the planner keeps adding stops until a plan meets it (up to *Max stations*); plans that break a ticked rule stay in the Results list marked **✗**, below the valid plans, and the summary names the rule. |
| **Tool geometry** | **Optimize tool**. **Pose tools:** tries head poses within the ranges (CMA-ES, *Shape / pose evaluations* candidates), plans stations for each (quick plan) and keeps the pose with the **best overall cost** (reach, stops, travel, manipulability, clearance, joint limits, and connector length as one term); the winner is planned once more with refinement. The table lists every candidate (connector length, full coverage, reached, stops, cost). **Legacy straight / bent tools:** the shortest L that still reaches everything within the station limit, then θ / d / roll (coverage-vs-length table). **Payload is not considered** here (no torque / payload limits, no torque cost); the feasibility check and the station planner still apply them. *Length tolerance* (mm) is up to you (no upper limit). **Cancel** stops within about a second and keeps the best result so far. **Apply to tool** writes the values into the Tool tab. **Save optimized tool to library** stores the tool with the optimized values under *Save as tool* (default `<tool>_opt`) without changing the tool in use. |

### Joint-configuration zones

A UR arm can reach most poses in up to 8 ways: **shoulder** (S), **elbow** (E) and **wrist** (W) each have two branches, written as a label like `S-E+W-`. Moving from one branch to another needs a big reconfiguration motion, which you rarely want in the middle of a sanding pass. With *Joint configuration* = **zones (minimize changes)**, the planner:

1. checks, for every point, which configurations can do it from its stop;
2. picks the **fewest configurations** that cover the stop's points (exact), each point using its best solution in one of them; a stop can therefore have more than one **zone**;
3. orders each zone's points as a serpentine **path** and picks joint values (±360° copies of the wrist joints) that keep the path continuous;
4. counts **configuration changes** = extra zones at a stop + remaining jumps of joint 4 or 6 above 180° between neighbouring points, and adds *Cost per config change* × changes to the cost J. Because the planner also offers plans with more stops, a plan with one more stop and no changes can win.

It also re-runs the tip-over and localization checks for the chosen solutions. It is about 3× slower than *best per point*. The Results tab lists the zones (`S1·Z1 S-E+W- 6 pts`), draws each zone's path as a thick coloured line, and **Targets → Display → Color points by** `zone` or `configuration` colours the points of the selected solution. The CSV and the HTML report list zone, path position and configuration per point. The Tool geometry optimizer uses the same setting.

### 6 Results tab

- **Solution**: the ranked list (best cost first); **◀ previous / next ▶** switches between the plans (with *Move robot with the plan* ticked the robot goes to the new plan's first stop), and a table compares all plans (stops, points reached, cost; ⚠ = out of date). The feasibility check, *Inspect result*, *Move robot to station* and the paths all use the selected plan; paths built for another plan are cleared. It shows a summary (with the zones and configuration changes when zoning is on), the cost breakdown bars and the station list.
- **Scene**: coloured station markers, **lines to served points**, **ghost robots** posed at a real solution, and the **station areas on the part**: the part surface painted in each stop's colour (the colour of its lines), a stop's second and third groups in lighter shades. A coarse part mesh is subdivided for this display only. If a stop's area is still in pieces after planning, the summary says so (⚠).
- **Move robot to station**: moves the carrier to the station's axis values (the robot base rides on it; static: the base itself) and the arm to the first point that station serves.
- **Paths & simulation** (folder, open, above Export; the planner and the tool optimizer point to it when they finish): **Build paths** connects the selected solution's points **stop by stop**: every point of a stop is done before the carrier moves on. Within a stop each group (zone, one joint configuration) is ordered by the **Pattern** (as in the sand-optimizer): **Zigzag** (rows back and forth), **Unidirectional** (every row the same way), **Cross-hatch** (zigzag, then again at +90°), **Concentric** (rings from the border, *Outside → in* or inside out), **Spiral**, **Hilbert** (space-filling curve), **Shortest path (TSP)**, or **Planner order** (the planner's own order). Rows run at *Raster angle* from the part X axis and start at the *Start corner* (`min-min` = lowest row, low end); *Row tolerance* (0 = half the point spacing) decides which points share a row or ring. Only the chosen pattern's settings are shown. Groups follow each other nearest first, and each later group starts at the corner nearest where the previous one ended. **Prevent crossing paths**: a move that would cross an earlier move of the same pass lifts off the surface by *Lift height* instead. **Home between stops** (on by default): each stop starts with a joint move from the robot's home pose and ends with one back to home; between stops the carrier moves with the arm at home (a rail along its travel, a mobile base along its contour path when both stops use the same robot side, otherwise in a straight line), and every carrier pose on the way is checked against keep-outs, the part standoff and stay-out zones. When that move is blocked, a **mobile base drives around** the stay-out zones, keep-outs and the part (a route planned on the floor, turning where needed) and keeps its **standoff distance** from the stay-out zones on the way (the *Fixed standoff* on a contour, else the minimum part standoff, at least the keep-out clearance; close to a stop the keep-out clearance the stop was planned with is enough). Rails and gantries cannot drive around: a blocked move is flagged `carrier_blocked`. With a **lift**, the lift moves only while the carrier stands: at the stop it lowers to the travel height (the lowest height, or the first of the two stops' heights and the lift levels at which the drive, the lift moves and the arm at home are all clear), the carrier drives at that height, and at the next stop the lift rises to that stop's height before the arm leaves home. The home pose is the one set under Robot & cell → *Home position*. **Moves to and from home are collision-checked and planned** against the part, the floor, stay-out zones and the carrier's own bodies: each stop starts with the joint values nearest home (the same arm pose, without a joint turning the long way round). A straight joint move, or one joint group at a time (arm, base, wrist), is used when it is clear. Otherwise the tool first backs straight off the surface (150 mm along its axis, same configuration) and the arm finds a way around. Only when no way is found is the straight move shown, with its collisions flagged. Between two points on the surface the TCP follows the part (sampled every *Path sample step*, projected onto the part, standoff kept) and turns smoothly from one planned tool orientation to the next; every sample is solved for the arm in the same joint configuration, closest to the previous sample. The summary lists flagged samples (red dots in the scene):
  - **unreachable** (no solution within the joint limits), **config flip** (only another shoulder / elbow / wrist configuration reaches it), **jump** (a joint moves more than *Max joint step* between neighbouring samples), **collision**;
  - **reconfigure**: the plan itself switches configuration there. The arm does not flip next to the part: it goes back to home and comes in again in the new configuration, with the same collision-free planned moves (directly, without home, when *Home between stops* is off). Plan with *Joint configuration* = zones to avoid configuration changes.
  - **carrier blocked**: a carrier pose between two stops hits a keep-out, the part standoff or a stay-out zone.

  The paths are drawn as coloured lines (one colour per zone); lifts, home moves and carrier moves are thin grey lines. **▶ Play** / **⏸ Pause** moves the arm along the samples and the carrier between the stops at *Speed (mm/s)*; *Sample* scrubs. **Clear paths** removes the built paths (lines, issue dots and playback) until you build them again. **Export path CSV** writes `export/paths.csv` (stop, kind = surface / lift / home / carrier, the carrier values, zone, from / to point (0 = home), TCP pose in the world in mm + quaternion, joints in degrees, issue per sample). It is a preview of the motion, not a timed robot program.
- **Export**:
  - **TCP poses (CSV)**
  - **Userframes (CSV)**: one row per stop: its carrier values and the **user frame**, the pose of the part frame in the robot's base frame at that stop (teach the points in it and they stay valid). The columns follow the robot model: UR arms `x_mm, y_mm, z_mm` + rotation vector `rx_rad, ry_rad, rz_rad` (as in a UR pose `p[...]`, with x y z converted to m); FANUC-style arms `X Y Z` (mm) + `W P R` (deg); every robot also gets the rotation matrix `r11 … r33`. The format comes from `frame_convention` in the robot's `spec.yaml` (`ur`, `fanuc` or `matrix`).
  - **Optimized tool URDF (zip)**
  - **HTML report**: self-contained, printable to PDF, with an optional 3D snapshot.

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

Shortcut: **Project** → *Examples* `mobile_amr` → **Open example** (four wheels, chassis, battery box and a column with the UR20 on top; tip-over check on).

1. External Axis → Carrier `mobile` → **Load carrier defaults** (a chassis and a riser).
2. Select *Body* `chassis`, set *Box L W H* to `1100, 700, 300` and *Position* to `0, 0, 100` (raised for wheels).
3. **Add cylinder** → *Name* `wheel_fl`, radius/height `100, 80`, *Rotation* `90, 0, 0` (axle along y), *Position* `450, 390, 100`, *Mass* `10`. **Duplicate body** and move the copies to `450,-310,100`, `-450,390,100`, `-450,-310,100`.
4. **Add box** → a battery or cabinet, or **Add STL/OBJ/PLY body** for a CAD model of your AMR (set *Mesh units* first).
5. *Mount on body* `riser` (or your column) → **Snap robot to top of body**.
6. Give the chassis a *Mass*, tick *Tip-over check*, **Apply base settings**, then Optimize → **Plan stations**.

### Example B3: avoid joint-configuration changes on the rail

1. **Project** → *Examples* `rail_long_part` → **Open example**. Targets: select the face pointing to the wall (y+) and generate points with *Spacing* `250`.
2. Optimize → *Station planner*: *Joint configuration* `best per point` → **Plan stations**. Targets → *Color points by* `configuration`: stops can mix two configurations, meaning the arm would reconfigure mid-pass (in the test version of this task every stop did).
3. Switch to `zones (minimize changes)` → **Plan stations** again. The Results summary now lists the zones and the number of configuration changes (0 in the test version of this task, at almost the same cost), and the zone paths are drawn in the scene.

### Example B4: tall wall with a gantry, floor X-Y axes or a mobile lift

The 3 m tall wall (`tall_wall`, 2 × 0.3 m) is out of reach from one fixed robot height. Three examples work it with different carriers (orbital sander, 250 mm grid):

* `gantry_overhead`: the UR20 hangs upside down from a carriage on an overhead X-Y gantry (two beams 4.4 m up on four posts). Targets: the top face and both long faces from 1.8 m up; the gantry's y axis takes the robot over to either side.
* `floor_xy_rails`: the UR20 stands on a riser on a carriage that moves along two crossed floor axes (x along the wall, y towards it). Targets: the front face and the end face from 0.25 m to 1.8 m (the sander cannot work right at the floor).
* `mobile_lift`: the mobile robot of `mobile_amr` with a 1 m lift (a *mast* on the chassis and a *lift_stage* that rises with the robot). Targets: the front face from 0.5 m to the top; every stop gets its own lift height, so the robot works the bottom low and the top raised.

Open one, then Optimize → **Plan stations**; Results lists each stop's carrier values (with the lift height for `mobile_lift`).

### Example C: the shortest probe for a deep hole

1. Cell → *Part* `deep_hole`, xyz `800, 0, 0` (100 mm hole, 300 mm deep).
2. Tool Setup → upload or define a slim straight tool. The test suite uses a probe with capsule radius 20 mm and no head; see `tests/test_tool_geometry.py::_probe`.
3. Targets → *Tool* `Single point` → click the hole bottom; group standoff `50` mm, tilt `0`.
4. Optimize → **Optimize tool**: L ≈ **260 mm** = depth − standoff + clearance (the wrist cannot enter the hole).

### Example D: your own robot cell

1. **Part**: Cell → **Upload mesh**, choose *Units* `mm` if your CAD exports in mm.
2. **Tool**: Tool Setup → **Upload tool URDF (.zip)** (URDF + meshes). Set the Process, place the TCP(s), pick the contact faces, set the extension ranges → **Save to library**.
3. **Targets**: if the STL has long thin triangles, **Tessellate part mesh** first. Then click-select the faces to process and generate, brush the areas you want with *brush points*, add single points, or **Import CSV** from a CAD/CAM tool (`x,y,z,nx,ny,nz[,group]`).
4. **Carrier** (optional): External Axis → build the rail or mobile robot from shapes or an STL and snap the robot onto it.
5. Then the same steps as Example A. Save the project so it can be reloaded later.

---

## 6. Files, folders and formats

| Path | Contents |
| --- | --- |
| `app.py` | GUI entry point (`--port`, `--host`, `--dark`, `--guide-port`, `--fresh`) |
| `scripts/run_app.bat` | Windows launcher using the `rso` env |
| `core/` | kinematics, IK, collision, tool model, targets, feasibility, carriers, exports (no GUI code) |
| `opt/` | reachability maps, station planner, tool-geometry optimizer, cost |
| `ui/` | the Viser GUI tabs |
| `assets/robots/ur20/` | UR20 URDF, meshes, `spec.yaml` (DH, limits, payload curve), licences |
| `assets/tools/`, `tools/*.yaml` | sample tool geometry and the tool library |
| `assets/parts/` | test parts (STL) |
| `examples/` | example projects (`rail_long_part/`, `mobile_amr/`, `gantry_overhead/`, `floor_xy_rails/`, `mobile_lift/`, regenerate with `scripts/make_examples.py`), `sander_mobile.json` (format sample) and `headless_plan.py` |
| `workspace/` (not in git) | your project folders: `project.json`, uploads, `export/` |
| `cache/` (not in git) | reachability maps |
| `docs/USER_GUIDE.md` / `.html` | this guide; the HTML is generated from the Markdown with `scripts/build_guide.py` |
| `docs/PROJECT_PLAN.md` | the design plan |

**Project file** (`project.json`): the robot and base pose, base mode and carrier (its `bodies`: shape, size or mesh file, pose, attachment, mass), tool, part, targets and groups, optimizer settings, and the saved solutions. Older projects with `chassis` / `riser` / `track` still load; the External Axis tab converts them to bodies. The `work` section holds where you left off: joint pose, selection, per-point joint solutions of the feasibility check and of each plan, and panel settings. Saves are atomic (written to a temporary file, then renamed). `workspace/last_project.txt` records the folder to reopen.

**TCP poses CSV** has one row per point, in station order:
- `point_id, group, status, reason, station, zone, seq, config, carrier` (zone / seq = path position / config label when zoning is on; rows then follow station → zone → path order);
- base pose (`base_x/y/z_mm, base_yaw_deg`);
- TCP in **world** (`tcp_world_x/y/z_mm`, quaternion `qw,qx,qy,qz`) and in the **robot base frame** (`tcp_base_*_mm`, `roll/pitch/yaw_deg`): use the base-frame values to program the robot at that station;
- joints `q1..q6_deg`;
- `spin_deg, tilt_deg, standoff_mm, clearance_mm, manipulability, torque_util, limit_margin_deg`.

Unreached points come last.

**Optimized tool URDF** (zip): `flange → mount → extension_k → head links → tcp_<name>`. The root link `flange` attaches to the robot's `tool0`. The extension is a cylinder with sphere caps and its own mass. Meshes are included.

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
| The app opens an old project at start | It reopens the last project. Use **Project → Load** for another one, or start with `--fresh`. |
| The planner puts a mobile base under a raised part | Standoff is 3D; add the part's stand to the mesh or draw a keep-out. |

**Known limitations (v1)**
- Planning features (heatmap, station planner, tool optimizer) need a **UR-type arm** (analytic IK and a full-turn base joint). Other uploaded robots can be displayed, jogged and feasibility-checked with numeric IK.
- Collisions are checked at each point only; the motion **between** points and stations is not checked (zones give an ordered path with continuous joints, but the path itself is not collision-checked). Travel distance ignores obstacles.
- Zoning works per stop with the stops the planner chose; it does not move points between stops to save a configuration change. The plans with other stop counts are the trade-off it offers.
- Payload checks are **static**: no acceleration loads. Beyond 800 mm COG offset the UR20 payload curve is a **linear extrapolation (estimate)**.
- The part footprint for standoffs is a convex hull, which is conservative.
- The **brush** works click by click (one dab per click): Viser reports clicks and box drags, not freehand mouse strokes. For big areas, select them and use **Fill selection with rows/columns**.
- Mesh bodies on a carrier are used as-is for collision; an STL with millions of triangles slows the checks. Simplify it in CAD first.
- **Not yet confirmed in a real browser** (only in automated and headless tests): gizmo dragging (parts, carrier bodies, robot mount), contact-face clicking, shift/ctrl modifiers, box select, brush and single-point clicks on the part, file upload (meshes, carrier STLs) and download. Please report anything odd.

---

## 9. For developers

```bash
PY=C:/Users/<you>/miniforge3/envs/rso/python.exe
RUFF=C:/Users/<you>/miniforge3/envs/rso/Library/bin/ruff.exe
$PY -m pytest -q                         # all tests (~8 min)
$RUFF check . && $RUFF format --check .   # lint + format
```

- `CLAUDE.md` holds the conventions (frames, units, cost), the milestone status with each done-when check, and design notes per module.
- `core/` and `opt/` must not import the GUI.
- **Edit `docs/USER_GUIDE.md`, then run `$PY scripts/build_guide.py`** to regenerate the HTML. `tests/test_guide.py` fails if the HTML is out of date.
- Each milestone's acceptance test lives in `tests/` (for example `test_kinematics.py`, `test_feasibility.py`, `test_reachability.py`, `test_placement.py`, `test_tool_geometry.py`, `test_carriers.py`, `test_export.py`). The carrier builder is covered by `test_carrier_builder.py` and `test_examples.py`, the brush / single points / tessellation by `test_point_tools.py`.

UR robot models © 2023 Universal Robots A/S. Use hereof is subject to Universal Robots A/S' Terms and Conditions for Use of Graphical Documentation (see `assets/robots/ur20/meshes/LICENSE.txt`).
