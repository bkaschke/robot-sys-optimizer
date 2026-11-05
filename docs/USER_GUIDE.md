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

Then try a full job on the test plate:

1. **Targets tab** → *Click on part* = `select surface` → click the top of the plate (it turns green).
2. Set *Spacing / step (mm)* to `100` → **Generate on selection**. Points with normals appear.
3. **Optimize tab** → **Check all targets**. The points turn green, amber or red (reachable, marginal, unreachable).
4. **Optimize tab** → *Base placement* → **Show base heatmap** (the first run builds a reachability map, about 1 min) → **Move base to best**.
5. **Optimize tab** → *Station planner* → **Plan stations**. The result appears in the **Results** tab.
6. **Optimize tab** → *Tool geometry* → **Find shortest tool** → **Apply to tool**.
7. **Results tab** → *Export* → **TCP poses (CSV)**, **Optimized tool URDF (zip)**, **HTML report**. Files are downloaded and also saved to `workspace/default/export/`.
8. **Project** (top of the panel) → **Save** stores everything in `workspace/default/project.json`.

Tick **Dark mode** at the top of the panel at any time.

---

## 2. Set up on a new computer

Tested on **Windows 10/11**. Linux and macOS should work with the same conda environment but have not been tested.

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

### Top of the panel

- **Dark mode**: switches the GUI and the floor grid. The `--dark` flag starts in dark mode.
- **Project**: *Folder* (relative to the repo, or absolute), **Save** / **Load** (`<folder>/project.json`), **Download project.json**, **Open project.json** (upload). Loading never half-applies: a broken project leaves the current session untouched.

### Tool Setup tab

| Section | What it does |
| --- | --- |
| **Library** | *Tool* picks a saved tool (`tools/*.yaml`). **Upload tool URDF (.zip)** imports a URDF plus meshes; `package://` paths are resolved. **Save to library** stores the tool and copies its geometry into `assets/tools/<name>/`. |
| **Process** | *Type* applies a preset: scanning, sanding, grinding, gripping, spraying or custom. You can then edit the force, tilt cone, standoff and spin. Contact-free processes (scanning, spraying) clear all contact areas. |
| **Flange mount** | Places the tool on the flange with a gizmo or typed values. |
| **Extension** | Straight or bent, with ranges for L, θ, d and roll, the capsule radius and mass per metre. **Rough placement** sliders set the current values. |
| **TCPs** | Add, rename or remove TCPs and place each with a gizmo. **Pick contact faces (click tool)**: tick it, then click the tool; one click selects the whole smooth face (*Region angle* sets how far it spreads), and clicking again removes it. **Whole-link contact** marks entire links. |
| **Mass** | Shows the total mass and COG relative to the flange. It can override the head mass and COG when the URDF has no inertials. |

Sample tools: **orbital_sander** (bent extension range, pad as the contact link) and **scanner** (contact-free). Both are simple stand-in geometry.

### Cell tab

| Section | What it does |
| --- | --- |
| **Robot** | Choose the UR20 from the registry, or **Upload URDF (.zip)** for another robot (numeric IK; planning features need a UR-type arm). The mount presets are floor, wall and ceiling. Move the base with the gizmo or the xyz/rpy fields. *Joints (deg)* sliders jog the arm; the readout shows the flange pose computed by Pinocchio. |
| **Part** | A test part (flat plate, curved panel, open box, long rail, deep hole) or **Upload mesh (.stl/.obj/.ply)** with *Units* (m / mm / in). Place it with the gizmo or the fields. |
| **Base mode** | **Static**: one base pose, and the planner places one station. **Movable**: choose a **carrier**, then **Load carrier defaults** and **Apply base settings**. |

**Carriers** are built from boxes and cylinders:
- **Rail**: a track along the frame's x axis; the carriage position *s* is the station variable.
- **Gantry**: an x/y carriage, usually with the robot mounted upside down.
- **Mobile base**: chassis x, y and yaw on the floor; position error (localization ±mm / ±deg) and an optional tip-over check.

The other base settings:
- **Standoff min / preferred**: carrier-to-part distance. The minimum is hard; closer than the preferred adds a cost. It is measured in 3D, so a carrier may sit partly under a raised part. Model fixtures and stands as part of the mesh or as keep-outs.
- **Keep-outs**: floor rectangles `x0,y0,x1,y1; ...` in mm.
- ***Carrier position*** sliders move the carrier, and the robot with it.
- **Drawing**: red line = minimum standoff, amber = preferred, red boxes = keep-outs.

### Targets tab

| Section | What it does |
| --- | --- |
| **Select surface** | *Click on part* = `select surface`: click selects a smooth region; **shift-click** adds, **ctrl-click** removes. **Select by normal** picks every face within an angle of a direction. **Box select (drag in view)** selects the faces inside a rectangle you drag. `place point` mode puts single points on the clicked surface. |
| **Generate points** | Methods run on the selection: **Projected grid** (spacing in mm, rays against the mean normal or a direction you give), **Poisson (even)** (even spacing on curved surfaces), **Raster lines** (step-over plus step along the line, alternating direction), **Selection edges** (along the borders, for deburring). Normals are *interpolated* by default (smooth), or *face*. |
| **Groups & tolerances** | Each point belongs to a group, and each group has spin, tilt, standoff and an allowed TCP list. **Apply to group** saves them. |
| **Edit points** | Assign to group, flip normals, or delete the points on the selected surface; delete all. **Import CSV** reads `x,y,z,nx,ny,nz[,group]` in mm or m. |
| **Display** | Colour points by *group* or *feasibility*; show normals; **Preview tool at point** draws a see-through tool aimed into the surface at any point. |

Points are stored **in the part frame**, so moving the part moves them with it.

### Optimize tab

| Section | What it does |
| --- | --- |
| **Feasibility check** | **Check all targets** from the current base, with clearance, joint margin and torque limit as settings. The summary gives counts and reasons. *Inspect result* → **Move robot to this solution** jogs the arm to any point's solution. |
| **Base placement (reachability)** | **Show base heatmap** draws where the base could stand on the floor (bright = all points reachable with the best worst-case manipulability) and marks the best spot. **Move base to best** moves it there. Ignores collisions: confirm with the feasibility check. |
| **Station planner** | **Plan stations** finds the fewest stations that reach every point, within *Max stations* (static mode = 1). Each solution goes to the Results tab as soon as it finishes. *Candidate grid* sets the search resolution; *Refine* turns the exact fine-tuning on or off. |
| **Tool geometry** | **Find shortest tool** first finds the shortest L that still reaches everything within the station limit, then tunes θ / d / roll. It shows a coverage-vs-length table. **Apply to tool** writes the values into Tool Setup. |

### Results tab

- **Solution**: the ranked list (best cost first). It shows a summary, the cost breakdown bars and the station list.
- **Scene**: coloured station markers, **lines to served points**, and **ghost robots** posed at a real solution.
- **Move base to station**: puts the live robot at a station.
- **Export**:
  - **TCP poses (CSV)**
  - **Optimized tool URDF (zip)**
  - **HTML report**: self-contained, printable to PDF, with an optional 3D snapshot.

  Files go to `<project folder>/export/` and also download in the browser.

---

## 5. Worked examples

### Example A: sand a flat plate from one base (static)

1. Cell → *Part* `flat_plate`, xyz `900, 0, 300` mm.
2. Targets → click the plate top → *Spacing* `100` → **Generate on selection**.
3. Optimize → **Check all targets**. Everything should be reachable from the origin.
4. Optimize → **Plan stations** → Results: "1 floor station: N/N reached".
5. Optimize → **Find shortest tool**. On this plate the shortest straight tool is the bottom of the range (50 mm), and the bend stage may lower the cost further.
6. Results → **HTML report**.

The same job without the GUI is in `examples/headless_plan.py` (section 7).

### Example B: a long rail part with a 7th-axis rail carrier

1. Cell → *Part* `long_rail`, xyz `0, 800, 500`.
2. Targets → *Select by normal* `(0,0,1)` within `5` → *Spacing* `250` → **Generate**.
3. Cell → *Base mode*: Mode `movable`, Carrier `rail` → **Load carrier defaults**. Set *Track/gantry frame xyz* to `0, 150, 0`, so the 0.5 m carriage sits exactly 300 mm from the part. Optionally add a wall keep-out `-5000,1300,5000,3000`. Click **Apply base settings**.
4. Optimize → *Max stations* `5` → **Plan stations**. Expect 2 rail stations for the top face; the Results tab shows the carriage positions *s*.
5. Try Carrier `mobile` with its defaults on the same targets and compare. Both need 2 stops on the top face. On the wall-side face (generate on the face pointing to the wall) the rail needs 4 and the mobile base 5.

### Example C: the shortest probe for a deep hole

1. Cell → *Part* `deep_hole`, xyz `800, 0, 0` (100 mm hole, 300 mm deep).
2. Tool Setup → upload or define a slim straight tool. The test suite uses a probe with capsule radius 20 mm and no head; see `tests/test_tool_geometry.py::_probe`.
3. Targets → *place point* on the hole bottom; group standoff `50` mm, tilt `0`.
4. Optimize → **Find shortest tool**: L ≈ **260 mm** = depth − standoff + clearance (the wrist cannot enter the hole).

### Example D: your own robot cell

1. **Part**: Cell → **Upload mesh**, choose *Units* `mm` if your CAD exports in mm.
2. **Tool**: Tool Setup → **Upload tool URDF (.zip)** (URDF + meshes). Set the Process, place the TCP(s), pick the contact faces, set the extension ranges → **Save to library**.
3. **Targets**: click-select the faces to process, or **Import CSV** from a CAD/CAM tool (`x,y,z,nx,ny,nz[,group]`).
4. Then the same steps as Example A. Save the project so it can be reloaded later.

---

## 6. Files, folders and formats

| Path | Contents |
| --- | --- |
| `app.py` | GUI entry point (`--port`, `--host`, `--dark`) |
| `scripts/run_app.bat` | Windows launcher using the `rso` env |
| `core/` | kinematics, IK, collision, tool model, targets, feasibility, carriers, exports (no GUI code) |
| `opt/` | reachability maps, station planner, tool-geometry optimizer, cost |
| `ui/` | the Viser GUI tabs |
| `assets/robots/ur20/` | UR20 URDF, meshes, `spec.yaml` (DH, limits, payload curve), licences |
| `assets/tools/`, `tools/*.yaml` | sample tool geometry and the tool library |
| `assets/parts/` | test parts (STL) |
| `examples/` | example project JSON and `headless_plan.py` |
| `workspace/` (not in git) | your project folders: `project.json`, uploads, `export/` |
| `cache/` (not in git) | reachability maps |
| `docs/PROJECT_PLAN.md` | the design plan |

**Project file** (`project.json`): the robot and base pose, base mode and carrier, tool, part, targets and groups, optimizer settings, and the saved solutions. Joint solutions per point are not saved; re-run the plan to get them back (for example to export the CSV).

**TCP poses CSV** has one row per point, in station order:
- `point_id, group, status, reason, station, carrier`;
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

For carriers, pass `sites=CarrierSites(Carrier(base), part_mesh, T_world_part)` to `plan_stations`. Build the collision scene with the carrier bodies: `bb, wb = sites.scene_bodies()`; `CollisionScene(..., base_bodies=bb, world_bodies=wb)`.

---

## 8. Troubleshooting and limitations

| Symptom | Fix |
| --- | --- |
| `conda env create` fails or `import pinocchio` errors | Use Miniforge (conda-forge), not pip. Keep the OpenBLAS line in `environment.yml`. |
| Python crashes silently on matrix maths | The MKL BLAS problem: run `conda install -n rso "libblas=*=*openblas"`. |
| The first heatmap or plan takes ~1 min | The reachability map is being built (once per tool shape); later runs are fast. |
| "Software WebGL rendering" banner | The browser has no GPU acceleration; enable hardware acceleration in its settings. |
| Port already in use | `app.py --port 8090`. Viser may also pick the next free port by itself; check the console output. |
| The planner puts a mobile base under a raised part | Standoff is 3D; add the part's stand to the mesh or draw a keep-out. |

**Known limitations (v1)**
- Planning features (heatmap, station planner, tool optimizer) need a **UR-type arm** (analytic IK and a full-turn base joint). Other uploaded robots can be displayed, jogged and feasibility-checked with numeric IK.
- Collisions are checked at each point only; the motion **between** points and stations is not checked. Travel distance ignores obstacles.
- Payload checks are **static**: no acceleration loads. Beyond 800 mm COG offset the UR20 payload curve is a **linear extrapolation (estimate)**.
- The part footprint for standoffs is a convex hull, which is conservative.
- **Not yet confirmed in a real browser** (only in automated and headless tests): gizmo dragging, contact-face clicking, shift/ctrl modifiers, box select, file upload and download. Please report anything odd.

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
- Each milestone's acceptance test lives in `tests/` (for example `test_kinematics.py`, `test_feasibility.py`, `test_reachability.py`, `test_placement.py`, `test_tool_geometry.py`, `test_carriers.py`, `test_export.py`).

UR robot models © 2023 Universal Robots A/S. Use hereof is subject to Universal Robots A/S' Terms and Conditions for Use of Graphical Documentation (see `assets/robots/ur20/meshes/LICENSE.txt`).
