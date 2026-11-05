# Robot Cell & End-Effector Optimizer — Project Plan

Nov 4, 2026 · @Ben

## Goal and scope

Build a browser-based cell-layout tool that answers two questions for a robotic process (scanning, sanding, gripping): **where should the robot base (or its carrier) stand**, and **what is the shortest end-effector that still reaches every target point without collision or overload**. It is a cost-based optimizer with a 3D HTML GUI.

**Inputs**

- Robot arm: UR20 in v1, loaded through a robot registry so more arms (e.g. FANUC M-710iC) can be added later without code changes elsewhere. Each arm brings its URDF, a payload spec sheet and an IK solver.
- End-effector URDF (scanner, sander, gripper) with mass and center of mass.
- A rough flange-to-tool placement the user sets with a 3D gizmo.
- Contact regions on the end effector (sander pad, scanner standoff point, gripper fingertips).
- A part mesh and a set of target points on its surface, each with z along the surface normal.
- Optional carrier: fixed pedestal, 2-axis gantry, or 2-axis/3-DOF mobile base.

**Outputs**

- Optimal base pose(s), or carrier stations, and the order to visit them.
- Optimal flange-to-TCP transform, minimum tool length, and an envelope for the end-effector geometry.
- Per-point report: reachable or not, which station, joint solution, clearance, manipulability, joint torque usage.
- Exports: project JSON, CSV of TCP poses per station, an updated end-effector URDF, and an HTML/PDF summary.

**Out of scope for v1:** full trajectory planning between points, cycle-time simulation, STEP import, and dynamic (acceleration) payload checks. Each is listed as a later milestone.

## Architecture and tech stack

Write the **whole tool in Python, using Viser for the 3D GUI**. Viser is a Python library that serves an interactive Three.js viewer and control panels to the browser, but you never write HTML, JavaScript or a REST API. It has built-in URDF display, transform gizmos, sliders, dropdowns, file upload and click events on the scene. That removes the separate frontend, the API layer and the type syncing, which is roughly half the project. Run `python app.py` and open the page.

| Layer | Choice | Why |
| --- | --- | --- |
| 3D viewer + GUI | Viser (`viser`, with its `ViserUrdf` helper on `yourdfpy`) | URDF display, `add_transform_controls` gizmos, GUI folders, tabs, sliders, file upload, scene click / pointer events, all from Python |
| Kinematics | Pinocchio (FK, Jacobians, gravity torques via RNEA) | C++ speed from Python, reads URDF directly |
| Analytic IK | UR analytic IK for the UR20 in v1; OPW kinematics added when the first 6R spherical-wrist industrial arm (e.g. FANUC) joins the registry | Returns all 8 solutions in microseconds; key for reachability sweeps |
| Fallback IK | Multi-start damped least squares on Pinocchio | Works for any URDF, including rail or gantry + arm as one chain |
| Collision | Coal (formerly hpp-fcl) via Pinocchio, or `python-fcl` | Distance queries, not just yes/no, so clearance and standoff can be costs |
| Mesh tools | `trimesh` (+ CoACD for convex decomposition) | Ray casting for surface picking, region growing, point sampling, normals |
| Optimizers | CMA-ES (`cma`), SciPy, OR-Tools or PuLP for set cover | Derivative-free outer loops; exact or greedy station selection |
| Parallelism | `multiprocessing` / `joblib` over candidates, optimizer in a background thread | Keeps the GUI responsive while results stream into the scene |
| Data | Pydantic models, saved as JSON / YAML | One schema for projects and the tool library |
| Tests | `pytest` + a few golden scenes | Regression on reachability counts and optimal lengths |

**Viewer options compared**

| Option | Effort | Fit | Verdict |
| --- | --- | --- | --- |
| Viser (Python, browser-rendered) | Low: one language, no frontend build | URDF, gizmos, panels and click picking built in | **Recommended** |
| Custom HTML: FastAPI + Three.js + `urdf-loader` | High: two codebases, API, type sync | Total control over UI, easiest to ship as a hosted web app later | Upgrade path if you outgrow Viser |
| PyVista / trame | Medium | Strong mesh picking (cell / area selection), weaker URDF and gizmo support | Good if surface selection becomes the hardest part |
| MeshCat | Low | Display only, no GUI controls | Too limited |
| RViz / Foxglove (ROS) | Medium–high | Great robot display, but requires ROS and is clumsy for setup forms | Only if you already live in ROS |

Keep the core math (`core/`, `opt/`) free of any viewer imports, so you can move to a custom Three.js frontend later without touching the optimizer.

## Core concepts and conventions

The single most important design decision: **the end effector is modeled as a fixed tool head plus a parametric extension**, so the optimizer can shorten or lengthen it and the collision geometry stretches with it. A static EE URDF alone cannot be optimized for length.

**Frames (write these into CLAUDE.md so every session uses them)**

- `world`: part and cell origin, z up, meters, radians internally (degrees in the GUI).
- `base`: robot base link; pose = carrier pose × mount offset.
- `flange`: tool0 / flange frame from the robot URDF.
- `tcp_i`: one frame per contact region on the tool. Its z axis points **out of the tool, toward the part**.
- `target_j`: one frame per surface point. Its z axis is the **outward surface normal**. A target is reached when `tcp.z = -target.z` (tool points into the surface) and origins coincide within tolerance.

**Target point tolerances** (per point or per group; these give the optimizer freedom and matter a lot)

- Spin about the normal: free (scanner, round sander) or a range (e.g. ±30° for an oriented gripper).
- Tilt cone: allowed angle between tool axis and normal (0° for a gripper, 5–15° for a sander, a few degrees for a scanner).
- Standoff: distance along the normal (0 for contact, e.g. 150–250 mm for a scanner's working range).

**Parametric end effector**

- `T_flange_mount`: fixed adapter (tool changer, insulating plate).
- Extension: a straight or one-bend arm of length L (and optional bend angle θ, lateral offset d). Collision = capsule or box of length L, sized by the user.
- Tool head: the loaded EE URDF meshes, attached at the end of the extension.
- `T_head_tcp_i`: contact frames on the head, defined in the GUI.
- Optimization variables: L, θ, d, and optionally the head roll about the flange axis, each bounded around the user's rough placement.

**Contact regions**

- The user picks links or mesh faces on the EE that are allowed to touch the part. Those faces are excluded from tool-to-part collision checks (or checked with a negative allowance).
- Each contact region gets a TCP frame. A target may be served by any TCP in its allowed set, e.g. either side of a two-sided sander.

**Payload model**

- EE mass and COM come from the URDF `<inertial>` tags; the extension adds mass per meter (user setting).
- Process force: a constant normal force (e.g. 50 N sanding) applied at the TCP along -target.z.
- Checks: static joint torques τ = g(q) + Jᵀ F against joint limits, plus the robot's payload diagram (COM distance vs mass) and wrist moment limits from the spec sheet.

## End-effector library and setup page

Add a separate **Tool Setup** tab where each end effector is configured once and saved to a reusable tool library; a project then just picks a tool and a part. This replaces GUI steps 3–6 below with a guided page.

**Tool library entry** (one YAML/JSON file + URDF + meshes per tool)

- Name, URDF, mass / COM (read from URDF, overridable), extension settings (straight or bent, length and angle ranges, radius, mass per meter).
- Process type (below), which loads default tolerances and checks.
- One or more TCPs, each with its frame on the tool head and its contact links or faces.
- Rough mounting on the flange (`T_flange_mount`), set with a gizmo.

**Process types and their defaults** (starting values only; every field is editable per tool and per point group)

| Process | Contact | Standoff along normal | Tilt cone | Spin about normal | Extra checks |
| --- | --- | --- | --- | --- | --- |
| Scanning / inspection | No; whole tool kept clear of part | User range, e.g. scanner working distance | Small, from scanner spec | Usually free | Optional line-of-sight ray from sensor to point |
| Sanding / polishing | Pad only | 0 (on surface) | Small, e.g. 0–10° | Free for round pads | Process force in torque check |
| Grinding / deburring | Wheel or bit | 0 | Process-specific lead angle | Constrained to travel direction | Process force; edge targets |
| Gripping / pick | Fingertips only | 0 at grasp, plus an approach distance | 0 | Constrained, e.g. 0° or 180° | Approach and retreat poses must also be reachable |
| Spraying / dispensing | No | Fixed nozzle distance | Small | Free | Clearance along spray cone |
| Custom | User-defined | User-defined | User-defined | User-defined | User-defined |

**TCP definition**

- Place a TCP frame on the tool with a gizmo, snap it to a clicked surface point or a link origin, or type xyz/rpy.
- Show the TCP axes in the scene; z points out of the tool toward the part.
- Pick contact faces by clicking the tool mesh; they turn a different color and are excluded from tool–part collision.

**Surface selection and point generation on the STL**

- **Pick a surface by click:** ray-cast the click onto the part mesh (trimesh), then grow a region across neighboring faces whose normals are within an angle threshold (e.g. 15°). One click selects a smooth face like a CAD face; shift-click adds more, a slider adjusts the threshold.
- **Box / lasso select** as a fallback for faceted meshes, and **select by normal direction** (e.g. all faces facing up ±30°).
- **Generate on the selection:** projected grid with mm spacing, Poisson-disk even spacing, raster lines with step-over, or edge-follow lines for deburring.
- **Edit points:** delete, flip normal, set group and tolerances; preview the tool at any point to sanity-check orientation.
- Store points in the part frame, so moving the part in the cell keeps them valid.

## Optimization formulation

Solve it as **three nested problems** rather than one giant one: tool geometry outside, base/station placement in the middle, per-point IK inside. Each layer is small enough to solve well, and the inner layers can be cached. A pipeline diagram is at the end of this section.

**Decision variables**

| Layer | Variables | Typical size | Solver |
| --- | --- | --- | --- |
| Outer: tool geometry | L, bend θ, lateral offset d, head roll | 1–4 continuous | CMA-ES or a 1D golden-section search on L when only length varies |
| Middle: placement | Per station: x, y, (z), yaw; number of stations k | 3–4 per station | Reachability-map scoring → set cover → CMA-ES refinement |
| Inner: per point | IK branch, spin about normal, tilt within cone, which TCP | 1–3 continuous + discrete | Analytic IK over a spin/tilt sample grid, then local polish |

**Hard constraints** (a candidate that violates any is infeasible for that point)

- An IK solution exists within joint limits (with a margin, e.g. 5°).
- No collision: robot self, robot–part, robot–environment, tool–part except contact faces, with a clearance margin (e.g. 10 mm).
- Static torque on every joint ≤ a utilization limit (e.g. 80%) and EE mass/COM inside the payload diagram.
- Carrier pose inside its travel limits and outside keep-out zones.

**Cost function** (all terms normalized to roughly 0–1, weights exposed in the GUI)

```latex
J = w_L \frac{L}{L_{ref}} + w_U N_{unreached} + w_S (k-1) + w_T \frac{D_{travel}}{D_{ref}} + w_M \overline{\frac{1}{1+\mu}} + w_C \overline{\max(0, 1 - \frac{c}{c_{ref}})} + w_J \overline{s_{limit}} + w_P \max \tau_{util}
```

- L: tool length (the headline objective).
- N\_unreached: unreachable points, with a huge weight so coverage dominates.
- k − 1: extra stations beyond one; D\_travel: carrier travel between stations (minimize base moves).
- μ: Yoshikawa manipulability at each point (avoid singularities); c: clearance; s\_limit: closeness to joint limits; τ\_util: worst joint torque utilization.
- Optional: joint-space distance between consecutive points and penalties for wrist or elbow flips within a station, so the path is smooth.

**Solver strategy (the part that makes it fast)**

1. **Reachability map (precompute once per robot + tool geometry).** Voxelize the space around the robot base (e.g. 25–50 mm). For each voxel, sample tool orientations, run analytic IK, store a reachability score and the best manipulability. This is the classic capability / inverse reachability map approach.
2. **Inverse reachability for each target.** Transform the map into the target's frame: every base pose from which that target is reachable. Accumulate votes over all targets on a base-pose grid (x, y, yaw). Peaks are good base locations. This is fast because it is lookups, not IK.
3. **Station selection as set cover.** Each candidate base pose covers a set of points. Pick the fewest poses that cover all points (greedy, then ILP with OR-Tools for under \~500 candidates), with travel distance as a tiebreaker. Order stations with a small TSP.
4. **Exact verification + refinement.** For the chosen stations, run full IK + collision + torque checks per point; refine each base pose with CMA-ES on the exact cost.
5. **Outer loop on tool length.** Repeat steps 1–4 for candidate L. Coverage vs L is mostly monotonic, so start with a bisection for the shortest L that keeps 100% coverage at the allowed station count, then let CMA-ES tune θ, d and roll.

**Caching:** the reachability map depends only on robot + tool geometry, so cache it to disk keyed by a hash of those inputs. Moving points or the part never invalidates it.

**Pipeline**

&#91;embedded content: solver pipeline · 3 nested loops\]

Only step 1 depends on tool geometry alone, so it is cached; steps 2–4 rerun cheaply whenever points, part or carrier limits change.

## Base mode and carriers: static, rail, gantry, mobile, positioner

Treat every carrier the same way: **a carrier is a small set of extra degrees of freedom that place the robot base**, with limits and a travel cost. The middle optimizer layer then works unchanged; only the search space and the travel metric differ.

**Base mode** is the first choice in the cell setup: **Static** (one fixed base pose, k = 1) or **Movable** (the base rides on a carrier and may stop at several stations). Choosing Movable then asks for the carrier type below.

| Base mode | Carrier | Variables | Search space | Travel cost | Notes |
| --- | --- | --- | --- | --- | --- |
| Static | Pedestal / floor / wall / ceiling | x, y, z (height), yaw, once | Floor region the user draws, height range | None | Mount upright, wall or inverted via a mount transform |
| Movable | 1-axis rail (7th axis) | Rail position s per station; rail pose itself optional to optimize | Rail travel limits | Rail distance | Rail can be floor or overhead; optimizing the rail's own placement answers "where should the track go" |
| Movable | 2-axis rail / gantry | X, Y carriage position per station; Z mount height fixed or a 3rd axis | Box = gantry travel limits | Gantry axis distance (or time with per-axis speed) | Robot usually inverted; optional coordinated mode treats axes as extra IK joints |
| Movable | Mobile base (AMR) | x, y, yaw per station | Floor polygon minus part footprint, keep-outs and standoff zones | Path length on the floor (straight line in v1, grid A\* later) | Verify reach with base pose perturbed by a localization tolerance, e.g. ±10 mm / ±1° |
| Movable | Part positioner | Part pose (turntable angle, or x/y/yaw on a fixture) | User-defined ranges | Positioner moves, usually cheaper than base moves | Same math; the relative pose base-to-part is what matters |

**Mobile base built from primitives (v1)**

In v1 the mobile base is not a vendor model. The user builds it from simple shapes in the Cell tab, and the tool turns those shapes plus a joint chain into a URDF. Pinocchio, the collision checker and Viser then load it like any other robot.

- **Geometry:** a chassis (box L × W × H or cylinder D × H), an optional riser or column (box or cylinder), an optional mount plate, and optional wheels drawn as cylinders for looks. All sizes in mm in the GUI.
- **Kinematics:** world → x (prismatic) → y (prismatic) → yaw (revolute) → optional lift (prismatic z) → mount (fixed) → UR20 `base_link`. Each joint has limits; x/y are also clipped by the floor polygon, keep-outs and standoff zones.
- **Mount frame:** xyz + rpy offset from the chassis origin to the UR20 base, set with a gizmo or typed in. Supports upright, side and angled mounts.
- **Collision:** the primitives are the collision geometry, so standoff and clearance checks are fast and exact.
- **Mass (optional):** chassis mass and COM enable a static tip-over check: the combined COM of base + arm + tool, plus process force, must stay inside the wheel support polygon with a margin.
- **Presets:** save a built base to a carrier library (one YAML per base), the same way tools are saved.
- **Same builder for rails and gantries:** a 1-axis rail is world → x → mount; a 2-axis gantry is world → x → y → mount, usually inverted. Only the joint list and the shapes change.

**Standoff distances between carrier and part**

- Each carrier (and the robot's base links) gets collision geometry: the URDF meshes, or a simple box / cylinder the user sizes for a mock model.
- The user sets a **minimum standoff** from carrier collision geometry to the part, e.g. 300 mm for a mobile base. This is a hard constraint: a station closer than the standoff is infeasible.
- An optional **preferred standoff** adds a soft cost when the base is closer than it, so the optimizer stays out of tight gaps unless it must.
- Separate standoffs per object class: carrier ↔ part, carrier ↔ fixtures / walls / keep-outs, and robot arm ↔ part (the clearance margin from the Optimization section).
- In the GUI, draw the standoff as a translucent offset shell around the part footprint on the floor, so the engineer sees where a base can stand.
- Implementation: a signed-distance check between carrier geometry and part (Coal distance query), plus a fast 2D pre-filter on the floor (part footprint polygon buffered by the standoff, using `shapely`) before any 3D checks.

**Two gantry modes worth supporting**

- **Station mode (v1):** the gantry stops at k positions and the arm does the work. Minimizes gantry moves, matches the stated goal.
- **Coordinated mode (later):** gantry axes move with the arm. Model gantry + arm as one URDF chain, use numerical IK with a secondary objective that keeps gantry motion minimal (null-space weighting).

**Mock carrier models:** ship simple parametric URDF generators in the repo (`gantry_2axis(x_travel, y_travel, z_height)`, `mobile_base(footprint, deck_height)`), so the user does not need a real model; add a 1-axis rail generator the same way. The robot URDF is attached to the carrier's mount link at load time.

## GUI spec

One Viser page: the 3D viewport plus a control panel with four tabs: Tool Setup, Cell, Optimize, Results. Steps 3–6 live in Tool Setup (see the End-effector section); the rest follow the workflow top to bottom.

1. **Robot.** Load URDF as a folder or .zip (resolve `package://` paths), or pick from a bundled library. Set mount type (floor, wall, inverted). Drag the base with a gizmo. Joint sliders to jog. Enter or import the payload spec sheet.
2. **Base mode.** Choose Static or Movable; for Movable pick 1-axis rail, 2-axis gantry, mobile base or positioner. Set travel limits, keep-out zones and carrier-to-part standoffs (minimum and preferred), shown as shells on the floor.
3. **End effector.** Load URDF. Place it on the flange with a gizmo (this becomes the rough `T_flange_head`). Set extension type (straight / bent), min–max length, bend range, capsule radius, mass per meter.
4. **Contact regions.** Click faces or links on the EE to mark them as contact; place a TCP frame on each with a gizmo; set process force.
5. **Part.** Load STL / OBJ / PLY; set units and pose.
6. **Targets.** Generate points (see below), then edit: select, delete, group, set tolerances per group (spin, tilt cone, standoff).
7. **Optimize.** Choose what is free (tool length only, base only, both, station count), set weights and limits, run. Progress streams live.
8. **Results.** Ranked candidate list; click one to load it into the scene. Points colored green / amber / red (reachable / marginal / unreachable), hover for details. Scrub through points to see the robot pose at each. Export.

**Point generation tools**

- **Projected grid:** define a plane and direction (or use the view direction), set spacing in mm, ray-cast onto the part; keep hits only on faces facing the ray.
- **Even surface sampling:** Poisson-disk sampling with a minimum spacing, optionally restricted to a painted region.
- **Raster paths:** parallel lines across a selected region at a step-over, sampled every N mm (natural for sanding and scanning).
- **Click to place:** single points on the surface.
- **Import CSV:** x, y, z, nx, ny, nz (+ optional group).
- Normals from area-weighted vertex normals with an option to use face normals; a toggle to flip normals if the mesh is inside-out; show normals as short arrows.

**Visual feedback that helps the engineer**

- Reachability heatmap on the floor for the current tool (from the inverse reachability votes), so they can see good base regions before optimizing.
- Ghosted robot at each station; lines from the chosen station to the points it serves.
- Collision pairs highlighted in red with the minimum distance labelled.
- A cost breakdown bar for the selected candidate so the weights make sense.

## Data model / project file

One JSON project file is the source of truth; the frontend edits it, the backend reads it. Define it with Pydantic on the backend and generate TypeScript types from the JSON Schema so both sides stay in sync. Asset files (URDFs, meshes) sit next to it in a project folder.

```json
{
  "version": 1,
  "units": "m",
  "robot": {
    "urdf": "assets/ur20/ur20.urdf",
    "flange_link": "tool0",
    "mount": "floor",
    "base_pose": {"xyz": [1.2, 0.0, 0.0], "rpy": [0, 0, 1.57]},
    "ik": "ur_analytic",
    "payload_spec": {"rated_kg": 20, "wrist_moment_Nm": [null, null, null], "joint_torque_Nm": [null]}
  },
  "base": {"mode": "movable", "carrier": "mobile", "chassis": {"shape": "box", "size": [1.2, 0.8, 0.4], "kg": 300}, "riser": {"shape": "cylinder", "radius": 0.15, "height": 0.3}, "joints": {"x": [0, 6.0], "y": [0, 4.0], "yaw_deg": [-180, 180], "lift": [0, 0.5]}, "T_mount": {"xyz": [0.2, 0, 0.7], "rpy": [0, 0, 0]}, "keepouts": [], "standoff_m": {"carrier_part_min": 0.3, "carrier_part_preferred": 0.5, "carrier_env_min": 0.1}},
  "end_effector": {
    "tool": "tools/orbital_sander.yaml", "process": "sanding", "urdf": "assets/sander/sander.urdf",
    "T_flange_mount": {"xyz": [0, 0, 0.02], "rpy": [0, 0, 0]},
    "extension": {"type": "bent", "L": [0.05, 0.6], "theta_deg": [0, 45], "radius": 0.04, "kg_per_m": 2.0},
    "tcps": [{"name": "pad", "T_head_tcp": {"xyz": [0, 0, 0.12], "rpy": [0, 0, 0]}, "contact_links": ["pad_link"]}],
    "process_force_N": 50
  },
  "part": {"mesh": "assets/panel.stl", "pose": {"xyz": [0, 0, 0.8], "rpy": [0, 0, 0]}},
  "targets": {
    "groups": [{"name": "top", "spin_deg": "free", "tilt_deg": 10, "standoff_m": 0, "tcps": ["pad"]}],
    "points": [{"id": 0, "p": [0.1, 0.2, 0.9], "n": [0, 0, 1], "group": "top"}]
  },
  "optimizer": {
    "free": ["L", "theta", "base"],
    "max_stations": 3,
    "weights": {"L": 1.0, "unreached": 1000, "stations": 5, "travel": 0.5, "manip": 0.3, "clearance": 0.3, "limits": 0.2, "torque": 0.5},
    "margins": {"clearance_m": 0.01, "joint_deg": 5, "torque_util": 0.8}
  },
  "results": []
}
```

The `payload_spec` values are placeholders: fill them from the manufacturer's datasheet for each robot. The tool should refuse to run payload checks (and say so) when they are missing, rather than guessing.

## Repo structure

```
cell-optimizer/
  CLAUDE.md                  # frames, units, conventions, commands, milestone status
  app.py                     # Viser server entry point
  ui/
    tool_setup.py            # Tool Setup tab: EE load, process type, TCPs, contact faces
    targets_ui.py            # surface picking, region growing, point generation, editing
    cell.py                  # Cell tab: robot, base mode, carrier, part, standoffs
    optimize.py              # Optimize tab: free variables, weights, run, progress
    results.py               # Results tab: candidates, heatmaps, scrubbing, export
  core/
    schema.py                # Pydantic project + tool-library models
    robot.py                 # URDF load, Pinocchio model, FK, Jacobian, gravity torques
    ik/opw.py                # analytic IK, 6R spherical wrist
    ik/ur.py                 # analytic IK, UR family
    ik/numeric.py            # multi-start DLS fallback
    collision.py             # Coal/FCL scene, distance queries, allowed-contact filter
    tool.py                  # parametric EE: mount + extension + head + TCPs
    process.py               # process-type presets and their default tolerances
    carrier.py               # primitive-based builder -> URDF for mobile / rail / gantry
    standoff.py              # 2D footprint buffers + 3D carrier-to-part distance
    targets.py               # region growing, grid projection, Poisson, raster, normals
    payload.py               # torque and payload-diagram checks
  opt/
    reachability.py          # reachability + inverse reachability maps, disk cache
    placement.py             # base voting, set cover, station ordering, refinement
    tool_geometry.py         # outer loop over L, theta, d
    cost.py                  # cost terms + breakdown
  tools/                     # tool library: one YAML per end effector
  assets/
    robots/                  # registry: ur20/ (URDF, meshes, spec.yaml) in v1
    tools/                   # EE URDFs and meshes
    parts/                   # flat plate, curved panel, box, long rail test parts
  examples/                  # ready-to-run project files
  tests/                     # pytest, golden scenes
```

**Robot registry:** each robot is a folder under `assets/robots/<name>/` holding a resolved URDF + meshes, `spec.yaml` (payload, joint torque limits, flange link name) and the name of its IK solver. Code looks robots up by name and never hard-codes the UR20. v1 ships only `ur20/`, taken from Universal Robots' ROS 2 description package; it ships as xacro, so run a one-time `xacro` → plain URDF conversion and commit the result after checking the license. Adding a robot later = add a folder, plus an IK solver if its kinematic family is new.

## Milestones for Claude Code

Build backend math first with tests, then the viewer, then wire them. Each milestone is sized for one or two Claude Code sessions and ends with a check you can run.

1. **M0 Skeleton.** Repo layout, CLAUDE.md, Viser app showing a grid and a test mesh, Pydantic schema, pytest and lint in place.
   - Done when: `python app.py` serves the page and `pytest` passes.
2. **M1 Robot kinematics.** Robot registry; load the UR20 into Pinocchio by name; FK, Jacobian, gravity torques; UR analytic IK; numeric IK fallback.
   - Done when: IK round-trip test (random q → FK → IK → FK) matches within 1e-6 m on 10,000 samples, all solution branches found.
3. **M2 Viewer.** Viser tabs (Tool Setup, Cell, Optimize, Results); load robot URDF (folder/zip), joint sliders, base gizmo, part mesh load, save/load project JSON.
   - Done when: robot pose in the viewer matches Pinocchio FK for the same q.
4. **M3 Tool Setup page.** EE URDF load, process-type presets, flange gizmo, extension with L/θ/d, TCP frames, contact face picking, mass model, save to the tool library.
   - Done when: changing L in the GUI stretches the collision capsule and moves the TCP correctly.
5. **M4 Targets.** Click-to-select surfaces with region growing, box and normal-direction select; then projected grid, Poisson sampling, raster and edge paths, CSV import, normals display, groups with tolerances.
   - Done when: a 10 mm grid on the curved test panel has normals within 1° of analytic normals.
6. **M5 Feasibility checker.** For a given scene, per-point IK over spin/tilt samples, collision with contact filtering, payload checks; colored points in the GUI.
   - Done when: on the golden scenes, reachable counts match hand-verified expectations.
7. **M6 Reachability maps.** Voxel reachability map, inverse map per target, base voting heatmap on the floor, disk cache.
   - Done when: heatmap peak lies within 50 mm of the exact best base found by brute-force grid search on the flat-plate scene.
8. **M7 Placement optimizer.** Set cover for stations, ordering, CMA-ES refinement, results list streamed over WebSocket.
   - Done when: on a scene designed to need 2 stations, it finds 2 and full coverage.
9. **M8 Tool-geometry optimizer.** Outer loop on L (bisection) then θ/d/roll (CMA-ES); cost breakdown.
   - Done when: on a scene with a known minimal L (a pocket of known depth), result is within 5 mm.
10. **M9 Carriers.** Static / movable base mode; primitive-based carrier builder (mobile base with chassis, riser, lift and mount frame first; then 1-axis rail and 2-axis gantry), optional tip-over check; station mode for all; keep-outs and carrier-to-part standoff zones (hard minimum + preferred); localization tolerance for mobile. Done when: on the long-rail scene, a 1-axis rail solution respects a 300 mm standoff and needs fewer stations than a mobile base.
11. **M10 Exports and report.** CSV poses per station, updated EE URDF with optimal geometry, HTML summary with screenshots.
12. **Later.** Additional robots (e.g. FANUC M-710iC with OPW IK), coordinated gantry mode, STEP import, collision-free paths between points, cycle time, dynamic payload.

**Golden test scenes** (build these in M0–M1, use them everywhere): a flat 1 × 0.6 m plate, a curved panel, an open box with a deep pocket (forces a long tool), and a long rail that a fixed robot cannot cover (forces stations).

## Risks, open decisions and Claude Code tips

**Risks**

- **URDF quirks:** xacro, `package://` paths, DAE units and up-axis differences. Mitigate with an import step that resolves and validates, and shows the model next to a 1 m reference cube.
- **Collision speed:** raw CAD meshes make distance queries slow. Use convex decomposition or simplified collision meshes; check broad-phase first.
- **IK multimodality:** numerical IK misses solutions and makes coverage noisy, which confuses the optimizer. Prefer analytic IK wherever the arm allows it.
- **Normals:** bad or flipped mesh normals silently ruin reachability. Show them and validate early.
- **Payload data:** torque and payload-diagram limits are not in URDFs and vary by variant. Keep them in a per-robot spec file filled from datasheets.
- **Cost tuning:** weighted sums can hide trade-offs. Show the cost breakdown and a coverage-vs-length curve so the engineer can choose.

**Open decisions** (decide before M5)

- [ ] Is Viser click picking + region growing good enough on your real part meshes, or is PyVista needed for selection? Prototype this first in M4.
- [ ] Tool geometry family for v1: straight only, or straight + one bend?
- [ ] Is a single tool length for all stations required, or can tools be swapped per station?
- [ ] Station mode only for the gantry in v1, or coordinated mode too?
- [ ] What output does the mechanical designer need: an envelope sketch, a URDF, or a STEP stub?

**Working with Claude Code**

- Put the Frames section and the cost definition into `CLAUDE.md`, plus the current milestone and the commands to run tests. This keeps conventions stable across sessions.
- Start each milestone in plan mode, paste the milestone text from this doc, approve the plan, then let it build.
- Ask for tests first for M1, M5 and M6: frame and sign errors are the most common bugs, and tests catch them.
- Keep one milestone per branch and commit when its check passes.
- Ask Claude Code to add a debug view for each math layer (draw frames, reachability voxels, candidate bases); visual checks find bugs quickly in robotics code.
