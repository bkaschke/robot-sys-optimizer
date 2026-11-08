# v4 plan: targets and paths like the sand-optimizer, station grouping, exports, speed (2026-11-08)

Branch `v5` (from `v4` at 7e07e75, which is not merged into `main` yet). Reference app: the
user's `sand-optimizer` (TypeScript, `../sand-optimizer`), whose surface selection, grid and
path patterns are to be matched. Decisions marked **(user)** were answered on 2026-11-08.

## 1. Target generation (Targets tab)

* **One *Tool* dropdown; only its settings are shown (user):**
  * **Brush + / Brush −**: a brush **handle you drag** over the part **(user)**. Viser reports
    no mouse drags on meshes (only clicks and box selects), so the brush is a small gizmo kept
    on the surface: every face within the *Brush radius* of its path is selected (+) or
    deselected (−) as it moves. A click also paints one dab.
  * **Fill + / Fill −**: click a face to (de)select the connected smooth region; neighbouring
    normals may differ by at most the *Fill angle limit*.
  * **Box + / Box −**: drag a rectangle in the view.
  * **Full surface**: every face.
  * **Single point**: a click adds one target, **Remove point**: a click removes the nearest.
  * *Select all / Clear / Invert*, and the selected face count and area.
* **Generate points on selection** like the sand-optimizer: the grid lies in the plane of the
  selection's mean facing direction; rows along the part X axis projected onto that plane,
  turned by *Row direction*; *Row spacing* and *Column spacing* (may differ); points closer than
  *Edge inset* to the selection boundary are dropped; each node is projected onto the selected
  faces (outermost hit); tool z into the surface along the smoothed normal, tool x along the rows.
* The current generators (projected grid, Poisson, raster lines, edges, CSV import) stay
  available under *Tool* = *Other generators*.

## 2. Path patterns (Results → Paths)

* Pattern dropdown like the sand-optimizer, without point filtering and without LS / CSV
  input **(user)**: **Zigzag, Unidirectional, Cross-hatch, Concentric, Spiral, Hilbert,
  Shortest path (TSP)**, with *Raster angle*, *Start corner*, *Row tolerance* (0 = automatic),
  *Outside → in* (concentric / spiral), and *Prevent crossing paths*; only the selected
  pattern's settings are shown.
* Applied per stop and per configuration group: **every point of a stop is done before the
  carrier moves on** **(user)**, groups in an optimized order.

## 3. Station areas do not overlap (user: station areas)

* After planning, points are reassigned so that each stop's points form **one connected area
  on the part**, no point of one stop lies inside another stop's area, and the stops' paths do
  not cross: a neighbourhood graph of the targets (k nearest within ~1.5 × spacing), points
  that sit among another stop's points move to that stop when it reaches them exactly
  (re-checked), repeated until stable. Plans where this leaves a stop split are flagged.
* **Surface colour layer:** the part surface is painted by stop (and group), using the same
  colours as the station lines, so the grouping is visible on the part; on / off in the
  object tree.

## 4. Path with carrier moves (user)

* Between stops: the arm goes to **home**, the carrier (AMR / rail) moves to the next stop, and
  the arm goes from home to the next stop's first point. These moves are part of the playback
  and of the path CSV (carrier values per sample), and are checked against stay-out zones.

## 5. Planner settings and costs

* **Max stations up to at least 30** (GUI limit 40). The planner stops adding stops once the
  cost has not improved for two counts, so 30 does not mean 30 full plans.
* **Editable cost weights** in the GUI (tool length, travel, manipulability, clearance, joint
  limits, torque, standoff) with the count costs spelled out **(user, 2026-11-08)**:
  * **Cost per stop** (each base / carrier position beyond the first: fewer base changes);
  * **Cost per group** (each extra configuration group at a stop: fewer groups);
  * **Cost per wrist flip** (a joint-4 / joint-6 jump > 180° inside a group's path).
  Today the first is the *stations* weight (5) and the last two are lumped into *Cost per
  config change* (1); they become separate weights (`Weights.group`, `Weights.wrist_flip`).
* **Must be met (user):** *All points reachable* and *No configuration change within a group*
  (a stop may hold several groups, each in one configuration). With them ticked the planner
  keeps adding stops until they hold (up to *Max stations*); plans that break a ticked rule are
  rejected (shown ✗, not ranked).

## 6. Speed (user: anything that helps)

* **Coarse to fine:** station search and the tool optimizer's candidates work on a subsample of
  the targets (farthest-point sampling, e.g. a quarter, at most ~150 points); the winners are
  verified on every point and repaired (an extra stop if some points are missed).
* Per-stop work in parallel processes where it pays (measured first; Pinocchio data is per
  process anyway).

## 7. USERFRAME export (user: robot-dependent CSV)

* **Results → Export → Userframes (CSV):** one row per stop: carrier values, and the transform
  from the **robot base frame to the part frame** in the loaded robot's convention: UR arms
  `x, y, z` (mm) + rotation vector `rx, ry, rz` (rad, as in a UR pose `p[...]`); FANUC-style
  arms `X Y Z W P R` (mm, deg); every robot also gets the 4 × 4 matrix columns. The convention
  comes from the robot's `spec.yaml` (`frame_convention`), so new robots only need that entry.

## Order

1 (targets) → 2 + 4 (patterns, carrier moves) → 3 (station areas + colour layer) → 5 (costs,
must-meet, 30 stops) → 6 (speed) → 7 (userframes, small, may go first).

## Status (2026-11-08, branch `v5`)

All seven items are implemented and tested (`tests/test_v5_*.py`). Deviations from the text above:

* **1:** the existing generators sit in a second *Generator* dropdown (with the new *Surface grid*
  and *Part-anchored lattice*) rather than under *Tool = Other generators*; each tool and each
  generator shows only its own settings.
* **3:** reassignment is region growing from the point nearest each base (exact checks only),
  then a majority clean-up; kept only when it leaves fewer pieces. Path crossings between stops
  are not checked separately.
* **5:** a wrist flip inside a group also breaks *No configuration change within a group*. The
  must-meet rules apply to the station planner; the tool optimizer keeps its own criterion
  (every target reached).
* **6:** coarse to fine is done (43 s -> 18 s on 357 points); parallel processes were measured and
  left out (a worker starts in ~2.7 s, about the time one stop's refinement takes).
