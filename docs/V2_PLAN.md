# v2 plan: user feedback on v1 (2026-11-05)

Source: `v1_feedback.txt` (user's notes after testing v1). Decisions marked **(user)** were answered by the user on 2026-11-05.

## Findings from the code (what causes the reported problems)

| Feedback | Cause found |
| --- | --- |
| Examples give poor plans | Both example projects save `optimizer.max_stations = 1`, so **Plan stations** on the mobile example reaches 13/25 points and on the rail 14/25. |
| Cancel doesn't work | `plan_stations` checks `cancelled()` once per station count k; the refine + verify + zoning of one k takes 5–30 s. The tool optimizer checks between L samples, each a full plan. |
| Moving things lags | Joint slider step 5 ms (ViserUrdf `update_cfg`). Carrier slider step 12–21 ms: `CarrierPanel.draw` removes and re-adds every body mesh (full vertex data to the browser) and recomputes the part footprint and zones each event. |
| UR doesn't stay on the carrier | The Cell base gizmo sets `T_world_base` independently; the carrier is drawn from its DOF sliders. |
| Move to solution leaves the carrier | `ResultsTab` calls `set_base(station_T)` only; the carrier DOF sliders are not set. |
| Safety zone ignores part moves | `cell.part_listeners` fire only on mesh/units changes; `_on_part_pose` notifies nobody. |
| Autosave changes examples | Opening an example replaces `workspace/examples/<name>/` and autosave then writes into it; nothing stops a project folder inside `examples/`. |

Measured baseline (this PC, map cached): mobile example 27 s (36 s with zoning); rail example 5 s (14 s with zoning). Exact checks (scalar IK, `check_point`) are about 60 % of planning time.

## Phase 1: fixes (small, do first)

1. Examples: `max_stations` 4; the movable-base default ≥ 3. A test that opens each example through the GUI and plans to full coverage with the panel defaults.
2. Cancel within about 1 s: `cancelled()` per point in `verify` / `zone_solution`, per CMA-ES evaluation, and a CP-SAT solution callback that stops the search. Tool optimizer passes it down to every plan.
3. **Move robot to solution** sets the carrier DOFs (`station.carrier`) through the carrier panel, not just the base pose.
4. Optimize → *Inspect* shows the per-point results of the solution selected in Results (and the tool optimizer's result), refreshed on every new plan.
5. Part pose changes notify `part_listeners` (throttled) so the standoff shells and the footprint distance follow the part.
6. *L tolerance*: user-defined, no upper limit (min > 0).
7. Jog lag: carrier bodies added once and only re-posed (`wxyz`/`position`) on jog; part footprint cached per (mesh, pose); slider events coalesced (latest value wins, a worker thread applies them) for joints, carrier and gizmos.
8. Examples are read-only templates: opening one creates an untitled copy that autosave does not write until **Save as**; saving into `examples/` is refused except via **Save as example** (name prompt, explicit).

## Phase 2: guided setup, robot + carrier assembly

- Tabs become steps **1 Robot & cell → 2 External axis → 3 Tool → 4 Part & targets → 5 Optimize → 6 Results**, each with a status (to do / confirmed) and **Confirm / Next**. **Free navigation (user):** any tab can be opened; Optimize warns when an earlier step is not confirmed.
- Mode-specific settings are hidden unless the mode is selected (static vs rail vs gantry vs mobile; zoning cost only when zoning is on; bent-tool fields only for bent tools until Phase 4 replaces them).
- **Confirm setup** on the External Axis step locks the robot to the carrier mount: the Cell base gizmo is hidden, the base pose is always `carrier.T_world_base(dofs)`, and moving the assembly means moving the carrier. **Edit setup** unlocks it.
- **Stale results:** changes to robot, tool, carrier, part pose/mesh or targets mark feasibility and every solution as out of date (banner, greyed entries, export asks first). Rerun to clear. **Save optimized tool to library** from Results / the tool panel.
- **Object tree** folder: robot, carrier (bodies), part, tool, targets, keep-outs; each with visible, select (shows its gizmo, hides the others) and jog.
- **Arrow jog**: ±X / ±Y / ±yaw (and ±s for a rail) button groups with a step size. Rail jog is clamped to the travel; mobile jog has no limits.

## Phase 3: mobile setup and planner speed

- **Mobile standoff (user):** one fixed distance measured from the **mobile base outline to the part outline** (top view). The user picks **which side of the mobile base faces the part** (front / back / left / right). Stops then lie on the offset contour of the part footprint at that distance, with the heading set by the contour normal and the chosen side: a 1-D "virtual rail" around the part instead of an x, y, yaw search. Localization tolerance stays.
- Exact checks in batches: candidate poses of a point go through `solve_batch` (about 7 µs per pose against 0.4 ms) before limits, torque and collision.
- One reachability map per robot at the flange (TCP target → flange pose per sampled spin), reused for every tool and shipped prebuilt in the portable zip. Removes the map build when the tool changes, and supports the Phase 4 tool model.
- Quick plan first (map + verify, no refinement) in seconds, refinement streamed after; refinement and verify spread over processes (Pinocchio data is not thread-safe).
- Target: the mobile example under 10 s; record timings in a benchmark test.

## Phase 4: tool geometry model

- The head pose relative to the flange is **x, y, z, rx, ry, rz**, each with min/max (min = max fixes it). It replaces L / θ / d / roll. Old tools convert (their T_flange_head becomes the nominal, ranges from the old ranges).
- **Payload is not used in tool optimization (user feedback).** The torque check stays in feasibility.
- **Connector = smooth spline tube** from the flange to the head: a cubic Hermite curve leaving along the flange z axis and arriving along the head mount axis, radius set by the user. Visual = tube mesh; collision = chain of capsules along the curve; mass per metre along the arc length.
- **Objective (user): best overall cost J** within the limits (stations, unreached, manipulability, clearance, travel, limits, with connector length as one weighted term). CMA-ES over the active components; stations re-planned per candidate as now, from the quick planner.
- URDF export includes the connector mesh; the envelope sketch shows the spline.

## Phase 5: paths and simulation

- **Paths connect the planned points (user):** each stop / zone's points in the planner's serpentine order; segments follow the part surface (sampled between points, projected onto the mesh, normals interpolated).
- Joint trajectory along each path with continuity (IK per sample, same branch, unwrapped), collision checks along the path, and gaps reported (unreachable sample, configuration flip, collision).
- Simulation: play / pause / speed / scrub slider animating robot, tool and carrier through stops and paths. Path CSV export (TCP poses + joints per sample).

## Order

Phase 1 → 2 → 3 → 4 → 5. Phase 3's flange map is a prerequisite for Phase 4's general TCP offsets.
