# v3 plan: second round of user feedback (2026-11-06)

Branch `v4` (from `main` at a572f3f). Decisions marked **(user)** were answered on 2026-11-06.

| # | Feedback | Plan |
| --- | --- | --- |
| 1 | Object tree on the left side to show / hide objects | A Viser panel docked on the left (`gui.add_panel().dock_left()`): a tree of the cell (robot + tool, carrier + each body, part + targets, stay-out zones, results: stops, ghosts, paths), each entry with show / hide and "gizmo on this"; the jog arrows and gizmo settings move there. |
| 2 | Which side the mobile robot can operate from | **Both (user):** robot sides that may face the part, any of **Front / Side 1 / Side 2 / Rear** (several allowed; the planner chooses per stop), and the **part sides** it may work from (part front / back / left / right, in the part frame). The contour is built per allowed robot side and only kept along the allowed part sides. |
| 3 | Force mobile robot / external axis parallel to the part | **(user: option 1 + robot sides):** rail / gantry: checkbox *Align track with the part* turns the track along the part's long axis (part footprint, minimum-area rectangle) and centres it on the part, keeping its current side and distance; re-applied when the part moves. Mobile: "parallel" = Side 1 / Side 2 facing the part (item 2). |
| 4 | Start a new study | **(user):** **New empty study**: no robot, no tool, no part, no stay-out geometry, no carrier (everything chosen from scratch; the GUI works without a robot until one is chosen). **New part, same cell**: clears the part, the surface selection, targets, results, stop locations, feasibility and paths; keeps robot, tool, carrier and stay-out zones. |
| 5 | Red zone on the floor; stay-out zones | The red floor zone is a carrier keep-out (2D, carrier only; the arm ignores it). **(user):** 3D stay-out **boxes and STL meshes** that the **arm, tool and carrier** all avoid (planner, check, paths), drawn as see-through red volumes, editable with a gizmo, saved in the project. |
| 6 | Default units mm for imports | Part upload and CSV import default to mm (carrier meshes already do). |
| 7 | STL see-through from one side | Part, carrier, tool head and stay-out meshes drawn double-sided. Normals are deliberately not "repaired" on load (flipping faces could silently flip the generated target normals of existing parts). |
| 8 | Gizmo size; gizmo inside a body cannot be grabbed | *Gizmo size* slider for all gizmos and *Gizmos always on top* (drawn through bodies, so they can be grabbed). |
| 9 | Paths should clear | Paths cleared when the station planner or the tool optimizer starts, or the targets change. |

## Status: all nine items done on branch `v4` (2026-11-06)

Tests: tests/test_v4_quick.py (9, 6, 7, 8), test_v4_study.py (4), test_v4_stayout.py (5), test_v4_sides.py (2, 3), test_v4_tree.py (1). Also: tests run in a private temporary workspace (tests/conftest.py, `RSO_WORKSPACE`) after test runs had been writing into the user's workspace; `scripts/test_batches.py` runs the suite in low-memory batches.

## Order

Quick items first (9, 6, 7, 8), then 4 (new part / new empty study), 5 (stay-out zones), 2 + 3 (mobile sides, rail alignment), 1 (left object tree, which then also lists the stay-out zones).
