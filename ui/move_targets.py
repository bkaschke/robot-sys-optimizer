"""The objects Move mode can move (ui.move_mode, v7 round 2 T11), keyed like the Objects tree:
"Part", "Robot base" (or its carrier), "Stay-out zone/<name>", "Reduced-speed zone/<name>",
"Carrier body/<name>"."""

from __future__ import annotations

from ui.move_mode import Target


def move_keys(objects) -> list[str]:
    s = objects.s
    keys = ["Robot base"]
    if s.part is not None:
        keys.append("Part")
    keys += [f"Stay-out zone/{z.name}" for z in s.stayouts if z.speed_pct is None]
    keys += [f"Reduced-speed zone/{z.name}" for z in s.stayouts if z.speed_pct is not None]
    carrier = objects.carrier
    if carrier is not None and s.base.mode == "movable" and not carrier.locked:
        keys += [f"Carrier body/{b.name}" for b in carrier.editor.bodies]
    return keys


def move_target(objects, key: str) -> Target | None:
    s, cell = objects.s, objects.cell
    kind, _, name = key.partition("/")
    if kind == "Part" and s.part is not None and cell is not None:
        return Target(
            "the part",
            lambda: s.part.T_world_part.copy(),
            lambda T: cell.part_pose._apply(T, source="code"),
            gizmo="Part",
            hold=cell.block_part_clicks,  # its gizmo, not the selection tools, gets the clicks
        )
    if kind == "Robot base" and cell is not None:
        if s.base.mode == "movable" and objects.carrier is not None:
            c = objects.carrier
            return Target(
                "the carrier (and the robot on it)",
                c.values,
                c.set_values,
                note="_Move it with the Jog panel (arrows or Carrier position sliders), then confirm._",
            )
        return Target("the robot base", lambda: s.T_world_base.copy(), cell.set_base, gizmo="Robot base")
    panel = {"Stay-out zone": objects.stayouts, "Reduced-speed zone": objects.slowzones}.get(kind)
    if panel is not None:
        p = panel
        if name not in [z.name for z in p.zones]:
            return None

        def get_zone():
            return next(z.pose for z in p.zones if z.name == name)

        def set_zone(pose) -> None:  # this panel's kind only: the other kind is kept
            p._store([z.model_copy(update={"pose": pose}) if z.name == name else z for z in p.zones])
            p.refresh(name)

        return Target(f"{kind.lower()} {name}", get_zone, set_zone, gizmo=kind, select=lambda: p.select(name))
    if kind == "Carrier body" and objects.carrier is not None:
        ed = objects.carrier.editor
        if name not in [b.name for b in ed.bodies]:
            return None

        def get_body():
            return next(b.pose for b in ed.bodies if b.name == name)

        def set_body(pose) -> None:
            ed._commit(
                None, [b.model_copy(update={"pose": pose}) if b.name == name else b for b in ed.bodies], name
            )

        def select() -> None:
            ed.pick.value = name

        return Target(f"carrier body {name}", get_body, set_body, gizmo="Carrier body", select=select)
    return None
