"""Move mode (v7 round 2, T11): Alt+click an object (or Objects -> Move…), drag its gizmo,
then **Confirm** or **Cancel**.

viser 1.1.1 reports no right-clicks, so the trigger is Alt+click (``on_click(modifier="alt")``)
on the part and the stay-out zones, plus the Objects panel's *Move…* for every object
(robot base / carrier, carrier bodies too). ``start`` saves the object's pose, turns that
object's gizmo on (every other one off, through Objects -> *Drag gizmo on*) and shows a
small floating panel near the click. ``confirm`` keeps the new pose: it is already in the
session, so a confirmed setup step shows "changed", as after any gizmo drag. ``cancel`` puts
the saved pose back exactly. Esc is not available (viser sends no key events): only the button.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

import viser


@dataclass
class Target:
    """A movable object: ``get`` returns its pose (any value, deep-copied), ``set`` puts one
    back; ``gizmo`` is its Objects -> *Drag gizmo on* entry ("" = none, e.g. a carrier
    moved with the Jog panel); ``select`` makes it the panel's current item first."""

    label: str
    get: Callable[[], object]
    set: Callable[[object], None]
    gizmo: str = ""
    select: Callable[[], None] | None = None
    note: str = ""
    hold: Callable[[bool], None] | None = None  # True while moving (e.g. the part ignores clicks)


class MoveMode:
    WIDTH = 240.0

    def __init__(
        self, server: viser.ViserServer, gizmo_choice=None, on_commit: Callable[[], None] | None = None
    ):
        self.server = server
        self.gizmo_choice = gizmo_choice  # Objects -> Drag gizmo on (dropdown handle)
        self.on_commit = on_commit
        self.key: str | None = None
        self.target: Target | None = None
        self._saved = None
        self._prev_gizmo: str | None = None
        self.panel = None
        self.title = self.confirm_btn = self.cancel_btn = None

    @property
    def active(self) -> bool:
        return self.key is not None

    def start(self, key: str, target: Target, event=None) -> None:
        """Begin moving ``target`` (any move in progress is cancelled first)."""
        if self.active:
            self.cancel()
        if target.select is not None:
            target.select()
        self.key, self.target = key, target
        self._saved = copy.deepcopy(target.get())
        if target.hold is not None:
            target.hold(True)
        if self.gizmo_choice is not None:
            self._prev_gizmo = self.gizmo_choice.value
            if target.gizmo and target.gizmo in self.gizmo_choice.options:
                self.gizmo_choice.value = target.gizmo  # this gizmo on, the others off
        self._show(event)

    def confirm(self) -> None:
        if not self.active:
            return
        self._end()
        if self.on_commit is not None:
            self.on_commit()

    def cancel(self) -> None:
        if not self.active:
            return
        self.target.set(copy.deepcopy(self._saved))
        self._end()

    # --- panel ------------------------------------------------------------------------------

    def _show(self, event) -> None:
        gui = self.server.gui
        self._close_panel()
        self.panel = gui.add_panel()
        with self.panel.add_tab("Move"):
            self.title = gui.add_markdown(
                f"**Move {self.target.label}**  \n"
                + (self.target.note or "_Drag its gizmo, then confirm. Cancel puts it back._")
            )
            self.confirm_btn = gui.add_button("Confirm", color="green")
            self.cancel_btn = gui.add_button("Cancel")
        self.confirm_btn.on_click(lambda e: self.confirm())
        self.cancel_btn.on_click(lambda e: self.cancel())
        x, y = 40.0, 80.0
        pos, cam = getattr(event, "screen_pos", None), getattr(getattr(event, "client", None), "camera", None)
        if pos is not None and cam is not None:
            try:  # next to the click (screen_pos is 0..1 of the 3D view)
                x = max(8.0, float(pos[0]) * cam.image_width + 16)
                y = max(8.0, float(pos[1]) * cam.image_height - 20)
            except (TypeError, AttributeError):
                pass
        self.panel.float(x=x, y=y, width=self.WIDTH)

    def _close_panel(self) -> None:
        if self.panel is not None:
            try:
                self.panel.remove()
            except Exception:  # already gone (stopped server)
                pass
        self.panel = None

    def _end(self) -> None:
        self._close_panel()
        if self.target is not None and self.target.hold is not None:
            self.target.hold(False)
        if self.gizmo_choice is not None and self._prev_gizmo is not None:
            self.gizmo_choice.value = self._prev_gizmo
        self.key = self.target = self._saved = self._prev_gizmo = None
