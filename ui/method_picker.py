"""Method dropdown with its description underneath (v8 T4).

    self.refine_method = MethodPicker(gui, "station_refine", "Refinement")
    key = self.refine_method.key          # "cmaes", "tpe", ...

The dropdown is ``.dropdown`` (``.value`` = the label, saved in panel settings like any other
dropdown); the muted line below shows the chosen method's one-line description and follows
every change. Hidden / shown together with ``visible``.
"""

from __future__ import annotations

from opt.methods import METHODS, by_name, options


class MethodPicker:
    def __init__(self, gui, slot: str, label: str, hint: str | None = None):
        self.slot = slot
        self.dropdown = gui.add_dropdown(
            label, options(slot), initial_value=METHODS[slot][0].name, hint=hint or "see the line below"
        )
        self.note = gui.add_markdown(self._text())
        self.dropdown.on_update(lambda _e: self.refresh())

    def _text(self) -> str:
        return f"_{by_name(self.slot, self.dropdown.value).short}._"

    def refresh(self) -> None:
        self.note.content = self._text()

    @property
    def value(self) -> str:
        return self.dropdown.value

    @value.setter
    def value(self, v: str) -> None:
        self.dropdown.value = by_name(self.slot, v).name
        self.refresh()

    @property
    def key(self) -> str:
        return by_name(self.slot, self.dropdown.value).key

    @property
    def visible(self) -> bool:
        return self.dropdown.visible

    @visible.setter
    def visible(self, on: bool) -> None:
        self.dropdown.visible = self.note.visible = bool(on)
