"""Optimize -> Tool geometry -> *Tool extension*: a second copy of the Tool tab's Extension
settings (type, ranges, capsule radius, rough placement) next to the tool optimizer.

Both copies edit the same tool: an edit here goes through ``ToolSetupTab._commit`` (the whole
tool re-validated) and refreshes the Tool tab; every Tool tab refresh refreshes this copy
(``ToolSetupTab.refresh_listeners``). Rough-placement sliders sync one value at a time so a
slider being dragged is never rebuilt under the mouse.
"""

from __future__ import annotations

from ui import tool_extension


class ExtensionMirror:
    def __init__(self, server, tool_tab, container):
        self.server, self.tab = server, tool_tab
        self._busy = False
        self._dyn: list = []
        self._nominal: dict = {}
        gui = server.gui
        with container:
            self.note = gui.add_markdown(
                "_The same settings as **3 Tool → Extension**: the ranges the optimizer searches "
                "and the tool's rough placement. Edits here change the tool._"
            )
            tool_extension.build(gui, self)
        for w in tool_extension.widgets(self):
            w.on_update(self._edit)
        tool_tab.refresh_listeners.append(self.refresh)
        tool_tab.nominal_listeners.append(self._from_tab)
        self.refresh()

    def _edit(self, event=None) -> None:
        if self._busy or self.tab.s.tool is None:
            return
        if self.tab._commit(lambda d: tool_extension.commit(self, d), event):
            self.tab._after(full=True)  # the Tool tab refreshes, and with it this copy
        else:
            self.refresh()

    def _on_nominal(self, key: str, scale: float):
        def handler(event) -> None:
            if self._busy:
                return
            val = self._nominal[key].value * scale
            if self.tab._commit(lambda d: d["extension"]["nominal"].update({key: val}), event):
                self.tab.set_nominal_slider(key, val)
                self.tab._after()

        return handler

    def _from_tab(self, key: str, val: float) -> None:
        """A Tool tab rough-placement slider moved: move the matching slider here."""
        w = self._nominal.get(key)
        if w is None:
            return
        self._busy = True
        try:
            w.value = val * tool_extension.FIELDS[key][1]
        finally:
            self._busy = False

    def refresh(self) -> None:
        self._busy = True
        try:
            for h in self._dyn:
                h.remove()
            self._dyn, self._nominal = [], {}
            ee = self.tab.s.tool
            for w in (*tool_extension.widgets(self), self.ext_note, self.legacy_note):
                w.visible = ee is not None
            if ee is None:
                self.note.content = "_No tool loaded (**3 Tool**)._"
                return
            self.note.content = (
                "_The same settings as **3 Tool → Extension**: the ranges the optimizer searches "
                "and the tool's rough placement. Edits here change the tool._"
            )
            tool_extension.refresh(self, ee.extension, self._on_nominal)
        finally:
            self._busy = False
