"""Carrier drive in the left panel's Jog (2026-11-11): *Show carrier gizmo* and a 2-axis
on-screen joystick.

Viser has no joystick widget, so the pad is HTML (``gui.add_html``): its inline
``onpointerdown`` handler (inline handlers run; ``<script>`` in innerHTML would not) captures
the pointer and, while held, writes ``joy:x,y,n`` (x right, y up, both -1..1; n counts up so
React always sees a change) into the *Joystick* text input every 100 ms, through the native
value setter + an ``input`` event, which viser sends to the server like typing. Release
sends 0,0. The server drives at ``RATE`` Hz while a command is fresh (``STALE_S``: a lost
release stops the carrier).

Axes: mobile *Drive and turn* = stick up drives forward along the heading, sideways turns;
*Strafe* = moves along / across the heading (holonomic). Rail = sideways moves s, up / down
the lift (if any). Gantry = x / y of the gantry frame. Rail / gantry stop at their travel; a
mobile base has no limits.
"""

from __future__ import annotations

import re
import threading
import time

import numpy as np
import viser

from ui.jog_gizmos import CarrierGizmo

RATE = 10.0
STALE_S = 0.5
DEAD = 0.05
MODES = ("Drive and turn", "Strafe")
PREFIX = "joy:"
_VALUE = re.compile(r"^joy:(-?[0-9.]+),(-?[0-9.]+)(?:,\d+)?$")

PAD_JS = (
    "(function(el,ev){var r=el.getBoundingClientRect(),cx=r.left+r.width/2,cy=r.top+r.height/2,"
    "R=r.width/2-16,k=el.querySelector('.rso-knob'),inp=[...document.querySelectorAll('input')]"
    ".find(function(i){return i.value.indexOf('joy:')==0});if(!inp)return;"
    "el.setPointerCapture(ev.pointerId);var x=0,y=0,n=0;"
    "var set=Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set;"
    "function send(){n++;set.call(inp,'joy:'+x.toFixed(2)+','+y.toFixed(2)+','+n);"
    "inp.dispatchEvent(new Event('input',{bubbles:true}))}"
    "function mv(e){var dx=e.clientX-cx,dy=e.clientY-cy,d=Math.hypot(dx,dy);"
    "if(d>R){dx*=R/d;dy*=R/d}x=dx/R;y=-dy/R;k.style.transform='translate('+dx+'px,'+dy+'px)'}"
    "mv(ev);send();var t=setInterval(send,100);"
    "function up(){clearInterval(t);x=0;y=0;k.style.transform='';send();"
    "el.removeEventListener('pointermove',mv);el.removeEventListener('pointerup',up);"
    "el.removeEventListener('pointercancel',up)}"
    "el.addEventListener('pointermove',mv);el.addEventListener('pointerup',up);"
    "el.addEventListener('pointercancel',up)})(this,event)"
)


def pad_html(size: int = 150) -> str:
    knob = 44
    off = (size - knob) // 2
    return (
        '<div style="display:flex;justify-content:center;padding:6px 0">'
        f'<div class="rso-joy" onpointerdown="{PAD_JS}" style="position:relative;width:{size}px;'
        f"height:{size}px;border-radius:50%;background:rgba(128,128,128,.18);"
        'border:2px solid rgba(128,128,128,.5);touch-action:none;user-select:none;cursor:grab">'
        f'<div style="position:absolute;left:50%;top:4px;transform:translateX(-50%);font-size:10px;'
        'opacity:.6;pointer-events:none">▲</div>'
        f'<div class="rso-knob" style="position:absolute;left:{off}px;top:{off}px;width:{knob}px;'
        f"height:{knob}px;border-radius:50%;background:#3b82f6;opacity:.85;pointer-events:none;"
        'box-shadow:0 1px 4px rgba(0,0,0,.4)"></div></div></div>'
    )


def parse(value: str) -> tuple[float, float] | None:
    m = _VALUE.match(value.strip())
    if m is None:
        return None
    x, y = (float(np.clip(float(g), -1.0, 1.0)) for g in m.groups())
    return x, y


def step_values(
    kind: str, mode: str, v: dict, x: float, y: float, speed: float, turn: float, dt: float
) -> dict:
    """Carrier DOFs after ``dt`` s of stick (x, y) at ``speed`` m/s / ``turn`` rad/s."""
    x, y = (0.0 if abs(a) < DEAD else a for a in (x, y))
    out = dict(v)
    if kind == "mobile":
        yaw = v.get("yaw", 0.0)
        c, s = np.cos(yaw), np.sin(yaw)
        if mode == "Strafe":
            fwd, left = y * speed * dt, -x * speed * dt
            out["x"] = v["x"] + c * fwd - s * left
            out["y"] = v["y"] + s * fwd + c * left
        else:
            fwd = y * speed * dt
            out["x"], out["y"] = v["x"] + c * fwd, v["y"] + s * fwd
            out["yaw"] = yaw - x * turn * dt
    elif kind == "rail":
        out["s"] = v["s"] + x * speed * dt
        if "lift" in v:
            out["lift"] = v["lift"] + y * speed * dt
    elif kind == "gantry":
        out["x"], out["y"] = v["x"] + x * speed * dt, v["y"] + y * speed * dt
    return out


class DrivePanel:
    """Carrier gizmo toggle + joystick (in ObjectsPanel.drive_slot)."""

    def __init__(self, server: viser.ViserServer, carrier, container) -> None:
        self.carrier, self.s = carrier, carrier.s
        self.gizmo = CarrierGizmo(server, carrier)
        self.cmd = (0.0, 0.0)
        self.t_cmd = 0.0
        self._thread = None
        self._lock = threading.Lock()
        gui = server.gui
        with container:
            self.show_gizmo = gui.add_checkbox(
                "Show carrier gizmo", False, hint="drag the mobile base / carriage in 3D along its axes"
            )
            self.pad = gui.add_html(pad_html())
            self.value = gui.add_text("Joystick", PREFIX + "0.00,0.00,0", hint="set by the pad above")
            self.mode = gui.add_dropdown(
                "Mobile drive", MODES, hint="stick sideways turns, or moves sideways"
            )
            self.speed = gui.add_number("Max speed (mm/s)", 300.0, min=1.0, max=5000.0, step=50.0)
            self.turn = gui.add_number("Max turn (deg/s)", 30.0, min=1.0, max=360.0, step=5.0)
            self.info = gui.add_markdown("")
        self.show_gizmo.on_update(lambda _e: self.gizmo.set_visible(self.show_gizmo.value))
        self.value.on_update(lambda _e: self.command_text(self.value.value))
        self.refresh()

    def refresh(self) -> None:
        """Hint for the current carrier (called after carrier changes)."""
        b = self.s.base
        if b.mode != "movable":
            text = "_No movable carrier: the joystick and gizmo drive a rail, gantry or mobile base._"
        else:
            how = {
                "mobile": "up = forward, sideways = turn (or strafe); no limits",
                "rail": "sideways = along the rail, up = lift",
                "gantry": "sideways = x, up = y (gantry frame)",
            }.get(b.carrier, "")
            text = f"_{b.carrier}: {how}._"
        if self.info.content != text:  # called on every base move: send only changes
            self.info.content = text
        self.gizmo.place()

    def command_text(self, text: str) -> None:
        xy = parse(text)
        if xy is not None:
            self.command(*xy)

    def command(self, x: float, y: float) -> None:
        with self._lock:
            self.cmd, self.t_cmd = (x, y), time.monotonic()
            idle = self._thread is None or not self._thread.is_alive()
            moving = abs(x) >= DEAD or abs(y) >= DEAD
            if idle and moving:
                self._thread = threading.Thread(target=self._loop, daemon=True)
                self._thread.start()

    def step(self, dt: float) -> bool:
        """One drive step (also used by tests). False when there is nothing to drive."""
        b = self.s.base
        if b.mode != "movable" or not self.carrier._dof:
            return False
        x, y = self.cmd
        c = self.carrier._carrier()
        v = self.carrier.current(c)
        new = step_values(
            c.kind,
            self.mode.value,
            v,
            x,
            y,
            self.speed.value / 1000,
            np.radians(self.turn.value),
            dt,
        )
        if new == v:
            return False
        self.carrier.set_values(new)
        return True

    def _loop(self) -> None:
        last = time.monotonic()
        while True:
            time.sleep(1.0 / RATE)
            now = time.monotonic()
            with self._lock:
                x, y = self.cmd
                fresh = now - self.t_cmd < STALE_S
            if not fresh or (abs(x) < DEAD and abs(y) < DEAD):
                return
            try:
                self.step(min(now - last, 0.3))
            except Exception:  # a carrier being rebuilt mid-drive: stop, the next push restarts
                return
            last = now
