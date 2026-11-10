"""Manufacturable tool output (v8 T17): standard tube sizes, bend angle steps and a STEP file.

- ``TUBE_OD_MM``: common round tube outer diameters (metric). The tool optimizer can use one as
  the extension's capsule (radius = OD / 2) instead of the tool's own radius.
- ``snap_values``: bend angles (theta, theta2, roll, pose rx / ry / rz) to multiples of a step
  (5 or 15 degrees), applied *inside* the optimizer to every candidate, so the result is a
  design a tube bender can make (rounded after the search it could stop reaching points).
- ``tool_step``: a STEP file (CadQuery / OpenCascade) of the extension as round tube legs
  (solid, or hollow with ``wall_mm``) with a ball at each bend, and an ISO 9409-1 flange plate
  at the mount (UR: ISO 9409-1-50-4-M6 = 4 x M6 holes on a 50 mm circle, 31.5 mm centring
  bore, plate 63 mm). The head is not included (it comes from its own CAD / URDF meshes). In
  the flange frame, millimetres.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TUBE_OD_MM = (12.0, 16.0, 20.0, 25.0, 30.0, 32.0, 40.0, 50.0, 60.0, 80.0)
BEND_STEPS = ("free", "5°", "15°")
ANGLE_KEYS = ("theta_deg", "theta2_deg", "roll_deg", "rx_deg", "ry_deg", "rz_deg")

# ISO 9409-1 flange patterns: name -> (pitch circle, holes, hole diameter, bore, plate diameter) in mm
FLANGES = {
    "ISO 9409-1-50-4-M6": (50.0, 4, 6.6, 31.5, 63.0),
    "ISO 9409-1-63-4-M6": (63.0, 4, 6.6, 40.0, 80.0),
    "ISO 9409-1-80-6-M8": (80.0, 6, 9.0, 50.0, 100.0),
}
PLATE_MM = 8.0


def step_deg(choice: str) -> float:
    return {"5°": 5.0, "15°": 15.0}.get(choice, 0.0)


def snap_values(v, step: float, ranges=None):
    """ExtensionValues with the bend angles rounded to ``step`` deg (kept inside ``ranges``:
    key -> (lo, hi) in deg, rounding toward the inside when the nearest step lies outside)."""
    if step <= 0:
        return v
    upd = {}
    for k in ANGLE_KEYS:
        a = float(getattr(v, k))
        s = round(a / step) * step
        if ranges is not None and k in ranges:
            lo, hi = ranges[k]
            if s < lo - 1e-9:
                s = np.ceil(lo / step) * step
            if s > hi + 1e-9:
                s = np.floor(hi / step) * step
            if not lo - 1e-9 <= s <= hi + 1e-9:  # no step inside a narrow range: keep the value
                s = a
        upd[k] = float(s)
    return v.model_copy(update=upd)


def tool_step(
    ee,
    values,
    path: Path,
    tube_od_mm: float | None = None,
    wall_mm: float = 0.0,
    flange: str = "ISO 9409-1-50-4-M6",
) -> Path:
    """Write the extension + flange plate as a STEP file; returns the path."""
    import cadquery as cq

    from core.tool import ToolModel

    tm = ToolModel(ee, None)
    r = (tube_od_mm / 2) if tube_od_mm else 1000 * ee.extension.radius
    T_fm = tm.T_flange_mount
    shape = None

    def add(s):
        nonlocal shape
        shape = s if shape is None else shape.fuse(s)

    segs = tm.extension_segments(values)
    for k, (a, b) in enumerate(segs):
        a, b = 1000 * np.asarray(a), 1000 * np.asarray(b)
        d = b - a
        L = float(np.linalg.norm(d))
        u = d / L
        add(cq.Solid.makeCylinder(r, L, cq.Vector(*a), cq.Vector(*u)))
        if k < len(segs) - 1:
            add(cq.Solid.makeSphere(r, cq.Vector(*b), angleDegrees1=-90, angleDegrees2=90))
    if wall_mm > 0 and r - wall_mm > 0.5:  # hollow tube: subtract the bore of every leg
        for a, b in segs:
            a, b = 1000 * np.asarray(a), 1000 * np.asarray(b)
            d = b - a
            L = float(np.linalg.norm(d))
            shape = shape.cut(cq.Solid.makeCylinder(r - wall_mm, L, cq.Vector(*a), cq.Vector(*(d / L))))
    pcd, n, hole, bore, plate = FLANGES[flange]
    z0 = 1000 * T_fm[:3, 3]
    zdir = T_fm[:3, 2]
    disk = cq.Solid.makeCylinder(plate / 2, PLATE_MM, cq.Vector(*(z0 - zdir * PLATE_MM)), cq.Vector(*zdir))
    xdir = T_fm[:3, 0]
    ydir = T_fm[:3, 1]
    for i in range(n):
        ang = 2 * np.pi * i / n + np.pi / n
        c = z0 - zdir * PLATE_MM + (xdir * np.cos(ang) + ydir * np.sin(ang)) * pcd / 2
        disk = disk.cut(cq.Solid.makeCylinder(hole / 2, PLATE_MM, cq.Vector(*c), cq.Vector(*zdir)))
    disk = disk.cut(
        cq.Solid.makeCylinder(bore / 2, PLATE_MM / 2, cq.Vector(*(z0 - zdir * PLATE_MM)), cq.Vector(*zdir))
    )
    add(disk)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(cq.Workplane().add(shape), str(path), exportType="STEP")
    return path
