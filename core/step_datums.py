"""Named coordinate systems (datum CSYS) in a STEP file, for placing a mesh tool's TCP or mount.

CAD systems export a coordinate-system feature as an ``AXIS2_PLACEMENT_3D`` entity carrying the
feature's name (``#12=AXIS2_PLACEMENT_3D('TCP',#13,#14,#15);``: origin point, z axis, x
reference direction). This reads every *named* placement straight from the STEP text (no CAD
kernel needed), converts it to metres with the file's length unit (mm, cm, m or inch) and
returns 4x4 poses in the file's frame, which is the frame of the tessellated mesh
(core.tool_import). Placements without a name (the ones every face / edge carries) are skipped.

Limit: the pose is the entity's own; for a datum inside a sub-part of an assembly the part's
placement in the assembly is not applied (fine for single-part files, the usual tool export).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_ENTITY = re.compile(r"#(\d+)\s*=\s*([A-Z0-9_]+)\s*\((.*?)\)\s*;", re.S)
_REF = re.compile(r"#(\d+)")
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?")
MAX_DATUMS = 200


@dataclass
class Datum:
    name: str
    T: np.ndarray  # 4x4 pose in the file frame, metres


def length_unit(text: str) -> float:
    """Metres per file length unit (default mm, the STEP norm for mechanical parts)."""
    t = text.upper()
    if re.search(r"CONVERSION_BASED_UNIT\s*\(\s*'INCH'", t):
        return 0.0254
    if re.search(r"SI_UNIT\s*\(\s*\.MILLI\.\s*,\s*\.METRE\.\s*\)", t):
        return 0.001
    if re.search(r"SI_UNIT\s*\(\s*\.CENTI\.\s*,\s*\.METRE\.\s*\)", t):
        return 0.01
    if re.search(r"SI_UNIT\s*\(\s*\$\s*,\s*\.METRE\.\s*\)", t):
        return 1.0
    return 0.001


def _vec(args: str) -> np.ndarray | None:
    m = re.search(r"\(([^()]*)\)\s*$", args.strip())
    if m is None:
        return None
    vals = [float(x) for x in _NUM.findall(m.group(1))]
    return np.array(vals[:3] + [0.0] * (3 - len(vals[:3])), float) if vals else None


def _name(args: str) -> str:
    m = re.match(r"\s*'((?:[^']|'')*)'", args)
    return m.group(1).replace("''", "'").strip() if m else ""


def read_datums(text: str) -> list[Datum]:
    """Named AXIS2_PLACEMENT_3D poses of a STEP file's text (metres); duplicate names get #id."""
    data = text[text.upper().find("DATA;") :] if "DATA;" in text.upper() else text
    ents = {int(i): (kind, args) for i, kind, args in _ENTITY.findall(data)}
    scale = length_unit(text)
    out: list[Datum] = []
    seen: dict[str, int] = {}
    for i, (kind, args) in sorted(ents.items()):
        if kind != "AXIS2_PLACEMENT_3D":
            continue
        name = _name(args)
        if not name:
            continue
        refs = args.split(",", 1)[1] if "," in args else ""
        parts = [p.strip() for p in refs.split(",")]
        ids = [int(_REF.match(p).group(1)) if _REF.match(p) else None for p in parts[:3]]
        while len(ids) < 3:
            ids.append(None)

        def get(k, want):
            e = ents.get(k) if k is not None else None
            return _vec(e[1]) if e and e[0] == want else None

        p = get(ids[0], "CARTESIAN_POINT")
        if p is None:
            continue
        z = get(ids[1], "DIRECTION")
        x = get(ids[2], "DIRECTION")
        z = np.array([0.0, 0.0, 1.0]) if z is None or np.linalg.norm(z) < 1e-12 else z / np.linalg.norm(z)
        if x is None or np.linalg.norm(np.cross(z, x)) < 1e-9:  # default / parallel ref: any perpendicular
            x = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = x - z * float(x @ z)
        x /= np.linalg.norm(x)
        T = np.eye(4)
        T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
        T[:3, 3] = p * scale
        if name in seen:
            name = f"{name} #{i}"
        seen[name] = i
        out.append(Datum(name, T))
        if len(out) >= MAX_DATUMS:
            break
    return out
