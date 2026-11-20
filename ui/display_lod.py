"""Display level of detail and a small cache for the part mesh (R1c).

A part of several hundred thousand triangles is one 9 MB message and stalls the browser. Above
``DISPLAY_FACES`` the viewer draws a simplified copy (``TARGET_FACES``, quadric decimation from
``core.collision_mesh``). Only the drawing changes: planning, collision and surface picking use
the full mesh. Picking casts the click ray (sent by the browser as origin and direction, not as
a face index) against the full mesh, so face ids always refer to the full mesh.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import trimesh

DISPLAY_FACES = 300_000  # above this many triangles the drawing is simplified
TARGET_FACES = 200_000  # about this many triangles in the simplified copy
KEEP = 2  # meshes kept (the current part and the one before it)


def file_key(path: Path, units: str) -> tuple:
    """Cache key: changes when the file is replaced or edited, or the units change."""
    p = Path(path)
    st = p.stat()  # a missing file raises FileNotFoundError, as loading would
    return (str(p.resolve()), units, st.st_size, st.st_mtime_ns)


class MeshCache:
    """The last ``keep`` entries by key (least recently used goes first)."""

    def __init__(self, keep: int = KEEP):
        self.keep = keep
        self._d: OrderedDict = OrderedDict()

    def get(self, key, make):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        value = make()
        self._d[key] = value
        while len(self._d) > self.keep:
            self._d.popitem(last=False)
        return value

    def clear(self) -> None:
        self._d.clear()


def display_copy(
    mesh: trimesh.Trimesh,
    limit: int = DISPLAY_FACES,
    target: int = TARGET_FACES,
    background: bool = False,
):
    """(mesh to draw, True when it is simplified). The mesh itself when it has at most ``limit``
    triangles, or when the simplifier is missing (a convex hull would hide the part). A simplified
    copy's largest deviation from ``mesh`` is measured once and remembered (``display_deviation``).
    ``background``: measure it in a thread (about 2 s for 500k triangles) so the GUI is not held;
    until it is ready ``display_deviation`` returns a conservative estimate, and the callbacks in
    ``on_deviation_ready`` run when the measured value differs materially from it."""
    if len(mesh.faces) <= limit:
        return mesh, False
    from core.collision_mesh import _reduce

    out, how = _reduce(mesh, target)
    if how != "reduce":
        return mesh, False
    fp = fingerprint(mesh)
    if not background:
        _DEVIATION[fp] = _measure(mesh, out)
        return out, True
    if fp not in _DEVIATION and fp not in _PENDING:
        _FALLBACK[fp] = fallback_deviation(mesh)
        while len(_FALLBACK) > 8:
            _FALLBACK.popitem(last=False)
        t = threading.Thread(target=_measure_bg, args=(fp, mesh, out), name="display-deviation", daemon=True)
        _PENDING[fp] = t
        t.start()
    return out, True


FALLBACK_FRACTION = 0.005  # of the bbox diagonal ...
FALLBACK_MAX = 0.005  # ... but at most 5 mm
MATERIAL_CHANGE = 2.5e-4  # a lift change below 0.25 mm is not worth a redraw
_FALLBACK: OrderedDict = OrderedDict()  # fingerprint -> estimate used until the measurement is in
_PENDING: dict = {}  # fingerprint -> measuring thread
on_deviation_ready: list[Callable[[], None]] = []  # app.py: redraw the overlays


def fallback_deviation(mesh: trimesh.Trimesh) -> float:
    """Conservative deviation (m) before the measurement: 0.5 % of the bbox diagonal, at most 5 mm."""
    import numpy as np

    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    return min(FALLBACK_MAX, FALLBACK_FRACTION * diag)


def _measure_bg(fp: tuple, full: trimesh.Trimesh, low: trimesh.Trimesh) -> None:
    try:
        d = _measure(full, low)
    except Exception:  # noqa: BLE001 - keep the conservative estimate
        _PENDING.pop(fp, None)
        return
    old = _FALLBACK.get(fp, 0.0)
    _DEVIATION[fp] = d
    _PENDING.pop(fp, None)
    if abs(d - old) > MATERIAL_CHANGE:
        for cb in list(on_deviation_ready):
            try:
                cb()
            except Exception:  # noqa: BLE001 - a redraw failure must not kill the thread
                pass


def wait_deviation(timeout: float = 30.0) -> None:
    """Block until background measurements finish (tests)."""
    for t in list(_PENDING.values()):
        t.join(timeout)


_DEVIATION: OrderedDict = OrderedDict()  # fingerprint of a full mesh -> deviation of its display copy (m)
DEV_VERTICES = 80_000  # vertices checked against the copy (a stride through all of them)


def fingerprint(mesh: trimesh.Trimesh) -> tuple:
    """Cheap identity of a mesh's geometry (part_mesh() returns copies, so ids do not work)."""
    v = mesh.vertices
    return (len(mesh.faces), len(v), float(v.sum()), float(abs(v).sum()))


def _measure(full: trimesh.Trimesh, low: trimesh.Trimesh) -> float:
    """Close lower bound of the largest distance from ``full`` to ``low`` (m): the exact
    closest-point distance of an even stride of at most DEV_VERTICES vertices (all of them
    would take several seconds for 500k triangles)."""
    import numpy as np

    from core.collision_mesh import deviation

    v = np.asarray(full.vertices, float)
    v = v[:: max(1, len(v) // DEV_VERTICES)]
    probe = trimesh.Trimesh(vertices=v, faces=np.zeros((1, 3), int), process=False)
    d = float(deviation(probe, low, samples=0))
    while len(_DEVIATION) > 8:
        _DEVIATION.popitem(last=False)
    return d


def display_deviation(mesh: trimesh.Trimesh) -> float:
    """Largest deviation (m) of the simplified display copy of ``mesh`` from it; 0 when the
    mesh is drawn as it is (or was never prepared by ``display_copy``)."""
    fp = fingerprint(mesh)
    d = _DEVIATION.get(fp)
    return d if d is not None else _FALLBACK.get(fp, 0.0)


def overlay_lift(mesh: trimesh.Trimesh, default: float) -> float:
    """How far to lift a selection / colour patch drawn from the full ``mesh`` so it does not
    sink under the simplified surface: ``default + deviation`` (the selection's 0.5 mm becomes
    deviation + 0.5 mm; the heatmap and area patches keep their order above it). ``default``
    for a mesh that is drawn as it is."""
    return default + display_deviation(mesh)


def note(full_faces: int, shown_faces: int) -> str:
    return (
        f"Display simplified: {full_faces / 1000:,.0f}k → {shown_faces / 1000:,.0f}k triangles "
        "(planning, collision and picking use the full mesh)"
    )
