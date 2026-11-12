"""One-time: low-detail collision meshes for a robot whose URDF uses its CAD (visual) meshes for
collision too (ABB GoFa / PoWa, 2026-11-11).

    python scripts/make_collision_meshes.py gofa10 gofa12 powa30 [--wrap-mm 5] [--faces 1500]

Each ``<collision>`` mesh of ``assets/robots/<name>/<name>.urdf`` is wrapped (core.collision_mesh,
a closed shell ``--wrap-mm`` outside the surface, reduced to about ``--faces`` triangles) and
written to ``meshes/collision/<file>.stl`` in the same units and frame; the URDF's collision
filename is pointed at it. The visual meshes are not touched. Runs again safely (it always
starts from the visual mesh of the same name).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.collision_mesh import Geom, wrap  # noqa: E402

_COLL = re.compile(r"(<collision>.*?<mesh filename=\")([^\"]+)(\"(?: scale=\"([^\"]+)\")?)", re.S)


def convert(name: str, wrap_mm: float, faces: int) -> None:
    folder = ROOT / "assets" / "robots" / name
    urdf = folder / f"{name}.urdf"
    text = urdf.read_text(encoding="utf-8")
    out_dir = folder / "meshes" / "collision"
    out_dir.mkdir(parents=True, exist_ok=True)
    done: dict[str, str] = {}

    def sub(m: re.Match) -> str:
        rel = m.group(2)
        src_rel = rel.replace("meshes/collision/", "meshes/")  # rerun: from the visual mesh again
        if src_rel not in done:
            scale = float((m.group(4) or "1").split()[0])
            mesh = trimesh.load(folder / src_rel, force="mesh")
            g = Geom("wrap", faces, wrap_mm)
            low = wrap(mesh, g.wrap_mm / 1000 / scale, g.faces)
            dst = out_dir / Path(src_rel).name
            low.export(dst)
            print(f"{name}: {Path(src_rel).name}: {len(mesh.faces)} -> {len(low.faces)} triangles")
            done[src_rel] = f"meshes/collision/{Path(src_rel).name}"
        return m.group(1) + done[src_rel] + m.group(3)

    text = _COLL.sub(sub, text)
    urdf.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("robots", nargs="+")
    ap.add_argument("--wrap-mm", type=float, default=5.0)
    ap.add_argument("--faces", type=int, default=1500)
    a = ap.parse_args()
    for name in a.robots:
        convert(name, a.wrap_mm, a.faces)


if __name__ == "__main__":
    main()
