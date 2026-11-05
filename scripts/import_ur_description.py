"""One-time import of a UR robot from Universal Robots' ROS 2 description package.

Converts the xacro to a plain URDF with relative mesh paths and copies meshes plus
both licenses into assets/robots/<ur_type>/. The resulting folder is committed, so
this only reruns when updating the model.

    git clone --depth 1 https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git <src>
    python scripts/import_ur_description.py <src> ur20
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="clone of Universal_Robots_ROS2_Description")
    ap.add_argument("ur_type", help="e.g. ur20")
    args = ap.parse_args()

    src: Path = args.src.resolve()
    if not (src / "config" / args.ur_type).is_dir():
        sys.exit(f"{src} has no config/{args.ur_type}")
    out = ROOT / "assets" / "robots" / args.ur_type
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        # xacro's $(find pkg) needs a ROS install; substitute the clone path instead.
        work = Path(tmp) / "ur_description"
        shutil.copytree(src / "urdf", work / "urdf")
        shutil.copytree(src / "config", work / "config")
        pkg = src.as_posix()
        for f in (work / "urdf").rglob("*.xacro"):
            text = f.read_text(encoding="utf-8")
            text = text.replace("$(find ur_description)", pkg)
            text = re.sub(r"\$\(find \$\{visual_params\['mesh'\]\['package'\]\}\)", pkg, text)
            text = text.replace(f"{pkg}/urdf", (work / "urdf").as_posix())
            f.write_text(text, encoding="utf-8")

        import xacro  # pip package; only needed for this one-time import

        doc = xacro.process_file(
            str(work / "urdf" / "ur.urdf.xacro"),
            mappings={"ur_type": args.ur_type, "name": args.ur_type},
        )
        urdf = doc.toprettyxml(indent="  ")

    # package://ur_description/meshes/<type>/visual/x.dae (or file:///...) -> meshes/visual/x.dae
    urdf, n = re.subn(
        r'filename="(?:package|file)://[^"]*?/meshes/' + re.escape(args.ur_type) + r'/([^"]+)"',
        r'filename="meshes/\1"',
        urdf,
    )
    if n == 0 or "file://" in urdf or "package://" in urdf:
        sys.exit("mesh path rewrite failed; check the generated URDF")

    (out / f"{args.ur_type}.urdf").write_text(urdf, encoding="utf-8")
    shutil.copytree(src / "meshes" / args.ur_type, out / "meshes", dirs_exist_ok=True)
    shutil.copy(src / "LICENSE", out / "LICENSE_ur_description_BSD3.txt")
    print(f"wrote {out} ({n} mesh references)")


if __name__ == "__main__":
    main()
