"""Write the golden test parts to assets/parts/*.stl. Run from the repo root."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.golden_parts import GOLDEN_PARTS  # noqa: E402

if __name__ == "__main__":
    out = ROOT / "assets" / "parts"
    out.mkdir(parents=True, exist_ok=True)
    for name, build in GOLDEN_PARTS.items():
        build().export(out / f"{name}.stl")
        print(f"wrote {out / name}.stl")
