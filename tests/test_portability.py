"""D2: Linux portability. Windows ignores letter case in paths and accepts backslashes; Linux does
neither. Every asset path in the shipped data must resolve with exact case and use '/'."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = {"urdf", "mesh", "file", "path", "collision_mesh"}
FILE_EXT = (".urdf", ".stl", ".dae", ".obj", ".step", ".stp", ".ply", ".glb", ".yaml", ".json")


def exact_exists(path: Path) -> bool:
    """True when every component of `path` exists with exactly this letter case."""
    cur = ROOT
    for part in path.relative_to(ROOT).parts:
        if part == "..":
            cur = cur.parent
            continue
        if part not in {p.name for p in cur.iterdir()}:
            return False
        cur = cur / part
    return True


def _walk(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in PATH_KEYS and isinstance(v, str) and v.lower().endswith(FILE_EXT):
                yield v
            else:
                yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _urdf_meshes(urdf: Path):
    return re.findall(r'filename="([^"]+)"', urdf.read_text(encoding="utf-8", errors="replace"))


def _refs() -> list[tuple[str, Path, str]]:
    """(source description, base folder the reference resolves against, reference)."""
    out: list[tuple[str, Path, str]] = []
    for spec in sorted((ROOT / "assets" / "robots").glob("*/spec.yaml")):
        data = yaml.safe_load(spec.read_text(encoding="utf-8"))
        out += [(str(spec.relative_to(ROOT)), spec.parent, r) for r in _walk(data)]
    for tool in sorted((ROOT / "tools").glob("*.yaml")):
        data = yaml.safe_load(tool.read_text(encoding="utf-8"))
        out += [(str(tool.relative_to(ROOT)), ROOT, r) for r in _walk(data)]
    for proj in sorted((ROOT / "examples").glob("*/project.json")):
        data = json.loads(proj.read_text(encoding="utf-8"))
        out += [(str(proj.relative_to(ROOT)), ROOT, r) for r in _walk(data)]
    for urdf in sorted((ROOT / "assets").rglob("*.urdf")):
        out += [(str(urdf.relative_to(ROOT)), urdf.parent, r) for r in _urdf_meshes(urdf)]
    return out


REFS = _refs()


def test_found_references():
    assert len(REFS) > 100


@pytest.mark.parametrize("src", sorted({r[0] for r in REFS}))
def test_asset_paths_resolve_with_exact_case(src):
    bad = []
    for s, base, ref in REFS:
        if s != src:
            continue
        if "\\" in ref:
            bad.append(f"{ref!r}: backslash")
            continue
        if ref.startswith(("package://", "http")):
            continue
        ref = ref.removeprefix("file://")
        target = Path(ref) if Path(ref).is_absolute() else base / ref
        try:
            target = target.resolve().relative_to(ROOT.resolve())
        except ValueError:
            bad.append(f"{ref!r}: outside the repository")
            continue
        if not exact_exists(ROOT / target):
            bad.append(f"{ref!r}: not found with exact case")
    assert not bad, f"{src}: " + "; ".join(bad)


def test_shell_launchers_have_matching_names():
    assert (ROOT / "scripts" / "run_app.bat").is_file()
    sh = ROOT / "scripts" / "run_app.sh"
    assert sh.is_file()
    assert "\r" not in sh.read_bytes().decode("utf-8"), "run_app.sh must use LF line endings"


def test_pdf_browser_lookup_names_linux_browsers(monkeypatch, tmp_path):
    """Linux browser names are searched (via shutil.which) and a missing browser says so."""
    import shutil

    from core import report

    seen = []
    monkeypatch.setattr(shutil, "which", lambda n: seen.append(n))
    monkeypatch.setattr(Path, "is_file", lambda self: False)  # Windows CI has Edge installed
    html = tmp_path / "a.html"
    html.write_text("<p>x</p>")
    with pytest.raises(report.PdfError, match="No Edge / Chrome"):
        report.html_to_pdf(html, tmp_path / "a.pdf")
    for name in ("chromium", "chromium-browser", "google-chrome", "microsoft-edge"):
        assert name in seen
