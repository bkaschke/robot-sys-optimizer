"""scripts/release_notes.py fills the release-notes template from a dist folder; app_version; --version."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from core import provenance, version

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dist(tmp_path, zip_too=True):
    names = ["RobotCellOptimizer-v1.2.3-setup.exe"] + (
        ["RobotCellOptimizer-v1.2.3-win64.zip"] if zip_too else []
    )
    for n in names:
        (tmp_path / n).write_bytes(b"x")
    (tmp_path / "SHA256SUMS.txt").write_text(
        "".join(f"{i:064x}  {n}\n" for i, n in enumerate(names, 1)), encoding="utf-8"
    )
    return tmp_path


def test_notes_filled_from_dist(tmp_path):
    rn = _load("rn", "scripts/release_notes.py")
    text = rn.fill(_dist(tmp_path))
    assert "{{" not in text
    assert "# Robot Cell Optimizer v1.2.3" in text
    assert "RobotCellOptimizer-v1.2.3-setup.exe" in text and "RobotCellOptimizer-v1.2.3-win64.zip" in text
    assert f"{1:064x}  RobotCellOptimizer-v1.2.3-setup.exe" in text
    assert "Get-FileHash" in text and "More info" in text and text.rstrip().endswith("## Changes")
    assert "9.9" in rn.fill(tmp_path, "9.9")


def test_notes_without_zip_drop_its_bullet(tmp_path):
    rn = _load("rn2", "scripts/release_notes.py")
    text = rn.fill(_dist(tmp_path, zip_too=False))
    assert "win64.zip" not in text and "{{" not in text


def test_changes_are_appended_only_when_given(tmp_path, monkeypatch):
    rn = _load("rn4", "scripts/release_notes.py")
    plain = rn.fill(_dist(tmp_path))
    # GitHub appends its own list: without --changes the text is what it always was
    assert rn.fill(tmp_path, None, None) == plain == rn.fill(tmp_path, changes="  \n")
    template = (ROOT / "packaging" / "RELEASE_NOTES.md.in").read_text(encoding="utf-8")
    assert plain.endswith(template[template.index("## Checksums") :].split("```")[-1])
    assert "github" not in template.lower() and "gitlab" not in template.lower()  # host-neutral wording
    text = rn.fill(tmp_path, changes="\n- one (abc1234)\n- two (def5678)\n\n")
    assert text.startswith(plain.rstrip("\n"))
    assert text.endswith("## Changes\n\n- one (abc1234)\n- two (def5678)\n")
    # the command line: --changes FILE, and no flag = the plain text
    (tmp_path / "c.md").write_text("- one (abc1234)\n", encoding="utf-8")
    for extra, want in (
        ([], plain),
        (["--changes", str(tmp_path / "c.md")], plain.rstrip("\n") + "\n\n- one (abc1234)\n"),
    ):
        out = tmp_path / "notes.md"
        monkeypatch.setattr(sys, "argv", ["release_notes.py", str(tmp_path), *extra, "--out", str(out)])
        rn.main()
        assert out.read_text(encoding="utf-8") == want


def test_notes_fail_clearly(tmp_path):
    rn = _load("rn3", "scripts/release_notes.py")
    with pytest.raises(SystemExit, match="setup.exe"):
        rn.fill(tmp_path)
    (tmp_path / "RobotCellOptimizer-v1-setup.exe").write_bytes(b"x")
    with pytest.raises(SystemExit, match="SHA256SUMS.txt"):
        rn.fill(tmp_path)


def test_app_version_source_mode(monkeypatch):
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    version.app_version.cache_clear()
    try:
        v = version.app_version()
        assert isinstance(v, str) and v
        monkeypatch.setattr(
            version.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no git"))
        )
        version.app_version.cache_clear()
        assert version.app_version() == version.UNKNOWN_SOURCE
    finally:
        version.app_version.cache_clear()


def test_app_version_frozen(tmp_path, monkeypatch):
    from core import data_paths

    (tmp_path / "_version.py").write_text(
        'VERSION = "v0.9.0"\nCOMMIT = "16de6e1"\nDATE = "2026-11-15 10:30"\n', encoding="utf-8"
    )
    monkeypatch.setattr(data_paths, "resources", lambda: tmp_path)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    version.app_version.cache_clear()
    try:
        assert version.app_version() == "v0.9.0 (16de6e1, 2026-11-15)"
        # an untagged build's version is the commit hash: it is not repeated
        (tmp_path / "_version.py").write_text(
            'VERSION = "55369ce-dirty"\nCOMMIT = "55369ce"\nDATE = "2026-11-15 10:30"\n', encoding="utf-8"
        )
        version.app_version.cache_clear()
        assert version.app_version() == "55369ce-dirty (2026-11-15)"
        (tmp_path / "_version.py").unlink()
        version.app_version.cache_clear()
        assert version.app_version() == "unknown"
    finally:
        version.app_version.cache_clear()
        provenance.git_version.cache_clear()


def test_numeric_version():
    assert version.numeric_version("v1.2.3-4-gabc") == "1.2.3.4"
    assert version.numeric_version("v0.9") == "0.9.0.0"
    # a bare commit hash is not a version: "98595ef" once became 98595.0.0.0, which Windows rejects
    assert version.numeric_version("16de6e1") == "0.0.0.0"
    assert version.numeric_version("98595ef") == "0.0.0.0"
    assert version.numeric_version("98595ef-dirty") == "0.0.0.0"
    assert version.numeric_version("v1.2.3-4-gabc123-dirty") == "1.2.3.4"
    assert version.numeric_version("v70000.1") == "65535.1.0.0"
    assert version.numeric_version("zzz") == "0.0.0.0"


def test_launch_version_flag():
    launch = _load("launch_v", "packaging/launch.py")
    assert launch.parse_args(["--version"])[0] == "version"
    assert launch.parse_args(["--port", "1", "--version"])[0] == "version"


def test_iss_takes_version_from_defines():
    iss = (ROOT / "packaging" / "RobotCellOptimizer.iss").read_text(encoding="utf-8")
    assert "AppVersion={#AppVersion}" in iss and "VersionInfoVersion={#NumericVersion}" in iss
    assert "NumericVersion=" in (ROOT / "scripts" / "build_exe.py").read_text(encoding="utf-8")
