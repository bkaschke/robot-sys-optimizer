"""Packaging: fast checks that need no build (spec, launcher arguments, offline guard, frozen version)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _launch():
    spec = importlib.util.spec_from_file_location("rso_launch_under_test", ROOT / "packaging" / "launch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_spec_file_compiles():
    src = (ROOT / "packaging" / "rso.spec").read_text(encoding="utf-8")
    compile(src, "rso.spec", "exec")
    assert "module_collection_mode" in src and "offset_wrist_nb" in src


def test_iss_is_per_user_and_keeps_data():
    iss = (ROOT / "packaging" / "RobotCellOptimizer.iss").read_text(encoding="utf-8")
    assert "PrivilegesRequired=lowest" in iss
    assert r"{localappdata}\Programs\RobotCellOptimizer" in iss
    assert "[UninstallDelete]" not in iss


def test_launch_arguments():
    launch = _launch()
    assert launch.parse_args(["--selftest"])[0] == "selftest"
    mode, browser, port, rest = launch.parse_args(["--no-browser", "--port", "9000", "--dark"])
    assert (mode, browser, port, rest) == ("app", False, "9000", ["--port", "9000", "--dark"])
    mode, browser, port, rest = launch.parse_args([])
    assert (mode, browser, port, rest) == ("app", True, "8080", [])
    assert launch.parse_args(["--port=1234"])[2] == "1234"


def test_offline_guard_blocks_non_loopback_and_allows_loopback():
    code = (
        "import os, socket, sys\n"
        "os.environ['RSO_OFFLINE_GUARD']='1'\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        f"sys.argv=['x']\n"
        "import importlib.util\n"
        f"spec=importlib.util.spec_from_file_location('l', {str(ROOT / 'packaging' / 'launch.py')!r})\n"
        "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "try:\n"
        "    socket.create_connection(('203.0.113.5', 80), timeout=1)\n"
        "    print('NOT-BLOCKED')\n"
        "except OSError as e:\n"
        "    print('blocked' if 'offline-guard' in str(e) else 'other:' + str(e))\n"
        "srv = socket.socket(); srv.bind(('127.0.0.1', 0)); srv.listen(1)\n"
        "c = socket.create_connection(('127.0.0.1', srv.getsockname()[1]), timeout=2)\n"
        "print('loopback-ok'); c.close(); srv.close()\n"
    )
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60)
    assert r.stdout.split() == ["blocked", "loopback-ok"], r.stdout + r.stderr
    assert "[offline-guard] blocked" in r.stderr


def test_provenance_reads_version_file_when_frozen(tmp_path, monkeypatch):
    from core import data_paths, provenance

    (tmp_path / "_version.py").write_text(
        'VERSION = "v1.2-3-gabc"\nCOMMIT = "abc1234"\nDATE = "2026-11-14 10:00"\n'
    )
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    provenance.git_version.cache_clear()
    try:
        assert data_paths.resources() == tmp_path
        assert provenance.frozen_version()["VERSION"] == "v1.2-3-gabc"
        assert provenance.git_version() == "abc1234"
    finally:
        provenance.git_version.cache_clear()


def test_provenance_frozen_without_file_is_unknown(tmp_path, monkeypatch):
    from core import provenance

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    provenance.git_version.cache_clear()
    try:
        assert provenance.git_version() == "unknown"
    finally:
        provenance.git_version.cache_clear()


def test_step_import_message_when_cadquery_missing(monkeypatch, tmp_path):
    from core import tool_import

    monkeypatch.setitem(sys.modules, "cadquery", None)  # import raises ImportError
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    with pytest.raises(tool_import.ToolImportError, match="not included in this build"):
        tool_import.step_mesh(tmp_path / "x.step")


def test_build_script_helpers(tmp_path):
    spec = importlib.util.spec_from_file_location("build_exe_under_test", ROOT / "scripts" / "build_exe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.write_version(tmp_path, {"VERSION": "v1", "COMMIT": "abc", "DATE": "d"})
    text = (tmp_path / "_version.py").read_text()
    assert 'COMMIT = "abc"' in text
    env = mod.build_env()
    assert env["PYTHONNOUSERSITE"] == "1" and env["PATH"].split(";" if sys.platform == "win32" else ":")[
        0
    ].endswith("bin")


def _portable():
    path = ROOT / "scripts" / "make_portable.py"
    spec = importlib.util.spec_from_file_location("make_portable_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# environment.yml name -> import name; build / dev-only entries map to None
_ENV_IMPORT = {"pyyaml": "yaml", "pillow": "PIL", "python-dateutil": "dateutil", "fonttools": "fontTools"}
_NOT_RUNTIME = {
    "python", "libblas", "pyinstaller", "pyinstaller-hooks-contrib", "pytest", "ruff", "markdown-it-py",
    "pip", "xacro",
}  # fmt: skip


def test_portable_check_covers_environment_yml():
    import re

    import yaml

    deps = yaml.safe_load((ROOT / "environment.yml").read_text(encoding="utf-8"))["dependencies"]
    pip = [d for e in deps if isinstance(e, dict) for d in e["pip"]]
    names = [d for d in deps if isinstance(d, str)] + pip
    wanted = set()
    for d in names:
        n = re.split(r"[=<>! ]", d, maxsplit=1)[0]
        if n not in _NOT_RUNTIME:
            wanted.add(_ENV_IMPORT.get(n, n))
    missing = wanted - set(_portable().RUNTIME_MODULES)
    assert not missing, f"add to make_portable.RUNTIME_MODULES (or _NOT_RUNTIME here): {sorted(missing)}"


def test_portable_check_env_passes_and_names_missing():
    mod = _portable()
    mod.check_env(sys.prefix, modules=("json",))
    with pytest.raises(SystemExit, match="no_such_module_rso"):
        mod.check_env(sys.prefix, modules=("json", "no_such_module_rso"))
