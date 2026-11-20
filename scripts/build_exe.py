"""Build the Windows executable folder (PyInstaller, onedir) and check it.

    python scripts/build_exe.py [--maps [--map-jobs N]] [--zip] [--installer] [--skip-selftest] [--no-cad]

Run it with the ``rso`` conda env's Python (the tested native libraries get bundled). Steps:
1. env check; PYTHONNOUSERSITE=1 and <env>/Library/bin on PATH;
2. ``_version.py`` (git describe, commit, date) into a stage folder that the spec bundles;
3. ``--maps``: prebuild the reachability maps of the bundled robots into cache/reachability
   (``--map-jobs N`` builds N at once: CI runners; default 1 for a PC short of memory);
4. PyInstaller with packaging/rso.spec -> dist/RobotCellOptimizer/;
5. the built exe's ``--selftest`` (RSO_DATA = temp folder, RSO_OFFLINE_GUARD=1); a failure fails the build;
6. ``--zip`` -> dist/RobotCellOptimizer-<version>-win64.zip; ``--installer`` -> Inno Setup (iscc) setup exe;
7. dist/SHA256SUMS.txt and a summary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
WORK = DIST / ".work"
APP = "RobotCellOptimizer"


def fail(msg: str) -> None:
    print(f"build_exe: {msg}", file=sys.stderr)
    sys.exit(1)


def check_env(allow_any: bool) -> None:
    name = Path(sys.prefix).name
    if name != "rso" and not allow_any:
        fail(f"run this with the 'rso' env's python (this is '{sys.prefix}'); --any-env skips the check")
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        fail("PyInstaller is missing: conda install -n rso -c conda-forge pyinstaller")


def build_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"  # the user site-packages must not leak into the bundle
    libbin = Path(sys.prefix) / "Library" / "bin"  # native DLLs of the conda env
    env["PATH"] = os.pathsep.join(p for p in (str(libbin), str(Path(sys.prefix)), env.get("PATH", "")) if p)
    return env


def git(*args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def version_info() -> dict[str, str]:
    desc = git("describe", "--tags", "--always", "--dirty") or "0.0.0-nogit"
    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    if git("status", "--porcelain", "--untracked-files=no"):
        commit += "+dirty"
    return {"VERSION": desc, "COMMIT": commit, "DATE": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}


def write_version(stage: Path, info: dict[str, str]) -> None:
    stage.mkdir(parents=True, exist_ok=True)
    text = "# written by scripts/build_exe.py\n" + "".join(f'{k} = "{v}"\n' for k, v in info.items())
    (stage / "_version.py").write_text(text, encoding="utf-8")


def _build_map(name: str) -> tuple[str, float, str]:
    """One robot's flange map (in a worker process with --map-jobs); (name, seconds, skip reason)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import numpy as np

    from core.robot import Robot
    from opt.reachability import DEFAULT_SETTINGS, ReachMap

    t0 = time.monotonic()
    try:
        ReachMap.load_or_build(Robot.from_registry(name), np.eye(4), DEFAULT_SETTINGS)
    except NotImplementedError as err:
        return name, 0.0, str(err)
    return name, time.monotonic() - t0, ""


def prebuild_maps(jobs: int = 1) -> int:
    """Flange reachability maps (shared by every straight or pose tool, see opt.reachability.map_for_tool)
    for each bundled robot with an analytic IK, into cache/reachability (the spec bundles that folder).
    ``jobs`` > 1 builds that many at once in worker processes (2026-11-19: one after another took 22 min
    of a 38 min GitHub release run)."""
    sys.path.insert(0, str(ROOT))
    from concurrent.futures import ProcessPoolExecutor

    from core.robot import list_robots

    names = list(list_robots())
    if jobs > 1:
        with ProcessPoolExecutor(min(jobs, len(names))) as pool:
            done = list(pool.map(_build_map, names))
    else:
        done = [_build_map(n) for n in names]
    n = 0
    for name, secs, skipped in done:
        if skipped:
            print(f"  map {name}: skipped ({skipped})")
            continue
        n += 1
        print(f"  map {name}: {secs:.1f} s")
    return n


def run_pyinstaller(stage: Path, env: dict[str, str], no_cad: bool) -> Path:
    env = {**env, "RSO_STAGE": str(stage)}
    if no_cad:
        env["RSO_NO_CAD"] = "1"
    out = DIST / APP
    shutil.rmtree(out, ignore_errors=True)
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--distpath",
        str(DIST),
        "--workpath",
        str(WORK / "pyi"),
    ]
    subprocess.run([*cmd, str(ROOT / "packaging" / "rso.spec")], cwd=ROOT, env=env, check=True)
    exe = out / f"{APP}.exe"
    if not exe.is_file():
        fail(f"PyInstaller finished but {exe} is missing")
    return exe


def selftest(exe: Path, env: dict[str, str]) -> bool:
    data = Path(tempfile.mkdtemp(prefix="rso-selftest-"))
    env = {**env, "RSO_DATA": str(data), "RSO_OFFLINE_GUARD": "1"}
    print(f"selftest: {exe} (RSO_DATA={data}, offline guard on)", flush=True)
    try:
        r = subprocess.run([str(exe), "--selftest"], env=env, timeout=1800)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        print("selftest: timed out after 30 min", file=sys.stderr)
        return False
    finally:
        shutil.rmtree(data, ignore_errors=True)


def make_zip(folder: Path, version: str) -> Path:
    path = DIST / f"{APP}-{version}-win64.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(folder.parent).as_posix())
    return path


def find_iscc() -> str | None:
    if found := shutil.which("iscc") or shutil.which("ISCC"):
        return found
    for var in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"):
        base = os.environ.get(var)
        for sub in ("Inno Setup 6", "Programs/Inno Setup 6"):
            if base and (Path(base) / sub / "ISCC.exe").is_file():
                return str(Path(base) / sub / "ISCC.exe")
    return None


def make_installer(version: str) -> Path | None:
    iscc = find_iscc()
    if not iscc:
        print("installer: skipped, Inno Setup (iscc) not found (jrsoftware.org; CI: choco install innosetup)")
        return None
    clean = re.sub(r"[^0-9A-Za-z._-]", "_", version)
    sys.path.insert(0, str(ROOT))
    from core.version import numeric_version

    subprocess.run(
        [
            iscc,
            f"/DAppVersion={clean}",
            f"/DNumericVersion={numeric_version(version)}",
            f"/DSourceDir={DIST / APP}",
            f"/O{DIST}",
            str(ROOT / "packaging" / f"{APP}.iss"),
        ],
        check=True,
    )
    return DIST / f"{APP}-{clean}-setup.exe"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def folder_stats(folder: Path) -> tuple[int, int]:
    files = [f for f in folder.rglob("*") if f.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--maps", action="store_true", help="prebuild the reachability maps into cache/reachability"
    )
    ap.add_argument(
        "--zip", action="store_true", help="zip the folder (dist/RobotCellOptimizer-<version>-win64.zip)"
    )
    ap.add_argument(
        "--map-jobs", type=int, default=1, metavar="N", help="with --maps: build N maps at once (default 1)"
    )
    ap.add_argument("--installer", action="store_true", help="build the Inno Setup installer (needs iscc)")
    ap.add_argument("--skip-selftest", action="store_true", help="do not run the exe's --selftest")
    ap.add_argument(
        "--no-cad", action="store_true", help="leave out CadQuery (about 500 MB smaller; no STEP)"
    )
    ap.add_argument("--any-env", action="store_true", help="allow a Python that is not the 'rso' env")
    a = ap.parse_args()
    t0 = time.monotonic()
    check_env(a.any_env)
    env = build_env()
    info = version_info()
    print(f"version {info['VERSION']} (commit {info['COMMIT']})")
    stage = WORK / "stage"
    write_version(stage, info)
    if a.maps:
        print("prebuilding reachability maps")
        t = time.monotonic()
        n = prebuild_maps(max(1, a.map_jobs))
        print(
            f"  {n} maps in cache/reachability ({time.monotonic() - t:.0f} s, {max(1, a.map_jobs)} at once)"
        )
    exe = run_pyinstaller(stage, env, a.no_cad)
    if not a.skip_selftest and not selftest(exe, env):
        fail("the built exe failed its selftest (output above); no release files were written")
    outputs: list[Path] = []
    if a.zip:
        outputs.append(make_zip(exe.parent, info["VERSION"]))
    if a.installer and (inst := make_installer(info["VERSION"])):
        outputs.append(inst)
    if outputs:
        (DIST / "SHA256SUMS.txt").write_text(
            "".join(f"{sha256(p)}  {p.name}\n" for p in outputs), encoding="utf-8"
        )
    n, size = folder_stats(exe.parent)
    print(f"\n{exe.parent}: {size / 1e6:.0f} MB, {n} files")
    for p in outputs:
        print(f"{p}: {p.stat().st_size / 1e6:.0f} MB")
    print(
        f"selftest: {'skipped' if a.skip_selftest else 'passed'}; total {(time.monotonic() - t0) / 60:.1f} min"  # noqa: E501
    )


if __name__ == "__main__":
    main()
