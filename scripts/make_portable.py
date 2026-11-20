"""Build a portable zip for a PC without conda or internet access.

    python scripts/make_portable.py [--prefix ENV_FOLDER] [--out dist]

The zip holds the committed repo files (``git archive HEAD``) plus the conda environment packed
with conda-pack under ``env/``, and the reachability maps already built on this PC
(``cache/reachability/*.npz``; keyed by robot + tool geometry, so the target PC reuses them).
On the target PC: unzip anywhere writable and double-click
``Start Robot Optimizer.bat``. The first start runs ``conda-unpack`` once to fix the paths
inside the environment; nothing is installed and no admin rights or network are needed.

Needs conda-pack in the Miniforge base env (``conda install -n base -c conda-forge conda-pack``).
Uncommitted changes are NOT included: commit first.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP = "robot-sys-optimizer"
START = """@echo off
rem Portable Robot Cell Optimizer: starts the app with the bundled Python in env\\.
rem Options are passed on, e.g. "Start Robot Optimizer.bat" --dark --port 8090
call "%~dp0scripts\\run_app.bat" %*
pause
"""


# Import names of the runtime packages in environment.yml (tests/test_packaging.py keeps this in sync).
# Checked in the env being packed before the zip is written: a missing package would otherwise only
# show up on the offline target PC.
RUNTIME_MODULES = (
    "pinocchio", "coal", "numpy", "scipy", "numba", "rtree", "yaml", "PIL", "lxml", "pyparsing",
    "dateutil", "fontTools", "viser", "yourdfpy", "pydantic", "trimesh", "ortools", "cma", "shapely",
    "optuna", "fast_simplification", "cadquery", "collada", "manifold3d",
)  # fmt: skip


def env_python(prefix: str | Path) -> Path:
    p = Path(prefix)
    for c in (p / "python.exe", p / "bin" / "python"):
        if c.exists():
            return c
    sys.exit(f"no Python found in the env to pack: {p}")


def check_env(prefix: str | Path, modules=RUNTIME_MODULES) -> None:
    """Import every runtime package with the env's own Python (isolated, no user site-packages,
    which conda-pack would not include). Exits with the missing packages named."""
    code = (
        "import importlib, sys\n"
        "bad = []\n"
        "for m in sys.argv[1:]:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except Exception as e:\n"
        "        bad.append(f'{m}: {type(e).__name__}: {e}')\n"
        "print('\\n'.join(bad))\n"
        "sys.exit(1 if bad else 0)\n"
    )
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    print(f"Checking {len(modules)} packages import in {prefix} ...", flush=True)
    r = subprocess.run(
        [str(env_python(prefix)), "-I", "-c", code, *modules], capture_output=True, text=True, env=env
    )
    if r.returncode != 0:
        lines = r.stdout.strip() or r.stderr.strip()[-2000:]
        sys.exit(
            "The env to pack is missing packages (the zip was not written):\n"
            f"{lines}\nUpdate it: conda env update -n rso -f environment.yml --prune"
        )


def find_conda_pack() -> str:
    for c in (
        os.environ.get("CONDA_PACK"),
        shutil.which("conda-pack"),
        str(Path.home() / "miniforge3" / "Scripts" / "conda-pack.exe"),
    ):
        if c and Path(c).exists():
            return c
    sys.exit("conda-pack not found: conda install -n base -c conda-forge conda-pack (or set CONDA_PACK)")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def copy_entries(src: zipfile.ZipFile, dst: zipfile.ZipFile, prefix: str) -> int:
    n = 0
    for info in src.infolist():
        if info.is_dir():
            continue
        out = zipfile.ZipInfo(prefix + info.filename, date_time=info.date_time)
        out.external_attr = info.external_attr
        out.compress_type = zipfile.ZIP_DEFLATED
        with src.open(info) as fi, dst.open(out, "w", force_zip64=True) as fo:
            shutil.copyfileobj(fi, fo, 1 << 20)
        n += 1
    return n


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--prefix", default=sys.prefix, help="conda env folder to pack (default: the running one)"
    )
    ap.add_argument("--out", type=Path, default=ROOT / "dist", help="output folder (default dist/)")
    ap.add_argument(
        "--no-maps", dest="maps", action="store_false", help="leave out the cached reachability maps (cache/)"
    )
    a = ap.parse_args(argv)

    if git("status", "--porcelain", "--untracked-files=no"):
        print("WARNING: uncommitted changes are not included (the bundle uses the last commit).")
    sha = git("rev-parse", "--short", "HEAD")
    check_env(a.prefix)
    a.out.mkdir(parents=True, exist_ok=True)
    final = a.out / f"{TOP}-portable-{sha}.zip"

    with tempfile.TemporaryDirectory() as tmp:
        env_zip, repo_zip = Path(tmp) / "env.zip", Path(tmp) / "repo.zip"
        print(f"Packing conda env {a.prefix} (a few minutes)...", flush=True)
        subprocess.run(
            [find_conda_pack(), "-p", a.prefix, "-o", str(env_zip), "--format", "zip", "--quiet"], check=True
        )
        subprocess.run(["git", "archive", "--format=zip", "-o", str(repo_zip), "HEAD"], cwd=ROOT, check=True)
        print(f"Writing {final} ...", flush=True)
        tmp_final = final.with_suffix(".part")
        with zipfile.ZipFile(tmp_final, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
            with zipfile.ZipFile(repo_zip) as r:
                n_repo = copy_entries(r, z, f"{TOP}/")
            with zipfile.ZipFile(env_zip) as e:
                n_env = copy_entries(e, z, f"{TOP}/env/")
            z.writestr(f"{TOP}/Start Robot Optimizer.bat", START.replace("\n", "\r\n"))
            maps = sorted((ROOT / "cache" / "reachability").glob("*.npz")) if a.maps else []
            for m in maps:  # prebuilt reachability maps: no ~1 min build on the target PC
                z.write(m, f"{TOP}/cache/reachability/{m.name}")
        tmp_final.replace(final)
    print(
        f"Done: {final} ({final.stat().st_size / 2**20:.0f} MB; {n_repo} repo files, {n_env} env files, "
        f"{len(maps)} reachability maps)"
    )
    return final


if __name__ == "__main__":
    main()
