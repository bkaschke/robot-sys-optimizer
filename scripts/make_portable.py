"""Build a portable zip for a PC without conda or internet access.

    python scripts/make_portable.py [--env rso] [--out dist]

The zip holds the committed repo files (``git archive HEAD``) plus the conda environment packed
with conda-pack under ``env/``. On the target PC: unzip anywhere writable and double-click
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
    ap.add_argument("--env", default="rso", help="conda environment to pack (default rso)")
    ap.add_argument("--out", type=Path, default=ROOT / "dist", help="output folder (default dist/)")
    a = ap.parse_args(argv)

    if git("status", "--porcelain", "--untracked-files=no"):
        print("WARNING: uncommitted changes are not included (the bundle uses the last commit).")
    sha = git("rev-parse", "--short", "HEAD")
    a.out.mkdir(parents=True, exist_ok=True)
    final = a.out / f"{TOP}-portable-{sha}.zip"

    with tempfile.TemporaryDirectory() as tmp:
        env_zip, repo_zip = Path(tmp) / "env.zip", Path(tmp) / "repo.zip"
        print(f"Packing conda env '{a.env}' (a few minutes)...", flush=True)
        subprocess.run(
            [find_conda_pack(), "-n", a.env, "-o", str(env_zip), "--format", "zip", "--quiet"], check=True
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
        tmp_final.replace(final)
    print(f"Done: {final} ({final.stat().st_size / 2**20:.0f} MB; {n_repo} repo files, {n_env} env files)")
    return final


if __name__ == "__main__":
    main()
