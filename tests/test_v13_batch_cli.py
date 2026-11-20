"""v13 #8: scripts/batch_run.py."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_batch_job import _make_batch, fake  # noqa: F401 (fixture)
from tests.test_v13_batch import RAIL, make_parts

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "batch_run.py"


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=600,
    )


def test_bad_arguments(tmp_path):
    from scripts.batch_run import main

    assert main([str(tmp_path), str(tmp_path), str(tmp_path / "o")]) == 2  # no project.json
    assert main([str(RAIL), str(tmp_path / "missing"), str(tmp_path / "o")]) == 2
    make_parts(tmp_path / "parts")
    assert main([str(RAIL), str(tmp_path / "parts"), str(tmp_path / "o"), "--parts", "zzz"]) == 2


@pytest.mark.slow
def test_runs_from_a_subprocess(tmp_path):
    make_parts(tmp_path / "parts", broken=True)
    p = _run(RAIL, tmp_path / "parts", tmp_path / "out", "--parts", "rail_a,bad_tcp", "--workers", "2")
    assert p.returncode == 1, p.stderr  # bad_tcp failed
    assert "rail_a: 25/25 reached" in p.stdout and "bad_tcp: failed" in p.stdout
    assert (tmp_path / "out" / "rail_a" / "project.json").is_file()
    assert (tmp_path / "out" / "summary.json").is_file() and not (tmp_path / "out" / "rail_b").exists()


def test_a_batch_folder_runs_its_queue(tmp_path, fake, capsys):  # noqa: F811
    """One positional argument with a batch.json: the queue's pending jobs (ui.batch_job)."""
    from scripts.batch_run import main

    _m, bdir = _make_batch(tmp_path, n=1)
    assert main([str(bdir)]) == 0
    assert "1/1 job(s) done" in capsys.readouterr().out and fake["seen"]
