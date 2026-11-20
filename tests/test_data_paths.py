"""core/paths.py: source mode equals the old expressions; frozen mode uses per-user folders."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from core import data_paths
from opt import reachability as reach

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("RSO_DATA", "RSO_WORKSPACE", "LOCALAPPDATA", "XDG_DATA_HOME"):
        monkeypatch.delenv(k, raising=False)


def test_source_mode_matches_old_expressions(clean_env):
    assert data_paths.resources() == REPO
    assert data_paths.data_dir() == REPO
    assert data_paths.workspace_dir() == REPO / "workspace"
    assert data_paths.last_project_file() == REPO / "workspace" / "last_project.txt"
    assert data_paths.cache_dir("reachability") == REPO / "cache" / "reachability"
    assert data_paths.cache_dir("collision") == REPO / "cache" / "collision"
    assert data_paths.tools_dir() == REPO / "tools"
    assert data_paths.asset_bases() == [REPO]
    # the module constants the app uses
    assert reach.CACHE_DIR == Path(reach.__file__).resolve().parents[1] / "cache" / "reachability"
    from core import collision_mesh, tool

    assert collision_mesh.CACHE == REPO / "cache" / "collision"
    assert tool.TOOLS_DIR == REPO / "tools" and tool.REPO == REPO


def test_workspace_override(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("RSO_WORKSPACE", str(tmp_path / "ws"))
    assert data_paths.workspace_dir() == tmp_path / "ws"
    assert data_paths.last_project_file() == tmp_path / "ws" / "last_project.txt"


def test_unknown_cache_kind(clean_env):
    with pytest.raises(ValueError, match="unknown cache kind"):
        data_paths.cache_dir("nope")


def test_frozen_windows(clean_env, monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert data_paths.resources() == bundle
    d = data_paths.data_dir()
    assert d == tmp_path / "local" / "RobotCellOptimizer" and d.is_dir()
    assert data_paths.workspace_dir() == d / "workspace"
    assert data_paths.cache_dir("numba") == d / "cache" / "numba"
    assert data_paths.asset_bases() == [d, bundle]


def test_frozen_linux(clean_env, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert data_paths.resources() == Path(sys.executable).resolve().parent
    assert data_paths.data_dir() == tmp_path / "xdg" / "robot-cell-optimizer"
    monkeypatch.delenv("XDG_DATA_HOME")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    assert data_paths.data_dir() == tmp_path / "home" / ".local" / "share" / "robot-cell-optimizer"


@pytest.mark.parametrize("frozen", [False, True])
def test_rso_data_override(clean_env, monkeypatch, tmp_path, frozen):
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    monkeypatch.setenv("RSO_DATA", str(tmp_path / "mine"))
    assert data_paths.data_dir() == tmp_path / "mine" and (tmp_path / "mine").is_dir()
    assert data_paths.workspace_dir() == tmp_path / "mine" / "workspace"


def test_numba_cache_only_when_frozen_or_overridden(clean_env, monkeypatch, tmp_path):
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    data_paths.configure_numba_cache()
    import os

    assert "NUMBA_CACHE_DIR" not in os.environ  # source checkout: numba default
    monkeypatch.setenv("RSO_DATA", str(tmp_path))
    data_paths.configure_numba_cache()
    assert Path(os.environ["NUMBA_CACHE_DIR"]) == tmp_path / "cache" / "numba"
    monkeypatch.delenv("NUMBA_CACHE_DIR")


def test_find_cached_reads_data_then_bundle(clean_env, monkeypatch, tmp_path):
    bundle, data = tmp_path / "bundle", tmp_path / "data"
    (bundle / "cache" / "reachability").mkdir(parents=True)
    (bundle / "cache" / "reachability" / "a.npz").write_bytes(b"seed")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setenv("RSO_DATA", str(data))
    assert data_paths.find_cached("reachability", "a.npz") == bundle / "cache" / "reachability" / "a.npz"
    assert data_paths.find_cached("reachability", "missing.npz") is None
    (data / "cache" / "reachability").mkdir(parents=True)
    (data / "cache" / "reachability" / "a.npz").write_bytes(b"user")
    assert data_paths.find_cached("reachability", "a.npz") == data / "cache" / "reachability" / "a.npz"


def test_reachmap_reads_seed_and_writes_only_to_data(clean_env, monkeypatch, tmp_path):
    bundle, data = tmp_path / "bundle", tmp_path / "data"
    seed = bundle / "cache" / "reachability"
    seed.mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setenv("RSO_DATA", str(data))
    monkeypatch.setattr(reach.ReachMap, "cache_key", classmethod(lambda cls, r, t, s: "k1"))
    arrs = dict(
        r0=0.1,
        z0=-1.0,
        res=0.05,
        dirs=np.zeros((2, 3)),
        reach=np.ones((2, 2, 2), bool),
        manip=np.zeros((2, 2, 2)),
        max_manip=0.5,
    )
    np.savez_compressed(seed / "k1.npz", **arrs)
    monkeypatch.setattr(reach, "CACHE_DIR", data_paths.cache_dir("reachability"))
    m = reach.ReachMap.load_or_build(None, np.eye(4), cache_dir=reach.CACHE_DIR)
    assert m.res == 0.05 and m.max_manip == 0.5
    assert not (data / "cache" / "reachability").exists()  # nothing written
    # a miss builds, then writes to data only
    monkeypatch.setattr(reach.ReachMap, "cache_key", classmethod(lambda cls, r, t, s: "k2"))
    monkeypatch.setattr(
        reach.ReachMap,
        "build",
        classmethod(lambda cls, *a, **k: cls(0.2, 0.0, 0.1, arrs["dirs"], arrs["reach"], arrs["manip"], 1.0)),
    )
    reach.ReachMap.load_or_build(None, np.eye(4), cache_dir=reach.CACHE_DIR)
    assert (data / "cache" / "reachability" / "k2.npz").is_file() and not (seed / "k2.npz").exists()
