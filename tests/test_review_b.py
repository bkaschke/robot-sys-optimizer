"""Review B: a broken check pool is evicted and the run falls back to serial checks."""

import multiprocessing as mp
import os

import numpy as np
import pytest

from opt import parallel


class _Scene:
    T_world_part = None


class _Checker:
    def __init__(self):
        self.scene = _Scene()
        self.T_world_part = np.eye(4)
        self.settings = None

    def set_base(self, T):
        pass

    def _check_point(self, pid, p, n, g, max_poses, clearance):
        return ("ok", pid)


class DiesInWorker:
    """build() kills pool workers (e.g. OOM) but works in the parent."""

    def build(self):
        if mp.parent_process() is not None:
            os._exit(3)
        return _Checker()

    def key(self):
        return "dies-in-worker"


class RaisesInWorker:
    def build(self):
        if mp.parent_process() is not None:
            raise ValueError("scene build exploded: bad mesh")
        return _Checker()

    def key(self):
        return "raises-in-worker"


def _calls(n):
    return [(np.eye(4), "point", i, None, None, None, 1, True) for i in range(n)]


@pytest.fixture(autouse=True)
def _clean():
    yield
    parallel.on_warn = None
    parallel.shutdown_all()


@pytest.mark.timeout(120)
@pytest.mark.parametrize("spec_cls", [DiesInWorker, RaisesInWorker])
def test_broken_pool_falls_back_and_is_replaced(spec_cls):
    msgs = []
    parallel.on_warn = msgs.append
    spec = spec_cls()
    p1 = parallel.pool_for(spec, 2)
    out = p1.run(None, _calls(8))  # before the fix: BrokenProcessPool
    assert out == [("ok", i) for i in range(8)]
    assert msgs and "parallel checks failed" in msgs[0] and "continuing with serial checks" in msgs[0]
    if spec_cls is RaisesInWorker:
        assert "scene build exploded: bad mesh" in msgs[0]  # the real cause is visible
    p2 = parallel.pool_for(spec, 2)  # before the fix: the same broken pool
    assert p2 is not p1
    assert p2.run(None, _calls(8)) == [("ok", i) for i in range(8)]
