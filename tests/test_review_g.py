"""Review M4: a collision scene that cannot be built is "unavailable", not "no collision"."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def app():
    import viser

    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8813, verbose=False)
    yield build_app(server)
    server.stop()


@pytest.fixture
def jog(app, monkeypatch):
    cell = app["cell"]
    assert app["project"].load_example("rail_long_part")
    jp = cell.jog_panel
    jp.point.value = jp.point.options[0]
    jp.allow.value = False
    jp.frame.value = "Joint"
    jp.refresh()
    cell.set_q(np.radians([0, -100, 100, -90, -90, 0]))
    yield jp
    jp.allow.value = False


def _break(monkeypatch, jog, msg="missing carrier mesh"):
    import core.collision as cc

    def boom(*a, **k):
        raise FileNotFoundError(msg)

    monkeypatch.setattr(cc, "CollisionScene", boom)
    jog.coll._key = jog.coll._fail = None  # force a scene rebuild
    getattr(jog.coll, "_logged", set()).clear()


def test_unavailable_check_blocks_step_and_warns(app, jog, monkeypatch):
    s = app["session"]
    _break(monkeypatch, jog)
    q0 = np.array(s.q, float)
    jog.update_readout()
    assert "collision check unavailable" in jog.readout.content
    assert "missing carrier mesh" in jog.readout.content
    assert not jog.jog("J3", +1)
    assert np.allclose(s.q, q0)  # the step was blocked
    jog.allow.value = True
    assert jog.jog("J3", +1) and not np.allclose(s.q, q0)  # allowed on request


def test_failure_logged_once_and_recovers(app, jog, monkeypatch):
    import core.collision as cc
    from ui import log_panel

    msgs = []
    monkeypatch.setattr(log_panel, "log", lambda m, level="info", since=None: msgs.append(m))
    real = cc.CollisionScene
    _break(monkeypatch, jog, "broken stay-out file")
    for _ in range(3):
        jog.update_readout()
    assert sum("broken stay-out file" in m for m in msgs) == 1
    monkeypatch.setattr(cc, "CollisionScene", real)
    jog.coll._fail = None  # as after RETRY_S
    jog.update_readout()
    assert "unavailable" not in jog.readout.content
    assert jog.coll.error is None
