"""R1d: playback sends fewer messages (status text throttled, one atomic batch per tick), the
unreached label cap has a summary label, and overlays lift above a simplified display copy."""

import time
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
import viser

from tests.helpers import confirm_all
from ui import display_lod, simulate


class Counting:
    """Records every assignment to ``content``."""

    def __init__(self):
        self.log = []

    @property
    def content(self):
        return self.log[-1] if self.log else ""

    @content.setter
    def content(self, v):
        self.log.append(v)


@pytest.fixture(scope="module")
def sim():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8806, verbose=False)
    st = build_app(server)
    assert st["project"].load_example("rail_long_part")
    confirm_all(st["stations"])
    st["stations"].start(block=True, quick=True)
    st["sim"].build(block=True)
    assert st["sim"].plan is not None and len(st["sim"].plan.samples) > 40
    yield st["sim"]
    server.stop()


def _play(sim, idx, throttle=True):
    q_seen = []
    real = sim.show_q
    sim.show_q = lambda q: (q_seen.append(np.array(q)), real(q))[1]
    sim.status = Counting()
    sim._status_t = 0.0
    sim._playing.set()
    try:
        for i in idx:
            sim.go_to(int(i))
    finally:
        sim._playing.clear()
        sim.show_q = real
    return q_seen, sim.status.log


def test_status_is_throttled_and_motion_unchanged(sim, monkeypatch):
    n = len(sim.plan.samples)
    idx = np.linspace(0, n - 1, 41).astype(int)[1:]
    monkeypatch.setattr(simulate, "STATUS_PERIOD", 0.0)  # old behaviour: every tick
    q_old, log_old = _play(sim, idx)
    monkeypatch.setattr(simulate, "STATUS_PERIOD", 0.25)
    t0 = time.monotonic()
    q_new, log_new = _play(sim, idx)
    dt = time.monotonic() - t0
    assert len(log_old) == 40
    assert 1 <= len(log_new) <= int(dt / 0.25) + 1
    assert len(q_new) == len(q_old) and all(np.array_equal(a, b) for a, b in zip(q_new, q_old, strict=True))
    sim.go_to(int(idx[-1]))  # playback stopped: the final state always shows
    assert sim.status.log[-1] == log_old[-1]


def test_tick_is_one_atomic_batch(sim):
    depth = []
    real = sim.server.atomic

    class Spy:
        def __enter__(self):
            depth.append(1)
            self.c = real()
            return self.c.__enter__()

        def __exit__(self, *a):
            return self.c.__exit__(*a)

    sim.server.atomic = lambda: Spy()
    try:
        _play(sim, [5, 9])
        n = len(depth)
        sim.go_to(3)  # not playing: no batch
    finally:
        del sim.server.atomic
    assert n == 2 and len(depth) == 2


def test_label_cap_and_summary():
    from ui.unreached_view import MAX_LABELS, draw_unreached

    assert MAX_LABELS <= 50
    labels = {}
    scene = SimpleNamespace(
        add_point_cloud=lambda *a, **k: None,
        add_line_segments=lambda *a, **k: None,
        add_label=lambda name, text, **k: labels.setdefault(name, text),
    )
    n = MAX_LABELS + 25
    ids = list(range(n))
    P, N = np.zeros((n, 3)), np.tile((0, 0, 1.0), (n, 1))
    out = draw_unreached(SimpleNamespace(scene=scene), "/r", ids, P, N, ids, 0.004, {})
    assert len(labels) == MAX_LABELS + 1
    assert labels["/r/unreached_label_more"] == "+25 more unreached"
    assert "unreached_label_more" in out


def test_overlay_lift_on_simplified_and_small_parts(monkeypatch):
    from ui.colour_mesh import lifted_patch, show_patch
    from ui.targets_draw import SEL_OFFSET, chunk_arrays

    big = trimesh.creation.icosphere(subdivisions=5, radius=0.2)
    small = trimesh.creation.icosphere(subdivisions=2, radius=0.2)
    monkeypatch.setattr(display_lod, "_DEVIATION", display_lod.OrderedDict())
    shown, simp = display_lod.display_copy(big, limit=3000, target=1000)
    assert simp
    dev = display_lod.display_deviation(big.copy())  # a copy, as part_mesh() returns
    assert dev > 5e-4 and abs(dev - display_lod._measure(big, shown)) < 1e-12
    faces = np.arange(50)
    V, _ = chunk_arrays(big, faces)
    used = np.unique(big.faces[faces])
    lift = np.linalg.norm(V - big.vertices[used], axis=1)
    assert np.allclose(lift, dev + SEL_OFFSET, atol=1e-6)
    sent = {}
    cache = SimpleNamespace(show=lambda key, build: build())
    scene = SimpleNamespace(add_glb=lambda name, blob: sent.setdefault("n", name))
    seen = []
    monkeypatch.setattr(
        "ui.colour_mesh.lifted_patch", lambda m, f, o: (seen.append(o), lifted_patch(m, f, o))[1]
    )
    show_patch(scene, cache, "/p", big, faces, np.full((50, 3), 200), 1.1e-3)
    assert seen == [pytest.approx(dev + 1.1e-3)]
    # a small part (drawn as it is): the default lifts
    assert display_lod.display_copy(small)[1] is False
    V, _ = chunk_arrays(small, faces)
    lift = np.linalg.norm(V - small.vertices[np.unique(small.faces[faces])], axis=1)
    assert np.allclose(lift, SEL_OFFSET, atol=1e-6)
    show_patch(scene, cache, "/p", small, faces, np.full((50, 3), 200), 1.1e-3)
    assert seen[-1] == 1.1e-3
