"""R1a: the Targets tab re-sends only what changed (ui.targets_draw).

Messages are recorded with the same patch as the R0 measurement script: viser's
WebsockMessageHandler.queue_message keeps every message the app queues (no browser needed).
"""

import collections

import msgspec
import numpy as np
import pytest
import viser
from viser.infra import _infra

from core.feasibility import PointResult
from ui.targets_draw import MAX_CHUNKS, PALETTE, SEL_OFFSET, STATUS_COLORS, UNCHECKED, point_colors

REC: list = []


@pytest.fixture(scope="module")
def app():
    from app import build_app

    orig = _infra.WebsockMessageHandler.queue_message

    def rec(self, message):
        REC.append(message)
        return orig(self, message)

    _infra.WebsockMessageHandler.queue_message = rec
    server = viser.ViserServer(host="127.0.0.1", port=8871, verbose=False)
    try:
        st = build_app(server)
        st["cell"].part_choice.value = "curved_panel"  # 19,760 faces
        tab = st["targets"]
        m = tab.mesh()
        rng = np.random.default_rng(0)
        fi = rng.choice(np.flatnonzero(m.face_normals[:, 2] > 0.5), 200, replace=False)
        tab._add(m.triangles_center[fi], m.face_normals[fi])
        yield st
    finally:
        _infra.WebsockMessageHandler.queue_message = orig
        server.stop()


def sent(fn) -> tuple[collections.Counter, collections.Counter]:
    """(messages, bytes) per message type queued while ``fn`` runs."""
    REC.clear()
    fn()
    n, b = collections.Counter(), collections.Counter()
    for msg in REC:
        bufs: list = []
        inner = msgspec.msgpack.encode(msg.as_serializable_dict(bufs))
        n[type(msg).__name__] += 1
        b[type(msg).__name__] += len(inner) + sum(x.nbytes for x in bufs)
    REC.clear()
    return n, b


HEAVY = ("MeshMessage", "PointCloudMessage", "LineSegmentsMessage", "LabelMessage", "GlbMessage")


def chunk_faces(tab) -> list[np.ndarray]:
    return [f for f, _ in tab.sel_layer.chunks]


def test_selection_add_sends_only_the_new_faces(app):
    tab = app["targets"]
    tab.select(np.zeros(0, int), "set")
    tab.select(np.arange(0, 6000), "add")
    n1, b1 = sent(lambda: tab.select(np.arange(6000, 7000), "add"))
    n2, b2 = sent(lambda: tab.select(np.arange(7000, 8000), "add"))
    assert n1["MeshMessage"] == n2["MeshMessage"] == 1
    assert "RemoveSceneNodeMessage" not in n1 and "PointCloudMessage" not in n1
    # bytes follow the added faces (1000 each time), not the selection (7000, 8000 faces)
    assert abs(b1["MeshMessage"] - b2["MeshMessage"]) < 0.1 * b1["MeshMessage"]
    _, full = sent(lambda: tab.sel_layer._rebuild(tab.mesh(), tab.selection))  # what the old code sent
    assert b2["MeshMessage"] < 0.2 * full["MeshMessage"]
    tab.select(np.zeros(0, int), "set")


def test_chunks_always_show_exactly_the_selection(app):
    tab = app["targets"]
    m = tab.mesh()
    rng = np.random.default_rng(3)
    tab.select(np.zeros(0, int), "set")
    for step in range(40):
        how = ("add", "add", "add", "remove", "set")[step % 5] if step % 13 else "set"
        faces = rng.choice(len(m.faces), int(rng.integers(0, 400)), replace=False)
        if step % 13 == 0:
            faces = faces[:0]  # clear
        tab.select(faces, how)
        parts = chunk_faces(tab)
        allf = np.concatenate(parts) if parts else np.zeros(0, int)
        assert len(allf) == len(np.unique(allf))  # chunks do not overlap
        assert np.array_equal(np.sort(allf), tab.selection)
        assert len(parts) <= MAX_CHUNKS
        assert all(not h._impl.removed for _, h in tab.sel_layer.chunks)
    assert tab._handles["sel"] is tab.sel_layer.frame


def test_many_adds_merge_into_one_chunk_and_look_is_unchanged(app):
    tab = app["targets"]
    m = tab.mesh()
    tab.select(np.zeros(0, int), "set")
    for k in range(MAX_CHUNKS + 1):
        tab.select(np.arange(k * 50, (k + 1) * 50), "add")
    assert len(tab.sel_layer.chunks) == 1  # the 17th add merged everything
    faces, h = tab.sel_layer.chunks[0]
    assert np.array_equal(faces, np.arange((MAX_CHUNKS + 1) * 50))
    # same triangles as the old single sub-mesh, lifted 0.5 mm along the normals
    tri = np.asarray(h.vertices)[np.asarray(h.faces)]
    want = m.vertices[m.faces[faces]] + SEL_OFFSET * m.vertex_normals[m.faces[faces]]
    assert np.allclose(tri, want, atol=1e-6)
    assert tuple(h.color) == (60, 200, 120) and h.opacity == 0.6 and h.side == "double"
    tab.select(np.zeros(0, int), "set")
    assert tab.sel_layer.chunks == [] and tab.sel_info.content == "_Nothing selected._"


def test_selection_only_refresh_sends_no_points_and_a_second_refresh_nothing(app):
    tab = app["targets"]
    tab.refresh()
    tab.selection = np.arange(500)  # changed behind the tab's back (as restore_work does)
    n, _ = sent(tab.refresh)
    assert n["MeshMessage"] == 1  # the selection only
    assert not any(n[k] for k in HEAVY[1:])
    n, _ = sent(tab.refresh)
    assert not any(n[k] for k in HEAVY) and "RemoveSceneNodeMessage" not in n
    assert sum(n.values()) <= 2  # nothing, or at most a GUI value echo
    tab.select(np.zeros(0, int), "set")


def test_points_update_in_place(app):
    s, tab = app["session"], app["targets"]
    h = tab._handles["points"]
    ids = [p.id for p in s.targets.points]
    s.feasibility = {i: PointResult(i, "reachable" if i % 2 else "marginal") for i in ids}
    n, _ = sent(lambda: setattr(tab.color_by, "value", "feasibility"))
    assert tab._handles["points"] is h and not any(n[k] for k in HEAVY)
    assert n["SceneNodeUpdateMessage"] == 1  # colours only
    want = np.array([STATUS_COLORS[s.feasibility[i].status] for i in ids])
    assert np.array_equal(np.asarray(h.colors), want)
    tab.color_by.value = "group"
    s.feasibility = None
    tab._draw_points()


def test_unreached_and_heatmap_redraw_only_when_their_inputs_change(app):
    s, tab = app["session"], app["targets"]
    ids = [p.id for p in s.targets.points]
    q = np.zeros(6)
    s.feasibility = {
        i: PointResult(i, "unreachable", "no_ik")
        if k < 3
        else PointResult(i, "reachable", q=q, clearance=0.001 * k)
        for k, i in enumerate(ids)
    }
    tab.selected_results = lambda: {}
    tab.heat.choice.value = "Clearance"
    tab.refresh()
    assert tab.heat._handles and len(tab._handles["unreached"].points) == 3
    n, _ = sent(tab.refresh)
    assert not any(n[k] for k in HEAVY)  # nothing changed: heatmap and markers stay
    s.feasibility[ids[10]] = PointResult(ids[10], "reachable", q=q, clearance=0.2)
    n, _ = sent(tab.refresh)
    assert n["GlbMessage"] >= 1 and not n["LabelMessage"]  # heatmap only (one vertex-coloured GLB)
    s.feasibility[ids[20]] = PointResult(ids[20], "unreachable", "collision")
    n, _ = sent(tab.refresh)
    assert n["LabelMessage"] == 4 and len(tab._handles["unreached"].points) == 4
    tab.heat.choice.value = "Joint-limit margin"  # the dropdown redraws it itself ...
    n, _ = sent(tab.refresh)
    assert not n["GlbMessage"]  # ... so the next refresh does not draw it again
    tab.heat.clear()  # drawn away from the tab (e.g. a draw/clear from another panel)
    tab.refresh()
    assert tab.heat._handles
    tab.heat.choice.value = "Reached / unreached (points)"
    s.feasibility = None
    tab.refresh()
    assert not tab.heat._handles and not any(k.startswith("unreached") for k in tab._handles)


def _old_colors(color_by, ids, G, names, feas, sel):
    """The pre-R1a per-point loop (reference)."""
    if color_by == "feasibility" and feas:
        return np.array([STATUS_COLORS[feas[i].status] if i in feas else UNCHECKED for i in ids])
    if color_by in ("zone", "configuration") and sel:
        labels = sorted({r.config for r in sel.values() if r.config})
        key = (lambda r: r.zone) if color_by == "zone" else (lambda r: labels.index(r.config))
        return np.array(
            [
                PALETTE[key(sel[i]) % len(PALETTE)]
                if i in sel and sel[i].config and sel[i].zone is not None
                else UNCHECKED
                for i in ids
            ]
        )
    return np.array([PALETTE[names.index(g) % len(PALETTE)] if g in names else PALETTE[-1] for g in G])


def test_vectorised_colours_match_the_old_loop():
    rng = np.random.default_rng(1)
    ids = np.arange(300)
    names = [f"g{k}" for k in range(8)]
    G = [names[k] if k < 8 else "other" for k in rng.integers(0, 9, 300)]
    st = ("reachable", "marginal", "unreachable")
    feas = {int(i): PointResult(int(i), st[i % 3]) for i in ids if i % 7}
    sel = {
        int(i): PointResult(
            int(i), "reachable", config=("S+E+W+", "S-E+W-", "")[i % 3], zone=int(i % 9) or None
        )
        for i in ids
        if i % 5
    }
    for by in ("group", "feasibility", "zone", "configuration"):
        a = point_colors(by, ids, G, names, feas, sel)
        assert a.dtype == np.uint8
        assert np.array_equal(a, _old_colors(by, ids, G, names, feas, sel)), by
    assert np.array_equal(
        point_colors("feasibility", ids, G, names, {}, {}), _old_colors("group", ids, G, names, {}, {})
    )
