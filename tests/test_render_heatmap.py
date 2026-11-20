"""R1b: the part heatmap and the station areas are ONE vertex-coloured GLB node each, cached:
switching back to a picture shown before only toggles visibility (ui.colour_mesh).

Messages are recorded with the same patch as the R0 measurement script: viser's
WebsockMessageHandler.queue_message keeps every message the app queues (no browser needed).
"""

import collections
import io
import json
import struct
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import trimesh
import viser
from viser.infra import _infra

from core.feasibility import PointResult
from ui.colour_mesh import NodeCache, glb, srgb_to_linear_u8, vertex_colours

REC: list = []


@pytest.fixture(scope="module")
def app():
    from app import build_app

    orig = _infra.WebsockMessageHandler.queue_message

    def rec(self, message):
        REC.append(message)
        return orig(self, message)

    _infra.WebsockMessageHandler.queue_message = rec
    server = viser.ViserServer(host="127.0.0.1", port=8873, verbose=False)
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


def _feasibility(s, k_scale=0.001):
    q = np.zeros(6)
    return {
        i: PointResult(i, "reachable", q=q, clearance=k_scale * (k % 60), manipulability=0.1 + 0.01 * k)
        for k, i in enumerate(p.id for p in s.targets.points)
    }


def test_glb_has_vertex_colours_and_a_matt_double_sided_material():
    m = trimesh.creation.icosphere(1)
    C = np.tile([[220, 40, 40]], (len(m.vertices), 1)).astype(np.uint8)
    b = glb(m.vertices, m.faces, C, opacity=0.55)
    magic, ver, total = struct.unpack("<III", b[:12])
    assert magic == 0x46546C67 and ver == 2 and total == len(b) and len(b) % 4 == 0
    n = struct.unpack("<I", b[12:16])[0]
    doc = json.loads(b[20 : 20 + n])
    prim = doc["meshes"][0]["primitives"][0]
    assert set(prim["attributes"]) == {"POSITION", "COLOR_0"}
    mat = doc["materials"][prim["material"]]
    assert mat["doubleSided"] and mat["alphaMode"] == "BLEND"
    assert mat["pbrMetallicRoughness"]["metallicFactor"] == 0.0  # the loader's default is metal
    assert mat["pbrMetallicRoughness"]["baseColorFactor"][3] == pytest.approx(0.55)
    back = trimesh.load(io.BytesIO(b), file_type="glb", force="mesh")  # a valid GLB
    assert len(back.faces) == len(m.faces)
    lin = srgb_to_linear_u8(np.array([[220, 40, 40]]))[0]
    view = doc["bufferViews"][doc["accessors"][prim["attributes"]["COLOR_0"]]["bufferView"]]
    blob = b[20 + n + 8 :]
    col = np.frombuffer(blob[view["byteOffset"] : view["byteOffset"] + view["byteLength"]], np.uint8)
    assert np.all(col.reshape(-1, 4)[:, :3] == lin)  # stored linear (glTF convention)
    assert tuple(srgb_to_linear_u8(np.array([0, 255, 188]))) == (0, 255, 128)


def test_vertex_colour_is_the_mean_of_its_faces():
    F = np.array([[0, 1, 2], [0, 2, 3]])
    c = vertex_colours(np.array([[200, 0, 0], [0, 200, 0]], float), F, 4)
    assert c.tolist() == [[100, 100, 0], [200, 0, 0], [100, 100, 0], [0, 200, 0]]


def test_node_cache_is_bounded_and_shows_one_at_a_time():
    class H:
        def __init__(self):
            self.visible, self._impl = True, SimpleNamespace(removed=False)

        def remove(self):
            self._impl.removed = True

    c, made = NodeCache(2), []

    def build():
        made.append(H())
        return made[-1]

    a = c.show("a", build)
    b = c.show("b", build)
    assert not a.visible and b.visible and c.show("a", build) is a and a.visible and not b.visible
    c.show("c", build)  # evicts b (least recently shown)
    assert b._impl.removed and len(c) == 2 and len(made) == 3
    c.hide()
    assert not any(h.visible for h in made)
    c.clear()
    assert all(h._impl.removed for h in made) and len(c) == 0


def test_heatmap_mode_switch_back_sends_almost_nothing(app):
    s, tab = app["session"], app["targets"]
    heat = tab.heat
    s.feasibility = _feasibility(s)
    tab.selected_results = lambda: {}
    n1, b1 = sent(lambda: setattr(heat.choice, "value", "Clearance"))
    assert n1["GlbMessage"] == 1 and not n1["MeshMessage"]  # one mesh, not one per bin
    assert len(heat._handles) == 1 and heat._handles[0].visible
    painted = heat.face_bins[heat.face_bins >= 0]
    assert len(painted) > 0 and len(set(painted.tolist())) > 1
    n2, b2 = sent(lambda: setattr(heat.choice, "value", "Manipulability"))
    assert n2["GlbMessage"] == 1
    first = heat._handles[0]
    n3, b3 = sent(lambda: setattr(heat.choice, "value", "Clearance"))  # back: cached
    assert not n3["GlbMessage"] and not n3["MeshMessage"]
    assert sum(b3.values()) < 2000 and n3["SetSceneNodeVisibilityMessage"] == 2
    assert heat._handles[0].visible and not first.visible
    # a recolour (new scores) sends at most one mesh's worth
    s.feasibility = _feasibility(s, 0.0007)
    n4, b4 = sent(heat.draw)
    assert n4["GlbMessage"] == 1 and b4["GlbMessage"] <= 1.05 * b1["GlbMessage"]
    assert sum(b4.values()) < 1.1 * b1["GlbMessage"] + 2000
    # switched off: hidden, legend empty, nothing uploaded; the cache stays bounded
    n5, _ = sent(lambda: setattr(heat.choice, "value", "Reached / unreached (points)"))
    assert not heat._handles and heat.legend.content == "" and not n5["GlbMessage"]
    assert not any(h.visible for h in heat._cache._nodes.values())
    for k in range(6):
        s.feasibility = _feasibility(s, 0.0001 * (k + 2))
        heat.choice.value = "Clearance"
        heat.draw()
    assert len(heat._cache) <= 4
    heat.choice.value = "Reached / unreached (points)"
    s.feasibility = None


def test_heatmap_refresh_with_the_heatmap_on_sends_no_mesh(app):
    s, tab = app["session"], app["targets"]
    s.feasibility = _feasibility(s)
    tab.selected_results = lambda: {}
    tab.heat.choice.value = "Joint-limit margin"
    tab.refresh()
    n, _ = sent(tab.refresh)
    assert not n["GlbMessage"] and not n["MeshMessage"]
    tab.heat.clear()  # cleared elsewhere: the next refresh shows the cached node again
    n, _ = sent(tab.refresh)
    assert tab.heat._handles and tab.heat._handles[0].visible and not n["GlbMessage"]
    tab.heat.choice.value = "Reached / unreached (points)"
    s.feasibility = None


def test_area_faces_match_the_per_group_meshes():
    from ui.station_areas import area_faces, area_meshes

    sol = SimpleNamespace(assignment={0: 0, 1: 0, 2: 0, 3: 1}, zones=[], label="fake", id=None)
    cfg = {0: "S+E+W+", 1: "S+E+W-", 2: "S+E+W-", 3: "S+E+W+"}
    mesh = trimesh.creation.box((4, 1, 0.01)).subdivide().subdivide().subdivide()
    P = np.array([[-1.5, 0, 0.005], [-0.5, 0, 0.005], [0.5, 0, 0.005], [1.5, 0, 0.005]])
    cols = [(200, 0, 0), (0, 0, 200)]
    old = {n: (len(f), c) for n, _, f, c in area_meshes(mesh, [0, 1, 2, 3], P, sol, cols, cfg)}
    faces, rgb, names = area_faces(mesh, [0, 1, 2, 3], P, sol, cols, cfg)
    assert names == sorted(old) and len(faces) == sum(n for n, _ in old.values())
    new = collections.Counter(map(tuple, rgb.tolist()))
    assert new == collections.Counter({c: n for n, c in old.values()})


def test_station_areas_are_one_cached_mesh(app):
    s, r = app["session"], app["results"]
    ids = [p.id for p in s.targets.points]
    sol = SimpleNamespace(assignment={i: k % 3 for k, i in enumerate(ids)}, zones=[], label="fake", id=None)
    n1, _ = sent(lambda: r.draw_areas(sol))
    assert n1["GlbMessage"] == 1 and len(r._areas) == 1 and set(r.area_names) == {"S1_G1", "S2_G1", "S3_G1"}
    h = r._areas[0]
    assert h.name.startswith("/cell/part/areas/")

    def redraw():
        r._drop()
        r.draw_areas(sol)

    n2, b2 = sent(redraw)  # the same picture: shown again, not re-sent
    assert not n2["GlbMessage"] and r._areas[0] is h and h.visible and sum(b2.values()) < 2000
    r._drop()
    assert not h.visible and not r._areas


def test_display_mesh_keeps_several_refined_meshes():
    from ui import station_areas as sa

    a = trimesh.creation.box((1, 1, 0.01))
    b = trimesh.creation.box((2, 1, 0.01))
    P = np.c_[np.linspace(-0.4, 0.4, 9), np.zeros(9), np.full(9, 0.005)]
    fa, fb = sa.display_mesh(a, P), sa.display_mesh(b, P)
    assert len(fa.faces) > len(a.faces) and len(fb.faces) > len(b.faces)
    assert sa.display_mesh(a, P) is fa and sa.display_mesh(b, P) is fb  # both still cached
    for k in range(sa.REFINED_KEEP + 1):
        sa.display_mesh(trimesh.creation.box((1 + k, 2, 0.01)), P)
    assert len(sa._REFINED) == sa.REFINED_KEEP
