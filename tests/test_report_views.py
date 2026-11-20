"""R3: framed report views (ui/report_views.py) with fake client and server."""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_report_make import planned  # noqa: F401  (fixture)
from ui import report_views as rv
from ui.floor import NODE


class Handle:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class Scene:
    def __init__(self):
        self.grid = Handle(visible=True)
        self.origin = Handle(visible=True)
        self.sel = Handle(visible=True)
        self.robot = Handle(visible=True)
        self.gizmo = Handle(visible=True, disable_axes=False, disable_sliders=False, disable_rotations=False)
        self._handle_from_node_name = {
            NODE: self.grid,
            "/world/origin": self.origin,
            "/cell/part/selection": self.sel,
            "/cell/base/urdf_root": self.robot,
            "/cell/base": self.gizmo,
        }

    def state(self):
        return (
            self.grid.visible,
            self.origin.visible,
            self.sel.visible,
            self.robot.visible,
            self.gizmo.disable_axes,
            self.gizmo.disable_sliders,
            self.gizmo.disable_rotations,
            self.gizmo.visible,
        )


class Server:
    def __init__(self):
        self.scene = Scene()


class Camera:
    def __init__(self, server, renders):
        self.position, self.look_at, self.up_direction = (1.0, 2.0, 3.0), (0.0, 0.0, 0.5), (0.0, 0.0, 1.0)
        self.server, self.renders, self.calls, self.seen = server, list(renders), 0, []

    def get_render(self, h, w, transport_format="png", timeout=None):
        self.calls += 1
        self.seen.append((self.server.scene.state(), tuple(self.position)))
        r = self.renders.pop(0) if self.renders else good()
        if isinstance(r, Exception):
            raise r
        return r


def blank():
    return np.zeros((8, 16, 4), np.uint8)  # transparent: an empty scene


def good():
    a = np.zeros((8, 16, 4), np.uint8)
    a[2:6, 4:12] = (200, 30, 30, 255)
    return a


class Client:
    def __init__(self, server, renders=()):
        self.camera = Camera(server, renders)


class Session:
    pass


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(rv, "SETTLE_S", 0.0)
    monkeypatch.setattr(rv, "RETRY_WAIT_S", 0.0)
    box = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 1.0]])
    monkeypatch.setattr(rv, "cell_bounds", lambda s: box)


def test_blank_then_good_retries():
    srv = Server()
    cl = Client(srv, [blank(), good()])
    out = rv.capture(srv, cl, Session(), views=("Iso",))
    assert [c for c, _ in out] == ["Iso view"]
    assert out[0][1].startswith(b"\x89PNG")
    assert cl.camera.calls == 2 and not out.notes


def test_always_blank_gives_note_after_three_tries():
    srv = Server()
    cl = Client(srv, [blank()] * 3)
    out = rv.capture(srv, cl, Session(), views=("Iso",))
    assert len(out) == 0 and cl.camera.calls == 3
    assert "stayed empty" in out.notes[0] and out.notes[0].startswith("3D views not included")


def test_state_restored_when_render_raises():
    srv = Server()
    before, cam0 = srv.scene.state(), None
    cl = Client(srv, [TimeoutError("no frame")])
    cam = cl.camera
    cam0 = (cam.position, cam.look_at, cam.up_direction)
    out = rv.capture(srv, cl, Session())
    assert len(out) == 0 and "TimeoutError" in out.notes[0]
    assert srv.scene.state() == before
    assert (cam.position, cam.look_at, cam.up_direction) == cam0


def test_helpers_hidden_at_render_time_and_camera_moved():
    srv = Server()
    before = srv.scene.state()
    cl = Client(srv)
    out = rv.capture(srv, cl, Session(), views=("Iso", "Top"))
    assert [c for c, _ in out] == ["Iso view", "Top view"]
    for (grid, origin, sel, robot, ax, sl, rot, gviz), _pos in cl.camera.seen:
        assert (grid, origin, sel) == (False, False, False)
        assert (ax, sl, rot) == (True, True, True)
        assert robot is True and gviz is True  # the gizmo's subtree (the robot) stays visible
    assert cl.camera.seen[0][1] != cl.camera.seen[1][1]
    assert srv.scene.state() == before


def test_no_client_and_empty_study(monkeypatch):
    out = rv.capture(Server(), None, Session())
    assert len(out) == 0 and out.notes[0].startswith("3D views not included")
    monkeypatch.setattr(rv, "cell_bounds", lambda s: None)
    out = rv.capture(Server(), Client(Server()), Session())
    assert len(out) == 0 and "nothing to show" in out.notes[0]


def test_make_report_with_images_and_note(planned):  # noqa: F811
    s, _panel, sol = planned
    from ui import report_make as rm

    png = rv.to_png(good())[0]
    html = rm.make_report(s, sol, images=[("Iso view", png), ("Top view", png)], notes=["x"])
    assert html.count("<figure>") == 2 and "Iso view" in html and "Top view" in html
    assert "class='figs'" in html
    none = rm.make_report(s, sol, notes=[rm.NO_VIEWS_BATCH])
    assert "<figure>" not in none and "3D views not included" in none and "rendered only from the GUI" in none
