"""v8 T20: the app in headless Edge: clicks, Alt+click Move, gizmo drag, stay-out zone pick,
dark-mode log colours. One app + one browser for the module (the scene takes ~10 s to stream).

The 3D points are projected to the canvas with the client's camera (viser reports it), and the
mouse events go through Chrome DevTools (``tests/e2e/cdp.py``)."""

import time

import numpy as np
import pytest
import viser

from tests.e2e import cdp

pytestmark = [pytest.mark.e2e, pytest.mark.skipif(not cdp.available(), reason="needs Edge + websockets")]

PORT, DEBUG = 8961, 9361
SHIFT, CTRL, ALT = 8, 2, 1


@pytest.fixture(scope="module")
def live():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=PORT, verbose=False)
    app = build_app(server)
    assert app["project"].load_example("rail_long_part")
    with cdp.Browser(f"http://127.0.0.1:{PORT}/", port=DEBUG) as b:
        t_end = time.time() + 60
        while not server.get_clients() and time.time() < t_end:
            time.sleep(0.25)
        client = next(iter(server.get_clients().values()))
        time.sleep(12)  # scene + meshes stream in
        rects = b.eval(
            "[...document.querySelectorAll('canvas')].map(c=>{const r=c.getBoundingClientRect();"
            "return [r.left,r.top,r.width,r.height]})"
        )
        canvas = max(rects, key=lambda r: r[2] * r[3])
        yield app, server, client, b, canvas
    server.stop()


def screen(client, canvas, p):
    """World point -> page pixels (pinhole camera from the client's reported camera)."""
    cam = client.camera
    pos, look, up = np.asarray(cam.position), np.asarray(cam.look_at), np.asarray(cam.up_direction)
    f = look - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = np.asarray(p, float) - pos
    x, y, z = d @ r, d @ u, d @ f
    t = np.tan(cam.fov / 2)
    nx, ny = x / (z * t * cam.aspect), y / (z * t)
    return canvas[0] + (nx + 1) / 2 * canvas[2], canvas[1] + (1 - ny) / 2 * canvas[3]


def part_top(app):
    s = app["session"]
    m, Tp = s.part_mesh(), s.part.T_world_part
    c = m.bounds.mean(axis=0)
    c[2] = m.bounds[1, 2]
    return Tp[:3, :3] @ c + Tp[:3, 3]


def settle(t=1.5):
    time.sleep(t)


def test_surface_selection_plain_shift_ctrl(live):
    app, _server, client, b, canvas = live
    tg = app["targets"]
    tg.surf.tool.value = "Fill +"
    tg.select(np.zeros(0, int), "set")
    x, y = screen(client, canvas, part_top(app))
    b.click(x, y)
    settle()
    n = len(tg.selection)
    assert n > 0  # plain click: the top face's region
    b.click(x, y, CTRL)
    settle()
    assert len(tg.selection) == 0  # ctrl inverts Fill + to remove
    b.click(x, y, SHIFT)
    settle()
    assert len(tg.selection) == n  # shift adds


def test_alt_click_opens_move_and_cancel_restores(live):
    app, _server, client, b, canvas = live
    mover, s = app["objects"].mover, app["session"]
    before = s.part.T_world_part.copy()
    x, y = screen(client, canvas, part_top(app))
    b.click(x, y, ALT)
    settle()
    assert mover.key is not None and "Part" in mover.key
    s.part.T_world_part = before @ np.diag([1.0, 1, 1, 1])  # unchanged copy
    mover.cancel()
    assert mover.key is None and np.allclose(s.part.T_world_part, before)


def test_gizmo_drag_moves_the_part(live):
    app, _server, client, b, canvas = live
    s, cell = app["session"], app["cell"]
    objects = app["objects"]
    objects.gizmo.value = "Part"  # Drag gizmo on: the part
    settle()
    before = s.part.T_world_part.copy()
    origin = before[:3, 3]
    g = cell.part_pose.gizmo
    scale = float(getattr(g, "scale", 0.5))
    a = screen(client, canvas, origin + np.array([0.0, 0.0, 0.75 * scale]))  # on the z arrow
    to = screen(client, canvas, origin + np.array([0.0, 0.0, 0.75 * scale + 0.3]))
    b.drag(a, to)
    settle(2.0)
    moved = s.part.T_world_part[:3, 3] - origin
    objects.gizmo.value = "None" if "None" in objects.gizmo.options else objects.gizmo.options[0]
    s.part.T_world_part = before
    cell.refresh_part()
    assert np.linalg.norm(moved) > 0.02, f"part did not move (gizmo scale {scale:.2f})"


def test_stayout_zone_click_selects_it(live):
    app, _server, client, b, canvas = live
    st = app["stayouts"]
    st.add_shape("box")
    first = app["session"].stayouts[-1]
    st.xyz.value = (0.0, -1500.0, 0.0)  # clear of the rail part (new zones start next to it)
    st.add_shape("box")
    first = next(z for z in app["session"].stayouts if z.name == first.name)
    assert st.pick.value != first.name  # the newest is picked
    c = np.array(first.pose.xyz) + np.array([0.0, 0.0, first.size[2] / 2])
    settle(2.5)  # adding a zone redraws every zone: let the browser get the new meshes first
    x, y = screen(client, canvas, c)
    b.click(x, y)
    settle()
    assert st.pick.value == first.name


def test_dark_mode_log_colours(live):
    app, _server, _client, b, _canvas = live
    from ui.log_panel import COLORS, log

    app["dark_toggle"].value = True
    log("e2e dark-mode check", "error")
    settle()
    html = b.eval("document.body.innerHTML")
    assert COLORS["error"][1] in html.lower() or COLORS["error"][1].upper() in html
    app["dark_toggle"].value = False


def test_pareto_dot_click_selects_plan(live):
    """A real mouse click on a dot of the Results -> Pareto front chart selects that plan."""
    app, _server, _client, b, _canvas = live
    from core.schema import Solution, Station

    s, results, view = app["session"], app["results"], app["pareto"]
    old = list(s.results)
    s.results = [
        Solution(
            label=f"Pareto {i}/3: test",
            stations=[Station(xyz=(0.0, -1.0, 0.0))] * i,
            cycle_s=100.0 - 20 * i,
            cost=float(i),
        )
        for i in (1, 2, 3)
    ]
    try:
        results.refresh()
        view.refresh()
        assert results.selected() is s.results[0]
        tab = b.eval(  # viser's tabs switch on a real mouse press, not element.click()
            "(()=>{const t=[...document.querySelectorAll('[role=tab]')].find(e=>/Results$/.test("
            "e.textContent.trim()));const r=t.getBoundingClientRect();"
            "return [r.left+r.width/2,r.top+r.height/2]})()"
        )
        b.click(tab[0], tab[1])
        settle(2.0)
        r = b.eval(
            "(()=>{const g=document.querySelector('g[data-plan=\"3\"]');if(!g)return null;"
            "g.scrollIntoView({block:'center'});const c=g.querySelector('circle').getBoundingClientRect();"
            "return [c.left+c.width/2,c.top+c.height/2]})()"
        )
        assert r is not None, "Pareto chart not rendered"
        settle(0.5)
        r = b.eval(
            "(()=>{const c=document.querySelector('g[data-plan=\"3\"] circle').getBoundingClientRect();"
            "return [c.left+c.width/2,c.top+c.height/2]})()"
        )
        b.click(r[0], r[1])
        settle(2.0)
        assert results.selected() is s.results[2]
    finally:
        s.results = old
        results.refresh()
        view.refresh()


def test_joystick_pad_drives_the_carrier(live):
    """Holding the joystick knob up in a real browser drives the carrier (the pad's inline
    handler writes into the Joystick text input, which viser sends to the server)."""
    app, _server, _client, b, _canvas = live
    s, car, drive = app["session"], app["carrier"], app["drive"]
    drive.speed.value = 1000.0
    pos = b.eval(  # open the folder (a real mouse press on its header)
        "(()=>{const h=[...document.querySelectorAll('*')].filter(e=>/^Carrier gizmo . joystick$/.test("
        "e.textContent.trim())).pop();if(!h)return null;h.scrollIntoView({block:'center'});"
        "const r=h.getBoundingClientRect();return [r.left+r.width/2,r.top+r.height/2]})()"
    )
    assert pos is not None, "folder not found"
    b.click(pos[0], pos[1])
    settle(1.5)
    pad = b.eval(
        "(()=>{const p=document.querySelector('.rso-joy');if(!p)return null;"
        "p.scrollIntoView({block:'center'});"
        "const r=p.getBoundingClientRect();return [r.left+r.width/2,r.top+r.height/2,r.width]})()"
    )
    assert pad is not None, "joystick pad not rendered"
    c = s.carrier()
    s0 = car.current(c)["s"]
    cx, cy, w = pad
    b.cdp("Input.dispatchMouseEvent", type="mouseMoved", x=cx, y=cy)
    b.cdp("Input.dispatchMouseEvent", type="mousePressed", x=cx, y=cy, button="left", clickCount=1)
    for k in range(1, 6):  # knob to the right: the rail drives toward +s
        b.cdp("Input.dispatchMouseEvent", type="mouseMoved", x=cx + k * w / 10, y=cy, button="left")
        time.sleep(0.05)
    time.sleep(1.5)  # held: the pad repeats every 100 ms
    b.cdp("Input.dispatchMouseEvent", type="mouseReleased", x=cx + w / 2, y=cy, button="left", clickCount=1)
    settle(1.0)
    s1 = car.current(c)["s"]
    settle(1.0)
    assert s1 - s0 > 0.3, f"carrier moved {s1 - s0:.3f} m"
    assert car.current(c)["s"] == s1  # released: stopped
    assert drive.value.value.startswith("joy:0.00,0.00")


def test_package_builder_opens_in_its_own_window(live):
    """The Package builder tab's button opens the builder (its own viser server) in a new
    window, which connects to that server, not the project's."""
    app, server, _client, b, _canvas = live
    w = app["builder_window"]
    assert w.port != PORT
    n_main = len(server.get_clients())
    pos = b.eval(
        "(()=>{const h=[...document.querySelectorAll('button,[role=tab]')].filter(e=>/Package builder$/.test("
        "e.textContent.trim())).pop();if(!h)return null;h.scrollIntoView({block:'center'});"
        "const r=h.getBoundingClientRect();return [r.left+r.width/2,r.top+r.height/2]})()"
    )
    assert pos is not None, "Package builder tab not found"
    b.click(pos[0], pos[1])
    settle(1.5)
    btn = b.eval(
        "(()=>{const h=[...document.querySelectorAll('button')].filter(e=>/Open package builder/.test("
        "e.textContent)).pop();if(!h)return null;h.scrollIntoView({block:'center'});"
        "const r=h.getBoundingClientRect();return [r.left+r.width/2,r.top+r.height/2]})()"
    )
    assert btn is not None, "open button not rendered"
    b.click(btn[0], btn[1])
    t_end = time.time() + 30
    while not w.server.get_clients() and time.time() < t_end:
        time.sleep(0.25)
    assert w.server.get_clients(), "the builder window did not connect"
    assert len(server.get_clients()) == n_main  # the project page did not navigate away
