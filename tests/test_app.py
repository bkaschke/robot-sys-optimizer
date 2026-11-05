import urllib.request

import viser

from app import TABS, build_app


def test_app_serves_page_and_scene():
    server = viser.ViserServer(host="127.0.0.1", port=8765, verbose=False)
    try:
        state = build_app(server)
        assert set(state["tabs"]) == set(TABS)
        assert state["part_name"] == "flat_plate"

        state["part_choice"].value = "curved_panel"
        assert state["part_name"] == "curved_panel"

        with urllib.request.urlopen(f"http://127.0.0.1:{server.get_port()}/", timeout=10) as r:
            assert r.status == 200
    finally:
        server.stop()
