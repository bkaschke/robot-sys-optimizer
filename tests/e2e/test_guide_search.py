"""v8 T4: the HTML guide's search box, in headless Edge."""

import pytest

from tests.e2e import cdp
from ui.guide import GUIDE, GuideServer

pytestmark = [pytest.mark.e2e, pytest.mark.skipif(not cdp.available(), reason="needs Edge + websockets")]


@pytest.fixture(scope="module")
def guide():
    srv = GuideServer("127.0.0.1", 8950)
    yield srv
    srv.stop()


def test_search_highlights_filters_and_jumps(guide):
    with cdp.Browser(f"http://127.0.0.1:{guide.port}/", port=9341) as b:
        b.wait_for("document.getElementById('search') !== null")
        nav_all = b.eval("document.querySelectorAll('nav a[href^=\"#\"]').length")
        n = b.eval("rsoGuideSearch('CMA-ES')")
        assert n >= 2  # the station planner and tool optimizer sections both mention it
        hidden = b.eval("document.querySelectorAll('nav a.hide').length")
        assert 0 < hidden < nav_all  # the sidebar keeps only sections with matches
        assert b.eval("document.querySelectorAll('mark.hit.cur').length") == 1
        first = b.eval(
            "[...document.querySelectorAll('mark.hit')].indexOf(document.querySelector('mark.cur'))"
        )
        b.eval("document.getElementById('search').focus()")
        b.key("Enter")
        assert (
            b.eval("[...document.querySelectorAll('mark.hit')].indexOf(document.querySelector('mark.cur'))")
            == (first + 1) % n
        )
        b.key("Escape")
        assert b.eval("document.querySelectorAll('mark.hit').length") == 0
        assert b.eval("document.querySelectorAll('nav a.hide').length") == 0
        text = GUIDE.read_text(encoding="utf-8")
        assert b.eval("document.querySelector('main').innerHTML.length") > 0.5 * text.count(
            "<"
        )  # text intact


def test_query_parameter_searches_on_load(guide):
    with cdp.Browser(f"http://127.0.0.1:{guide.port}/?q=stay-out", port=9342) as b:
        b.wait_for("document.querySelectorAll('mark.hit').length > 0")
        assert b.eval("document.getElementById('search').value") == "stay-out"
