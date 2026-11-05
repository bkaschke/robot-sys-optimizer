"""User guide as HTML: in sync with the Markdown, links resolve, served on its own port with
nothing else exposed, and linked from the panel."""

import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_guide  # noqa: E402

from ui.guide import GUIDE, GuideServer, link_html  # noqa: E402


def test_html_is_up_to_date_with_markdown():
    assert GUIDE.read_text(encoding="utf-8") == build_guide.build(), "run python scripts/build_guide.py"


def test_every_internal_link_has_a_target():
    page = GUIDE.read_text(encoding="utf-8")
    ids = set(re.findall(r' id="([^"]+)"', page))
    hrefs = set(re.findall(r'href="#([^"]+)"', page))
    assert hrefs and hrefs <= ids, hrefs - ids
    assert "<script>" in page and page.count("<script") == 1  # only the theme toggle
    assert "src=" not in page and "<link" not in page  # self-contained


def test_slug_matches_github():
    assert build_guide.slug("1. Quickstart (5 minutes)") == "1-quickstart-5-minutes"
    assert build_guide.slug("4. Tools and features, tab by tab") == "4-tools-and-features-tab-by-tab"


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as err:
        return err.code, b""


def test_guide_server_serves_only_the_guide():
    g = GuideServer("127.0.0.1", 8781)
    try:
        base = f"http://127.0.0.1:{g.port}"
        status, body = _get(base + "/")
        assert status == 200 and body == GUIDE.read_bytes()
        assert _get(base + "/USER_GUIDE.html")[0] == 200
        for bad in ("/USER_GUIDE.md", "/../app.py", "/%2e%2e/CLAUDE.md", "/docs/USER_GUIDE.html"):
            assert _get(base + bad)[0] == 404, bad
        g2 = GuideServer("127.0.0.1", g.port)  # port taken: moves to the next free one
        assert g2.port != g.port
        g2.stop()
    finally:
        g.stop()


def test_panel_link_and_download(tmp_path):
    import viser

    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8782, verbose=False)
    try:
        state = build_app(server, guide_port=8783)
        assert ":8783/" in state["guide"]["link"].content and 'target="_blank"' in link_html(8783)
        assert "download" in state["guide"]
        bare = build_app(viser.ViserServer(host="127.0.0.1", port=8784, verbose=False))
        assert "link" not in bare["guide"]  # no guide server: download button only
    finally:
        server.stop()


@pytest.mark.parametrize("arg", ["--check"])
def test_check_mode(arg, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_guide.py", arg])
    assert build_guide.main() == 0
