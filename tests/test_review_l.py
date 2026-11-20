"""Review fixes (WP L): delete-all confirm + undo, max_manip reset on load, joint-limit warning."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import viser

from core import schema
from ui.session import Session, load_project, save_project


class _Ctx:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def close(self):
        self.log.append("closed")


class _Btn:
    def __init__(self, label):
        self.label = label
        self.cb = None

    def on_click(self, fn):
        self.cb = fn
        return fn


class _Gui:
    def __init__(self):
        self.modals, self.texts, self.buttons = [], [], {}

    def add_modal(self, title):
        self.modals.append(title)
        return _Ctx(self.modals)

    def add_markdown(self, text):
        self.texts.append(text)

    def add_button(self, label, **kw):
        self.buttons[label] = _Btn(label)
        return self.buttons[label]


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8791, verbose=False)
    yield build_app(server)
    server.stop()


def test_delete_all_confirms_and_undo_restores(app):
    tab, s = app["targets"], app["session"]
    tab.tools.undo_stack.clear()
    s.targets = schema.Targets()
    tab.tools.add(np.array([[0.0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]]), np.tile([0.0, 0, 1], (3, 1)))
    tab.tools.undo_stack.clear()
    before = [p.id for p in s.targets.points]
    assert len(before) == 3
    gui = _Gui()
    ev = SimpleNamespace(client=SimpleNamespace(gui=gui, add_notification=lambda *a, **k: None))
    tab._ask_delete_all(ev)
    assert len(s.targets.points) == 3  # nothing deleted before the user confirms
    assert gui.modals and any("3" in t and "out of date" in t for t in gui.texts)
    gui.buttons["Cancel"].cb(ev)
    assert len(s.targets.points) == 3
    gui.buttons["Delete all points"].cb(ev)
    assert s.targets.points == []
    assert tab.tools.undo_last(ev) == 3
    assert [p.id for p in s.targets.points] == before


def test_max_manip_reset_across_load(tmp_path):
    s = Session()
    assert s.max_manip is None
    s.max_manip = 0.7
    save_project(s, tmp_path / "p")
    load_project(s, tmp_path / "p")
    assert s.max_manip is None


def test_summary_card_without_max_manip():
    from core.feasibility import PointResult
    from core.schema import Solution, Station
    from ui.summary_card import card

    s = Session()
    s.targets = schema.Targets(
        groups=[schema.TargetGroup(name="default")],
        points=[schema.TargetPoint(id=0, p=(0, 0, 0), n=(0, 0, 1), group="default")],
    )
    sol = Solution(label="x", stations=[Station(xyz=(0, 0, 0))], assignment={0: 0})
    s.result_points["x"] = {
        0: PointResult(0, "reachable", q=[0.0] * 6, manipulability=0.3, limit_margin_deg=40.0)
    }
    md = card(s, sol)
    assert "Lowest manipulability" in md and " %" not in md.split("Lowest manipulability |")[1].split("|")[1]


def test_bad_joint_limits_warn(tmp_path):
    s = Session()
    s.load_robot_from_registry("ur20")
    save_project(s, tmp_path / "p")
    f = tmp_path / "p" / "project.json"
    d = json.loads(f.read_text())
    d["robot"]["joint_limits_deg"] = [[10.0, 5.0]] * 6  # empty range on every joint
    f.write_text(json.dumps(d))
    load_project(s, tmp_path / "p")
    assert s.load_warnings and "Joint limits" in s.load_warnings[0]
    load_project(s, tmp_path / "p")
    assert len(s.load_warnings) == 1  # one warning per load, not accumulating
