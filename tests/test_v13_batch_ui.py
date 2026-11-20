"""Optimize -> Several parts (batch): a folder of v13 part packages queued and run from the panel."""

import pytest
import viser

from core import batch as bt
from tests.test_v13_batch import make_parts


@pytest.fixture(scope="module")
def app():
    from app import build_app

    server = viser.ViserServer(host="127.0.0.1", port=8987, verbose=False)
    a = build_app(server)
    assert a["project"].load_example("rail_long_part")
    yield a
    a["builder_window"].stop()
    server.stop()


def test_add_folder_of_packages(app, tmp_path):
    from tests.helpers import confirm_all

    b = app["batch"]
    b.batch_dir = None
    confirm_all(app["session"])
    assert "Add jobs" in b.why_not()
    make_parts(tmp_path / "parts")
    (tmp_path / "parts" / "junk").mkdir()
    (tmp_path / "parts" / "junk" / "part.json").write_text("{}", encoding="utf-8")
    (tmp_path / "parts" / "other").mkdir()  # not a package: ignored
    queued, skipped = b.add_folder(str(tmp_path / "parts"))
    assert len(queued) == 2 and len(skipped) == 1 and skipped[0].startswith("junk: ")
    jobs = b.manifest().jobs
    assert [(j.name, j.source) for j in jobs] == [("rail_a", "package"), ("rail_b", "package")]
    for j in jobs:
        assert (b.batch_dir / j.package / "part.json").is_file() and (b.batch_dir / j.part).is_file()
        assert bt.check_job(j, b.batch_dir) is None and j.summary["targets_preview"] > 0
    assert j.pose[0][3] == pytest.approx(0.4)  # the package's world pose
    assert "package" in b.table.content and b.why_not() == ""


@pytest.mark.slow
def test_run_from_the_panel(app, tmp_path):
    from tests.helpers import confirm_all

    b = app["batch"]
    b.batch_dir = None
    confirm_all(app["session"])
    make_parts(tmp_path / "parts")
    b.add_folder(str(tmp_path / "parts"))
    b.pick.value = b.pick.options[1]
    assert b.remove()
    opened = []
    b.on_done = opened.append
    b.run(block=True)
    m = b.manifest()
    assert [j.status for j in m.jobs] == ["done"], [j.error for j in m.jobs]
    assert opened == [b.batch_dir / bt.RESULTS]
    assert (b.batch_dir / bt.SETUP / "project.json").is_file()
    assert (b.batch_dir / m.jobs[0].out_dir / "project.json").is_file() and m.jobs[0].summary["cycle_s"]
    assert b.run_btn.visible and b.run_btn.disabled and "No pending jobs" in b.reason_md.content  # all done
