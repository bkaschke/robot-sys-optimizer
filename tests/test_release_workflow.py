"""Structure checks of .github/workflows/release.yml (it cannot run here) and the serve-check script."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
WF = yaml.safe_load(TEXT)
TRIGGERS = WF.get("on", WF.get(True))  # PyYAML reads the key `on` as True


def steps(job: str) -> list[dict]:
    return WF["jobs"][job]["steps"]


def step(job: str, needle: str) -> dict:
    hits = [s for s in steps(job) if needle in s.get("name", "")]
    assert hits, f"no step named like {needle!r}"
    return hits[0]


def test_triggers():
    assert TRIGGERS["push"]["tags"] == ["v*"]
    draft = TRIGGERS["workflow_dispatch"]["inputs"]["draft"]
    assert draft["default"] is True


def test_windows_job_setup():
    job = WF["jobs"]["windows"]
    assert job["runs-on"] == "windows-latest"
    assert job["timeout-minutes"] == 45
    assert any(s.get("with", {}).get("fetch-depth") == 0 for s in steps("windows"))
    assert any("setup-miniconda" in s.get("uses", "") for s in steps("windows"))
    cache = step("windows", "Reachability map cache")
    assert "runner.os" in cache["with"]["key"]
    assert "innosetup" in step("windows", "Inno Setup")["run"]
    assert "scripts/build_exe.py --maps --zip --installer --map-jobs 4" in step("windows", "Build exe")["run"]


def test_default_permissions_are_read_only_and_write_is_scoped_to_release():
    assert WF["permissions"] == {"contents": "read"}
    assert "permissions" not in WF["jobs"]["windows"]
    assert WF["jobs"]["release"]["permissions"] == {"contents": "write"}
    assert TEXT.count("contents: write") == 1


def test_firewall_rule_blocks_outbound_and_is_always_removed():
    block = step("windows", "Block outbound")["run"]
    assert "New-NetFirewallRule" in block and "-Direction Outbound" in block
    assert "-Action Block" in block and "-RemoteAddress Internet" in block
    remove = step("windows", "Remove firewall rule")
    assert remove["if"] == "always()"
    assert "Remove-NetFirewallRule" in remove["run"]
    names = [s.get("name", "") for s in steps("windows")]
    i_block, i_rm = names.index("Block outbound network (firewall rule)"), names.index(remove["name"])
    i_self = names.index("Offline selftest of the built exe")
    i_serve = names.index("Offline serve check (exe + headless Edge)")
    assert i_block < i_self < i_serve < i_rm


def test_offline_steps_set_the_guard_and_a_data_folder():
    st = step("windows", "Offline selftest")
    assert st["env"]["RSO_OFFLINE_GUARD"] == "1"
    assert "--selftest" in st["run"] and "RSO_DATA" in st["run"]
    sv = step("windows", "Offline serve check")
    assert sv["env"]["RSO_OFFLINE_GUARD"] == "1" and "RSO_DATA" in sv["run"]
    assert "ci_serve_check.ps1" in sv["run"] and "8765" in sv["run"]
    assert (ROOT / "packaging" / "ci_serve_check.ps1").is_file()


def test_installer_is_install_tested_silently_per_user():
    run = step("windows", "Install test")["run"]
    for flag in ("/VERYSILENT", "/CURRENTUSER", "/SUPPRESSMSGBOXES"):
        assert flag in run
    assert "--selftest" in run and "unins" in run


def test_artifacts_cover_zip_setup_and_sums():
    up = [s for s in steps("windows") if "upload-artifact" in s.get("uses", "")][0]
    paths = up["with"]["path"]
    assert "win64.zip" in paths and "setup.exe" in paths and "SHA256SUMS.txt" in paths


def test_release_job_makes_a_draft_and_checks_sizes():
    job = WF["jobs"]["release"]
    assert job["needs"] == "windows"
    assert "refs/tags/v" in job["if"] and "workflow_dispatch" in job["if"]
    check = step("release", "Check asset sizes")["run"]
    assert "2 * 1024 * 1024 * 1024" in check and "exit 1" in check
    create = step("release", "Create the release")
    assert "gh release create" in create["run"] and "--draft" in create["run"]
    # a tag push is always a draft; the dispatch input can only switch it off when unticked
    # format() keeps a ticked-off `false` from falling through `a && b || c` to the default
    assert create["env"]["DRAFT"].endswith("format('{0}', inputs.draft) || 'true' }}")
    assert not any("action-gh-release" in s.get("uses", "") for s in steps("release"))


def test_release_job_writes_notes_from_template_and_appends_generated_ones():
    job = WF["jobs"]["release"]
    checkout = [s for s in steps("release") if "actions/checkout" in s.get("uses", "")]
    assert checkout and "release_notes.py" in checkout[0]["with"]["sparse-checkout"]
    assert "RELEASE_NOTES.md.in" in checkout[0]["with"]["sparse-checkout"]
    assert job["permissions"] == {"contents": "write"}
    notes = step("release", "Write the release notes")["run"]
    assert "scripts/release_notes.py release" in notes and "--out notes.md" in notes
    names = [s.get("name", "") for s in steps("release")]
    assert names.index(step("release", "Write the release notes")["name"]) < names.index(
        step("release", "Create the release")["name"]
    )
    create = step("release", "Create the release")["run"]
    assert "--notes-file notes.md" in create and "--generate-notes" in create
    assert (ROOT / "scripts" / "release_notes.py").is_file()
    assert (ROOT / "packaging" / "RELEASE_NOTES.md.in").is_file()


def test_linux_job_is_only_a_comment():
    assert set(WF["jobs"]) == {"windows", "release"}
    assert "# linux:" in TEXT and "AppImage" in TEXT


def test_every_action_is_pinned_to_a_major_version():
    for job in WF["jobs"].values():
        for s in job["steps"]:
            if "uses" in s:
                assert "@v" in s["uses"], s["uses"]


@pytest.mark.skipif(not (ROOT / "packaging" / "ci_serve_check.ps1").is_file(), reason="script missing")
def test_serve_check_script_waits_for_the_ready_line_the_app_prints():
    ps = (ROOT / "packaging" / "ci_serve_check.ps1").read_text(encoding="utf-8")
    assert "Robot Cell Optimizer ready:" in ps
    assert "Robot Cell Optimizer ready:" in (ROOT / "app.py").read_text(encoding="utf-8")
    assert "--dump-dom" in ps and "--no-browser" in ps
