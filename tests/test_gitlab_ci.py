"""Structure checks of the GitLab pipeline (it cannot run here) against the GitHub workflows it mirrors,
and scripts/gitlab_release.py with a fake HTTP layer."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MAIN = yaml.safe_load((ROOT / ".gitlab-ci.yml").read_text(encoding="utf-8"))
PARTS = {p: (ROOT / p).read_text(encoding="utf-8") for p in (i["local"] for i in MAIN["include"])}
JOBS: dict = {}
for _text in PARTS.values():
    JOBS.update(yaml.safe_load(_text))
GH_TESTS = yaml.safe_load((ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8"))
GH_RELEASE = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")


def job(name: str) -> dict:
    """The job with its `extends` chain merged like GitLab does (later wins, dicts merge one level)."""
    out: dict = {}
    raw = JOBS[name]
    parents = raw.get("extends", [])
    for parent in [parents] if isinstance(parents, str) else parents:
        for k, v in job(parent).items():
            out[k] = {**out[k], **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    for k, v in raw.items():
        out[k] = {**out[k], **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def text(name: str, key: str = "script") -> str:
    return "\n".join(job(name).get(key, []))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "gitlab_release.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_includes_exist_and_runner_assumptions_are_top_level_variables():
    assert set(PARTS) == {".gitlab/ci/setup.yml", ".gitlab/ci/tests.yml", ".gitlab/ci/release.yml"}
    v = MAIN["variables"]
    for key in ("RUNNER_TAG_WINDOWS", "RUNNER_TAG_LINUX", "LINUX_TEST_IMAGE", "LINUX_RELEASE_IMAGE"):
        assert isinstance(v[key], str) and v[key]
    assert v["PYTHONUTF8"] == "1"
    assert job("fast:windows")["tags"] == ["$RUNNER_TAG_WINDOWS"] == job("windows")["tags"]
    assert job("fast:linux")["tags"] == ["$RUNNER_TAG_LINUX"] == job("release")["tags"]
    assert job("fast:linux")["image"] == "$LINUX_TEST_IMAGE"
    assert job("release")["image"] == "$LINUX_RELEASE_IMAGE" == job("release-notes")["image"]
    # the prefilled variables of a manual run: the default is "no release" and "wait before publishing"
    assert v["RELEASE_RUN"]["value"] == "false" and v["RELEASE_DRAFT"]["value"] == "true"
    for key in ("RELEASE_RUN", "RELEASE_DRAFT"):
        assert v[key]["value"] in v[key]["options"] and v[key]["description"]


def test_no_duplicate_branch_and_merge_request_pipelines():
    rules = MAIN["workflow"]["rules"]
    assert rules[0] == {"if": "$CI_COMMIT_TAG =~ /^manual-/", "when": "never"}  # tags the release job creates
    ifs = [r["if"] for r in rules]
    assert "$CI_COMMIT_TAG" in ifs and '$CI_PIPELINE_SOURCE == "merge_request_event"' in ifs
    dup = [r for r in rules if "$CI_OPEN_MERGE_REQUESTS" in r["if"]]
    assert dup and dup[0]["when"] == "never"
    assert ifs.index(dup[0]["if"]) < ifs.index("$CI_COMMIT_BRANCH")


def test_fast_jobs_run_the_same_checks_as_the_github_workflow():
    gh = GH_TESTS["jobs"]["fast"]
    # the checks only: GitHub's conditional env step (cache miss) is GitLab's before_script
    checks = [s for s in gh["steps"] if "run" in s and "if" not in s]
    gh_cmds = [ln.strip() for s in checks for ln in s["run"].splitlines() if ln.strip()]
    for name in ("fast:windows", "fast:linux"):
        j = job(name)
        assert j["script"] == gh_cmds
        assert j["timeout"] == f"{gh['timeout-minutes']} minutes"
        assert j["stage"] == "test"
        # not in a manual release run, otherwise always
        assert j["rules"][0]["when"] == "never" and "RELEASE_RUN" in j["rules"][0]["if"]
        assert "environment.yml" in text(name, "before_script")
    assert "conda activate rso" in text("fast:linux", "before_script")
    assert "env create" in text("fast:windows", "before_script")


def test_lint_job_matches_the_github_lint_job():  # v12 #12: lint on Linux with the pinned ruff
    gh = GH_TESTS["jobs"]["lint"]
    gh_cmds = [ln.strip() for s in gh["steps"] if "run" in s for ln in s["run"].splitlines() if ln.strip()]
    assert job("lint")["script"] == gh_cmds
    assert job("lint")["timeout"] == f"{gh['timeout-minutes']} minutes"


@pytest.mark.parametrize("name", ["fast:windows", "fast:linux", "windows"])
def test_reachability_cache_has_the_github_key_inputs(name):
    steps = GH_TESTS["jobs"]["fast"]["steps"]
    gh_key = [s for s in steps if s.get("name") == "Reachability map cache"][0]["with"]
    inputs = re.findall(r"'([^']+)'", gh_key["key"])
    maps = [c for c in job(name)["cache"] if c["paths"] == [gh_key["path"]]]
    assert len(maps) == 1
    assert maps[0]["key"]["files"] == inputs
    assert maps[0]["key"]["prefix"] == ("maps-linux" if name.endswith("linux") else "maps-windows")
    assert len(job(name)["cache"]) <= 4  # GitLab's limit per job
    assert all(len(c["key"]["files"]) <= 2 for c in job(name)["cache"])  # the documented default limit
    for pattern in inputs:
        assert list(ROOT.glob(pattern)), pattern


def test_release_jobs_run_on_v_tags_and_manual_runs_only():
    for name in ("windows", "release-notes"):
        assert [r["if"] for r in job(name)["rules"]] == [
            "$CI_COMMIT_TAG =~ /^v/",
            '$CI_PIPELINE_SOURCE == "web" && $RELEASE_RUN == "true"',
        ]
    assert job("windows")["needs"] == [] and job("windows")["timeout"] == "45 minutes"
    assert job("windows")["variables"]["GIT_DEPTH"] == "0"  # git describe needs the tags
    assert job("release-notes")["needs"] == ["windows"]
    assert job("release")["needs"] == ["windows", "release-notes"]


def test_release_waits_for_a_click_unless_a_manual_run_says_otherwise():
    rules = job("release")["rules"]
    assert 'RELEASE_DRAFT == "false"' in rules[0]["if"] and '"web"' in rules[0]["if"]
    assert "when" not in rules[0]
    assert [r["when"] for r in rules[1:]] == ["manual", "manual"]
    assert rules[1]["if"] == "$CI_COMMIT_TAG =~ /^v/"  # a tag push always waits
    assert "allow_failure" not in job("release")  # so the pipeline is blocked, not passed


def test_windows_job_carries_every_github_verification_step():
    s = text("windows")
    script = job("windows")["script"]
    assert "python scripts/build_exe.py --maps --zip --installer --map-jobs 4" in script
    assert "WindowsBuiltInRole]::Administrator" in script[0]  # checked before the long build
    order = [
        "choco install innosetup",
        "scripts/build_exe.py --maps --zip --installer",
        "New-NetFirewallRule",
        "https://example.com",
        "RobotCellOptimizer.exe --selftest",
        "packaging\\ci_serve_check.ps1",
        "Remove-NetFirewallRule -DisplayName",
        "/VERYSILENT",
        "& $exe --selftest",
        "unins*.exe",
        "SHA256SUMS.txt",
    ]
    at = [s.index(x, s.index("New-NetFirewallRule") if x.startswith("Remove") else 0) for x in order]
    assert at == sorted(at), list(zip(order, at, strict=True))
    for needle in (
        "-Direction Outbound",
        "-Action Block",
        "-RemoteAddress Internet",
        "-Port 8765",
        "finally",
    ):
        assert needle in s
    for flag in ("/VERYSILENT", "/CURRENTUSER", "/SUPPRESSMSGBOXES", "/NORESTART", "/NOCANCEL"):
        assert flag in s
    assert s.count('$env:RSO_OFFLINE_GUARD = "1"') == 2 and s.count("$env:RSO_DATA = ") == 3
    assert "Remove-NetFirewallRule" in text("windows", "after_script")
    # the PowerShell that both hosts share is still the same text in the GitHub workflow
    for line in ("New-NetFirewallRule -DisplayName", '$dir = Join-Path $env:LOCALAPPDATA "Programs'):
        assert line in GH_RELEASE and line in s
    assert (ROOT / "packaging" / "ci_serve_check.ps1").is_file()


def test_artifacts_are_the_three_release_files():
    art = job("windows")["artifacts"]
    gh = yaml.safe_load(GH_RELEASE)["jobs"]["windows"]["steps"]
    up = [st for st in gh if "upload-artifact" in st.get("uses", "")][0]["with"]
    assert art["paths"] == up["path"].split() and art["name"] == up["name"]
    assert job("release-notes")["artifacts"]["paths"] == ["notes.md"]


def test_release_scripts_exist_and_take_the_flags_the_pipeline_passes(tmp_path, capsys):
    notes, pub = text("release-notes"), text("release")
    assert "RELEASE_MAX_FILE_BYTES" in notes and "exit 1" in notes
    assert 'python3 scripts/gitlab_release.py changes --tag "$TAG" --out changes.md' in notes
    assert (
        'python3 scripts/release_notes.py dist --version "$TAG" --changes changes.md --out notes.md' in notes
    )
    assert 'python3 scripts/gitlab_release.py publish dist --tag "$TAG" --notes notes.md' in pub
    for s in (notes, pub):
        assert 'TAG="${CI_COMMIT_TAG:-manual-$CI_PIPELINE_IID}"' in s
    gl = _load("gl_cli")
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "notes.md").write_text("n", encoding="utf-8")
    out = tmp_path / "changes.md"
    gl.main(["changes", "--tag", "v9", "--out", str(out)])
    assert out.read_text(encoding="utf-8").startswith("- ")
    for f in (
        "scripts/release_notes.py",
        "scripts/build_exe.py",
        "scripts/build_guide.py",
        "scripts/test_batches.py",
    ):
        assert (ROOT / f).is_file()


ENV = {
    "CI_API_V4_URL": "https://gitlab.example/api/v4",
    "CI_PROJECT_ID": "42",
    "CI_JOB_TOKEN": "job-token",
    "CI_COMMIT_SHA": "abc123",
}


def _folder(tmp_path):
    for n in ("RobotCellOptimizer-v1.2.3-setup.exe", "RobotCellOptimizer-v1.2.3-win64.zip", "SHA256SUMS.txt"):
        (tmp_path / n).write_bytes(b"data-" + n.encode())
    return tmp_path


def test_publish_uploads_each_file_then_creates_the_release(tmp_path):
    gl = _load("gl_pub")
    calls = []

    def send(method, url, headers, data=None, timeout=0):
        calls.append((method, url, headers, data if isinstance(data, bytes) else data.read()))
        return b"{}"

    body = gl.publish(_folder(tmp_path), "v1.2.3", "# notes", {**ENV, "CI_COMMIT_TAG": "v1.2.3"}, send=send)
    assert [c[0] for c in calls] == ["PUT", "PUT", "PUT", "POST"]
    base = "https://gitlab.example/api/v4/projects/42/packages/generic/RobotCellOptimizer/v1.2.3/"
    assert [c[1] for c in calls[:3]] == [base + f.name for f in sorted(tmp_path.iterdir())]
    for (_, _, head, data), f in zip(calls[:3], sorted(tmp_path.iterdir()), strict=True):
        assert head["JOB-TOKEN"] == "job-token" and head["Content-Length"] == str(f.stat().st_size)
        assert data == f.read_bytes()
    method, url, head, data = calls[3]
    assert url == "https://gitlab.example/api/v4/projects/42/releases" and head["JOB-TOKEN"] == "job-token"
    assert json.loads(data) == body
    assert body["tag_name"] == "v1.2.3" and body["name"] == "Robot Cell Optimizer v1.2.3"
    assert body["description"] == "# notes" and "ref" not in body  # the tag exists: nothing is created
    links = body["assets"]["links"]
    assert [ln["name"] for ln in links] == [f.name for f in sorted(tmp_path.iterdir())]
    assert all(ln["link_type"] == "package" and ln["direct_asset_path"] == "/" + ln["name"] for ln in links)
    assert [ln["url"] for ln in links] == [c[1] for c in calls[:3]]


def test_publish_of_a_manual_run_creates_the_tag_and_marks_a_prerelease(tmp_path):
    gl = _load("gl_manual")
    sent = []
    env = {**ENV, "RELEASE_TOKEN": "pat", "RELEASE_PACKAGE": "pkg"}
    body = gl.publish(_folder(tmp_path), "manual-7", "n", env, send=lambda *a, **k: sent.append(a) or b"")
    assert body["ref"] == "abc123" and body["name"] == "Robot Cell Optimizer manual-7 (pre-release)"
    assert "/packages/generic/pkg/manual-7/" in sent[0][1]
    assert sent[0][2]["PRIVATE-TOKEN"] == "pat" and "JOB-TOKEN" not in sent[0][2]


def test_publish_dry_run_and_clear_failures(tmp_path):
    gl = _load("gl_fail")

    def boom(*a, **k):
        raise AssertionError("a dry run must not send")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no files"):
        gl.publish(empty, "v1", "n", ENV, send=boom)
    (empty / "f.bin").write_bytes(b"1")
    assert (
        gl.publish(empty, "v1", "n", {**ENV, "CI_JOB_TOKEN": ""}, dry_run=True, send=boom)["tag_name"] == "v1"
    )
    with pytest.raises(SystemExit, match="no token"):
        gl.publish(empty, "v1", "n", {**ENV, "CI_JOB_TOKEN": ""}, send=boom)
    with pytest.raises(SystemExit, match="CI_PROJECT_ID"):
        gl.publish(empty, "v1", "n", {"CI_API_V4_URL": "x"}, send=boom)


def test_changes_lists_commits_since_the_previous_tag(monkeypatch):
    gl = _load("gl_changes")
    asked = []

    def fake_git(*args):
        asked.append(args)
        return "v1.0.0" if args[0] == "describe" else "- fix a thing (abc1234)\n- add b (def5678)"

    monkeypatch.setattr(gl, "git", fake_git)
    out = gl.changes("v1.1.0", "https://gitlab.example/g/p/")
    assert out == (
        "- fix a thing (abc1234)\n- add b (def5678)\n\n"
        "**Full changes**: https://gitlab.example/g/p/-/compare/v1.0.0...v1.1.0\n"
    )
    assert asked[0][-1] == "HEAD^" and asked[1][-1] == "v1.0.0..HEAD" and "--no-merges" in asked[1]
    # first release (no earlier tag) or no git: still valid Markdown, no link
    monkeypatch.setattr(gl, "git", lambda *a: None)
    assert gl.changes("v1", "https://x") == "- (no commits listed)\n"
