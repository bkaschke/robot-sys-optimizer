"""GitLab side of a release (.gitlab/ci/release.yml); GitHub uses ``gh release create`` instead.

    python scripts/gitlab_release.py changes [--tag TAG] [--out FILE]
    python scripts/gitlab_release.py publish <dir> --tag TAG --notes FILE [--dry-run]

``changes``: the commit subjects since the previous ``v*`` tag and a compare link, as Markdown (the
closest GitLab has to GitHub's generated release notes). ``publish``: uploads every file in ``<dir>``
to the project's generic package registry, then creates the release ``TAG`` with links to those files
(GitLab releases hold links, not files). A tag that does not exist yet is created at ``CI_COMMIT_SHA``
and the release is named a pre-release.

Standard library only (it runs in a plain python image). Reads the predefined CI variables
``CI_API_V4_URL``, ``CI_PROJECT_ID``, ``CI_JOB_TOKEN``, ``CI_COMMIT_SHA``, ``CI_COMMIT_TAG``,
``CI_PROJECT_URL``; ``RELEASE_PACKAGE`` names the package. Set ``RELEASE_TOKEN`` (a project access
token with the api scope) only if the job token may not create the tag or the release.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_COMMITS = 200
TITLE = "Robot Cell Optimizer"


def fail(msg: str) -> None:
    raise SystemExit(f"gitlab_release: {msg}")


def git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def previous_tag(ref: str = "HEAD") -> str | None:
    """The newest v* tag before ``ref`` (None for the first release or a shallow clone)."""
    return git("describe", "--tags", "--abbrev=0", "--match", "v*", f"{ref}^") or None


def changes(tag: str, project_url: str = "", ref: str = "HEAD") -> str:
    prev = previous_tag(ref)
    log = git(
        "log",
        "--no-merges",
        f"--max-count={MAX_COMMITS}",
        "--pretty=format:- %s (%h)",
        *([f"{prev}..{ref}"] if prev else [ref]),
    )
    lines = [ln for ln in (log or "").splitlines() if ln.strip()] or ["- (no commits listed)"]
    if len(lines) == MAX_COMMITS:
        lines.append(f"- ... (only the newest {MAX_COMMITS} commits are listed)")
    if prev and project_url:
        lines += ["", f"**Full changes**: {project_url.rstrip('/')}/-/compare/{prev}...{tag}"]
    return "\n".join(lines) + "\n"


def auth_header(env: dict[str, str]) -> dict[str, str]:
    if env.get("RELEASE_TOKEN"):
        return {"PRIVATE-TOKEN": env["RELEASE_TOKEN"]}
    if env.get("CI_JOB_TOKEN"):
        return {"JOB-TOKEN": env["CI_JOB_TOKEN"]}
    fail("no token: CI_JOB_TOKEN (set by GitLab CI) or RELEASE_TOKEN is needed")


def package_url(env: dict[str, str], tag: str, name: str) -> str:
    q = urllib.parse.quote
    package = env.get("RELEASE_PACKAGE") or "RobotCellOptimizer"
    return (
        f"{env['CI_API_V4_URL']}/projects/{env['CI_PROJECT_ID']}/packages/generic/"
        f"{q(package, safe='')}/{q(tag, safe='')}/{q(name, safe='')}"
    )


def release_payload(tag: str, notes: str, links: list[tuple[str, str]], ref: str | None) -> dict:
    """Body of POST /projects/:id/releases. ``ref`` set = the tag is new (a manual run)."""
    body = {
        "tag_name": tag,
        "name": f"{TITLE} {tag}" + (" (pre-release)" if ref else ""),
        "description": notes,
        "assets": {
            "links": [
                {"name": n, "url": u, "direct_asset_path": f"/{n}", "link_type": "package"} for n, u in links
            ]
        },
    }
    if ref:
        body["ref"] = ref
    return body


def send(method: str, url: str, headers: dict[str, str], data=None, timeout: float = 3600) -> bytes:
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        fail(f"{method} {url.split('?')[0]} -> HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        fail(f"{method} {url.split('?')[0]} failed: {e.reason}")


def publish(
    folder: Path, tag: str, notes: str, env: dict[str, str], dry_run: bool = False, send=send
) -> dict:
    for key in ("CI_API_V4_URL", "CI_PROJECT_ID"):
        if not env.get(key):
            fail(f"{key} is not set (this runs in GitLab CI)")
    files = sorted(f for f in folder.iterdir() if f.is_file())
    if not files:
        fail(f"no files in {folder}")
    head = {} if dry_run else auth_header(env)
    links = []
    for f in files:
        url = package_url(env, tag, f.name)
        size = f.stat().st_size
        print(f"upload {f.name} ({size} bytes) -> {url}", flush=True)
        if not dry_run:
            with f.open("rb") as fh:  # streamed: Content-Length is given, so the file is not read into memory
                send(
                    "PUT",
                    url,
                    {**head, "Content-Length": str(size), "Content-Type": "application/octet-stream"},
                    fh,
                )
        links.append((f.name, url))
    # the tag exists on a tag pipeline; a manual run creates it at the pipeline's commit
    ref = (
        None
        if env.get("CI_COMMIT_TAG") == tag
        else env.get("CI_COMMIT_SHA") or fail("CI_COMMIT_SHA is not set")
    )
    body = release_payload(tag, notes, links, ref)
    url = f"{env['CI_API_V4_URL']}/projects/{env['CI_PROJECT_ID']}/releases"
    print(f"create release {body['name']!r}" + (f" (new tag at {ref})" if ref else ""), flush=True)
    if not dry_run:
        send("POST", url, {**head, "Content-Type": "application/json"}, json.dumps(body).encode("utf-8"))
        if env.get("CI_PROJECT_URL"):
            print(f"release: {env['CI_PROJECT_URL']}/-/releases/{urllib.parse.quote(tag, safe='')}")
    return body


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("changes", help="Markdown change list since the previous v* tag")
    c.add_argument(
        "--tag", default=None, help="name used in the compare link (default: CI_COMMIT_TAG or HEAD)"
    )
    c.add_argument("--out", type=Path, default=None)
    p = sub.add_parser("publish", help="upload the files and create the release")
    p.add_argument("dir", type=Path)
    p.add_argument("--tag", required=True)
    p.add_argument("--notes", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true", help="print what would be sent, send nothing")
    a = ap.parse_args(argv)
    env = dict(os.environ)
    if a.cmd == "changes":
        text = changes(a.tag or env.get("CI_COMMIT_TAG") or "HEAD", env.get("CI_PROJECT_URL", ""))
        if a.out:
            a.out.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
    else:
        if not a.dir.is_dir():
            fail(f"{a.dir} is not a folder")
        publish(a.dir, a.tag, a.notes.read_text(encoding="utf-8"), env, a.dry_run)


if __name__ == "__main__":
    main()
