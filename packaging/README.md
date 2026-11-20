# Windows executable (PyInstaller, onedir)

Files: `rso.spec` (PyInstaller spec), `launch.py` (entry point), `RobotCellOptimizer.iss` (Inno Setup),
`../scripts/build_exe.py` (the build). Nothing is downloaded at install or run time.

## Build locally

```
PY=<path to the rso env>/python.exe
$PY scripts/build_exe.py --zip                   # build + selftest + dist/RobotCellOptimizer-<version>-win64.zip
$PY scripts/build_exe.py --maps --zip --installer  # also prebuild reachability maps; installer needs Inno Setup (iscc)
```

Options: `--skip-selftest`, `--no-cad` (without CadQuery, about 500 MB smaller, no STEP import/export),
`--any-env` (skip the check that Python is the `rso` env). The script must run in the `rso` env (the same
conda env the tests use, so the native libraries are the tested ones). It sets `PYTHONNOUSERSITE=1` (packages
in the user site-packages must not leak into the bundle) and puts `<env>\Library\bin` on PATH. Output:
`dist/RobotCellOptimizer/RobotCellOptimizer.exe`, `dist/SHA256SUMS.txt`, the zip and/or setup exe.
The git version goes in a generated `_version.py` (the exe has no git); `core.provenance` reads it.

## Selftest

`RobotCellOptimizer.exe --selftest` runs headless and exits 0 or 1. The build runs it with `RSO_DATA` set to a
temp folder and `RSO_OFFLINE_GUARD=1`, and fails when it fails. It checks: the Viser web client is served
(loopback HTTP, proxy ignored); the example `mobile_amr` loads; a quick plan runs in the worker process
(spawn workers and the check pool work frozen); the HTML report is written; STEP export works (or reports "not
included in this build"); numba compiles the CRX offset-wrist IK from the bundle.

## Offline guard

`RSO_OFFLINE_GUARD=1` makes any non-loopback socket connect or DNS lookup in Python (main process and spawn
workers) raise and print `[offline-guard] blocked ...` on stderr. Loopback (127.x, ::1, localhost) is allowed.
It does not cover a browser or native code that opens its own sockets, so CI also uses a firewall rule.
Other variables: `--no-browser` (do not open a browser), `RSO_SELFTEST_PORT` (selftest server port, default
free port), `RSO_DATA` (data folder; default `%LOCALAPPDATA%\RobotCellOptimizer`).

## Installer

Per-user (`PrivilegesRequired=lowest`), into `%LOCALAPPDATA%\Programs\RobotCellOptimizer`, Start-menu entry,
uninstaller. The data folder `%LOCALAPPDATA%\RobotCellOptimizer` (projects, caches, tools) is NOT removed on
uninstall. Unsigned: Windows SmartScreen shows a warning once.

## Why onedir

D0 spike numbers (CadQuery included): 1,011 MB on disk, 2,135 files, 321 MB zipped; CadQuery is about 495 MB
of that (mostly VTK). Server answers in 1.1 s, GUI ready in about 11 s (same as from source). Selftest 55 s cold
(reachability map and numba compile built), 18.5 s warm. One-file mode was rejected: it unpacks for about 5 s
on every start, leaves about 1 GB in `%TEMP%` when killed (and managed PCs often block running from `%TEMP%`),
and recompiles numba every start. numba needs `module_collection_mode={"core.ik.offset_wrist_nb": "py"}` so its
cache uses `NUMBA_CACHE_DIR` (set by `core.data_paths.configure_numba_cache`).

## Releases from CI (`.github/workflows/release.yml`, `.gitlab/ci/release.yml`)

The repository carries the same release pipeline for GitHub Actions and for GitLab CI. Each host builds and
publishes its own release; the build, the checks and the three files are the same. GitHub first, then
[GitLab](#gitlab).

Cut a release (GitHub):
1. Merge to the branch you want to ship, then `git tag vX.Y.Z` and `git push origin vX.Y.Z`. (Or Actions ->
   release -> Run workflow; its `draft` input is ticked by default.)
2. The `windows` job builds with `scripts/build_exe.py --maps --zip --installer --map-jobs 4` (prebuilt reachability maps,
   zip, Inno Setup installer, SHA256SUMS.txt). The version comes from `git describe`, so the tag is the version.
3. The `release` job creates a **draft** GitHub release `vX.Y.Z` with the zip, the setup exe and SHA256SUMS.txt
   (it refuses files over GitHub's 2 GB per-file limit; use `--no-cad` if that ever happens).
4. Open the draft on GitHub, check the generated notes and the files, then **Publish**. Nothing is published
   automatically. A manual run on a branch makes a prerelease draft named `manual-<run number>`.

What CI checks before the draft exists (any failure stops the release):
- the build's own `--selftest` (headless plan, report, STEP, numba, bundled web client);
- with outbound network blocked by a Windows firewall rule and `RSO_OFFLINE_GUARD=1`: `--selftest` again, then
  the exe is started with `--no-browser --port 8765`, `packaging/ci_serve_check.ps1` waits for the "Robot Cell
  Optimizer ready:" line, loads the page in headless Edge and checks that the Viser client rendered (root
  populated and a canvas); the rule is removed in an `always()` step. A hidden download fails the build, not a customer;
- the installer is installed silently per user (`/VERYSILENT /CURRENTUSER /SUPPRESSMSGBOXES`), the installed
  exe's selftest runs, and it is uninstalled silently.

Run the serve check locally (not the firewall part, which needs admin) with
`powershell -File packaging\ci_serve_check.ps1 -Exe dist\RobotCellOptimizer\RobotCellOptimizer.exe`
(about 3 min; Edge in swiftshader mode is slow).

Linux AppImage: not built yet (decision: after the first Windows release); a commented placeholder job is in the workflow.

### GitLab

`.gitlab-ci.yml` includes `.gitlab/ci/setup.yml` (conda env, caches), `tests.yml` (lint, guide check, fast
tests on Windows and Linux for pushes, merge requests and tags) and `release.yml`. **It has not run on a
GitLab instance yet**; `tests/test_gitlab_ci.py` checks its structure against the GitHub workflows.

Cut a release (GitLab):
1. `git tag vX.Y.Z` and `git push <gitlab remote> vX.Y.Z`. (Or Build -> Pipelines -> New pipeline, pick the
   branch, set `RELEASE_RUN` to `true`; `RELEASE_DRAFT` is `true` by default.)
2. The `windows` job runs the same build and the same checks as on GitHub (list above) and keeps the zip, the
   setup exe and SHA256SUMS.txt as job artifacts (90 days).
3. The `release-notes` job checks the file sizes and writes `notes.md` (artifact): the template plus the commit
   subjects since the previous `v*` tag.
4. The pipeline now shows **blocked**. Download the artifacts, read `notes.md`, then start the `release` job
   (the play button). It uploads the three files to the package registry (Deploy -> Package registry, package
   `RobotCellOptimizer`, version = the tag) and creates the release `vX.Y.Z` with links to them. Nothing is
   published before that click. A manual run makes the tag `manual-<pipeline number>` at that commit.

Set up once in the GitLab project:
- **Runners.** The tags and images are the variables at the top of `.gitlab-ci.yml` (`RUNNER_TAG_WINDOWS`,
  `RUNNER_TAG_LINUX`, `LINUX_TEST_IMAGE`, `LINUX_RELEASE_IMAGE`, `WINDOWS_CI_DIR`); the defaults are
  gitlab.com's hosted runners. A self-hosted Windows runner needs: shell executor (powershell or pwsh), git,
  an **administrator** account (the firewall rule; the job stops at once with a message when it is not
  elevated), Microsoft Edge, and Inno Setup 6 or Chocolatey (missing tools are installed with `choco`). The
  release job blocks outbound traffic on that machine for a few minutes, so use a dedicated machine with one
  job at a time. Miniforge is installed into `WINDOWS_CI_DIR` when no conda is found. The Linux jobs need a
  docker executor.
- **Settings -> General -> Visibility:** Releases and Package registry enabled.
- **Settings -> Repository -> Protected tags:** protect `v*` (Maintainers may create), so only maintainers can
  start a release. Leave `manual-*` unprotected, or the manual run cannot create its tag.
- **Tokens:** none to create. The jobs use the job token (`CI_JOB_TOKEN`), which has the rights of the person
  who pushed the tag or started the pipeline (Developer or higher; for a protected tag, someone allowed to
  create it). Only if your instance refuses that: add a project access token (scope `api`) as the masked,
  protected CI/CD variable `RELEASE_TOKEN`.
- **Settings -> CI/CD -> General pipelines -> Timeout:** the `windows` job asks for 45 minutes; a runner's own
  maximum job timeout must allow that.
- **Limits:** job artifacts of one job must fit the instance limit (gitlab.com: 1 GB; the zip alone is
  about 0.3 GB with CadQuery, the setup exe is not measured yet). `RELEASE_MAX_FILE_BYTES` (default 5 GB, gitlab.com's package file limit) is
  the per-file check; use `--no-cad` if a file is ever too large.

Differences from GitHub:

| GitHub | GitLab |
| --- | --- |
| draft release, then **Publish** | no drafts: the `release` job waits for a click, then the release is public at once |
| `draft` input of a manual run | variable `RELEASE_DRAFT` (`false` = no waiting) |
| pre-release flag on manual runs | none: the release name ends with "(pre-release)" |
| generated notes (merged pull requests) | commit subjects since the previous `v*` tag and a compare link |
| files attached to the release | files in the generic package registry, linked from the release |
| 2 GB per release file | the package registry's limit (`RELEASE_MAX_FILE_BYTES`) |
| env via `setup-miniconda`, activated in bash | Miniforge installed in `before_script`; PowerShell with the env on PATH |
| fresh VM for every job | hosted runners: the same; self-hosted: the machine is reused (the job cleans up after itself) |
| push and pull-request runs side by side | one pipeline per change (merge request pipeline when one is open) |

To publish a release again after a failed `release` job, delete the release (and the package version) first:
GitLab refuses a second release for the same tag.

### SmartScreen (unsigned builds)

The exe and installer are not code-signed. On first run Windows SmartScreen may show "Windows protected your
PC": click **More info**, then **Run anyway**. Managed PCs may block unsigned programs: IT has to allow the
file (by the SHA-256 in SHA256SUMS.txt, or by path), or the build has to be signed later. The zip variant needs
no install and no admin rights; unblock it first (file Properties -> Unblock) if Windows marks the download.
