"""Entry point of the frozen Robot Cell Optimizer (PyInstaller; built by scripts/build_exe.py).

    RobotCellOptimizer.exe [--dark] [--port 8080] [--host ...] [--open FOLDER] [--no-browser]
    RobotCellOptimizer.exe --version        (print the version, exit 0; no server)
    RobotCellOptimizer.exe --selftest       (headless: plan one small job in the worker, export, exit 0/1)

Order matters:
1. numba's cache folder (core.data_paths; inherited by the spawn workers through the environment);
2. ``multiprocessing.freeze_support()``: a spawn worker branches off there and never returns;
3. the app (``app.main``) or the selftest.
Read-only resources and the per-user data folder come from ``core.data_paths`` (RSO_DATA overrides).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if not getattr(sys, "frozen", False):  # run as a script from a checkout: the repo is the parent folder
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import data_paths  # noqa: E402  (stdlib only: cheap in every spawn worker)

data_paths.configure_numba_cache()  # before anything imports numba
RES = data_paths.resources()
DATA = data_paths.data_dir()


def _offline_guard() -> None:
    """RSO_OFFLINE_GUARD=1 (selftest): any non-loopback connection or DNS lookup from Python
    raises and is reported on stderr (main process and spawn workers: the variable is inherited).
    It does not cover the browser or native code that opens its own sockets."""
    import socket

    local = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "", None}
    real_gai, real_connect = socket.getaddrinfo, socket.socket.connect

    def blocked(what: str):
        msg = f"[offline-guard] blocked {what} (pid {os.getpid()})"
        print(msg, file=sys.stderr, flush=True)
        return OSError(msg)

    def getaddrinfo(host, *a, **k):
        if host not in local and not str(host).startswith("127."):
            raise blocked(f"DNS lookup of {host!r}")
        return real_gai(host, *a, **k)

    def connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if (
            self.family in (socket.AF_INET, socket.AF_INET6)
            and host not in local
            and not str(host).startswith("127.")
        ):
            raise blocked(f"connect to {address!r}")
        return real_connect(self, address)

    socket.getaddrinfo, socket.socket.connect = getaddrinfo, connect


if os.environ.get("RSO_OFFLINE_GUARD") == "1":
    _offline_guard()


def selftest() -> int:
    """Load an example project, quick-plan it in the worker process, export report + STEP."""
    import shutil
    import tempfile
    import time
    import urllib.request

    t0 = time.monotonic()
    import viser

    from core.version import app_version

    print(f"[selftest] Robot Cell Optimizer {app_version()}")

    import app
    from ui import jobs

    jobs.USE_WORKER = True
    jobs.CHECK_WORKERS = 2  # the worker's own spawn pool (opt.parallel) for the exact checks
    DATA.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="selftest-", dir=DATA))
    proj = work / "mobile_amr"
    shutil.copytree(RES / "examples" / "mobile_amr", proj)
    server = viser.ViserServer(host="127.0.0.1", port=int(os.environ.get("RSO_SELFTEST_PORT", "0") or 0))
    ok = True
    try:
        if os.environ.get("RSO_OFFLINE_GUARD") == "1":
            import socket

            try:
                socket.getaddrinfo("example.com", 443)
                print("[selftest] offline guard NOT active")
                ok = False
            except OSError as err:
                print(f"[selftest] offline guard active ({'offline-guard' in str(err)})")
        port = server.get_port()
        no_proxy = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # HTTP_PROXY may be set
        html = no_proxy.open(f"http://127.0.0.1:{port}/", timeout=10).read()
        print(f"[selftest] viser page: {len(html)} bytes, viser in page: {b'viser' in html.lower()}")
        ok &= len(html) > 10_000
        state = app.build_app(server)
        s = state["session"]
        ok &= bool(state["project"]._load(None, proj))
        print(
            f"[selftest] loaded {proj.name}: {len(s.targets.points)} targets ({time.monotonic() - t0:.1f} s)"
        )
        st = state["stations"]
        from ui.workflow import setup_reason

        if reason := setup_reason(s):
            print(f"[selftest] setup not confirmed ({reason}); confirming steps")
            for step in ("Cell", "External Axis", "Tool Setup", "Targets"):
                state["workflow"].confirm(step)
        t1 = time.monotonic()
        st.start(quick=True)
        if st._thread is None:
            print(f"[selftest] plan did not start: {st.quick_status.content}")
            return 1
        st._thread.join(600)
        print(f"[selftest] status: {st.quick_status.content!r} ({time.monotonic() - t1:.1f} s)")
        print(f"[selftest] worker pid: {jobs.runner().pid} (this process {os.getpid()})")
        from ui import log_panel

        for _t, level, msg in log_panel.entries():  # a pool fallback or worker error shows here
            if level in ("warn", "error"):
                print(f"[selftest] log {level}: {msg.splitlines()[0][:200]}")
                ok &= "parallel" not in msg and "check worker" not in msg
        sols = [r for r in s.results if r.tool_values is None]
        ok &= bool(sols)
        if sols:
            sol = sols[0]
            ex = state["results"].export
            path, data = ex.export_report(sol)
            ext = [u for u in (b"http://", b"https://") if u in data]
            print(f"[selftest] report: {path} {len(data)} bytes; external URL markers: {ext}")
            ok &= len(data) > 1000
            import importlib.util

            if importlib.util.find_spec("cadquery") is None:  # a build without CadQuery (RSO_NO_CAD)
                try:
                    ex.export_step(sol)
                    print("[selftest] STEP: exported without CadQuery?")
                    ok = False
                except ValueError as err:  # the user-facing message
                    print(f"[selftest] STEP: not in this build, message: {err}")
            else:
                try:
                    path, data = ex.export_step(sol)
                    print(f"[selftest] STEP: {path} {len(data)} bytes")
                    ok &= data.startswith(b"ISO-10303-21")
                except Exception as err:
                    print(f"[selftest] STEP export failed: {type(err).__name__}: {err}")
                    ok = False
        # numba (offset-wrist IK, cached compile) in this process
        from core.ik import make_ik
        from core.robot import Robot

        t2 = time.monotonic()
        r = Robot.from_registry("crx10ia")
        ik = make_ik(r)
        import numpy as np

        sol_q = ik.solve(ik.fk(np.full(6, 0.3)))
        print(
            f"[selftest] numba IK crx10ia: {type(ik).__name__}, {len(sol_q)} solutions "
            f"({time.monotonic() - t2:.1f} s)"
        )
    except Exception:
        import traceback

        traceback.print_exc()
        ok = False
    finally:
        jobs.runner().stop()
        from opt.parallel import shutdown_all

        shutdown_all()
        server.stop()
        shutil.rmtree(work, ignore_errors=True)
    print(f"[selftest] {'PASS' if ok else 'FAIL'} in {time.monotonic() - t0:.1f} s; data in {DATA}")
    return 0 if ok else 1


def parse_args(argv: list[str]) -> tuple[str, bool, str, list[str]]:
    """(mode, open the browser?, port, args left for app.main). Mode: "version", "selftest" or "app"."""
    if "--version" in argv:
        return "version", False, "", []
    if "--selftest" in argv:
        return "selftest", False, "", []
    rest = [a for a in argv if a != "--no-browser"]
    port = "8080"
    for i, a in enumerate(rest):
        if a == "--port" and i + 1 < len(rest):
            port = rest[i + 1]
        elif a.startswith("--port="):
            port = a.split("=", 1)[1]
    return "app", "--no-browser" not in argv, port, rest


def main() -> None:
    mode, browser, port, rest = parse_args(sys.argv[1:])
    if mode == "version":
        from core.version import app_version

        print(f"Robot Cell Optimizer {app_version()}")
        sys.exit(0)
    if mode == "selftest":
        sys.exit(selftest())
    sys.argv = [sys.argv[0], *rest]
    if browser:
        import threading
        import webbrowser

        threading.Timer(4.0, lambda: webbrowser.open(f"http://localhost:{port}/")).start()
    import app

    app.main()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
