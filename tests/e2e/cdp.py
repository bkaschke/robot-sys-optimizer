"""Minimal Chrome DevTools Protocol driver for headless Edge (v8 T4 / T20 browser tests).

    with Browser("http://127.0.0.1:8080/") as b:
        b.eval("document.title")
        b.click(x, y, modifiers=8)   # alt 1, ctrl 2, meta 4, shift 8

Synchronous wrapper over ``websockets`` (asyncio in a private loop). Edge is found in its
usual Windows locations (or ``RSO_EDGE``); ``available()`` is False when it isn't installed,
so the tests skip.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

EDGE_PATHS = (
    os.environ.get("RSO_EDGE", ""),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)


def edge() -> str | None:
    return next((p for p in EDGE_PATHS if p and Path(p).is_file()), None)


def available() -> bool:
    try:
        import websockets  # noqa: F401
    except ImportError:
        return False
    return edge() is not None


class Browser:
    def __init__(self, url: str, port: int = 9333, size: tuple[int, int] = (1400, 900)):
        self.url, self.port, self.size = url, port, size
        self._loop = asyncio.new_event_loop()
        self._id = 0

    def __enter__(self) -> Browser:
        self._prof = tempfile.mkdtemp()
        self.proc = subprocess.Popen(
            [
                edge(),
                "--headless=new",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
                f"--remote-debugging-port={self.port}",
                f"--user-data-dir={self._prof}",
                f"--window-size={self.size[0]},{self.size[1]}",
                self.url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        page = None
        for _ in range(80):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json").read())
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.25)
        if page is None:
            raise RuntimeError("headless Edge did not start")
        import websockets

        self.ws = self._run(websockets.connect(page["webSocketDebuggerUrl"], max_size=None))
        return self

    def __exit__(self, *exc) -> None:
        try:
            self._run(self.ws.close())
        except Exception:
            pass
        self.proc.kill()
        self._loop.close()

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def cdp(self, method: str, **params) -> dict:
        async def call():
            self._id += 1
            await self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
            while True:
                msg = json.loads(await self.ws.recv())
                if msg.get("id") == self._id:
                    if "error" in msg:
                        raise RuntimeError(msg["error"])
                    return msg.get("result", {})

        return self._run(call())

    def eval(self, expression: str):
        r = self.cdp("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"])
        return r["result"].get("value")

    def wait_for(self, expression: str, timeout: float = 30.0):
        t_end = time.time() + timeout
        while time.time() < t_end:
            v = self.eval(expression)
            if v:
                return v
            time.sleep(0.25)
        raise TimeoutError(expression)

    def key(self, key: str, modifiers: int = 0) -> None:
        for typ in ("keyDown", "keyUp"):
            self.cdp("Input.dispatchKeyEvent", type=typ, key=key, code=key, modifiers=modifiers)

    def click(self, x: float, y: float, modifiers: int = 0) -> None:
        for typ in ("mouseMoved", "mousePressed", "mouseReleased"):
            self.cdp(
                "Input.dispatchMouseEvent",
                type=typ,
                x=x,
                y=y,
                button="left",
                clickCount=1,
                modifiers=modifiers,
            )
            time.sleep(0.05)

    def drag(
        self, a: tuple[float, float], b: tuple[float, float], steps: int = 12, modifiers: int = 0
    ) -> None:
        self.cdp("Input.dispatchMouseEvent", type="mouseMoved", x=a[0], y=a[1], modifiers=modifiers)
        self.cdp(
            "Input.dispatchMouseEvent",
            type="mousePressed",
            x=a[0],
            y=a[1],
            button="left",
            clickCount=1,
            modifiers=modifiers,
        )
        for k in range(1, steps + 1):
            x, y = a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps
            self.cdp(
                "Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y, button="left", modifiers=modifiers
            )
            time.sleep(0.03)
        self.cdp(
            "Input.dispatchMouseEvent",
            type="mouseReleased",
            x=b[0],
            y=b[1],
            button="left",
            clickCount=1,
            modifiers=modifiers,
        )
