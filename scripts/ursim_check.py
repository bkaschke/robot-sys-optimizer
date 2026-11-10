"""Check an exported plan in URSim (v8 T19): does each stop's motion run to the end?

    python scripts/ursim_check.py <project>/export/tcp_poses.csv --host 127.0.0.1
    python scripts/ursim_check.py <project>/export/tcp_poses.csv --dry-run

From the TCP poses CSV (Results -> Export -> TCP poses) it writes one URScript program per stop
(``<csv folder>/ursim/stop_<k>.script``): a ``movej`` to every reached point of the stop in the
planned order, with the planned joint values (so the robot takes the planned configuration).
Then, per stop, it sends the program to URSim's secondary interface (port 30002), polls the
dashboard server (port 29999) until the program has stopped, and reads the safety status: a
protective stop (joint limit, self-collision in URSim's model, singularity during a move) fails
the stop. The carrier moves between stops are not part of it (move the URSim base yourself, or
run one stop at a time with ``--stop k``).

URSim: Universal Robots' simulator (e.g. the ``universalrobots/ursim_e-series`` Docker image,
ports 29999, 30002 published). ``--dry-run`` only writes the programs (no URSim needed).
"""

from __future__ import annotations

import argparse
import csv
import socket
import sys
import time
from pathlib import Path


def read_stops(csv_path: Path) -> dict[int, list[list[float]]]:
    """{stop: [q (deg) per reached point, in CSV order]}."""
    stops: dict[int, list[list[float]]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "unreachable" or not row.get("q1_deg"):
                continue
            k = int(float(row["station"]))
            stops.setdefault(k, []).append([float(row[f"q{i}_deg"]) for i in range(1, 7)])
    return stops


def urscript(name: str, qs: list[list[float]], speed: float, acc: float) -> str:
    lines = [f"def {name}():"]
    for q in qs:
        rad = ", ".join(f"d2r({v:.4f})" for v in q)
        lines.append(f"  movej([{rad}], a={acc}, v={speed})")
    lines.append("end")
    return "\n".join(lines) + "\n"


def dashboard(host: str, cmd: str, timeout: float = 5.0) -> str:
    with socket.create_connection((host, 29999), timeout=timeout) as s:
        s.recv(1024)  # greeting
        s.sendall((cmd + "\n").encode())
        return s.recv(1024).decode(errors="replace").strip()


def run_stop(host: str, script: str, timeout: float) -> tuple[bool, str]:
    dashboard(host, "unlock protective stop")
    with socket.create_connection((host, 30002), timeout=5.0) as s:
        s.sendall(script.encode())
    time.sleep(1.0)
    t_end = time.time() + timeout
    while time.time() < t_end:
        state = dashboard(host, "programState")
        safety = dashboard(host, "safetystatus")
        if "NORMAL" not in safety.upper():
            return False, f"safety status: {safety}"
        if state.upper().startswith("STOPPED"):
            return True, "ran to the end"
        time.sleep(0.5)
    return False, f"still running after {timeout:.0f} s"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", type=Path)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--speed", type=float, default=1.05, help="movej speed (rad/s)")
    ap.add_argument("--acc", type=float, default=1.4, help="movej acceleration (rad/s^2)")
    ap.add_argument("--stop", type=int, default=None, help="only this stop (as numbered in the CSV, 1-based)")
    ap.add_argument("--timeout", type=float, default=300.0, help="per stop (s)")
    ap.add_argument("--dry-run", action="store_true", help="write the programs only")
    a = ap.parse_args(argv)
    stops = read_stops(a.csv)
    if not stops:
        print("no reached points in the CSV")
        return 1
    out = a.csv.parent / "ursim"
    out.mkdir(exist_ok=True)
    failed = 0
    for k in sorted(stops):
        if a.stop is not None and k != a.stop:
            continue
        script = urscript(f"stop_{k}", stops[k], a.speed, a.acc)
        path = out / f"stop_{k}.script"
        path.write_text(script, encoding="utf-8")
        if a.dry_run:
            print(f"stop {k}: {len(stops[k])} points -> {path}")
            continue
        try:
            ok, msg = run_stop(a.host, script, a.timeout)
        except OSError as err:
            print(f"stop {k}: cannot reach URSim at {a.host} ({err})")
            return 2
        failed += not ok
        print(f"stop {k}: {'OK' if ok else 'FAILED'} ({msg})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
