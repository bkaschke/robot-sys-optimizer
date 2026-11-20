"""The status line shown under the playback slider (split out of simulate.py)."""

from __future__ import annotations


def status_text(p) -> str:
    """One line for the path sample ``p``: stop, zone, what the robot does, progress, issue."""
    zone = "" if p.zone is None else f", zone {p.zone + 1}"
    flag = f" · **{p.issue.replace('_', ' ')}**" if p.issue else ""
    ends = ("home" if p.a == 0 else p.a, "home" if p.b == 0 else p.b)
    what = {
        "carrier": "carrier moving, arm parked",
        "home": f"{ends[0]} → {ends[1]} (joint move)",
        "retreat": f"backing off point {p.a}",
        "lift": f"point {p.a} → {p.b} (lifted)",
    }.get(p.kind, f"point {p.a} → {p.b}")
    return f"Stop S{p.station + 1}{zone}: {what} ({100 * p.s:.0f} %){flag}"
