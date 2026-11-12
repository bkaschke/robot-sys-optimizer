"""What a collision hits, in words (split from core.collision, which re-exports these names):
``describe_hits`` turns ``CollisionScene.colliding_pairs`` into "part (tool, forearm); floor
(upper_arm)"."""

from __future__ import annotations

import re

HITS = "collides with: "  # PointResult.notes prefix of ``describe_hits``


def _mover(name: str) -> str:
    """A moving geometry's short name: tool, hose, or the robot link (``forearm_link_0`` ->
    ``forearm``)."""
    if name.startswith("tool_"):
        return "tool"
    if name.startswith("hose_"):
        return "hose"
    base = re.sub(r"_\d+$", "", name)
    return re.sub(r"_link$", "", base)


def _obstacle(name: str) -> str | None:
    if name in ("part", "floor"):
        return name
    if name.startswith("carrier_"):
        return f"carrier {name[len('carrier_') :]}"
    if name.startswith("stayout_"):
        return f"stay-out zone {name[len('stayout_') :]}"
    return None


def describe_hits(pairs: list[tuple[str, str]]) -> str:
    """``HITS`` + what collides, from ``CollisionScene.colliding_pairs``: "part (tool,
    forearm); floor (upper_arm)", "arm itself (...)", "tool vs arm (...)"; "" for no pairs."""
    hit: dict[str, list[str]] = {}
    for a, b in pairs:
        oa, ob = _obstacle(a), _obstacle(b)
        if oa or ob:
            key, movers = (oa, [b]) if oa else (ob, [a])
        elif a.startswith("tool_") or b.startswith("tool_"):
            key, movers = "tool vs arm", [b if a.startswith("tool_") else a]
        else:
            key, movers = "arm itself", [a, b]
        lst = hit.setdefault(key, [])
        lst += [m for m in map(_mover, movers) if m not in lst]
    if not hit:
        return ""
    return HITS + "; ".join(f"{k} ({', '.join(v)})" for k, v in hit.items())
