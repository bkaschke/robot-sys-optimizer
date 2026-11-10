"""Shared test helpers."""

from ui.staleness import STEP_COMPONENTS, confirm_step


def confirm_all(target) -> None:
    """Confirm setup steps 1-4 (the optimizers refuse to run before that, v7 round 2 T5).
    ``target`` is a Session or anything with one as ``.s`` (a panel)."""
    s = getattr(target, "s", target)
    for step in STEP_COMPONENTS:
        confirm_step(s, step)
