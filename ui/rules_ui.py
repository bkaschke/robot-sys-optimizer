"""Station planner -> *Mode* **Rules only** (v7 round 2, T15): widgets and the planning run.

Cost based (the default) is the planner as before. Rules only hides *Costs & requirements*
and shows the *Objective* (Fewest stations / Fewest groups), the *Singularity limit* (% of the
robot's max manipulability) and the fixed list of rules; the plan comes from opt.rules_plan.
"""

from __future__ import annotations

from dataclasses import replace

from opt.placement import plan_stations
from opt.rules_plan import OBJECTIVES, RULES, plan_rules, rule_counts, rule_texts

MODES = ("Cost based", "Rules only")


def build(gui, panel) -> None:
    panel.mode = gui.add_dropdown("Mode", MODES, hint="Rules only: no costs, every rule must hold")
    panel.rules_folder = gui.add_folder("Rules", visible=False)
    with panel.rules_folder:
        panel.objective = gui.add_dropdown("Objective", OBJECTIVES)
        panel.sing_pct = gui.add_number(
            "Singularity limit (% of max manipulability)",
            5.0,
            min=0.0,
            max=50.0,
            step=1.0,
            hint="arm poses with a lower manipulability (|det J|) do not count as reaching a point",
        )
        gui.add_markdown(
            "_All rules must hold:_  \n" + "  \n".join(f"{k}. {text}" for k, text in RULES.items())
        )
    panel.mode.on_update(lambda e: show(panel))


def rules_mode(panel) -> bool:
    return panel.mode.value == MODES[1]


def show(panel) -> None:
    on = rules_mode(panel)
    panel.rules_folder.visible = on
    panel.costs.folder.visible = not on
    panel.refine.visible = not on  # rules mode never refines (no cost to optimize)
    panel.zoning.visible = not on  # rule 5: zones always on
    panel.zoning_note.visible = not on and panel.zoning.value == panel.zoning.options[1]


def run(panel, rmap, checker, args: dict, streamed, say, cancelled) -> tuple[list, str]:
    """The rules-mode plan (in the planner thread). Returns ([(Solution, results)] or [],
    status markdown)."""
    objective = panel.objective.value
    checker.settings.min_manip = float(panel.sing_pct.value) / 100.0 * float(rmap.max_manip)
    base = args["settings"]

    def run_k(k: int, exclude: tuple | None):
        st = replace(
            base,
            first_k=k,
            max_stations=k,
            refine=False,
            zoning=True,
            require_all=True,
            require_one_config=True,
            patience=0,
            exclude=exclude or (),
            skip_below_min=exclude is not None,
        )
        out = plan_stations(
            rmap,
            checker,
            args["t"],
            args["T_base0"],
            args["fp"],
            args["weights"],
            st,
            args["L"],
            args["L_ref"],
            cancelled=cancelled,
            sites=args["sites"],
        )
        return out[0] if out else None

    res = plan_rules(
        run_k,
        objective,
        base.max_stations,
        patience=base.patience,
        cancelled=cancelled,
        say=lambda msg: say(msg, 0.5),
    )
    if res.best is None:
        return [], "_No plan found (rules only)._"
    sol, results = res.best
    counts = rule_counts(sol, results)
    sol.violations = rule_texts(counts, len(sol.stations))
    sol.planner = f"rules: {objective.lower()}"
    streamed(sol, results)
    if res.passed:
        status = (
            f"**Rules: all met.** Best ({objective.lower()}): **{sol.label}**, "
            f"{len(sol.zones)} group(s), {len(res.tried)} plan(s) tried."
        )
    else:
        status = (
            "**✘ No plan meets every rule.** Closest: **"
            + sol.label
            + "**:  \n"
            + "  \n".join(sol.violations)
        )
    return [res.best], status
