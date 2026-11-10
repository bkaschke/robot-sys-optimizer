"""Process-type presets (plan § End-effector library). Starting values only: every field
stays editable per tool and per target group."""

from __future__ import annotations

from dataclasses import dataclass

from core.schema import EndEffector, ProcessType, Tolerances


@dataclass(frozen=True)
class ProcessPreset:
    label: str
    contact: str  # "none": whole tool must stay clear of the part; "tcp": contact faces may touch
    tolerances: Tolerances
    line_of_sight: bool = False
    approach_m: float = 0.0
    notes: str = ""


PRESETS: dict[str, ProcessPreset] = {
    "scanning": ProcessPreset(
        "Scanning / inspection",
        "none",
        Tolerances(spin_deg="free", tilt_deg=3.0, standoff_m=(0.15, 0.25)),
        line_of_sight=True,
        notes="Standoff = scanner working range; tilt from the scanner spec.",
    ),
    "sanding": ProcessPreset(
        "Sanding / polishing",
        "tcp",
        Tolerances(spin_deg="free", tilt_deg=10.0, standoff_m=0.0),
        notes="Pad only touches the part.",
    ),
    "grinding": ProcessPreset(
        "Grinding / deburring",
        "tcp",
        Tolerances(spin_deg=0.0, tilt_deg=15.0, standoff_m=0.0),
        notes="Spin fixed to the travel direction; edge targets.",
    ),
    "gripping": ProcessPreset(
        "Gripping / pick",
        "tcp",
        Tolerances(spin_deg=0.0, tilt_deg=0.0, standoff_m=0.0),
        approach_m=0.1,
        notes="Approach and retreat poses must also be reachable.",
    ),
    "spraying": ProcessPreset(
        "Spraying / dispensing",
        "none",
        Tolerances(spin_deg="free", tilt_deg=5.0, standoff_m=0.2),
        notes="Fixed nozzle distance; keep the spray cone clear.",
    ),
    "custom": ProcessPreset("Custom", "tcp", Tolerances()),
}


def apply_preset(ee: EndEffector, process: ProcessType) -> EndEffector:
    """Copy of ``ee`` with the preset's defaults for ``process`` applied.

    For contact-free processes the TCP contact sets are cleared, because no part of the
    tool may touch the part.
    """
    p = PRESETS[process]
    tcps = ee.tcps
    if p.contact == "none":
        tcps = [t.model_copy(update={"contact_links": [], "contact_faces": {}}) for t in ee.tcps]
    return ee.model_copy(
        update={
            "process": process,
            "tolerances": p.tolerances,
            "line_of_sight": p.line_of_sight,
            "approach_m": p.approach_m,
            "tcps": tcps,
        }
    )
