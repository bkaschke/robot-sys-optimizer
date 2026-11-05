"""Project-file models for the work state: where the user left off (joint pose, selection,
feasibility and plan joint solutions, panel settings). Kept apart from core.schema to keep
that file small; core.schema re-exports both models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PointResultData(_Model):
    """A saved core.feasibility.PointResult (joint solution + quality of one target)."""

    id: int
    status: Literal["reachable", "marginal", "unreachable"]
    reason: str = ""
    q: list[float] | None = None
    tcp: str | None = None
    spin_deg: float = 0.0
    tilt_deg: float = 0.0
    standoff: float = 0.0
    clearance: float | None = None  # None = no collision geometry nearby (infinite)
    manipulability: float = 0.0
    torque_util: float = 0.0
    limit_margin_deg: float = 0.0
    notes: list[str] = Field(default_factory=list)
    config: str = ""  # joint configuration label (core.configs)
    zone: int | None = None
    seq: int | None = None


class Zone(_Model):
    """Points of one station done in one joint configuration, in path order."""

    station: int = Field(ge=0)
    config: str
    points: list[int] = Field(default_factory=list)  # point ids, path order
    wrap_jumps: int = Field(default=0, ge=0)  # joint 4/6 jumps > 180 deg left in the path


class WorkState(_Model):
    """Session state that is not part of the cell definition: the arm's joint pose, the
    surface selection, feasibility and plan joint solutions, and panel settings."""

    q: list[float] | None = None
    selection_mesh: str | None = None  # the selection is only valid on this part mesh
    selection: list[int] = Field(default_factory=list)
    feasibility: list[PointResultData] = Field(default_factory=list)
    result_points: dict[str, list[PointResultData]] = Field(default_factory=dict)  # solution label -> points
    ui: dict[str, Any] = Field(default_factory=dict)  # panel settings: "panel.widget" -> value
