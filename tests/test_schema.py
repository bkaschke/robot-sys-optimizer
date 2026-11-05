import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.schema import Project, load_project, save_project

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sander_mobile.json"


def _example_dict() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def test_plan_example_loads():
    p = load_project(EXAMPLE)
    assert p.robot.ik == "ur_analytic"
    assert p.base.carrier == "mobile"
    assert p.end_effector.tcps[0].name == "pad"
    assert p.targets.points[0].n == (0.0, 0.0, 1.0)


def test_round_trip(tmp_path):
    p = load_project(EXAMPLE)
    out = tmp_path / "p.json"
    save_project(p, out)
    assert load_project(out) == p


def test_normals_are_normalized():
    d = _example_dict()
    d["targets"]["points"][0]["n"] = [0, 0, 5]
    assert Project.model_validate(d).targets.points[0].n == (0.0, 0.0, 1.0)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["targets"]["points"][0].update(n=[0, 0, 0]),  # zero normal
        lambda d: d["targets"]["points"][0].update(group="nope"),  # unknown group
        lambda d: d["targets"]["groups"][0].update(tcps=["ghost"]),  # unknown TCP
        lambda d: d["end_effector"]["extension"].update(L=[0.6, 0.05]),  # inverted range
        lambda d: d["base"].update(carrier=None),  # movable without carrier
        lambda d: d["base"]["standoff_m"].update(carrier_part_preferred=0.1),  # pref < min
        lambda d: d.update(version=2),  # unsupported version
        lambda d: d["robot"].update(typo_field=1),  # unknown key
        lambda d: d["robot"]["base_pose"].update(xyz=[float("nan"), 0, 0]),  # non-finite
    ],
)
def test_invalid_projects_rejected(mutate):
    d = _example_dict()
    mutate(d)
    with pytest.raises(ValidationError):
        Project.model_validate(d)


def test_payload_spec_placeholder_is_incomplete():
    assert not load_project(EXAMPLE).robot.payload_spec.is_complete()
