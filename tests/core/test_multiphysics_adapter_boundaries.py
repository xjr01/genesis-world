from pathlib import Path

import pytest

SCENARIO_ROOT = Path(__file__).parents[2] / "examples" / "multiphysics"
SCENARIOS = ("coffee_water", "table_wiping", "litter_scoop", "garment_folding", "butter_spreading")
FORBIDDEN_RUNTIME_TOKENS = ("scene.sim", "._solvers", "particle_start", "particle_end")


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_normalized_scenario_does_not_reach_into_engine_private_dispatch(scenario):
    for source_path in (SCENARIO_ROOT / scenario).glob("*.py"):
        source = source_path.read_text(encoding="utf-8")
        for token in FORBIDDEN_RUNTIME_TOKENS:
            assert token not in source, f"{source_path} reaches through the normalized adapter boundary with {token!r}"


def test_table_wiping_is_independent_of_research_example_runtime():
    for source_path in (SCENARIO_ROOT / "table_wiping").glob("*.py"):
        source = source_path.read_text(encoding="utf-8")
        assert "examples.multiflow" not in source
