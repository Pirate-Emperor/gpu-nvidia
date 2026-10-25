from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.legal_config_registry import (
    DEFAULT_CONFIG_PATH,
    load_config,
    load_config_data,
    validate_config_data,
)


def read_default() -> dict:
    return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def test_default_registry_preserves_original_29_configurations() -> None:
    config = load_config()
    assert len(config["cases"]) == 7
    assert len(config["tile_candidates"]) == 5
    assert config["legal_pair_count"] == 29


def test_adding_a_workload_expands_the_matrix() -> None:
    data = read_default()
    data["cases"].append(
        {
            "name": "extra_medium_d64",
            "B": 1,
            "H": 4,
            "N": 256,
            "D": 64,
            "Dv": 64,
            "seed": 7777,
        }
    )
    cases, tiles, _ = validate_config_data(data)
    legal_pairs = [
        (case["name"], tile["tile_q"], tile["tile_k"])
        for case in cases
        for tile in tiles
        if case["N"] % tile["tile_q"] == 0 and case["N"] % tile["tile_k"] == 0
    ]
    assert len(legal_pairs) == 34
    assert sum(name == "extra_medium_d64" for name, _, _ in legal_pairs) == 5


def test_adding_a_tile_candidate_expands_the_matrix() -> None:
    data = read_default()
    data["tile_candidates"].append({"tile_q": 16, "tile_k": 64})
    cases, tiles, _ = validate_config_data(data)
    legal_count = sum(
        1
        for case in cases
        for tile in tiles
        if case["N"] % tile["tile_q"] == 0 and case["N"] % tile["tile_k"] == 0
    )
    assert legal_count == 36


def test_non_power_of_two_tile_is_rejected() -> None:
    data = read_default()
    data["tile_candidates"].append({"tile_q": 48, "tile_k": 64})
    with pytest.raises(ValueError, match="not a power of two"):
        validate_config_data(data)


def test_unsupported_head_dimension_is_rejected() -> None:
    data = read_default()
    data["cases"][0]["D"] = 96
    data["cases"][0]["Dv"] = 96
    with pytest.raises(ValueError, match="unsupported"):
        validate_config_data(data)


def test_mismatched_value_dimension_is_rejected() -> None:
    data = read_default()
    data["cases"][0]["Dv"] = 64
    with pytest.raises(ValueError, match="must equal D"):
        validate_config_data(data)


def test_workload_without_any_legal_tile_is_rejected() -> None:
    data = read_default()
    data["cases"].append(
        {
            "name": "no_matching_tile",
            "B": 1,
            "H": 1,
            "N": 96,
            "D": 32,
            "Dv": 32,
            "seed": 7780,
        }
    )
    with pytest.raises(ValueError, match="not divisible"):
        validate_config_data(data)


def test_duplicate_workload_name_is_rejected() -> None:
    data = read_default()
    data["cases"].append(copy.deepcopy(data["cases"][0]))
    with pytest.raises(ValueError, match="Duplicate case name"):
        validate_config_data(data)


def test_duplicate_tile_candidate_is_rejected() -> None:
    data = read_default()
    data["tile_candidates"].append(copy.deepcopy(data["tile_candidates"][0]))
    with pytest.raises(ValueError, match="Duplicate tile candidate"):
        validate_config_data(data)


def test_manual_configuration_file_is_loaded_and_validated(tmp_path: Path) -> None:
    destination = tmp_path / "legal_attention_configs.json"
    destination.write_text(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    cases, tiles, warnings = load_config_data(destination)
    assert len(cases) == 7
    assert len(tiles) == 5
    assert isinstance(warnings, list)
