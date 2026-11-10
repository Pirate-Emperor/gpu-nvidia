import numpy as np
import pytest

from contract import TileConfig, validate_launch_contract
from reference import AttentionValidationError


def make_inputs(N=64, D=64, Dv=64, B=1, H=2):
    q = np.zeros((B, H, N, D), dtype=np.float16)
    k = np.zeros((B, H, N, D), dtype=np.float16)
    v = np.zeros((B, H, N, Dv), dtype=np.float16)
    return q, k, v


def test_valid_divisible_configuration():
    q, k, v = make_inputs(128)
    validate_launch_contract(q, k, v, TileConfig(query=64, key=64))


@pytest.mark.parametrize(
    "n,tq,tk",
    [
        (64, 32, 64),
        (128, 64, 128),
        (256, 128, 64),
    ],
)
def test_multiple_valid_tile_configurations(n, tq, tk):
    q, k, v = make_inputs(n)
    validate_launch_contract(q, k, v, TileConfig(query=tq, key=tk))


@pytest.mark.parametrize(
    "n,tq,tk,match",
    [
        (64, 30, 64, "query sequence length"),
        (64, 64, 30, "key sequence length"),
    ],
)
def test_non_divisible_sequence_rejected(n, tq, tk, match):
    q, k, v = make_inputs(n)
    with pytest.raises(AttentionValidationError, match=match):
        validate_launch_contract(q, k, v, TileConfig(query=tq, key=tk))


def test_required_head_dimensions_are_accepted():
    for d in (32, 64, 128):
        q, k, v = make_inputs(N=128, D=d, Dv=d)
        validate_launch_contract(q, k, v, TileConfig(query=64, key=64))


def test_unsupported_head_dimension_rejected():
    q, k, v = make_inputs(N=64, D=16, Dv=16)
    with pytest.raises(AttentionValidationError, match="unsupported"):
        validate_launch_contract(q, k, v, TileConfig(query=64, key=64))


@pytest.mark.parametrize("D,Dv", [(32, 64), (64, 128), (128, 32)])
def test_independent_value_dimensions_are_accepted(D, Dv):
    q, k, v = make_inputs(N=64, D=D, Dv=Dv)
    validate_launch_contract(q, k, v, TileConfig(query=64, key=64))


def test_unsupported_value_dimension_rejected():
    q, k, v = make_inputs(N=64, D=64, Dv=16)
    with pytest.raises(AttentionValidationError, match="value dimension"):
        validate_launch_contract(q, k, v, TileConfig(query=64, key=64))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"B": 17}, "batch size"),
        ({"H": 33}, "head count"),
        ({"N": 8193}, "sequence length"),
    ],
)
def test_assignment_upper_bounds_are_enforced(kwargs, match):
    q, k, v = make_inputs(**kwargs)
    with pytest.raises(AttentionValidationError, match=match):
        validate_launch_contract(q, k, v, TileConfig(query=64, key=64))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"B": 1, "H": 1, "N": 1},
        {"B": 16, "H": 2, "N": 64},
        {"B": 1, "H": 32, "N": 64},
        {"B": 1, "H": 1, "N": 8192},
    ],
)
def test_assignment_contract_boundaries_are_accepted(kwargs):
    q, k, v = make_inputs(**kwargs)
    validate_launch_contract(
        q, k, v, TileConfig(query=1 if kwargs.get("N", 64) == 1 else 64, key=1 if kwargs.get("N", 64) == 1 else 64)
    )


@pytest.mark.parametrize(
    "tq,tk",
    [
        (0, 64),
        (64, 0),
        (-1, 64),
        (64, -1),
    ],
)
def test_invalid_tile_size_rejected(tq, tk):
    q, k, v = make_inputs(64)
    with pytest.raises(AttentionValidationError, match="tile dimensions"):
        validate_launch_contract(q, k, v, TileConfig(query=tq, key=tk))
