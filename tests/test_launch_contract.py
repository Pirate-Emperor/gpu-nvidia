import numpy as np
import pytest

from contract import TileConfig, validate_launch_contract
from reference import AttentionValidationError


def make_inputs(N=64):
    q = np.zeros((1, 2, N, 64), dtype=np.float16)
    k = np.zeros((1, 2, N, 64), dtype=np.float16)
    v = np.zeros((1, 2, N, 64), dtype=np.float16)
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
