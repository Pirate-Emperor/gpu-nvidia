import numpy as np
import pytest

from contract import AttentionValidationError
from cutile_attention import _validate_cutile_config


def make_inputs(N=64, D=64, Dv=64):
    q = np.zeros((1, 1, N, D), dtype=np.float16)
    k = np.zeros((1, 1, N, D), dtype=np.float16)
    v = np.zeros((1, 1, N, Dv), dtype=np.float16)
    return q, k, v


@pytest.mark.parametrize(
    "tile_q,tile_k",
    [
        (64, 64),
        (32, 64),
        (64, 32),
        (128, 64),
    ],
)
def test_valid_cutile_configs(tile_q, tile_k):
    q, k, v = make_inputs(N=128)
    _validate_cutile_config(q, k, v, tile_q, tile_k)


def test_all_required_head_dimensions_are_accepted():
    for d in (32, 64, 128):
        q, k, v = make_inputs(N=128, D=d, Dv=d)
        _validate_cutile_config(q, k, v, 64, 64)


@pytest.mark.parametrize(
    "tile_q,tile_k",
    [
        (48, 64),
        (64, 48),
        (0, 64),
        (64, 0),
        (96, 64),
    ],
)
def test_invalid_cutile_tile_shapes(tile_q, tile_k):
    q, k, v = make_inputs(N=128)
    with pytest.raises(AttentionValidationError):
        _validate_cutile_config(q, k, v, tile_q, tile_k)


def test_unsupported_head_dimension_rejected_for_cutile():
    q, k, v = make_inputs(D=16, Dv=16)
    with pytest.raises(AttentionValidationError, match="unsupported"):
        _validate_cutile_config(q, k, v, 64, 64)


def test_value_dimension_mismatch_rejected_for_cutile():
    q, k, v = make_inputs(D=64, Dv=32)
    with pytest.raises(AttentionValidationError, match="Dv == D"):
        _validate_cutile_config(q, k, v, 64, 64)


def test_sequence_divisibility_checked_before_cutile_launch():
    q, k, v = make_inputs(N=96)
    with pytest.raises(AttentionValidationError, match="query sequence length"):
        _validate_cutile_config(q, k, v, 64, 64)
