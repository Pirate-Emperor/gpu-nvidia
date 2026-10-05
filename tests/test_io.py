import numpy as np
import pytest

from attention_io import load_attention_npz, make_npz, save_attention_npy
from reference import AttentionValidationError


def make_inputs():
    rng = np.random.default_rng(20261006)
    q = rng.standard_normal((1, 2, 8, 8), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((1, 2, 8, 8), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((1, 2, 8, 5), dtype=np.float32).astype(np.float16)
    return q, k, v


def test_npz_round_trip(tmp_path):
    q, k, v = make_inputs()
    path = tmp_path / "input.npz"
    make_npz(path, q, k, v)

    q2, k2, v2 = load_attention_npz(path)
    np.testing.assert_array_equal(q2, q)
    np.testing.assert_array_equal(k2, k)
    np.testing.assert_array_equal(v2, v)
    assert q2.flags.c_contiguous
    assert k2.flags.c_contiguous
    assert v2.flags.c_contiguous


def test_npz_requires_exact_keys(tmp_path):
    q, k, _ = make_inputs()
    path = tmp_path / "bad.npz"
    np.savez(path, Q=q, K=k)

    with pytest.raises(AttentionValidationError):
        load_attention_npz(path)


def test_npz_rejects_extra_keys(tmp_path):
    q, k, v = make_inputs()
    path = tmp_path / "bad_extra.npz"
    np.savez(path, Q=q, K=k, V=v, EXTRA=np.zeros(1, dtype=np.float32))

    with pytest.raises(AttentionValidationError):
        load_attention_npz(path)


def test_save_npy_round_trip(tmp_path):
    out = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    path = tmp_path / "output.npy"
    save_attention_npy(path, out)

    restored = np.load(path, allow_pickle=False)
    np.testing.assert_array_equal(restored, out)


def test_missing_input_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_attention_npz(tmp_path / "missing.npz")
