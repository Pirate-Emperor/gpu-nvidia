import numpy as np

from attention_io import load_attention_npz
from generate_inputs import generate


def test_generator_is_deterministic():
    a = generate(1, 2, 64, 64, 64, 4242, "normal")
    b = generate(1, 2, 64, 64, 64, 4242, "normal")
    for xa, xb in zip(a, b):
        np.testing.assert_array_equal(xa, xb)


def test_generator_modes_are_finite():
    for mode in ("normal", "small", "large_logits", "constant"):
        q, k, v = generate(1, 1, 64, 64, 64, 7, mode)
        assert q.dtype == np.float16
        assert k.dtype == np.float16
        assert v.dtype == np.float16
        assert np.isfinite(q).all()
        assert np.isfinite(k).all()
        assert np.isfinite(v).all()


def test_constant_mode_is_exactly_constant():
    q, k, v = generate(1, 1, 64, 64, 64, 7, "constant")
    np.testing.assert_array_equal(q, np.ones_like(q))
    np.testing.assert_array_equal(k, np.ones_like(k))
    np.testing.assert_array_equal(v, np.ones_like(v))


def test_large_logits_mode_stays_finite():
    q, k, v = generate(1, 1, 64, 64, 64, 7, "large_logits")
    assert np.max(np.abs(q.astype(np.float32))) > 16.0
    assert np.isfinite(q).all() and np.isfinite(k).all() and np.isfinite(v).all()


def test_generated_file_round_trips(tmp_path):
    path = tmp_path / "generated.npz"
    q, k, v = generate(1, 2, 64, 64, 64, 7, "normal")

    from attention_io import make_npz

    make_npz(path, q, k, v)
    q2, k2, v2 = load_attention_npz(path)
    np.testing.assert_array_equal(q2, q)
    np.testing.assert_array_equal(k2, k)
    np.testing.assert_array_equal(v2, v)
