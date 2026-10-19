import os

import pytest

if os.environ.get("REQUIRE_CUTILE") == "1":
    try:
        import cupy as cp
    except ImportError as exc:
        pytest.fail(f"CuPy required for cuTile/PyTorch optional gates: {exc}", pytrace=False)
    try:
        import cuda.tile as ct
    except ImportError as exc:
        pytest.fail(f"cuTile required for cuTile/PyTorch optional gates: {exc}", pytrace=False)
    try:
        import torch
    except ImportError as exc:
        pytest.fail(f"PyTorch required for cuTile/PyTorch optional gates: {exc}", pytrace=False)
else:
    cp = pytest.importorskip("cupy")
    ct = pytest.importorskip("cuda.tile")
    torch = pytest.importorskip("torch")

import torch.nn.functional as F

from torch_integration import cutile_attention_torch, make_attention_module


@pytest.mark.parametrize("n_heads", [1, 2])
def test_torch_autograd_bridge_matches_sdpa_output_and_gradients(n_heads):
    torch.manual_seed(1200 + n_heads)
    q = torch.randn((1, n_heads, 64, 32), device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn((1, n_heads, 64, 32), device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn((1, n_heads, 64, 32), device="cuda", dtype=torch.float16, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    actual = cutile_attention_torch(q, k, v, tile_q=64, tile_k=64)
    expected = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, dropout_p=0.0, is_causal=True)
    assert actual.shape == expected.shape
    assert actual.dtype == torch.float16
    assert torch.isfinite(actual).all().item()
    assert torch.allclose(actual, expected, rtol=3e-2, atol=3e-2)

    probe = torch.randn_like(actual)
    (actual.float() * probe.float()).sum().backward()
    (expected.float() * probe.float()).sum().backward()
    for actual_grad, expected_grad in (
        (q.grad, q_ref.grad), (k.grad, k_ref.grad), (v.grad, v_ref.grad)
    ):
        assert actual_grad is not None
        assert torch.isfinite(actual_grad).all().item()
        assert torch.allclose(actual_grad, expected_grad, rtol=0.15, atol=0.15)


def test_model_block_matches_sdpa_reference_block():
    torch.manual_seed(1337)
    CuTileBlock, SDPABlock = make_attention_module(torch)
    cutile_model = CuTileBlock(128, 4, tile_q=64, tile_k=64).cuda().half()
    sdpa_model = SDPABlock(128, 4).cuda().half()
    sdpa_model.load_state_dict(cutile_model.state_dict())

    x = torch.randn((1, 64, 128), dtype=torch.float16, device="cuda")
    actual = cutile_model(x)
    expected = sdpa_model(x)
    assert actual.shape == expected.shape
    assert actual.dtype == torch.float16
    assert torch.isfinite(actual).all().item()
    assert torch.allclose(actual, expected, rtol=3e-2, atol=3e-2)
