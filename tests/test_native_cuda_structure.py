"""Static design-contract checks for the non-cuTile CUDA surrogates."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_src(name: str) -> str:
    return (ROOT / "src" / name).read_text(encoding="utf-8")


def test_shared_tile_is_tile_oriented_and_online():
    src = read_src("shared_tile_cuda_attention.py")
    assert "__shared__ half k_shared" in src
    assert "__shared__ half v_shared" in src
    assert "TILE_Q = WARPS_PER_BLOCK" in src
    assert "run_max" in src
    assert "run_norm" in src
    assert "out_acc" in src
    assert "__shfl_down_sync" in src
    assert 'backend="nvcc"' in src


def test_fused_cuda_variants_do_not_materialize_n_by_n_scores():
    for name in (
        "custom_cuda_attention.py",
        "warp_cuda_attention.py",
        "shared_tile_cuda_attention.py",
    ):
        src = read_src(name).replace(" ", "").lower()
        assert "n,n" not in src
        assert "n*n" not in src
        assert "n**2" not in src


def test_native_surrogates_use_fp16_storage_and_fp32_math():
    for name in (
        "custom_cuda_attention.py",
        "warp_cuda_attention.py",
        "shared_tile_cuda_attention.py",
    ):
        src = read_src(name)
        assert "cuda_fp16.h" in src
        assert "__half2float" in src
        assert "__float2half" in src
        assert "float" in src
