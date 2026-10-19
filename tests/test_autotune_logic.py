from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "autotune_cutile_tiles.py"
SPEC = spec_from_file_location("autotune_cutile_tiles", SCRIPT)
AUTOTUNER = module_from_spec(SPEC)
SPEC.loader.exec_module(AUTOTUNER)


def test_recommends_repeatable_candidate_with_clear_gain():
    summaries = {
        "64x64": {"median_ms": 1.000, "range_pct": 0.4},
        "64x128": {"median_ms": 0.960, "range_pct": 0.5},
    }
    result = AUTOTUNER.recommend_tile(summaries, min_gain_pct=1.0)
    assert (result["tile_q"], result["tile_k"]) == (64, 128)
    assert result["credible"] is True
    assert result["gain_pct_vs_default"] > 1.0


def test_rejects_gain_inside_measurement_spread():
    summaries = {
        "64x64": {"median_ms": 1.000, "range_pct": 2.0},
        "64x128": {"median_ms": 0.995, "range_pct": 2.1},
    }
    result = AUTOTUNER.recommend_tile(summaries, min_gain_pct=1.0)
    assert (result["tile_q"], result["tile_k"]) == (64, 64)
    assert result["credible"] is False
    assert "inconclusive" in result["status"]


def test_keeps_default_when_default_is_fastest():
    summaries = {
        "64x64": {"median_ms": 0.950, "range_pct": 0.4},
        "64x128": {"median_ms": 0.970, "range_pct": 0.5},
    }
    result = AUTOTUNER.recommend_tile(summaries)
    assert (result["tile_q"], result["tile_k"]) == (64, 64)
    assert result["credible"] is True
