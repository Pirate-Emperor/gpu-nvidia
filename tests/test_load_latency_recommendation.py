from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_load_latency_hints.py"
SPEC = spec_from_file_location("benchmark_load_latency_hints", SCRIPT)
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_recommends_latency_hint_only_when_gain_clears_noise():
    by_hint = {
        0: [{"median_ms": 1.00}, {"median_ms": 1.01}, {"median_ms": 0.99}],
        8: [{"median_ms": 0.92}, {"median_ms": 0.93}, {"median_ms": 0.92}],
    }
    result = MODULE.hint_recommendation(by_hint, minimum_gain_pct=1.0)
    assert result["latency_hint"] == 8
    assert result["credible"] is True
    assert result["gain_pct_vs_default"] > result["required_gain_pct"]


def test_rejects_latency_hint_gain_within_session_spread():
    by_hint = {
        0: [{"median_ms": 1.00}, {"median_ms": 1.03}, {"median_ms": 0.98}],
        4: [{"median_ms": 0.999}, {"median_ms": 1.02}, {"median_ms": 0.981}],
    }
    result = MODULE.hint_recommendation(by_hint, minimum_gain_pct=1.0)
    assert result["latency_hint"] == 0
    assert result["credible"] is False
    assert "no hint" in result["status"]
