"""Frozen test/benchmark case definitions shared by GPU test code."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionCase:
    name: str
    B: int
    H: int
    N: int
    D: int
    Dv: int
    seed: int


# These are deliberately conservative A100-friendly cases. The course requires
# at least six representative configurations, including short/long sequences
# and head dimensions 64 and 128; these six satisfy that minimum. Additional
# instructor-provided hidden tests remain authoritative.
MANDATORY_CASES = (
    AttentionCase("short_d64", 1, 1, 64, 64, 64, 1001),
    AttentionCase("short_multihead_d64", 1, 4, 128, 64, 64, 1002),
    AttentionCase("medium_d64", 1, 8, 256, 64, 64, 1003),
    AttentionCase("medium_d128", 1, 8, 512, 128, 128, 1004),
    AttentionCase("long_d128", 1, 16, 1024, 128, 128, 1005),
    AttentionCase("longer_batched_d128", 2, 8, 2048, 128, 128, 1006),
)

HEAD_DIMENSIONS = (64, 128)
