#!/usr/bin/env python3
"""Generate deterministic course-format attention inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from io import make_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--mode",
        choices=("normal", "small", "large_logits", "constant"),
        default="normal",
    )
    return parser.parse_args()


def generate(
    batch: int,
    heads: int,
    seq: int,
    head_dim: int,
    value_dim: int,
    seed: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if min(batch, heads, seq, head_dim, value_dim) <= 0:
        raise ValueError("all dimensions must be positive")

    rng = np.random.default_rng(seed)

    if mode == "normal":
        scale = 1.0
    elif mode == "small":
        scale = 1e-3
    elif mode == "large_logits":
        # 64 is finite in FP16 and creates deliberately large finite
        # FP32 logits after the dot product and scale operation.
        scale = 64.0
    else:
        q = np.ones((batch, heads, seq, head_dim), dtype=np.float16)
        k = np.ones((batch, heads, seq, head_dim), dtype=np.float16)
        v = np.arange(
            batch * heads * seq * value_dim, dtype=np.float32
        ).reshape(batch, heads, seq, value_dim).astype(np.float16)
        make_npz(args_output_placeholder, q, k, v)
        return q, k, v

    q = (
        rng.standard_normal((batch, heads, seq, head_dim), dtype=np.float32) * scale
    ).astype(np.float16)
    k = (
        rng.standard_normal((batch, heads, seq, head_dim), dtype=np.float32) * scale
    ).astype(np.float16)
    v = (
        rng.standard_normal((batch, heads, seq, value_dim), dtype=np.float32) * scale
    ).astype(np.float16)
    return q, k, v


args_output_placeholder: Path | None = None


def main() -> int:
    global args_output_placeholder
    args = parse_args()
    value_dim = args.head_dim if args.value_dim is None else args.value_dim
    args_output_placeholder = args.output

    q, k, v = generate(
        args.batch,
        args.heads,
        args.seq,
        args.head_dim,
        value_dim,
        args.seed,
        args.mode,
    )

    make_npz(args.output, q, k, v)
    print(f"wrote {args.output}")
    print(f"Q={q.shape} K={k.shape} V={v.shape} dtype={q.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
