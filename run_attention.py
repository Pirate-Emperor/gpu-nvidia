#!/usr/bin/env python3
"""Course-required CLI for the fused cuTile attention backend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from attention_io import load_attention_npz, save_attention_npy
from cutile_attention import CuTileUnavailableError, forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("cutile",))
    parser.add_argument("--tile-q", type=int, default=64)
    parser.add_argument("--tile-k", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        q, k, v = load_attention_npz(args.input)
        out = forward(
            q,
            k,
            v,
            tile_q=args.tile_q,
            tile_k=args.tile_k,
        )
        save_attention_npy(args.output, out)

        print(f"backend={args.backend}")
        print(f"Q={q.shape} K={k.shape} V={v.shape}")
        print(f"output={out.shape} dtype={out.dtype}")
        return 0
    except (ValueError, OSError, CuTileUnavailableError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
