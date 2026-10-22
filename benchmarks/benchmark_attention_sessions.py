#!/usr/bin/env python3
"""Run three order-rotated sessions of the full cuTile tile benchmark.

Each session uses the existing benchmark harness and writes its own raw JSON/CSV.
The combined result preserves the session medians and computes aggregate median,
P90, speedup and session-range values rather than averaging away variability.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_TILES = ("32x64", "64x64", "64x128", "128x64", "128x128")
PROTOCOL_SESSIONS = (
    BASE_TILES,
    BASE_TILES[1:] + BASE_TILES[:1],
    tuple(reversed(BASE_TILES)),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repeat the complete tile sweep for three order-varied sessions."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cutile-attention-3session.json"),
        help="combined JSON output path; a sibling CSV and per-session raw files are written",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats-per-sample", type=int, default=5)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 30 or args.repeats_per_sample < 1:
        parser.error("warmup >= 1, iterations >= 30, repeats-per-sample >= 1 required")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_payloads = []
    session_meta = []
    for session_index, tile_order in enumerate(PROTOCOL_SESSIONS, 1):
        raw_path = output.with_name(f"{output.stem}_session{session_index}.json")
        command = [
            sys.executable,
            str(ROOT / "benchmarks" / "benchmark_attention.py"),
            "--output", str(raw_path),
            "--warmup", str(args.warmup),
            "--iterations", str(args.iterations),
            "--repeats-per-sample", str(args.repeats_per_sample),
            "--tiles", *tile_order,
        ]
        print("")
        print("=" * 88)
        print(f"SESSION {session_index}/3 | tile order: {' -> '.join(tile_order)}")
        print(f"RAW_JSON={raw_path}")
        subprocess.run(command, cwd=ROOT, check=True)
        if not raw_path.is_file():
            raise RuntimeError(f"benchmark returned successfully but did not create {raw_path}")
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        raw_payloads.append(payload)
        session_meta.append({
            "session": session_index,
            "tile_order": list(tile_order),
            "json": str(raw_path),
            "csv": str(raw_path.with_suffix(".csv")),
        })

    combined = {}
    for session_index, payload in enumerate(raw_payloads, 1):
        for row in payload["results"]:
            key = (row["case"], int(row["tile_q"]), int(row["tile_k"]))
            combined.setdefault(key, []).append((session_index, row))

    aggregated_rows = []
    for key, measurements in combined.items():
        if len(measurements) != 3:
            raise RuntimeError(f"expected three sessions for {key}, got {len(measurements)}")
        session_rows = [item[1] for item in measurements]
        medians = [float(row["fused_median_ms"]) for row in session_rows]
        p90s = [float(row["fused_p90_ms"]) for row in session_rows]
        baselines = [float(row["baseline_median_ms"]) for row in session_rows]
        speedups = [float(row["speedup_vs_unfused"]) for row in session_rows]
        baseline_drifts = [float(row["baseline_bracket_drift_pct"]) for row in session_rows]
        aggregate_median = float(statistics.median(medians))
        range_pct = (
            100.0 * (max(medians) - min(medians)) / aggregate_median
            if aggregate_median > 0 else 0.0
        )
        aggregate = {
            "case": key[0],
            "B": session_rows[0]["B"],
            "H": session_rows[0]["H"],
            "N": session_rows[0]["N"],
            "D": session_rows[0]["D"],
            "Dv": session_rows[0]["Dv"],
            "tile_q": key[1],
            "tile_k": key[2],
            "fused_median_ms": aggregate_median,
            "fused_p90_ms": float(statistics.median(p90s)),
            "baseline_median_ms": float(statistics.median(baselines)),
            "speedup_vs_unfused": float(statistics.median(speedups)),
            "session_range_pct": range_pct,
            "max_baseline_bracket_drift_pct": max(baseline_drifts),
            "session_medians_ms": medians,
            "session_p90_ms": p90s,
            "session_baseline_medians_ms": baselines,
            "session_speedups": speedups,
            "session_baseline_drift_pct": baseline_drifts,
            "first_call_ms_by_session": [row.get("jit_ms") for row in session_rows],
            "sessions": 3,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats_per_sample": args.repeats_per_sample,
            "measured_launches_per_session": args.iterations * args.repeats_per_sample,
        }
        aggregated_rows.append(aggregate)

    expected_rows = 29
    if len(aggregated_rows) != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} legal case/tile rows across all sessions; "
            f"got {len(aggregated_rows)}"
        )

    # Preserve the first session's case/tile ordering so the printed table is easy to compare.
    first_order = {
        (row["case"], int(row["tile_q"]), int(row["tile_k"])): index
        for index, row in enumerate(raw_payloads[0]["results"])
    }
    aggregated_rows.sort(
        key=lambda row: first_order[(row["case"], row["tile_q"], row["tile_k"])]
    )

    payload = {
        "title": "Three-session cuTile attention tile experiment",
        "method": (
            "Run the same seven mandatory workloads in three separate benchmark processes. "
            "Tile order is base, left-rotated, then reversed. Each session brackets the "
            "tile sweep with baseline measurements. Aggregates are medians across the "
            "three session summaries; raw session JSON/CSV files are kept separately."
        ),
        "environment": raw_payloads[0].get("environment", {}),
        "protocol": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats_per_sample": args.repeats_per_sample,
            "sessions": 3,
            "tiles": list(BASE_TILES),
            "tile_orders": [list(x) for x in PROTOCOL_SESSIONS],
            "legal_case_tile_rows": len(aggregated_rows),
            "measured_launches_per_configuration_across_sessions": (
                3 * args.iterations * args.repeats_per_sample
            ),
            "baseline": "deliberately unfused FP32 CuPy; not FlashAttention/optimized SDPA",
        },
        "raw_sessions": session_meta,
        "results": aggregated_rows,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    fields = [
        "case", "B", "H", "N", "D", "Dv", "tile_q", "tile_k",
        "fused_median_ms", "fused_p90_ms", "baseline_median_ms",
        "speedup_vs_unfused", "session_range_pct",
        "max_baseline_bracket_drift_pct", "session_medians_ms",
        "session_p90_ms", "session_speedups", "session_baseline_drift_pct",
        "first_call_ms_by_session", "sessions", "warmup", "iterations",
        "repeats_per_sample", "measured_launches_per_session",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in aggregated_rows:
            writer.writerow({
                **row,
                "session_medians_ms": json.dumps(row["session_medians_ms"]),
                "session_p90_ms": json.dumps(row["session_p90_ms"]),
                "session_speedups": json.dumps(row["session_speedups"]),
                "session_baseline_drift_pct": json.dumps(row["session_baseline_drift_pct"]),
                "first_call_ms_by_session": json.dumps(row["first_call_ms_by_session"]),
            })

    print("")
    print("THREE-SESSION AGGREGATE (milliseconds; speedup is vs unfused FP32 CuPy)")
    print(
        f"{'CASE':25s} {'TILE':9s} {'MEDIAN':>10s} {'P90':>10s} "
        f"{'BASELINE':>10s} {'SPEEDUP':>9s} {'RANGE %':>9s} {'DRIFT %':>9s}"
    )
    for row in aggregated_rows:
        print(
            f"{row['case']:25s} "
            f"{row['tile_q']}x{row['tile_k']:<5d} "
            f"{row['fused_median_ms']:10.6f} "
            f"{row['fused_p90_ms']:10.6f} "
            f"{row['baseline_median_ms']:10.6f} "
            f"{row['speedup_vs_unfused']:8.2f}x "
            f"{row['session_range_pct']:8.2f}% "
            f"{row['max_baseline_bracket_drift_pct']:8.2f}%"
        )
    print(f"LEGAL_CASE_TILE_ROWS={len(aggregated_rows)}")
    print(f"RAW_SESSION_JSON={[item['json'] for item in session_meta]}")
    print(f"AGGREGATE_JSON={output}")
    print(f"AGGREGATE_CSV={csv_path}")
    print("NOTE: clean-cache JIT time is not guaranteed; first-call numbers may include a compiler-cache hit.")
    print("NOTE: the baseline is unfused FP32 CuPy, not optimized vendor attention.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
