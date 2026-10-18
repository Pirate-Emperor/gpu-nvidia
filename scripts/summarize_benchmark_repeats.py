#!/usr/bin/env python3
"""Aggregate repeated cuTile benchmark JSON sessions into auditable CSV/JSON.

This tool reads only the explicitly supplied benchmark reports and writes its
summary to the requested project-local paths. It does not invoke GPU code.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from collections import defaultdict


def median(values):
    return float(statistics.median(values)) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()

    sessions = []
    grouped = defaultdict(list)
    for path in args.inputs:
        data = json.loads(path.read_text(encoding="utf-8"))
        sessions.append({
            "input": str(path),
            "git_commit": data.get("environment", {}).get("git_commit"),
            "gpu": data.get("environment", {}).get("gpu"),
            "warmup": data.get("warmup"),
            "iterations": data.get("iterations"),
            "repeats_per_sample": data.get("repeats_per_sample", 1),
            "baseline_measurement": data.get("baseline_measurement"),
            "tiles": data.get("tiles"),
        })
        for row in data["results"]:
            key = (row["case"], int(row["tile_q"]), int(row["tile_k"]))
            grouped[key].append({**row, "session_input": str(path)})

    results = []
    for (case, tile_q, tile_k), rows in sorted(grouped.items()):
        fused = [float(r["fused_median_ms"]) for r in rows]
        fused_p90 = [float(r["fused_p90_ms"]) for r in rows]
        baseline = [float(r["baseline_median_ms"]) for r in rows]
        baseline_p90 = [float(r["baseline_p90_ms"]) for r in rows]
        speedups = [float(r["speedup_vs_unfused"]) for r in rows]
        center = median(fused)
        spread = max(fused) - min(fused)
        results.append({
            "case": case,
            "tile_q": tile_q,
            "tile_k": tile_k,
            "repeat_count": len(rows),
            "median_fused_ms": center,
            "median_fused_p90_ms": median(fused_p90),
            "min_fused_ms": min(fused),
            "max_fused_ms": max(fused),
            "fused_range_pct_of_median": (100.0 * spread / center) if center else None,
            "median_baseline_ms": median(baseline),
            "median_baseline_p90_ms": median(baseline_p90),
            "median_speedup_vs_unfused": median(speedups),
            "min_speedup_vs_unfused": min(speedups),
            "max_speedup_vs_unfused": max(speedups),
            "jit_ms_by_session": [r.get("jit_ms") for r in rows],
            "session_inputs": [r["session_input"] for r in rows],
        })

    best_by_case = {}
    for row in results:
        prev = best_by_case.get(row["case"])
        if prev is None or row["median_fused_ms"] < prev["median_fused_ms"]:
            best_by_case[row["case"]] = row
    for row in results:
        best = best_by_case[row["case"]]
        row["best_median_tile_for_case"] = (
            f"{best['tile_q']}x{best['tile_k']}"
        )
        row["is_best_median_tile_for_case"] = row is best

    payload = {
        "summary": "Across-session medians; raw session JSON files remain the source of record.",
        "session_count": len(sessions),
        "sessions": sessions,
        "result_count": len(results),
        "results": results,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fields = [
        "case", "tile_q", "tile_k", "repeat_count", "median_fused_ms",
        "median_fused_p90_ms", "min_fused_ms", "max_fused_ms",
        "fused_range_pct_of_median", "median_baseline_ms",
        "median_baseline_p90_ms", "median_speedup_vs_unfused",
        "min_speedup_vs_unfused", "max_speedup_vs_unfused",
        "jit_ms_by_session", "best_median_tile_for_case",
        "is_best_median_tile_for_case", "session_inputs",
    ]
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({**row, "jit_ms_by_session": json.dumps(row["jit_ms_by_session"]),
                             "session_inputs": json.dumps(row["session_inputs"])})

    for row in results:
        if row["is_best_median_tile_for_case"]:
            print(
                f"BEST {row['case']:24s} tile={row['tile_q']}x{row['tile_k']} "
                f"median={row['median_fused_ms']:.6f} ms "
                f"across {row['repeat_count']} sessions "
                f"speedup={row['median_speedup_vs_unfused']:.2f}x "
                f"range={row['fused_range_pct_of_median']:.1f}%"
            )
    print(f"SUMMARY_JSON={args.json_output}")
    print(f"SUMMARY_CSV={args.csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
