# Results and Experiment Artifacts

This directory stores raw reproducible outputs rather than hand-edited tables.

For each benchmark/profiling session, preserve:

- JSON benchmark results.
- CSV benchmark results.
- Markdown interpretation notes.
- Nsight Compute reports when available.
- Exported profiler summaries when useful.

## Naming

Use:

YYYYMMDD_GPU_CASE_BACKEND_TILE.EXT

Example:

20261006_A100_long_d128_cutile_64x64.json

Never overwrite an older run. Keep the run and environment metadata so every reported number is auditable.

## Required benchmark metadata

Every benchmark result must identify:

- GPU model.
- Driver version.
- CUDA runtime/toolkit version.
- cuTile version when applicable.
- Python/CuPy/PyTorch versions when applicable.
- Tile configuration.
- Warm-up count.
- Measured iteration count (>=30).
- Median latency.
- P90 latency.
- JIT/cold-start time separately.
- Speedup relative to the deliberately unfused baseline.
- Effective output throughput.
- Clock/power policy when known.

Final report tables should be generated from these raw files rather than manually typed.
