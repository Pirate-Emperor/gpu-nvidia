# Test matrix

This file separates previously recorded results from the checks still needed on current main.

## Core coverage

| Check | Coverage |
|---|---|
| Reference comparisons | 29 legal workload/tile combinations |
| GPU correctness suite | 48 test executions across correctness and edge-case files |
| CLI output | Saved FP16 result compared with the independent NumPy reference |
| Runtime | Genuine cuTile kernel compile, launch and result check |
| Performance | Three historical timing sessions; five tile choices on seven workloads |
| Profiling | Two historical cuTile configurations, with resource and scheduler data |

## Last complete native run

[Run #39422402438](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438) on commit bf5a623 passed the core tests, numerical matrix and CLI checks. The numerical matrix had zero tolerance violations and maximum absolute error 0.0009765625. The run also recorded three timing sessions and two Nsight Compute configurations.

The kernel was compacted after this run. Current main needs a fresh native check before these results can be treated as validation of the current source.

## Performance limits

The benchmark uses an unfused FP32 CuPy baseline. Its speedups do not compare against FlashAttention or optimized PyTorch SDPA. Some long-specialization first-call timings were not measured from a clean cache. Those values must remain open until measured properly.

## Nsight Compute

The saved profiles show high register/shared-memory use and low achieved occupancy for the two sampled configurations. Scheduler data shows few eligible warps in many cycles. These are leads for another experiment, not proof of a single bottleneck. The next capture should include Details and Raw views as well as occupancy, memory, scheduler and warp-state sections.

## Next checks

- [ ] Current main: runtime, 48 GPU tests and CLI check.
- [ ] Clean-cache JIT measurements for long specializations.
- [ ] Compatible framework attention comparison.
- [ ] Matched before/after profile for one kernel change.
- [ ] Final report and contribution notes.

See [execution evidence](EXECUTION_EVIDENCE_INDEX.md) for run links and artifacts.
