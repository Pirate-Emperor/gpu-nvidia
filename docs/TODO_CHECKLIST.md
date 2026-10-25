# Project checklist

Updated: 24 October 2026

The project implements fused causal self-attention in NVIDIA cuTile. Q, K and V use FP16; score and softmax state accumulate in FP32. The kernel processes tiles and does not write the full N-by-N score or probability matrix to global memory.

## Done earlier

- [x] Implemented the cuTile forward kernel.
- [x] Added an independent FP32 NumPy reference and an unfused FP32 CuPy baseline.
- [x] Added input-contract, causal-mask, numerical-stability and regression tests.
- [x] Ran the 48-test GPU suite on the RTX 4060.
- [x] Checked 29 supported workload/tile combinations against the reference.
- [x] Added a CLI to load Q/K/V, run attention and save the output.
- [x] Recorded three benchmark sessions and two Nsight Compute configurations.

The latest full recorded GPU run is #39422402438 on commit bf5a623. The kernel was compacted later, so these results do not replace a fresh check of current main.

## Still to do

- [ ] Rerun the runtime check, all 48 tests and the CLI on the current source.
- [ ] Capture first-call/JIT time in a clean cache for the long workloads.
- [ ] Compare against a compatible framework attention implementation; record the backend and verify output agreement.
- [ ] Review matched Nsight Compute reports for the compared tile choices.
- [ ] Use the profile to choose one change, then rerun correctness, timing and profiling.
- [ ] Finish the report and add a truthful contribution note for each team member.

## Rules for calling the work complete

- Do not count a skipped GPU test as a pass.
- Do not call the fastest tile from one run; use repeated timings and report variation.
- Keep the unfused CuPy result separate from any framework attention result.
- Keep the original Nsight reports and logs with the screenshots.
- If a change fails, record it and leave the last passing kernel in place.

See the [course checklist](COURSE_PROJECT_COMPLETION_CHECKLIST.md) for the assignment gates and the [execution evidence index](EXECUTION_EVIDENCE_INDEX.md) for run links.
