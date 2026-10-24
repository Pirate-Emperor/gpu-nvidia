# Project handoff

Continue the cuTile attention project in xemptynothing/gpu-nvidia.

Start with README.md, docs/TODO_CHECKLIST.md, docs/COURSE_PROJECT_COMPLETION_CHECKLIST.md, Report.md and docs/EXECUTION_EVIDENCE_INDEX.md. Check the current main commit and workflow results before using older numbers.

## Project

- Main kernel: src/cutile_attention.py
- GPU target: Windows RTX 4060, compute capability 8.9
- A100 work: ordinary CUDA C++ comparisons, not cuTile
- Correctness suite: 48 GPU test executions
- Separate numerical matrix: 29 legal case/tile combinations
- Benchmark reference: deliberately unfused FP32 CuPy

## Status

The last complete native GPU run is #39422402438 on commit bf5a623. It passed the core tests, numerical matrix, CLI check, three benchmark sessions and two Nsight Compute configurations. The kernel was compacted afterward; confirm a fresh native run for current main before relying on the old result.

## Open work

1. Revalidate current main on the RTX 4060.
2. Measure clean-cache first-call/JIT time for long specializations.
3. Complete a compatible framework attention comparison and record the backend.
4. Use detailed Nsight Compute evidence to choose one change, then test correctness, timing and the matching profile.
5. Finish the report and record actual team contributions.

Keep code changes small. Run correctness before timing. Do not call a timing improvement real unless it exceeds the observed variation. Keep old evidence, but label it by source commit and date.
