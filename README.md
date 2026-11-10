# Fused causal self-attention with NVIDIA cuTile

**Assignment status: complete for the six-page course brief (5 November 2026).**

This repository implements tiled causal scaled dot-product self-attention as a fused NVIDIA cuTile GPU kernel. The primary validation target is a native Windows NVIDIA GeForce RTX 4060 (SM 8.9). The Linux A100 experiments are ordinary CUDA C++ comparison controls and are not presented as cuTile results.

The Linux A100 experiments are CUDA C++ comparison controls, **not actual NVIDIA cuTile** results; none of their measurements are counted as cuTile validation.

## Submission

- **[Six-page final technical report (PDF)](docs/SUBMISSION_REPORT_2026-11-03.pdf)**
- [Final report source (Markdown)](docs/SUBMISSION_REPORT_2026-11-03.md)
- [Page-by-page assignment audit](docs/ASSIGNMENT_BRIEF_AUDIT.md)
- [Course completion checklist](docs/COURSE_PROJECT_COMPLETION_CHECKLIST.md)
- [Current RTX 4060 and framework evidence](docs/CURRENT_RUN_2026-11-02.md)

- [Week 3 tile-choice NCU analysis](docs/ncu/rtx4060/WEEK3_TILE_CHOICE_2026-11-05.md)

## Implementation

The kernel performs QKᵀ, scaling, causal masking, online softmax and P×V in one cuTile launch per output call. Q/K/V are FP16; score, softmax and P×V accumulation are FP32; output is FP16. A program instance owns a query tile and streams K/V tiles, avoiding materialization of full N×N score or probability matrices in global memory.

The implemented contract follows the brief: independent D and Dv in {32, 64, 128}; 1≤B≤16, 1≤H≤32, 1≤N≤8192; contiguous FP16 self-attention inputs; and sequence lengths divisible by the selected tile dimensions. Partial edge tiles are outside the fixed-tile contract.

## Verified results

| Gate | Evidence |
|---|---|
| Native cuTile smoke and required GPU suite | Pass; 57 native cuTile correctness tests on RTX 4060, REQUIRE_CUTILE=1 (including 64×32 across all seven divisible workload lengths) |
| Numerical case/tile matrix | 29/29 passed, zero mixed-tolerance violations |
| CLI | NPZ → cuTile → NPY verified against independent NumPy FP32 oracle |
| Trusted framework correctness reference | CUDA FP32 PyTorch SDPA vs NumPy: 7/7 workloads pass; max absolute discrepancy 0.00048828125 |
| PyTorch efficient-SDPA comparison | 10 configurations × 3 sessions; 30/30 paired correctness checks; backend observed as aten::_efficient_attention_forward |
| Benchmark matrix | 29 legal tile rows over 3 order-varied sessions; 13 pass both ≤5% stability filters |
| Clean-cache first call | Two long specializations measured separately from steady-state timing |
| Nsight Compute | 27 native RTX 4060 reports with per-profile Details/Raw exports and PTX/SASS analysis |
| GUI pane proof | 216/216 SASS/PTX pane checks across 27 profiles; zero failures; 486 images in 27 archives |
| Week 2 profiler-guided tuning | LOAD_LATENCY=10 accepted for medium D=128 / 64×64; global hint remains 0 |
| Week 3 profiler-guided tuning | Longer-batched D=128: 64×128 is 1.60% faster than 64×64 across three stable sessions; matched NCU duration is 0.68% lower, with higher register/shared-memory use and unchanged occupancy |
| Global defaults | 64×64 and LOAD_LATENCY=0; narrower optimizations remain workload-specific |

Expanded native GPU validation: [Actions run #42075683189](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42075683189) (57 passed, 29/29 numerical matrix). Matched Week 3 profiler pair: [run #42076178947](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42076178947), [artifact #13440087603](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42076178947/artifacts/13440087603). Earlier baseline native validation: [run #41499939303](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41499939303). The FP32 reference and efficient-SDPA recheck: [run #41511770773](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41511770773), with raw evidence published in [commit 5e834fb](https://github.com/xemptynothing/gpu-nvidia/commit/5e834fbfebe3e451edc14a60d3fc978149bee449). Latest CPU/reference CI: [run #42080908745](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42080908745) (801 passed, 70 expected skips).

The 29-row speedup matrix uses the deliberately unfused FP32 CuPy baseline. It does not claim those ratios against FlashAttention or optimized vendor attention. PyTorch efficient SDPA is an additional same-machine comparison, not the official speedup denominator. Sixteen benchmark rows have noisy ranges/drift and are retained but not promoted as headline results.

## Reproduction

Use the project-local Windows environment on the RTX 4060; no new CUDA/cuTile installation is needed.

~~~powershell
$py = ".\.venv-cutile-windows\Scripts\python.exe"
& $py scripts\cutile_smoke.py
$env:REQUIRE_CUTILE = "1"
& $py -m pytest -v -ra tests\test_gpu_correctness.py tests\test_gpu_edge_cases.py
& $py scripts\mentor_numerical_matrix.py --output .tmp\validation-matrix.json
& $py run_attention.py --input INPUT.npz --output OUTPUT.npy --backend cutile --tile-q 64 --tile-k 64
& $py scripts\verify_cli_output.py --input INPUT.npz --output OUTPUT.npy
~~~

## Important scope and limitations

The assignment's core scope is forward-only causal self-attention under the fixed-tile contract. Backward, arbitrary edge tiles, BF16/FP8, GQA/MQA, packed variable-length sequences and architecture-specific asynchronous pipelines are optional extensions. The remaining performance bottleneck is high per-CTA shared-memory/register use, low occupancy and a small eligible-warp pool; these limits are documented rather than hidden.

An additional owner-requested literal transcription of every detail in all 972 original NCU screenshots is not a requirement in the six-page PDF and remains separate from assignment completion. The 972 captures, 27 contact sheets, per-profile analyses and full-matrix pane-movement evidence are preserved.

## Source map

| Path | Purpose |
|---|---|
| src/cutile_attention.py | Fused cuTile kernel and device launch |
| src/contract.py | Assignment input and launch contract |
| src/reference.py | Independent full-matrix FP32 NumPy oracle |
| src/online_reference.py | Tiled online-softmax reference |
| src/torch_reference.py | CUDA FP32 PyTorch SDPA correctness reference |
| src/baseline.py | Deliberately unfused FP32 CuPy baseline |
| run_attention.py | NPZ input / NPY output CLI |
| tests/ | Contract, CPU, numerical and native GPU tests |
| benchmarks/, scripts/ | Timing, NCU, validation and evidence workflows |
| results/, docs/ncu/ | Raw evidence and profile analyses |

The older Report.md and dated October report are historical working notes. The final report linked above is the submission source of truth.

## Project tracking and evidence archive

The current assignment status is summarized above. Historical investigation notes, detailed test contracts and archived experiment data remain available here:

- [TODO checklist](docs/TODO_CHECKLIST.md)
- [Documentation index](docs/README.md)
- [Experiment report history](docs/EXPERIMENT_REPORT.md)
- [Legacy RTX 4060 results snapshot](docs/CUTILE_RTX4060_RESULTS_2026-10-19.md)
- [Testing contract](docs/TESTING_CONTRACT.md)
- [Test matrix](docs/TEST_MATRIX.md)
- [Progress log](docs/PROGRESS.md)
- [Historical status snapshot](results/STATUS.md)
