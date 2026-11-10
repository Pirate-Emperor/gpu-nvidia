# Current-source milestone evidence — updated 5 November 2026

**Assignment source:** nvidia2-cutile.pdf (six-page brief; SHA-256 b4bdef5a0ede51c5661dfc006b8556a437c23764476e0fee656bb907fb030946).  
**Latest expanded native RTX 4060 validation:** [#42075683189](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42075683189), passed 57 native cuTile correctness tests, 29/29 numerical configurations, CLI/NumPy oracle, three-session benchmarks, clean-cache JIT and profiler-hint measurements.  
**Matched Week 3 tile-choice NCU pair:** [#42076178947](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42076178947), [artifact #13440087603](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42076178947/artifacts/13440087603).  
**Framework FP32 reference:** [#41511770773](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41511770773), passed 7/7 workload families against the independent NumPy FP32 oracle.  
**Hosted CPU/reference CI:** [#42080908745](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42080908745), 801 passed, 70 expected skips.  
**Full screenshot evidence commit:** [058e6ea](https://github.com/xemptynothing/gpu-nvidia/commit/058e6ea6d87844db037e9a5b5ef043810663f4da).

The 5 November addenda below are the authoritative latest-source status. The earlier experiment tables are retained as a historical snapshot, not as the newest measurements.

## Historical snapshot — 2 November 2026 (superseded by the 5 November addenda)

## Hardware and software

| Item | Recorded value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060, compute capability 8.9 (SM89) |
| Driver | 616.92 |
| Python / NumPy | 3.12.10 / 2.5.4 |
| CuPy / CUDA Tile | 14.2.0 / 1.6.0 |
| CUDA Toolkit packages | 13.4.3 |
| CUDA runtime API / driver API | 13.2 (13020) / 13.4 (13040) |
| Windows | Windows 11 build 26300 |
| Sampled SM clock | 2700 MHz (point-in-time sample) |
| Power limit / draw | 115 W reported / unavailable ([N/A]) |

No locked-clock or fixed power-policy condition is claimed.

## Milestone 1 — kernel, contract and correctness

**Complete for the tested contract and workload matrix.**

- Genuine cuTile smoke compiled and launched on the RTX 4060; output check passed.
- Required actual-cuTile correctness suite: **57 passed**.
- Optional GQA/MQA/packed-variable-length GPU suite: **3 passed** (optional extensions, not core requirements).
- Independent numerical report: **29/29 legal case/tile configurations passed**, zero mixed-tolerance violations; worst maximum absolute error was **0.0009765625**.
- Deterministic NPZ → CLI → NPY path passed the independent NumPy output check.
- Independent legal D and Dv support is GPU-tested. Shared host and direct launch validation enforce PDF limits for B, H and N; boundary/just-outside contract tests pass.
- Current hosted CPU/reference suite: **801 passed, 70 skipped**. These skips include unavailable GPU packages on the hosted runner and reference cases whose tiles do not divide N; they are not counted as GPU proof.

## Milestone 2 — benchmark matrix

Current implementation measurements cover seven workload families and **29 legal case/tile rows**, in three order-varied sessions. Protocol: 10 warmups, 30 CUDA-event samples per configuration, five launches per sample, and an unfused FP32 CuPy baseline measured before and after the tile sweep. Speedup here means only speedup against that deliberately unfused CuPy baseline.

A conservative row is called stable when both three-session fused-median range and maximum baseline-bracket drift are ≤5%. **13/29 rows meet both checks; 16/29 are noisy.** All raw rows are retained; noisy rows must not be used as headline results.

| Representative case | Tile | Fused median | Unfused median | Speedup | Session range | Baseline drift |
|---|---:|---:|---:|---:|---:|---:|
| Long D=128, N=1024 | 64×64 | 0.266445 ms | 4.383078 ms | 16.45× | 0.53% | 0.063% |
| Longer batched D=128, N=2048 | 64×128 | 0.922214 ms | 17.109811 ms | 18.54× | 0.09% | 0.031% |
| Short D=32 | 64×64 | 0.015770 ms | 0.292509 ms | 18.48× | 3.90% | 2.56% |
| Short D=32 | 32×64 | 0.016877 ms | 0.292509 ms | 17.33× | 4.72% | 2.56% |
| Short D=64 | 32×64 | 0.015850 ms | 0.272538 ms | 18.55× | 1.57% | 1.85% |

The long and longer-batched D=128 families each have five stable tile rows; the short D=32 and D=64 families provide additional stable cases. This exceeds the six-configuration minimum while covering short/long and D=64/D=128. Medium D=128 shows 7–9% fused spread and about 10.37% baseline drift; medium D=64 shows 20–65% fused spread. Those rows remain in the raw dataset but are not reliable headline claims.

Raw data lives under results/rtx4060-validation/41499590898/, including the three session JSON/CSV pairs and aggregated summary JSON/CSV.

## Milestones 2–3 — profiler-guided optimization

### LOAD_LATENCY experiment

A three-session sweep evaluated hints 0, 2, 4, 6, 8 and 10 (10 warmups, 30 event samples and five launches/sample).

| Workload | Selected hint | Median | Hint-0 median | Gain | Decision |
|---|---:|---:|---:|---:|---|
| Medium D=128, 64×64 | 10 | 0.056320 ms | 0.060723 ms | 7.25% | Credible targeted gain |
| Longer batched D=128, 64×128 | 0 | 0.915866 ms | 0.915866 ms | Best alternative only 0.034% | Keep compiler default |

For medium D=128, the noise-derived required gain was 2.63%; hint 10 cleared it and its session range was 0.55%. Hint 10 is a workload-specific tuning choice, **not** a global default. The kernel API exposes the load_latency parameter; all other cases retain compiler-inferred scheduling (0).

### Matched Nsight Compute comparison

Current-source medium D=128 / 64×64 profiles compare hint 0 with hint 10.

| Metric | Hint 0 | Hint 10 |
|---|---:|---:|
| Kernel duration | 53.12 µs | 47.39 µs |
| DRAM throughput vs peak | 22.80% | 25.75% |
| Memory workload throughput | 60.06 GB/s | 67.93 GB/s |
| SM compute throughput vs peak | 18.85% | 21.10% |
| Eligible warps / scheduler | 0.14 | 0.16 |
| No-eligible scheduler cycles | 85.85% | 84.06% |
| Warp cycles / issued instruction | 7.09 | 6.28 |
| Registers / thread | 231 | 238 |
| Static shared memory / CTA | 65.54 KiB | 98.30 KiB |
| Achieved occupancy | 8.32% | 8.35% |
| Local-memory spills | 0 | 0 |

The matched NCU duration is about 10.8% lower for hint 10. Throughput and eligible warps improve modestly, while static shared memory rises about 50%; occupancy remains around 8.3%, with no local spills. This supports a latency/scheduling improvement for the medium specialization, not a global setting or a universal bandwidth-bound diagnosis.

Two current-source profiles were also captured for medium D=128 / 64×64 and longer-batched D=128 / 64×128. The repository retains 27 earlier native NCU reports and per-configuration analyses.

**Remaining bottleneck:** the medium profile has high register use and large static shared memory, about one CTA/four active warps per SM and about 8.3% occupancy. DRAM and SM throughput are far below peak; the kernel is not saturating memory bandwidth or compute. High no-eligible-warp cycles and modest eligible warps point to latency/parallelism pressure. Reducing per-CTA shared-memory footprint or improving eligible warps is the next optimization hypothesis; it requires a matched benchmark and correctness rerun.

## Clean-cache first-call measurements

Each specialization ran in a fresh Python process with initially empty cuTile and CUDA driver cache directories. Context creation, imports and input preparation were excluded. The first-call timer includes compilation, first launch and synchronization; it is not a compiler-only duration.

| Specialization | First call | Steady median | Steady P90 | NumPy check |
|---|---:|---:|---:|---|
| Long D=128 / 64×64 | 2170.951 ms | 0.287027 ms | 0.289371 ms | Pass |
| Longer batched D=128 / 64×128 | 3059.894 ms | 0.988467 ms | 0.994714 ms | Pass |

Both workers verified empty cuTile and driver-cache directories before timing.

## Framework/vendor attention reference

The focused framework run used PyTorch 2.14.0+cu130 with CUDA build 13.0 on the same RTX 4060. Backend identification was captured by torch.profiler as **aten::_efficient_attention_forward** (PyTorch efficient SDPA), not FlashAttention.

- Ten configurations, three order-varied sessions and 30 paired raw rows.
- Each backend/config/session used 10 warmups, 30 CUDA-event samples and five launches per sample.
- All 30 SDPA-vs-NumPy, cuTile-vs-NumPy and cuTile-vs-SDPA correctness checks passed.
- cuTile was faster than PyTorch efficient SDPA in all ten configuration medians, with observed ratios from **1.08× to 2.08×**.
- Six configurations had ≤5% session spread for both backends; short/medium outliers remain flagged in the JSON and must not be overstated.

Representative medians:

| Case / tile | PyTorch efficient SDPA | cuTile | cuTile advantage |
|---|---:|---:|---:|
| Medium D=128 / 64×64 | 0.070301 ms | 0.058931 ms | 1.19× |
| Long D=128 / 64×128 | 0.291558 ms | 0.270173 ms | 1.08× |
| Longer batched D=128 / 64×128 | 1.011546 ms | 0.921734 ms | 1.10× |
| Short D=32 / 64×64 | 0.044653 ms | 0.021446 ms | 2.08× |
| Short D=64 / 64×64 | 0.069901 ms | 0.036413 ms | 1.92× |

Full outputs are published under results/sdpa-reference/41501308083/. This framework comparison is supplementary; the assignment's speedup denominator remains the deliberately unfused FP32 CuPy baseline.

## Full 27-profile SASS/PTX pane movement evidence

The original screenshot set was validated and published without repeating the GUI capture:
- 27 profiles, both SASS and PTX.
- Four panes per mode: left code, left metrics, right code and right metrics.
- 216 pane movement/restore checks with zero recorded failures.
- 486 original PNGs at 1936×1048.
- 27 per-profile ZIP archives, 18 images each.

Index and archives: [verified pane evidence README](ncu/rtx4060/horizontal-proof-2026-11-02/README.md). This closes full-matrix pane movement proof. It does **not** close the user's separate request for literal manual transcription of every instruction and small visual detail in all 972 original full-resolution GUI screenshots. That manual transcription remains open; contact sheets and representative full-resolution checks are not substitutes.

## Assignment status

| Milestone | Status | Evidence |
|---|---|---|
| Week 1 — implementation and correctness | Complete for tested contract/workload matrix | 57 native GPU tests (includes 64×32 on all seven divisible lengths), 29/29 accuracy configurations, CLI, smoke, host/device contract |
| Week 2 — initial benchmark and first optimization | Complete | 29 tile rows; 13 stable; first hint optimization measured with current-source three-session timing and matched NCU evidence |
| Week 3 — iterative profiler-guided optimization | Complete for the tested two-step tuning loop | Week 2 medium-D128 `LOAD_LATENCY=10`; Week 3 longer-batched D128 64×128 tile choice supported by stable three-session benchmarking and matched NCU. Trade-offs documented; global defaults unchanged. |
| Week 4 — final reproducibility, analysis and report | Complete for PDF scope | Current-source results, reproducibility instructions, limitations, verifiable contribution attribution and final 5–8 page report/PDF are published |

The six-page assignment scope is complete and the final 5–8 page report/PDF is generated. The only remaining project item tracked here is the separate owner-requested literal transcription of all 972 GUI screenshots; that task is not a requirement in the assignment PDF. Contributor attribution is limited to verifiable Git history rather than invented off-repository percentages.


## Addendum — CUDA FP32 framework reference verification (3 November 2026)

The assignment's independent trusted framework reference was executed explicitly, not inferred from the FP16 performance comparison. Focused native RTX 4060 workflow [#41511770773](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41511770773) invoked src/torch_reference.py::forward for seven distinct workload families. It copied FP16 Q/K/V to CUDA FP32 tensors, used torch.nn.functional.scaled_dot_product_attention(is_causal=True, dropout_p=0.0), cast the result back to FP16, and checked it against src/reference.py's NumPy FP32 oracle. **All seven workloads passed** the fixed mixed tolerance abs(error) <= 0.02 + 0.02*abs(reference); the maximum absolute discrepancy was **0.00048828125**.

The same run re-executed the FP16 efficient-SDPA benchmark: 10 legal config/tile pairs × 3 order-varied sessions = 30 raw rows; all SDPA-vs-NumPy, cuTile-vs-NumPy, and cuTile-vs-SDPA checks passed. torch.profiler observed aten::_efficient_attention_forward on PyTorch 2.14.0+cu130 / CUDA 13.0. This is efficient SDPA, not a claim of FlashAttention. Raw JSON/CSV, per-case FP32-reference errors, status and environment logs were published under results/sdpa-reference/41511770773/ by [evidence commit 5e834fb](https://github.com/xemptynothing/gpu-nvidia/commit/5e834fbfebe3e451edc14a60d3fc978149bee449).


## Addendum — expanded native validation and Week 3 second optimization (5 November 2026)

### Expanded native cuTile verification

Native RTX 4060 workflow [#42075683189](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42075683189) passed **57 GPU correctness tests** in 14.67 seconds. This expands the optional 64×32 exploratory configuration from one test length to all seven MANDATORY_CASES whose N is divisible by both tile dimensions. The mandatory numerical matrix remains **29/29 passed**, zero mixed-tolerance violations, worst max absolute error 0.0009765625. CLI/NumPy oracle, clean-cache first-call workers, the three-session benchmark matrix, PyTorch FP32 framework reference, hint sweep and the standard two NCU captures also passed. Logs and JSON/CSV outputs are under results/rtx4060-validation/42075683189/.

### Week 3 second profiler-guided improvement: tile choice for longer-batched D=128

The assignment requires a second distinct Week 3 profiler-guided optimization loop. This is separate from the Week 2 LOAD_LATENCY=10 hint tuning. On the expanded current source, the longer-batched case (B=2,H=8,N=2048,D=Dv=128) compared tiles 64×64 and 64×128. Both were correct in the 29-row numerical matrix.

| Tile | Fused median | Session range | Max baseline-bracket drift | Speedup vs unfused FP32 CuPy |
|---|---:|---:|---:|---:|
| 64×64 | 0.936346 ms | 0.525% | 0.055% | 18.29× |
| 64×128 | 0.921395 ms | 0.456% | 0.055% | 18.60× |

The 64×128 tile improved the stable three-session median by **1.60%**. A same-source matched NCU pair [#42076178947](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42076178947), artifact [#13440087603](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42076178947/artifacts/13440087603), measured:

| Metric | 64×64 | 64×128 |
|---|---:|---:|
| NCU duration | 997.73 µs | 990.98 µs |
| Memory throughput | 19.65% | 20.21% |
| DRAM throughput | 11.67% | 11.81% |
| SM throughput | 29.03% | 30.23% |
| Registers/thread | 227 | 255 |
| Static shared memory/CTA | 65.54 KiB | 81.92 KiB |
| Achieved occupancy | 8.33% | 8.33% |
| Eligible warps/scheduler | 0.16 | 0.15 |
| No-eligible cycles | 84.48% | 85.24% |

The profiler shows a modest duration and throughput improvement, but the larger tile increases registers and shared memory, does not improve occupancy and slightly worsens scheduler eligibility. Accept 64×128 only for the longer-batched D=128 workload; keep 64×64 as the global default. Two isolated paired retries are recorded as non-headline variance results because one or both choices failed the predeclared 5% spread gate (runs [#42073544473](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42073544473) and [#42073654984](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42073654984)).

This leaves the central bottleneck unchanged: low eligible-warp count and CTA resource pressure (register/shared-memory footprint), not saturated DRAM bandwidth. See [the full Week 3 NCU analysis](docs/ncu/rtx4060/WEEK3_TILE_CHOICE_2026-11-05.md).
