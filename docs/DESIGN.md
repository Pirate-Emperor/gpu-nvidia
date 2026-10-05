# NVIDIA cuTile Fused Attention — Full Design Overview

## 0. Goal

Implement the course-required forward pass of causal scaled dot-product attention as a fused cuTile kernel, optimized and validated on NVIDIA A100 (SM80).

The project has two layers:

1. **Course-contract layer:** exact required behavior, interface, correctness, benchmark, and profiling requirements.
2. **Performance/research layer:** profiler-guided A100 optimization and carefully isolated extensions.

The mandatory implementation is never weakened by an extension.

## 1. Functional contract

Inputs:
- Q: [B, H, Nq, D], contiguous, float16
- K: [B, H, Nk, D], contiguous, float16
- V: [B, H, Nk, Dv], contiguous, float16

Required workload:
- Causal self-attention only.
- Nq == Nk for the required path.
- Supported sequence lengths must be divisible by the selected query/key tile sizes.
- Reject unsupported shapes/dtypes before kernel launch with a clear nonzero exit.

Reference:
Attention(Q,K,V) = softmax((QK^T)/sqrt(D) + causal_mask) V.

Output:
- [B, H, Nq, Dv]
- One .npy output file.
- Output dtype selected once, documented, and used consistently.

CLI:
python run_attention.py --input INPUT.npz --output OUTPUT.npy --backend cutile

## 2. Kernel architecture

Logical execution:
- Grid identifies attention work units (batch/head/query-tile).
- Each block owns a Q tile and streams K/V tiles across the reduction dimension.
- Q/K/V are loaded from global memory into cuTile tiles.
- Score tile = Q_tile @ K_tile^T.
- Apply scale 1/sqrt(D).
- Apply causal mask based on absolute query/key indices.
- Update online-softmax state.
- Accumulate weighted V tile.
- Store normalized output tile once.

No full N x N score or probability matrix is stored in global memory.

The online state per query row is:
- running max m
- running normalization l
- running output accumulator o

For each new score tile with tile max m_t:
m_new = max(m, m_t)
l_new = exp(m - m_new) * l + sum(exp(s_tile - m_new))
o_new = exp(m - m_new) * o + sum(exp(s_tile - m_new) * V_tile)

Final output:
o / l

Exact implementation equations will be kept in the code/report comments and independently validated against a trusted FP32 framework reference.

## 3. Precision

Mandatory:
- Q/K/V storage: FP16
- Accumulation/reduction state: FP32

No fast-math/numerical shortcuts will be introduced until a correctness baseline is established.

A separate BF16 extension may be added after all mandatory tests pass.

FP8 is not a target optimization for the A100 mandatory environment.

## 4. Host/runtime architecture

Repository components:

src/
  cutile_attention.py or equivalent kernel/launch implementation
  reference.py
  baseline.py
  io.py
  validation.py

tests/
  test_shapes.py
  test_correctness.py
  test_rejection.py
  test_numerical_stability.py
  test_regression.py

benchmarks/
  benchmark_attention.py
  benchmark_configs.json
  collect_results.py

profiles/
  ncu_commands.md
  metric_schema.md

scripts/
  environment_check.py
  benchmark_sweep.py
  with_core_config.py

docs/
  DESIGN.md
  TESTING_CONTRACT.md
  OPTIMIZATION_LOG.md

.github/workflows/
  gpu-smoke.yml
  gpu-correctness.yml
  gpu-benchmark.yml

The exact filenames may change during implementation, but the separation of kernel, reference, validation, benchmark and profiling responsibilities should remain.

## 5. Correctness architecture

Correctness is a hard gate.

Every kernel change must pass:
1. input validation tests,
2. shape/edge tests,
3. trusted FP32 reference comparison,
4. NaN/Inf checks,
5. numerical-stability tests,
6. regression suite.

The trusted reference must be independent of the cuTile implementation. It must compute FP32 intermediates and use an explicit causal mask.

We will never use the optimized kernel to generate its own expected answers.

## 6. Test taxonomy

### A. Contract tests
- Correct CLI.
- Missing input/output.
- Wrong dtype.
- Wrong rank.
- Non-contiguous input rejection or explicit normalization policy.
- Nq/Nk mismatch for required causal self-attention.
- Unsupported head dimension.
- Non-divisible sequence/tile dimensions.
- Unsupported tile configuration.
- Correct nonzero exit status.

### B. Shape tests
Across every required head dimension:
- sequence length exactly one tile;
- multiple tiles;
- short sequence;
- long sequence;
- batch = 1 and >1;
- heads = 1 and >1.

### C. Causal-mask tests
- Tiny hand-computable examples.
- Strictly upper triangular positions must contribute zero.
- Lower triangle and diagonal must contribute.
- Different query rows must see different valid key ranges.

### D. Numerical tests
- normal random inputs;
- very small magnitudes;
- large positive/negative logits;
- constant Q/K/V;
- repeated values;
- mixed-sign values;
- NaN/Inf rejection;
- fixed seeds.

### E. Regression tests
- Every previously fixed failure becomes a permanent test.
- Every optimization is tested against the same oracle.

### F. Performance sanity tests
- Warm-up before timing.
- GPU synchronization.
- CUDA-event timings.
- Median over >=30 iterations.
- JIT compilation measured separately.
- Kernel-only timing kept separate from any optional end-to-end timing.

## 7. Test oracle policy

For each test:
- Generate CPU-side NumPy inputs with a fixed seed.
- Copy the exact same values to GPU.
- Compute reference in FP32.
- Run cuTile kernel.
- Copy output back.
- Check shape, dtype, finite values.
- Check absolute and relative error against a documented mixed tolerance.

Critical hand-written tests will include tiny dimensions where the causal result can be independently inspected element-by-element.

Before finalizing the tolerance, we will collect empirical FP16-input/FP32-accumulation error distributions and compare them against the course's recommended tolerance.

## 8. Mandatory benchmark matrix

At minimum:
- short sequence configuration;
- long sequence configuration;
- head dimension 64;
- head dimension 128;
- >= 6 representative tile-divisible configurations.

Each configuration:
- warm-up;
- >=30 measured iterations;
- median latency;
- speedup vs unfused baseline;
- effective output throughput;
- JIT compilation time separately.

References:
- unfused attention baseline;
- vendor/framework implementation as performance reference.

## 9. Profiling plan

Profile at least:
- one representative short/medium workload;
- one long workload;
- additional configuration when a major optimization is introduced.

Collect:
- global-memory traffic;
- achieved occupancy;
- arithmetic throughput;
- memory throughput;
- register/resource usage where available;
- limiting bottleneck.

Every optimization claim must have:
1. before measurement,
2. code/configuration change,
3. after measurement,
4. profiler evidence,
5. explanation.

## 10. A100 optimization roadmap

Phase 0: Environment validation
- Identify A100/SM80.
- Record driver, CUDA, Python, cuTile/TileIR versions.
- Run NVIDIA cuTile smoke sample.
- Run Nsight Compute smoke profile.

Phase 1: Correct baseline
- Reference implementation.
- Unfused baseline.
- Minimal correct cuTile kernel.
- Mandatory correctness suite.

Phase 2: Initial tile exploration
- At least two legal tile configurations.
- Benchmark and profile.
- Establish baseline bottleneck.

Phase 3: A100 optimization
- Sweep legal tile shapes.
- Investigate work assignment.
- Investigate memory access/reuse.
- Investigate compiler configuration/resource trade-offs.
- Keep configuration/result metadata machine-readable.

Phase 4: Autotuning
- Build workload-aware configuration selection.
- Search only configurations known to satisfy correctness/shape constraints.
- Cache benchmark results.
- Select fastest stable configuration, not one based on a single noisy run.

Phase 5: Precision extension
- Add BF16 path only after FP16 path is fully stable.
- Re-run full correctness and benchmark matrices.

Phase 6: OEC-aligned extensions
- GQA/MQA.
- KV-cache/decode-oriented attention.
- asymmetric spatial-Q/text-KV cross-attention.
- MLA-inspired compressed-KV research path.

Extensions remain separate backends/experiments and cannot alter mandatory behavior.

## 11. Performance methodology

All measurements must record:
- GPU name and architecture;
- driver;
- CUDA version;
- cuTile version;
- Python version;
- tile configuration;
- compilation mode;
- clock/power policy when known;
- git commit;
- test seed;
- benchmark configuration.

Use median latency rather than a single observation.

We will preserve raw benchmark results as CSV/JSON so plots are reproducible.

## 12. Reproducibility

The repository must provide:
- environment verification script;
- deterministic test seed;
- benchmark configuration file;
- one-command correctness run;
- one-command benchmark sweep;
- profiling command documentation;
- raw results;
- optimization log.

A clean checkout must reproduce the documented results within a stated variance envelope.

## 13. CI / self-hosted A100 runner

The intended runner:
- name: gpu-project
- labels: gpu,a100
- repository: SpryzenHell/gpu-nvidia
- working directory: ~/gpu/project/_work

GPU jobs should initially be manually dispatched to avoid accidental competition with unrelated workloads.

Network access from the compute node requires importing the site's core_config before invoking commands that need the configured proxy.

For subprocess-based commands, use a wrapper that imports core_config and then launches the child process so proxy environment variables are inherited.

No proxy URLs, access tokens, or GitHub runner registration tokens are stored in the repository.

## 14. OEC/MTP relationship

OEC is not the execution environment for this project.

Only the research extensions are motivated by OEC workloads:
- OEC MLA -> compressed-KV experiment.
- OEC spatial-Q/text-KV -> asymmetric cross-attention.
- OEC inference workloads -> KV-cache/decode study.

The NVIDIA project remains independently reproducible on one A100.

## 15. Definition of done

Mandatory:
- all course requirements pass;
- all correctness tests pass;
- no NaN/Inf output;
- required rejection cases pass;
- benchmark methodology is compliant;
- required profiler evidence exists;
- clean checkout reproduces tests.

Performance/research:
- at least one profiler-backed optimization;
- tile sweep;
- autotuning result;
- detailed A100 analysis;
- optional extensions isolated and benchmarked;
- optimization log includes failures and lessons.

## 16. Non-negotiable correctness principles

1. Never optimize before establishing a trusted oracle.
2. Never remove a test because an optimization makes it inconvenient.
3. Never change tolerance solely to make a failing optimization pass.
4. Never compare against a value produced by the optimized kernel.
5. Never claim speedup without repeated measurements.
6. Never mix mandatory and experimental behavior in a way that can alter the course-required interface.
7. Every discovered bug becomes a regression test.
