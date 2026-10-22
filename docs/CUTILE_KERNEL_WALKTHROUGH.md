# cuTile Attention Kernel — Mentor Walkthrough

**Source:** [`src/cutile_attention.py`](../src/cutile_attention.py)  
**Comparison baseline:** [`src/baseline.py`](../src/baseline.py)  
**Independent numerical reference:** [`src/reference.py`](../src/reference.py)

This walkthrough explains the code that actually executes on the native Windows RTX 4060. It intentionally excludes Nsight Compute; the profiler demonstration is a separate session.

## 1. What operation is being implemented?

The kernel implements causal scaled dot-product self-attention:

1. Compute the score matrix (S=QK^T/\sqrt{D}).
2. Mask future keys for each query.
3. Apply a row-wise softmax to get probabilities (P).
4. Multiply by (V), giving (O=PV).

For a sequence of length (N), a conventional implementation materializes score/probability matrices with (N\times N) entries per head. The fused cuTile kernel forms only one `TQ × TK` score tile at a time, updates the running softmax state and writes the output tile. It does not store the full (N\times N) score or probability matrix in global memory.

## 2. Start with the baseline: `src/baseline.py`

The baseline is deliberately **unfused FP32 CuPy attention**, not FlashAttention and not optimized PyTorch SDPA.

| Step in baseline | Why it is there |
|---|---|
| Convert Q/K/V to FP32 for the comparison path | Reduce input-precision differences in the reference timing path. |
| `scores = cp.matmul(q, cp.swapaxes(k, -1, -2))` | Materialize the full (N\times N) score matrix. |
| Multiply by (1/\sqrt{D}) | Apply scaled dot-product attention's standard score scale. |
| Build an upper-triangular future-key mask | Enforce causal semantics. |
| `max → exp → sum → divide` | Stable row-wise softmax. |
| `cp.matmul(probs, v)` | Weighted sum of value vectors. |
| Cast output to FP16 | Match the project's output contract. |

Each CuPy array operation may become one or more GPU operations; the baseline intentionally does not fuse them into one attention kernel. The benchmark measures both paths on already-resident GPU arrays, so the headline device-latency comparison is not dominated by copying Q/K/V from the CPU for every timed call.

**Interpretation limit:** a measured cuTile speedup is a comparison with this unfused baseline only. It is not evidence of a speedup over a vendor-optimized attention implementation.

## 3. Read the cuTile implementation in execution order

The public route is `forward → launch → launch_device → _build_kernel`.

### 3.1 Validation and specialization

- `forward` accepts NumPy FP16 Q/K/V, calls the launch path, synchronizes the CUDA stream and returns a NumPy FP16 output.
- `launch` validates the host-array contract and copies Q/K/V to the device.
- `launch_device` checks the device-array dimensions, matching Q/K/V sequence and head dimensions, the supported head dimensions (D\in\{32,64,128\}), the self-attention condition (Dv=D), power-of-two tile sizes and divisibility of (N) by both tile dimensions.
- `_build_kernel` is cached by tile shape and dimensions. cuTile specializes the kernel with `TQ`, `TK`, (D), and (Dv); this explains why the first use of a specialization can be much slower than later launches.
- The launch grid is `(batch, heads, sequence_length // tile_q)`: one program instance owns one query tile for one batch item and one query head.

The divisibility checks are intentional. Partial edge tiles are not silently claimed to be supported by this core contract. Illegal shapes should fail clearly before a kernel launch.

### 3.2 Load a query tile and initialize online softmax

Each program instance obtains its batch index, head index and query-block index from `ct.bid`. It loads (Q_t\in\mathbb{R}^{TQ\times D}).

For every query row, the online-softmax state begins as:

- (m=-\infty): running maximum score;
- (\ell=0): running softmax normalization sum;
- (u=0): running unnormalized weighted-value sum.

These three pieces of state are enough to combine later key/value tiles without saving every score or probability.

### 3.3 Skip key/value tiles that are entirely in the future

The loop bound is computed as `nkt = ct.cdiv((qb + 1) * tq, tk)`. The end of the query tile is an exclusive index. The bound therefore includes every key tile that can contain a key at or before at least one query in this query block, but omits later tiles that are wholly beyond every query row's causal boundary.

This is a semantic optimization, not an approximate mask: a tile that cannot contribute under causal attention is not loaded or multiplied at all.

### 3.4 Compute a tile of scores and apply the causal mask

For each key/value block, the kernel loads (K_t) and (V_t), transposes the key tile for matrix multiplication, then executes the equivalent of:

[
S_t = Q_tK_t^T/\sqrt{D}.
]

The row query indices and column key indices are built from tile-local indices plus their block offsets. For any score whose key index is larger than its query index, the kernel substitutes (-\infty). Its exponent is therefore zero after the max-shifted softmax step.

### 3.5 Update the running softmax without storing a full probability matrix

For the current masked score tile, the kernel computes its row-wise maximum (m_t), then the new maximum (m'=\max(m,m_t)). It rescales old state to this maximum:

[
\alpha=\exp(m-m'),\qquad
P_t=\exp(S_t-m').
]

The normalization and weighted-value state are updated as:

[
\ell'=\alpha\ell+\sum_j P_{t,j},\qquad
u'=\alpha u+P_tV_t.
]

The rescaling is important. The old and new tiles may have different score maxima; they must be expressed on the same exponential scale before being combined. At the end, the output tile is (O_t=u/\ell).

The online maximum, normalization and value accumulator use FP32. The current implementation casts the probability operand to FP16 for the (P_tV_t) matrix multiplication while keeping the matrix multiplication accumulator in FP32. This mixed-precision choice is retained because the independent numerical gates passed; it is not justified by a type name alone.

### 3.6 Store the result

The normalized output is reshaped to the 4-D output tile and stored as FP16. The output allocation matches ((B,H,N,Dv)).

## 4. Algorithm sketch to say aloud

```text
for each (batch, head, query_tile):
    load Q tile
    initialize running maximum m, normalization l, value accumulator u
    for each K/V tile that can contribute causally:
        load K and V tile
        score = Q_tile @ K_tile.T / sqrt(D)
        mask keys where key_index > query_index with -infinity
        merge this tile into (m, l, u) using max-shifted online softmax
    write FP16(u / l)
```

## 5. Why these tests are meaningful

There are **48 parameterized test executions** across two files, not 48 distinct Python test functions. The files are run independently in the walkthrough so the mentor can see each assertion family and its own pytest summary.

### `tests/test_gpu_correctness.py`: 34 executions

| Family | Count | Why it matters |
|---|---:|---|
| Every legal mandatory case/tile pair vs independent NumPy reference | 29 | Checks numerical output across all seven representative workload shapes and the supported tile candidates that divide the sequence length. |
| Exploratory (64×32) tile | 1 | Checks an asymmetric legal tile, beyond the main candidate set. |
| Small and large score magnitudes | 2 | Looks for instability, overflow or non-finite output under very different logit scales. |
| Change future values and check earlier query outputs | 1 | Directly tests causal isolation. |
| Reject a non-divisible sequence | 1 | Confirms the fixed-tile launch contract is enforced rather than silently producing an incomplete result. |

### `tests/test_gpu_edge_cases.py`: 14 executions

| Family | Count | Why it matters |
|---|---:|---|
| Constant (V), for (D=32,64,128) | 3 | A weighted average of a constant vector must return that same vector, regardless of attention weights. |
| Zero logits | 1 | Equal scores imply uniform probabilities over the causal prefix; the output should be the prefix average of (V). |
| Four tile choices compared to the independent reference and each other | 1 | Detects tile-dependent logic or numerical divergence. |
| Future (V) mutation | 1 | Future values must not influence earlier outputs. |
| Future (K) mutation | 1 | Future keys must not influence earlier outputs. |
| NaN, (+\infty), and (-\infty) rejection | 3 | Invalid values must be rejected before they corrupt kernel arithmetic. |
| Non-contiguous input rejection | 1 | Enforces the memory-layout contract expected by the implementation. |
| Unsupported (D) or (Dv\ne D) | 2 | Rejects unsupported dimensions and a value width inconsistent with self-attention. |
| Non-divisible sequence under three tile settings | 1 | Ensures both query and key tile divisibility requirements are enforced. |

The native GPU run set `REQUIRE_CUTILE=1`. This is deliberate: missing CuPy or cuTile must fail the required GPU gate, not produce a misleading green result because pytest skipped the file.

## 6. Numerical contract and recorded outcome

The independent NumPy reference does not call the cuTile kernel to generate expected values. Every legal case/tile output is checked for shape, FP16 dtype and finiteness before the mixed tolerance:

[
|O_{\text{actual}}-O_{\text{reference}}|
\le 0.02+0.02|O_{\text{reference}}|.
]

Recorded native RTX 4060 result:

| Metric | Recorded outcome |
|---|---:|
| Core GPU tests | 48 passed, 0 failed |
| Independent legal case/tile configurations | 29/29 passed |
| Maximum absolute error | 0.0009765625 |
| Largest per-configuration P99 absolute error | 0.00048828125 |
| Largest per-configuration RMSE | 0.000086860741 |
| Tolerance violations | 0 |

The maximum tolerance-envelope ratio was 2.4367%: this means the worst error consumed 2.4367% of its *allowed tolerance*, not that the result had 2.4367% output error.

## 7. What the performance experiments showed

These are recorded three-session measurements, not guaranteed outcomes for a fresh run.

### 7.1 Tile-size experiment

| Workload | Tile | Median | P90 | Speedup vs unfused FP32 CuPy |
|---|---:|---:|---:|---:|
| Medium (D=128) | 32×64 | 0.069632 ms | 0.070659 ms | 5.62× |
| Medium (D=128) | 64×64 | 0.058368 ms | 0.058814 ms | 6.70× |
| Medium (D=128) | 64×128 | 0.058368 ms | 0.059044 ms | 6.70× |
| Medium (D=128) | 128×64 | 0.079053 ms | 0.079258 ms | 4.94× |
| Medium (D=128) | 128×128 | 0.107315 ms | 0.107950 ms | 3.64× |
| Longer batched (D=128) | 32×64 | 1.142989 ms | 1.146900 ms | 14.97× |
| Longer batched (D=128) | 64×64 | 0.932966 ms | 0.936468 ms | 18.33× |
| Longer batched (D=128) | 64×128 | 0.915558 ms | 0.921457 ms | 18.69× |
| Longer batched (D=128) | 128×64 | 1.017139 ms | 1.020314 ms | 16.83× |
| Longer batched (D=128) | 128×128 | 1.285120 ms | 1.289237 ms | 13.32× |

**Decision:** keep 64×64 as the general default. The 64×128 tile was about 1.86% faster for the longer batched workload, but 64×64 was better for the long unbatched workload and tied 64×128 in the medium D=128 median. The 128×128 tile was consistently worse on these long D=128 cases. Bigger is not automatically faster.

### 7.2 DRAM load-latency hint experiment

The `benchmarks/benchmark_load_latency_hints.py` experiment first checks the output against independent NumPy and then tests hints `0, 2, 4, 6, 8, 10` for two fixed workloads. Each candidate uses three order-rotated sessions, 10 warmups, 30 CUDA-event samples and five launches per sample.

| Workload | Default hint 0 | Best measured candidate | Observed gain | Decision |
|---|---:|---:|---:|---|
| Medium (D=128), tile 64×64 | 0.058342 ms | Hint 10: 0.053248 ms | 8.73% | Timing candidate; do not promote globally until matched profile review |
| Longer batched (D=128), tile 64×128 | 0.922010 ms | Hint 2: ~0.920576 ms | 0.16% | Keep hint 0; gain is below observed variation |

Hint 10 on the medium workload is the interesting result. On the longer batched case, the observed improvement is much smaller than the noise-aware threshold. The kernel therefore retains compiler-inferred `LOAD_LATENCY=0` as its default.

### 7.3 Experiments that were not promoted

- **128×128 tile:** higher latency and greater shared-memory/resource pressure on the long D=128 workloads.
- **Shared-tile v4 / v5 prototypes:** these are historical ordinary-CUDA/A100 control experiments, not actual-cuTile evidence. Correctness passed in their own workflows, but the resource/timing results were mixed; they were not promoted.
- **PyTorch SDPA comparison:** currently blocked before the cuTile bridge because a plain `torch.bmm` fails with `CUBLAS_STATUS_INVALID_VALUE` in the pinned CUDA 13.0 PyTorch wheel. No PyTorch speedup is claimed.
- **First-call/JIT:** measured separately from steady-state latency, but historical first-call observations sometimes look like cache hits. Do not call them clean-cache compile times.

### 7.4 Historical baseline-ladder experiments (ordinary CUDA controls, not cuTile)

The repository also retains v1/v2/v3/current and v4/v5 experiments from the earlier CUDA C++ implementation ladder. They help explain why kernel structure matters, but **they were run as ordinary CUDA C++ control kernels on the A100 path—not as the current NVIDIA cuTile Python kernel on the RTX 4060**. Their timings must not be placed in the same speedup ranking as the actual-cuTile results above.

Best observed median per approach in one historical ladder run (ms; lowest measured median over `tile_k=16/32/64` within that run):

| Workload | v1 | v2 | v3 | Current CUDA control |
|---|---:|---:|---:|---:|
| short D=32 | 0.111 | 0.066 | 0.055 | 0.055 |
| short D=64 | 0.119 | 0.073 | 0.060 | 0.061 |
| short multi-head D=64 | 0.385 | 0.102 | 0.077 | 0.081 |
| medium D=64 | 1.727 | 0.234 | 0.180 | 0.193 |
| medium D=128 | — | 0.947 | 0.815 | 0.792 |
| long D=128 | — | 7.188 | 6.556 | 6.131 |
| longer batched D=128 | — | 22.895 | 26.533 | 25.678 |

The earlier code ladder changed the work decomposition progressively. The measured first-compile wall times in that ladder run were v1 1652 ms, v2 717 ms, v3 1063 ms and current 808 ms. Those are compile/JIT-related times and must not be reported as steady-state kernel latency. Absolute medians differed across workflow runs, so use these as historical engineering evidence rather than current RTX 4060 headline results.

The v4 persistent-Q-fragment variant passed its 30 correctness tests but had mixed performance: it was near the current control on short cases, faster on some D=128 cases and substantially slower on longer-batched inputs with smaller K tiles. Its PTXAS report found no resource advantage over the then-current code: v2/current/v4 each reported 32 registers, 33,792 bytes shared memory and zero spills; current and v4 cubins were both 15,136 bytes. Decision: **not promoted**.

The v5 half2-oriented Q/K-load variant passed 30/30 correctness tests and a three-session paired timing study. Its results were mixed for short inputs. On long D=128, measured improvements were only about 0.3–0.4%; on longer-batched D=128 they were about 0.4–0.6%. Some medium-D=64 rows improved by roughly 2–11%, but these gains did not establish a reliable, general speedup. Decision: **not promoted**. The full per-case/per-tile v5 table remains in [the historical experiment report](EXPERIMENT_REPORT.md).

These negative results matter: passing correctness is necessary but not sufficient to promote an optimization. An optimization should change one well-defined factor, pass the same independent correctness tests, beat the existing implementation by more than observed timing variation, and avoid disproportionate resource cost.

## 8. Suggested mentor narrative

> “The baseline explicitly materializes the full score and probability matrices, while the cuTile kernel streams K/V tiles and fuses score calculation, causal masking, softmax-state updates and the weighted-value product. The running maximum, normalization and value accumulator are the online-softmax state. I first validate every legal shape and tile against independent NumPy, then benchmark the already-resident GPU functions. The measurements show workload-dependent tile choices, not a universal winner. The latency-hint experiment found a promising medium-workload candidate, but the global default is unchanged until that candidate clears the remaining review.”

## 9. Evidence and related pages

- [Step-by-step mentor walkthrough](MENTOR_STEP_BY_STEP_WALKTHROUGH.md)
- [Testing contract](TESTING_CONTRACT.md)
- [Test matrix / validation ledger](TEST_MATRIX.md)
- [Recorded RTX 4060 results](CUTILE_RTX4060_RESULTS_2026-10-20.md)
- [Complete experiment report](../Report.md)
- [Runtime tile-sweep guide](MENTOR_RUNTIME_TILE_SWEEP.md)
