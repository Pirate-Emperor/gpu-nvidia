# Exhaustive Test Plan

## Oracle validation
- Independent scalar FP64 causal-attention oracle for small tensors.
- Vectorized FP32 reference is compared against the scalar oracle for N=1..12, D={1,2,3,4,8}, Dv={1,2,5}, and B/H combinations {(1,1),(1,2),(2,1)}.
- Online-softmax reference is compared against the scalar oracle for N=1..16, multiple D/Dv pairs, and every legal query/key tile pair formed from divisors of N.
- Explicit regression covers an entirely future K-tile, which previously exposed an online-softmax NaN-state bug.

## Deterministic mathematical invariants
- Causality: future K/V changes cannot alter earlier query rows.
- Query locality: changing query row i cannot alter rows before i.
- Batch/head isolation: batched execution agrees with independently sliced execution.
- Single-token identity: output equals V for N=1.
- Constant-V invariance: output equals the constant V tensor.
- Zero-logit prefix mean: equal logits produce the arithmetic mean of the causal prefix.
- Identical-key reduction: repeated keys produce prefix means.
- Common-key-shift invariance: adding a common key vector preserves softmax probabilities within FP16 rounding.
- Affine value transform: output follows the same affine transformation within FP16 rounding.
- Convex-hull property: each output coordinate lies within the causal prefix value range.
- Extreme finite logits: output remains finite and stable.

## GPU non-cuTile coverage
- Ordinary CUDA baselines are tested independently against the mathematical oracle.
- Block-row, warp-row, and shared-tile implementations are cross-validated.
- Shared-tile coverage spans D={32,64,128}, N={64,128,256}, tile_k={32,64}, plus batch/head isolation, tile-size invariance, zero logits, constant V, extreme logits, causal future-K/V invariance, nonfinite rejection, and no-partial-query-tile validation.
- Both exposed A100 devices are exercised for native CUDA execution and cross-device consistency.
- Historical v1 and optimized v2 shared-tile implementations are regression-tested against each other and the oracle.

## Contract/error coverage
- dtype, rank, contiguity, finite-input, batch/head, sequence, D, Dv, and tile-divisibility validation.
- Required D values {32,64,128}.
- Invalid and non-divisible tile configurations.

## What 'exhaustive' means here
The CPU mathematical/reference layer is exhaustively enumerated over a deliberately bounded small state space, including every legal tiling for N=1..16. GPU coverage is broad deterministic coverage over the required production dimensions plus adversarial cases; it is not every possible B,H,N,D combination. Full numerical claims must be based on the frozen GPU matrix and recorded raw results.

## Future enhancement test buckets
- Tensor-core MMA variants.
- Alternative tile shapes and autotuning.
- Longer sequences and larger batch/head combinations.
- BF16/FP8 only as optional extensions after the mandatory FP16 path is stable.
- GQA/MQA, variable-length inputs, backward, and dropout remain outside the mandatory project scope.