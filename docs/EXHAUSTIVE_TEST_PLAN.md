# Test plan

## Reference checks

- Compare the vectorised FP32 attention reference against a scalar FP64 reference for small tensors (N=1–12).
- Compare the tiled online-softmax reference against the scalar reference for N=1–16 and every legal tile pair.
- Keep a regression for a fully future K tile; it previously exposed an invalid online-softmax state.

## Mathematical properties

- Changing future K/V values cannot change earlier outputs.
- Changing query row i cannot change rows before i.
- Batched output agrees with separate batch/head slices.
- For N=1, output equals V.
- Constant V stays constant.
- Zero scores give the average of each causal prefix.
- Repeated keys give causal-prefix means.
- Adding a common key vector preserves probabilities within FP16 rounding.
- Affine transformations of V are preserved within FP16 rounding.
- Output coordinates stay within the allowed value range.
- Large finite inputs produce finite output.

## GPU coverage

- Check each ordinary CUDA implementation against both references.
- Cross-check block-row, warp-row and shared-tile variants.
- Cover D={32,64,128}, N={64,128,256} and tile_k={32,64} where legal.
- Check batch/head isolation, tile agreement, causal invariance, constant values, zero logits, extreme values, invalid inputs and non-divisible shapes.
- Keep cross-device checks separate from cuTile validation.

## Input-contract checks

Reject invalid dtype, rank, contiguity, batch/head dimensions, sequence lengths, D, Dv, non-finite values and illegal tile divisibility before launching a kernel.

## What “exhaustive” means here

The CPU reference tests enumerate a bounded small state space. GPU tests cover the required dimensions and selected edge cases; they do not cover every possible shape.

Autotuning, BF16/FP8, backward, GQA/MQA, packed sequences and dropout are separate extensions, not part of the required FP16 forward path.
