# Kernel design

## Goal

Implement the forward pass of causal scaled dot-product attention in one NVIDIA cuTile kernel.

## Contract

Inputs Q, K and V are contiguous FP16 arrays with logical shape [B,H,N,D]. The required path uses self-attention, so all sequence lengths match and Dv = D. Supported head dimensions are 32, 64 and 128. The selected query and key tile sizes must divide the sequence length.

The output has shape [B,H,N,D] and uses FP16. Invalid shapes, layouts, dtypes and non-finite values are rejected before launch.

## Kernel steps

1. A program handles one batch/head/query-tile.
2. Load the query tile and initialise the online-softmax state.
3. Load key/value tiles in sequence.
4. Compute scaled QK scores and mask all future key positions.
5. Update the row-wise maximum, normalisation sum and weighted-value accumulator.
6. Divide by the final normalisation sum and store the output tile.

The full N-by-N score and probability matrices are never written to global memory by the fused path.

For each query row, let m be the running maximum, l the running softmax sum and a the weighted-value accumulator. For a new tile with maximum mt:

mnew = max(m, mt)

lnew = exp(m - mnew) * l + sum(exp(s - mnew))

anew = exp(m - mnew) * a + sum(exp(s - mnew) * V)

The output row is a/l.

## Precision

Q/K/V and the output use FP16. Score accumulation, running softmax values and P×V accumulation use FP32. The probability operand is converted to FP16 for P×V, while its accumulator stays FP32.

## Correctness and performance

The expected output comes from the independent FP32 NumPy reference. The required suite has 48 GPU test executions; the separate numerical matrix has 29 legal case/tile pairs.

Timing uses CUDA events, warmups, repeated samples and rotated tile order. The speedup baseline is deliberately unfused FP32 CuPy. It is not an optimized attention library.

The implementation target is the native Windows RTX 4060. Ordinary CUDA versions on the A100 are separate comparison kernels.
