# Qwen Gated DeltaNet execution design

This is the CUDA execution contract for the 30 linear-attention layers in
`nvidia/Qwen3.6-35B-A3B-NVFP4`. Decode and prefill deliberately use different
algorithms while sharing the same recurrent and convolution state layout.

## State and projections

Each request slot owns, per GDN layer:

- a BF16 causal-convolution tail `[8192, 3]`; and
- an FP32 recurrent matrix `[32, 128, 128]`.

The input projections produce Q and K with 16 heads of width 128, V and Z with
32 heads of width 128, and headwise `a`/`b` gates with 32 values. Value head
`h_v` uses key/query head `h_v / 2`.

The convolution is depthwise, width four, causal, and followed by SiLU. For
each token the recurrent transition is evaluated in FP32:

```text
g     = -exp(A_log) * softplus(a + dt_bias)
alpha = exp(g)
beta  = sigmoid(b)
q, k  = l2_normalize(q), l2_normalize(k)
S     = alpha*S + beta*(v - S*k)*k^T
o     = S*(q/sqrt(128))
```

The output applies per-value-head RMSNorm with the checkpoint's 128-element
weight, then multiplies by `silu(z)` before the output projection.

## Decode

Decode is a fused recurrent update. A persistent launch maps each
`(request, value_head, value_block)` to one worker, loads its FP32 state tile,
performs the one-token update, stores the state, and emits BF16 output. The
convolution tail is updated in a preceding fused depthwise-convolution launch.
No request-id map or state allocation exists inside the executor: the batch
contains scheduler-owned recurrent slot indices.

## Prefill

Prefill uses chunks, not a token-serial recurrent kernel. For a chunk of length
`C`, define the cumulative decay `gamma` and the strict-lower triangular
matrix

```text
M = I + tril(B K K^T, -1).
```

The second system needed by the gated update is diagonally similar:

```text
G = diag(gamma)
N = G M G^-1
N^-1 = G M^-1 G^-1.
```

Therefore the FP32 forward-substitution factorization of `M` is computed once
per chunk/head and reused for both `W` and the gate-rescaled `U`. The remaining
state transition and causal chunk output are tensor-core GEMMs. This is the
single-inversion formulation described in [Simple math to speed up GDN
prefill](https://veitner.bearblog.dev/simple-math-to-speed-up-gdn-prefill/)
and is consistent with the single-inverse observation in
[Comba](https://arxiv.org/abs/2506.02475).

Chunk boundaries are internal kernel choices. Request boundaries, initial
state selection, and final-state commit come from `query_start_offsets` and
the scheduler's recurrent slots; no chunk may cross a request boundary.

## Verification gates

1. The fused decode transition is compared with an FP32 host recurrence over
   randomized states, gates, Q/K/V, and multiple request slots.
2. Chunked prefill is compared against that recurrence for boundary lengths
   around the selected chunk size, including nonzero initial state.
3. Running prefill followed by decode must equal one uninterrupted recurrence.
4. State-slot permutation must only permute outputs; cancellation reuse must
   begin from zeroed state.
5. Compute Sanitizer and retained A100 timing cover both one-token decode and
   long packed prefill.
