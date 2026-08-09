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
`C`, let `ell = cumsum(log(alpha))`. The numerically stable system is

```text
N = I + tril(B (exp(ell_i - ell_j) .* K K^T), -1).
```

For causal entries `i > j`, `ell_i - ell_j <= 0`, so the exponential cannot
grow. The kernel never materializes `gamma = exp(ell)` or `gamma^-1`. A single
FP32 forward-substitution factorization of `N` is reused for

```text
U = N^-1 (B V)
W = N^-1 (B exp(ell) K).
```

This is algebraically identical to factoring the ungated
`M = I + tril(B K K^T, -1)`, because with `G = diag(exp(ell))`,
`N = G M G^-1` and `N^-1 = G M^-1 G^-1`. Factoring `N` is preferable on the
GPU because it expresses every decay as a bounded log-space difference. The
remaining state transition and causal chunk output are tensor-core GEMMs. This
is the single-inversion identity described in [Simple math to speed up GDN
prefill](https://veitner.bearblog.dev/simple-math-to-speed-up-gdn-prefill/),
the orientation used by SGLang's fused KKT solve, and is consistent with the
single-inverse observation in [Comba](https://arxiv.org/abs/2506.02475).

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
