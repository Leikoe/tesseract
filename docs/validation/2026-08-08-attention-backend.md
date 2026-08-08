# Attention backend validation

Validated on 2026-08-08 at Git revision
`b4ecef2d7637669a840a32d7e176c217413c76ff`.

## Change under test

`AttentionBackend` is a crate-private, statically composed operation-family
contract. Its associated `LayerState` makes physical state ownership explicit.
The two required execution paths are cohesive: `enqueue_eager` consumes typed
ragged metadata, while `record_decode` records the same implementation into a
CUDA graph using stable buffers.

`CudaLlama<A>` is generic over the backend. The initial
`DirectFlatKvAttention` owns every layer's key/value tensors and the RoPE tables,
and implements query rotation, fused key rotation/KV write, and ragged flat-KV
attention for both eager and captured execution. Q/K/V and output projections
remain model-program responsibilities. There is no per-layer dynamic dispatch
or runtime plugin registry.

## Verification

- Local `cargo test`: 39 passed.
- Local `cargo clippy --all-targets -- -D warnings`: passed.
- CUDA-feature compilation and strict Clippy passed on the A100.
- A100 `scripts/node/verify-a100.sh`: passed on an NVIDIA A100-SXM4-80GB with
  CUDA 13.3 and Rust 1.89.0.
- The real BF16 forward check still predicted token 12366 (`" Paris"`), proving
  that eager prefill plus the attention state path remained numerically intact.
- The serving benchmark exercised pre-captured batched decode graphs, proving
  the graph-recording implementation remained valid after extraction.

## Performance

The retained workload uses concurrency 8, 16 requests, at most 16 generated
tokens, and two benchmark warmup requests on the same A100 host.

| Revision/run | First traffic tok/s | Repeated tok/s | Repeated inter-token |
| --- | ---: | ---: | ---: |
| Generic CUDA executor | 1,672.04 | 1,663.19 | 3.35 ms |
| Attention backend, first suite | 1,640.20 | 1,585.73 | 3.47 ms |
| Attention backend, repeated suite | 1,691.32 | 1,666.46 | 3.34 ms |

The first suite suggested a 4.7% warm regression, but the immediate full repeat
matched the prior revision (+0.2% warm) and had slightly lower inter-token
latency. Since the captured operations are unchanged and the two runs straddle
the baseline, the evidence supports a performance-neutral static abstraction;
both runs are reported to make that conclusion auditable.

Compact summaries and the exact environment for the retained repeat are in
[`../benchmarks/2026-08-08-attention-backend/`](../benchmarks/2026-08-08-attention-backend/README.md).
