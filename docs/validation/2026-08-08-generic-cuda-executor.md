# Generic CUDA executor validation

Validated on 2026-08-08 at Git revision
`99ac26e1f88adccb01e9c9d2f2d9c440f317d8ef`.

## Change under test

`CudaExecutor<P>` is now the reusable CUDA execution shell. It lowers a
validated `ForwardBatch`, submits one statically composed `ModelProgram`,
validates program-output cardinality, applies the selected host-logits sampler,
maps tokens back to request IDs, and publishes the typed completion ticket.

Llama implements `ModelProgram` and sees only a whole `CudaBatch`. It returns
either tokens or explicitly host-resident logits; it no longer implements the
engine-facing executor protocol or handles sampling and completion. This is
source-level extensibility with monomorphized composition, not runtime plugin
dispatch.

## Verification

- Local `cargo test`: 39 passed.
- Local `cargo clippy --all-targets -- -D warnings`: passed.
- Three CUDA-feature contract tests passed on the A100: mixed-batch metadata
  lowering, output-cardinality rejection, and one-logit-row-per-sample mapping.
- A100 `scripts/node/verify-a100.sh`: passed on an NVIDIA A100-SXM4-80GB with
  CUDA 13.3 and Rust 1.89.0.
- The real BF16 forward check still predicted token 12366 (`" Paris"`).
- Two independent API requests with `temperature=0.8`, `top_p=0.9`, and
  `seed=4242` produced the same 12-token completion after output normalization
  moved into `CudaExecutor`.

## Performance

The retained workload uses concurrency 8, 16 requests, at most 16 generated
tokens, and two benchmark warmup requests on the same A100 host.

| Revision/workload | Output tok/s | Mean TTFT | Mean inter-token |
| --- | ---: | ---: | ---: |
| CUDA batch/sampler, first traffic | 1,648.89 | 12.34 ms | 3.35 ms |
| Generic CUDA executor, first traffic | 1,672.04 | 11.55 ms | 3.40 ms |
| CUDA batch/sampler, repeated | 1,685.06 | 11.99 ms | 3.29 ms |
| Generic CUDA executor, repeated | 1,663.19 | 11.95 ms | 3.35 ms |

First-pass throughput was 1.4% higher and repeated throughput was 1.3% lower.
The opposing movements and sub-0.06 ms inter-token differences support a
performance-neutral ownership extraction, not a speed claim.

Compact summaries and the exact environment are retained in
[`../benchmarks/2026-08-08-generic-cuda-executor/`](../benchmarks/2026-08-08-generic-cuda-executor/README.md).
