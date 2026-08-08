# CUDA batch and sampler extraction validation

Validated on 2026-08-08 at Git revision
`efb076670fbbaffebb7fe969855c3b1d9c46dc4c`.

## Change under test

Validated `ForwardBatch` lowering is now implemented once in the model-neutral
`cuda::batch::CudaBatch`. It creates aligned token, position, current-slot,
request-index, context-length, context-slot, and sample-row metadata while
preserving the stable prefill/decode row partition. Llama consumes this physical
materialization instead of rebuilding scheduler metadata itself.

Host-resident logits are now sampled by an explicit `HostLogitsSampler` beside
the executor. Greedy, temperature, top-p, and deterministic draw behavior have
typed inputs and errors. Its documentation forbids silently copying device
logits to the host as a fallback; the current stochastic eager path already
returns host logits and opts into this implementation explicitly.

## Verification

- Local `cargo test`: 39 passed.
- Local `cargo clippy --all-targets -- -D warnings`: passed.
- The host sampler property test generated arbitrary finite logit vectors,
  temperatures, top-p values, and draws, always selecting an in-range token.
- Greedy tie ordering and invalid-logit handling have focused tests.
- The CUDA-feature-only mixed-batch lowering test passed on the A100 and checks
  every flattened metadata vector and sampled row.
- A100 `scripts/node/verify-a100.sh`: passed on an NVIDIA A100-SXM4-80GB with
  CUDA 13.3 and Rust 1.89.0.
- The real BF16 forward check still predicted token 12366 (`" Paris"`).
- Two independent API requests with `temperature=0.8`, `top_p=0.9`, and
  `seed=4242` produced the same 12-token completion through the extracted
  sampler.

## Performance

The retained workload uses concurrency 8, 16 requests, at most 16 generated
tokens, and two benchmark warmup requests on the same A100 host.

| Revision/workload | Output tok/s | Mean TTFT | Mean inter-token |
| --- | ---: | ---: | ---: |
| Execution tickets, first traffic | 1,622.77 | 12.44 ms | 3.43 ms |
| CUDA batch/sampler, first traffic | 1,648.89 | 12.34 ms | 3.35 ms |
| Execution tickets, repeated | 1,687.61 | 10.66 ms | 3.36 ms |
| CUDA batch/sampler, repeated | 1,685.06 | 11.99 ms | 3.29 ms |

First-pass throughput was 1.6% higher and repeated throughput was 0.2% lower.
Mean inter-token latency was lower in both passes. Repeated mean TTFT was 1.33
ms higher, so this supports a performance-neutral extraction rather than a
speed claim.

Compact summaries and the exact environment are retained in
[`../benchmarks/2026-08-08-cuda-batch-sampler/`](../benchmarks/2026-08-08-cuda-batch-sampler/README.md).
