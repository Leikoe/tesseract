# Engine request-authority validation

Validated on 2026-08-08 at Git revision
`9f5c8caa527a647a1ced44dab06eeb5f23ca4327`.

## Change under test

The engine request record is now the sole semantic authority for prompt and
generated token history, incremental decoding, sampling state, request-local
randomness, stop handling, progress, and logical KV-slot history. The Llama
CUDA backend no longer has an `add_request`/`remove_request` lifecycle or a
per-request semantic-state map. It consumes a validated `ForwardBatch` carrying
only the exact batch-local token, position, slot, context, and sampling inputs
needed for execution.

The boundary is enforced with semantic `TokenId` and `SamplingInput` types and
fallible batch construction. Duplicate, missing, or unexpected backend outputs
fail requests and reclaim their KV state rather than partially committing
scheduler progress.

## Verification

- Local `cargo test`: 34 passed.
- Local `cargo clippy --all-targets -- -D warnings`: passed.
- Local formatting and whitespace checks: passed.
- A100 `scripts/node/verify-a100.sh`: passed on an NVIDIA A100-SXM4-80GB with
  CUDA 13.3 and Rust 1.89.0.
- The real BF16 Llama forward check predicted token 12366 (`" Paris"`) for
  `The capital of France is`.
- Two independent API requests with `temperature=0.8`, `top_p=0.9`, and
  `seed=4242` produced the same 12-token completion, directly exercising the
  engine-owned stochastic sampling state on CUDA.
- The cuTile smoke kernel and model/CUDA allocation checks passed.
- Scheduler property tests cover arbitrary scheduling/KV state machines and
  exact batch metadata; focused tests cover duplicate backend results and
  reclamation.

## Performance

The retained comparison uses concurrency 8, 16 requests, at most 16 generated
tokens, and two benchmark warmup requests on the same A100 host.

| Revision/workload | Output tok/s | Mean TTFT | Mean inter-token |
| --- | ---: | ---: | ---: |
| Prior redesign, first traffic | 1,554.25 | 12.59 ms | 3.60 ms |
| Engine authority, first traffic | 1,654.93 | 12.37 ms | 3.35 ms |
| Prior redesign, repeated | 1,538.57 | 12.74 ms | 3.67 ms |
| Engine authority, repeated | 1,623.51 | 13.98 ms | 3.34 ms |

Throughput was 6.5% higher on first traffic and 5.5% higher on the repeated
pass. Mean inter-token latency was lower in both passes. Repeated mean TTFT was
1.24 ms higher, so the result should be read as no throughput regression, not
as a universal latency improvement from an ownership-only refactor.

Raw compact summaries and the exact environment are retained in
[`../benchmarks/2026-08-08-engine-request-authority/`](../benchmarks/2026-08-08-engine-request-authority/README.md).
