# A100 batching redesign validation

Validated 2026-08-08 on one NVIDIA A100-SXM4-80GB with CUDA 13.3 and Rust
1.89. The machine ran a Git-less source snapshot labeled
`uncommitted-source-snapshot-20260808`; no Git history or credentials were
copied to the node.

## Diagnosis

The historical server left most of the A100 unused for four independent
reasons:

1. It invoked the model once per request instead of flattening scheduled work
   into one ragged forward.
2. Every layer materialized gathered KV tensors rather than reading the flat
   physical cache directly.
3. Hot-path synchronization and initialized output allocation serialized work
   and launched unnecessary fills.
4. Runtime tensor/scalar combinations triggered cuTile frontend compilation
   after readiness. Nsight and backend timing separated the cold path: prompt
   tokenization took 0.14 ms, while the first prefill took 671 ms. After fixing
   metadata extents, the remaining 347 ms was isolated to `rope_kv_write` and
   its host `valid_rows` scalar.

The final implementation uses one validated `ScheduledBatch`, direct flat-KV
ragged attention, power-of-two query/decode buckets, fixed eager metadata
extents, isolated KV scratch slots for padding, and 72 pre-captured and
pre-replayed decode graphs. No logical row count remains in a cuTile kernel
signature.

## Performance

The retained workload is concurrency 8, 16 requests, at most 16 generated
tokens, with two benchmark warmup requests. “First traffic” is the first such
workload after server readiness; “warm” is the immediately repeated workload.

| Build/workload | Output tok/s | Mean TTFT | Mean inter-token |
| --- | ---: | ---: | ---: |
| Historical retained baseline, warm | 610.04 | 74.24 ms | 7.36 ms |
| Redesign, first traffic | 1,554.25 | 12.59 ms | 3.60 ms |
| Redesign, repeated | 1,538.57 | 12.74 ms | 3.67 ms |

First traffic is 2.55 times the historical warm throughput, with 83% lower
mean TTFT and 51% lower mean inter-token latency. More importantly, first and
repeated throughput differ by only 1.0%; the request-time compilation cliff is
gone.

Raw results:

- [`concurrent-first.json`](../benchmarks/2026-08-08-a100-shape-stable/concurrent-first.json)
- [`concurrent-warm.json`](../benchmarks/2026-08-08-a100-shape-stable/concurrent-warm.json)

After both passes the server reported 36 completed requests, zero failed or
cancelled requests, maximum batch size 8, 72 graph captures, 67 graph replays,
67 packed decode forwards covering 348 request rows, and zero allocated logical
KV tokens. The capture count did not increase after readiness.

## Correctness and safety

- `cargo test --all-targets`: 31 passed, including Proptest-generated scheduler
  state machines and bucket-coverage properties.
- Local and CUDA-feature strict Clippy passed with warnings denied.
- A pinned PyTorch 2.8 / Transformers 4.55 BF16 reference matched all three
  next-token IDs. Shared top-logit maximum absolute differences were 0.125,
  0.4375, and 0.125 under the 0.5 tolerance.
- Compute Sanitizer initially exposed the validation binary under-declaring its
  query bound after isolated padding slots were introduced. The runtime now
  rejects query buckets above the configured scratch bound, the validator
  derives that bound from its encoded prompt, and the final 16-layer eager
  prefill/decode run reported `ERROR SUMMARY: 0 errors` while predicting token
  12366 (`" Paris"`).

Stable Rust constructors and typed errors enforce runtime contracts. Nightly
`core::contracts` and panic-oriented procedural contract macros are deliberately
not dependencies. Property tests complement those boundary checks by exploring
multi-step scheduling, KV allocation/release, progress, fairness, and all
configured execution buckets.
