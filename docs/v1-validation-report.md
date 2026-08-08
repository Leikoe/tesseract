# Tesseract v1 validation report

Validated 2026-08-08 UTC (2026-08-07 America/Los_Angeles).

This report audits every item in [`v1-acceptance.md`](v1-acceptance.md). It
distinguishes retained measurements from design claims and identifies the exact
revision used for the final clean-checkout gate.

## Verdict

Tesseract v1 satisfies the acceptance contract for one A100 80 GB serving
`meta-llama/Llama-3.2-1B-Instruct` in BF16.

- Final implementation revision: `8df163f9944db43916d54131c3527c8fc23d52ed`
- Final A100 verifier: `scripts/node/verify-a100.sh`
- Final verifier result: `a100_node_validation=ok`
- Local and A100 unit/integration result: 21 passed, 0 failed
- A100 checkout after verification and HTTP probes: clean and aligned with
  `origin/main`

The retained benchmark was run on implementation revision `9ebb36b`. Changes
between that revision and the final implementation revision only retain the raw
benchmark evidence, add acceptance tests and request lifecycle logging, improve
documentation, and extend the verifier with strict CUDA Clippy. They do not
change model execution or scheduling.

## Supported deployment

| Requirement | Direct evidence | Result |
| --- | --- | --- |
| One NVIDIA A100 80 GB (`sm_80`) on Linux | Final verifier reported NVIDIA A100-SXM4-80GB, compute capability 8.0, 81,920 MiB. The node image is Ubuntu 22.04. | Pass |
| CUDA 13.2+ and Rust 1.89+ | Final verifier reported CUDA 13.3 build `compiler.38244171_0` and enforced Rust 1.89.0. | Pass |
| Llama 3.2 1B Instruct from SafeTensors | `model-check` loaded 146 tensors for `meta-llama/Llama-3.2-1B-Instruct`; `src/model/weights.rs` uses the official `safetensors` crate. | Pass |
| BF16 weights, activations, and KV | Model validation reported `dtype=bfloat16`; `model-cuda-check` loaded 2,471,628,800 bytes; the model config rejects non-BF16 weights and CUDA KV tensors are BF16. FP32 is confined to allowed reductions/sampling math. | Pass |
| One replica and clean future boundaries | One dedicated backend is constructed per process. Generic `Model` and `Backend` traits isolate the engine while all Llama configuration, template, tokenizer, tensor names, and execution remain in `src/model/llama_3_2.rs`. | Pass |

Multi-GPU, quantization, multimodal input, LoRA, speculative decoding, and
constrained decoding remain intentionally outside v1.

## Serving contract

| Requirement | Direct evidence | Result |
| --- | --- | --- |
| OpenAI-compatible chat completions | API integration tests validate the non-streaming response envelope, model, choices, finish reason, and usage. A real-model A100 request returned `Paris`. | Pass |
| SSE streaming and `[DONE]` | Integration tests validate role/content/usage chunks and `[DONE]`. The retained A100 smoke emitted a terminal finish chunk, usage chunk, and `[DONE]`. | Pass |
| Greedy, temperature/top-p, max tokens, stop, seed | The A100 smoke validated greedy and concurrent seeded temperature/top-p requests. Length termination has a scheduler test. Stop suppression has a scheduler test and a final real-model HTTP probe. | Pass |
| Disconnect and explicit cancellation release state | Tests cover stream-drop and explicit cancellation. The real disconnect probe returned running requests and KV usage to zero and incremented cancellation metrics. | Pass |
| Bounded admission | Atomic frontend admission plus bounded Tokio channels reject excess work. The HTTP test asserts 429 with `rate_limit_error`. | Pass |
| Live, ready, and metrics endpoints | API tests exercise liveness and metrics. The A100 server returned ready only after backend loading and graph warmup. | Pass |
| Graceful shutdown | The shutdown test cancels active work and clears readiness. Real SIGINT probes exited without a remaining server process or allocated request KV. | Pass |

The final A100 HTTP probe additionally established:

- an empty `messages` array returns HTTP 400 and OpenAI-shaped
  `invalid_request_error`;
- stop string `Paris` terminates with `finish_reason=stop` and is absent from
  the response content;
- the completion lifecycle log contains request ID, finish reason, prompt and
  generated token counts, and end-to-end latency (`1639.246 ms` in that probe),
  without prompt text, generated text, or credentials.

## Engine contract

| Requirement | Direct evidence | Result |
| --- | --- | --- |
| Tokio HTTP; no model execution on workers | `src/main.rs` runs Axum on Tokio while `EngineHandle::spawn_with_factory` creates the named `tesseract-engine` OS thread and initializes the backend there. | Pass |
| Dedicated owner of CUDA/model/scheduler state | `EngineWorker` owns the backend, scheduler state, KV allocator, and request queues; the CUDA backend is created and used on that thread. | Pass |
| Continuous batching | Commands are admitted between engine iterations and newly admitted requests join existing running work. Concurrent A100 workloads reached batch sizes 2 and 8. | Pass |
| Token-budgeted, chunked prefill | `build_batch` caps total scheduled tokens and each prefill item. Exhaustive tests vary budgets 3–9 and chunk sizes 1–4; decode-priority behavior is separately tested. | Pass |
| Flat preallocated KV and explicit maps | Each layer owns flat BF16 K/V tensors. `KvSlots` reserves/allocates physical slots and scheduled work carries the logical request's explicit slot vector into KV write/gather kernels. Non-aliasing, reservation, and reuse are tested. | Pass |
| Deterministic teardown | Completion, failure, cancellation, disconnect, and shutdown all release `KvSlots` and backend request state. Tests and A100 metrics end at zero KV use. | Pass |
| CUDA graphs plus eager fallback | Full 16-layer decode graphs include embedding through LM head and greedy argmax. Batch-1 context buckets prewarm; exact larger batch/context shapes capture lazily. Shape failures are remembered and route to eager execution. A100 counters prove captures and replays. | Pass |
| cuBLAS and cuTile division | cuBLAS performs linear projections. cuTile Rust implements fused elementwise operations, RoPE, flat-KV access, attention, and greedy sampling on `sm_80`. Temperature/top-p sampling uses a correct FP32 host implementation in v1. | Pass |

cuTile's persistent CUBIN cache is explicitly enabled with a private 2 GiB soft
capacity. A cold process compiled 11 kernels; the next process loaded all 11
from disk in under 1 ms each. The final verifier required cached `.cubin`
artifacts to exist.

## Correctness gates

| Requirement | Direct evidence | Result |
| --- | --- | --- |
| Local tests | `cargo test --all-targets`: 21 passed, 0 failed. Strict local Clippy also passed. | Pass |
| API/SSE/overload/cancel/shutdown tests | Named integration tests directly cover every listed behavior. | Pass |
| Scheduler properties | Exhaustive small-domain tests cover total token budget, chunk cap, decode priority, physical-slot non-aliasing, exact slot count, reservation, and reuse. Cancellation paths are independently exercised. | Pass |
| SafeTensors/config validation before GPU allocation | `Llama32::load` parses and validates architecture, BF16 dtype, dimensions, tokenizer, tensor names, shapes, and dtypes before `CudaLlama::load` allocates device state. `model-check` is the host-only gate. | Pass |
| Pinned independent logits reference | PyTorch 2.8.0 / Transformers 4.55.0 BF16 eager reference matched all three next-token IDs. Shared top-logit maximum absolute differences were 0.125, 0.4375, and 0.125 under the documented 0.5 tolerance. | Pass |
| Generated-text edge cases | Empty input: API plus Llama-template tests and final A100 HTTP 400. One token: real `Paris` response. EOS: real count-to-three stream. Maximum length and stop: scheduler tests plus real stop probe. Concurrent requests: real France/Germany packed decode and concurrency-8 benchmark. | Pass |
| CUDA memory checking | Compute Sanitizer 2026.2.1 reported zero errors for project kernels, eager cuBLAS, captured replay, a full 16-layer next-token forward, and an HTTP workload with packed batch-2 graph replay. | Pass |

The detailed correctness and sanitizer transcript is retained in
[`validation/2026-08-07-a100-server-smoke.md`](validation/2026-08-07-a100-server-smoke.md).

## Production and performance gates

| Requirement | Direct evidence | Result |
| --- | --- | --- |
| Release from clean checkout | The A100 fast-forwarded to exact revision `8df163f`, passed the verifier's clean-tracked-worktree guard, built release CUDA binaries, and remained clean after probes. | Pass |
| Readiness after load/warmup | Backend initialization loads model state and captures all configured batch-1 graph buckets before returning. The worker sets readiness afterward. A100 logs show graph warmup before the ready log; default capacity warmed 12 buckets. | Pass |
| Safe lifecycle logs | Admission and finish logs carry request IDs; finish/failure logs include lifecycle latency. Default logs contain no prompt, generated text, token IDs, or credentials. | Pass |
| Required metrics | Prometheus output includes queue/running/KV gauges; request outcome, prompt/generated token, engine/batch, graph/eager, and packed-decode counters; and TTFT, inter-token, and request-duration summaries. | Pass |
| Reproducible benchmark metadata | The pure-standard-library harness records GPU, driver, CUDA, Rust, Git, exact model revision, config checksum, server args, prompts, raw request timing/usage, and percentile summaries. | Pass |
| Batch-1 and mixed concurrent workloads | Retained raw results include eight serial requests and two concurrency-8 mixed-prompt passes. | Pass |
| Measured vs targets and raw retention | The benchmark report labels all values as measurements, explains first-shape capture versus warm steady state, makes no target claim, and retains every raw JSON report plus final metrics in Git. | Pass |

Measured A100 results from the retained benchmark:

| Workload | Output tok/s | Mean TTFT | TTFT p99 | Mean inter-token | Request p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Batch 1 | 27.97 | 437.28 ms | 1,793.16 ms | 2.95 ms | 1,838.31 ms |
| Concurrency 8, first shape pass | 47.37 | 320.70 ms | 845.97 ms | 114.50 ms | 2,577.64 ms |
| Concurrency 8, warm shapes | 584.46 | 76.27 ms | 117.67 ms | 7.48 ms | 210.66 ms |

The raw files and fuller interpretation are in
[`benchmarks/2026-08-08-a100/`](benchmarks/2026-08-08-a100/README.md).

## Reproduction

New-node bootstrap, pinned versions, model revision, persistent cache location,
reference setup, sanitizer commands, and normal spot-instance workflow are in
[`a100-node-setup.md`](a100-node-setup.md). The canonical sequence remains:

1. validate locally;
2. commit and push;
3. fast-forward the node;
4. execute `scripts/node/verify-a100.sh`;
5. copy unique evidence back;
6. commit retained evidence locally.
