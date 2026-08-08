# Tesseract v1 acceptance contract

Last updated: 2026-08-07

This document defines what “v1” means. Passing a hello-world request is not
enough: v1 is complete only when every requirement below has direct evidence.

## Supported deployment

- One NVIDIA A100 80 GB (`sm_80`) on Linux.
- CUDA 13.2 or newer and Rust 1.89 or newer.
- `meta-llama/Llama-3.2-1B-Instruct` loaded from SafeTensors.
- BF16 weights, activations, and KV cache. Numerically sensitive reductions,
  normalization statistics, softmax state, and sampling math may use FP32.
- One model replica per process. Multi-GPU, quantization, multimodal input,
  LoRA, speculative decoding, and constrained decoding are explicitly outside
  v1, but the request and backend boundaries must not preclude them.

## Serving contract

- `POST /v1/chat/completions` supports OpenAI-compatible chat requests.
- Streaming uses SSE and terminates with `[DONE]`; non-streaming returns one
  complete response.
- v1 supports greedy decoding and temperature/top-p sampling, `max_tokens`,
  stop strings, and a request-scoped seed.
- Client disconnect and explicit cancellation release scheduler and KV state.
- Admission is bounded. Overload returns a deliberate error instead of
  creating unbounded queues or memory growth.
- `GET /health/live`, `GET /health/ready`, and `GET /metrics` are available.
- Shutdown stops admission, drains or cancels active requests, and releases GPU
  resources without hanging.

## Engine contract

- HTTP work runs on Tokio; model execution never runs on a Tokio worker.
- A dedicated engine thread owns scheduling, CUDA context/streams, model
  weights, graphs, and KV memory.
- Continuous batching admits new work between decoding iterations.
- Prefill work is token-budgeted and can be chunked so a long prompt cannot
  indefinitely block active decodes.
- Request logical positions map through an explicit table to physical KV
  slots. Physical K/V storage is flat and preallocated per layer.
- KV allocation, release, and request teardown are deterministic and tested.
- The decode path supports CUDA graph capture/replay for stable batch buckets,
  with a correct eager fallback.
- Linear projections use cuBLASLt/cuBLAS initially; fused elementwise,
  positional, KV-write, attention, and sampling kernels use cuTile Rust where
  supported on `sm_80`.

## Correctness gates

- Local Rust unit and integration tests pass with no GPU required.
- API schema, SSE framing, overload, cancellation, and shutdown have tests.
- Scheduler property tests cover token budgets, fairness, cancellation, and KV
  slot non-aliasing/reuse.
- SafeTensors names, shapes, dtype, and model configuration are validated before
  GPU allocation.
- On the A100, greedy next-token IDs and logits agree with a pinned reference
  implementation for fixed prompts within documented BF16 tolerances.
- Generated text is checked for at least: empty input rejection, one-token
  generation, EOS, maximum length, stop strings, and concurrent requests.
- Compute Sanitizer or equivalent memory checking passes representative eager
  kernel tests where the CUDA/cuTile toolchain supports it.

## Production and performance gates

- Release build starts from a clean checkout using documented commands.
- Readiness remains false until model load, kernel compilation/warmup, and graph
  capture required by configured buckets complete.
- Logs contain request IDs and lifecycle timings without prompts, tokens, or
  credentials by default.
- Metrics include queue depth, running requests, prompt/generated token counts,
  TTFT, inter-token latency, request latency, failures, cancellations, KV usage,
  batch size, and graph/eager execution counts.
- A reproducible benchmark records hardware, driver, CUDA, Git revision, model
  revision, server arguments, workload, TTFT, TPOT, throughput, and latency
  percentiles.
- Validation includes batch-1 decode and a concurrent mixed-prompt workload.
- The final report distinguishes measured results from targets and retains raw
  benchmark output in the local Git repository.

## Canonical workflow for the spot A100

1. Make changes locally.
2. Run local formatting, linting, and CPU-independent tests.
3. Commit and push the exact revision.
4. Fast-forward the A100 checkout from GitHub.
5. Build and validate that revision on the A100.
6. Copy any unique logs, profiles, or benchmark results back locally.
7. Commit important evidence before relying on the spot instance again.

