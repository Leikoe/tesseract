# tesseract

Tesseract is a production-oriented, BF16 inference server for NVIDIA A100
GPUs. v1 serves `meta-llama/Llama-3.2-1B-Instruct` through an OpenAI-compatible
chat-completions API while keeping model-specific behavior behind the model
backend boundary.

## Architecture

Tokio, Axum, and Tower own the HTTP control plane. A dedicated engine thread
owns scheduling and every CUDA resource, so model execution never blocks a
Tokio worker. The scheduler provides bounded admission, chunked prefill,
continuous batching, decode priority, deterministic cancellation, and a flat
preallocated BF16 KV cache with explicit logical-to-physical slot maps.

Linear algebra uses cuBLAS. cuTile Rust kernels implement embedding,
normalization, RoPE, direct flat-KV attention, activation, residuals, and greedy
argmax. The backend flattens prompt chunks and decode rows into one ragged model
batch. Pure greedy decode uses full-model CUDA graph replay; prefill, mixed, and
stochastic batches share a batched eager path. cuTile's persistent CUBIN cache
is enabled at startup.

## Run on the validated A100 stack

The canonical, idempotent setup is documented in
[`docs/a100-node-setup.md`](docs/a100-node-setup.md). After downloading the
pinned model revision, start the release server with:

```bash
cargo +1.89.0 run --release --features cuda --bin tesseract -- \
  --model-path /home/ubuntu/models/Llama-3.2-1B-Instruct
```

The default listener is `0.0.0.0:8000`. Liveness, readiness, and Prometheus
metrics are available at `/health/live`, `/health/ready`, and `/metrics`.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "temperature": 0,
    "max_tokens": 8
  }'
```

## Validation

Run CPU-independent tests and lint locally:

```bash
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

On the documented A100 image, `scripts/node/verify-a100.sh` additionally runs
strict CUDA lint, model/SafeTensors validation, project-owned cuTile kernels,
the full next-token path, the persistent-CUBIN-cache check, and a pinned
upstream cuTile smoke test. Retained correctness, sanitizer, API, and benchmark
evidence lives under `docs/validation/` and `docs/benchmarks/`.

## Benchmark

The built-in benchmark client targets any running OpenAI-compatible server. It
does not require CUDA and keeps open-loop request rate separate from the
client-side concurrency limit:

```bash
cargo run --bin tesseract -- bench \
  --base-url http://127.0.0.1:8000 \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer /path/to/model/tokenizer.json \
  --input-len 1024 \
  --output-len 128 \
  --num-prompts 1000 \
  --max-concurrency 32 \
  --output results.json
```

The default random dataset constructs deterministic token-length prompts with a
local tokenizer, including optional length variation and a shared prefix.
Streaming is enabled so the report includes request and token throughput, TTFT,
and inter-token latency. Use `tesseract bench --help` for all controls.


## Research notes

- [Inference engine architecture reference](docs/inference-engine-architecture-reference.md) — vLLM, SGLang, and SGLang's 2026 unified KV-memory work.
- [Batching architecture](docs/batching-architecture.md) — Tesseract's scheduler/backend contract, ragged tensor layout, execution modes, and validation gates.
- [v1 acceptance contract](docs/v1-acceptance.md) — required serving, engine, correctness, and performance gates.
- [v1 validation report](docs/v1-validation-report.md) — requirement-by-requirement evidence and measured A100 results.
- [A100 node setup](docs/a100-node-setup.md) — reproducible bootstrap and verification for the spot-worker base image.
