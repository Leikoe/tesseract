# A100 server smoke validation — 2026-08-07

This is durable evidence from the disposable A100 worker. It records measured
results, not performance targets.

## Revision and environment

- Git revision: `82ba95d6a57131a1c9bd3ae32875d25a18f64f58`
- GPU: NVIDIA A100-SXM4-80GB (`sm_80`)
- Driver: 580.126.09
- CUDA toolkit: 13.3.73
- Rust: 1.89.0
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Server: release build with `--features cuda`
- Runtime arguments: `--kv-capacity-tokens 4096 --max-sequence-length
  4096 --max-running 8 --max-batch-tokens 512 --prefill-chunk-tokens 128`

Strict `cargo clippy --features cuda --all-targets -- -D warnings` passed before
the runtime checks.

## Persistent CUBIN cache

A cold `cuda-check` process compiled 11 distinct kernels and reported
`stage2_source=tileiras`. A second process reported `stage2_source=disk` for
all 11. Cold stage-2 compilation ranged from about 61 ms to 412 ms per kernel;
disk stage-2 loading was below 1 ms per kernel in this run. The cache contained
11 `.cubin` entries after the test.

## Real model/API results

Readiness returned `{"status":"ready"}` after loading the model on the engine
thread. Model discovery returned the configured Llama model.

A non-streaming greedy chat request asking for the capital of France produced:

```json
{
  "content": "Paris",
  "finish_reason": "length",
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 1,
    "total_tokens": 25
  }
}
```

A streaming greedy request to count from one to three emitted `One`, newline,
`Two`, newline, `Three`, then a terminal chunk with `finish_reason: "stop"`, a
usage-only chunk (`20` prompt, `6` completion, `26` total), and `data: [DONE]`.

Two concurrent requests with `temperature=0.8`, `top_p=0.9`, and `seed=4242`
both generated the same six-token prefix, `"Hello, how are you`, demonstrating
request-scoped deterministic sampling while two requests shared the scheduler.

An SSE client was intentionally disconnected after one second during a
128-token request. The post-disconnect metrics were:

```text
tesseract_running_requests 0
tesseract_kv_tokens_used 0
tesseract_requests_cancelled_total 1
```

After all completed requests, queue depth, running requests, and KV usage were
all zero. Four prior requests were recorded as started and completed, with 19
generated tokens over 14 engine steps. Sending SIGINT terminated the process;
no `tesseract` process remained.

## Finding requiring follow-up

The runtime is correct enough to serve real requests, but it is not yet a v1
performance result. cuTile compiled new attention specializations as context
length changed. Stable decode buckets and CUDA graph capture/replay are still
required to remove shape-specific JIT work from steady-state decoding. The
current backend also executes scheduled requests serially inside a scheduler
step rather than packing them into one GPU batch. These are explicit remaining
acceptance gaps, not hidden by this smoke result.

## Independent BF16 logits reference

The isolated reference environment used PyTorch 2.8.0, Transformers 4.55.0,
BF16 weights/compute, and eager attention. `scripts/reference/llama_logits.py`
compared Tesseract against the reference for three raw prompts. In every case:

- tokenizer input IDs and prompt-token counts matched;
- greedy next-token IDs matched;
- every reference top-10 token appeared in Tesseract's top-20;
- the top-20 overlap was 20, 19, and 20 tokens respectively;
- maximum absolute differences over shared top logits were 0.125, 0.4375,
  and 0.125.

The gate uses a documented maximum absolute tolerance of 0.5. This accounts
for BF16 logit quantization and different reduction order between cuTile/cuBLAS
and PyTorch eager; it does not permit a different greedy token. The prompts and
results were:

| Prompt | Input IDs | Reference/Tesseract next token | Max abs logit diff |
| --- | --- | ---: | ---: |
| `The capital of France is` | `128000,791,6864,315,9822,374` | `12366` | 0.125 |
| `2 + 2 =` | `128000,17,489,220,17,284` | `220` | 0.4375 |
| `Rust is a programming` | `128000,49,592,374,264,15840` | `4221` | 0.125 |
