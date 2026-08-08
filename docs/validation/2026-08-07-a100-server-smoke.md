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

Follow-up revision `5b924e9` moved the changing context length and query start
into device metadata and padded gathered KV to power-of-two context buckets. A
19-token prefill followed by ten decode tokens then compiled only one attention
specialization for that bucket, rather than one per context length, and still
generated `One, Two, Three, Four, Five.`. The three-prompt logits gate continued
to pass. Full model graph replay remains pending.

A subsequent CUDA smoke gate captured an allocation-free BF16 cuBLAS GEMM in a
CUDA graph, replayed it, and compared the replay output to the eager result.
This proves the selected cuBLAS/CUDA/cuTile stack supports graph capture; it is
not evidence that the complete Llama decode path is captured yet.

The complete decode path was then converted to static per-bucket buffers and a
single graph spanning embedding, all 16 transformer layers, flat-KV writes and
gathers, attention, MLPs, final normalization, and the tied LM head. For a
six-token completion, metrics directly reported one eager prefill, one graph
capture, and five graph replays; output remained `One\nTwo\nThree` and stopped
on EOS. That run measured 1.557 s TTFT and 1.285 s total inter-token time across
five intervals (about 257 ms/interval). These are unoptimized measurements, not
targets: request metadata still uses allocating host-to-device updates,
stochastic sampling still copies logits to the host, and scheduled requests are
executed serially.

Startup warmup was subsequently enabled for every power-of-two context bucket.
With a 256-token test capacity, the backend logged five warmed graph buckets
before the server's ready log. Before any request, metrics showed readiness 1,
five graph captures, zero eager forwards, and zero graph replays.

Greedy graph replay was then extended with a two-stage BF16 cuTile argmax. The
argmax is part of the captured graph, so steady-state greedy decode copies one
`u32` token ID to the host instead of 128,256 BF16 logits. On the real model,
the same 22-token chat prompt was run for eight completion tokens through both
the temperature-zero GPU-token path and the nonzero-temperature host-logits
path (`temperature=0.000001`, which makes the distribution effectively
greedy). Both produced exactly `Here is the count from one to three`. Metrics
reported two eager prefills, 14 graph replays, and 16 generated tokens. Startup
still warmed five graph buckets and reported ready only after capture.

The backend was subsequently changed to execute aligned greedy decode requests
as one packed GPU forward. It flattens each request's independent physical KV
slot map into a ragged, padded batch; cuTile gathers K/V by row, attention reads
a device context length per row, cuBLAS projects all rows together, and a
batched cuTile argmax returns one token ID per request. Two simultaneous
19-token prompts completed with eight tokens each as `The capital of France is
Paris.` and `The capital of Germany is Berlin.`. Eight scheduler steps executed
only nine eager forwards: two first-step prefills and seven two-row packed
decode forwards. The same requests run separately through batch-1 CUDA graphs
produced byte-identical text. The maximum scheduled batch was two, and KV usage
and running requests returned to zero.

Packed decode was then moved into CUDA graphs keyed by exact batch size and the
power-of-two maximum context bucket. Batch-1 buckets remained prewarmed. The
first two-request, bucket-32 workload raised graph captures from five to six;
the second identical workload left captures at six while graph replays rose
from eight to sixteen. Across both runs, packed decode counters rose from six
forwards/12 request rows to 12 forwards/24 request rows. Both runs retained the
same France/Paris and Germany/Berlin outputs, with zero KV and running-request
gauges afterward. A graph capture or replay failure is remembered per shape
and routes subsequent work through the packed eager implementation.

## Default-capacity and memory-safety gates

The release server was also started with no capacity overrides. It prewarmed 12
batch-1 context buckets for the default 32,768-token KV capacity, reported
ready, and used 6,042 MiB according to `nvidia-smi`. A one-token request returned
`Paris` with 23 prompt tokens and one completion token.

Compute Sanitizer 2026.2.1 initially exposed stream-ordered lifetime hazards:
some last-owner tensors were moved into asynchronous operations and dropped on
cuTile's separate deallocator stream before the requested execution stream had
synchronized. Tesseract now retains an `Arc` owner through each host copy and
each eager cuBLAS `sync_on` call. The following gates then reported `ERROR
SUMMARY: 0 errors`:

```bash
compute-sanitizer --tool memcheck --error-exitcode=99 \
  ./target/release/cuda-check

compute-sanitizer --tool memcheck --error-exitcode=99 \
  ./target/release/next-token-check \
  --model-path /home/ubuntu/models/Llama-3.2-1B-Instruct \
  --prompt "The capital of France is"
```

The second command covered a complete 16-layer eager forward and retained next
token `12366` (` Paris`). Finally, the HTTP server itself ran under memcheck at
256-token capacity. Two concurrent eight-token requests exercised eager
prefill, one batch-2 graph capture, seven packed graph replays, flat KV access,
and GPU argmax; both responses were correct. After SIGINT shutdown, Compute
Sanitizer again reported zero errors.

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
