# Qwen3.6 NVFP4 serving benchmark on A100

This directory retains the raw reports for a text-only serving benchmark of
`nvidia/Qwen3.6-35B-A3B-NVFP4` on one NVIDIA A100-SXM4-80GB. The benchmark is
intended to expose latency, saturation, queueing, and genuinely long-context
behavior; a short-prompt throughput number is not treated as a complete serving
result.

## Reproducibility

- Date: 2026-08-09
- GPU: NVIDIA A100-SXM4-80GB, compute capability 8.0, 81,920 MiB
- Driver: 580.126.09
- Host: Linux 6.8.0-90-generic x86_64
- Server source checkout: `7a7aa316d532c042960d7b66e4c77ffb7aa88486`
- CUDA server binary: built after model changes at `a4779e8`; SHA-256
  `64f5751f71ff9fc1cc6b8f2f773bf701d89dcc1ab74f9d2dad25036919a4cb83`
- HTTP-only benchmark binary: SHA-256
  `78f86ece3730af2f2ebf0568fa0afebf94fa0530b929381a86c4b43656cf5e84`
- Model config SHA-256:
  `58aefa1c9eff7989f431d748f2ddec39446cb1fd2a69acc46e285c6a37b0cecc`

The model declares a 262,144-token text context window, 40 layers (30 linear
attention and 10 full attention), 16 query heads, 2 KV heads, and head dimension
256.

The isolated server used:

```text
target/release/tesseract \
  --listen 127.0.0.1:18100 \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --model-path /home/ubuntu/models/Qwen3.6-35B-A3B-NVFP4 \
  --max-running 4 \
  --max-batch-tokens 4096 \
  --prefill-chunk-tokens 512 \
  --max-sequence-length 262144 \
  --kv-capacity-tokens 140000 \
  --log warn
```

All requests use the built-in `tesseract bench` HTTP client, the checkpoint's
actual tokenizer, deterministic random prompts, SSE streaming, and server
reported token usage. Cold and warm reports are retained separately. The
open-loop load tests use Poisson arrivals and report both service time and time
spent waiting behind the client concurrency bound.

The half-window case generates 131,062 prompt tokens before chat templating.
The template contributes exactly 10 tokens, so the server observes exactly
131,072 prompt tokens: 50% of the declared context window. Eight requested
output tokens make the resulting sequence longer than 50% of the window.

## Results

The short and medium-context results are measurements, not performance targets.
The 1,024-token cases are labeled medium prefill and are not presented as
long-context tests.

Closed-loop results:

| Scenario | Phase | Requests | Peak client concurrency | Request/s | Input tok/s | Output tok/s | Mean TTFT | p99 TTFT | Mean ITL | Failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 input, 32 output, latency | fresh server | 16 | 1 | 0.362 | 49.92 | 11.58 | 509.87 ms | 511.52 ms | 72.73 ms | 0 |
| 128 input, 32 output, latency | warm | 32 | 1 | 0.332 | 45.84 | 10.61 | 732.83 ms | 4,493.50 ms | 73.63 ms | 0 |
| 128 input, 32 output, throughput | fresh server | 32 | 8 | 1.150 | 158.94 | 36.80 | 3,208.49 ms | 6,575.51 ms | 120.90 ms | 0 |
| 128 input, 32 output, throughput | warm | 64 | 8 | 1.269 | 175.55 | 40.61 | 2,556.09 ms | 5,792.45 ms | 120.91 ms | 0 |
| 1,024 input, 8 output, medium prefill | fresh server | 16 | 4 | 0.442 | 457.07 | 3.54 | 8,369.81 ms | 8,375.81 ms | 97.00 ms | 0 |
| 1,024 input, 8 output, medium prefill | warm | 32 | 4 | 0.443 | 457.73 | 3.54 | 8,361.86 ms | 8,380.85 ms | 96.54 ms | 0 |

The warm batch-one TTFT distribution has one substantial outlier: p50 is
509.93 ms while p99 is 4,493.50 ms. The raw report is retained rather than
discarding that request as noise.

<!-- The half-window row and queue-aware open-loop table are filled after those
     retained runs complete. -->

The exact half-window run was interrupted by the announced spot-node shutdown
before its first token. At the final captured sample it had completed 120,832
of 131,072 prompt positions (92.1875%) in 65m33s, across 237 serial 512-token
engine steps. It had generated zero tokens and recorded zero failures. This is
retained as [`half-window-partial.json`](half-window-partial.json), explicitly
as partial progress evidence rather than a completed TTFT or throughput result.

The same request was reproduced on a replacement A100 from checkout `9f74573`.
It reached 42,496 positions (32.421875%) in 9m36s and was deliberately stopped
once the repeated behavior was established. The SSE keep-alive change was then
verified end to end: the server cancelled the disconnected request and returned
KV use to zero. The independent observation is retained as
[`half-window-new-node-partial.json`](half-window-new-node-partial.json).

## SGLang control

SGLang 0.5.17 was installed in an isolated virtual environment on the
replacement A100 and pointed at the same local checkpoint. It auto-detected
`modelopt_mixed`, used Marlin for the SM80 NVFP4/FP8 weight-only paths, Triton
for GDN, FlashInfer for attention, and BF16 KV. Prefill and decode CUDA graphs
were disabled, so this is not a graph-assisted best case.

After SGLang's model-load and one-time kernel-compilation warmup, its official
serving benchmark sent integer token IDs directly to `/generate`:

```text
python -m sglang.benchmark.serving \
  --backend sglang \
  --base-url http://127.0.0.1:18200 \
  --model /home/ubuntu/models/Qwen3.6-35B-A3B-NVFP4 \
  --tokenizer /home/ubuntu/models/Qwen3.6-35B-A3B-NVFP4 \
  --dataset-name random-ids \
  --tokenize-prompt \
  --num-prompts 1 \
  --random-input-len 131072 \
  --random-output-len 8 \
  --random-range-ratio 1.0 \
  --request-rate inf \
  --max-concurrency 1 \
  --warmup-requests 0 \
  --seed 42
```

This excludes server-side tokenization and guarantees the exact input length.
The content is deterministic random token IDs rather than the Tesseract
client's decoded random text, so the comparison is shape-equivalent rather than
token-for-token identical.

| Engine | 131,072-token TTFT | E2E for 8 output tokens | Input throughput | Status |
| --- | ---: | ---: | ---: | --- |
| SGLang 0.5.17 | 15.982 s | 16.372 s | 8,000.27 tok/s | completed |
| Tesseract, replacement A100 | >576 s | >576 s | — | deliberately stopped before first token |
| Tesseract, prior A100 | >3,933 s | >3,933 s | — | spot shutdown before first token |

The observed TTFT gap is therefore **greater than 36.0×** against the shorter
Tesseract observation and **greater than 246.1×** against the longer one. These
are conservative lower bounds because neither Tesseract request reached its
first token. The raw SGLang report and filtered server-side batch trace are
retained as
[`sglang-native-half-window.jsonl`](sglang-native-half-window.jsonl) and
[`sglang-native-half-window-server.log`](sglang-native-half-window-server.log).
The trace contains exactly sixteen 8,192-token prefill batches.

The two open-loop files were produced before client queue time was added to the
report schema. They are retained because the node was being reclaimed, but
their service-time E2E values must not be interpreted as scheduled E2E under
overload. The 2 req/s case reached only 1.366 req/s with peak client concurrency
16, which demonstrates saturation; a future rerun must use the queue-aware
client from `7a7aa31`.

## Long-context interpretation

The reference audit changes the diagnosis: full attention is structurally poor,
but the first-order defect is host-side serialization throughout every model
step.

Each of the 30 GDN layers performs roughly 17 `sync_on(stream)` allocations or
uploads in one prefill forward. The barriers are visible around convolution,
gates, KKT, W/U, recurrent output, and output gating in
`src/cuda/gdn.rs:1344-1475` and `src/cuda/gdn.rs:1622-1701`. That is roughly 510
forced synchronizations per scheduler forward before embedding, metadata, or
MoE is counted. Every one of the 40 MoE layers then drains the stream in
`src/cuda/moe.rs:293-329`; `src/cuda/qwen3_5_moe.rs:1315-1338` relies on that
drain so the workspace can be reclaimed. Seven batch arrays and four GDN plan
arrays are also uploaded synchronously (`src/cuda/qwen3_5_moe.rs:483-505` and
`src/cuda/gdn.rs:1264-1302`).

The fixed 512-token scheduler chunk repeats this barrier waterfall 256 times for
a 131,072-token prompt. SGLang uses an 8,192-token A100 default and separates
the scheduler chunk from its internal GDN kernel tile. A larger chunk alone is
not the final solution, but Tesseract's current value multiplies every other
defect by 16 relative to that production baseline.

The full-attention kernel remains a serious long-context bottleneck.
`ragged_attention_bf16` assigns one CTA to one query row and head (`BM = 1`),
then traverses the context in 16-key blocks. Each query row repeats the physical
slot lookup and K/V gather and cannot reuse a staged K/V tile with neighboring
queries. vLLM on SM80 selects FlashAttention 2; SGLang's SM80 Triton extend
kernel uses 64-query by 64-key tiles. FlashAttention 2 does not split one
softmax row across CTAs on A100, so the relevant improvement is multi-query
tiling, coalesced paged-KV access, tensor-core work, and on-chip K/V reuse—not a
claim that the serial online-softmax recurrence disappears.

During the half-window run `nvidia-smi` consistently reported 99–100% activity
but only about 209–212 W. That is consistent with a GPU kept occupied by small,
synchronization-heavy kernels and inefficient memory traffic rather than a
well-saturated production prefill path.

## Reference-backed production order

The clean implementation order is:

1. Lower batch metadata once into persistent device buffers and make every GDN
   stage enqueue asynchronously into a pointer-stable, stream-ordered shape
   arena. Permit one synchronization only at the final host-visible boundary.
2. Remove the per-MoE-layer stream drain. Use static shape slabs, ping-pong
   arenas, or event epochs so scratch lifetime follows stream order.
3. Fuse the separate QKV/A/B/Z GDN projections, adopt a production chunked GDN
   backend, and tune its A100 tile independently of scheduler chunk size.
4. Fuse and tune routed/shared MoE work, including gate+up projection and
   activation, using actual device-produced expert counts rather than repeated
   worst-case padded capacity where possible.
5. Replace `BM = 1` attention with a paged, multi-query FlashAttention-style
   backend and reusable planned outputs/workspace.
6. Increase and dynamically choose the scheduler prefill chunk using memory and
   decode-SLO pressure. Add piecewise prefill CUDA graphs only after addresses
   and lifetimes are stable.

This ordering follows the actual vLLM and SGLang implementations under
`references/`, not merely their public APIs. SGLang fuses GDN projections and
enters one backend extend call without host synchronization
(`references/sglang/python/sglang/srt/layers/attention/linear/gdn_backend.py:563-684`).
vLLM selects its SM80 GDN backend once and warms fixed-shape chunk kernels before
serving
(`references/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:88-136`
and `:991-1014`). Both keep attention/model execution asynchronous and reuse
planned buffers.

## Benchmark finding: disconnected prefill

During the first half-window attempt, the benchmark timeout was discovered to
be too short and the HTTP client was terminated. The server did not observe the
disconnect while it was still producing prefill chunks: it retained the
request, continued consuming the GPU, and kept its KV allocation until the
isolated server was stopped. This is a production cancellation defect exposed
by the benchmark, not a result sample. The invalid attempt is excluded from the
raw reports. Commit `fd07675` adds five-second SSE keep-alives so the HTTP
transport can observe a dead peer during a long prefill and drop the request
stream, which activates the scheduler's existing cancellation path.

The exact case was restarted with a three-hour timeout. This timeout was chosen
from observed progress rather than prompt length alone: full-attention work per
512-token chunk grows with the accumulated context, so extrapolating linearly
from the first chunks materially underestimates total TTFT.
