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
