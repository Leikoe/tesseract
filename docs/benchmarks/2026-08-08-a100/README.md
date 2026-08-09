# A100 v1 benchmark — 2026-08-08

These are measured results, not performance targets. The raw reports were
produced by the legacy benchmark harness and copied back from the spot node.

## Environment

- GPU: NVIDIA A100-SXM4-80GB, compute capability 8.0, 81,920 MiB
- Driver: 580.126.09
- CUDA: 13.3 (`compiler.38244171_0`)
- Rust: 1.89.0
- Tesseract: `9404e47a6c4bdd0881727a81a68793a34f2dc831`
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Model revision: `9213176726f574b556790deb65791e0c5aa438b6`
- Server: release build, all production defaults except binding to
  `127.0.0.1:8000`

Every request used streaming SSE, greedy decoding, a request-scoped seed, and
up to 16 generated tokens. The eight fixed prompts vary in length and task.
The harness records individual raw requests in addition to the summaries.

## Results

| Workload | Output tok/s | Mean TTFT | TTFT p99 | Mean inter-token | Inter-token p99 | Mean request | Request p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Batch 1, 8 requests | 32.42 | 372.21 ms | 1,382.35 ms | 2.94 ms | 3.04 ms | 408.19 ms | 1,427.57 ms |
| Concurrency 8, first shape pass | 79.30 | 316.89 ms | 838.45 ms | 61.22 ms | 139.31 ms | 1,080.17 ms | 1,668.46 ms |
| Concurrency 8, warm shapes | 610.04 | 74.24 ms | 115.46 ms | 7.36 ms | 20.20 ms | 156.33 ms | 207.80 ms |

The first concurrency-8 pass is intentionally retained: it includes lazy
capture of previously unseen exact batch/context shapes and makes that cost
visible. The identical second pass reuses those graphs and represents warm
steady state. Production deployments that require predictable first-request
latency should prewarm their expected batch shapes; v1 currently prewarms all
batch-1 context buckets and captures larger exact batch shapes on demand.

Final Prometheus evidence recorded a maximum batch size of 8, 21 graph
captures, 183 graph replays, 64 packed decode forwards covering 389 request
rows, 554 generated tokens, and 46 completed requests. Running requests and KV
usage were both zero after the benchmark.

## Raw evidence

- `tesseract-a100-batch1.json`: batch-1 raw requests and summary
- `tesseract-a100-mixed-c8.json`: first concurrency-8 pass
- `tesseract-a100-mixed-c8-warm.json`: identical warm concurrency-8 pass
- `tesseract-a100-final.metrics`: server metrics after all passes
