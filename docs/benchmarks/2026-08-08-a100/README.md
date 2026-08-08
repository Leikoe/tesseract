# A100 v1 benchmark — 2026-08-08

These are measured results, not performance targets. The raw reports were
produced by `scripts/benchmark/a100_v1.py` and copied back from the spot node.

## Environment

- GPU: NVIDIA A100-SXM4-80GB, compute capability 8.0, 81,920 MiB
- Driver: 580.126.09
- CUDA: 13.3 (`compiler.38244171_0`)
- Rust: 1.89.0
- Tesseract: `9ebb36ba26e7c606bcd6e6976bb8023f9482a6be`
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
| Batch 1, 8 requests | 27.97 | 437.28 ms | 1,793.16 ms | 2.95 ms | 3.04 ms | 473.30 ms | 1,838.31 ms |
| Concurrency 8, first shape pass | 47.37 | 320.70 ms | 845.97 ms | 114.50 ms | 275.69 ms | 1,755.17 ms | 2,577.64 ms |
| Concurrency 8, warm shapes | 584.46 | 76.27 ms | 117.67 ms | 7.48 ms | 20.62 ms | 159.62 ms | 210.66 ms |

The first concurrency-8 pass is intentionally retained: it includes lazy
capture of previously unseen exact batch/context shapes and makes that cost
visible. The identical second pass reuses those graphs and represents warm
steady state. Production deployments that require predictable first-request
latency should prewarm their expected batch shapes; v1 currently prewarms all
batch-1 context buckets and captures larger exact batch shapes on demand.

Final Prometheus evidence recorded a maximum batch size of 8, 22 graph
captures, 183 graph replays, 65 packed decode forwards covering 390 request
rows, 554 generated tokens, and 46 completed requests. Running requests and KV
usage were both zero after the benchmark.

## Raw evidence

- `tesseract-a100-batch1.json`: batch-1 raw requests and summary
- `tesseract-a100-mixed-c8.json`: first concurrency-8 pass
- `tesseract-a100-mixed-c8-warm.json`: identical warm concurrency-8 pass
- `tesseract-a100-final.metrics`: server metrics after all passes
