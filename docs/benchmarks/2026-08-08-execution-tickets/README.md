# Execution-ticket A100 benchmark

These are measured results, not performance targets.

## Environment

- Git: `dea5e06a8e4a32be8393f81f692b8c519df1cbff`
- Command: `target/release/tesseract bench --output /tmp/tesseract-bench-dea5e06`
- GPU: NVIDIA A100-SXM4-80GB, 8.0, 580.126.09, 81920 MiB
- CUDA: Build cuda_13.3.r13.3/compiler.38244171_0
- Rust: rustc 1.89.0 (29483883e 2025-08-04)
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Model revision: `9213176726f574b556790deb65791e0c5aa438b6`
- Server arguments: `--listen 127.0.0.1:8000 --model meta-llama/Llama-3.2-1B-Instruct`

## Results

| Workload | Output tok/s | Mean TTFT | TTFT p99 | Mean inter-token | Request p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Batch 1 | 342.54 | 7.09 ms | 7.72 ms | 2.54 ms | 45.64 ms |
| Concurrency, first traffic | 1,622.77 | 12.44 ms | 16.70 ms | 3.43 ms | 71.99 ms |
| Concurrency, repeated | 1,687.61 | 10.66 ms | 14.97 ms | 3.36 ms | 66.76 ms |

The suite used 8 batch-1 requests and two identical concurrency-8 passes of 16
requests, with at most 16 generated tokens and two warmup requests. The server
exited cleanly. Compact machine-readable concurrent summaries are retained
beside this file.
