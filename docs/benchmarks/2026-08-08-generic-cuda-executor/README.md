# Generic CUDA executor A100 benchmark

These are measured results, not performance targets.

## Environment

- Git: `99ac26e1f88adccb01e9c9d2f2d9c440f317d8ef`
- Command: `target/release/tesseract bench --output /tmp/tesseract-bench-99ac26e`
- GPU: NVIDIA A100-SXM4-80GB, 8.0, 580.126.09, 81920 MiB
- CUDA: Build cuda_13.3.r13.3/compiler.38244171_0
- Rust: rustc 1.89.0 (29483883e 2025-08-04)
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Model revision: `9213176726f574b556790deb65791e0c5aa438b6`
- Server arguments: `--listen 127.0.0.1:8000 --model meta-llama/Llama-3.2-1B-Instruct`

## Results

| Workload | Output tok/s | Mean TTFT | TTFT p99 | Mean inter-token | Request p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Batch 1 | 335.84 | 7.60 ms | 8.02 ms | 2.56 ms | 46.47 ms |
| Concurrency, first traffic | 1,672.04 | 11.55 ms | 16.02 ms | 3.40 ms | 69.40 ms |
| Concurrency, repeated | 1,663.19 | 11.95 ms | 15.41 ms | 3.35 ms | 67.87 ms |

The suite used 8 batch-1 requests and two identical concurrency-8 passes of 16
requests, with at most 16 generated tokens and two warmup requests. The server
exited cleanly. Compact machine-readable concurrent summaries are retained
beside this file.
