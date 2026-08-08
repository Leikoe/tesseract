# CUDA batch and sampler extraction A100 benchmark

These are measured results, not performance targets.

## Environment

- Git: `efb076670fbbaffebb7fe969855c3b1d9c46dc4c`
- Command: `target/release/tesseract bench --output /tmp/tesseract-bench-efb0766`
- GPU: NVIDIA A100-SXM4-80GB, 8.0, 580.126.09, 81920 MiB
- CUDA: Build cuda_13.3.r13.3/compiler.38244171_0
- Rust: rustc 1.89.0 (29483883e 2025-08-04)
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Model revision: `9213176726f574b556790deb65791e0c5aa438b6`
- Server arguments: `--listen 127.0.0.1:8000 --model meta-llama/Llama-3.2-1B-Instruct`

## Results

| Workload | Output tok/s | Mean TTFT | TTFT p99 | Mean inter-token | Request p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Batch 1 | 337.48 | 7.55 ms | 7.96 ms | 2.56 ms | 46.50 ms |
| Concurrency, first traffic | 1,648.89 | 12.34 ms | 16.54 ms | 3.35 ms | 70.07 ms |
| Concurrency, repeated | 1,685.06 | 11.99 ms | 16.57 ms | 3.29 ms | 68.89 ms |

The suite used 8 batch-1 requests and two identical concurrency-8 passes of 16
requests, with at most 16 generated tokens and two warmup requests. The server
exited cleanly. Compact machine-readable concurrent summaries are retained
beside this file.
