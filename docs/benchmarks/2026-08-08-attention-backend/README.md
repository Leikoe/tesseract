# Attention backend A100 benchmark

These are measured results, not performance targets.

## Environment

- Git: `b4ecef2d7637669a840a32d7e176c217413c76ff`
- Command: `target/release/tesseract bench --output /tmp/tesseract-bench-b4ecef2-repeat`
- GPU: NVIDIA A100-SXM4-80GB, 8.0, 580.126.09, 81920 MiB
- CUDA: Build cuda_13.3.r13.3/compiler.38244171_0
- Rust: rustc 1.89.0 (29483883e 2025-08-04)
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Model revision: `9213176726f574b556790deb65791e0c5aa438b6`
- Server arguments: `--listen 127.0.0.1:8000 --model meta-llama/Llama-3.2-1B-Instruct`

## Results

| Workload | Output tok/s | Mean TTFT | TTFT p99 | Mean inter-token | Request p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Batch 1 | 344.81 | 6.88 ms | 7.23 ms | 2.54 ms | 45.41 ms |
| Concurrency, first traffic | 1,691.32 | 12.23 ms | 16.78 ms | 3.29 ms | 67.87 ms |
| Concurrency, repeated | 1,666.46 | 12.41 ms | 16.13 ms | 3.34 ms | 68.74 ms |

This is the second complete suite on the revision. The first measured 1,640.20
tok/s first traffic and 1,585.73 tok/s repeated; retaining the repeat avoids
mistaking a single noisy run for a regression while the validation document
reports both. Compact machine-readable summaries of the repeat are retained
beside this file.
