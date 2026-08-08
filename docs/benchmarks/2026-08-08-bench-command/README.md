# Tesseract A100 serving benchmark

These are measured results, not performance targets.

## Environment

- Git: `1fa6583b33c261d65b5ee87f8caa12848475e4ff`
- Command: `target/release/tesseract bench --output /tmp/tesseract-bench-clap-1fa6583`
- GPU: NVIDIA A100-SXM4-80GB, 8.0, 580.126.09, 81920 MiB
- CUDA: Build cuda_13.3.r13.3/compiler.38244171_0
- Rust: rustc 1.89.0 (29483883e 2025-08-04)
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Model revision: `9213176726f574b556790deb65791e0c5aa438b6`
- Server arguments: `--listen 127.0.0.1:8000 --model meta-llama/Llama-3.2-1B-Instruct`

## Results

| Workload | Output tok/s | Mean TTFT | TTFT p99 | Mean inter-token | Request p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Batch 1 | 25.69 | 478.20 ms | 1799.53 ms | 3.02 ms | 1844.99 ms |
| Concurrency, first shape pass | 63.75 | 384.17 ms | 1057.35 ms | 76.27 ms | 2101.32 ms |
| Concurrency, warm shapes | 573.40 | 78.46 ms | 118.57 ms | 6.87 ms | 208.19 ms |

The first concurrent pass includes lazy capture of exact batch/context
shapes. The identical warm pass reuses those graphs.

Raw request results, final Prometheus metrics, the server log, and the
suite manifest are retained beside this file.
