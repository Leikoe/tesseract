# Tesseract A100 serving benchmark

These are measured results, not performance targets.

## Environment

- Git: `d7ff81f56e3dc23989aa6f402cc67fb8e223544e`
- Command: `cargo bench-a100 --output /tmp/tesseract-bench-command-d7ff81f`
- GPU: NVIDIA A100-SXM4-80GB, 8.0, 580.126.09, 81920 MiB
- CUDA: Build cuda_13.3.r13.3/compiler.38244171_0
- Rust: rustc 1.89.0 (29483883e 2025-08-04)
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Model revision: `9213176726f574b556790deb65791e0c5aa438b6`
- Server arguments: `--listen 127.0.0.1:8000 --model meta-llama/Llama-3.2-1B-Instruct`

## Results

| Workload | Output tok/s | Mean TTFT | TTFT p99 | Mean inter-token | Request p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Batch 1 | 28.09 | 435.58 ms | 1627.21 ms | 2.93 ms | 1672.46 ms |
| Concurrency, first shape pass | 66.29 | 358.56 ms | 994.72 ms | 91.49 ms | 2116.56 ms |
| Concurrency, warm shapes | 595.74 | 74.83 ms | 116.64 ms | 6.72 ms | 208.15 ms |

The first concurrent pass includes lazy capture of exact batch/context
shapes. The identical warm pass reuses those graphs.

Raw request results, final Prometheus metrics, the server log, and the
suite manifest are retained beside this file.
