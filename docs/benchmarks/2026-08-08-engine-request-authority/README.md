# Engine request-authority A100 benchmark

These are measured results, not performance targets.

## Environment

- Git: `9f5c8caa527a647a1ced44dab06eeb5f23ca4327`
- Command: `target/release/tesseract bench --output /tmp/tesseract-bench-9f5c8ca`
- GPU: NVIDIA A100-SXM4-80GB, 8.0, 580.126.09, 81920 MiB
- CUDA: Build cuda_13.3.r13.3/compiler.38244171_0
- Rust: rustc 1.89.0 (29483883e 2025-08-04)
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Model revision: `9213176726f574b556790deb65791e0c5aa438b6`
- Server arguments: `--listen 127.0.0.1:8000 --model meta-llama/Llama-3.2-1B-Instruct`

## Results

| Workload | Output tok/s | Mean TTFT | TTFT p99 | Mean inter-token | Request p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Batch 1 | 335.10 | 7.73 ms | 8.59 ms | 2.56 ms | 47.13 ms |
| Concurrency, first traffic | 1,654.93 | 12.37 ms | 17.76 ms | 3.35 ms | 70.31 ms |
| Concurrency, repeated | 1,623.51 | 13.98 ms | 20.66 ms | 3.34 ms | 72.82 ms |

The suite used 8 batch-1 requests and two identical concurrency-8 passes of 16
requests, with at most 16 generated tokens and two warmup requests. The server
exited cleanly. Compact machine-readable concurrent summaries are retained
beside this file; the complete raw suite remains reproducible through the
recorded command.
