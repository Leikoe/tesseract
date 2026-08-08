# Reusable CUDA batch benchmark

Revision `d6ec8c34e24e1081159af7b1f22d6caecaa8f2d7` on an NVIDIA
A100-SXM4-80GB with CUDA 13.3, Rust 1.89.0, and the pinned
Llama-3.2-1B-Instruct artifact.

| Run | Batch-1 tok/s | Concurrent first tok/s | Concurrent warm tok/s |
| --- | ---: | ---: | ---: |
| First | 336.88 | 1,650.39 | 1,583.52 |
| Immediate repeat | 342.09 | 1,699.52 | 1,682.64 |

Each batch-1 workload used eight sequential requests. Each concurrent workload
used sixteen requests at concurrency eight, two warmup requests, and at most 16
generated tokens per request. The first concurrent pass may capture graph
shapes; the warm pass repeats identical shapes.

Both runs lie within prior node variance. These measurements show no regression
but do not isolate a statistically credible throughput gain from host allocation
reuse. The adjacent JSON files retain the exact aggregate summaries.
