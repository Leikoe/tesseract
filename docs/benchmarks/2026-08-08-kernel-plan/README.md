# Typed kernel plan benchmark

Revision `9b109446583adc41b2dc72f9498092d48fd26227` on an
NVIDIA A100-SXM4-80GB with CUDA 13.3 and Llama-3.2-1B-Instruct.

The construction-time plan is performance-neutral: it resolves before warmup
and its immutable block/shape choices replace the same constants previously
embedded in Llama. The retained concurrency-8 workload reached 1,670.95
completion tokens/s on the first-shape run and 1,719.79 completion tokens/s on
the identical warm-shape run. This is within ordinary run-to-run variation of
the preceding attention-backend measurements (1,691.32 and 1,666.46 tokens/s).

The JSON files retain the reproducibility metadata and aggregate summaries from
the generated benchmark directory on the A100 node.
