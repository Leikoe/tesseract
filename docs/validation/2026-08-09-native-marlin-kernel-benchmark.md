# Native Marlin extraction and A100 kernel comparison

This record covers Tesseract's framework-independent extraction of the SM80
Marlin dense weight-only GEMM and its first direct comparison with the existing
cuTile kernels. It does not compare through SGLang, vLLM, Torch, Python, or TVM.

## Implementation under test

The Apache-2.0 Marlin kernel, dequantization, MMA, and pipeline headers were
extracted from the pinned vLLM reference at
`f7ef489e93cf92b8d6ce7403b49f1db867bcc35e`. The repack kernel is derived from
TokenSpeed's vendored Marlin source under its included MIT notice. Tesseract
supplies its own narrow C ABI, SM80 specializations, Rust allocation and stream
ownership, ModelOpt layout conversion, scale permutation, and error handling.

The extracted paths are:

- BF16 activation × E4M3 weight → BF16 output, FP32 accumulation;
- BF16 activation × packed E2M1 weight with E4M3 group-16 scales → BF16
  output, FP32 accumulation.

Weights are repacked at construction and remain 8-bit or packed 4-bit. NVFP4
scales remain 8-bit in Marlin's permuted S0E5M3 representation. No persistent
BF16 weight expansion is used. The built-in A100 differential probe reports
zero maximum absolute error for both extracted paths and both existing cuTile
paths on its deterministic oracle cases.

## Timing method

Revision `c540456` produced the retained result on one A100-SXM4-80GB. The
built-in `cuda-kernel-bench` uses the same BF16 activations and original
ModelOpt bytes/scales for each implementation. Allocation, upload, JIT, and
load-time repacking occur before the timing boundary. Each individual launch is
timed with CUDA driver events on the same stream after two warmups. Seven raw
samples are retained per row; the table reports their median.

This first series deliberately measures hot-weight steady state: every timed
iteration reuses the same packed artifact. It is valid for compute-heavy
prefill and for repeated decode access, but it must not be presented as a
cold-weight DRAM-bandwidth result. A cache-rotating series remains required.

The exact command was:

```text
cuda-kernel-bench --rows 16,512,8192 \
  --warmup-iterations 2 --iterations 7 \
  --output /tmp/marlin-vs-cutile-production-shapes.json
```

## Result

| Quantization and logical shape | Rows | cuTile median | Marlin median | Speedup |
| --- | ---: | ---: | ---: | ---: |
| FP8 `K=2048,N=8192` | 16 | 0.248832 ms | 0.023552 ms | 10.56x |
| FP8 `K=2048,N=8192` | 512 | 4.712448 ms | 0.145408 ms | 32.41x |
| FP8 `K=2048,N=8192` | 8192 | 60.726273 ms | 1.643520 ms | 36.95x |
| NVFP4 `K=2048,N=512` | 16 | 0.107520 ms | 0.022528 ms | 4.77x |
| NVFP4 `K=2048,N=512` | 512 | 0.280576 ms | 0.031744 ms | 8.84x |
| NVFP4 `K=2048,N=512` | 8192 | 2.675712 ms | 0.119808 ms | 22.33x |
| NVFP4 `K=512,N=2048` | 16 | 0.059392 ms | 0.013312 ms | 4.46x |
| NVFP4 `K=512,N=2048` | 512 | 0.492544 ms | 0.029696 ms | 16.59x |
| NVFP4 `K=512,N=2048` | 8192 | 5.809152 ms | 0.143360 ms | 40.52x |

At 8,192 rows, extracted Marlin reaches 167.25 logical TFLOP/s for W8A16,
143.40 TFLOP/s for the `2048→512` W4A16 shape, and 119.84 TFLOP/s for the
`512→2048` W4A16 shape. The corresponding cuTile results are 4.53, 6.42, and
2.96 TFLOP/s.

The raw report is
[`marlin-vs-cutile-production-shapes.json`](../benchmarks/2026-08-09-qwen-serving/post-tiled/marlin-vs-cutile-production-shapes.json).
The preliminary three-sample small-M run is retained separately as
[`marlin-vs-cutile-quick.json`](../benchmarks/2026-08-09-qwen-serving/post-tiled/marlin-vs-cutile-quick.json).

## Grouped Marlin MoE follow-up

Revision `92caa8f` additionally extracts TokenSpeed's grouped Marlin MoE device
kernel behind a framework-independent C ABI. The selector is deliberately
specialized to the checkpoint's BF16 × NVFP4/E4M3 contract; unrelated integer,
FP8, MXFP4, zero-point, and act-order specializations are not compiled. The
artifact owns all 256 repacked experts, retains 8-bit block scales and one BF16
global scale per expert, and consumes device-resident padded expert blocks.

A permanent two-expert differential test uses distinct packed weights, block
scales, and global scales for each expert. It passed on the A100 with zero
maximum absolute error. The benchmark then compared the same 256-expert
artifact and BF16 inputs under two sorted routing distributions:

- `uniform`: padded blocks distributed across the expert range;
- `skewed`: 75% of padded blocks assigned to expert 0, with the remainder
  distributed across other experts.

The row count is the padded dispatched-row count seen by the expert GEMM, not
the number of logical prompt tokens. Rows 512 and 8,192 use 64-row expert
blocks; row 16 uses one 16-row block. Allocation, upload, and expert-bank
repacking remain outside the CUDA-event interval.

| Expert projection | Rows | Routing | cuTile grouped | Marlin grouped | Marlin speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| Gate/up `K=2048,N=512` | 16 | uniform | 0.105472 ms | 0.112640 ms | 0.94x |
| Gate/up `K=2048,N=512` | 512 | uniform | 0.108544 ms | 0.114688 ms | 0.95x |
| Gate/up `K=2048,N=512` | 512 | skewed | 0.102400 ms | 0.115712 ms | 0.88x |
| Gate/up `K=2048,N=512` | 8192 | uniform | 1.059840 ms | 0.328704 ms | 3.22x |
| Gate/up `K=2048,N=512` | 8192 | skewed | 1.024000 ms | 0.318464 ms | 3.22x |
| Down `K=512,N=2048` | 16 | uniform | 0.063488 ms | 0.036864 ms | 1.72x |
| Down `K=512,N=2048` | 512 | uniform | 0.172032 ms | 0.056320 ms | 3.05x |
| Down `K=512,N=2048` | 512 | skewed | 0.169984 ms | 0.057344 ms | 2.96x |
| Down `K=512,N=2048` | 8192 | uniform | 2.143232 ms | 0.354304 ms | 6.05x |
| Down `K=512,N=2048` | 8192 | skewed | 2.089984 ms | 0.333824 ms | 6.26x |

This rejects a single unconditional backend choice. The existing persistent
cuTile gate/up kernel remains slightly faster at small dispatched batches;
grouped Marlin wins gate/up once the dispatched batch is large and wins the
down projection at every measured size. Production selection must therefore
include projection geometry and dispatched rows. The raw reports are
[`marlin-grouped-vs-cutile-quick.json`](../benchmarks/2026-08-09-qwen-serving/post-tiled/marlin-grouped-vs-cutile-quick.json)
and
[`marlin-grouped-vs-cutile-production.json`](../benchmarks/2026-08-09-qwen-serving/post-tiled/marlin-grouped-vs-cutile-production.json).

The gate/up rows above still time one projection. The production-equivalent
follow-up compares Tesseract's fused two-bank cuTile kernel (independent gate
and up global scales, register SwiGLU, one BF16 store) against two grouped
Marlin launches with independent scale domains followed by the same BF16 SiLU
kernel. Keeping independent scales is intentional: SGLang's SM80 Marlin
fallback collapses differing gate/up global scales to the gate scale and warns
that accuracy may be affected. Tesseract does not make that approximation.

| Dispatched rows | Routing | Fused cuTile gate/up+SiLU | Faithful Marlin gate/up+SiLU | Marlin speedup |
| ---: | --- | ---: | ---: | ---: |
| 512 | uniform | 0.208896 ms | 0.232448 ms | 0.90x |
| 512 | skewed | 0.199680 ms | 0.230400 ms | 0.87x |
| 640 | uniform | 0.198656 ms | 0.196608 ms | 1.01x |
| 640 | skewed | 0.197632 ms | 0.195584 ms | 1.01x |
| 768 | uniform | 0.198656 ms | 0.164864 ms | 1.21x |
| 768 | skewed | 0.196608 ms | 0.161792 ms | 1.22x |
| 1024 | uniform | 0.385024 ms | 0.177152 ms | 2.17x |
| 2048 | uniform | 0.580608 ms | 0.216064 ms | 2.69x |
| 4096 | uniform | 0.906240 ms | 0.371712 ms | 2.44x |
| 8192 | uniform | 1.720320 ms | 0.675840 ms | 2.55x |
| 8192 | skewed | 1.713152 ms | 0.653312 ms | 2.62x |

The measured conservative crossover is 768 padded dispatched rows. Down uses
grouped Marlin for every measured size; gate/up should retain fused cuTile
below 768 and use faithful dual-Marlin above it. The retained pipeline reports
are
[`marlin-gate-up-pipeline-quick.json`](../benchmarks/2026-08-09-qwen-serving/post-tiled/marlin-gate-up-pipeline-quick.json),
[`marlin-gate-up-threshold.json`](../benchmarks/2026-08-09-qwen-serving/post-tiled/marlin-gate-up-threshold.json),
[`marlin-gate-up-crossover.json`](../benchmarks/2026-08-09-qwen-serving/post-tiled/marlin-gate-up-crossover.json),
and
[`marlin-gate-up-pipeline-production.json`](../benchmarks/2026-08-09-qwen-serving/post-tiled/marlin-gate-up-pipeline-production.json).

## Remaining scope boundary

These are steady-state synthetic uniform/skewed routing distributions. A trace
captured from this exact model and representative prompts is still required to
set and validate the production crossover policy. Cold-weight cache rotation
is also pending. The measurements establish the kernel-level choice but do not
yet claim end-to-end server parity.
