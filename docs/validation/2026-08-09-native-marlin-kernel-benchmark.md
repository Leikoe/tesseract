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

## Scope boundary and next gate

The NVFP4 rows above compare dense GEMMs at the routed expert projection
dimensions; they do not yet model a 256-expert routing histogram or invoke the
grouped Marlin MoE kernel. They directly apply to the shared expert and prove
the dequant/MMA pipeline gap. Production routed-MoE selection requires the same
native extraction, correctness oracle, and routing-distribution benchmark for
Marlin MoE. Cold-weight cache rotation is also pending. Neither omission
invalidates the dense result, but both prevent claiming final server parity.
