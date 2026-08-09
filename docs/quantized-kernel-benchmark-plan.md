# Quantized kernel benchmark contract

This benchmark compares implementations of the same mathematical operation; it
is not an end-to-end serving benchmark and it does not replace serving TTFT.
The production checkpoint representation is authoritative. FP8 weights remain
FP8 bytes and NVFP4 weights remain packed FP4 for every timed implementation.
An expanded-BF16 implementation may be reported only as a separately labelled
compute ceiling and is never eligible for production backend selection.

## Comparability contract

Every row in a comparison fixes all of the following:

- GPU, clocks, CUDA version, build profile, commit, and stream;
- logical `M`, `N`, and `K`, expert count, and the expert-row histogram;
- identical BF16 activations, packed weight bytes, block scales, global scales,
  routing weights, padding sentinels, and output layout;
- BF16 tensor-core multiplication, FP32 accumulation, and BF16 output;
- warmup count, timed iterations, cache policy, and CUDA-event timing boundary.

The timing boundary begins immediately before the kernel sequence and ends
immediately after its final output kernel. It excludes checkpoint reads,
load-time repacking, allocation, JIT compilation, host-to-device copies, and
output downloads. Load time, repacked device bytes, and temporary workspace are
reported separately because they remain production costs.

Candidates first pass the existing differential oracle, extended to cover every
FP4 code, FP8 normal and subnormal values, both signs, non-unit scales, ragged
expert segments, padding, and repeated token destinations. A faster candidate
with different BF16 output is rejected rather than hidden behind throughput.

## Required implementations

The stable implementation identifiers are:

- `cutile_legacy_expanded_scales`: the current persistent cuTile byte-weight
  kernel, which expands E4M3 block scales to BF16 and is baseline-only;
- `cutile_storage_faithful`: a cuTile kernel retaining both weight and scale
  tensors in their manifest-declared byte representations;
- `cutile_repacked`: any load-time-repacked and staged cuTile successor;
- `marlin`: the SM80 Marlin implementation using its required load-time layout;
- `bf16_expanded_ceiling`: cuBLAS with expanded BF16 weights, diagnostic only.

Backend selection uses measured workload classes at construction or warmup. It
does not inspect formats or choose a kernel inside the token hot path.

`cutile_legacy_expanded_scales` is not eligible for production selection. It is
retained so changing scale representation cannot be mistaken for a kernel
speedup. A repacked scale layout is eligible only if its elements remain E4M3;
reordering storage is different from widening it to BF16.

## Shape matrix

Dense W8A16 covers the model's FP8 attention projections:

| Workload | M | K | N |
|---|---:|---:|---:|
| decode projection | 16, 32, 64 | 2048 | 512, 2048, 4096, 8192 |
| prefill projection | 512, 2048, 8192 | 2048 | 512, 2048, 4096, 8192 |
| GDN output | 512, 2048, 8192 | 4096 | 2048 |

Dense W4A16 covers shared experts and the LM head:

| Workload | M | K | N |
|---|---:|---:|---:|
| shared gate/up | 16, 512, 2048, 8192 | 2048 | 512 |
| shared down | 16, 512, 2048, 8192 | 512 | 2048 |
| LM head | 1, 16 | 2048 | 248320 |

Grouped W4A16 uses 256 experts and measures routed gate+up/SwiGLU and routed
down separately. Prefill cases use real router histograms retained from the
model benchmark when available, plus uniform and deliberately skewed synthetic
histograms. Decode cases cover 1, 2, 4, 8, 16, 32, and 64 requests at top-8.
Both 16-row and 64-row segment padding policies are measured rather than
assuming a universal tile.

## Reported measurements

For each candidate and shape, retain raw per-window CUDA-event samples and
report median, p10, p90, and minimum latency. Derived metrics include logical
TFLOP/s, packed-weight GB/s, total bytes read/written, padding amplification,
workspace bytes, repacked weight bytes, and load/repack time. Kernel launch
count is part of the result: fused gate/up/SwiGLU is compared to the complete
unfused sequence, not to one constituent launch.

Timed windows rotate enough independent activation/output buffers to exceed the
A100 L2 cache when measuring cold-weight behavior. A second hot-weight series
is retained because decode can reuse an expert's weights. Neither series is
silently substituted for the other.

The benchmark emits versioned JSON and a compact CSV summary. Raw artifacts and
the exact command line are committed under `docs/benchmarks/`; a backend may be
selected only from a retained result produced by the same correctness-passing
revision.
