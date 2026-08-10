# A100 cuTile W4A16 versus Marlin

Date: 2026-08-09  
GPU: NVIDIA A100-SXM4-80GB (`sm_80`)  
Model geometry: `nvidia/Qwen3.6-35B-A3B-NVFP4`

## Contract

The comparison keeps the checkpoint representation intact:

- weights remain packed E2M1 FP4 (two values per byte),
- per-16-weight scales remain E4M3 FP8 bytes,
- activations and outputs are BF16,
- MMA accumulates in FP32,
- every timing candidate runs after the existing differential probes.

This is W4A16 emulation on Ampere. cuTile's native scaled FP4 MMA is an
`sm_100+` operation; on `sm_80`, a competitive kernel must explicitly decode
packed FP4/FP8 fragments to BF16 and feed BF16 tensor-core MMA. See
`references/cutile-rs/cutile-book/tutorials/11-nvfp4-inference.md`.

## Reference-derived schedule

The implementation and sweep follow four concrete reference findings:

1. cuTile's persistent GEMM maps a bounded logical output grid onto a fixed
   physical worker grid (`references/cutile-rs/cutile-examples/examples/persistent_gemm.rs`).
2. TileGym's A100 static-persistent GEMM search uses M/N tiles in `{64, 128}`,
   K=32, and occupancy 1 or 2
   (`references/TileGym/src/tilegym/ops/cutile/matmul.py`).
3. TileGym launches up to `SM count * occupancy` workers; an occupancy compiler
   hint alone does not create those CTAs
   (`references/TileGym/src/tilegym/ops/cutile/bmm.py`).
4. Marlin's selected NVFP4 path uses bit-level E2M1/E4M3 conversion and a
   256-thread CTA (`src/cuda/marlin/native/moe/marlin/dequant.h` and
   `src/cuda/marlin/native/moe/moe_wna16_marlin.cuh`).

The resulting built-in benchmark sweeps K32/N64 and K32/N128 with occupancy
1/2/4. Candidate names include the eight-worker-warp CTA contract.

## Changes validated

- N and K storage shapes are compile-time specializations, exposing full
  strides to Tile IR instead of `?` outer strides.
- The persistent physical grid is capped at `SMs * occupancy`, not merely
  `SMs`.
- Packed weights and FP8 scales are never expanded at model load.
- E2M1 decoding constructs exact BF16 bits using masks/shifts and one
  subnormal select instead of seven value selects.
- E4M3 decoding uses masks/shifts and two subnormal range selects instead of
  seven value selects.
- FP4 and FP8 bit-decoder differential probes report zero maximum error.

## Median kernel time

Uniform routing, three warmups, eleven measured iterations:

| Projection (K→N) | Rows | Best cuTile | Marlin | cuTile / Marlin |
|---|---:|---:|---:|---:|
| routed gate/up (2048→512) | 512 | 0.088 ms | 0.114 ms | 0.78× |
| routed gate/up (2048→512) | 8192 | 0.572 ms | 0.338 ms | 1.69× |
| routed down (512→2048) | 512 | 0.055 ms | 0.058 ms | 0.95× |
| routed down (512→2048) | 8192 | 0.503 ms | 0.360 ms | 1.40× |

The initial static-stride candidates were 0.934 ms for large gate/up and
2.109 ms for large down. Correct persistent occupancy and bit-level decoding
therefore improve those paths by approximately 1.63× and 4.19× respectively.

The authoritative final sweep is
`docs/benchmarks/2026-08-09-qwen-serving/post-tiled/cutile-bitdecode-v2-vs-marlin.json`.
Intermediate artifacts in the same directory preserve each schedule change.

## Rejected hypotheses

- N128 is not universally better. It helps the large 2048→512 geometry at
  occupancy 1, but regresses small batches and is inferior to N64 for down.
- Applying two-way occupancy and eight worker warps to the existing fused K16
  gate/up kernel regresses 8192 rows from 1.40 ms to 1.68 ms. The rejected
  result is preserved in `cutile-fused-grid-vs-marlin.json` and the code change
  was reverted.
- A hand-authored blocked weight layout failed the differential oracle and was
  fully removed. No layout repack should return without a permanent pack/load
  property test.
- Tile IR's generic packed `i4` `unpack` operation was numerically exact but
  did not reproduce Marlin's fragment behavior on `sm_80`. At 512 rows it was
  flat for gate/up (0.088 ms) and regressed the best down schedule to 0.061 ms.
  At 8192 rows its best gate/up and down medians were 0.721 ms and 0.591 ms,
  versus Marlin's 0.345 ms and 0.362 ms. The implementation was reverted; the
  raw results remain in `cutile-tile-ir-unpack-512-rejected.json` and
  `cutile-tile-ir-unpack-8192-rejected.json`.

## Status and next gate

cuTile now matches or beats Marlin at 512 routed rows, but it has not matched
Marlin for throughput-sized batches. Production therefore retains the existing
load-time-selected Marlin fallback above its row threshold.

The next production candidate should give cuTile and Marlin a shared,
property-tested prepared-NVFP4 artifact: checkpoint-native storage remains the
source of truth, while load-time preparation produces Marlin's exact blocked
weight and scale layout. The cuTile consumer must then preserve those packed
words through its fragment load instead of applying generic logical nibble
unpacking. Promotion requires numerical parity and no regression on both 512
and 8192 rows for uniform and skewed routing.
