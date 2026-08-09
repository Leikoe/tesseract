# SM80 grouped NVFP4 kernel design

This note pins the upstream evidence used to design Tesseract's A100
ModelOpt-W4A16 MoE kernels. The references are local, immutable checkouts:

- `references/cutile-python` at `cc98c055622fbd98c4471f23c05cc4118873fa40`
- `references/TileGym` at `5902f99ee59b01ccc484554a3a6734f9e5607e59`
- `references/cutile-rs` at `9fe5756f861bc40f098e6981ac2dff6cf5d3d0e4`

## Non-negotiable arithmetic contract

The A100 path is W4A16, not W4A8. It loads BF16 activations and packed NVFP4
weights, decodes E2M1 nibbles and E4M3 group scales in the tile, casts the
scaled weights to BF16 for tensor-core MMA, accumulates in FP32, and casts to
BF16 only immediately before the final store. Packed weights must never be
expanded into a persistent BF16 weight matrix.

## Routing and memory contract

The NVIDIA cuTile MoE sample (`samples/MoE.py`) sorts token replicas by expert,
pads each expert segment to `TILE_M`, and records one expert id per row tile.
The kernel gathers activation rows through `sorted_token_ids` and scatters the
result back through the same mapping. Tesseract should follow that contract:
padding is logical scheduling metadata, not a reason to materialize a padded
activation copy. The first projection gathers original token rows; the second
reads expert-contiguous intermediate rows. The down projection applies routing
weights before the final reduction/scatter.

The production MoE leaf therefore needs a device-resident batch plan containing
at least:

- sorted token-replica ids, including a sentinel for padding;
- one expert id per row tile;
- the valid assignment count and padded row-tile count;
- routing weights and the destination token ids needed by the down projection.

No host loop may launch one GEMM per expert.

## Persistent scheduling

cuTile-Python's `samples/MatMul.py` caps the physical launch grid at the GPU SM
count and has each worker stride over the full logical output-tile space.
TileGym's `ops/cutile/group_gemm.py` applies the same scheme across multiple
GEMM problems. Its Unsloth grouped MoE kernel additionally derives expert-local
coordinates from device-resident group sizes, avoiding a GPU-to-CPU sync.

cuTile-RS exposes the safer equivalent: map a logical output partition onto a
bounded physical grid and iterate `MappedPartitionMut::iter_indices()`. The
committed cuTile-RS benchmark reports that this checked mapped schedule is
within 0.3% of its raw-pointer persistent implementation. Tesseract should use
that proof-carrying API rather than hand-written unchecked output pointers.

The persistent worker count is a launch-policy input obtained from device
properties and capped at the number of logical jobs. It is not a model constant.
The same packed artifact and route plan must also support a conventional grid so
the backend can select the measured winner for a workload class at construction
or warmup time.

## Tile policy and validation

The upstream examples do not establish one universal tile. NVIDIA's sample MoE
uses BF16 `128x128x64`; TileGym searches pre-SM90 dense/persistent candidates
across M/N tiles of 64 or 128, K tiles of 32, 64, or 128, and occupancy 1 or 2.
Decode MoE has much smaller, irregular expert-local M than those dense examples,
so copying `128x128x64` without measurement would waste most row work.

For the target A100 model, benchmark at minimum:

- small-M grouped gate/up and down projections at representative decode batch
  sizes and top-k routing distributions;
- prefill routing distributions separately;
- conventional versus persistent schedules;
- candidate N/K tiles and occupancy supported by cuTile on SM80;
- fused gate+up plus SiLU and routing-weight application versus unfused leaves.

Every candidate must first pass differential tests against a host/reference
dequantization, including all FP4 codes, non-unit E4M3 and global scales,
negative values, ragged expert counts, padding sentinels, and repeated token
destinations. CUDA sanitizer and multi-request batch tests remain release gates.
