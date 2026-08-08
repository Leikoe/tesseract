# Reusable CUDA batch-materialization validation

Revision: `d6ec8c3`.

The generic `CudaExecutor<P>` now owns one reusable `CudaBatch`. Lowering clears
and repopulates its typed host vectors rather than allocating a new collection
for every submission. Per-request context vectors are retained as a high-water
pool when request count shrinks, so their allocations are available again when
the batch grows.

Packed decode now consumes the request indices, context lengths, and sample
rows produced by generic lowering. It no longer reconstructs three equivalent
vectors inside `DenseDecoder` on every step. Architecture programs continue to
receive a fully lowered batch and do not recover scheduler semantics.

The CUDA-feature regression test lowers a larger batch, then a smaller decode
batch, then the larger batch again. It proves backing storage is reused and
that no token, position, slot, context, sampling, or row metadata survives from
the previous logical batch.

The complete `scripts/node/verify-a100.sh` suite passed on the A100-SXM4-80GB:
48 ordinary and CUDA-feature tests, strict ordinary and CUDA Clippy, checkpoint
and device-weight validation, cuTile smoke tests, the real Llama forward
producing token 12366 (`" Paris"`), and pinned upstream cuTile hello world.

Two retained serving-benchmark runs produced:

| Run | Batch-1 tok/s | Concurrent first tok/s | Concurrent warm tok/s |
| --- | ---: | ---: | ---: |
| First | 336.88 | 1,650.39 | 1,583.52 |
| Immediate repeat | 342.09 | 1,699.52 | 1,682.64 |

The concurrent results remain inside the node's previously observed range of
roughly 1,586–1,720 tok/s. They establish no regression, but the run variance is
larger than the likely benefit of these small host allocations, so this is not
claimed as a measured throughput improvement. The allocation-lifetime change
itself is established by the regression test. Compact benchmark summaries are
retained in `docs/benchmarks/2026-08-08-reusable-cuda-batch/`.
