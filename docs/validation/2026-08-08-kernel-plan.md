# Typed CUDA kernel plan validation

Revision: `9b109446583adc41b2dc72f9498092d48fd26227`.

The initial `KernelCatalog` now resolves the complete BF16 dense-decoder leaf
set before device allocation and graph capture. Each semantic operation has a
stable name and revision plus its exact compile-time geometry. Prefill/mixed
and decode attention are separate plan entries even though both currently
resolve to the same ragged cuTile kernel. Runtime bucket selection also comes
from the immutable plan.

The plan rejects unsupported compute capability, malformed head/rotary
geometry, and tile divisibility constraints with typed errors. A Proptest
property checks that query/context dispatch covers all positive logical sizes,
rounds upward, remains a power of two, and is bounded by the next bucket.

Validation on the A100 comprised:

- `cargo check --features cuda --all-targets`;
- focused CUDA tests, including all three `kernel_plan` tests;
- `scripts/node/verify-a100.sh`, covering 39 host tests, strict ordinary and
  CUDA Clippy, release model checks, BF16 device loading, a cuTile smoke test,
  the actual Llama forward, and the upstream cuTile example;
- the actual forward predicted token 12366, `" Paris"`;
- the retained benchmark reached 1,670.95 first-shape and 1,719.79 warm-shape
  completion tokens/s at concurrency 8.

The plan introduces no hot-path discovery, registry lookup, trait object, or
string matching. Its diagnostic summary is emitted once during construction.
