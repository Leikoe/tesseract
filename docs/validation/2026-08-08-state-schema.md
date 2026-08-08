# State-schema and arena validation

Revision: `a4c9016`.

Logical KV allocation and physical executor storage now share an explicit,
immutable `StateSchema`. The schema has a process-unique `StateArenaId` and a
list of typed state groups; the first implemented group is flat KV.

The contract is enforced at two boundaries:

- engine startup rejects a requested logical KV capacity larger than the
  executor's physical flat-KV group; and
- every `ForwardBatch` carries the allocator's arena ID, which executors check
  before batch lowering or device execution.

This prevents an otherwise-valid numeric `KvSlot` from being interpreted in a
different executor's address space. The schema is deliberately data, not a new
runtime plugin interface. Attention continues to own its concrete storage and
layout.

Local validation passed 44 tests and strict Clippy. New coverage includes:

- distinct arena construction never aliases;
- a 128-case property test over valid flat-KV capacities;
- startup rejection of insufficient physical capacity;
- executor rejection of a batch from a foreign arena; and
- the existing shrinking scheduler state-machine properties, now running every
  generated allocation and batch through the arena capability.

The complete `scripts/node/verify-a100.sh` suite passed on the A100-SXM4-80GB:
all 44 ordinary and CUDA-feature tests, strict ordinary and CUDA Clippy, model
and 2,471,628,800-byte device-weight validation, cuTile smoke tests, the real
Llama forward producing token 12366 (`" Paris"`), and pinned upstream cuTile
hello world.

This is a correctness and construction-boundary change. It does not alter
kernels, launch shapes, CUDA graph replay, batching policy, or sampling, so the
retained throughput benchmark remains applicable.
