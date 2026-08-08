# Variable generation-output validation

Revision: `32fe2f3`.

`ExecutionOutput::Generation` now returns one `GeneratedTokens` group per
sampled request rather than hard-coding one token per request. This is the
executor contract needed by speculative acceptance, jump decoding, and future
composite executors while preserving current one-token CUDA execution.

The engine validates request-level cardinality before committing results:

- every sampled request has exactly one output group;
- duplicate and unexpected request IDs fail the batch;
- ordinary generation rejects an empty group instead of entering a no-progress
  scheduler stall; and
- tokens are consumed in order only until the first stop string, EOS token, or
  request length boundary. Any later candidates are discarded.

Local validation passed 48 tests and strict Clippy. New coverage includes
empty-result failure, length truncation, terminal-token truncation, and a
256-case shrinking property showing that arbitrary executor overproduction
never commits or reports more than `max_tokens`.

The complete `scripts/node/verify-a100.sh` suite passed on the A100-SXM4-80GB:
all 48 ordinary and CUDA-feature tests, strict ordinary and CUDA Clippy, model
and device-weight validation, cuTile smoke tests, the real Llama forward
producing token 12366 (`" Paris"`), and pinned upstream cuTile hello world.

This changes an engine protocol and scheduler commit path, not CUDA kernels,
launches, graph shapes, or the current one-token model output. The retained
throughput benchmark therefore remains applicable.
