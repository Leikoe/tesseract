# Tesseract Batching Architecture

Status: implementation contract, updated 2026-08-08.

This document defines the production batching boundary between the engine and
model backends. The upstream analysis behind these choices is recorded in
`inference-engine-architecture-reference.md`.

## Design choice

Tesseract schedules token progress like vLLM and retains phase information like
SGLang:

- The scheduler advances each request from `position` by `num_tokens` under one
  global token budget. Prompt chunks and decode rows can coexist in a batch.
- `WorkPhase` remains explicit so the backend can select a phase-specialized
  execution path without inferring phase from incidental token counts.
- The backend flattens all scheduled query tokens into one ragged model batch.
  It does not execute one model forward per request.
- Pure greedy decode uses fixed-shape CUDA graphs. Prefill, mixed batches, and
  stochastic decode use the ragged eager path until equivalent graph-safe
  buckets are implemented.

This separates scheduling policy from model layout. The engine does not know
about tokenization, attention heads, RoPE, or model-specific tensors. The model
backend does not decide admission, fairness, or physical KV ownership.

## Scheduler-to-backend contract

`ScheduledBatch` is immutable after validation and contains one contiguous
`ScheduledWork` range per request. Construction guarantees:

1. Every work item has at least one query token.
2. Every query token has exactly one newly allocated physical KV slot.
3. A request and physical KV slot appear at most once in a batch.
4. Decode work contains exactly one token and samples that token.
5. Position and aggregate token arithmetic cannot overflow.
6. `query_start_offsets` is the prefix sum of work-item token counts, begins at
   zero, and ends at `num_tokens`.
7. Request `i` owns flattened query rows
   `query_start_offsets[i]..query_start_offsets[i + 1]`.

The scheduler may construct an empty batch to indicate that no model execution
is ready, but it never passes that batch to `Backend::step`.

These are enforced contracts, but not Rust contract attributes. Tesseract uses
stable Rust 1.89; `core::contracts` is nightly-only, and a procedural contract
macro would turn malformed runtime work into assertion panics. Fallible
constructors return typed errors at trust boundaries, private fields prevent
unchecked construction, and property tests exercise cross-step invariants.
`debug_assert!` is reserved for redundant checks of internal programmer-only
assumptions.

The scheduler owns KV reservation and slot allocation. The backend validates
that `position` equals its committed request history before executing. Logical
KV history is committed only after the GPU forward succeeds. Failed requests
release their reservations, so writes made by a failed forward are disposable
and cannot become visible to another live request before release.

## Flattened model batch

For `T` scheduled query tokens and `R` requests, the Llama backend builds:

| Tensor or vector | Shape | Meaning |
| --- | --- | --- |
| `token_ids` | `[T]` | Input token for every flattened query row |
| `positions` | `[T]` | Absolute RoPE position per row |
| `current_slots` | `[T]` | Physical destination for the row's new K/V |
| `request_indices` | `[T]` | Selects the request/context table row |
| `context_lengths` | `[T]` | Causal key boundary for this query row |
| `context_slots` | `[request_bucket, context_bucket]` | Fixed-extent logical-to-physical KV map; only the first `R` rows are logical requests |
| `sample_rows` | `[S]` | Strictly increasing query rows requiring logits |

For a prompt chunk beginning at position `P`, its rows have context lengths
`P + 1, P + 2, ...`. This is what makes multiple prompt chunks and decode rows
correct inside the same forward: attention uses a per-query causal boundary,
not one sequence length for the entire batch.

Only `sample_rows` pass through the language-model head. Intermediate prompt
chunks populate KV without computing or copying unused vocabulary logits.

## KV attention

K/V storage is flat physical storage with a sentinel plus isolated padding
scratch slots:

```text
[physical_slot, kv_head, head_dim]
```

RoPE writes new K/V directly into `current_slots`. Ragged attention reads the
request's logical slot table and loads K/V directly from the flat cache. It
must not materialize a gathered `[request, context, head, dim]` cache per layer.
The sentinel is reserved for padded table entries and remains initialized.
Padded query rows write to distinct scratch slots beyond logical allocator
capacity. They therefore cannot race with each other, alias a live request, or
force `valid_rows` into a cuTile kernel signature.

The same attention operation is used by eager mixed/prefill execution and
captured decode graphs. Decode graphs provide static request indices
`0..batch_size`; their token IDs, positions, current slots, context tables, and
context lengths are updated before replay.

## Shapes, compilation, and workspaces

Exact query-row shapes must not be allowed to create unbounded cuTile frontend
specializations. The production shape policy is:

- Decode graphs use the full cross-product of power-of-two batch buckets and
  context buckets. With the default limits this is 6 batch buckets times 12
  context buckets, or 72 graphs; every graph is captured and replayed before
  readiness.
- Ragged query rows are padded to a configured, finite power-of-two token-bucket
  set. Eager request and context metadata use fixed extents derived from
  `max_running` and KV capacity, with logical counts and causal lengths carried
  separately. This prevents request-time cuTile specialization on combinations
  of prompt count and context length.
- All bucket shapes required by the serving configuration must compile before
  readiness, or readiness must report the unsupported shape policy.
- Captured graphs and reusable workspaces own stable tensor addresses.
- A bucket may execute fewer logical rows, but padding rows must use distinct
  scratch KV slots, a valid dummy context, and may never appear in
  `sample_rows`. Runtime checks reject query buckets larger than the scratch
  region configured at load time.

Model outputs are allocated uninitialized only when the immediately following
kernel, GEMM (`beta = 0`), or memcpy overwrites every element. KV caches,
sentinel data, partially written buffers, and readable padding must be
initialized. This rule is centralized in the audited `output_buffer` helper.

Eager kernels are enqueued asynchronously on the backend-owned stream. A
transformer-layer boundary drains the stream before layer-local tensors are
released, because cuTile uses a separate deallocator stream. Sampling performs
the final synchronization while copying token IDs or logits; prompt chunks that
do not sample synchronize explicitly before their KV slots are committed. An
error-path completion guard drains queued KV writes before the scheduler can
release and recycle those slots. Per-operation `sync_on` is forbidden in the
model hot path.

Readiness is a device-wide quiescence boundary after tokenizer warmup, eager
token-bucket warmup, and graph capture/replay. This drains both the compute
stream and cuTile's deallocator stream. Debug-level latency events report slow
CUDA enqueues, full model batches, and tokenization without logging prompt or
generated text.

The remaining workspace milestone is to retain eager intermediates by token
bucket rather than allocating them for every engine iteration. A workspace is
owned by the backend thread and cannot be reused until its stream-ordered work
has completed or a dependency proves that reuse is safe.

## Execution modes

| Scheduled work | Sampling | Execution |
| --- | --- | --- |
| Pure decode | All greedy and every row samples | Fixed-shape full-model CUDA graph |
| Pure decode | Any stochastic request | Ragged eager forward, batched logits |
| Pure/chunked prefill | Any | One ragged eager forward |
| Mixed prefill/decode | Any | One ragged eager forward |

There is no serial per-request fallback. If a ragged batch cannot execute, the
backend returns a batch error and the scheduler fails the affected requests;
silently decomposing it would hide a production performance regression.

## Required validation

Completion requires evidence at every boundary:

- Unit and property tests prove batch validation, prefix offsets, token budgets,
  phase priority, fairness, KV non-aliasing, progress, output cardinality, and
  cancellation/release behavior. Generated scheduler state machines must shrink
  failures into reproducible minimal cases.
- CUDA correctness compares ragged and graph outputs with the prior trusted
  path for prompt-only, decode-only, mixed, and stochastic batches.
- Compute Sanitizer reports no invalid access or race for direct flat-KV reads.
- Serving tests prove multiple prompt requests execute in one backend forward
  and mixed prompt/decode work stays in one forward.
- A cold-shape test proves readiness/warmup prevents request-time frontend JIT
  cliffs for every configured token bucket.
- Nsight Systems confirms gathered-KV kernels are absent, zero-fill kernels are
  limited to intentionally initialized state, and allocator/stream-sync counts
  no longer scale with transformer layer count.
- A100 benchmark artifacts record TTFT, inter-token latency, request latency,
  request/token throughput, graph counters, eager counters, and batch shape
  distributions for cold and warm runs.

The 2026-08-08 A100 validation passed the cold/warm shape-stability benchmark,
strict CUDA Clippy, independent BF16 logits comparison, and a final full-model
Compute Sanitizer run. The retained measurements and exact scope are recorded
in `validation/2026-08-08-a100-batching-redesign.md`.
