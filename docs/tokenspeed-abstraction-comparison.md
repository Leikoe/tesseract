# TokenSpeed Compared with the Tesseract Engine Design

Date: 2026-08-08.

This review compares the proposed Tesseract design against the checked-out
TokenSpeed source at revision `27008e78184ca0403e597cb9cc103c476544ba66`.
It treats TokenSpeed as three related systems: its Python serving runtime, its
C++ scheduler/cache library, and `tokenspeed-kernel`.

Source paths below are relative to `references/tokenspeed/` unless they already
include that prefix.

## Verdict

TokenSpeed is the strongest confirmation so far that Tesseract is choosing the
right large boundaries. It independently separates scheduling from execution,
uses explicit scheduler plans and completion events, makes attention a stateful
backend, describes heterogeneous cache groups, composes common model structure,
and selects implementations by operation family and capability.

It also shows where Tesseract can be cleaner:

- distinguish a lifecycle-bearing layer backend from a fine-grained kernel
  implementation selected beneath it;
- resolve kernel choices into an immutable execution plan during construction
  instead of consulting a stringly typed global registry in the hot path;
- make mixed prefill/decode a first-class batch kind;
- model MoE as a planned route/dispatch/experts/combine pipeline whose phases
  may be fused, not one opaque kernel call;
- retain private validated Rust batches and typed identifiers rather than
  parallel public vectors, string group IDs, and `**kwargs`;
- use model-based property tests with shrinking for scheduler/cache state rather
  than only fixed-seed randomized scenarios.

The conclusion is not to reproduce TokenSpeed's plugin system. Its useful idea
is the layered contract and selection model. Tesseract can preserve that source
extensibility with linked Rust factories and static hot-path composition.

## Current implementation versus target design

The comparison above is primarily validation of the target architecture, not a
claim that the current Rust implementation has reached it. The distinction is
important:

| Concern | Current Tesseract | Target confirmed by TokenSpeed |
| --- | --- | --- |
| Scheduled work | private, validated `ForwardBatch` with typed positions/KV slots and an explicit mixed partition | retain this boundary and extend it with request-slot generations, output selection, and grouped state views |
| Engine/executor protocol | `ModelExecutor::{submit,poll}` with typed completion tickets and one in-flight synchronous CUDA submission | enable event-backed overlap, multiple tickets, and epoch-fenced reclamation without changing the boundary |
| Executor request state | the engine owns prompt/generated/decoder/sampling state; the executor consumes batch-local materializations and owns physical KV | retain engine authority while allowing explicit versioned, non-authoritative device mirrors when they are performance-justified |
| Physical state | one flat `KvSlot` domain | a schema of attention/recurrent/cache groups with typed arena identity |
| Model program | Llama architecture, batching, graphs, sampling, and CUDA lifecycle remain combined in a 2,770-line file | small private architecture adapter constructing a shared decoder program |
| Operation implementations | direct concrete kernels | stateful `AttentionBackend`, planned `MoeBackend`, and construction-time leaf-kernel plans |

The typed mixed batch, engine-owned request record, and ticketed executor are
the first three realized slices of the design. They remove semantic request
authority from the Llama executor and make completion-gated reclamation
explicit. Device execution is still synchronous and one-at-a-time, so overlap
and model isolation remain later steps.

## The three levels must remain distinct

TokenSpeed reveals three different substitution boundaries:

| Level | TokenSpeed | Tesseract choice |
| --- | --- | --- |
| Engine execution | large `ModelExecutor` orchestration object | `ModelExecutor` boundary around a concrete executor |
| Stateful operation family | `AttentionBackend` owns metadata, cache integration, and graph state | sealed `AttentionBackend`; analogous planned `MoeBackend` |
| Leaf kernel | `KernelSpec` plus `select_kernel` for attention, GEMM, MoE, norm, sampling, and other operations | construction-time typed kernel catalog producing concrete plans |

`tokenspeed-kernel` explicitly presents the third layer as public operation APIs
over a registry keyed by family, mode, format, features, platform, and traits
(`references/tokenspeed/tokenspeed-kernel/README.md:34` and
`tokenspeed-kernel/python/tokenspeed_kernel/selection.py:464`). Registered specs
also carry priority bands and tags
(`tokenspeed-kernel/python/tokenspeed_kernel/registry.py:75` and `:156`).

By contrast, TokenSpeed's runtime `AttentionBackend` is not just a callable. It
owns cache-pool attachment, eager metadata, capture buffers, replay updates,
prefill/decode dispatch, and PD cache-step coordination
(`python/tokenspeed/runtime/layers/attention/backends/base.py:61`). This is the
same demonstrated boundary represented by Tesseract's `AttentionBackend`.

These levels should not collapse into one universal `Kernel` trait. Attention
metadata and graph lifetime are coherent at the backend level, while a GEMM or
RMSNorm implementation is ordinarily a leaf choice inside an already-built
layer. Conversely, choosing a callable leaf kernel does not satisfy the
lifecycle contract of an attention backend.

## Kernel selection: adopt the catalog, reject hot-path discovery

TokenSpeed makes adding a leaf implementation local: register one callable with
format signatures, platform requirements, features, traits, and priority. The
selector filters candidates, consults an optional family oracle, ranks them, and
caches the winner (`tokenspeed-kernel/python/tokenspeed_kernel/selection.py:258`
and `:464`). It also supplies standalone numerical references, benchmarking,
shape capture, and overrides (`tokenspeed-kernel/README.md:113`). Those are all
useful ideas.

The Python implementation has costs that Rust need not inherit:

- identity and most capability dimensions are strings and dictionaries;
- missing trait declarations may be treated as unconstrained during selection
  (`selection.py:295`);
- environment and process-global overrides participate in selection
  (`selection.py:501`);
- selection is lazy on first call, so construction does not by itself prove that
  every required operation has a compatible implementation.

Tesseract should use a typed `KernelCatalog` only during executor construction.
The architecture and selected backends submit closed `KernelRequirement` values;
the builder resolves every requirement or fails before allocating/capturing the
executor. The result is an immutable `KernelPlan` stored by the concrete program.
A plan need not mean one kernel for every runtime shape: it may contain a
validated bucket table or typed decision tree over dynamic batch geometry. What
must be absent from the transformer loop is global discovery, capability
filtering, string comparison, or virtual dispatch.

An implementation descriptor should still record a stable name and revision so
profiles and failures identify the chosen code. Explicit development overrides
are useful, but they should alter the build request and be reported in the final
plan rather than invisibly modifying a global selector.

TokenSpeed's out-of-tree plugin mechanism is narrower than a general runtime
component system. Discovery is an explicit startup action over the
`tokenspeed_kernel.plugins` Python entry-point group, and each package merely
calls a registration function
(`tokenspeed-kernel/python/tokenspeed_kernel/plugins/__init__.py:21` and `:153`).
Tesseract's linked-crate registration function is the source-level equivalent:
it can add descriptors to the cold catalog without loading code dynamically or
changing the hot-path ownership model.

## Models and layer composition

TokenSpeed's dense Llama implementation is 395 lines, versus the current
2,904-line `src/model/llama_3_2.rs`, because generic execution, cache, graph,
sampling, and common transformer mechanics live elsewhere. Llama defines its
MLP, projections/RoPE/attention computation, layer resolvers, and weight mapping
(`python/tokenspeed/runtime/models/llama.py:63`, `:115`, and `:290`). Shared
decoder and causal-LM bases provide the repeated structure
(`python/tokenspeed/runtime/models/base/decoder_layer.py:68` and
`models/base/causal_lm.py:43`).

This supports moving Tesseract's execution machinery out of the Llama file and
constructing a shared `DenseDecoder` program. It does not imply inheritance in
Rust. Validated concrete specs plus generic composition express the reuse more
directly.

TokenSpeed also has an opt-in layer compiler. A layer declares an ordered list
of modules with closed `ModuleKind`, `CallConvention`, placement, fusion,
aux-capture, and idle behavior
(`python/tokenspeed/runtime/models/base/module_spec.py:29`). The compiler inserts
communication and produces executable steps
(`models/base/execution.py:40` and `models/base/decoder_layer.py:275`).

Tesseract should not introduce a public universal `Layer` trait or dynamic graph
IR for dense single-GPU v1. It should, however, preserve room for a sealed
construction-time `LayerPlan` when distributed placement and communication
fusion arrive. That plan should be a closed enum over semantic operations, not
arbitrary callbacks, and compile to concrete steps before serving begins.

## Attention

TokenSpeed confirms all parts of the proposed `AttentionBackend` contract:

- eager metadata initialization (`base.py:126`);
- separate graph-state initialization, capture, and replay (`base.py:135-193`);
- explicit decode versus extend execution (`base.py:236-313`);
- backend-owned cache-pool and runtime configuration (`base.py:90-98` and
  `:196`);
- distinct cache group/table capabilities (`base.py:64-80`);
- architecture-aware construction and validation in a named registry
  (`python/tokenspeed/runtime/layers/attention/registry.py:170`).

Tesseract's trait is cleaner because associated metadata and graph-state types
make backend state explicit and static. One amendment is necessary: a backend
must be able to expose separate prepared paths for prefill, decode, and mixed
execution, potentially using different leaf kernel plans. This need not mean
three trait objects; it can be associated plan data inside one concrete backend.

TokenSpeed also keeps a lightweight per-layer `PagedAttention` object while the
selected backend owns batch metadata and graph state. The Rust contract should
make the same split explicit with an associated `LayerState`: construction
validates layer-specific cache/scaling information once, and every enqueue takes
that typed state. Otherwise layer identifiers and scale/layout requirements are
likely to return as untyped fields inside `AttentionOperation`.

## MoE

TokenSpeed treats MoE selection as a whole plan. `MoELayer` builds a plan from
quantization, activation, routing mode, all-to-all backend, topology, geometry,
and operator choice, then creates weights in the layout required by the winning
plan (`python/tokenspeed/runtime/layers/moe/expert.py:221`). Capabilities include
whether the backend owns routing and whether finalization may be deferred
(`expert.py:261-273`). Execution can receive precomputed top-k values or raw
router logits and can expose an overlap callback inside dispatch
(`expert.py:287`).

This shows that Tesseract's current `MoeBackend::plan`/`enqueue` sketch is too
opaque. Its prepared layer should expose a closed pipeline plan with semantic
phases:

```text
route -> dispatch -> experts -> combine -> finalize
```

A backend may fuse adjacent or all phases, but it must declare whether it owns
routing, what expert-weight layout it requires, which collective topology it
uses, whether finalization is deferred, and what overlap/capture guarantees it
provides. The scheduler still sees none of these details.

Dense feed-forward does not yet have comparable lifecycle evidence. Keep
`FeedForwardBackend` narrow and construction-time, or represent it as a concrete
`DenseMlp<KernelPlan>` until multiple dense implementations require shared
workspace/graph hooks. Attention and MoE are the clearly justified boundaries.

## Batch shape and executor mirrors

TokenSpeed's scheduler builds `PrefillOperation` and `DecodeOperation` values,
then stable-partitions them into one `ForwardBatch` containing the prefill rows
before decode rows (`tokenspeed-scheduler/csrc/scheduler/operations/forward.h:34`).
Its runtime has explicit `EXTEND`, `DECODE`, `MIXED`, and `IDLE` modes
(`python/tokenspeed/runtime/execution/forward_batch_info.py:32`).

Tesseract therefore needs `ForwardKind::Mixed`; treating a mixed batch as merely
an implementation detail would lose its ordering and output-selection contract.
The current representation proves this with disjoint prefill and decode ranges
over one stable-partitioned sequence array. Separate owning sub-batches are not
necessary.

TokenSpeed also keeps device-resident per-request state such as future input
tokens and valid cache lengths, then updates it after a forward
(`python/tokenspeed/runtime/execution/model_executor.py:870`). This further
supports Tesseract's versioned non-authoritative `RequestMirror`. TokenSpeed's
large mutable `ForwardContext`, which accumulates optional feature-specific
fields (`python/tokenspeed/runtime/execution/context.py:39`), is the outcome the
private typed Tesseract batch and `RuntimePolicy` are intended to avoid.

## Scheduler, completion, and cache ownership

The C++ scheduler has a clean external protocol: submit requests, obtain an
`ExecutionPlan`, and advance it with an `ExecutionEvent`
(`tokenspeed-scheduler/csrc/scheduler/scheduler.h:49`). Plans contain a sequence
of closed operations; events contain a sequence of closed completion variants
(`scheduler/execution_plan.h:33` and `scheduler/execution_event.h:29`). Request
states are a `std::variant`, and transitions consume a state to produce the next
state (`scheduler/request.h:41`). This strongly supports Tesseract's plan/ticket/
completion split and closed request-state enum.

TokenSpeed permits one additional overlapped schedule step and tracks pending
forward results (`scheduler/scheduler.cpp:84` and `scheduler/scheduler.h:147`).
Tesseract's ticket IDs, epochs, and deferred reclamation generalize that protocol
without fixing the pipeline depth at one.

Cache ownership is also close to the proposed design. TokenSpeed has one
coordinator over heterogeneous cache groups and a shared physical block pool
(`tokenspeed-scheduler/csrc/cache/kv_cache_coordinator.h:47`). `CacheBlockRef` is
an explicit pool-scoped shared owner that releases the physical block when its
last reference dies (`cache/cache_block_ref.h:50` and `:96`). This validates the
revised invariant of shared immutable prefixes and an exclusive writable tail.

Rust can encode more of the remaining contract: typed group IDs, `ArenaId`,
generational request slots, distinct shared-prefix and writable-tail lease types,
and commit/abort APIs that surface errors rather than relying on destructor
cleanup or unchecked partial commit.

## Contracts and property testing

TokenSpeed validates many boundaries eagerly, uses closed C++ variants for
request state, and has extensive scheduler/cache scenario tests. Its strongest
property-like test executes 200 fixed-seed randomized cache/eviction sequences
and verifies that a jointly matched prefix is recoverable by every cache group
(`tokenspeed-scheduler/tests/cpp/test_joint_match_invariants.cpp:91`).

That test is good evidence for Tesseract's proposed reference-model testing, but
it is not a shrinking property-testing system. Tesseract should use `proptest`
for command sequences over a small reference engine:

- admit, prefill, decode, cancel, finish, share prefix, evict, and complete;
- compare optimized and reference request/cache state after every command;
- assert one writer per writable location and exact shared-reference counts;
- assert no reclamation before the last relevant completion epoch;
- assert batch lowering preserves row ownership and mode/output cardinality;
- assert mixed batches are equivalent to their ordered prefill and decode
  components where the backend declares that equivalence;
- assert every advertised backend capability has eager/reference parity and, if
  advertised, graph parity.

Ordinary Rust types, private constructors, and explicit validation remain the
primary contract mechanism. Contract macros may document local preconditions,
but they cannot replace ownership types, a scheduler reference model, or backend
conformance suites.

## Concrete amendments to the Tesseract design

1. Keep `AttentionBackend` sealed and statically composed.
2. Give `AttentionBackend` typed per-layer state in addition to eager metadata
   and graph state, and require explicit prefill/decode/mixed capability.
3. Add a construction-time typed kernel catalog that resolves an immutable
   shape-dispatch `KernelPlan`; do not add a universal hot-path `Kernel` trait.
4. Retain `ForwardKind::Mixed` with its validated prefill/decode row partition.
5. Expand the MoE contract into declared pipeline phases and fusion/capture/
   overlap capabilities.
6. Keep dense feed-forward concrete until a second lifecycle-bearing
   implementation proves the trait boundary.
7. Reserve a closed construction-time `LayerPlan` for future distributed
   placement and communication compilation; do not expose it to the engine.
8. Model scheduler output as plans and executor feedback as completion events,
   tied together by ticket IDs and reclamation epochs.
9. Add shrinking reference-model properties for scheduler/cache behavior and a
   reusable numerical/graph conformance suite for every backend implementation.

## What to copy and what not to copy

Copy the boundaries: stable-partitioned mixed batches, scheduler plans and
completion events, executor-resident request mirrors, stateful attention
backends, whole-pipeline MoE planning, heterogeneous cache groups, and standalone
kernel numerics/benchmarks.

Do not copy the representation accidents: parallel public vectors, string cache
group IDs, mutable optional-field contexts, signature introspection, plan
dictionaries, process-global overrides, or repeated registry resolution in an
operation call. Rust lets Tesseract make those contracts closed, typed, and
construction-validated without giving up the extensibility that motivated them.
