# Tesseract Abstractions Compared with vLLM and SGLang

Date: 2026-08-08.

This review compares `engine-abstraction-design.md` against the checked-in
upstream sources, not documentation or remembered architecture.

| Project | Local revision |
| --- | --- |
| vLLM | `f7ef489e93cf92b8d6ce7403b49f1db867bcc35e` |
| SGLang | `55f02e68875a27eef0efedfc9d5845e066b184c0` |

## Verdict

Tesseract's strongest proposed improvements over both projects are:

- private validated host batches rather than mutable bags of optional tensors;
- semantic integer newtypes and typed errors;
- explicit transactional cache ownership and reclamation;
- one set of model operations for eager and graph execution;
- property-tested scheduling and allocation state machines;
- only two engine-visible runtime traits.

The original proposal was nevertheless too specialized for synchronous,
single-stage causal-LM decoding. Source review required six material revisions:

1. permit versioned, non-authoritative executor request mirrors;
2. replace synchronous execute/commit with in-flight tickets and epochs;
3. replace logits-only `CausalLm` with a stage-aware `ModelProgram`;
4. admit a sealed, statically dispatched attention-backend boundary;
5. represent grouped/shared/recurrent state rather than one flat KV layout;
6. separate architecture selection from checkpoint format and transport.

## Side-by-side boundaries

| Concern | vLLM | SGLang | Revised Tesseract choice |
| --- | --- | --- | --- |
| Authoritative request state | Scheduler `Request` | Scheduler `Req` | Engine request table |
| Device request state | Persistent runner mirror | Scheduler/runner objects and pools | Versioned non-authoritative mirror |
| Scheduled work | `SchedulerOutput` deltas | Mutable `ScheduleBatch` | Private validated `ForwardBatch` |
| Local execution | `GPUModelRunner` | `ModelRunner` | `CudaExecutor<P,R,A,S>` |
| Distributed execution | `Executor` implementations | Worker/process topology | `ModelExecutor` implementation |
| Model boundary | model `forward` + capability protocols | model `forward` + convention | sealed `ModelProgram` |
| Runtime model policy | V2 `ModelState` | fields/hooks in runner and batch | sealed `RuntimePolicy` |
| Attention | backend + metadata builder | graph-aware backend | sealed static `AttentionBackend` |
| Sampling | runner-owned sampler | runner-owned sampler | executor-owned sampler |
| Cache | cache groups/coordinator/block pool | radix cache + pool allocators | `StateSchema`, coordinator/arena pair |
| Graphs | model runner/graph manager | model runner/graph runners | executor-owned graph cache |
| Model registry | architecture registry | `EntryClass` registry | `ArchitectureFactory` registry |
| Weight loading | independent format loaders | independent format loaders | independent `WeightSource` |

## Request authority and executor mirrors

vLLM's scheduler owns the authoritative request objects
(`references/vllm/vllm/v1/core/sched/scheduler.py:440`). Its scheduler output
sends full state for new requests and deltas for cached requests
(`references/vllm/vllm/v1/core/sched/output.py:193`). The newer GPU runner keeps
persistent token, progress, and sampled-token buffers in `RequestState`
(`references/vllm/vllm/v1/worker/gpu/states.py:27`) and creates ephemeral
`InputBatch` views over stable buffers
(`references/vllm/vllm/v1/worker/gpu/input_batch.py:12`).

SGLang's `Req` owns input/output tokens, sampling parameters, decoding state,
and cache progress (`references/sglang/python/sglang/srt/managers/schedule_batch.py:786`).
It also contains execution-facing pool indices and model-specific state. The
large `ScheduleBatch` and `ForwardBatch` structures then mix scheduler,
allocator, host, device, and feature metadata
(`references/sglang/python/sglang/srt/managers/schedule_batch.py:1965` and
`references/sglang/python/sglang/srt/model_executor/forward_batch_info.py:412`).

The evidence rejects the original absolute rule that an executor must hold no
request state. Restaging complete histories, block tables, penalties, and RNG
state on every token is unnecessary overhead. Tesseract instead distinguishes:

```text
engine state:   authoritative semantics and scheduling decisions
executor state: versioned materialization for efficient device execution
```

Stable `RequestSlot { index, generation }` handles and rejected stale deltas
make that distinction enforceable.

## Batch and output contracts

vLLM's newer `InputBatch` explicitly represents padded versus logical extents,
request mappings, multiple logits per request, speculative expansion, and
structured-output presence
(`references/vllm/vllm/v1/worker/gpu/input_batch.py:36`). vLLM outputs
variable-length token lists because speculative acceptance counts differ
(`references/vllm/vllm/v1/outputs.py:261`).

SGLang has explicit forward modes for extend, decode, mixed, target verify,
draft, and split prefill
(`references/sglang/python/sglang/srt/model_executor/forward_batch_info.py:98`).
Its broad mutable `ForwardBatch` demonstrates what happens when new modes are
added as unrelated optional fields.

Tesseract should keep a small validated common batch, then use closed
`ForwardKind` and `OutputSelection` enums with typed payloads. This supports
prefill with no output, ordinary generation, speculative variable output,
prompt logprobs, pooling, encoder work, and pipeline completion without a giant
optional-field structure.

## Model program and output phases

vLLM separates model body and output head. Llama returns hidden states and
exposes `compute_logits` separately
(`references/vllm/vllm/model_executor/models/llama.py:517` and `:529`). The
runner selects required rows before the LM head
(`references/vllm/vllm/v1/worker/gpu_model_runner.py:4569`). This enables
pipeline intermediates, prompt logprobs, pooling, and auxiliary hidden-state
consumers.

SGLang calls the same model forward for eager execution and decode graph capture
(`references/sglang/python/sglang/srt/model_executor/runner/eager_runner.py:222`
and `runner/decode_cuda_graph_runner.py:1023`). Its prefill graph can capture the
transformer body while leaving the vocabulary-sized logits tail eager
(`runner/prefill_cuda_graph_runner.py:510`).

The correct rule is therefore:

> Eager and graph paths use the same model operations, while graph policy may
> capture a declared subregion.

`ModelProgram` consequently exposes stage-aware inputs and outputs rather than
one opaque `enqueue_logits` call.

## Attention backend

Both projects prove that attention is a real substitution boundary, not merely
a model configuration field.

vLLM's `AttentionBackend` defines cache shape and stride requirements, supported
data/head/block geometries, and implementation/metadata builders
(`references/vllm/vllm/v1/attention/backend.py:56`). Its metadata builder
reports graph support and batch-reordering constraints at `:623`.

SGLang's backend separates eager, outside-graph, inside-graph, capture, and
replay metadata preparation, then provides decode/extend execution
(`references/sglang/python/sglang/srt/layers/attention/base_attn_backend.py:21`).
The runner can select different prefill and decode backends
(`references/sglang/python/sglang/srt/model_executor/model_runner.py:920`).

Tesseract should not create a dynamic attention object for every layer. It
should use one sealed startup-selected backend strategy, statically dispatched
through `DenseDecoder<A>`, with associated metadata and graph-state types.

## Cache/state ownership

vLLM describes per-layer cache specs and groups for full attention, sliding
window, MLA, Mamba, encoder-only, and cross-attention
(`references/vllm/vllm/v1/kv_cache_interface.py:94`). Cache blocks are
reference-counted and may be shared by multiple running requests
(`references/vllm/vllm/v1/core/kv_cache_utils.py:117`).

SGLang couples scheduling directly to request-to-token and token-to-KV
allocators (`references/sglang/python/sglang/srt/managers/schedule_batch.py:1971`)
and supports distinct MHA, MLA, and Mamba pool schemas
(`references/sglang/python/sglang/srt/mem_cache/unified_memory_pool.py:71`).

This revealed two flaws in the first Tesseract draft:

- a bare `KvSlot` did not prove which device arena it belonged to;
- “no slot has two owners” incorrectly prohibited prefix sharing.

The revised design creates an engine-owned coordinator and executor-owned arena
as an `ArenaId`-matched pair. `StateSchema` describes groups and addressing.
Leases distinguish shared immutable prefixes from exclusive writable tails.

## Asynchrony and disaggregation

vLLM advances scheduled progress optimistically and fences reclamation by
completed scheduling epochs
(`references/vllm/vllm/v1/core/sched/scheduler.py:1326` and `:1693`). Its KV
manager also supports delayed publication and transferred-token accounting
(`references/vllm/vllm/v1/core/kv_cache_manager.py:345`).

SGLang's disaggregation connector exposes explicit sender/receiver lifecycle
states and polling
(`references/sglang/python/sglang/srt/disaggregation/base/conn.py:89`).

A synchronous `execute -> commit` transaction is safe but eventually prevents
multiple batches in flight. Tesseract uses `BatchTicket`s containing completion
fences, provisional state, optimistic deltas, rollback information, and a
reclamation epoch. A `StateStore` trait is added only when local and remote
state policies actually become independently selectable.

## Loading

Both projects separate architecture selection from checkpoint format:

- vLLM architecture registry:
  `references/vllm/vllm/model_executor/models/registry.py:1060`
- vLLM format loader selection:
  `references/vllm/vllm/model_executor/model_loader/__init__.py:50`
- SGLang architecture registry:
  `references/sglang/python/sglang/srt/models/registry.py:19`
- SGLang loader base and selection:
  `references/sglang/python/sglang/srt/model_loader/loader.py:342` and `:4104`

Tesseract follows this proven split with `ArchitectureFactory` and
`WeightSource`, avoiding an architecture-by-storage-format matrix.

## Sampling and operational invariants

Both systems keep sampling outside model files. vLLM selects model rows, computes
logits, and invokes a runner-owned sampler. SGLang's sampler consumes logits and
batched sampling metadata
(`references/sglang/python/sglang/srt/layers/sampler.py:70`). Tesseract keeps
semantic sampling state in the engine and device penalty/RNG buffers in the
versioned executor mirror.

Tesseract's property-testing proposal is stronger than either inspected codebase.
SGLang does add useful live invariant checks for capacity conservation,
committed-versus-allocated progress, free-list uniqueness, and use-after-free
(`references/sglang/python/sglang/srt/managers/scheduler_components/invariant_checker.py:64`
and `:300`). Tesseract should use both property state machines and cheap runtime
probes.

## Economical final abstraction set

Engine-visible hot-path boundaries:

1. `ModelFrontend`
2. `ModelExecutor`
3. `StateStore`, only once independently selectable remote/offloaded state exists

Sealed, statically dispatched executor boundaries:

1. `ModelProgram`
2. `RuntimePolicy`
3. `AttentionBackend`
4. sampler strategy

Cold-path boundaries:

1. `ArchitectureFactory`
2. `WeightSource`
3. executor administration for adapter load/unload/pin/reset

This keeps the scheduler independent of architecture and CUDA details while
representing the production variation that vLLM and SGLang have demonstrated.

## Registries and external plugins

vLLM has an actual package plugin loader based on Python entry points. Its
declared groups are `vllm.general_plugins`, `vllm.io_processor_plugins`,
`vllm.platform_plugins`, `vllm.stat_logger_plugins`, and
`vllm.endpoint_plugins`
(`references/vllm/vllm/plugins/__init__.py:16`). Logits processors use an
additional `vllm.logits_processors` group
(`references/vllm/vllm/v1/sample/logits_processor/__init__.py:48`). General
plugins execute their registration functions in every process at
`references/vllm/vllm/plugins/__init__.py:77`; the project's own general
plugins register filesystem and Hugging Face LoRA resolvers
(`references/vllm/pyproject.toml:46`). Endpoint plugins are more strictly
allowlisted because they add network routes (`plugins/__init__.py:93`).

SGLang primarily uses explicit Python registries rather than one universal
package-entry-point layer:

- attention factories:
  `references/sglang/python/sglang/srt/layers/attention/attention_registry.py:31`
- sampler factories:
  `references/sglang/python/sglang/srt/layers/sampler.py:527`
- prefix-cache factories:
  `references/sglang/python/sglang/srt/mem_cache/registry.py:49`
- model entry classes:
  `references/sglang/python/sglang/srt/models/registry.py:19`

The important upstream idea is the combination of a behavior contract with a
named factory and capability validation. The Python import mechanism itself is
not appropriate to copy into Rust. Tesseract will use an explicit typed registry
and build-time-linked plugin crates first. Runtime native plugins remain deferred
until a versioned C ABI, GPU resource-ownership contract, and allowlist exist.
