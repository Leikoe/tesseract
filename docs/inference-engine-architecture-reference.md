# Inference Engine Architecture Reference

## vLLM, SGLang, and SGLang's 2026 Unified KV Memory Work

Last updated: 2026-08-07

This document records the source analysis and online research performed for
Tesseract on 2026-08-07. It is intended to preserve the architectural mental
models, implementation entry points, terminology, and design implications that
would otherwise be easy to lose as the upstream projects evolve.

## Scope and revisions

The local source analysis was pinned to:

| Project | Revision | Description |
| --- | --- | --- |
| vLLM | `f7ef489e93cf92b8d6ce7403b49f1db867bcc35e` | `v0.26.1rc0-504-gf7ef489e9` |
| SGLang | `55f02e68875a27eef0efedfc9d5845e066b184c0` | SGLang `main` shallow checkout |

The online SGLang research covers first-party material available through
2026-08-07, especially the June/July 2026 page-major and unified-memory work.
Any statement about current feature support should be rechecked against the
upstream repository before implementation.

## Executive summary

Both vLLM and SGLang implement continuous batching, paged KV storage, prefix
reuse, distributed execution, CUDA graphs, speculative decoding, constrained
generation, multimodal input, and LoRA. The major difference is where each
system puts its semantic center of gravity.

- **vLLM V1 is request-progress-centric.** Prefill, decode, prefix hits,
  speculative tokens, and chunking are represented as progress from a
  request's computed-token count toward its current token target.
- **SGLang is cache- and phase-centric.** It explicitly constructs
  prefill/extend and decode batches, while a radix tree participates in prefix
  discovery, request ordering, KV ownership, protection, and eviction.
- **vLLM's strongest abstraction boundary is frontend -> EngineCore ->
  executor/worker.**
- **SGLang's online serving path is a process pipeline:** tokenizer/frontend ->
  scheduler ranks -> detokenizer.
- **SGLang's 2026 unified-memory work does not replace RadixAttention.** It
  replaces separately sized physical pools beneath the radix index with one
  byte buffer dynamically shared by heterogeneous state types.

A compact mental model is:

```text
vLLM:   schedule request token progress, then assign paged blocks
SGLang: discover reusable prefixes, then build prefill/decode work around them
```

## Side-by-side comparison

| Dimension | vLLM V1 | SGLang |
| --- | --- | --- |
| Core scheduling model | Advance `num_computed_tokens` toward a token target | Explicit prefill/extend and decode transitions |
| Prefix representation | Chained hashes over full KV blocks | Compressed radix tree over token sequences |
| Physical cache ownership | Block-pool reference counts and LRU-like free/eviction queue | Page allocator plus radix-node lock references |
| Cache-aware ordering | Policy plus prefix-block lookup | Longest-prefix and DFS-weight policies are first-class |
| Serving process shape | Frontend -> EngineCore -> executor/workers | Tokenizer -> scheduler ranks -> detokenizer |
| Detokenization | Frontend `OutputProcessor` | Dedicated detokenizer process |
| CPU/GPU overlap | Async EngineCore, nonblocking execution, PP batch queue | Explicit overlap scheduler and stream WAR fences |
| Data parallelism | Independent EngineCore/scheduler/KV replicas | DP controller dispatching to independent TP x PP groups |
| Prefill/decode disaggregation | KV connector framework | Scheduler-level prefill/decode state machines |
| Extensibility style | Broad abstract interfaces and plugin registries | Many specialized cache, scheduler, model, and backend paths |

## vLLM V1

### Architectural shape

```text
HTTP/OpenAI or Python LLM
    |
    v
Renderer / tokenizer / multimodal processing
    |
    v
InputProcessor -> EngineCoreRequest
    |
    v
EngineCoreClient -- in-process or ZMQ --> EngineCore
                                             |
                                  Scheduler + KVCacheManager
                                             |
                                             v
                              Executor: uni / mp / Ray / external
                                             |
                                             v
                                  Worker / GPUModelRunner
                                             |
                                             v
                                      logits / sampler
                                             |
    <----------- OutputProcessor / detokenization / streams
```

Responsibilities are divided into three principal layers:

1. The frontend owns user-facing API contracts, prompt rendering,
   tokenization, multimodal preprocessing, per-request output queues,
   incremental detokenization, and stop-string handling.
2. `EngineCore` owns admission, request state, continuous scheduling, KV-cache
   allocation, structured-output state, and model execution coordination.
3. Executors and workers own process topology, distributed control, model
   weights, device buffers, kernels, and sampling.

Important source entry points:

- `references/vllm/vllm/v1/engine/async_llm.py`
- `references/vllm/vllm/v1/engine/core_client.py`
- `references/vllm/vllm/v1/engine/core.py`
- `references/vllm/vllm/v1/core/sched/scheduler.py`
- `references/vllm/vllm/v1/core/kv_cache_manager.py`
- `references/vllm/vllm/v1/core/block_pool.py`
- `references/vllm/vllm/v1/worker/gpu/model_runner.py`

### Online request lifecycle

1. `AsyncLLM` renders or tokenizes the input. Raw tokenization and multimodal
   processing run asynchronously so they do not block the HTTP event loop.
2. The frontend registers an output collector and converts the input into an
   `EngineCoreRequest`. Requests with `n > 1` are expanded into child requests.
3. `EngineCoreClient` sends the request in-process or through ZeroMQ to an
   EngineCore process. Online serving normally uses the asynchronous
   multiprocessing path.
4. EngineCore adds the request to the scheduler, schedules work, dispatches a
   model execution, obtains grammar masks, processes concurrent aborts, and
   updates scheduler state from the result.
5. The GPU runner mirrors request and block-table deltas, packs active requests
   into a flattened ragged batch, executes the model, and samples or verifies
   tokens.
6. The scheduler commits accepted tokens, rolls back rejected speculative
   positions, checks core-side termination, and emits compact core output.
7. The frontend incrementally detokenizes, applies stop strings, constructs the
   public response, and feeds the request's async collector.

The normal inner loop is effectively:

```text
schedule -> execute model -> sample/verify -> update scheduler -> emit output
```

Pipeline parallelism may keep several such batches in a bounded FIFO. Filling
the pipeline is prioritized over waiting for the oldest result, which reduces
pipeline bubbles but requires strict association between each future and the
scheduler state that produced it.

### Scheduler model

vLLM deliberately avoids a fundamental prefill/decode split. Each request has:

```text
num_computed_tokens -> num_tokens_with_spec
```

The target includes prompt, output, and speculative tokens. Each iteration
assigns work that advances the computed count toward the target. The same model
covers:

- ordinary prefill;
- chunked prefill;
- one-token decode;
- local or remote prefix hits;
- speculative verification;
- mixed prompt/decode batches.

The scheduler considers running requests before waiting requests and applies a
shared token budget. Per-request work is further constrained by context length,
prefill chunk thresholds, encoder budgets, structured-grammar readiness,
remote-KV state, available KV pages, speculative lookahead, and distributed
synchronization.

Allocation failure causes recomputation-based preemption: a victim's KV and
encoder state are released, its computed progress is reset, and it returns to
the waiting queue. This preserves liveness under uncertain output lengths but
increases latency and duplicate work for victims.

Important invariants:

- Computed-token counts can include in-flight work.
- Deferred block freeing prevents asynchronous GPU writes from targeting
  recycled blocks.
- Partial prompt chunks populate KV but do not emit user-visible output.
- Rejected speculative tokens roll optimistic counters back.
- Pipeline batches remain FIFO-associated with their scheduler outputs.

### KV cache and prefix reuse

vLLM combines paged physical storage with block-hash prefix caching:

- `KVCacheManager` handles per-request lookup and allocation.
- `KVCacheCoordinator` reconciles cache groups for full attention, sliding
  windows, Mamba/SSM, cross-attention, and hybrid models.
- `BlockPool` owns fixed physical blocks, reference counts, the free/eviction
  queue, and the hash-to-block index.
- Per-request block tables map logical context pages to physical blocks.

Prefix hashes are chained across full token blocks. A released hashed block can
remain in the free queue as a reusable eviction candidate; it loses its cached
identity only when allocation overwrites it.

The final prompt token is generally recomputed even after a complete cache hit,
because logits are required at the prompt boundary. Prefix reuse therefore
stops at or before `prompt_length - 1`.

For hybrid models, prefix reuse must be valid across every cache group. The
coordinator reconciles the deepest prefix simultaneously supported by each
group rather than allowing the groups to disagree about request progress.

### GPU and attention path

The GPU runner incrementally mirrors EngineCore state:

- removes completed or preempted requests;
- adds new requests;
- appends block IDs for existing requests;
- applies copy-on-write block copies;
- updates per-request scheduled-token counts;
- constructs dense request-slot metadata and flat token arrays.

Attention metadata contains cumulative query offsets, sequence lengths, block
tables, and per-token physical slot mappings. Attention first scatters new K/V
into paged storage and then reads the block tables through the selected backend.
Backends are chosen based on platform, dtype, head geometry, KV layout, MLA or
sliding-window requirements, and parallel configuration.

### Parallelism

- **TP** shards a model replica and adds intra-layer collectives.
- **PP** partitions layers and relies on concurrent batch queues to reduce
  bubbles.
- **DP** normally creates independent EngineCore, scheduler, worker, and KV
  worlds, then routes requests among them.
- **Expert parallelism** changes MoE weight ownership and communication.
- **Prefill context parallelism** expands prompt execution without simply
  duplicating the same KV layout.

TP and PP scale one request-processing replica. DP creates additional
independently scheduled replicas and therefore duplicates scheduler and cache
state.

### Compilation and advanced features

vLLM supports eager execution, standard `torch.compile`, and a custom compiler
pipeline. CUDA-graph modes include none, piecewise, full, and combinations such
as full decode graphs plus piecewise mixed/prefill graphs. More capture sizes
reduce padding and fallback, but cost startup time and graph memory.

Structured output applies grammar masks during sampling rather than validating
after generation. Speculative decoding reserves lookahead KV, verifies draft
tokens, and rolls rejected positions back. LoRA adapters are cached and applied
with adapter-aware kernels. Multimodal preprocessing stays primarily in the
frontend while encoder compute and cache capacity participate in scheduling.

## SGLang

### Architectural shape

```text
HTTP / OpenAI service
        |
        v
TokenizerManager
tokenization, multimodal processing, request state
        |
        | ZMQ PUSH
        v
Scheduler process(es), one per local TP x PP rank
prefix lookup, admission, KV ownership, GPU execution
        |
        | ZMQ
        v
DetokenizerManager
incremental text reconstruction
        |
        v
TokenizerManager -> HTTP/SSE response
```

The process split keeps tokenizer and detokenizer CPU work outside the GPU
scheduler. Each TP rank has scheduler/model-runner state, while a designated
ingress rank consumes frontend traffic and collectives keep other ranks in
lockstep.

Important source entry points:

- `references/sglang/python/sglang/srt/entrypoints/engine.py`
- `references/sglang/python/sglang/srt/managers/tokenizer_manager.py`
- `references/sglang/python/sglang/srt/managers/scheduler.py`
- `references/sglang/python/sglang/srt/managers/schedule_policy.py`
- `references/sglang/python/sglang/srt/managers/detokenizer_manager.py`
- `references/sglang/python/sglang/srt/mem_cache/radix_cache.py`
- `references/sglang/python/sglang/srt/mem_cache/memory_pool.py`
- `references/sglang/python/sglang/srt/model_executor/model_runner.py`

### Request lifecycle

1. `TokenizerManager` normalizes and validates request arguments, resolves LoRA
   identity, tokenizes text or accepts token IDs/embeddings, performs multimodal
   processing, installs request-ID state, and dispatches the request.
2. The scheduler receives requests into a waiting queue, performs prefix lookup
   and scheduling, allocates request and KV slots, and executes a prefill/extend
   or decode batch.
3. Scheduler output tokens go to a dedicated `DetokenizerManager`, which retains
   incremental per-request decoding state.
4. The detokenizer returns text/token deltas to `TokenizerManager`, which
   delivers them to the appropriate HTTP or SSE waiter.

Incremental detokenization uses bounded read/surrogate windows. An incomplete
Unicode suffix is not committed until later tokens make the sequence printable,
avoiding corrupted byte-pair boundaries.

### Scheduler model

SGLang is continuously batched but preserves explicit forward modes. Persistent
state includes a waiting queue, a running decode batch, previous/current forward
batches, and optionally a parked chunked-prefill request.

Each scheduler iteration approximately:

1. processes arrivals and previous results;
2. caches a completed prefill chunk if necessary;
3. merges prefill-complete requests into the running decode batch;
4. tries to construct and admit a new prefill batch;
5. otherwise updates and runs the decode batch;
6. optionally mixes decodes into an extend batch.

`PrefillAdder` accounts for physical page capacity, estimated future decode
growth, the maximum prefill-token budget, chunk size, request count, current
running batch, backend tile geometry, and priority/preemption thresholds.

Under pressure, requests may be retracted into the waiting queue. The scheduler
adapts its estimate of future output memory after retraction and relaxes it when
capacity remains healthy.

The overlap scheduler prepares the next batch and processes the previous result
while the GPU runs. Shared request/KV metadata is protected with precise GPU
read-completion events when available and whole-stream waits otherwise.

### Radix cache and KV ownership

SGLang uses two levels of indirection plus a radix index:

```text
ReqToTokenPool[request slot, position]
        |
        v
physical KV slot/page
        ^
        |
RadixCache[token sequence + namespace] -> physical slot slices
```

`ReqToTokenPool` maps request positions to physical KV indices. A separate
allocator owns free physical slots. `RadixCache` is a compressed token-prefix
tree whose values are copied slices of physical indices, not KV tensors.

Keys include an optional namespace (`extra_key`). It prevents unsafe sharing
between requests whose token IDs match but whose KV differs because of LoRA,
cache salt, or other context.

On lookup, the tree finds the longest prefix and may split a compressed node at
an exact match boundary. On insertion, it canonicalizes the physical indices,
frees duplicates, rewrites the request table to the canonical mapping, and
moves lock ownership to the new terminal node.

Ancestor lock references distinguish protected active paths from evictable
leaves. The radix tree is therefore simultaneously:

- a prefix index;
- an ownership graph;
- an eviction structure;
- a cache-locality input to scheduling.

Cache-aware scheduling policies include longest-prefix match and DFS weighting.
They can trade strict arrival ordering for reduced prefill work and improved
locality.

Correctness depends on token identity uniquely determining KV inside a
namespace. Direct embeddings and positional embedding overrides can invalidate
that property and therefore force cache misses or disable radix caching.

### Model runner, attention, and graphs

`ModelRunner` owns model weights, distributed groups, samplers, physical pools,
attention backends, CUDA-graph runners, LoRA state, and optional draft models.

Attention backends distinguish decode, extend/prefill, mixed, and idle paths.
SGLang can choose different implementations for prefill and decode and wrap
them in a hybrid backend. Dynamic metadata preparation is kept outside captured
graphs; graph-recordable updates use fixed-shape buffers.

CUDA graphs depend on stable addresses, captured batch-size buckets, and a
reserved padding request slot. Attention metadata, LoRA mappings, speculative
state, and KV write locations must all honor that address-stability contract.
Unsupported modes and shapes fall back to compiled or eager execution.

### Parallelism and disaggregation

SGLang supports TP, PP, DP, attention context/data parallelism, and MoE
expert/data parallelism. Ordinary DP places independent TP x PP groups behind a
`DataParallelController`. Some attention/MoE decompositions require explicit
idle or synchronization batches so all ranks enter collectives consistently.

Prefill/decode disaggregation is represented as scheduler state, not merely as
a proxy boundary:

- A prefill worker establishes destination metadata, computes the prompt,
  transfers KV, and reports completion.
- A decode worker preallocates destination slots, polls transfer progress,
  constructs a prebuilt extend batch that skips prompt computation, and merges
  the request into continuous decode.

This permits independent TTFT and decode-throughput scaling, at the cost of
bootstrap, topology, transfer-capacity, page-layout, and distributed-failure
constraints.

## SGLang's page-major and unified-memory work

### Terminology

The phrase **flat KV cache** is an understandable informal description, but the
upstream names are more precise:

| Term | Meaning |
| --- | --- |
| Page-major layout | A physical page's K/V or state across layers is one contiguous envelope |
| `UnifiedKVPool` / unified memory | One physical byte buffer dynamically shared by two heterogeneous sub-pools |
| `UnifiedRadixCache` | One logical radix controller coordinating typed FULL/SWA/MAMBA cache components |
| Flat transfer layout | A separate PD-transfer detail: one flat buffer per layer or state component |

Unified memory does **not** remove the radix tree or paging. It changes the
physical allocator beneath them.

### Timeline

| Date | Change |
| --- | --- |
| 2026-06-29 | SGLang merged PR #29533, adding opt-in page-major KV/state layout |
| 2026-07-01 | SGLang merged PR #29678, adding the opt-in unified memory pool |
| 2026-07-27 | The Kimi K3 launch post publicly explained the design and motivation |
| 2026-08-03 | PR #33362 opened to add PD-disaggregation support for the Kimi-linear unified pool |

### The original problem

Hybrid models can maintain state types with radically different allocation
units. Kimi K3 is the clearest example:

- MLA/full-attention state is append-only and allocated per token.
- KDA recurrent state is mutable and allocated as a large fixed-size block per
  active request.

Separate fixed pools encode a startup-time workload guess:

```text
[ fixed attention-KV pool ][ fixed recurrent-state pool ]
```

Concurrency-heavy workloads can exhaust recurrent-state slots while attention
KV remains unused. Long-context workloads can exhaust token KV while the state
pool has slack.

### Page-major layout

Historically, a page was scattered across separate per-layer tensors:

```text
layer 0: every page
layer 1: every page
layer 2: every page
```

The page-major layout flips the outer physical grouping:

```text
page 0: layer 0 K/V, layer 1 K/V, layer 2 K/V, ...
page 1: layer 0 K/V, layer 1 K/V, layer 2 K/V, ...
```

At page size one, this is a per-token, all-layer envelope. It enables whole-page
movement, transfer, offload, and compaction using contiguous byte ranges.
Unified memory implies this layout.

### Unified physical pool

The new allocator reserves one GPU `uint8` buffer:

```text
low addresses                                      high addresses
[ recurrent/SWA state --->       free gap       <--- full KV ]
```

Two `MultiEndedAllocator` instances grow from opposite ends. The free bytes in
the middle belong to neither side and can satisfy whichever state type needs
them next.

The public Kimi K3 explanation cited approximately 54 MB for one TP=8 KDA state
block across 69 layers and approximately 27 KB for one MLA token across 24
layers. A common fixed page size would be inappropriate for units differing by
roughly three orders of magnitude; sharing raw byte capacity avoids that
constraint.

### Virtual IDs and compaction

The layout is physically flat but logically indirect:

```text
radix nodes / request mappings
            |
            | stable virtual slot IDs
            v
virtual-to-physical tables
            |
            v
shared physical byte buffer
```

When freeing creates a hole, an end allocation can be moved into it. Only the
allocator's translation tables change; radix nodes and request mappings retain
their virtual IDs. This avoids rewriting every reference and keeps captured
graph input buffers stable.

Read and write locations are translated before graph replay into capture-stable
metadata buffers. The physical stores do not perform arbitrary translation
inside the graph hot path.

### Scheduler accounting

Admission can no longer ask only how many pages one independent pool has left.
It must reason about shared bytes and whether capacity is actually realizable by
a particular side.

For example, nominal full-KV evictable bytes cannot satisfy a new recurrent
slot if allocator geometry or non-drainable holes prevent forming that slot.
SGLang's prefill admission therefore accounts for:

- shared-gap bytes;
- peer-compaction holes;
- radix entries evictable by the requesting side;
- each sub-pool's entry size and growth direction.

### Current adoption status

As of 2026-08-07, unified memory is merged and documented but remains opt-in:

```text
--enable-unified-memory
```

It is intended for hybrid Mamba/GDN plus full-attention models and hybrid SWA
plus full-attention models. It is not a universal replacement for the ordinary
homogeneous Transformer KV allocator.

At the analyzed revision, important restrictions include:

- only two sub-pools;
- restricted attention, linear-attention, and Mamba backend combinations;
- page-major layout is mandatory;
- monolithic decode CUDA graphs only;
- no generic speculative-decoding integration;
- no decode context parallelism;
- no general hierarchical/host-tier cache integration;
- PD-disaggregation support still under active follow-up work.

The July feature should therefore be viewed as a memory-efficiency and
correctness foundation. The original PR explicitly did not claim a throughput
benchmark; its immediate benefit is eliminating stranded capacity and manual
pool-split tuning.

### Relationship to Kimi K3

Kimi K3 interleaves 69 KDA linear-attention layers with 24 MLA layers. KDA state
is overwritten in place, while MLA KV is append-only. This combination exposed
both problems:

1. recurrent state needed safe copy-on-write, snapshot, donation, sparse
   checkpointing, and replay semantics for radix reuse;
2. recurrent-state and token-KV capacity needed a runtime-dependent physical
   split.

The first problem is logical state management over the radix tree. The second
is the unified physical allocator. They complement one another but are not the
same feature.

## Design implications for Tesseract

### Preserve logical identity across physical movement

Stable logical or virtual cache IDs are valuable if physical pages may move,
be compacted, migrate across tiers, or change backing layout. Direct physical
addresses should not leak into long-lived scheduler or prefix-index state.

### Account in the scarcest realizable unit

A shared byte count is necessary but not sufficient. Admission should answer:

```text
Can the allocator realize the exact required objects without violating
alignment, page geometry, protected ownership, or in-flight operations?
```

This is especially important when heterogeneous state types have very
different allocation sizes.

### Separate cache identity, ownership, and storage

Three layers should remain conceptually distinct even if optimized together:

1. identity: which computation produced this reusable state;
2. ownership/protection: who may reference or evict it;
3. physical storage: where its bytes currently live.

SGLang's radix tree combines the first two closely, while the 2026 allocator
work increasingly decouples the third.

### Choose the scheduler's unifying abstraction carefully

vLLM demonstrates the benefits of reducing modes to token progress. SGLang
demonstrates the benefits of retaining explicit batch modes when specialized
cache and hardware paths differ materially. Tesseract should avoid accidental
mode proliferation while leaving room for optimizations whose invariants truly
differ between prefill, decode, and heterogeneous recurrent state.

### Make asynchronous ownership transitions explicit

Overlap, compaction, offload, and RDMA make `free` a synchronization operation,
not merely a free-list update. Blocks cannot move or be recycled until every
reader/writer that may reference the old physical location is fenced.

### Treat CUDA-graph compatibility as a data-layout contract

Captured graphs require stable addresses and shapes. Translation should happen
into stable metadata buffers before replay when possible. Features such as
LoRA, speculation, cache relocation, and padding need explicit graph-safe
contracts rather than ad hoc exceptions.

### Measure capacity benefits separately from throughput

Unified memory primarily improves usable capacity and reduces configuration
risk. Compaction and translation can add work even when they enable more
concurrency. Benchmarks should separate:

- maximum admitted requests or tokens;
- stranded memory;
- compaction bytes and frequency;
- scheduler overhead;
- latency/throughput below the old capacity limit;
- throughput after the old design begins queueing or OOMing.

## Source map

### vLLM local sources

- `references/vllm/vllm/v1/engine/async_llm.py`
- `references/vllm/vllm/v1/engine/core_client.py`
- `references/vllm/vllm/v1/engine/core.py`
- `references/vllm/vllm/v1/core/sched/scheduler.py`
- `references/vllm/vllm/v1/core/kv_cache_manager.py`
- `references/vllm/vllm/v1/core/block_pool.py`
- `references/vllm/vllm/v1/core/kv_cache_coordinator.py`
- `references/vllm/vllm/v1/worker/gpu/model_runner.py`
- `references/vllm/vllm/v1/attention/backends/flash_attn.py`
- `references/vllm/vllm/compilation/cuda_graph.py`

### SGLang local sources

- `references/sglang/python/sglang/srt/entrypoints/engine.py`
- `references/sglang/python/sglang/srt/managers/tokenizer_manager.py`
- `references/sglang/python/sglang/srt/managers/scheduler.py`
- `references/sglang/python/sglang/srt/managers/schedule_policy.py`
- `references/sglang/python/sglang/srt/managers/schedule_batch.py`
- `references/sglang/python/sglang/srt/managers/detokenizer_manager.py`
- `references/sglang/python/sglang/srt/mem_cache/radix_cache.py`
- `references/sglang/python/sglang/srt/mem_cache/memory_pool.py`
- `references/sglang/python/sglang/srt/mem_cache/layout/page_major.py`
- `references/sglang/python/sglang/srt/mem_cache/unified_memory_pool.py`
- `references/sglang/python/sglang/srt/mem_cache/multi_ended_allocator.py`
- `references/sglang/python/sglang/srt/mem_cache/kv_cache_configurator.py`
- `references/sglang/python/sglang/srt/model_executor/model_runner.py`
- `references/sglang/python/sglang/srt/disaggregation/prefill.py`
- `references/sglang/python/sglang/srt/disaggregation/decode.py`

### First-party online sources

- [SGLang and Miles Add Day-0 Support for Kimi K3](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support)
- [SGLang and Miles Add Day-0 Support for Inkling](https://www.lmsys.org/blog/2026-07-15-inkling-day0-support)
- [PR #29533: page-major KV/state layout](https://github.com/sgl-project/sglang/pull/29533)
- [PR #29678: unified memory pool for hybrid Mamba/SWA models](https://github.com/sgl-project/sglang/pull/29678)
- [PR #33362: PD support for unified memory](https://github.com/sgl-project/sglang/pull/33362)
- [SGLang server-argument documentation](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/server_arguments.mdx)

## Questions to revisit

- When does unified memory become enabled by default, if ever?
- Does support expand beyond exactly two heterogeneous sub-pools?
- Which PD, speculative-decoding, DCP, and HiCache restrictions have been
  removed since the pinned revision?
- What is the measured compaction frequency on real mixed workloads?
- How much extra scheduler and metadata cost does virtual-to-physical
  translation add below the capacity limit?
- Does SGLang publish the promised implementation-focused follow-up post?
- Can the same physical allocator serve remote or persistent cache tiers without
  weakening the stable virtual-ID abstraction?
