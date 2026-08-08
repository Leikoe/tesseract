# Tesseract Engine Abstraction Design

Status: proposed design, revised after source-level vLLM, SGLang, and TokenSpeed
review, 2026-08-08.

This document defines the boundary between request semantics, scheduling,
model architecture, and CUDA execution. The goal is not to maximize the number
of interfaces. It is to make ownership and invariants obvious, keep model
details contained, and preserve a zero-overhead hot path.

## Design rules

1. A trait must represent a real substitution boundary. Data variation uses
   structs and enums, not traits.
2. Dynamic dispatch is allowed at startup and once per engine batch. It is not
   used per token, transformer layer, tensor, or kernel launch.
3. The engine owns authoritative request progress, token history, sampling
   state, decoding state, and logical cache policy. Executors may retain
   versioned, non-authoritative device materializations of that state.
4. The executor owns device memory, workspaces, CUDA graphs, streams, kernels,
   and physical execution policy.
5. A deployed model artifact owns request presentation and tokenization. A
   model architecture owns configuration validation, tensor-name mapping, and
   construction of a shared executable program. These are independent axes.
6. Invalid states are excluded with private fields, newtypes, and fallible
   constructors. Assertions are not a substitute for validating a trust
   boundary.
7. Eager and graph execution use the same model operations. Graph policy may
   capture a validated subregion such as the transformer body without creating
   a second model implementation.

## Target dependency direction

```text
api -> ModelFrontend -> authoritative engine request state
                              |
                              v
                  scheduler + cache coordinator
                              |
                  BatchPlan + ExecutionTicket
                              |
                              v
                 ModelExecutor (local/distributed)
                              |
                    CudaExecutor<P, R, A, S>
                       /       |       \
             ModelProgram  RuntimePolicy  Sampler
                  |            |             |
          Attention/MoE backends + immutable KernelPlan
                              ^
                              |
       architecture factory + KernelCatalog + independent weight source
```

Dependencies only point downward. The scheduler cannot import CUDA or model
architecture modules. CUDA cannot import API request types. A model module may
construct shared programs, but no other module may inspect its private config
or weight types.

## Runtime boundaries

The local baseline engine should see only two runtime traits, with `StateStore`
added later if remote/offloaded state becomes independently selectable.
Executor-internal substitution points are sealed and statically dispatched.
Counting every internal strategy as an engine abstraction would repeat the
complexity of the upstream systems; pretending the internal variation does not
exist would bake Llama assumptions into the runner.

### 1. `ModelFrontend`

This is the CPU-side semantic boundary. It is called during admission and after
sampled tokens return, never inside the GPU layer loop.

```rust
pub trait ModelFrontend: Send + Sync + 'static {
    fn info(&self) -> &ModelInfo;

    fn prepare(
        &self,
        input: &RequestInput<'_>,
    ) -> Result<PreparedInput, PrepareError>;

    fn decoder(&self) -> Box<dyn TokenDecoder>;

    fn is_eos(&self, token: TokenId) -> bool;
}

pub trait TokenDecoder: Send {
    fn push(&mut self, token: TokenId) -> Result<&str, DecodeError>;
}
```

`PreparedInput` contains token IDs plus typed modality or encoder work when the
artifact requires it. `ModelFrontend` replaces the current ambiguous `Model`
name. It is constructed from the deployed tokenizer, processor, generation
configuration, and chat-template artifacts. It is not assumed to be uniquely
determined by the neural-network architecture.

### 2. `ModelExecutor`

This is the engine-to-execution boundary. An implementation may be a local CUDA
executor, a tensor/pipeline-parallel worker group, or a composite speculative
executor. It consumes validated work and returns mode-aware variable output.

```rust
pub trait ModelExecutor: 'static {
    fn model_info(&self) -> &ModelInfo;
    fn limits(&self) -> ExecutionLimits;

    fn submit(
        &mut self,
        batch: &ForwardBatch<'_>,
    ) -> Result<BatchTicket, ExecutionError>;

    fn poll(
        &mut self,
        completion: CompletionId,
    ) -> Result<Option<ExecutionOutput>, ExecutionError>;

    fn take_stats(&mut self) -> ExecutionStats;
    fn shutdown(&mut self) -> Result<(), ExecutionError>;
}
```

`ExecutionOutput` is a tagged result. Generation output contains zero or more
accepted tokens per request; other variants carry prompt logprobs, pooling
results, intermediate completion, or cache-transfer events. Cardinality is
validated against the batch's `OutputSelection`, not a universal one-token rule.

The outer engine remains generic over `E: ModelExecutor`, so ordinary serving
does not require a virtual call. A `Box<dyn ModelExecutor>` is used only by the
model registry and command-line construction path where heterogeneous model
selection is actually required.

The engine remains the semantic authority, but the executor may maintain stable
`RequestSlot`s and device-resident mirrors of tokens, positions, block tables,
sampling penalties, and RNG state. Every slot carries a generation/version;
admission installs a snapshot, later batches send deltas, and stale deltas are
rejected. These mirrors exist for performance and never decide scheduling or
user-visible progress.

### 3. Sealed `ModelProgram`

This is the boundary between generic CUDA lifecycle machinery and an executable
model program. It is called once for a whole batch. It does not expose layers to
the engine.

```rust
pub(crate) trait ModelProgram: 'static {
    fn execution_spec(&self) -> &ExecutionSpec;
    fn state_schema(&self) -> &StateSchema;

    fn enqueue(
        &self,
        context: &mut CudaExecutionContext<'_>,
        batch: &ProgramBatch<'_>,
        workspace: &mut ModelWorkspace,
        outputs: &mut ProgramOutputs,
    ) -> Result<(), ExecutionError>;
}
```

`ProgramBatch` and `ProgramOutputs` are closed enums for decoder hidden states,
selected logits, encoder work, pooling, requested auxiliary hidden states, and
pipeline intermediates. This is data variation, not public trait proliferation.
The generation pipeline is deliberately split into model body, output-row
selection, task head, and sampling so a prefill graph can capture the body
without capturing a vocabulary-sized tail.

`CudaExecutor<P, R, A, S>` is generic over a `ModelProgram`, a model-dependent
`RuntimePolicy`, an `AttentionBackend`, and a sampler. The default runtime policy
is zero-sized. Specialized policies handle genuinely model-dependent runtime
state such as multidimensional RoPE, recurrent-state movement, encoder inputs,
or auxiliary hidden-state capture without polluting the generic executor.

```rust
pub(crate) trait RuntimePolicy<P: ModelProgram>: 'static {
    type RequestState;
    type BatchState;

    fn install_request(
        &mut self,
        snapshot: &RequestSnapshot<'_>,
    ) -> Result<Self::RequestState, ExecutionError>;

    fn prepare_batch(
        &mut self,
        program: &P,
        batch: &DeviceBatch,
    ) -> Result<Self::BatchState, ExecutionError>;

    fn complete_batch(
        &mut self,
        state: Self::BatchState,
        output: &ProgramOutputs,
    ) -> Result<(), ExecutionError>;
}
```

### 4. Sealed `AttentionBackend`

Attention implementation is a demonstrated substitution boundary. Kernel
choice, metadata, cache layout, capture safety, and prefill/decode support vary
together in both upstream engines.

```rust
pub(crate) trait AttentionBackend: 'static {
    type LayerState;
    type Metadata;
    type GraphState;

    fn capabilities(&self) -> AttentionCapabilities;
    fn prepare_layer(
        &self,
        spec: &AttentionLayerSpec,
    ) -> Result<Self::LayerState, LoadError>;
    fn prepare(&mut self, batch: &DeviceBatch) -> Result<Self::Metadata, ExecutionError>;
    fn prepare_capture(&mut self, shape: ExecutionShape)
        -> Result<Self::GraphState, ExecutionError>;
    fn prepare_replay(
        &mut self,
        state: &mut Self::GraphState,
        batch: &DeviceBatch,
    ) -> Result<(), ExecutionError>;
    fn enqueue(
        &self,
        layer: &Self::LayerState,
        metadata: &Self::Metadata,
        operation: AttentionOperation<'_>,
    ) -> Result<(), ExecutionError>;
}
```

There is one startup-selected backend strategy per compatible execution path,
not a trait object stored in every layer. Each layer stores only the backend's
typed, construction-validated `LayerState`; batch metadata and graph state
remain backend/executor-owned. `DenseDecoder<A>` uses static dispatch. Prefill,
decode, and mixed execution may resolve to different prepared leaf plans inside
one backend, or to different concrete backends when their capabilities require
it.

There are no per-layer trait objects or universal `Layer`/`Kernel` interfaces.
Operation-family backends are statically composed by shared program types.

## Load-time boundary

Architecture selection and checkpoint transport are independent startup axes:

```rust
pub trait ArchitectureFactory: Send + Sync {
    fn probe(&self, manifest: &ModelManifest) -> Probe;
    fn instantiate(
        &self,
        config: &ModelConfig,
        target: &ExecutionTarget,
    ) -> Result<UnloadedProgram, LoadError>;
}

pub trait WeightSource {
    fn tensors(
        &mut self,
    ) -> Result<Box<dyn Iterator<Item = NamedTensor> + '_>, LoadError>;
}

pub struct LoadedModel {
    pub frontend: Arc<dyn ModelFrontend>,
    pub executor: Box<dyn ModelExecutor>,
}
```

The registry contains architecture factories, not model-name checks scattered
through the engine. SafeTensors, sharded, remote, quantized, and test-generated
weight sources feed the same named-tensor interface. A Llama factory parses its
private configuration, validates and maps tensors, and constructs a shared
`DenseDecoder` program. Only `ModelFrontend` and `ModelExecutor` reach the
engine.

Adapter identity participates in request and prefix-cache identity. Loading,
unloading, pinning, or resetting adapters is a cold control-plane operation,
not part of `submit`. An `ExecutorAdmin` command channel may expose those
operations without widening the hot-path executor trait.

Rust does not need inheritance between model families. Common architecture is
represented by concrete shared programs:

```text
Llama 3 / Mistral / Qwen2 ----> DenseDecoder
Mixtral / Qwen-MoE -----------> MoeDecoder
Mamba ------------------------> StateSpaceProgram
multimodal model -------------> EncoderDecoderProgram
```

Each program has a validated configuration and weight layout. Small variations
such as RMSNorm epsilon, GQA geometry, activation, RoPE variant, sliding-window
attention, tied embeddings, and bias presence are data in that program. A new
program type is warranted only when the computation or persistent state is
fundamentally different.

## Extensible layer implementations

The goal is source-level extensibility: adding another compiled implementation
of an operation family without changing model architecture or engine code. A
trait defines the semantic and lifecycle contract; a named factory makes the
implementation selectable during executor construction.

The boundary belongs around an operation family whose implementations differ in
more than one kernel call. Attention varies in metadata, cache addressing,
workspace, graph preparation, and kernels. MoE varies in routing, expert weight
layout, dispatch/combine, collective communication, workspace, and kernels.
Those deserve separate backend traits.

```rust
pub(crate) trait MoeBackend: 'static {
    type Layer;
    type BatchPlan;
    type Workspace;

    fn capabilities(&self) -> MoePipelineCapabilities;

    fn prepare_layer(
        &self,
        spec: &MoeSpec,
        weights: MoeWeights,
    ) -> Result<Self::Layer, LoadError>;

    fn plan(
        &self,
        layer: &Self::Layer,
        input: MoeBatchInput<'_>,
    ) -> Result<Self::BatchPlan, ExecutionError>;

    fn enqueue(
        &self,
        layer: &Self::Layer,
        plan: &Self::BatchPlan,
        hidden: HiddenStateViewMut<'_>,
        workspace: &mut Self::Workspace,
        context: &mut CudaExecutionContext<'_>,
    ) -> Result<(), ExecutionError>;
}
```

`MoePipelineCapabilities` declares whether the backend owns routing, its expert
weight layout, collective topology, graph legality, overlap support, and whether
finalization may be deferred. `BatchPlan` represents the semantic pipeline
`route -> dispatch -> experts -> combine -> finalize`; an implementation may
fuse any adjacent phases, including the whole pipeline. This keeps fusion an
implementation property without making the model or scheduler understand it.

Dense feed-forward is initially a concrete `DenseMlp<KernelPlan>`, not a peer
backend trait. Fused and unfused GEMM/activation choices are resolved in its
construction-time kernel plan. Extract a `FeedForwardBackend` only when a second
implementation demonstrates shared lifecycle, workspace, or graph behavior
beyond leaf-kernel selection.

The program types compose these statically:

```rust
pub(crate) struct DenseDecoder<A: AttentionBackend> {
    // model geometry and layers
}

pub(crate) struct MoeDecoder<A: AttentionBackend, E: MoeBackend> {
    // model geometry and layers
}
```

There is no trait object per layer. Every layer stores the concrete associated
`Layer` type, and calls are monomorphized. If expert-parallel dispatch later
varies independently of expert computation, `MoeBackend` may be composed with a
sealed `ExpertDispatcher`; it should not leak into the scheduler.

RMSNorm, activation, RoPE, and elementary GEMM calls remain ordinary typed
operations until a second implementation demonstrates a larger lifecycle or
storage contract. We should not create a trait for every kernel merely to make
the type graph look uniform.

Adding a lifecycle-bearing backend implementation requires four things:

1. implement the operation-family trait;
2. declare capabilities and reject incompatible geometry at construction;
3. register a factory that closes the concrete generic types;
4. pass the shared conformance suite: reference correctness, eager/graph parity,
   every supported bucket, deterministic behavior where promised, and Compute
   Sanitizer.

The typed construction registry is:

```rust
pub struct BackendDescriptor {
    pub name: BackendName,
    pub version: BackendVersion,
    pub capabilities: Capabilities,
}

pub trait ExecutorFactory: Send + Sync {
    fn descriptor(&self) -> &BackendDescriptor;

    fn build(
        &self,
        request: &BuildRequest,
    ) -> Result<Box<dyn ModelExecutor>, LoadError>;
}

pub struct RegistryBuilder {
    architectures: BTreeMap<ArchitectureName, Arc<dyn ArchitectureFactory>>,
    weight_sources: BTreeMap<FormatName, Arc<dyn WeightSourceFactory>>,
    executors: BTreeMap<BackendName, Arc<dyn ExecutorFactory>>,
    frontends: BTreeMap<ProcessorName, Arc<dyn FrontendFactory>>,
}
```

The registry selects and validates once. A factory constructs a fully concrete
`CudaExecutor<P, R, A, S>` behind the single outer `ModelExecutor` object, so
hot-path component dispatch remains static. Built-ins are registered explicitly
by the binary. A new implementation can live in another crate linked into the
binary and export a plain registration function; no runtime plugin mechanism is
required.

Factories register supported backend bundles rather than promising every
possible Cartesian product of implementations. A helper can close a new
`AttentionBackend` with the default dense kernel plan, sampler, and runtime
policy;
an MoE bundle can close the same attention implementation with a new
`MoeBackend`. This avoids existential associated types, per-layer dynamic
dispatch, and uncontrolled monomorphization while keeping additions local.

Layer backends and leaf kernels are separate levels. During construction, a
typed `KernelCatalog` resolves closed `KernelRequirement` values over operation,
mode, tensor formats, device capability, geometry, and required semantics. It
must resolve the complete set or fail before executor capture. The result is an
immutable `KernelPlan` stored by the concrete program or backend:

```rust
pub enum KernelRequirement {
    Gemm(GemmRequirement),
    Norm(NormRequirement),
    Activation(ActivationRequirement),
    AttentionLeaf(AttentionKernelRequirement),
    MoeLeaf(MoeKernelRequirement),
    Sampling(SamplingKernelRequirement),
}

pub struct KernelPlan {
    descriptor: KernelPlanDescriptor,
    // concrete, already-validated operation handles and shape dispatch tables
}
```

This catalog is a cold construction mechanism, not a universal hot-path
`Kernel` trait. A requirement may resolve to one implementation or to an
immutable bucket table/typed decision tree when the best kernel depends on
runtime batch geometry. There is no global registry lookup, string matching,
capability filtering, or dynamic dispatch in the transformer loop. Explicit
development overrides alter `BuildRequest` and are recorded in
`KernelPlanDescriptor`, so profiles and failures always report the exact
selected implementation.

The demonstrated internal strategy interfaces are:

| Strategy | Tesseract status | Why it varies |
| --- | --- | --- |
| `AttentionBackend` | define now | kernels, metadata, KV layout, graph legality |
| dense `KernelPlan` | define with decoder | fused/unfused GEMM and activation selection |
| `MoeBackend` | add with first MoE | routing, experts, dispatch/combine, collectives |
| sampler backend | concrete generic now | greedy/random kernels and RNG handling |
| logits processor | add with penalties/grammar | stateful batched logit transforms |
| `StateStore`/connector | add with remote state | prefix/offload/disaggregated lifecycle |
| quantization method | add after BF16 v1 | weight creation, packing, GEMM kernels |
| expert dispatcher | split from MoE only if needed | communication topology and overlap |
| platform backend | defer | CUDA is the only production platform in scope |

Runtime-loaded `.so` plugins are explicitly out of scope. This design concerns
compiled implementations selected at startup.

## Typed batch contract

The current batch path passes several unrelated `u32` and `usize` arrays. Use
newtypes at the host boundary so positions, rows, slots, and token IDs cannot be
accidentally exchanged:

```rust
pub struct TokenId(u32);
pub struct KvSlot(u32);
pub struct Position(u32);
pub struct QueryRow(u32);
pub struct SequenceIndex(u32);
pub struct RequestSlot { index: u32, generation: u32 }
pub struct ArenaId(u64);
```

`ForwardBatch` has private fields and is created only by `BatchLowerer`:

```rust
pub struct ForwardBatch<'a> {
    kind: ForwardKind<'a>,
    sequences: Vec<SequenceView<'a>>,
    updates: Vec<RequestDelta<'a>>,
    output: OutputSelection,
}

pub struct SequenceView<'a> {
    request_id: RequestId,
    request_slot: RequestSlot,
    state: StateView<'a>,
    query: Range<usize>,
}

pub enum ForwardKind<'a> {
    Prefill(TokenBatch<'a>),
    Decode(TokenBatch<'a>),
    Mixed(MixedBatch<'a>),
    TargetVerify(VerificationBatch<'a>),
    Draft(DraftBatch<'a>),
    Encode(EncoderBatch<'a>),
    Pool(PoolingBatch<'a>),
}

pub struct MixedBatch<'a> {
    prefill: TokenBatch<'a>,
    decode: TokenBatch<'a>,
}

pub enum OutputSelection {
    None,
    Generate(Vec<GenerationSelection>),
    PromptLogprobs(Vec<LogprobSelection>),
    Pool(Vec<PoolingSelection>),
}
```

Construction proves the invariants already documented in
`batching-architecture.md`: aligned token metadata, non-aliasing destination
slots, increasing selected rows, valid sequence indices, causal context bounds,
mode-specific output cardinality, and the prefill-before-decode row partition of
mixed batches. CUDA code receives a valid value rather
than revalidating parallel vectors on every execution branch. Closed mode and
output enums avoid the giant mutable structure of unrelated optional tensors
seen in mature upstream runners.

`DeviceBatch` is not a public contract. It is an executor-owned set of stable
device buffers populated from `ForwardBatch`. Graph padding and bucket extents
exist only in `DeviceBatch`; logical engine batches never contain fake requests
or scratch slots.

## Cache and state topology

A singular flat `KvLayout` is not general enough for sliding-window attention,
MLA, cross-attention, recurrent state, or hybrid models. The executable program
declares a validated schema:

```rust
pub struct StateSchema {
    groups: Vec<StateGroupSpec>,
}

pub enum StateGroupKind {
    Attention { addressing: KvAddressing },
    Recurrent,
    CrossAttention,
    Encoder,
}

pub enum KvAddressing {
    FlatSlots,
    PagedBlocks { block_size: NonZeroU32 },
}
```

The engine-owned `StateCoordinator` owns allocation and prefix-cache policy.
The executor owns the corresponding physical `StateArena`. They are constructed
as a matched pair and share an `ArenaId`; leases from one arena cannot be sent
to another executor.

Prefix sharing distinguishes immutable shared prefix references from
exclusively writable tail reservations. Copy-on-write is used for partial tails.
The correct invariant is not “one live owner per slot,” but “one writer per
writable location, and every shared location's reference count equals its live
references.” Kernel-specific block tables or flat slot mappings are executor
materializations, not scheduler contracts.

When local-only and remote/disaggregated state become independently selectable,
a fourth engine-facing `StateStore` trait is justified. Its state machine must
represent `PendingLoad`, `Resident`, `PendingStore`, and `Failed`, and surface
transfer completions to scheduling. It does not belong in `ModelProgram`.

## Transactional and asynchronous execution

KV allocation and request progress must be committed atomically around model
execution:

```text
reserve request capacity
    -> create BatchPlan + provisional state leases
    -> lower to ForwardBatch
    -> submit and receive BatchTicket(completion fence, epoch, deltas)
    -> allow other independent batches in flight
    -> complete in dependency order
    -> validate mode-specific output
    -> commit or correct progress and leases
    -> append accepted tokens and decode text
```

`BatchTicket` carries provisional allocations, optimistic progress deltas,
completion fences, rollback data, and a reclamation epoch. Multiple tickets may
coexist. Physical state cannot be reclaimed before every relevant fence has
completed, including after cancellation or speculative rejection.

Commit and abort are explicit fallible operations. `Drop` is only a leak-safe
fallback because it cannot safely wait for CUDA or report synchronization
failure. If completion cannot be established, the executor is poisoned and the
affected state is quarantined rather than recycled.

This removes the authoritative `slots`, `prompt`, and `generated` collections
now hidden in the Llama backend while permitting compact versioned mirrors in
the executor. It prevents partial scheduler commits without serializing all GPU
work behind synchronous execute-then-commit.

## Generic CUDA executor

`CudaExecutor<P, R, A, S>` owns every mechanism that should be identical across
models using the selected program, runtime policy, attention backend, and
sampler:

- host-to-device batch lowering;
- token, request, and context bucketing;
- reusable eager workspaces;
- CUDA graph capture, update, replay, and warmup;
- physical state arenas, sentinel/scratch regions, and address materialization;
- stream and completion-fence discipline;
- sampled-row gathering, logits processing, and sampling;
- execution counters and latency instrumentation;
- readiness and shutdown synchronization.

Its important internal types are concrete, not traits:

```rust
pub(crate) struct BucketPolicy { /* finite configured shapes */ }
pub(crate) struct WorkspacePool { /* keyed by ExecutionShape */ }
pub(crate) struct GraphCache { /* keyed by ExecutionShape */ }
pub(crate) struct StateArena { /* storage described by StateSchema */ }
pub(crate) struct RequestMirror { /* versioned derived device state */ }

pub(crate) struct ExecutionShape {
    kind: ForwardKindTag,
    query_bucket: usize,
    request_bucket: usize,
    context_bucket: usize,
    tokens_per_request: TokenShape,
    output: OutputMode,
    attention_variant: AttentionVariant,
    adapter_count: usize,
    distributed_padding: DistributedPadding,
    capture: CaptureVariant,
}
```

Graph and eager selection belongs here because it is execution policy, not
Llama semantics. The graph cache owns capture/replay, while `ModelProgram`,
`RuntimePolicy`, and `AttentionBackend` expose only the hooks required to make
their operations and metadata capture-safe.

## Sampling

Sampling is model-neutral and belongs beside the executor, not in a model file.
`SamplingParams` is validated data. A concrete `CudaSampler` can support greedy,
temperature, top-p, and later penalties or constrained decoding. CPU sampling
may exist as a test oracle, but it is not a silent production fallback.

The engine owns semantic sampling state. Device penalty tables, RNG buffers,
and adapter selection are versioned executor mirrors updated by request deltas,
not restaged in full for every token.

Greedy graph execution captures:

```text
model program -> sampled-row logits -> argmax -> token output
```

Stochastic graph execution can later capture the same pipeline when RNG state
and shape contracts are graph-safe. Until then, the generic executor selects
the eager path explicitly.

## Error boundaries

Errors retain typed sources until the API boundary:

```text
PrepareError   tokenizer, prompt, chat-template, request validation
LoadError      manifest, config, weights, device initialization
ScheduleError  capacity, arithmetic, invalid state transition
BatchError     malformed lowering input or violated batch invariant
ExecutionError CUDA enqueue, graph, kernel, output, synchronization
DecodeError    incremental token-to-text conversion
```

All are `thiserror` enums. The engine must not turn a model or CUDA error into a
`String` and later wrap that string in another error. Public API responses may
redact internal sources while tracing retains the complete chain.

## Contracts and property tests

The contract strategy remains:

- fallible constructors at trust boundaries;
- private fields for validated values;
- newtypes for semantically distinct integers;
- explicit tickets and typestate for provisional/committed state ownership;
- `debug_assert!` only for redundant internal facts;
- property tests for combinatorial and state-machine behavior.

Property tests should generate request arrivals, prompt sizes, token budgets,
prefill chunk sizes, cancellations, execution failures, and completions. After
every transition they prove:

1. no writable state location has multiple writers;
2. every shared prefix location's reference count equals its live references;
3. reserved, shared, exclusively owned, quarantined, and free capacity is
   conserved;
4. computed and in-flight progress never exceeds available token history;
5. failed or rejected batches apply exactly their declared correction;
6. output request IDs and cardinalities match `OutputSelection` exactly;
7. decode receives priority without starving prefill peers;
8. lowering preserves query ranges and causal context lengths;
9. stale request-slot generations and cross-arena leases are rejected;
10. completed reclamation epochs restore allocator state, while unresolved
    fences keep their locations unavailable;
11. mixed-batch lowering preserves the validated prefill-before-decode row
    partition and agrees with separate execution where equivalence is promised.

CUDA differential tests then compare eager and graph results for every supported
bucket and compare shared-program outputs with the trusted Llama path.
Cheap runtime invariant probes should also check capacity conservation,
reference counts, committed-versus-allocated progress, free-list uniqueness,
and use-after-free conditions. Property tests validate the model; probes detect
divergence in production-only paths.

## Proposed module layout

```text
src/
  engine/
    request.rs          authoritative request/token/frontend state
    scheduler.rs        policy, tickets, epochs, progress transitions
    state.rs            coordinator, prefixes, leases, reclamation
    batch.rs            mode-aware validated ForwardBatch
    executor.rs         ModelExecutor submission/completion contract
  model/
    registry.rs         ArchitectureFactory registry
    frontend.rs         ModelFrontend, TokenDecoder, processor types
    weight_source.rs    format/transport-independent named tensors
    programs/
      dense_decoder.rs  shared dense causal decoder
      moe_decoder.rs    added only when required
    llama_3_2.rs        private config/tensor mapping; builds DenseDecoder
  cuda/
    executor.rs         CudaExecutor<P, R, A, S>
    batch.rs            stable DeviceBatch buffers and uploads
    request_mirror.rs   versioned device-side derived request state
    workspace.rs        bucket-owned reusable intermediates
    graph.rs            generic graph cache and capture/replay
    state_arena.rs      grouped physical KV/recurrent storage
    attention.rs        sealed graph-aware backend strategy
    kernel_catalog.rs   cold typed selection into immutable kernel plans
    sampler.rs          model-neutral device sampling
    kernels.rs
    cublas.rs
```

## Migration sequence

The refactor should remain runnable after every step:

1. Add characterization tests for current Llama eager, graph, mixed-batch, and
   failure behavior.
2. Introduce semantic newtypes and validated `ForwardBatch` without changing
   execution.
3. Move authoritative prompt/generated tokens, decoder, sampling state, and
   logical slot history from `LlamaCudaBackend` into engine request state;
   retain only explicit versioned executor mirrors.
4. Replace `Backend::{add_request,remove_request,step}` with mode-aware
   `ModelExecutor::{submit,poll}` and execution tickets. Start with one in-flight
   ticket while proving the API, then enable overlap without redesign.
5. Extract sampling, device batching, workspaces, and graph management into
   `CudaExecutor` while the existing Llama computation becomes its initial
   `ModelProgram`.
6. Extract a sealed `AttentionBackend` and grouped `StateSchema`, initially
   implemented only by the current direct flat-KV path.
7. Add the typed construction-time `KernelCatalog`; resolve the current kernels
   into one immutable default `KernelPlan` without changing their execution.
8. Extract the common transformer computation into `DenseDecoder`; make the
   Llama module validate and construct it.
9. Add the architecture registry and independent `WeightSource`; remove
   model-name and checkpoint-format switches from serving code.
10. Run property tests, strict Clippy, CUDA differential tests, Compute
   Sanitizer, and the retained A100 benchmark after each performance-sensitive
   extraction.

The second supported model is the architectural acceptance test. Adding a
Llama-like model should require a private config/factory/frontend implementation and
construction of `DenseDecoder`, with no scheduler, engine, CUDA graph, sampler,
or kernel-lifecycle changes.

## Rejected designs

### One backend implementation per model

This duplicates graph capture, batching, KV metadata, sampling, synchronization,
and error handling. It is the current scaling problem.

### One trait per transformer component

This creates a large object graph and makes lifetimes, graph capture, and
workspace reuse harder. Most component choice is static program data. Attention
is the deliberate exception because its metadata, cache layout, kernels, and
graph legality vary as one strategy; it remains sealed and statically
dispatched.

### A universal tensor-operation IR first

An IR could eventually enable compilation and fusion, but building a complete
compiler abstraction before a second architecture exists would obscure the
required ownership refactor. Shared executable program types give the same
model isolation now and leave room to introduce an IR later behind
`ModelProgram`.

### Model-specific capability checks in the scheduler

The scheduler should reason about execution limits and batch contracts, not
architecture names. Execution-path selection belongs in the executor.

## Acceptance criteria

The design is realized when:

- `llama_3_2.rs` contains no scheduler backend, CUDA graph cache, sampling
  algorithm, workspace lifecycle, or request table;
- the engine has one authoritative semantic request state and every executor
  mirror is versioned and explicitly non-authoritative;
- eager and graph paths use the same model operations, even when a graph
  captures only a declared subregion;
- malformed batches are unconstructable outside the batch module;
- executor failures, cancellations, and rejected speculation cannot
  prematurely recycle state or leave progress uncorrected;
- shared prefix references and writable tails satisfy their distinct ownership
  invariants;
- a second dense decoder model adds no changes to engine or CUDA lifecycle
  modules;
- property, differential CUDA, sanitizer, and A100 performance gates pass.
