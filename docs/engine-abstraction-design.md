# Tesseract Engine Abstraction Design

Status: proposed design, 2026-08-08.

This document defines the boundary between request semantics, scheduling,
model architecture, and CUDA execution. The goal is not to maximize the number
of interfaces. It is to make ownership and invariants obvious, keep model
details contained, and preserve a zero-overhead hot path.

## Design rules

1. A trait must represent a real substitution boundary. Data variation uses
   structs and enums, not traits.
2. Dynamic dispatch is allowed at startup and once per engine batch. It is not
   used per token, transformer layer, tensor, or kernel launch.
3. The engine owns request progress, token history, sampling state, text
   decoding, and logical KV allocation.
4. The executor owns device memory, workspaces, CUDA graphs, streams, kernels,
   and physical execution policy.
5. A model family owns configuration validation, prompt syntax, tensor-name
   mapping, and the construction of a shared executable program. Model-specific
   types do not cross the model module boundary.
6. Invalid states are excluded with private fields, newtypes, and fallible
   constructors. Assertions are not a substitute for validating a trust
   boundary.
7. One model program is used by eager execution and CUDA graph capture. A graph
   is a captured execution of the normal program, not a second implementation.

## Target dependency direction

```text
api
 |
 v
text model ---------> engine request state
                         |
                         v
                 scheduler + KV allocator
                         |
                    BatchPlan + KvLease
                         |
                         v
                    BatchLowerer
                         |
                    ForwardBatch
                         |
                         v
              ModelExecutor (one batch call)
                         |
                CudaExecutor<M, S>
                   /             \
          CausalLm program      Sampler
                   |
        shared dense/MoE/etc. program
                   ^
                   |
       model module builds program + weights
```

Dependencies only point downward. The scheduler cannot import CUDA or model
architecture modules. CUDA cannot import API request types. A model module may
construct shared programs, but no other module may inspect its private config
or weight types.

## The three runtime traits

The serving path needs only three genuine runtime traits.

### 1. `TextModel`

This is the CPU-side semantic boundary. It is called during admission and after
sampled tokens return, never inside the GPU layer loop.

```rust
pub trait TextModel: Send + Sync + 'static {
    fn info(&self) -> &ModelInfo;

    fn render_chat(
        &self,
        messages: &[ChatMessage<'_>],
    ) -> Result<String, PrepareError>;

    fn encode(&self, text: &str) -> Result<Vec<TokenId>, PrepareError>;

    fn decoder(&self) -> Box<dyn TokenDecoder>;

    fn is_eos(&self, token: TokenId) -> bool;
}

pub trait TokenDecoder: Send {
    fn push(&mut self, token: TokenId) -> Result<&str, DecodeError>;
}
```

`TextModel` replaces the current ambiguous `Model` name. Its implementation may
remain architecture-specific because tokenizers and chat templates are not part
of device execution.

### 2. `ModelExecutor`

This is the only engine-to-device boundary. It consumes a fully validated,
model-neutral batch and returns exactly one token for every requested sample.

```rust
pub trait ModelExecutor: 'static {
    fn model_info(&self) -> &ModelInfo;
    fn limits(&self) -> ExecutionLimits;

    fn execute(
        &mut self,
        batch: &ForwardBatch<'_>,
    ) -> Result<ExecutionOutput, ExecutionError>;

    fn take_stats(&mut self) -> ExecutionStats;
    fn shutdown(&mut self) -> Result<(), ExecutionError>;
}
```

The outer engine remains generic over `E: ModelExecutor`, so ordinary serving
does not require a virtual call. A `Box<dyn ModelExecutor>` is used only by the
model registry and command-line construction path where heterogeneous model
selection is actually required.

`ModelExecutor` has no `add_request` or `remove_request`. Those methods force the
executor to maintain a second, hidden copy of request token state. The engine
already has the authoritative request table and should lower it into each
`ForwardBatch` directly.

### 3. `CausalLm`

This is the boundary between generic CUDA lifecycle machinery and an executable
model program. It is called once for a whole batch. It does not expose layers to
the engine.

```rust
pub(crate) trait CausalLm: 'static {
    fn geometry(&self) -> ModelGeometry;
    fn kv_layout(&self) -> KvLayout;

    fn enqueue_logits(
        &self,
        context: &mut CudaExecutionContext<'_>,
        batch: &DeviceBatch,
        workspace: &mut ModelWorkspace,
        logits: &mut LogitsBuffer,
    ) -> Result<(), ExecutionError>;
}
```

`CudaExecutor<M, S>` is generic over `M: CausalLm` and a concrete sampler. It
owns the stream, device KV cache, bucket policy, workspaces, graph cache, batch
uploads, synchronization, and statistics. The same `enqueue_logits` call is
made during eager execution and inside graph capture.

There are deliberately no `Attention`, `Mlp`, `Norm`, `Layer`, or `Kernel`
traits. Those operations do not vary independently at runtime. Shared program
types compose them statically.

## Load-time boundary

Model selection is a startup concern. It may use a fourth trait, kept entirely
outside the hot path:

```rust
pub trait ModelLoader: Send + Sync {
    fn probe(&self, manifest: &ModelManifest) -> Probe;

    fn load(
        &self,
        source: &ModelSource,
        target: &ExecutionTarget,
    ) -> Result<LoadedModel, LoadError>;
}

pub struct LoadedModel {
    pub text: Arc<dyn TextModel>,
    pub executor: Box<dyn ModelExecutor>,
}
```

The registry contains loaders, not checks for model names scattered through
the engine. A Llama loader parses its private configuration, validates tensor
names and shapes, and constructs a shared `DenseDecoder` program. Only
`TextModel` and `ModelExecutor` leave the module.

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
```

`ForwardBatch` has private fields and is created only by `BatchLowerer`:

```rust
pub struct ForwardBatch<'a> {
    token_ids: Cow<'a, [TokenId]>,
    positions: Cow<'a, [Position]>,
    current_slots: Cow<'a, [KvSlot]>,
    sequences: Vec<SequenceView<'a>>,
    query_sequence: Cow<'a, [SequenceIndex]>,
    context_lengths: Cow<'a, [u32]>,
    samples: Vec<SampleSpec>,
}

pub struct SequenceView<'a> {
    request_id: RequestId,
    context_slots: &'a [KvSlot],
    query: Range<usize>,
}

pub struct SampleSpec {
    request_id: RequestId,
    row: QueryRow,
    params: SamplingParams,
}
```

Construction proves the invariants already documented in
`batching-architecture.md`: aligned token metadata, non-aliasing destination
slots, increasing sample rows, valid sequence indices, causal context bounds,
and one sample per decode request. CUDA code receives a valid value rather than
revalidating parallel vectors on every execution branch.

`DeviceBatch` is not a public contract. It is an executor-owned set of stable
device buffers populated from `ForwardBatch`. Graph padding and bucket extents
exist only in `DeviceBatch`; logical engine batches never contain fake requests
or scratch slots.

## Transactional execution

KV allocation and request progress must be committed atomically around model
execution:

```text
reserve request capacity
    -> create provisional BatchPlan and KvLease
    -> lower to ForwardBatch
    -> execute and validate output cardinality
    -> commit KvLease and request progress
    -> append sampled tokens and decode text
```

Dropping an uncommitted `KvLease` returns its slots. An execution failure first
waits for the executor's completion fence, then releases the lease; a slot can
never be recycled while a failed asynchronous KV write may still target it.

This removes the duplicated `slots`, `prompt`, and `generated` collections now
held by the Llama backend. It also prevents partial scheduler commits when an
executor returns malformed output.

## Generic CUDA executor

`CudaExecutor<M, S>` owns every mechanism that should be identical across
models:

- host-to-device batch lowering;
- token, request, and context bucketing;
- reusable eager workspaces;
- CUDA graph capture, update, replay, and warmup;
- flat physical KV allocation and sentinel/scratch regions;
- stream and completion-fence discipline;
- sampled-row gathering, logits processing, and sampling;
- execution counters and latency instrumentation;
- readiness and shutdown synchronization.

Its important internal types are concrete, not traits:

```rust
pub(crate) struct BucketPolicy { /* finite configured shapes */ }
pub(crate) struct WorkspacePool { /* keyed by ExecutionShape */ }
pub(crate) struct GraphCache { /* keyed by ExecutionShape */ }
pub(crate) struct DeviceKvCache { /* storage described by KvLayout */ }

pub(crate) struct ExecutionShape {
    mode: ForwardMode,
    query_bucket: usize,
    request_bucket: usize,
    context_bucket: usize,
    sampling: SamplingMode,
}
```

Graph and eager selection belongs here because it is execution policy, not
Llama semantics. `CausalLm` supplies only the model program and its storage
geometry.

## Sampling

Sampling is model-neutral and belongs beside the executor, not in a model file.
`SamplingParams` is validated data. A concrete `CudaSampler` can support greedy,
temperature, top-p, and later penalties or constrained decoding. CPU sampling
may exist as a test oracle, but it is not a silent production fallback.

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
- typestate or RAII for provisional/committed KV ownership;
- `debug_assert!` only for redundant internal facts;
- property tests for combinatorial and state-machine behavior.

Property tests should generate request arrivals, prompt sizes, token budgets,
prefill chunk sizes, cancellations, execution failures, and completions. After
every transition they prove:

1. no physical KV slot has two live owners;
2. reserved plus free capacity is conserved;
3. computed progress never exceeds available token history;
4. failed batches do not commit progress;
5. output request IDs equal sampled request IDs exactly;
6. decode receives priority without starving prefill peers;
7. lowering preserves query ranges and causal context lengths;
8. dropping any provisional lease restores allocator state.

CUDA differential tests then compare eager and graph results for every supported
bucket and compare shared-program outputs with the trusted Llama path.

## Proposed module layout

```text
src/
  engine/
    request.rs          authoritative request/token/decoder state
    scheduler.rs        policy and progress transitions
    kv.rs               reservations and transactional KvLease
    batch.rs            BatchPlan, BatchLowerer, ForwardBatch
    executor.rs         ModelExecutor contract and output validation
  model/
    registry.rs         ModelLoader registry
    text.rs             TextModel, TokenDecoder, common semantic types
    programs/
      dense_decoder.rs  shared dense causal decoder
      moe_decoder.rs    added only when required
    llama_3_2.rs        private config/chat/tensor mapping; builds DenseDecoder
    weights.rs          SafeTensors access and typed weight loading helpers
  cuda/
    executor.rs         CudaExecutor<M, S>
    batch.rs            stable DeviceBatch buffers and uploads
    workspace.rs        bucket-owned reusable intermediates
    graph.rs            generic graph cache and capture/replay
    kv.rs               device flat-KV storage
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
3. Move prompt/generated tokens, decoder, sampling state, and logical slot
   history from `LlamaCudaBackend` into engine request state.
4. Replace `Backend::{add_request,remove_request,step}` with the narrow
   `ModelExecutor::execute` contract and transactional `KvLease` commits.
5. Extract sampling, device batching, workspaces, and graph management into
   `CudaExecutor` while the existing Llama computation remains its
   `CausalLm` implementation.
6. Extract the common transformer computation into `DenseDecoder`; make the
   Llama module validate and construct it.
7. Add the loader registry and remove model-name switches from serving code.
8. Run property tests, strict Clippy, CUDA differential tests, Compute
   Sanitizer, and the retained A100 benchmark after each performance-sensitive
   extraction.

The second supported model is the architectural acceptance test. Adding a
Llama-like model should require a private config/loader/text implementation and
construction of `DenseDecoder`, with no scheduler, engine, CUDA graph, sampler,
or kernel-lifecycle changes.

## Rejected designs

### One backend implementation per model

This duplicates graph capture, batching, KV metadata, sampling, synchronization,
and error handling. It is the current scaling problem.

### One trait per transformer component

This creates a large object graph and makes lifetimes, graph capture, and
workspace reuse harder. Component choice is static program data, not an online
substitution boundary.

### A universal tensor-operation IR first

An IR could eventually enable compilation and fusion, but building a complete
compiler abstraction before a second architecture exists would obscure the
required ownership refactor. Shared executable program types give the same
model isolation now and leave room to introduce an IR later behind `CausalLm`.

### Model-specific capability checks in the scheduler

The scheduler should reason about execution limits and batch contracts, not
architecture names. Execution-path selection belongs in the executor.

## Acceptance criteria

The design is realized when:

- `llama_3_2.rs` contains no scheduler backend, CUDA graph cache, sampling
  algorithm, workspace lifecycle, or request table;
- the engine has one authoritative copy of request token and logical KV state;
- eager and graph paths call the same model program;
- malformed batches are unconstructable outside the batch module;
- executor failures cannot commit progress or prematurely recycle KV slots;
- a second dense decoder model adds no changes to engine or CUDA lifecycle
  modules;
- property, differential CUDA, sanitizer, and A100 performance gates pass.
