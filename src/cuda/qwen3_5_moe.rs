use std::sync::Arc;

use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream};
use cutile::{
    api,
    core::bf16,
    tensor::{PartitionMut, Reshape, Tensor, TensorView, ToHostVec},
    tile_kernel::TileKernel,
};

use crate::{
    engine::{ExecutionError, ModelExecutor, RequestId, SamplingInput, StateSchema, TokenId},
    model::{
        CudaForwardReport, CudaModelReport, CudaTokenLogit, Model, ModelError,
        weights::{WeightDtype, WeightSource},
    },
};

use super::{
    batch::{CudaBatch, SampleTarget},
    cublas,
    execution::StreamExecution,
    executor::{CudaExecutor, ModelProgram, ProgramOutput},
    gdn::{self as gdn_backend, GdnPrefillPlan, GdnState},
    kernels,
    linear::{ExpertProjection, Fp8W8A16Linear, GroupedNvfp4W4A16, Nvfp4W4A16Linear},
    moe::{self as moe_backend, RoutingPlan},
    qwen_attention::{AttentionInput, QwenFlatKvAttention},
};

type Bf16Tensor = Arc<Tensor<bf16>>;
type F32Tensor = Arc<Tensor<f32>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerKind {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone)]
pub(crate) struct Config {
    pub(crate) attn_output_gate: bool,
    pub(crate) head_dim: usize,
    pub(crate) hidden_act: String,
    pub(crate) hidden_size: usize,
    pub(crate) layers: Vec<LayerKind>,
    pub(crate) linear_conv_kernel_dim: usize,
    pub(crate) linear_key_head_dim: usize,
    pub(crate) linear_num_key_heads: usize,
    pub(crate) linear_num_value_heads: usize,
    pub(crate) linear_value_head_dim: usize,
    pub(crate) mamba_ssm_dtype: String,
    pub(crate) max_position_embeddings: usize,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) num_experts: usize,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) partial_rotary_factor: f32,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_interleaved: bool,
    pub(crate) rope_section: Vec<usize>,
    pub(crate) rope_theta: f32,
    pub(crate) rope_type: String,
    pub(crate) shared_expert_intermediate_size: usize,
    pub(crate) vocab_size: usize,
}

pub(crate) struct Artifact {
    pub(crate) model: Arc<dyn Model>,
    pub(crate) config: Config,
    pub(crate) weights: Arc<dyn WeightSource>,
}

pub(crate) struct Checkpoint {
    embedding: Bf16Tensor,
    final_norm: Bf16Tensor,
    lm_head: Nvfp4W4A16Linear,
    layers: Vec<Layer>,
}

struct Program {
    model: Arc<dyn Model>,
    config: Config,
    checkpoint: Checkpoint,
    stream: Arc<Stream>,
    attention: QwenFlatKvAttention,
    recurrent: Vec<Option<GdnState>>,
    state_schema: StateSchema,
    kv_capacity: usize,
    recurrent_capacity: usize,
    max_batch_tokens: usize,
}

struct PaddedBatch {
    token_ids: Vec<u32>,
    positions: Vec<u32>,
    current_slots: Vec<u32>,
    request_indices: Vec<u32>,
    recurrent_slots: Vec<i32>,
    query_start_offsets: Vec<u32>,
    context_lengths: Vec<i32>,
    context_slots: Vec<u32>,
    context_bucket: usize,
    requests: usize,
    reset_recurrent_slots: Vec<i32>,
    logical_rows: usize,
    rows: usize,
}

const ROW_ALIGNMENT: usize = 16;
const PRIVATE_PADDING_SLOTS: usize = ROW_ALIGNMENT - 1;
const HIDDEN_SIZE: usize = 2048;
const EMBEDDING_BLOCK: usize = 256;

impl PaddedBatch {
    fn new(
        batch: &CudaBatch,
        kv_capacity: usize,
        recurrent_capacity: usize,
        max_batch_tokens: usize,
    ) -> Result<Self, ModelError> {
        let logical_rows = batch.num_tokens();
        if logical_rows == 0 || logical_rows > max_batch_tokens {
            return Err(ModelError::Cuda(format!(
                "Qwen batch has {logical_rows} tokens; configured maximum is {max_batch_tokens}"
            )));
        }
        let rows = logical_rows.div_ceil(ROW_ALIGNMENT) * ROW_ALIGNMENT;
        let padding = rows - logical_rows;
        let mut token_ids = batch.token_ids.clone();
        let mut positions = batch.positions.clone();
        let mut current_slots = batch.current_slots.clone();
        let mut request_indices = batch.request_indices.clone();
        let mut context_lengths = batch.context_lengths.clone();
        let mut query_start_offsets = batch.query_start_offsets.clone();
        let mut recurrent_slots = batch
            .recurrent_slots
            .iter()
            .map(|slot| {
                let slot = slot.ok_or_else(|| {
                    ModelError::Cuda("Qwen batch is missing a recurrent-state slot".into())
                })?;
                i32::try_from(slot)
                    .map_err(|_| ModelError::Cuda("Qwen recurrent slot exceeds i32".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if current_slots
            .iter()
            .any(|slot| *slot as usize >= kv_capacity)
            || recurrent_slots
                .iter()
                .any(|slot| *slot < 0 || *slot as usize >= recurrent_capacity)
        {
            return Err(ModelError::Cuda(
                "Qwen batch references state outside its scheduler-owned capacity".into(),
            ));
        }
        let logical_requests = batch.request_count();
        let mut reset_recurrent_slots = Vec::with_capacity(logical_requests);
        for (request, slot) in recurrent_slots.iter().copied().enumerate() {
            let row = batch.query_start_offsets[request] as usize;
            if batch.positions[row] == 0 {
                reset_recurrent_slots.push(slot);
            }
        }
        let mut contexts = batch.contexts().to_vec();
        for index in 0..padding {
            let private_kv = kv_capacity
                .checked_add(index)
                .and_then(|slot| u32::try_from(slot).ok())
                .ok_or_else(|| ModelError::Cuda("Qwen private KV slot overflowed".into()))?;
            let private_recurrent = recurrent_capacity
                .checked_add(index)
                .and_then(|slot| i32::try_from(slot).ok())
                .ok_or_else(|| ModelError::Cuda("Qwen private recurrent slot overflowed".into()))?;
            token_ids.push(0);
            positions.push(0);
            current_slots.push(private_kv);
            request_indices.push(
                u32::try_from(logical_requests + index)
                    .map_err(|_| ModelError::Cuda("Qwen request index overflowed".into()))?,
            );
            context_lengths.push(1);
            recurrent_slots.push(private_recurrent);
            let next = query_start_offsets
                .last()
                .copied()
                .and_then(|offset| offset.checked_add(1))
                .ok_or_else(|| ModelError::Cuda("Qwen query offset overflowed".into()))?;
            query_start_offsets.push(next);
            contexts.push(vec![private_kv]);
        }
        let requests = contexts.len();
        let context_bucket = contexts
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(1)
            .div_ceil(16)
            * 16;
        let sentinel = u32::try_from(kv_capacity)
            .map_err(|_| ModelError::Cuda("Qwen KV sentinel overflowed".into()))?;
        let mut context_slots = Vec::with_capacity(requests * context_bucket);
        for context in contexts {
            context_slots.extend_from_slice(&context);
            context_slots.resize(
                context_slots.len() + context_bucket - context.len(),
                sentinel,
            );
        }
        Ok(Self {
            token_ids,
            positions,
            current_slots,
            request_indices,
            recurrent_slots,
            query_start_offsets,
            context_lengths,
            context_slots,
            context_bucket,
            requests,
            reset_recurrent_slots,
            logical_rows,
            rows,
        })
    }
}

struct Layer {
    input_norm: Bf16Tensor,
    post_attention_norm: Bf16Tensor,
    attention: Attention,
    moe: Moe,
}

enum Attention {
    Linear(LinearAttention),
    Full(FullAttention),
}

struct LinearAttention {
    a_log: F32Tensor,
    conv1d: Bf16Tensor,
    dt_bias: F32Tensor,
    input_a: Bf16Tensor,
    input_b: Bf16Tensor,
    input_qkv: Fp8W8A16Linear,
    input_z: Fp8W8A16Linear,
    norm: Bf16Tensor,
    output: Fp8W8A16Linear,
}

struct FullAttention {
    key: Fp8W8A16Linear,
    key_norm: Bf16Tensor,
    output: Fp8W8A16Linear,
    query: Fp8W8A16Linear,
    query_norm: Bf16Tensor,
    value: Fp8W8A16Linear,
}

struct Moe {
    router: Bf16Tensor,
    routed_gate: GroupedNvfp4W4A16,
    routed_up: GroupedNvfp4W4A16,
    routed_down: GroupedNvfp4W4A16,
    shared_gate: Nvfp4W4A16Linear,
    shared_up: Nvfp4W4A16Linear,
    shared_down: Nvfp4W4A16Linear,
    shared_router: Bf16Tensor,
}

impl Checkpoint {
    fn load(artifact: &Artifact, stream: &Arc<Stream>) -> Result<Self, ModelError> {
        let source = artifact.weights.as_ref();
        let embedding = load_bf16(source, "model.language_model.embed_tokens.weight", stream)?;
        let final_norm = load_bf16(source, "model.language_model.norm.weight", stream)?;
        let lm_head = Nvfp4W4A16Linear::load(source, "lm_head", stream)?;
        let mut layers = Vec::with_capacity(artifact.config.layers.len());
        for (index, kind) in artifact.config.layers.iter().copied().enumerate() {
            let prefix = format!("model.language_model.layers.{index}");
            let input_norm =
                load_bf16(source, &format!("{prefix}.input_layernorm.weight"), stream)?;
            let post_attention_norm = load_bf16(
                source,
                &format!("{prefix}.post_attention_layernorm.weight"),
                stream,
            )?;
            let attention = match kind {
                LayerKind::LinearAttention => {
                    let prefix = format!("{prefix}.linear_attn");
                    Attention::Linear(LinearAttention {
                        a_log: load_f32(source, &format!("{prefix}.A_log"), stream)?,
                        conv1d: load_bf16_as(
                            source,
                            &format!("{prefix}.conv1d.weight"),
                            &[8192, 4],
                            stream,
                        )?,
                        dt_bias: load_f32(source, &format!("{prefix}.dt_bias"), stream)?,
                        input_a: load_bf16(source, &format!("{prefix}.in_proj_a.weight"), stream)?,
                        input_b: load_bf16(source, &format!("{prefix}.in_proj_b.weight"), stream)?,
                        input_qkv: Fp8W8A16Linear::load(
                            source,
                            &format!("{prefix}.in_proj_qkv"),
                            stream,
                        )?,
                        input_z: Fp8W8A16Linear::load(
                            source,
                            &format!("{prefix}.in_proj_z"),
                            stream,
                        )?,
                        norm: load_bf16(source, &format!("{prefix}.norm.weight"), stream)?,
                        output: Fp8W8A16Linear::load(
                            source,
                            &format!("{prefix}.out_proj"),
                            stream,
                        )?,
                    })
                }
                LayerKind::FullAttention => {
                    let prefix = format!("{prefix}.self_attn");
                    Attention::Full(FullAttention {
                        key: Fp8W8A16Linear::load(source, &format!("{prefix}.k_proj"), stream)?,
                        key_norm: load_bf16(source, &format!("{prefix}.k_norm.weight"), stream)?,
                        output: Fp8W8A16Linear::load(source, &format!("{prefix}.o_proj"), stream)?,
                        query: Fp8W8A16Linear::load(source, &format!("{prefix}.q_proj"), stream)?,
                        query_norm: load_bf16(source, &format!("{prefix}.q_norm.weight"), stream)?,
                        value: Fp8W8A16Linear::load(source, &format!("{prefix}.v_proj"), stream)?,
                    })
                }
            };
            let moe_prefix = format!("{prefix}.mlp");
            let experts_prefix = format!("{moe_prefix}.experts");
            let shared_prefix = format!("{moe_prefix}.shared_expert");
            let moe = Moe {
                router: load_bf16(source, &format!("{moe_prefix}.gate.weight"), stream)?,
                routed_gate: GroupedNvfp4W4A16::load(
                    source,
                    &experts_prefix,
                    ExpertProjection::Gate,
                    artifact.config.num_experts,
                    stream,
                )?,
                routed_up: GroupedNvfp4W4A16::load(
                    source,
                    &experts_prefix,
                    ExpertProjection::Up,
                    artifact.config.num_experts,
                    stream,
                )?,
                routed_down: GroupedNvfp4W4A16::load(
                    source,
                    &experts_prefix,
                    ExpertProjection::Down,
                    artifact.config.num_experts,
                    stream,
                )?,
                shared_gate: Nvfp4W4A16Linear::load(
                    source,
                    &format!("{shared_prefix}.gate_proj"),
                    stream,
                )?,
                shared_up: Nvfp4W4A16Linear::load(
                    source,
                    &format!("{shared_prefix}.up_proj"),
                    stream,
                )?,
                shared_down: Nvfp4W4A16Linear::load(
                    source,
                    &format!("{shared_prefix}.down_proj"),
                    stream,
                )?,
                shared_router: load_bf16(
                    source,
                    &format!("{moe_prefix}.shared_expert_gate.weight"),
                    stream,
                )?,
            };
            layers.push(Layer {
                input_norm,
                post_attention_norm,
                attention,
                moe,
            });
        }
        Ok(Self {
            embedding,
            final_norm,
            lm_head,
            layers,
        })
    }

    fn device_bytes(&self) -> usize {
        self.embedding.num_bytes()
            + self.final_norm.num_bytes()
            + self.lm_head.device_bytes()
            + self.layers.iter().map(Layer::device_bytes).sum::<usize>()
    }
}

impl Program {
    fn load(
        artifact: Artifact,
        device_id: usize,
        kv_capacity: usize,
        max_batch_tokens: usize,
        max_running: usize,
    ) -> Result<Self, ModelError> {
        let physical_kv_capacity = kv_capacity
            .checked_add(PRIVATE_PADDING_SLOTS)
            .ok_or_else(|| ModelError::Cuda("Qwen physical KV capacity overflowed".into()))?;
        let physical_recurrent_capacity = max_running
            .checked_add(PRIVATE_PADDING_SLOTS)
            .ok_or_else(|| ModelError::Cuda("Qwen recurrent capacity overflowed".into()))?;
        let device = Device::new(device_id).map_err(|error| {
            ModelError::Cuda(format!("initialize device {device_id}: {error:?}"))
        })?;
        let stream = device
            .new_stream()
            .map_err(|error| ModelError::Cuda(format!("create stream: {error:?}")))?;
        let checkpoint = Checkpoint::load(&artifact, &stream)?;
        let full_layers = artifact
            .config
            .layers
            .iter()
            .filter(|kind| **kind == LayerKind::FullAttention)
            .count();
        let attention = QwenFlatKvAttention::load(
            full_layers,
            physical_kv_capacity,
            artifact.config.max_position_embeddings,
            artifact.config.rope_theta,
            &stream,
        )?;
        let mut recurrent = Vec::with_capacity(artifact.config.layers.len());
        for kind in &artifact.config.layers {
            recurrent.push(match kind {
                LayerKind::LinearAttention => {
                    Some(GdnState::zeros(physical_recurrent_capacity, &stream)?)
                }
                LayerKind::FullAttention => None,
            });
        }
        let state_schema = StateSchema::try_hybrid(kv_capacity, max_running)
            .map_err(|error| ModelError::InvalidConfig(error.to_string()))?;
        Ok(Self {
            model: artifact.model,
            config: artifact.config,
            checkpoint,
            stream,
            attention,
            recurrent,
            state_schema,
            kv_capacity,
            recurrent_capacity: max_running,
            max_batch_tokens,
        })
    }

    fn forward(&mut self, batch: &CudaBatch) -> Result<ProgramOutput, ModelError> {
        let padded = PaddedBatch::new(
            batch,
            self.kv_capacity,
            self.recurrent_capacity,
            self.max_batch_tokens,
        )?;
        let rows = padded.rows;
        let logical_rows = padded.logical_rows;
        let stream = &self.stream;
        let token_ids = upload_u32(&padded.token_ids, stream, "Qwen token IDs")?;
        let positions = upload_u32(&padded.positions, stream, "Qwen positions")?;
        let current_slots = upload_u32(&padded.current_slots, stream, "Qwen current KV slots")?;
        let request_indices = upload_u32(&padded.request_indices, stream, "Qwen request indices")?;
        let recurrent_slots =
            upload_i32_named(&padded.recurrent_slots, stream, "Qwen recurrent slots")?;
        let context_lengths =
            upload_i32_named(&padded.context_lengths, stream, "Qwen context lengths")?;
        let context_storage = upload_u32(&padded.context_slots, stream, "Qwen context slots")?;
        let context_slots = context_storage
            .view(&[padded.requests, padded.context_bucket])
            .map_err(|error| ModelError::Cuda(format!("view Qwen context slots: {error:?}")))?;
        let mut hidden = api::zeros::<bf16>(&[rows, HIDDEN_SIZE])
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("allocate Qwen embedding: {error:?}")))?;
        let (_, _, hidden_partition) = kernels::embedding_bf16(
            &token_ids,
            &*self.checkpoint.embedding,
            (&mut hidden).partition([1, EMBEDDING_BLOCK]),
        )
        .generics(vec![HIDDEN_SIZE.to_string(), EMBEDDING_BLOCK.to_string()])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("execute Qwen embedding: {error:?}")))?;
        drop(hidden_partition);
        let hidden = Arc::new(hidden);
        let prefill = if batch.num_prefill_tokens > 0 {
            Some(GdnPrefillPlan::from_offsets(
                &padded.query_start_offsets,
                stream,
            )?)
        } else {
            None
        };
        let mut execution = StreamExecution::new(stream);

        let mut pending: Option<(Bf16Tensor, Bf16Tensor)> = None;
        let mut full_layer = 0usize;
        for layer_index in 0..self.checkpoint.layers.len() {
            let layer = &self.checkpoint.layers[layer_index];
            let previous = pending.take();
            let result = match &layer.attention {
                Attention::Linear(_) => {
                    let state = self.recurrent[layer_index].as_mut().ok_or_else(|| {
                        ModelError::Cuda(format!(
                            "Qwen linear layer {layer_index} has no recurrent state"
                        ))
                    })?;
                    state.reset_slots(&padded.reset_recurrent_slots, &mut execution)?;
                    layer.forward_linear(
                        previous
                            .as_ref()
                            .map(|(residual, _)| residual.clone())
                            .unwrap_or_else(|| hidden.clone()),
                        previous.as_ref().map(|(_, update)| update.clone()),
                        state,
                        recurrent_slots.clone(),
                        prefill.as_ref(),
                        rows,
                        logical_rows,
                        self.config.rms_norm_eps,
                        &mut execution,
                    )?
                }
                Attention::Full(_) => {
                    let result = layer.forward_full(
                        previous
                            .as_ref()
                            .map(|(residual, _)| residual.clone())
                            .unwrap_or_else(|| hidden.clone()),
                        previous.as_ref().map(|(_, update)| update.clone()),
                        &self.attention,
                        full_layer,
                        positions.clone(),
                        current_slots.clone(),
                        request_indices.clone(),
                        &context_slots,
                        context_lengths.clone(),
                        rows,
                        logical_rows,
                        self.config.rms_norm_eps,
                        &mut execution,
                    )?;
                    full_layer += 1;
                    result
                }
            };
            pending = Some(result);
        }

        if batch.samples.is_empty() {
            execution.synchronize("complete Qwen forward")?;
            return Ok(ProgramOutput::None);
        }
        let (residual, update) =
            pending.ok_or_else(|| ModelError::Cuda("Qwen checkpoint has no layers".into()))?;
        let (final_hidden, _) = gemma_add_rms_norm(
            residual,
            update,
            self.checkpoint.final_norm.clone(),
            rows,
            self.config.rms_norm_eps,
            stream,
        )?;
        self.sample(final_hidden, batch, &mut execution)
    }

    fn sample(
        &self,
        final_hidden: Bf16Tensor,
        batch: &CudaBatch,
        execution: &mut StreamExecution<'_>,
    ) -> Result<ProgramOutput, ModelError> {
        const ARGMAX_BLOCK: usize = 256;
        const ARGMAX_REDUCE_BLOCK: usize = 1024;
        let samples = batch.samples.len();
        let sample_rows = samples.div_ceil(ROW_ALIGNMENT) * ROW_ALIGNMENT;
        let mut padded_sample_rows = batch.sample_rows.clone();
        padded_sample_rows.resize(sample_rows, batch.sample_rows[0]);
        let sample_rows_device = upload_u32(&padded_sample_rows, &self.stream, "Qwen sample rows")?;
        let mut sampled_hidden = api::zeros::<bf16>(&[sample_rows, HIDDEN_SIZE])
            .sync_on(&self.stream)
            .map_err(|error| {
                ModelError::Cuda(format!("allocate Qwen sampled hidden states: {error:?}"))
            })?;
        let (_, _, sampled_partition) = unsafe {
            kernels::gather_rows_bf16(
                final_hidden.device_pointer(),
                &sample_rows_device,
                (&mut sampled_hidden).partition([1, EMBEDDING_BLOCK]),
            )
        }
        .generics(vec![HIDDEN_SIZE.to_string(), EMBEDDING_BLOCK.to_string()])
        .sync_on(&self.stream)
        .map_err(|error| ModelError::Cuda(format!("gather Qwen sample rows: {error:?}")))?;
        drop(sampled_partition);
        let logits =
            self.checkpoint
                .lm_head
                .enqueue(Arc::new(sampled_hidden), sample_rows, execution)?;
        if batch.all_samples_greedy {
            let blocks = self.config.vocab_size.div_ceil(ARGMAX_BLOCK);
            let mut block_max = api::zeros::<f32>(&[sample_rows, blocks])
                .sync_on(&self.stream)
                .map_err(|error| ModelError::Cuda(format!("allocate Qwen argmax: {error:?}")))?;
            let mut block_index = api::zeros::<u32>(&[sample_rows, blocks])
                .sync_on(&self.stream)
                .map_err(|error| {
                    ModelError::Cuda(format!("allocate Qwen argmax indices: {error:?}"))
                })?;
            let (_, block_max_partition, block_index_partition, _) =
                kernels::argmax_blocks_batch_bf16(
                    Arc::new(logits),
                    (&mut block_max).partition([1, 1]),
                    (&mut block_index).partition([1, 1]),
                    self.config.vocab_size as i32,
                )
                .generics(vec![ARGMAX_BLOCK.to_string()])
                .sync_on(&self.stream)
                .map_err(|error| ModelError::Cuda(format!("execute Qwen argmax: {error:?}")))?;
            drop(block_max_partition);
            drop(block_index_partition);
            let mut sampled = api::zeros::<u32>(&[sample_rows])
                .sync_on(&self.stream)
                .map_err(|error| {
                    ModelError::Cuda(format!("allocate Qwen sampled tokens: {error:?}"))
                })?;
            let (_, _, sampled_partition, _) = kernels::argmax_reduce_batch_bf16(
                Arc::new(block_max),
                Arc::new(block_index),
                (&mut sampled).partition([1]),
                blocks as i32,
            )
            .generics(vec![ARGMAX_REDUCE_BLOCK.to_string()])
            .sync_on(&self.stream)
            .map_err(|error| ModelError::Cuda(format!("reduce Qwen argmax: {error:?}")))?;
            drop(sampled_partition);
            let mut sampled = sampled
                .to_host_vec()
                .sync_on(&self.stream)
                .map_err(|error| {
                    ModelError::Cuda(format!("download Qwen sampled tokens: {error:?}"))
                })?;
            execution.mark_synchronized();
            sampled.truncate(samples);
            return Ok(ProgramOutput::Tokens(
                sampled.into_iter().map(TokenId::new).collect(),
            ));
        }
        let logits = logits
            .to_host_vec()
            .sync_on(&self.stream)
            .map_err(|error| ModelError::Cuda(format!("download Qwen logits: {error:?}")))?;
        execution.mark_synchronized();
        let mut logits = logits.into_iter().map(bf16::to_f32).collect::<Vec<_>>();
        logits.truncate(samples * self.config.vocab_size);
        Ok(ProgramOutput::HostLogits {
            values: logits,
            vocab_size: self.config.vocab_size,
        })
    }
}

impl ModelProgram for Program {
    fn model(&self) -> Arc<dyn Model> {
        self.model.clone()
    }

    fn state_schema(&self) -> &StateSchema {
        &self.state_schema
    }

    fn execute(&mut self, batch: &CudaBatch) -> Result<ProgramOutput, ExecutionError> {
        self.forward(batch)
            .map_err(|error| ExecutionError::Execution(error.to_string()))
    }
}

pub(crate) fn load_executor(
    artifact: Artifact,
    device_id: usize,
    kv_capacity_tokens: usize,
    max_batch_tokens: usize,
    max_running: usize,
) -> Result<Box<dyn ModelExecutor>, ModelError> {
    let program = Program::load(
        artifact,
        device_id,
        kv_capacity_tokens,
        max_batch_tokens,
        max_running,
    )?;
    tracing::info!(
        kv_capacity_tokens,
        max_batch_tokens,
        max_running,
        "Qwen hybrid CUDA executor loaded"
    );
    Ok(Box::new(CudaExecutor::new(program)))
}

pub(crate) fn forward_report(
    model_id: &str,
    artifact: Artifact,
    device_id: usize,
    prompt: &str,
) -> Result<CudaForwardReport, ModelError> {
    let token_ids = artifact.model.encode(prompt)?;
    if token_ids.is_empty() {
        return Err(ModelError::InvalidInput(
            "prompt must encode to at least one token".into(),
        ));
    }
    let rows = token_ids.len();
    let capacity = rows
        .checked_add(1)
        .ok_or_else(|| ModelError::InvalidInput("prompt is too long".into()))?;
    let slots = (0..rows)
        .map(|slot| {
            u32::try_from(slot).map_err(|_| ModelError::InvalidInput("prompt is too long".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let sampling = SamplingInput::try_new(1.0, 1.0, 0.0)
        .map_err(|error| ModelError::InvalidInput(error.to_string()))?;
    let batch = CudaBatch {
        token_ids,
        positions: (0..rows as u32).collect(),
        current_slots: slots.clone(),
        request_indices: vec![0; rows],
        recurrent_slots: vec![Some(0)],
        query_start_offsets: vec![0, rows as u32],
        context_lengths: (1..=rows)
            .map(|length| {
                i32::try_from(length)
                    .map_err(|_| ModelError::InvalidInput("prompt is too long".into()))
            })
            .collect::<Result<Vec<_>, _>>()?,
        context_storage: vec![slots],
        num_requests: 1,
        sample_rows: vec![(rows - 1) as u32],
        samples: vec![SampleTarget {
            request_id: RequestId::now_v7(),
            sampling,
        }],
        all_samples_greedy: false,
        num_prefill_tokens: rows,
    };
    let mut program = Program::load(artifact, device_id, capacity, rows, 1)?;
    let ProgramOutput::HostLogits { values, vocab_size } = program.forward(&batch)? else {
        return Err(ModelError::Cuda(
            "Qwen forward omitted requested logits".into(),
        ));
    };
    let mut ranked = values
        .into_iter()
        .enumerate()
        .map(|(token_id, logit)| CudaTokenLogit {
            token_id: token_id as u32,
            logit,
        })
        .collect::<Vec<_>>();
    if ranked.len() != vocab_size {
        return Err(ModelError::Cuda(format!(
            "Qwen language-model head returned {} logits for vocabulary size {vocab_size}",
            ranked.len()
        )));
    }
    ranked.sort_unstable_by(|left, right| right.logit.total_cmp(&left.logit));
    ranked.truncate(20);
    let next_token_id = ranked
        .first()
        .map(|entry| entry.token_id)
        .ok_or_else(|| ModelError::Cuda("Qwen language-model head returned no logits".into()))?;
    let next_token_text = program.model.decoder().push(next_token_id)?;
    Ok(CudaForwardReport {
        model_id: model_id.into(),
        prompt_tokens: rows,
        next_token_id,
        next_token_text,
        top_logits: ranked,
    })
}

impl Layer {
    fn device_bytes(&self) -> usize {
        self.input_norm.num_bytes()
            + self.post_attention_norm.num_bytes()
            + self.attention.device_bytes()
            + self.moe.device_bytes()
    }

    fn forward_linear(
        &self,
        residual: Bf16Tensor,
        update: Option<Bf16Tensor>,
        state: &mut GdnState,
        state_slots: Arc<Tensor<i32>>,
        prefill: Option<&GdnPrefillPlan>,
        rows: usize,
        logical_rows: usize,
        epsilon: f32,
        execution: &mut StreamExecution<'_>,
    ) -> Result<(Bf16Tensor, Bf16Tensor), ModelError> {
        let stream = execution.stream().clone();
        let (attention_input, residual) = match update {
            Some(update) => gemma_add_rms_norm(
                residual,
                update,
                self.input_norm.clone(),
                rows,
                epsilon,
                &stream,
            )?,
            None => (
                gemma_rms_norm(
                    residual.clone(),
                    self.input_norm.clone(),
                    rows,
                    epsilon,
                    &stream,
                )?,
                residual,
            ),
        };
        let Attention::Linear(attention) = &self.attention else {
            return Err(ModelError::Cuda(
                "linear decode called for a full-attention layer".into(),
            ));
        };
        let attention_output = attention.forward(
            attention_input,
            state,
            state_slots,
            prefill,
            rows,
            epsilon,
            execution,
        )?;
        let (moe_input, residual) = gemma_add_rms_norm(
            residual,
            Arc::new(attention_output),
            self.post_attention_norm.clone(),
            rows,
            epsilon,
            &stream,
        )?;
        let moe_output = self.moe.forward(moe_input, rows, logical_rows, execution)?;
        Ok((residual, Arc::new(moe_output)))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_full(
        &self,
        residual: Bf16Tensor,
        update: Option<Bf16Tensor>,
        backend: &QwenFlatKvAttention,
        full_layer: usize,
        positions: Arc<Tensor<u32>>,
        current_slots: Arc<Tensor<u32>>,
        request_indices: Arc<Tensor<u32>>,
        context_slots: &TensorView<'_, u32>,
        context_lengths: Arc<Tensor<i32>>,
        rows: usize,
        logical_rows: usize,
        epsilon: f32,
        execution: &mut StreamExecution<'_>,
    ) -> Result<(Bf16Tensor, Bf16Tensor), ModelError> {
        let stream = execution.stream().clone();
        let (attention_input, residual) = match update {
            Some(update) => gemma_add_rms_norm(
                residual,
                update,
                self.input_norm.clone(),
                rows,
                epsilon,
                &stream,
            )?,
            None => (
                gemma_rms_norm(
                    residual.clone(),
                    self.input_norm.clone(),
                    rows,
                    epsilon,
                    &stream,
                )?,
                residual,
            ),
        };
        let Attention::Full(attention) = &self.attention else {
            return Err(ModelError::Cuda(
                "full-attention forward called for a linear-attention layer".into(),
            ));
        };
        let attention_output = attention.forward(
            attention_input,
            backend,
            full_layer,
            positions,
            current_slots,
            request_indices,
            context_slots,
            context_lengths,
            rows,
            epsilon,
            execution,
        )?;
        let (moe_input, residual) = gemma_add_rms_norm(
            residual,
            Arc::new(attention_output),
            self.post_attention_norm.clone(),
            rows,
            epsilon,
            &stream,
        )?;
        let moe_output = self.moe.forward(moe_input, rows, logical_rows, execution)?;
        Ok((residual, Arc::new(moe_output)))
    }
}

impl Attention {
    fn device_bytes(&self) -> usize {
        match self {
            Self::Linear(weights) => weights.device_bytes(),
            Self::Full(weights) => weights.device_bytes(),
        }
    }
}

impl LinearAttention {
    fn device_bytes(&self) -> usize {
        self.a_log.num_bytes()
            + self.conv1d.num_bytes()
            + self.dt_bias.num_bytes()
            + self.input_a.num_bytes()
            + self.input_b.num_bytes()
            + self.input_qkv.device_bytes()
            + self.input_z.device_bytes()
            + self.norm.num_bytes()
            + self.output.device_bytes()
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: Bf16Tensor,
        state: &mut GdnState,
        state_slots: Arc<Tensor<i32>>,
        prefill: Option<&GdnPrefillPlan>,
        rows: usize,
        epsilon: f32,
        execution: &mut StreamExecution<'_>,
    ) -> Result<Tensor<bf16>, ModelError> {
        const HIDDEN_SIZE: usize = 2048;
        const VALUE_SIZE: usize = 4096;
        if !rows.is_multiple_of(16) || hidden.shape() != [rows as i32, HIDDEN_SIZE as i32] {
            return Err(ModelError::Cuda("invalid Qwen GDN input geometry".into()));
        }
        let stream = execution.stream().clone();
        let mixed_qkv = Arc::new(self.input_qkv.enqueue(hidden.clone(), rows, execution)?);
        let mixed_qkv = Arc::new(match prefill {
            Some(plan) => state.prefill_conv(
                mixed_qkv,
                self.conv1d.clone(),
                plan.query_start_offsets(),
                state_slots.clone(),
                rows,
                plan.requests(),
                &stream,
            )?,
            None => state.decode_conv(
                mixed_qkv,
                self.conv1d.clone(),
                state_slots.clone(),
                rows,
                &stream,
            )?,
        });
        let a = bf16_gemm(
            self.input_a.clone(),
            hidden.clone(),
            32,
            rows,
            HIDDEN_SIZE,
            "Qwen GDN a projection",
            &stream,
        )?;
        let b = bf16_gemm(
            self.input_b.clone(),
            hidden.clone(),
            32,
            rows,
            HIDDEN_SIZE,
            "Qwen GDN b projection",
            &stream,
        )?;
        let recurrent = Arc::new(match prefill {
            Some(plan) => state.prefill(
                mixed_qkv,
                a,
                b,
                self.a_log.clone(),
                self.dt_bias.clone(),
                state_slots,
                plan,
                &stream,
            )?,
            None => state.decode(
                mixed_qkv,
                a,
                b,
                self.a_log.clone(),
                self.dt_bias.clone(),
                state_slots,
                rows,
                &stream,
            )?,
        });
        let gate = self
            .input_z
            .enqueue(hidden, rows, execution)?
            .reshape(&[rows, 32, 128])
            .map_err(|error| ModelError::Cuda(format!("reshape Qwen GDN z gate: {error:?}")))?;
        let gated = gdn_backend::output_gate(
            recurrent,
            Arc::new(gate),
            self.norm.clone(),
            epsilon,
            rows,
            &stream,
        )?
        .reshape(&[rows, VALUE_SIZE])
        .map_err(|error| ModelError::Cuda(format!("reshape Qwen GDN output: {error:?}")))?;
        self.output.enqueue(Arc::new(gated), rows, execution)
    }
}

impl FullAttention {
    fn device_bytes(&self) -> usize {
        self.key.device_bytes()
            + self.key_norm.num_bytes()
            + self.output.device_bytes()
            + self.query.device_bytes()
            + self.query_norm.num_bytes()
            + self.value.device_bytes()
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: Bf16Tensor,
        backend: &QwenFlatKvAttention,
        layer: usize,
        positions: Arc<Tensor<u32>>,
        current_slots: Arc<Tensor<u32>>,
        request_indices: Arc<Tensor<u32>>,
        context_slots: &TensorView<'_, u32>,
        context_lengths: Arc<Tensor<i32>>,
        rows: usize,
        epsilon: f32,
        execution: &mut StreamExecution<'_>,
    ) -> Result<Tensor<bf16>, ModelError> {
        let stream = execution.stream().clone();
        let q_gate = Arc::new(self.query.enqueue(hidden.clone(), rows, execution)?);
        let key = Arc::new(self.key.enqueue(hidden.clone(), rows, execution)?);
        let value = Arc::new(self.value.enqueue(hidden, rows, execution)?);
        let attention = backend.enqueue(AttentionInput {
            layer,
            q_gate,
            key,
            value,
            q_weight_delta: self.query_norm.clone(),
            k_weight_delta: self.key_norm.clone(),
            positions,
            current_slots,
            request_indices,
            context_slots,
            context_lengths,
            rows,
            epsilon,
            stream: &stream,
        })?;
        self.output.enqueue(Arc::new(attention), rows, execution)
    }
}

impl Moe {
    fn device_bytes(&self) -> usize {
        self.router.num_bytes()
            + self.routed_gate.device_bytes()
            + self.routed_up.device_bytes()
            + self.routed_down.device_bytes()
            + self.shared_gate.device_bytes()
            + self.shared_up.device_bytes()
            + self.shared_down.device_bytes()
            + self.shared_router.num_bytes()
    }

    fn forward(
        &self,
        hidden: Bf16Tensor,
        rows: usize,
        logical_rows: usize,
        execution: &mut StreamExecution<'_>,
    ) -> Result<Tensor<bf16>, ModelError> {
        const HIDDEN_SIZE: usize = 2048;
        if logical_rows == 0
            || logical_rows > rows
            || !rows.is_multiple_of(16)
            || hidden.shape() != [rows as i32, HIDDEN_SIZE as i32]
        {
            return Err(ModelError::Cuda("invalid Qwen MoE input geometry".into()));
        }

        let stream = execution.stream().clone();
        let router_logits = bf16_gemm(
            self.router.clone(),
            hidden.clone(),
            256,
            rows,
            HIDDEN_SIZE,
            "Qwen routed-expert logits",
            &stream,
        )?;
        let routing = RoutingPlan::build(router_logits, logical_rows, execution)?;
        let dispatched = routing.dispatch(hidden.clone(), logical_rows, HIDDEN_SIZE, execution)?;
        let dispatched_rows = routing.max_dispatched_rows;
        let routed_gate = self.routed_gate.enqueue_device_plan(
            dispatched.hidden.clone(),
            dispatched_rows,
            dispatched.expert_by_row_tile.clone(),
            execution,
        )?;
        let routed_up = self.routed_up.enqueue_device_plan(
            dispatched.hidden,
            dispatched_rows,
            dispatched.expert_by_row_tile.clone(),
            execution,
        )?;
        let routed_activated = silu_mul(
            Arc::new(routed_gate),
            Arc::new(routed_up),
            dispatched_rows,
            self.routed_gate.output_size(),
            &stream,
        )?;
        let routed_down = self.routed_down.enqueue_device_plan(
            Arc::new(routed_activated),
            dispatched_rows,
            dispatched.expert_by_row_tile,
            execution,
        )?;
        let routed = routing.combine(
            Arc::new(routed_down),
            logical_rows,
            rows,
            HIDDEN_SIZE,
            execution,
        )?;

        let shared_gate = self.shared_gate.enqueue(hidden.clone(), rows, execution)?;
        let shared_up = self.shared_up.enqueue(hidden.clone(), rows, execution)?;
        let shared_activated = silu_mul(
            Arc::new(shared_gate),
            Arc::new(shared_up),
            rows,
            self.shared_gate.output_size(),
            &stream,
        )?;
        let shared = self
            .shared_down
            .enqueue(Arc::new(shared_activated), rows, execution)?;
        let shared_logits = bf16_gemm(
            self.shared_router.clone(),
            hidden,
            1,
            rows,
            HIDDEN_SIZE,
            "Qwen shared-expert gate",
            &stream,
        )?;
        moe_backend::combine_shared(
            Arc::new(routed),
            Arc::new(shared),
            shared_logits,
            rows,
            HIDDEN_SIZE,
            execution,
        )
    }
}

fn bf16_gemm(
    weight: Bf16Tensor,
    input: Bf16Tensor,
    output_size: usize,
    rows: usize,
    input_size: usize,
    operation: &'static str,
    stream: &Arc<Stream>,
) -> Result<Bf16Tensor, ModelError> {
    let output = api::zeros::<bf16>(&[rows, output_size])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate {operation} output: {error:?}")))?;
    let output = cublas::gemm_bf16(weight, input, output, output_size, rows, input_size)
        .map_err(|error| ModelError::Cuda(format!("prepare {operation}: {error}")))?
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("schedule {operation}: {error:?}")))?
        .map_err(|error| ModelError::Cuda(format!("execute {operation}: {error}")))?;
    Ok(Arc::new(output))
}

fn silu_mul(
    gate: Bf16Tensor,
    up: Bf16Tensor,
    rows: usize,
    width: usize,
    stream: &Arc<Stream>,
) -> Result<Tensor<bf16>, ModelError> {
    const BLOCK: usize = 256;
    if width == 0
        || !width.is_multiple_of(BLOCK)
        || gate.shape() != [rows as i32, width as i32]
        || up.shape() != gate.shape()
    {
        return Err(ModelError::Cuda("invalid Qwen SiLU gate geometry".into()));
    }
    let mut output = api::zeros::<bf16>(&[rows, width])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate Qwen SiLU output: {error:?}")))?;
    let (_, _, output_partition) =
        kernels::silu_mul_bf16(gate, up, (&mut output).partition([1, BLOCK]))
            .generics(vec![BLOCK.to_string()])
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("execute Qwen SiLU gate: {error:?}")))?;
    drop(output_partition);
    Ok(output)
}

fn gemma_rms_norm(
    input: Bf16Tensor,
    weight_delta: Bf16Tensor,
    rows: usize,
    epsilon: f32,
    stream: &Arc<Stream>,
) -> Result<Bf16Tensor, ModelError> {
    const HIDDEN_SIZE: usize = 2048;
    const BLOCK: usize = 256;
    if input.shape() != [rows as i32, HIDDEN_SIZE as i32]
        || weight_delta.shape() != [HIDDEN_SIZE as i32]
    {
        return Err(ModelError::Cuda(
            "invalid Qwen Gemma RMSNorm geometry".into(),
        ));
    }
    let mut output = api::zeros::<bf16>(&[rows, HIDDEN_SIZE])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate Qwen normalized output: {error:?}")))?;
    let (_, _, output_partition, _) = unsafe {
        kernels::gemma_rms_norm_bf16(
            input,
            weight_delta,
            (&mut output).partition([1, HIDDEN_SIZE]),
            epsilon,
        )
    }
    .generics(vec![HIDDEN_SIZE.to_string(), BLOCK.to_string()])
    .sync_on(stream)
    .map_err(|error| ModelError::Cuda(format!("execute Qwen Gemma RMSNorm: {error:?}")))?;
    drop(output_partition);
    Ok(Arc::new(output))
}

fn gemma_add_rms_norm(
    residual: Bf16Tensor,
    update: Bf16Tensor,
    weight_delta: Bf16Tensor,
    rows: usize,
    epsilon: f32,
    stream: &Arc<Stream>,
) -> Result<(Bf16Tensor, Bf16Tensor), ModelError> {
    const HIDDEN_SIZE: usize = 2048;
    const BLOCK: usize = 256;
    if residual.shape() != [rows as i32, HIDDEN_SIZE as i32]
        || update.shape() != residual.shape()
        || weight_delta.shape() != [HIDDEN_SIZE as i32]
    {
        return Err(ModelError::Cuda(
            "invalid Qwen fused add Gemma RMSNorm geometry".into(),
        ));
    }
    let mut normalized = api::zeros::<bf16>(&[rows, HIDDEN_SIZE])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate Qwen normalized output: {error:?}")))?;
    let mut combined = api::zeros::<bf16>(&[rows, HIDDEN_SIZE])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate Qwen residual output: {error:?}")))?;
    let (_, _, _, normalized_partition, combined_partition, _) = unsafe {
        kernels::gemma_add_rms_norm_bf16(
            residual,
            update,
            weight_delta,
            (&mut normalized).partition([1, HIDDEN_SIZE]),
            (&mut combined).partition([1, HIDDEN_SIZE]),
            epsilon,
        )
    }
    .generics(vec![HIDDEN_SIZE.to_string(), BLOCK.to_string()])
    .sync_on(stream)
    .map_err(|error| {
        ModelError::Cuda(format!("execute Qwen fused add Gemma RMSNorm: {error:?}"))
    })?;
    drop(normalized_partition);
    drop(combined_partition);
    Ok((Arc::new(normalized), Arc::new(combined)))
}

pub(crate) fn checkpoint_report(
    model_id: &str,
    artifact: Artifact,
    device_id: usize,
) -> Result<CudaModelReport, ModelError> {
    let device = Device::new(device_id)
        .map_err(|error| ModelError::Cuda(format!("initialize device {device_id}: {error:?}")))?;
    let stream = device
        .new_stream()
        .map_err(|error| ModelError::Cuda(format!("create stream: {error:?}")))?;
    let checkpoint = Checkpoint::load(&artifact, &stream)?;
    let bytes = checkpoint.device_bytes();
    drop(checkpoint);
    Ok(CudaModelReport {
        model_id: model_id.into(),
        device_id,
        tensors: artifact.weights.tensor_count(),
        bytes,
    })
}

fn load_bf16(
    source: &dyn WeightSource,
    name: &str,
    stream: &Arc<Stream>,
) -> Result<Bf16Tensor, ModelError> {
    let shape = source.tensor(name)?.shape().to_vec();
    load_bf16_as(source, name, &shape, stream)
}

fn load_bf16_as(
    source: &dyn WeightSource,
    name: &str,
    shape: &[usize],
    stream: &Arc<Stream>,
) -> Result<Bf16Tensor, ModelError> {
    let tensor = source.tensor(name)?;
    if tensor.dtype() != &WeightDtype::Bf16 {
        return Err(ModelError::WrongDtype {
            name: name.into(),
            expected: WeightDtype::Bf16.to_string(),
            actual: tensor.dtype().to_string(),
        });
    }
    let host = Arc::new(
        tensor
            .bytes()
            .chunks_exact(2)
            .map(|bytes| bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect::<Vec<_>>(),
    );
    let device = api::copy_host_vec_to_device(&host)
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("upload `{name}`: {error:?}")))?
        .reshape(shape)
        .map_err(|error| ModelError::Cuda(format!("reshape `{name}`: {error:?}")))?;
    Ok(Arc::new(device))
}

fn load_f32(
    source: &dyn WeightSource,
    name: &str,
    stream: &Arc<Stream>,
) -> Result<F32Tensor, ModelError> {
    let tensor = source.tensor(name)?;
    let host = Arc::new(match tensor.dtype() {
        WeightDtype::F32 => tensor
            .bytes()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect::<Vec<_>>(),
        WeightDtype::Bf16 => tensor
            .bytes()
            .chunks_exact(2)
            .map(|bytes| bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
            .collect::<Vec<_>>(),
        actual => {
            return Err(ModelError::WrongDtype {
                name: name.into(),
                expected: format!("{} or {}", WeightDtype::F32, WeightDtype::Bf16),
                actual: actual.to_string(),
            });
        }
    });
    let device = api::copy_host_vec_to_device(&host)
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("upload `{name}`: {error:?}")))?
        .reshape(tensor.shape())
        .map_err(|error| ModelError::Cuda(format!("reshape `{name}`: {error:?}")))?;
    Ok(Arc::new(device))
}

fn upload_u32(
    values: &[u32],
    stream: &Arc<Stream>,
    name: &str,
) -> Result<Arc<Tensor<u32>>, ModelError> {
    api::copy_host_vec_to_device(&Arc::new(values.to_vec()))
        .sync_on(stream)
        .map(Arc::new)
        .map_err(|error| ModelError::Cuda(format!("upload {name}: {error:?}")))
}

fn upload_i32_named(
    values: &[i32],
    stream: &Arc<Stream>,
    name: &str,
) -> Result<Arc<Tensor<i32>>, ModelError> {
    api::copy_host_vec_to_device(&Arc::new(values.to_vec()))
        .sync_on(stream)
        .map(Arc::new)
        .map_err(|error| ModelError::Cuda(format!("upload {name}: {error:?}")))
}
