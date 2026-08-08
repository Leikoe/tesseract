use std::{
    collections::{HashMap, HashSet},
    f32::consts::TAU,
    sync::Arc,
    time::Instant,
};

use cuda_async::{
    cuda_graph::{CudaGraph, Scope},
    device_operation::DeviceOp,
    error::DeviceError,
};
use cuda_core::{Device, Stream};
use cutile::{
    api,
    core::bf16,
    tensor::{IntoPartition, PartitionMut, Reshape, Tensor, ToHostVec},
    tile_kernel::TileKernel,
};

use crate::{
    cuda::{
        attention::{AttentionBackend, DecodeGraphAttention, EagerAttention},
        batch::CudaBatch,
        cublas,
        executor::{CudaExecutor, ModelProgram, ProgramOutput},
        kernel_plan::{
            AttentionKernelPlan, ComputeCapability, DecoderKernelRequirement, KernelCatalog,
            KernelPlan,
        },
        kernels,
    },
    engine::{ExecutionError, ExecutionStats, ModelExecutor, TokenId},
    model::{
        CudaForwardReport, CudaModelReport, CudaTokenLogit, Model, ModelError, weights::WeightStore,
    },
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Llama3RopeConfig {
    pub factor: f32,
    pub high_frequency_factor: f32,
    pub low_frequency_factor: f32,
    pub original_max_positions: usize,
    pub theta: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DenseDecoderConfig {
    pub bos_token_id: u32,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope: Llama3RopeConfig,
    pub vocab_size: usize,
}

impl DenseDecoderConfig {
    fn q_width(self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    fn kv_width(self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DenseLayerWeightNames {
    pub input_norm: String,
    pub post_norm: String,
    pub query: String,
    pub key: String,
    pub value: String,
    pub output: String,
    pub gate: String,
    pub up: String,
    pub down: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DenseDecoderWeightNames {
    pub embedding: String,
    pub final_norm: String,
    pub lm_head: Option<String>,
    pub layers: Vec<DenseLayerWeightNames>,
}

pub(crate) struct DenseDecoderArtifact {
    model: Arc<dyn Model>,
    config: DenseDecoderConfig,
    weights: Arc<WeightStore>,
    weight_names: DenseDecoderWeightNames,
}

impl DenseDecoderArtifact {
    pub fn try_new(
        model: Arc<dyn Model>,
        config: DenseDecoderConfig,
        weights: Arc<WeightStore>,
        weight_names: DenseDecoderWeightNames,
    ) -> Result<Self, ModelError> {
        if !valid_dense_decoder_config(config) {
            return Err(ModelError::InvalidConfig(
                "invalid dense decoder geometry or RoPE contract".into(),
            ));
        }
        if weight_names.layers.len() != config.num_hidden_layers {
            return Err(ModelError::InvalidConfig(format!(
                "dense decoder has {} layer weight mappings for {} configured layers",
                weight_names.layers.len(),
                config.num_hidden_layers
            )));
        }
        Ok(Self {
            model,
            config,
            weights,
            weight_names,
        })
    }
}

fn valid_dense_decoder_config(config: DenseDecoderConfig) -> bool {
    config.hidden_size > 0
        && config.intermediate_size > 0
        && config.head_dim > 0
        && config.head_dim.is_multiple_of(2)
        && config.num_attention_heads > 0
        && config.num_key_value_heads > 0
        && config
            .num_attention_heads
            .is_multiple_of(config.num_key_value_heads)
        && config.num_attention_heads.checked_mul(config.head_dim) == Some(config.hidden_size)
        && config.num_hidden_layers > 0
        && config.max_position_embeddings > 0
        && config.vocab_size > 0
        && (config.bos_token_id as usize) < config.vocab_size
        && config.rms_norm_eps.is_finite()
        && config.rms_norm_eps > 0.0
        && config.rope.factor.is_finite()
        && config.rope.factor > 0.0
        && config.rope.high_frequency_factor.is_finite()
        && config.rope.high_frequency_factor > 0.0
        && config.rope.low_frequency_factor.is_finite()
        && config.rope.low_frequency_factor > 0.0
        && config.rope.original_max_positions > 0
        && config.rope.original_max_positions <= config.max_position_embeddings
        && config.rope.theta.is_finite()
        && config.rope.theta > 0.0
}

type Bf16Tensor = Arc<Tensor<bf16>>;
type F32Tensor = Arc<Tensor<f32>>;
type RopeTables = (F32Tensor, F32Tensor);

struct DeviceWeights {
    tensors: HashMap<String, Bf16Tensor>,
    bytes: usize,
}

impl DeviceWeights {
    fn load(store: &WeightStore, stream: &Arc<Stream>) -> Result<Self, ModelError> {
        let names = store.names();
        let mut tensors = HashMap::with_capacity(names.len());
        let mut bytes = 0usize;
        for name in names {
            let tensor = store.load_device_bf16(&name, stream)?;
            bytes = bytes
                .checked_add(tensor.num_bytes())
                .ok_or_else(|| ModelError::Cuda("device weight byte count overflowed".into()))?;
            tensors.insert(name, Arc::new(tensor));
        }
        Ok(Self { tensors, bytes })
    }

    fn get(&self, name: &str) -> Result<Bf16Tensor, ModelError> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| ModelError::MissingTensor(name.into()))
    }
}

struct LayerWeights {
    input_norm: Bf16Tensor,
    post_norm: Bf16Tensor,
    query: Bf16Tensor,
    key: Bf16Tensor,
    value: Bf16Tensor,
    output: Bf16Tensor,
    gate: Bf16Tensor,
    up: Bf16Tensor,
    down: Bf16Tensor,
}

struct RuntimeWeights {
    embedding: Bf16Tensor,
    final_norm: Bf16Tensor,
    lm_head: Bf16Tensor,
    layers: Vec<LayerWeights>,
}

impl RuntimeWeights {
    fn load(artifact: &DenseDecoderArtifact, stream: &Arc<Stream>) -> Result<Self, ModelError> {
        let all = DeviceWeights::load(&artifact.weights, stream)?;
        let names = &artifact.weight_names;
        let embedding = all.get(&names.embedding)?;
        let final_norm = all.get(&names.final_norm)?;
        let lm_head = match &names.lm_head {
            Some(name) => all.get(name)?,
            None => embedding.clone(),
        };
        let mut layers = Vec::with_capacity(names.layers.len());
        for layer in &names.layers {
            layers.push(LayerWeights {
                input_norm: all.get(&layer.input_norm)?,
                post_norm: all.get(&layer.post_norm)?,
                query: all.get(&layer.query)?,
                key: all.get(&layer.key)?,
                value: all.get(&layer.value)?,
                output: all.get(&layer.output)?,
                gate: all.get(&layer.gate)?,
                up: all.get(&layer.up)?,
                down: all.get(&layer.down)?,
            });
        }
        Ok(Self {
            embedding,
            final_norm,
            lm_head,
            layers,
        })
    }
}

struct DenseDecoder<A> {
    model: Arc<dyn Model>,
    config: DenseDecoderConfig,
    stream: Arc<Stream>,
    weights: RuntimeWeights,
    attention: A,
    kernel_plan: KernelPlan,
    capacity: usize,
    max_decode_batch_bucket: usize,
    max_query_bucket: usize,
    decode_graphs: HashMap<(usize, usize), DecodeGraph>,
    failed_decode_graphs: HashSet<(usize, usize)>,
    execution_stats: ExecutionStats,
}

struct FlatKvLayerState {
    key_cache: Bf16Tensor,
    value_cache: Bf16Tensor,
}

struct DirectFlatKvAttention {
    layers: Vec<FlatKvLayerState>,
    cosine: Arc<Tensor<f32>>,
    sine: Arc<Tensor<f32>>,
    num_attention_heads: usize,
    plan: AttentionKernelPlan,
}

enum ForwardOutput {
    None,
    Logits(Vec<f32>),
    BatchLogits(Vec<f32>),
    Token(u32),
    Tokens(Vec<u32>),
}

struct EagerBatch<'a> {
    token_ids: &'a [u32],
    positions: &'a [u32],
    current_slots: &'a [u32],
    contexts: &'a [Vec<u32>],
    request_indices: &'a [u32],
    context_lengths: &'a [i32],
    sample_rows: &'a [u32],
    greedy: bool,
}

impl EagerBatch<'_> {
    fn validate(&self, capacity: usize) -> Result<(), ModelError> {
        let rows = self.token_ids.len();
        let valid = rows > 0
            && self.positions.len() == rows
            && self.current_slots.len() == rows
            && self.request_indices.len() == rows
            && self.context_lengths.len() == rows
            && !self.contexts.is_empty()
            && self
                .current_slots
                .iter()
                .all(|slot| (*slot as usize) < capacity)
            && self.contexts.iter().all(|context| {
                !context.is_empty() && context.iter().all(|slot| (*slot as usize) < capacity)
            })
            && self
                .request_indices
                .iter()
                .all(|index| (*index as usize) < self.contexts.len())
            && self.context_lengths.iter().all(|length| *length > 0)
            && self
                .request_indices
                .iter()
                .zip(self.context_lengths)
                .all(|(index, length)| *length as usize <= self.contexts[*index as usize].len())
            && self.sample_rows.iter().all(|row| (*row as usize) < rows)
            && self.sample_rows.windows(2).all(|pair| pair[0] < pair[1]);
        if valid {
            Ok(())
        } else {
            Err(ModelError::Cuda("invalid forward batch metadata".into()))
        }
    }
}

impl DirectFlatKvAttention {
    fn load(
        config: &DenseDecoderConfig,
        stream: &Arc<Stream>,
        capacity: usize,
        scratch_slots: usize,
        cosine: Arc<Tensor<f32>>,
        sine: Arc<Tensor<f32>>,
        plan: AttentionKernelPlan,
    ) -> Result<Self, ModelError> {
        let cache_shape = [
            capacity
                .checked_add(1)
                .and_then(|slots| slots.checked_add(scratch_slots))
                .ok_or_else(|| {
                    ModelError::Cuda("KV cache sentinel allocation overflowed".into())
                })?,
            config.num_key_value_heads,
            config.head_dim,
        ];
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let key_cache = api::zeros::<bf16>(&cache_shape)
                .sync_on(stream)
                .map_err(|error| {
                    ModelError::Cuda(format!("allocate layer {layer} key cache: {error:?}"))
                })?;
            let value_cache =
                api::zeros::<bf16>(&cache_shape)
                    .sync_on(stream)
                    .map_err(|error| {
                        ModelError::Cuda(format!("allocate layer {layer} value cache: {error:?}"))
                    })?;
            layers.push(FlatKvLayerState {
                key_cache: Arc::new(key_cache),
                value_cache: Arc::new(value_cache),
            });
        }
        Ok(Self {
            layers,
            cosine,
            sine,
            num_attention_heads: config.num_attention_heads,
            plan,
        })
    }
}

impl AttentionBackend for DirectFlatKvAttention {
    type LayerState = FlatKvLayerState;
    type Error = ModelError;

    fn layer_state(&self, layer: usize) -> Result<&Self::LayerState, Self::Error> {
        self.layers
            .get(layer)
            .ok_or_else(|| ModelError::Cuda(format!("attention layer {layer} is out of range")))
    }

    fn enqueue_eager(&self, input: EagerAttention<'_>) -> Result<Tensor<bf16>, Self::Error> {
        let state = self.layer_state(input.layer)?;
        let head_dim = self.plan.head_dim.get();
        let rotated_query = output_buffer_async::<bf16>(
            &[input.rows, self.num_attention_heads, head_dim],
            input.stream,
            "rotated query",
        )?
        .partition([1, 1, head_dim]);
        let (_, _, _, _, rotated_query) = enqueue(
            kernels::rope_q_bf16(
                input.query,
                input.positions,
                &self.cosine,
                &self.sine,
                rotated_query,
            )
            .generics(vec![
                head_dim.to_string(),
                self.plan.rope.rotary_dim.get().to_string(),
            ]),
            input.stream,
            "query RoPE",
        )?;
        let rotated_query = Arc::new(rotated_query.unpartition());

        let rotated_key = output_buffer_async::<bf16>(
            &[input.rows, self.plan.kv_heads.get(), head_dim],
            input.stream,
            "rotated key",
        )?
        .partition([1, 1, head_dim]);
        let (_, _, _, _, _, _, _, _, _rotated_key) = enqueue(
            unsafe {
                kernels::rope_kv_write_bf16(
                    input.key,
                    input.value,
                    input.positions,
                    input.current_slots,
                    &self.cosine,
                    &self.sine,
                    state.key_cache.device_pointer(),
                    state.value_cache.device_pointer(),
                    rotated_key,
                )
            }
            .generics(vec![
                head_dim.to_string(),
                self.plan.rope.rotary_dim.get().to_string(),
                self.plan.kv_heads.get().to_string(),
            ]),
            input.stream,
            "key RoPE and flat KV write",
        )?;

        let attention = output_buffer_async::<bf16>(
            &[input.rows, self.num_attention_heads, head_dim],
            input.stream,
            "ragged attention output",
        )?
        .partition([1, 1, head_dim]);
        let (_, _, _, _, _, _, attention, _, _) = enqueue(
            unsafe {
                kernels::ragged_attention_bf16(
                    &rotated_query,
                    input.request_indices,
                    input.context_slots,
                    input.context_lengths,
                    state.key_cache.device_pointer(),
                    state.value_cache.device_pointer(),
                    attention,
                    1.0 / (head_dim as f32).sqrt(),
                    self.plan.query_heads_per_kv.get() as i32,
                )
            }
            .generics(vec![
                self.plan.prefill_mixed.key_block.get().to_string(),
                head_dim.to_string(),
                self.plan.kv_heads.get().to_string(),
            ]),
            input.stream,
            "ragged flat-KV attention",
        )?;
        Ok(attention.unpartition())
    }

    fn record_decode(&self, input: DecodeGraphAttention<'_>) -> Result<(), Self::Error> {
        let state = self.layer_state(input.layer)?;
        let head_dim = self.plan.head_dim.get();
        input
            .scope
            .record(
                kernels::rope_q_bf16(
                    input.query,
                    input.positions,
                    &self.cosine,
                    &self.sine,
                    input.rotated_query.partition([1, 1, head_dim]),
                )
                .generics(vec![
                    head_dim.to_string(),
                    self.plan.rope.rotary_dim.get().to_string(),
                ]),
            )
            .map_err(|error| cuda_error("record query RoPE", error))?;
        input
            .scope
            .record(
                unsafe {
                    kernels::rope_kv_write_bf16(
                        input.key,
                        input.value,
                        input.positions,
                        input.current_slots,
                        &self.cosine,
                        &self.sine,
                        state.key_cache.device_pointer(),
                        state.value_cache.device_pointer(),
                        input.rotated_key.partition([1, 1, head_dim]),
                    )
                }
                .generics(vec![
                    head_dim.to_string(),
                    self.plan.rope.rotary_dim.get().to_string(),
                    self.plan.kv_heads.get().to_string(),
                ]),
            )
            .map_err(|error| cuda_error("record key RoPE and flat KV write", error))?;
        input
            .scope
            .record(
                unsafe {
                    kernels::ragged_attention_bf16(
                        &*input.rotated_query,
                        input.request_indices,
                        input.context_slots,
                        input.context_lengths,
                        state.key_cache.device_pointer(),
                        state.value_cache.device_pointer(),
                        input.attention.partition([1, 1, head_dim]),
                        1.0 / (head_dim as f32).sqrt(),
                        self.plan.query_heads_per_kv.get() as i32,
                    )
                }
                .generics(vec![
                    self.plan.decode.key_block.get().to_string(),
                    head_dim.to_string(),
                    self.plan.kv_heads.get().to_string(),
                ]),
            )
            .map_err(|error| cuda_error("record ragged flat-KV attention", error))?;
        Ok(())
    }
}

impl DenseDecoder<DirectFlatKvAttention> {
    fn load(
        artifact: DenseDecoderArtifact,
        device_id: usize,
        capacity: usize,
        max_running: usize,
        max_batch_tokens: usize,
    ) -> Result<Self, ModelError> {
        let DenseDecoderArtifact {
            model,
            config,
            weights,
            weight_names,
        } = artifact;
        let artifact = DenseDecoderArtifact {
            model: model.clone(),
            config,
            weights,
            weight_names,
        };
        let compute_capability = ComputeCapability::detect(device_id)
            .map_err(|error| ModelError::Cuda(error.to_string()))?;
        let kernel_plan = KernelCatalog
            .resolve(
                compute_capability,
                DecoderKernelRequirement {
                    hidden_size: config.hidden_size,
                    intermediate_size: config.intermediate_size,
                    head_dim: config.head_dim,
                    attention_heads: config.num_attention_heads,
                    kv_heads: config.num_key_value_heads,
                    vocab_size: config.vocab_size,
                },
            )
            .map_err(|error| ModelError::Cuda(error.to_string()))?;
        tracing::info!(plan = %kernel_plan.diagnostic_summary(), "resolved CUDA kernel plan");
        let device = Device::new(device_id).map_err(|error| {
            ModelError::Cuda(format!("initialize device {device_id}: {error:?}"))
        })?;
        let stream = device
            .new_stream()
            .map_err(|error| ModelError::Cuda(format!("create stream: {error:?}")))?;
        let runtime_weights = RuntimeWeights::load(&artifact, &stream)?;
        let (cosine, sine) = rope_tables(&config, &stream)?;
        let max_decode_batch_bucket = kernel_plan
            .shapes
            .query_bucket(max_running)
            .ok_or_else(|| ModelError::Cuda("max running batch bucket overflowed".into()))?;
        let max_query_bucket = kernel_plan
            .shapes
            .query_bucket(max_batch_tokens)
            .ok_or_else(|| ModelError::Cuda("max query bucket overflowed".into()))?;
        let attention = DirectFlatKvAttention::load(
            &config,
            &stream,
            capacity,
            max_decode_batch_bucket.max(max_query_bucket),
            cosine,
            sine,
            kernel_plan.attention,
        )?;
        Ok(Self {
            model,
            config,
            stream,
            weights: runtime_weights,
            attention,
            kernel_plan,
            capacity,
            max_decode_batch_bucket,
            max_query_bucket,
            decode_graphs: HashMap::new(),
            failed_decode_graphs: HashSet::new(),
            execution_stats: ExecutionStats::default(),
        })
    }
}

impl<A: AttentionBackend<Error = ModelError>> DenseDecoder<A> {
    fn forward(
        &mut self,
        token_ids: &[u32],
        positions: &[u32],
        current_slots: &[u32],
        context_slots: &[u32],
        return_logits: bool,
        greedy: bool,
    ) -> Result<ForwardOutput, ModelError> {
        if return_logits && token_ids.len() == 1 && positions[0] > 0 {
            let context_bucket = self
                .kernel_plan
                .shapes
                .context_bucket(context_slots.len())
                .ok_or_else(|| ModelError::Cuda("decode context bucket overflowed".into()))?;
            let key = (1, context_bucket);
            if !self.failed_decode_graphs.contains(&key) {
                match self.decode_graph(
                    token_ids[0],
                    positions[0],
                    current_slots[0],
                    context_slots,
                    greedy,
                ) {
                    Ok(output) => {
                        self.execution_stats.graph_replays += 1;
                        return Ok(output);
                    }
                    Err(error) => {
                        tracing::warn!(
                            batch_size = 1,
                            context_bucket = key.1,
                            %error,
                            "decode CUDA graph failed; using eager fallback"
                        );
                        self.failed_decode_graphs.insert(key);
                    }
                }
            }
        }
        let output = self.forward_eager(
            token_ids,
            positions,
            current_slots,
            context_slots,
            return_logits,
            greedy,
        );
        if output.is_ok() {
            self.execution_stats.eager_forwards += 1;
        }
        output
    }

    fn forward_eager(
        &self,
        token_ids: &[u32],
        positions: &[u32],
        current_slots: &[u32],
        context_slots: &[u32],
        return_logits: bool,
        greedy: bool,
    ) -> Result<ForwardOutput, ModelError> {
        let request_indices = vec![0; token_ids.len()];
        let context_lengths = positions
            .iter()
            .map(|position| {
                i32::try_from(*position)
                    .ok()
                    .and_then(|position| position.checked_add(1))
                    .ok_or_else(|| ModelError::Cuda("attention position exceeds i32".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let sample_rows = return_logits.then(|| vec![(token_ids.len() - 1) as u32]);
        self.forward_eager_impl(EagerBatch {
            token_ids,
            positions,
            current_slots,
            contexts: &[context_slots.to_vec()],
            request_indices: &request_indices,
            context_lengths: &context_lengths,
            sample_rows: sample_rows.as_deref().unwrap_or_default(),
            greedy,
        })
    }

    fn forward_decode_batch(
        &mut self,
        token_ids: &[u32],
        positions: &[u32],
        current_slots: &[u32],
        contexts: &[Vec<u32>],
    ) -> Result<Vec<u32>, ModelError> {
        let batch_bucket = self
            .kernel_plan
            .shapes
            .query_bucket(token_ids.len())
            .ok_or_else(|| ModelError::Cuda("decode batch bucket overflowed".into()))?;
        let context_bucket = self
            .kernel_plan
            .shapes
            .context_bucket(contexts.iter().map(Vec::len).max().unwrap_or(0))
            .ok_or_else(|| ModelError::Cuda("decode context bucket overflowed".into()))?;
        let key = (batch_bucket, context_bucket);
        let request_indices: Vec<u32> = (0..token_ids.len())
            .map(|index| {
                u32::try_from(index)
                    .map_err(|_| ModelError::Cuda("decode batch exceeds u32".into()))
            })
            .collect::<Result<_, _>>()?;
        let context_lengths: Vec<i32> = contexts
            .iter()
            .map(|context| {
                i32::try_from(context.len())
                    .map_err(|_| ModelError::Cuda("decode context exceeds i32".into()))
            })
            .collect::<Result<_, _>>()?;
        let sample_rows: Vec<u32> = request_indices.clone();
        let output = if self.failed_decode_graphs.contains(&key) {
            self.forward_eager_impl(EagerBatch {
                token_ids,
                positions,
                current_slots,
                contexts,
                request_indices: &request_indices,
                context_lengths: &context_lengths,
                sample_rows: &sample_rows,
                greedy: true,
            })?
        } else {
            match self.decode_graph_batch(token_ids, positions, current_slots, contexts, true) {
                Ok(output) => {
                    self.execution_stats.graph_replays += 1;
                    output
                }
                Err(error) => {
                    tracing::warn!(
                        batch_size = key.0,
                        context_bucket = key.1,
                        %error,
                        "packed decode CUDA graph failed; using eager fallback"
                    );
                    self.failed_decode_graphs.insert(key);
                    self.forward_eager_impl(EagerBatch {
                        token_ids,
                        positions,
                        current_slots,
                        contexts,
                        request_indices: &request_indices,
                        context_lengths: &context_lengths,
                        sample_rows: &sample_rows,
                        greedy: true,
                    })?
                }
            }
        };
        if self.failed_decode_graphs.contains(&key) {
            self.execution_stats.eager_forwards += 1;
        }
        match output {
            ForwardOutput::Tokens(tokens) => Ok(tokens),
            ForwardOutput::Token(token) => Ok(vec![token]),
            ForwardOutput::None | ForwardOutput::Logits(_) | ForwardOutput::BatchLogits(_) => Err(
                ModelError::Cuda("packed decode omitted sampled tokens".into()),
            ),
        }
    }

    fn forward_eager_impl(&self, batch: EagerBatch<'_>) -> Result<ForwardOutput, ModelError> {
        batch.validate(self.capacity)?;
        let EagerBatch {
            token_ids,
            positions,
            current_slots,
            contexts,
            request_indices,
            context_lengths,
            sample_rows,
            greedy,
        } = batch;
        let logical_rows = token_ids.len();
        let cfg = &self.config;
        let stream = &self.stream;
        let rows = self
            .kernel_plan
            .shapes
            .query_bucket(logical_rows)
            .ok_or_else(|| ModelError::Cuda("query token bucket size overflowed".into()))?;
        if rows > self.max_query_bucket {
            return Err(ModelError::Cuda(format!(
                "query bucket {rows} exceeds configured maximum {}",
                self.max_query_bucket
            )));
        }
        let mut token_ids = token_ids.to_vec();
        let mut positions = positions.to_vec();
        let mut current_slots = current_slots.to_vec();
        let mut request_indices = request_indices.to_vec();
        let mut context_lengths = context_lengths.to_vec();
        token_ids.resize(rows, cfg.bos_token_id);
        positions.resize(rows, 0);
        for padding_offset in 1..=rows - logical_rows {
            let padding_slot = self
                .capacity
                .checked_add(padding_offset)
                .and_then(|slot| u32::try_from(slot).ok())
                .ok_or_else(|| ModelError::Cuda("eager padding slot overflowed".into()))?;
            current_slots.push(padding_slot);
        }
        request_indices.resize(rows, 0);
        context_lengths.resize(rows, 1);
        let token_ids = copy_u32(&token_ids, stream, "token IDs")?;
        let positions = copy_u32(&positions, stream, "positions")?;
        let current_slots = copy_u32(&current_slots, stream, "current KV slots")?;
        let request_indices = copy_u32(&request_indices, stream, "request indices")?;
        let context_lengths = copy_i32(&context_lengths, stream, "query context lengths")?;
        let context_bucket = self
            .kernel_plan
            .shapes
            .context_bucket(self.capacity)
            .ok_or_else(|| ModelError::Cuda("context bucket size overflowed".into()))?;
        let request_bucket = self.max_decode_batch_bucket;
        if contexts.len() > request_bucket {
            return Err(ModelError::Cuda(format!(
                "eager batch has {} requests but its fixed metadata table supports {request_bucket}",
                contexts.len()
            )));
        }
        let context_elements = request_bucket
            .checked_mul(context_bucket)
            .ok_or_else(|| ModelError::Cuda("ragged context table size overflowed".into()))?;
        let mut padded = Vec::with_capacity(context_elements);
        for context in contexts {
            padded.extend_from_slice(context);
            padded.resize(
                padded.len() + context_bucket - context.len(),
                self.capacity as u32,
            );
        }
        padded.resize(context_elements, self.capacity as u32);
        let context_slots = copy_u32(&padded, stream, "ragged context KV slots")?;
        let context_slots = context_slots
            .view(&[request_bucket, context_bucket])
            .map_err(|error| cuda_error("view ragged context slots", error))?;
        let mut completion = StreamCompletionGuard::new(stream);

        let hidden =
            output_buffer_async::<bf16>(&[rows, cfg.hidden_size], stream, "embedding output")?
                .partition([1, self.kernel_plan.dense.embedding.block.get()]);
        let (_, _, hidden) = enqueue(
            kernels::embedding_bf16(&token_ids, &*self.weights.embedding, hidden).generics(vec![
                cfg.hidden_size.to_string(),
                self.kernel_plan.dense.embedding.block.get().to_string(),
            ]),
            stream,
            "embedding",
        )?;
        let hidden = Arc::new(hidden.unpartition());

        let mut pending: Option<(Bf16Tensor, Bf16Tensor)> = None;
        for (layer_index, layer) in self.weights.layers.iter().enumerate() {
            // Keep the previous layer's pair alive until this layer has
            // synchronized. The normalization launch only borrows it, and
            // cuTile frees the final Arc on a separate deallocator stream.
            let previous = pending.take();
            let (attention_input, residual) = match previous.as_ref() {
                None => (
                    self.rms_norm(&hidden, &layer.input_norm, rows)?,
                    hidden.clone(),
                ),
                Some((residual, update)) => {
                    let (normalized, combined) =
                        self.add_rms_norm(residual, update, &layer.input_norm, rows)?;
                    (normalized, combined)
                }
            };

            let query = self.gemm(
                &layer.query,
                &attention_input,
                cfg.q_width(),
                rows,
                cfg.hidden_size,
                "query projection",
            )?;
            let key = self.gemm(
                &layer.key,
                &attention_input,
                cfg.kv_width(),
                rows,
                cfg.hidden_size,
                "key projection",
            )?;
            let value = self.gemm(
                &layer.value,
                &attention_input,
                cfg.kv_width(),
                rows,
                cfg.hidden_size,
                "value projection",
            )?;
            let query = query
                .view(&[rows, cfg.num_attention_heads, cfg.head_dim])
                .map_err(|error| cuda_error("view query heads", error))?;
            let key = key
                .view(&[rows, cfg.num_key_value_heads, cfg.head_dim])
                .map_err(|error| cuda_error("view key heads", error))?;
            let value = value
                .view(&[rows, cfg.num_key_value_heads, cfg.head_dim])
                .map_err(|error| cuda_error("view value heads", error))?;

            let attention = self.attention.enqueue_eager(EagerAttention {
                layer: layer_index,
                query: &query,
                key: &key,
                value: &value,
                positions: &positions,
                current_slots: &current_slots,
                request_indices: &request_indices,
                context_slots: &context_slots,
                context_lengths: &context_lengths,
                rows,
                stream,
            })?;
            let attention = Arc::new(
                attention
                    .reshape(&[rows, cfg.hidden_size])
                    .map_err(|error| cuda_error("reshape attention output", error))?,
            );
            let attention_output = self.gemm(
                &layer.output,
                &attention,
                cfg.hidden_size,
                rows,
                cfg.hidden_size,
                "attention output projection",
            )?;
            let (mlp_input, hidden_after_attention) =
                self.add_rms_norm(&residual, &attention_output, &layer.post_norm, rows)?;
            let gate = self.gemm(
                &layer.gate,
                &mlp_input,
                cfg.intermediate_size,
                rows,
                cfg.hidden_size,
                "MLP gate projection",
            )?;
            let up = self.gemm(
                &layer.up,
                &mlp_input,
                cfg.intermediate_size,
                rows,
                cfg.hidden_size,
                "MLP up projection",
            )?;
            let activated = output_buffer_async::<bf16>(
                &[rows, cfg.intermediate_size],
                stream,
                "MLP activation",
            )?
            .partition([1, self.kernel_plan.dense.silu_mul.block.get()]);
            let (_, _, activated) = enqueue(
                kernels::silu_mul_bf16(&gate, &up, activated).generics(vec![
                    self.kernel_plan.dense.silu_mul.block.get().to_string(),
                ]),
                stream,
                "SiLU gated activation",
            )?;
            let activated = Arc::new(activated.unpartition());
            let down = self.gemm(
                &layer.down,
                &activated,
                cfg.hidden_size,
                rows,
                cfg.intermediate_size,
                "MLP down projection",
            )?;
            pending = Some((hidden_after_attention, down));
            // cuTile frees dropped tensors on a dedicated deallocator
            // stream. Drain this layer before its temporary projections go
            // out of scope; the retained residual/down pair feeds the next
            // layer. This collapses dozens of synchronization points into
            // one without permitting cross-stream lifetime races.
            synchronize_stream(stream, "complete transformer layer")?;
        }

        if sample_rows.is_empty() {
            completion.synchronize("complete ragged forward")?;
            return Ok(ForwardOutput::None);
        }

        let (residual, update) =
            pending.ok_or_else(|| ModelError::Cuda("model has no transformer layers".into()))?;
        let (final_hidden, _final_residual) =
            self.add_rms_norm(&residual, &update, &self.weights.final_norm, rows)?;
        let samples = sample_rows.len();
        let sample_bucket = self
            .kernel_plan
            .shapes
            .query_bucket(samples)
            .ok_or_else(|| ModelError::Cuda("sample row bucket size overflowed".into()))?;
        let mut padded_sample_rows = sample_rows.to_vec();
        padded_sample_rows.resize(sample_bucket, sample_rows[0]);
        let sample_rows_device = copy_u32(&padded_sample_rows, stream, "sample rows")?;
        let sampled_hidden = output_buffer_async::<bf16>(
            &[sample_bucket, cfg.hidden_size],
            stream,
            "sampled hidden states",
        )?
        .partition([1, self.kernel_plan.dense.gather_rows.block.get()]);
        let (_, _, sampled_hidden) = enqueue(
            unsafe {
                kernels::gather_rows_bf16(
                    final_hidden.device_pointer(),
                    &sample_rows_device,
                    sampled_hidden,
                )
            }
            .generics(vec![
                cfg.hidden_size.to_string(),
                self.kernel_plan.dense.gather_rows.block.get().to_string(),
            ]),
            stream,
            "gather sampled hidden states",
        )?;
        let sampled_hidden = Arc::new(sampled_hidden.unpartition());
        let logits = self.gemm(
            &self.weights.lm_head,
            &sampled_hidden,
            cfg.vocab_size,
            sample_bucket,
            cfg.hidden_size,
            "ragged language-model head",
        )?;
        if greedy {
            let argmax_block = self.kernel_plan.sampling.argmax.block.get();
            let argmax_blocks = cfg.vocab_size.div_ceil(argmax_block);
            let reduce_block = self
                .kernel_plan
                .shapes
                .query_bucket(argmax_blocks)
                .ok_or_else(|| ModelError::Cuda("argmax reduce bucket overflowed".into()))?;
            let block_max = output_buffer_async::<f32>(
                &[sample_bucket, argmax_blocks],
                stream,
                "ragged argmax maxima",
            )?
            .partition([1, 1]);
            let block_index = output_buffer_async::<u32>(
                &[sample_bucket, argmax_blocks],
                stream,
                "ragged argmax indices",
            )?
            .partition([1, 1]);
            let (_, block_max, block_index, _) = enqueue(
                kernels::argmax_blocks_batch_bf16(
                    &logits,
                    block_max,
                    block_index,
                    cfg.vocab_size as i32,
                )
                .generics(vec![argmax_block.to_string()]),
                stream,
                "ragged logits argmax blocks",
            )?;
            let block_max = block_max.unpartition();
            let block_index = block_index.unpartition();
            let sampled =
                output_buffer_async::<u32>(&[sample_bucket], stream, "ragged sampled tokens")?
                    .partition([1]);
            let (_, _, sampled, _) = enqueue(
                kernels::argmax_reduce_batch_bf16(
                    &block_max,
                    &block_index,
                    sampled,
                    argmax_blocks as i32,
                )
                .generics(vec![reduce_block.to_string()]),
                stream,
                "reduce ragged logits argmax",
            )?;
            let sampled = Arc::new(sampled.unpartition());
            let mut sampled = sampled
                .clone()
                .to_host_vec()
                .sync_on(stream)
                .map_err(|error| cuda_error("copy ragged sampled tokens to host", error))?;
            completion.mark_synchronized();
            sampled.truncate(samples);
            return if samples == 1 {
                Ok(ForwardOutput::Token(sampled[0]))
            } else {
                Ok(ForwardOutput::Tokens(sampled))
            };
        }
        let logits: Vec<bf16> = logits
            .clone()
            .to_host_vec()
            .sync_on(stream)
            .map_err(|error| cuda_error("copy logits to host", error))?;
        completion.mark_synchronized();
        let mut logits: Vec<f32> = logits.into_iter().map(bf16::to_f32).collect();
        logits.truncate(samples * cfg.vocab_size);
        if samples == 1 {
            Ok(ForwardOutput::Logits(logits))
        } else {
            Ok(ForwardOutput::BatchLogits(logits))
        }
    }

    fn decode_graph(
        &mut self,
        token_id: u32,
        position: u32,
        current_slot: u32,
        context_slots: &[u32],
        greedy: bool,
    ) -> Result<ForwardOutput, ModelError> {
        self.decode_graph_batch(
            &[token_id],
            &[position],
            &[current_slot],
            &[context_slots.to_vec()],
            greedy,
        )
    }

    fn decode_graph_batch(
        &mut self,
        token_ids: &[u32],
        positions: &[u32],
        current_slots: &[u32],
        contexts: &[Vec<u32>],
        greedy: bool,
    ) -> Result<ForwardOutput, ModelError> {
        let batch_size = token_ids.len();
        if batch_size == 0
            || positions.len() != batch_size
            || current_slots.len() != batch_size
            || contexts.len() != batch_size
            || current_slots
                .iter()
                .any(|slot| *slot as usize >= self.capacity)
            || contexts.iter().any(|context| {
                context.is_empty() || context.iter().any(|slot| *slot as usize >= self.capacity)
            })
        {
            return Err(ModelError::Cuda("invalid decode graph metadata".into()));
        }
        let batch_bucket = self
            .kernel_plan
            .shapes
            .query_bucket(batch_size)
            .filter(|bucket| *bucket <= self.max_decode_batch_bucket)
            .ok_or_else(|| ModelError::Cuda("decode graph batch bucket overflowed".into()))?;
        let context_bucket = self
            .kernel_plan
            .shapes
            .context_bucket(contexts.iter().map(Vec::len).max().unwrap_or(0))
            .ok_or_else(|| ModelError::Cuda("decode graph context bucket overflowed".into()))?;
        let key = (batch_bucket, context_bucket);
        if !self.decode_graphs.contains_key(&key) {
            let graph = DecodeGraph::capture(self, batch_bucket, context_bucket)?;
            self.decode_graphs.insert(key, graph);
            self.execution_stats.graph_captures += 1;
        }
        let graph = self
            .decode_graphs
            .get_mut(&key)
            .ok_or_else(|| ModelError::Cuda("decode graph cache insertion failed".into()))?;
        graph.forward(
            token_ids,
            positions,
            current_slots,
            contexts,
            self.capacity as u32,
            greedy,
        )
    }

    fn warm_execution_buckets(
        &mut self,
        max_batch_tokens: usize,
        max_running: usize,
    ) -> Result<(), ModelError> {
        if self.capacity == 0 || max_batch_tokens == 0 || max_running == 0 {
            return Err(ModelError::Cuda(
                "KV capacity and execution limits must be positive before warmup".into(),
            ));
        }
        let max_logical_queries = max_batch_tokens
            .min(self.capacity)
            .min(self.config.max_position_embeddings);
        let max_logical_samples = max_running.min(self.capacity);
        let max_sample_bucket = self
            .kernel_plan
            .shapes
            .query_bucket(max_logical_samples)
            .ok_or_else(|| ModelError::Cuda("sample warmup bucket overflowed".into()))?;
        let decode_batch_warmup_sizes = warmup_logical_sizes(max_logical_samples, 1)
            .ok_or_else(|| ModelError::Cuda("decode batch warmup sizes overflowed".into()))?;
        let query_warmup_sizes = warmup_logical_sizes(max_logical_queries, 1)
            .ok_or_else(|| ModelError::Cuda("query warmup sizes overflowed".into()))?;
        for logical_queries in query_warmup_sizes {
            let query_bucket = self
                .kernel_plan
                .shapes
                .query_bucket(logical_queries)
                .ok_or_else(|| ModelError::Cuda("query warmup bucket overflowed".into()))?;
            let token_ids = vec![self.config.bos_token_id; logical_queries];
            let positions = vec![0; logical_queries];
            let slots: Vec<u32> = (0..logical_queries)
                .map(|slot| {
                    u32::try_from(slot)
                        .map_err(|_| ModelError::Cuda("warmup slot exceeds u32".into()))
                })
                .collect::<Result<_, _>>()?;
            let warm_requests = logical_queries.min(max_logical_samples);
            let contexts: Vec<Vec<u32>> = (0..warm_requests)
                .map(|request| vec![slots[request]])
                .collect();
            let request_indices: Vec<u32> = (0..logical_queries)
                .map(|row| {
                    u32::try_from(row % warm_requests)
                        .map_err(|_| ModelError::Cuda("warmup request index exceeds u32".into()))
                })
                .collect::<Result<_, _>>()?;
            let context_lengths = vec![1; logical_queries];
            let sample_count = if query_bucket <= max_sample_bucket {
                logical_queries.min(max_logical_samples)
            } else {
                1
            };
            let sample_rows: Vec<u32> = (0..sample_count)
                .map(|row| {
                    u32::try_from(row)
                        .map_err(|_| ModelError::Cuda("warmup sample row exceeds u32".into()))
                })
                .collect::<Result<_, _>>()?;
            self.forward_eager_impl(EagerBatch {
                token_ids: &token_ids,
                positions: &positions,
                current_slots: &slots,
                contexts: &contexts,
                request_indices: &request_indices,
                context_lengths: &context_lengths,
                sample_rows: &sample_rows,
                greedy: true,
            })?;
        }

        let context_warmup_sizes = warmup_logical_sizes(self.capacity, 16)
            .ok_or_else(|| ModelError::Cuda("context warmup sizes overflowed".into()))?;
        for logical_context in context_warmup_sizes {
            let bucket = self
                .kernel_plan
                .shapes
                .context_bucket(logical_context)
                .ok_or_else(|| ModelError::Cuda("context warmup bucket overflowed".into()))?;
            // Compile every shape and initialize the cuBLAS handle before
            // capture. Repeating slot zero is safe during warmup and is
            // overwritten when the allocator first assigns that slot.
            let context_slots = vec![0u32; bucket];
            let position = (bucket - 1)
                .min(self.capacity - 1)
                .min(self.config.max_position_embeddings - 1) as u32;
            self.forward_eager(
                &[self.config.bos_token_id],
                &[position],
                &[0],
                &context_slots,
                true,
                true,
            )?;
            for logical_batch in &decode_batch_warmup_sizes {
                let batch_bucket = self
                    .kernel_plan
                    .shapes
                    .query_bucket(*logical_batch)
                    .ok_or_else(|| {
                        ModelError::Cuda("decode batch warmup bucket overflowed".into())
                    })?;
                let mut graph = DecodeGraph::capture(self, batch_bucket, bucket)?;
                // Capture records work but does not pay every first-launch
                // cost. Replay each graph once before readiness using one
                // real warmup row and isolated padding slots so user
                // requests never trigger module/graph initialization.
                graph.forward(
                    &[self.config.bos_token_id],
                    &[0],
                    &[0],
                    &[vec![0]],
                    self.capacity as u32,
                    true,
                )?;
                self.decode_graphs.insert((batch_bucket, bucket), graph);
            }
        }
        // Readiness is a device-wide quiescence boundary, not merely a
        // compute-stream boundary. Warmup drops many temporary tensors;
        // cuTile queues their frees on a separate deallocator stream.
        // Drain the whole context so the first request cannot inherit
        // allocator/free backlog from initialization.
        unsafe { self.stream.device().synchronize() }
            .map_err(|error| cuda_error("complete CUDA warmup", error))?;
        self.execution_stats = ExecutionStats {
            graph_captures: self.decode_graphs.len() as u64,
            ..ExecutionStats::default()
        };
        Ok(())
    }

    fn gemm(
        &self,
        weight: &Bf16Tensor,
        input: &Bf16Tensor,
        output_size: usize,
        rows: usize,
        input_size: usize,
        operation: &'static str,
    ) -> Result<Bf16Tensor, ModelError> {
        let output =
            output_buffer_async::<bf16>(&[rows, output_size], &self.stream, "GEMM output")?;
        let output = enqueue(
            cublas::gemm_bf16(
                weight.clone(),
                input.clone(),
                output,
                output_size,
                rows,
                input_size,
            )
            .map_err(|error| ModelError::Cuda(format!("{operation}: {error}")))?,
            &self.stream,
            operation,
        )?
        .map_err(|error| ModelError::Cuda(format!("{operation}: {error}")))?;
        Ok(Arc::new(output))
    }

    fn rms_norm(
        &self,
        input: &Bf16Tensor,
        weight: &Bf16Tensor,
        rows: usize,
    ) -> Result<Bf16Tensor, ModelError> {
        let output = output_buffer_async::<bf16>(
            &[rows, self.config.hidden_size],
            &self.stream,
            "RMSNorm output",
        )?
        .partition([1, self.config.hidden_size]);
        let (_, _, output, _) = enqueue(
            unsafe { kernels::rms_norm_bf16(input, weight, output, self.config.rms_norm_eps) }
                .generics(vec![
                    self.config.hidden_size.to_string(),
                    self.kernel_plan.dense.rms_norm.block.get().to_string(),
                ]),
            &self.stream,
            "RMSNorm",
        )?;
        Ok(Arc::new(output.unpartition()))
    }

    fn add_rms_norm(
        &self,
        residual: &Bf16Tensor,
        update: &Bf16Tensor,
        weight: &Bf16Tensor,
        rows: usize,
    ) -> Result<(Bf16Tensor, Bf16Tensor), ModelError> {
        let normalized = output_buffer_async::<bf16>(
            &[rows, self.config.hidden_size],
            &self.stream,
            "fused RMSNorm output",
        )?
        .partition([1, self.config.hidden_size]);
        let combined = output_buffer_async::<bf16>(
            &[rows, self.config.hidden_size],
            &self.stream,
            "residual output",
        )?
        .partition([1, self.config.hidden_size]);
        let (_, _, _, normalized, combined, _) = enqueue(
            unsafe {
                kernels::add_rms_norm_bf16(
                    residual,
                    update,
                    weight,
                    normalized,
                    combined,
                    self.config.rms_norm_eps,
                )
            }
            .generics(vec![
                self.config.hidden_size.to_string(),
                self.kernel_plan.dense.add_rms_norm.block.get().to_string(),
            ]),
            &self.stream,
            "fused residual RMSNorm",
        )?;
        Ok((
            Arc::new(normalized.unpartition()),
            Arc::new(combined.unpartition()),
        ))
    }
}

struct DecodeLayerBuffers {
    attention_input: Tensor<bf16>,
    residual: Tensor<bf16>,
    query: Tensor<bf16>,
    key: Tensor<bf16>,
    value: Tensor<bf16>,
    rotated_query: Tensor<bf16>,
    rotated_key: Tensor<bf16>,
    attention: Tensor<bf16>,
    attention_flat: Tensor<bf16>,
    attention_output: Tensor<bf16>,
    mlp_input: Tensor<bf16>,
    hidden_after_attention: Tensor<bf16>,
    gate: Tensor<bf16>,
    up: Tensor<bf16>,
    activated: Tensor<bf16>,
    down: Tensor<bf16>,
}

impl DecodeLayerBuffers {
    fn allocate(
        cfg: &DenseDecoderConfig,
        batch_size: usize,
        stream: &Arc<Stream>,
    ) -> Result<Self, ModelError> {
        Ok(Self {
            attention_input: output_bf16(&[batch_size, cfg.hidden_size], stream)?,
            // Layer zero does not consume this field, so unlike the other
            // graph buffers it is not guaranteed to be overwritten.
            residual: api::zeros::<bf16>(&[batch_size, cfg.hidden_size])
                .sync_on(stream)
                .map_err(|error| cuda_error("initialize graph residual", error))?,
            query: output_bf16(&[batch_size, cfg.q_width()], stream)?,
            key: output_bf16(&[batch_size, cfg.kv_width()], stream)?,
            value: output_bf16(&[batch_size, cfg.kv_width()], stream)?,
            rotated_query: output_bf16(
                &[batch_size, cfg.num_attention_heads, cfg.head_dim],
                stream,
            )?,
            rotated_key: output_bf16(&[batch_size, cfg.num_key_value_heads, cfg.head_dim], stream)?,
            attention: output_bf16(&[batch_size, cfg.num_attention_heads, cfg.head_dim], stream)?,
            attention_flat: output_bf16(&[batch_size, cfg.hidden_size], stream)?,
            attention_output: output_bf16(&[batch_size, cfg.hidden_size], stream)?,
            mlp_input: output_bf16(&[batch_size, cfg.hidden_size], stream)?,
            hidden_after_attention: output_bf16(&[batch_size, cfg.hidden_size], stream)?,
            gate: output_bf16(&[batch_size, cfg.intermediate_size], stream)?,
            up: output_bf16(&[batch_size, cfg.intermediate_size], stream)?,
            activated: output_bf16(&[batch_size, cfg.intermediate_size], stream)?,
            down: output_bf16(&[batch_size, cfg.hidden_size], stream)?,
        })
    }
}

struct DecodeGraphStorage {
    _embedding_hidden: Tensor<bf16>,
    _layers: Vec<DecodeLayerBuffers>,
    _final_hidden: Tensor<bf16>,
    _final_residual: Tensor<bf16>,
    _argmax_block_max: Tensor<f32>,
    _argmax_block_index: Tensor<u32>,
}

struct DecodeGraph {
    graph: CudaGraph<()>,
    stream: Arc<Stream>,
    token_ids: Tensor<u32>,
    positions: Tensor<u32>,
    current_slots: Tensor<u32>,
    _request_indices: Tensor<u32>,
    context_slots: Tensor<u32>,
    context_lengths: Tensor<i32>,
    logits: Arc<Tensor<bf16>>,
    sampled_token: Arc<Tensor<u32>>,
    batch_size: usize,
    context_bucket: usize,
    _storage: DecodeGraphStorage,
}

impl DecodeGraph {
    fn capture<A: AttentionBackend<Error = ModelError>>(
        runtime: &DenseDecoder<A>,
        batch_size: usize,
        context_bucket: usize,
    ) -> Result<Self, ModelError> {
        let cfg = &runtime.config;
        let stream = runtime.stream.clone();
        let token_ids = copy_u32(&vec![0; batch_size], &stream, "graph token IDs")?;
        let positions = copy_u32(&vec![0; batch_size], &stream, "graph positions")?;
        let current_slots = copy_u32(
            &vec![runtime.capacity as u32; batch_size],
            &stream,
            "graph current slots",
        )?;
        let request_indices = copy_u32(
            &(0..batch_size)
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| ModelError::Cuda("decode graph batch exceeds u32".into()))?,
            &stream,
            "graph request indices",
        )?;
        let context_slots = copy_u32(
            &vec![runtime.capacity as u32; batch_size * context_bucket],
            &stream,
            "graph context slots",
        )?;
        let context_lengths = copy_i32(&vec![1; batch_size], &stream, "graph context lengths")?;
        let mut embedding_hidden = output_bf16(&[batch_size, cfg.hidden_size], &stream)?;
        let mut layers: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|_| DecodeLayerBuffers::allocate(cfg, batch_size, &stream))
            .collect::<Result<_, _>>()?;
        let mut final_hidden = output_bf16(&[batch_size, cfg.hidden_size], &stream)?;
        let mut final_residual = output_bf16(&[batch_size, cfg.hidden_size], &stream)?;
        let logits = Arc::new(output_bf16(&[batch_size, cfg.vocab_size], &stream)?);
        let hidden_block = runtime.kernel_plan.dense.embedding.block.get();
        let rms_norm_block = runtime.kernel_plan.dense.rms_norm.block.get();
        let add_rms_norm_block = runtime.kernel_plan.dense.add_rms_norm.block.get();
        let mlp_block = runtime.kernel_plan.dense.silu_mul.block.get();
        let argmax_block = runtime.kernel_plan.sampling.argmax.block.get();
        let argmax_blocks = cfg.vocab_size.div_ceil(argmax_block);
        let argmax_reduce_block = runtime
            .kernel_plan
            .shapes
            .query_bucket(argmax_blocks)
            .ok_or_else(|| ModelError::Cuda("graph argmax bucket overflowed".into()))?;
        let mut argmax_block_max =
            output_buffer::<f32>(&[batch_size, argmax_blocks], &stream, "graph argmax maxima")?;
        let mut argmax_block_index = output_buffer::<u32>(
            &[batch_size, argmax_blocks],
            &stream,
            "graph argmax indices",
        )?;
        let mut sampled_token =
            output_buffer::<u32>(&[batch_size], &stream, "graph sampled token")?;

        let graph = CudaGraph::scope(&stream, |scope| {
            scope.record(
                kernels::embedding_bf16(
                    &token_ids,
                    &runtime.weights.embedding,
                    (&mut embedding_hidden).partition([1, hidden_block]),
                )
                .generics(vec![cfg.hidden_size.to_string(), hidden_block.to_string()]),
            )?;

            for layer_index in 0..cfg.num_hidden_layers {
                if layer_index == 0 {
                    let layer = &mut layers[0];
                    scope.record(
                        unsafe {
                            kernels::rms_norm_bf16(
                                &embedding_hidden,
                                &runtime.weights.layers[0].input_norm,
                                (&mut layer.attention_input).partition([1, cfg.hidden_size]),
                                cfg.rms_norm_eps,
                            )
                        }
                        .generics(vec![
                            cfg.hidden_size.to_string(),
                            rms_norm_block.to_string(),
                        ]),
                    )?;
                } else {
                    let (previous, current) = layers.split_at_mut(layer_index);
                    let previous = &previous[layer_index - 1];
                    let current = &mut current[0];
                    scope.record(
                        unsafe {
                            kernels::add_rms_norm_bf16(
                                &previous.hidden_after_attention,
                                &previous.down,
                                &runtime.weights.layers[layer_index].input_norm,
                                (&mut current.attention_input).partition([1, cfg.hidden_size]),
                                (&mut current.residual).partition([1, cfg.hidden_size]),
                                cfg.rms_norm_eps,
                            )
                        }
                        .generics(vec![
                            cfg.hidden_size.to_string(),
                            add_rms_norm_block.to_string(),
                        ]),
                    )?;
                }

                let layer_weights = &runtime.weights.layers[layer_index];
                let layer = &mut layers[layer_index];
                record_gemm(
                    scope,
                    &layer_weights.query,
                    &layer.attention_input,
                    &layer.query,
                    cfg.q_width(),
                    batch_size,
                    cfg.hidden_size,
                )?;
                record_gemm(
                    scope,
                    &layer_weights.key,
                    &layer.attention_input,
                    &layer.key,
                    cfg.kv_width(),
                    batch_size,
                    cfg.hidden_size,
                )?;
                record_gemm(
                    scope,
                    &layer_weights.value,
                    &layer.attention_input,
                    &layer.value,
                    cfg.kv_width(),
                    batch_size,
                    cfg.hidden_size,
                )?;

                let query = layer
                    .query
                    .view(&[batch_size, cfg.num_attention_heads, cfg.head_dim])
                    .map_err(device_error)?;
                let key = layer
                    .key
                    .view(&[batch_size, cfg.num_key_value_heads, cfg.head_dim])
                    .map_err(device_error)?;
                let value = layer
                    .value
                    .view(&[batch_size, cfg.num_key_value_heads, cfg.head_dim])
                    .map_err(device_error)?;
                let context_slots_view = context_slots
                    .view(&[batch_size, context_bucket])
                    .map_err(device_error)?;
                runtime
                    .attention
                    .record_decode(DecodeGraphAttention {
                        scope,
                        layer: layer_index,
                        query: &query,
                        key: &key,
                        value: &value,
                        positions: &positions,
                        current_slots: &current_slots,
                        request_indices: &request_indices,
                        context_slots: &context_slots_view,
                        context_lengths: &context_lengths,
                        rotated_query: &mut layer.rotated_query,
                        rotated_key: &mut layer.rotated_key,
                        attention: &mut layer.attention,
                    })
                    .map_err(|error| DeviceError::Internal(error.to_string()))?;
                scope.record(api::memcpy(&mut layer.attention_flat, &layer.attention))?;
                record_gemm(
                    scope,
                    &layer_weights.output,
                    &layer.attention_flat,
                    &layer.attention_output,
                    cfg.hidden_size,
                    batch_size,
                    cfg.hidden_size,
                )?;
                let residual = if layer_index == 0 {
                    &embedding_hidden
                } else {
                    &layer.residual
                };
                scope.record(
                    unsafe {
                        kernels::add_rms_norm_bf16(
                            residual,
                            &layer.attention_output,
                            &layer_weights.post_norm,
                            (&mut layer.mlp_input).partition([1, cfg.hidden_size]),
                            (&mut layer.hidden_after_attention).partition([1, cfg.hidden_size]),
                            cfg.rms_norm_eps,
                        )
                    }
                    .generics(vec![
                        cfg.hidden_size.to_string(),
                        add_rms_norm_block.to_string(),
                    ]),
                )?;
                record_gemm(
                    scope,
                    &layer_weights.gate,
                    &layer.mlp_input,
                    &layer.gate,
                    cfg.intermediate_size,
                    batch_size,
                    cfg.hidden_size,
                )?;
                record_gemm(
                    scope,
                    &layer_weights.up,
                    &layer.mlp_input,
                    &layer.up,
                    cfg.intermediate_size,
                    batch_size,
                    cfg.hidden_size,
                )?;
                scope.record(
                    kernels::silu_mul_bf16(
                        &layer.gate,
                        &layer.up,
                        (&mut layer.activated).partition([1, mlp_block]),
                    )
                    .generics(vec![mlp_block.to_string()]),
                )?;
                record_gemm(
                    scope,
                    &layer_weights.down,
                    &layer.activated,
                    &layer.down,
                    cfg.hidden_size,
                    batch_size,
                    cfg.intermediate_size,
                )?;
            }

            let last = layers
                .last()
                .ok_or_else(|| DeviceError::Internal("model has no layers".into()))?;
            scope.record(
                unsafe {
                    kernels::add_rms_norm_bf16(
                        &last.hidden_after_attention,
                        &last.down,
                        &runtime.weights.final_norm,
                        (&mut final_hidden).partition([1, cfg.hidden_size]),
                        (&mut final_residual).partition([1, cfg.hidden_size]),
                        cfg.rms_norm_eps,
                    )
                }
                .generics(vec![
                    cfg.hidden_size.to_string(),
                    add_rms_norm_block.to_string(),
                ]),
            )?;
            record_gemm(
                scope,
                &runtime.weights.lm_head,
                &final_hidden,
                &logits,
                cfg.vocab_size,
                batch_size,
                cfg.hidden_size,
            )?;
            scope.record(
                kernels::argmax_blocks_batch_bf16(
                    &logits,
                    (&mut argmax_block_max).partition([1, 1]),
                    (&mut argmax_block_index).partition([1, 1]),
                    cfg.vocab_size as i32,
                )
                .generics(vec![argmax_block.to_string()]),
            )?;
            scope.record(
                kernels::argmax_reduce_batch_bf16(
                    &argmax_block_max,
                    &argmax_block_index,
                    (&mut sampled_token).partition([1]),
                    argmax_blocks as i32,
                )
                .generics(vec![argmax_reduce_block.to_string()]),
            )?;
            Ok(())
        })
        .map_err(|error| cuda_error("capture decode CUDA graph", error))?;

        Ok(Self {
            graph,
            stream,
            token_ids,
            positions,
            current_slots,
            _request_indices: request_indices,
            context_slots,
            context_lengths,
            logits,
            sampled_token: Arc::new(sampled_token),
            batch_size,
            context_bucket,
            _storage: DecodeGraphStorage {
                _embedding_hidden: embedding_hidden,
                _layers: layers,
                _final_hidden: final_hidden,
                _final_residual: final_residual,
                _argmax_block_max: argmax_block_max,
                _argmax_block_index: argmax_block_index,
            },
        })
    }

    fn forward(
        &mut self,
        token_ids: &[u32],
        positions: &[u32],
        current_slots: &[u32],
        contexts: &[Vec<u32>],
        sentinel_slot: u32,
        greedy: bool,
    ) -> Result<ForwardOutput, ModelError> {
        let logical_batch = token_ids.len();
        if logical_batch == 0
            || logical_batch > self.batch_size
            || positions.len() != logical_batch
            || current_slots.len() != logical_batch
            || contexts.len() != logical_batch
            || contexts
                .iter()
                .any(|context| context.is_empty() || context.len() > self.context_bucket)
            || (!greedy && logical_batch != 1)
        {
            return Err(ModelError::Cuda(
                "invalid decode graph batch metadata".into(),
            ));
        }
        let mut padded_token_ids = token_ids.to_vec();
        padded_token_ids.resize(self.batch_size, token_ids[0]);
        let mut padded_positions = positions.to_vec();
        padded_positions.resize(self.batch_size, 0);
        let mut padded_current_slots = current_slots.to_vec();
        let mut padded_context_slots = Vec::with_capacity(self.batch_size * self.context_bucket);
        let mut context_lengths = Vec::with_capacity(self.batch_size);
        for context in contexts {
            context_lengths.push(
                i32::try_from(context.len())
                    .map_err(|_| ModelError::Cuda("decode graph context exceeds i32".into()))?,
            );
            padded_context_slots.extend_from_slice(context);
            padded_context_slots.resize(
                padded_context_slots.len() + self.context_bucket - context.len(),
                sentinel_slot,
            );
        }
        for padding_index in logical_batch..self.batch_size {
            let offset = u32::try_from(padding_index - logical_batch + 1)
                .map_err(|_| ModelError::Cuda("decode padding slot exceeds u32".into()))?;
            let padding_slot = sentinel_slot
                .checked_add(offset)
                .ok_or_else(|| ModelError::Cuda("decode padding slot overflowed".into()))?;
            padded_current_slots.push(padding_slot);
            context_lengths.push(1);
            padded_context_slots.push(padding_slot);
            padded_context_slots.resize(
                padded_context_slots.len() + self.context_bucket - 1,
                sentinel_slot,
            );
        }
        let token_ids = copy_u32(&padded_token_ids, &self.stream, "decode graph tokens")?;
        let positions = copy_u32(&padded_positions, &self.stream, "decode graph positions")?;
        let current_slots = copy_u32(&padded_current_slots, &self.stream, "decode graph KV slots")?;
        let context_slots = copy_u32(
            &padded_context_slots,
            &self.stream,
            "decode graph context slots",
        )?;
        let context_lengths = copy_i32(
            &context_lengths,
            &self.stream,
            "decode graph context lengths",
        )?;

        self.graph
            .update(api::memcpy(&mut self.token_ids, &token_ids))
            .map_err(|error| cuda_error("update graph token", error))?;
        self.graph
            .update(api::memcpy(&mut self.positions, &positions))
            .map_err(|error| cuda_error("update graph position", error))?;
        self.graph
            .update(api::memcpy(&mut self.current_slots, &current_slots))
            .map_err(|error| cuda_error("update graph KV slot", error))?;
        self.graph
            .update(api::memcpy(&mut self.context_slots, &context_slots))
            .map_err(|error| cuda_error("update graph context slots", error))?;
        self.graph
            .update(api::memcpy(&mut self.context_lengths, &context_lengths))
            .map_err(|error| cuda_error("update graph context lengths", error))?;
        self.graph
            .launch()
            .sync_on(&self.stream)
            .map_err(|error| cuda_error("replay decode CUDA graph", error))?;
        if greedy {
            let sampled: Vec<u32> = self
                .sampled_token
                .clone()
                .to_host_vec()
                .sync_on(&self.stream)
                .map_err(|error| cuda_error("copy graph sampled token to host", error))?;
            if sampled.len() != self.batch_size {
                return Err(ModelError::Cuda(format!(
                    "graph argmax returned {} tokens for batch {}",
                    sampled.len(),
                    self.batch_size
                )));
            }
            if logical_batch == 1 {
                return Ok(ForwardOutput::Token(sampled[0]));
            }
            return Ok(ForwardOutput::Tokens(
                sampled.into_iter().take(logical_batch).collect(),
            ));
        }
        let logits: Vec<bf16> = self
            .logits
            .clone()
            .to_host_vec()
            .sync_on(&self.stream)
            .map_err(|error| cuda_error("copy graph logits to host", error))?;
        Ok(ForwardOutput::Logits(
            logits.into_iter().map(bf16::to_f32).collect(),
        ))
    }
}

fn record_gemm(
    scope: &Scope,
    matrix: &Tensor<bf16>,
    rhs: &Tensor<bf16>,
    out: &Tensor<bf16>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), DeviceError> {
    scope
        .record(
            cublas::gemm_bf16_into(matrix, rhs, out, m, n, k)
                .map_err(|error| DeviceError::Internal(error.to_string()))?,
        )?
        .map_err(|error| DeviceError::Internal(error.to_string()))
}

/// Allocates storage for an output that the immediately following operation
/// overwrites in full. Callers must never use this for partially-written
/// tensors or readable padding/sentinel regions.
fn output_buffer<T: cutile::DType>(
    shape: &[usize],
    stream: &Arc<Stream>,
    operation: &'static str,
) -> Result<Tensor<T>, ModelError> {
    let len = shape.iter().try_fold(1usize, |len, dimension| {
        len.checked_mul(*dimension)
            .ok_or_else(|| ModelError::Cuda(format!("{operation} shape overflowed")))
    })?;
    if len == 0 {
        return Err(ModelError::Cuda(format!(
            "{operation} output cannot be empty"
        )));
    }
    let tensor = Tensor::<T>::uninitialized(len)
        .sync_on(stream)
        .map_err(|error| cuda_error(operation, error))?;
    // SAFETY: this helper is restricted to output storage. Every caller
    // immediately hands the entire tensor to a kernel, GEMM, or memcpy that
    // writes all elements before any read is enqueued on the same stream.
    unsafe { tensor.assume_init() }
        .reshape(shape)
        .map_err(|error| cuda_error(operation, error))
}

fn output_buffer_async<T: cutile::DType>(
    shape: &[usize],
    stream: &Arc<Stream>,
    operation: &'static str,
) -> Result<Tensor<T>, ModelError> {
    let len = shape.iter().try_fold(1usize, |len, dimension| {
        len.checked_mul(*dimension)
            .ok_or_else(|| ModelError::Cuda(format!("{operation} shape overflowed")))
    })?;
    if len == 0 {
        return Err(ModelError::Cuda(format!(
            "{operation} output cannot be empty"
        )));
    }
    // SAFETY: the executor owns this stream and enqueues every dependent
    // operation on it. The forward synchronizes before exposing host output
    // or committing scheduler-visible KV state.
    let tensor = unsafe { Tensor::<T>::uninitialized(len).async_on(stream) }
        .map_err(|error| cuda_error(operation, error))?;
    // SAFETY: callers use this only for fully-overwritten output storage;
    // see `output_buffer` above. Initialization and every read are ordered
    // on `stream`, and the forward synchronizes before returning.
    unsafe { tensor.assume_init() }
        .reshape(shape)
        .map_err(|error| cuda_error(operation, error))
}

fn enqueue<T>(
    operation: impl DeviceOp<Output = T>,
    stream: &Arc<Stream>,
    name: &'static str,
) -> Result<T, ModelError> {
    // SAFETY: DenseDecoder is thread-confined and all eager work uses this one
    // stream. Inputs remain alive until dependent work is enqueued, async
    // frees are stream ordered, and the forward synchronizes at its boundary.
    let started = Instant::now();
    let output = unsafe { operation.async_on(stream) }.map_err(|error| cuda_error(name, error));
    let elapsed = started.elapsed();
    if elapsed.as_millis() >= 5 {
        tracing::debug!(
            name,
            elapsed_ms = elapsed.as_secs_f64() * 1_000.0,
            "slow CUDA enqueue"
        );
    }
    output
}

fn synchronize_stream(stream: &Arc<Stream>, operation: &'static str) -> Result<(), ModelError> {
    // SAFETY: the stream is owned by the thread-confined executor.
    let started = Instant::now();
    let result = unsafe { stream.synchronize() }.map_err(|error| cuda_error(operation, error));
    let elapsed = started.elapsed();
    if elapsed.as_millis() >= 5 {
        tracing::debug!(
            operation,
            elapsed_ms = elapsed.as_secs_f64() * 1_000.0,
            "slow CUDA synchronization"
        );
    }
    result
}

struct StreamCompletionGuard<'a> {
    stream: &'a Arc<Stream>,
    synchronized: bool,
}

impl<'a> StreamCompletionGuard<'a> {
    fn new(stream: &'a Arc<Stream>) -> Self {
        Self {
            stream,
            synchronized: false,
        }
    }

    fn synchronize(&mut self, operation: &'static str) -> Result<(), ModelError> {
        synchronize_stream(self.stream, operation)?;
        self.synchronized = true;
        Ok(())
    }

    fn mark_synchronized(&mut self) {
        self.synchronized = true;
    }
}

impl Drop for StreamCompletionGuard<'_> {
    fn drop(&mut self) {
        if !self.synchronized {
            // Error paths must drain queued K/V writes before the scheduler
            // may release and recycle physical slots. The original error is
            // preserved; a later CUDA call will surface synchronization loss.
            let _ = unsafe { self.stream.synchronize() };
        }
    }
}

fn output_bf16(shape: &[usize], stream: &Arc<Stream>) -> Result<Tensor<bf16>, ModelError> {
    output_buffer(shape, stream, "allocate decode graph output")
}

fn device_error(error: impl std::fmt::Debug) -> DeviceError {
    DeviceError::Internal(format!("decode graph tensor view: {error:?}"))
}

impl<A: AttentionBackend<Error = ModelError>> ModelProgram for DenseDecoder<A> {
    fn model(&self) -> Arc<dyn Model> {
        self.model.clone()
    }

    fn execute(&mut self, batch: &CudaBatch) -> Result<ProgramOutput, ExecutionError> {
        let started = Instant::now();
        let forward = if batch.is_packed_greedy_decode() {
            let output = self
                .forward_decode_batch(
                    &batch.token_ids,
                    &batch.positions,
                    &batch.current_slots,
                    &batch.contexts,
                )
                .map_err(execution)?;
            self.execution_stats.packed_decode_forwards += 1;
            self.execution_stats.packed_decode_requests += batch.request_count() as u64;
            ForwardOutput::Tokens(output)
        } else {
            let output = self
                .forward_eager_impl(EagerBatch {
                    token_ids: &batch.token_ids,
                    positions: &batch.positions,
                    current_slots: &batch.current_slots,
                    contexts: &batch.contexts,
                    request_indices: &batch.request_indices,
                    context_lengths: &batch.context_lengths,
                    sample_rows: &batch.sample_rows,
                    greedy: batch.all_samples_greedy,
                })
                .map_err(execution)?;
            self.execution_stats.eager_forwards += 1;
            output
        };
        tracing::debug!(
            requests = batch.request_count(),
            tokens = batch.num_tokens(),
            prefill_tokens = batch.num_prefill_tokens,
            elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0,
            "model batch executed"
        );

        match forward {
            ForwardOutput::None => Ok(ProgramOutput::None),
            ForwardOutput::Token(token) => Ok(ProgramOutput::Tokens(vec![TokenId::new(token)])),
            ForwardOutput::Tokens(tokens) => Ok(ProgramOutput::Tokens(
                tokens.into_iter().map(TokenId::new).collect(),
            )),
            ForwardOutput::Logits(values) | ForwardOutput::BatchLogits(values) => {
                Ok(ProgramOutput::HostLogits {
                    values,
                    vocab_size: self.config.vocab_size,
                })
            }
        }
    }

    fn take_execution_stats(&mut self) -> ExecutionStats {
        std::mem::take(&mut self.execution_stats)
    }
}

fn execution(error: ModelError) -> ExecutionError {
    ExecutionError::Execution(error.to_string())
}

pub(crate) fn load_executor(
    artifact: DenseDecoderArtifact,
    device_id: usize,
    kv_capacity_tokens: usize,
    max_batch_tokens: usize,
    max_running: usize,
) -> Result<Box<dyn ModelExecutor>, ModelError> {
    let mut runtime = DenseDecoder::load(
        artifact,
        device_id,
        kv_capacity_tokens,
        max_running,
        max_batch_tokens,
    )?;
    runtime.warm_execution_buckets(max_batch_tokens, max_running)?;
    tracing::info!(
        graph_buckets = runtime.decode_graphs.len(),
        kv_capacity_tokens,
        max_batch_tokens,
        max_running,
        "dense CUDA decoder loaded and execution buckets warmed"
    );
    Ok(Box::new(CudaExecutor::new(runtime)))
}

pub(crate) fn validate(
    model_id: &str,
    weights: &WeightStore,
    probe_name: &str,
    device_id: usize,
) -> Result<CudaModelReport, ModelError> {
    let device = Device::new(device_id)
        .map_err(|error| ModelError::Cuda(format!("initialize device {device_id}: {error:?}")))?;
    let stream = device
        .new_stream()
        .map_err(|error| ModelError::Cuda(format!("create stream: {error:?}")))?;
    let device_weights = DeviceWeights::load(weights, &stream)?;

    let expected = weights.tensor(probe_name)?;
    let actual: Vec<bf16> = device_weights
        .get(probe_name)?
        .to_host_vec()
        .sync_on(&stream)
        .map_err(|error| ModelError::Cuda(format!("verify `{probe_name}`: {error:?}")))?;
    let matches = actual
        .iter()
        .zip(expected.data().chunks_exact(2))
        .all(|(actual, bytes)| actual.to_bits() == u16::from_le_bytes([bytes[0], bytes[1]]));
    if !matches || actual.len() * 2 != expected.data().len() {
        return Err(ModelError::Cuda(format!(
            "BF16 round-trip mismatch for `{probe_name}`"
        )));
    }
    Ok(CudaModelReport {
        model_id: model_id.into(),
        device_id,
        tensors: device_weights.tensors.len(),
        bytes: device_weights.bytes,
    })
}

pub(crate) fn validate_next_token(
    artifact: DenseDecoderArtifact,
    device_id: usize,
    prompt: &str,
) -> Result<CudaForwardReport, ModelError> {
    let token_ids = artifact.model.encode(prompt)?;
    if token_ids.is_empty() {
        return Err(ModelError::InvalidInput(
            "prompt must encode to at least one token".into(),
        ));
    }
    let capacity = token_ids.len() + 1;
    let positions: Vec<u32> = (0..token_ids.len() as u32).collect();
    let slots: Vec<u32> = (0..token_ids.len())
        .map(|index| (capacity - 1 - index) as u32)
        .collect();
    let mut runtime = DenseDecoder::load(artifact, device_id, capacity, 1, token_ids.len())?;
    let logits = match runtime.forward(&token_ids, &positions, &slots, &slots, true, false)? {
        ForwardOutput::Logits(logits) => logits,
        ForwardOutput::None
        | ForwardOutput::Token(_)
        | ForwardOutput::Tokens(_)
        | ForwardOutput::BatchLogits(_) => {
            return Err(ModelError::Cuda("forward omitted requested logits".into()));
        }
    };
    let mut ranked: Vec<_> = logits
        .iter()
        .enumerate()
        .map(|(token_id, logit)| CudaTokenLogit {
            token_id: token_id as u32,
            logit: *logit,
        })
        .collect();
    ranked.sort_unstable_by(|left, right| right.logit.total_cmp(&left.logit));
    ranked.truncate(20);
    let next_token_id = ranked
        .first()
        .map(|entry| entry.token_id)
        .ok_or_else(|| ModelError::Cuda("language-model head returned no logits".into()))?;
    let next_token_text = runtime.model.decoder().push(next_token_id)?;
    Ok(CudaForwardReport {
        model_id: runtime.model.id().into(),
        prompt_tokens: token_ids.len(),
        next_token_id,
        next_token_text,
        top_logits: ranked,
    })
}

fn rope_tables(
    config: &DenseDecoderConfig,
    stream: &Arc<Stream>,
) -> Result<RopeTables, ModelError> {
    let half = config.head_dim / 2;
    let mut inverse_frequency = Vec::with_capacity(half);
    for index in 0..half {
        let exponent = (2 * index) as f32 / config.head_dim as f32;
        let base_frequency = 1.0 / config.rope.theta.powf(exponent);
        let wavelength = TAU / base_frequency;
        let low_wavelength =
            config.rope.original_max_positions as f32 / config.rope.low_frequency_factor;
        let high_wavelength =
            config.rope.original_max_positions as f32 / config.rope.high_frequency_factor;
        let frequency = if wavelength < high_wavelength {
            base_frequency
        } else if wavelength > low_wavelength {
            base_frequency / config.rope.factor
        } else {
            let smooth = (config.rope.original_max_positions as f32 / wavelength
                - config.rope.low_frequency_factor)
                / (config.rope.high_frequency_factor - config.rope.low_frequency_factor);
            (1.0 - smooth) * base_frequency / config.rope.factor + smooth * base_frequency
        };
        inverse_frequency.push(frequency);
    }
    let elements = config
        .max_position_embeddings
        .checked_mul(half)
        .ok_or_else(|| ModelError::Cuda("RoPE table size overflowed".into()))?;
    let mut cosine = Vec::with_capacity(elements);
    let mut sine = Vec::with_capacity(elements);
    for position in 0..config.max_position_embeddings {
        for frequency in &inverse_frequency {
            let angle = position as f32 * frequency;
            cosine.push(angle.cos());
            sine.push(angle.sin());
        }
    }
    let cosine = api::copy_host_vec_to_device(&Arc::new(cosine))
        .sync_on(stream)
        .map_err(|error| cuda_error("upload RoPE cosine table", error))?
        .reshape(&[config.max_position_embeddings, half])
        .map_err(|error| cuda_error("reshape RoPE cosine table", error))?;
    let sine = api::copy_host_vec_to_device(&Arc::new(sine))
        .sync_on(stream)
        .map_err(|error| cuda_error("upload RoPE sine table", error))?
        .reshape(&[config.max_position_embeddings, half])
        .map_err(|error| cuda_error("reshape RoPE sine table", error))?;
    Ok((Arc::new(cosine), Arc::new(sine)))
}

fn copy_u32(
    values: &[u32],
    stream: &Arc<Stream>,
    operation: &'static str,
) -> Result<Tensor<u32>, ModelError> {
    api::copy_host_vec_to_device(&Arc::new(values.to_vec()))
        .sync_on(stream)
        .map_err(|error| cuda_error(operation, error))
}

fn copy_i32(
    values: &[i32],
    stream: &Arc<Stream>,
    operation: &'static str,
) -> Result<Tensor<i32>, ModelError> {
    api::copy_host_vec_to_device(&Arc::new(values.to_vec()))
        .sync_on(stream)
        .map_err(|error| cuda_error(operation, error))
}

fn cuda_error(operation: &'static str, error: impl std::fmt::Debug) -> ModelError {
    ModelError::Cuda(format!("{operation}: {error:?}"))
}

fn execution_bucket(logical_size: usize, minimum: usize) -> Option<usize> {
    if logical_size == 0 || minimum == 0 {
        return None;
    }
    logical_size
        .checked_next_power_of_two()
        .map(|bucket| bucket.max(minimum))
}

fn warmup_logical_sizes(maximum: usize, minimum_bucket: usize) -> Option<Vec<usize>> {
    let maximum_bucket = execution_bucket(maximum, minimum_bucket)?;
    let mut bucket = execution_bucket(minimum_bucket, 1)?;
    let mut sizes = Vec::new();
    loop {
        sizes.push(bucket.min(maximum));
        if bucket == maximum_bucket {
            break;
        }
        bucket = bucket.checked_mul(2)?;
    }
    Some(sizes)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use proptest::prelude::*;

    use super::*;

    #[test]
    fn dense_decoder_config_rejects_inconsistent_head_geometry() {
        let mut config = valid_config();
        config.hidden_size += 1;
        assert!(!valid_dense_decoder_config(config));
    }

    #[test]
    fn execution_buckets_reject_empty_dimensions() {
        assert_eq!(execution_bucket(0, 1), None);
        assert_eq!(execution_bucket(1, 0), None);
        assert_eq!(warmup_logical_sizes(0, 1), None);
    }

    fn valid_config() -> DenseDecoderConfig {
        DenseDecoderConfig {
            bos_token_id: 128_000,
            head_dim: 64,
            hidden_size: 2_048,
            intermediate_size: 8_192,
            max_position_embeddings: 131_072,
            num_attention_heads: 32,
            num_hidden_layers: 16,
            num_key_value_heads: 8,
            rms_norm_eps: 1.0e-5,
            rope: Llama3RopeConfig {
                factor: 32.0,
                high_frequency_factor: 4.0,
                low_frequency_factor: 1.0,
                original_max_positions: 8_192,
                theta: 500_000.0,
            },
            vocab_size: 128_256,
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn warmup_sizes_cover_every_runtime_bucket(
            maximum in 1usize..4097,
            minimum in prop_oneof![Just(1usize), Just(16usize)],
        ) {
            let sizes = warmup_logical_sizes(maximum, minimum).unwrap();
            prop_assert_eq!(sizes.last().copied(), Some(maximum));
            prop_assert!(sizes.windows(2).all(|pair| pair[0] < pair[1]));

            let warmed: BTreeSet<_> = sizes
                .iter()
                .map(|size| execution_bucket(*size, minimum).unwrap())
                .collect();
            let expected_last = execution_bucket(maximum, minimum).unwrap();
            prop_assert_eq!(warmed.last().copied(), Some(expected_last));
            let every_size_is_covered = (1..=maximum)
                .all(|size| warmed.contains(&execution_bucket(size, minimum).unwrap()));
            prop_assert!(every_size_is_covered);
        }

        #[test]
        fn decode_graph_warmup_covers_runtime_bucket_pairs(
            max_batch in 1usize..65,
            max_context in 1usize..4097,
            batch_seed in any::<usize>(),
            context_seed in any::<usize>(),
        ) {
            let batch = batch_seed % max_batch + 1;
            let context = context_seed % max_context + 1;
            let warmed_batches: BTreeSet<_> = warmup_logical_sizes(max_batch, 1)
                .unwrap()
                .into_iter()
                .map(|size| execution_bucket(size, 1).unwrap())
                .collect();
            let warmed_contexts: BTreeSet<_> = warmup_logical_sizes(max_context, 16)
                .unwrap()
                .into_iter()
                .map(|size| execution_bucket(size, 16).unwrap())
                .collect();

            prop_assert!(warmed_batches.contains(&execution_bucket(batch, 1).unwrap()));
            prop_assert!(warmed_contexts.contains(&execution_bucket(context, 16).unwrap()));
        }
    }
}
