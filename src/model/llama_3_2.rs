use std::{path::Path, sync::Arc};

use serde::Deserialize;

use super::{
    ChatMessage, ChatRole, IncrementalDecoder, Model, ModelError, ModelSummary, read_file,
    weights::WeightStore,
};

#[cfg(feature = "cuda")]
use super::{CudaForwardReport, CudaModelReport, CudaTokenLogit};

const MODEL_ID: &str = "meta-llama/Llama-3.2-1B-Instruct";

pub(super) fn supports(model_id: &str) -> bool {
    model_id == MODEL_ID
}

pub(super) fn load(model_id: &str, model_dir: &Path) -> Result<Llama32, ModelError> {
    Llama32::load(model_id, model_dir)
}

#[cfg(feature = "cuda")]
pub(super) fn validate_cuda(
    model_id: &str,
    model_dir: &Path,
    device_id: usize,
) -> Result<CudaModelReport, ModelError> {
    cuda_impl::validate(model_id, model_dir, device_id)
}

#[cfg(feature = "cuda")]
pub(super) fn validate_cuda_next_token(
    model_id: &str,
    model_dir: &Path,
    device_id: usize,
    prompt: &str,
) -> Result<CudaForwardReport, ModelError> {
    cuda_impl::validate_next_token(model_id, model_dir, device_id, prompt)
}

#[cfg(feature = "cuda")]
pub(super) fn load_cuda_backend(
    model_id: &str,
    model_dir: &Path,
    device_id: usize,
    kv_capacity_tokens: usize,
) -> Result<Box<dyn crate::engine::Backend>, ModelError> {
    cuda_impl::load_backend(model_id, model_dir, device_id, kv_capacity_tokens)
}

#[derive(Debug, Clone, Deserialize)]
struct RopeScaling {
    factor: f32,
    high_freq_factor: f32,
    low_freq_factor: f32,
    original_max_position_embeddings: usize,
    rope_type: String,
}

#[derive(Debug, Clone, Deserialize)]
struct Config {
    architectures: Vec<String>,
    attention_bias: bool,
    bos_token_id: u32,
    eos_token_id: Vec<u32>,
    head_dim: usize,
    hidden_act: String,
    hidden_size: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    mlp_bias: bool,
    model_type: String,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f32,
    rope_scaling: RopeScaling,
    rope_theta: f32,
    tie_word_embeddings: bool,
    torch_dtype: String,
    vocab_size: usize,
}

impl Config {
    fn load(model_dir: &Path) -> Result<Self, ModelError> {
        let path = model_dir.join("config.json");
        let text = read_file(&path)?;
        let config: Self = serde_json::from_str(&text).map_err(|source| ModelError::Json {
            path: path.clone(),
            source,
        })?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), ModelError> {
        let invalid = |message: &str| Err(ModelError::InvalidConfig(message.into()));
        if self.model_type != "llama"
            || !self
                .architectures
                .iter()
                .any(|architecture| architecture == "LlamaForCausalLM")
        {
            return invalid("Llama 3.2 requires LlamaForCausalLM");
        }
        if self.torch_dtype != "bfloat16" {
            return invalid("Llama 3.2 v1 requires torch_dtype=bfloat16");
        }
        if self.hidden_act != "silu" {
            return invalid("Llama 3.2 requires the SiLU gated MLP");
        }
        if self.attention_bias || self.mlp_bias {
            return invalid("Llama 3.2 does not use attention or MLP bias tensors");
        }
        if self.num_hidden_layers == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.vocab_size == 0
        {
            return invalid("model dimensions must be positive");
        }
        if self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return invalid("attention heads must be divisible by KV heads");
        }
        if self.hidden_size != self.num_attention_heads * self.head_dim {
            return invalid("hidden_size must equal num_attention_heads * head_dim");
        }
        if self.rms_norm_eps <= 0.0 || !self.rms_norm_eps.is_finite() {
            return invalid("rms_norm_eps must be finite and positive");
        }
        if self.rope_theta <= 0.0 || !self.rope_theta.is_finite() {
            return invalid("rope_theta must be finite and positive");
        }
        if self.rope_scaling.rope_type != "llama3"
            || self.rope_scaling.factor <= 0.0
            || self.rope_scaling.high_freq_factor <= 0.0
            || self.rope_scaling.low_freq_factor <= 0.0
            || self.rope_scaling.original_max_position_embeddings == 0
        {
            return invalid("Llama 3.2 requires valid Llama 3 RoPE scaling");
        }
        if self.max_position_embeddings < self.rope_scaling.original_max_position_embeddings {
            return invalid("max context is smaller than the original RoPE context");
        }
        if self.eos_token_id.is_empty() {
            return invalid("at least one EOS token is required");
        }
        if self.bos_token_id as usize >= self.vocab_size
            || self
                .eos_token_id
                .iter()
                .any(|token| *token as usize >= self.vocab_size)
        {
            return invalid("BOS and EOS token IDs must be inside the vocabulary");
        }
        Ok(())
    }

    fn q_width(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    fn kv_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

struct Tokenizer {
    inner: Arc<tokenizers::Tokenizer>,
}

impl Tokenizer {
    fn load(model_dir: &Path) -> Result<Self, ModelError> {
        let path = model_dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&path)
            .map_err(|error| ModelError::Tokenizer(error.to_string()))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, ModelError> {
        self.inner
            .encode(text, true)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| ModelError::Tokenizer(error.to_string()))
    }

    fn decode(&self, ids: &[u32]) -> Result<String, ModelError> {
        self.inner
            .decode(ids, true)
            .map_err(|error| ModelError::Tokenizer(error.to_string()))
    }
}

impl Clone for Tokenizer {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

pub(super) struct Llama32 {
    id: String,
    config: Config,
    tokenizer: Tokenizer,
    weights: WeightStore,
}

impl Llama32 {
    fn load(model_id: &str, model_dir: &Path) -> Result<Self, ModelError> {
        let config = Config::load(model_dir)?;
        let weights = WeightStore::open(model_dir)?;
        validate_weights(&weights, &config)?;
        let tokenizer = Tokenizer::load(model_dir)?;
        Ok(Self {
            id: model_id.into(),
            config,
            tokenizer,
            weights,
        })
    }
}

impl Model for Llama32 {
    fn id(&self) -> &str {
        &self.id
    }

    fn render_chat(&self, messages: &[ChatMessage<'_>]) -> Result<String, ModelError> {
        render_chat(messages)
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, ModelError> {
        self.tokenizer.encode(text)
    }

    fn decoder(&self) -> Box<dyn IncrementalDecoder> {
        Box::new(Decoder {
            tokenizer: self.tokenizer.clone(),
            token_ids: Vec::new(),
            decoded: String::new(),
        })
    }

    fn eos_token_ids(&self) -> &[u32] {
        &self.config.eos_token_id
    }

    fn summary(&self) -> ModelSummary {
        ModelSummary {
            id: self.id.clone(),
            architecture: self.config.architectures[0].clone(),
            dtype: self.config.torch_dtype.clone(),
            layers: self.config.num_hidden_layers,
            hidden_size: self.config.hidden_size,
            attention_heads: self.config.num_attention_heads,
            kv_heads: self.config.num_key_value_heads,
            vocab_size: self.config.vocab_size,
            tensors: self.weights.tensor_count(),
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use std::{
        collections::{HashMap, HashSet},
        f32::consts::TAU,
        path::Path,
        sync::Arc,
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
        tensor::{PartitionMut, Reshape, Tensor, ToHostVec},
        tile_kernel::{PartitionOp, TileKernel},
    };

    use crate::{
        cuda::{cublas, kernels},
        engine::{
            Backend, BackendError, BackendExecutionStats, GenerateRequest, PreparedRequest,
            RequestId, ScheduledWork, StepOutput,
        },
        model::{IncrementalDecoder, Model},
    };

    use super::{
        CudaForwardReport, CudaModelReport, CudaTokenLogit, Llama32, ModelError, WeightStore,
    };

    const HIDDEN_BLOCK: usize = 512;
    const MLP_BLOCK: usize = 512;
    const ATTENTION_QUERY_BLOCK: usize = 1;
    const ATTENTION_KEY_BLOCK: usize = 16;
    const ARGMAX_BLOCK: usize = 256;

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
                bytes = bytes.checked_add(tensor.num_bytes()).ok_or_else(|| {
                    ModelError::Cuda("device weight byte count overflowed".into())
                })?;
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
        fn load(model: &Llama32, stream: &Arc<Stream>) -> Result<Self, ModelError> {
            let all = DeviceWeights::load(&model.weights, stream)?;
            let embedding = all.get("model.embed_tokens.weight")?;
            let final_norm = all.get("model.norm.weight")?;
            let lm_head = if model.config.tie_word_embeddings {
                embedding.clone()
            } else {
                all.get("lm_head.weight")?
            };
            let mut layers = Vec::with_capacity(model.config.num_hidden_layers);
            for layer in 0..model.config.num_hidden_layers {
                let prefix = format!("model.layers.{layer}");
                layers.push(LayerWeights {
                    input_norm: all.get(&format!("{prefix}.input_layernorm.weight"))?,
                    post_norm: all.get(&format!("{prefix}.post_attention_layernorm.weight"))?,
                    query: all.get(&format!("{prefix}.self_attn.q_proj.weight"))?,
                    key: all.get(&format!("{prefix}.self_attn.k_proj.weight"))?,
                    value: all.get(&format!("{prefix}.self_attn.v_proj.weight"))?,
                    output: all.get(&format!("{prefix}.self_attn.o_proj.weight"))?,
                    gate: all.get(&format!("{prefix}.mlp.gate_proj.weight"))?,
                    up: all.get(&format!("{prefix}.mlp.up_proj.weight"))?,
                    down: all.get(&format!("{prefix}.mlp.down_proj.weight"))?,
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

    struct CudaLlama {
        model: Arc<Llama32>,
        stream: Arc<Stream>,
        weights: RuntimeWeights,
        cosine: Arc<Tensor<f32>>,
        sine: Arc<Tensor<f32>>,
        key_cache: Vec<Bf16Tensor>,
        value_cache: Vec<Bf16Tensor>,
        capacity: usize,
        decode_graphs: HashMap<(usize, usize), DecodeGraph>,
        failed_decode_graphs: HashSet<(usize, usize)>,
        execution_stats: BackendExecutionStats,
    }

    enum ForwardOutput {
        None,
        Logits(Vec<f32>),
        Token(u32),
        Tokens(Vec<u32>),
    }

    impl CudaLlama {
        fn load(
            model: Arc<Llama32>,
            device_id: usize,
            capacity: usize,
        ) -> Result<Self, ModelError> {
            let device = Device::new(device_id).map_err(|error| {
                ModelError::Cuda(format!("initialize device {device_id}: {error:?}"))
            })?;
            let stream = device
                .new_stream()
                .map_err(|error| ModelError::Cuda(format!("create stream: {error:?}")))?;
            let weights = RuntimeWeights::load(&model, &stream)?;
            let (cosine, sine) = rope_tables(&model.config, &stream)?;
            let mut key_cache = Vec::with_capacity(model.config.num_hidden_layers);
            let mut value_cache = Vec::with_capacity(model.config.num_hidden_layers);
            let cache_shape = [
                capacity.checked_add(1).ok_or_else(|| {
                    ModelError::Cuda("KV cache sentinel allocation overflowed".into())
                })?,
                model.config.num_key_value_heads,
                model.config.head_dim,
            ];
            for layer in 0..model.config.num_hidden_layers {
                let key = api::zeros::<bf16>(&cache_shape)
                    .sync_on(&stream)
                    .map_err(|error| {
                        ModelError::Cuda(format!("allocate layer {layer} key cache: {error:?}"))
                    })?;
                let value = api::zeros::<bf16>(&cache_shape)
                    .sync_on(&stream)
                    .map_err(|error| {
                        ModelError::Cuda(format!("allocate layer {layer} value cache: {error:?}"))
                    })?;
                key_cache.push(Arc::new(key));
                value_cache.push(Arc::new(value));
            }
            Ok(Self {
                model,
                stream,
                weights,
                cosine,
                sine,
                key_cache,
                value_cache,
                capacity,
                decode_graphs: HashMap::new(),
                failed_decode_graphs: HashSet::new(),
                execution_stats: BackendExecutionStats::default(),
            })
        }

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
                let key = (1, context_slots.len().next_power_of_two().max(16));
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
        ) -> Result<ForwardOutput, ModelError> {
            self.forward_eager_impl(
                token_ids,
                positions,
                current_slots,
                context_slots,
                None,
                return_logits,
            )
        }

        fn forward_decode_batch(
            &mut self,
            token_ids: &[u32],
            positions: &[u32],
            current_slots: &[u32],
            contexts: &[Vec<u32>],
        ) -> Result<Vec<u32>, ModelError> {
            let context_bucket = contexts
                .iter()
                .map(Vec::len)
                .max()
                .unwrap_or(1)
                .next_power_of_two()
                .max(16);
            let key = (token_ids.len(), context_bucket);
            let output = if self.failed_decode_graphs.contains(&key) {
                self.forward_eager_impl(
                    token_ids,
                    positions,
                    current_slots,
                    &[],
                    Some(contexts),
                    true,
                )?
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
                        self.forward_eager_impl(
                            token_ids,
                            positions,
                            current_slots,
                            &[],
                            Some(contexts),
                            true,
                        )?
                    }
                }
            };
            if self.failed_decode_graphs.contains(&key) {
                self.execution_stats.eager_forwards += 1;
            }
            match output {
                ForwardOutput::Tokens(tokens) => Ok(tokens),
                ForwardOutput::None | ForwardOutput::Logits(_) | ForwardOutput::Token(_) => Err(
                    ModelError::Cuda("packed decode omitted sampled tokens".into()),
                ),
            }
        }

        fn forward_eager_impl(
            &self,
            token_ids: &[u32],
            positions: &[u32],
            current_slots: &[u32],
            context_slots: &[u32],
            decode_contexts: Option<&[Vec<u32>]>,
            return_logits: bool,
        ) -> Result<ForwardOutput, ModelError> {
            let rows = token_ids.len();
            if rows == 0
                || positions.len() != rows
                || current_slots.len() != rows
                || current_slots
                    .iter()
                    .any(|slot| *slot as usize >= self.capacity)
            {
                return Err(ModelError::Cuda("invalid forward batch metadata".into()));
            }
            match decode_contexts {
                Some(contexts)
                    if contexts.len() == rows
                        && contexts.iter().all(|context| {
                            !context.is_empty()
                                && context.iter().all(|slot| (*slot as usize) < self.capacity)
                        }) => {}
                Some(_) => {
                    return Err(ModelError::Cuda(
                        "invalid packed decode context metadata".into(),
                    ));
                }
                None if context_slots.is_empty()
                    || context_slots
                        .iter()
                        .any(|slot| *slot as usize >= self.capacity) =>
                {
                    return Err(ModelError::Cuda("invalid forward context metadata".into()));
                }
                None => {}
            }
            let cfg = &self.model.config;
            let stream = &self.stream;
            let query_start = positions[0] as i32;
            let token_ids = copy_u32(token_ids, stream, "token IDs")?;
            let positions = copy_u32(positions, stream, "positions")?;
            let current_slots = copy_u32(current_slots, stream, "current KV slots")?;
            let (context_bucket, context_slots, attention_metadata, decode_context_lengths) =
                if let Some(contexts) = decode_contexts {
                    let context_bucket = contexts
                        .iter()
                        .map(Vec::len)
                        .max()
                        .unwrap_or(1)
                        .next_power_of_two()
                        .max(16);
                    let mut padded = Vec::with_capacity(rows * context_bucket);
                    let mut lengths = Vec::with_capacity(rows);
                    for context in contexts {
                        lengths.push(i32::try_from(context.len()).map_err(|_| {
                            ModelError::Cuda("packed decode context exceeds i32".into())
                        })?);
                        padded.extend_from_slice(context);
                        padded.resize(
                            padded.len() + context_bucket - context.len(),
                            self.capacity as u32,
                        );
                    }
                    (
                        context_bucket,
                        copy_u32(&padded, stream, "packed decode context KV slots")?,
                        None,
                        Some(copy_i32(&lengths, stream, "packed decode context lengths")?),
                    )
                } else {
                    let context_len = context_slots.len();
                    let context_bucket = context_len.next_power_of_two().max(16);
                    let mut padded = context_slots.to_vec();
                    padded.resize(context_bucket, self.capacity as u32);
                    let metadata = copy_i32(
                        &[
                            i32::try_from(context_len).map_err(|_| {
                                ModelError::Cuda("attention context exceeds i32".into())
                            })?,
                            query_start,
                        ],
                        stream,
                        "attention metadata",
                    )?;
                    (
                        context_bucket,
                        copy_u32(&padded, stream, "padded context KV slots")?,
                        Some(metadata),
                        None,
                    )
                };

            let hidden = api::zeros::<bf16>(&[rows, cfg.hidden_size])
                .partition([1, HIDDEN_BLOCK])
                .sync_on(stream)
                .map_err(|error| cuda_error("allocate embedding output", error))?;
            let (_, _, hidden) =
                kernels::embedding_bf16(&token_ids, &*self.weights.embedding, hidden)
                    .generics(vec![cfg.hidden_size.to_string(), HIDDEN_BLOCK.to_string()])
                    .sync_on(stream)
                    .map_err(|error| cuda_error("embedding", error))?;
            let hidden = Arc::new(hidden.unpartition());

            let mut pending: Option<(Bf16Tensor, Bf16Tensor)> = None;
            for (layer_index, layer) in self.weights.layers.iter().enumerate() {
                let (attention_input, residual) = match pending.take() {
                    None => (
                        self.rms_norm(hidden.clone(), layer.input_norm.clone(), rows)?,
                        hidden.clone(),
                    ),
                    Some((residual, update)) => {
                        let (normalized, combined) =
                            self.add_rms_norm(residual, update, layer.input_norm.clone(), rows)?;
                        (normalized, combined)
                    }
                };

                let query = self.gemm(
                    layer.query.clone(),
                    attention_input.clone(),
                    cfg.q_width(),
                    rows,
                    cfg.hidden_size,
                    "query projection",
                )?;
                let key = self.gemm(
                    layer.key.clone(),
                    attention_input.clone(),
                    cfg.kv_width(),
                    rows,
                    cfg.hidden_size,
                    "key projection",
                )?;
                let value = self.gemm(
                    layer.value.clone(),
                    attention_input,
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

                let rotated_query =
                    api::zeros::<bf16>(&[rows, cfg.num_attention_heads, cfg.head_dim])
                        .partition([1, 1, cfg.head_dim])
                        .sync_on(stream)
                        .map_err(|error| cuda_error("allocate rotated query", error))?;
                let (_, _, _, _, rotated_query) = kernels::rope_q_bf16(
                    &query,
                    &positions,
                    &self.cosine,
                    &self.sine,
                    rotated_query,
                )
                .generics(vec![
                    cfg.head_dim.to_string(),
                    (cfg.head_dim / 2).to_string(),
                ])
                .sync_on(stream)
                .map_err(|error| cuda_error("query RoPE", error))?;
                let rotated_query = Arc::new(rotated_query.unpartition());

                let rotated_key =
                    api::zeros::<bf16>(&[rows, cfg.num_key_value_heads, cfg.head_dim])
                        .partition([1, 1, cfg.head_dim])
                        .sync_on(stream)
                        .map_err(|error| cuda_error("allocate rotated key", error))?;
                let (_, _, _, _, _, _, _, _, rotated_key) = unsafe {
                    kernels::rope_kv_write_bf16(
                        &key,
                        &value,
                        &positions,
                        &current_slots,
                        &self.cosine,
                        &self.sine,
                        self.key_cache[layer_index].device_pointer(),
                        self.value_cache[layer_index].device_pointer(),
                        rotated_key,
                    )
                }
                .generics(vec![
                    cfg.head_dim.to_string(),
                    (cfg.head_dim / 2).to_string(),
                    cfg.num_key_value_heads.to_string(),
                ])
                .sync_on(stream)
                .map_err(|error| cuda_error("key RoPE and flat KV write", error))?;
                drop(rotated_key);

                let attention = if let Some(context_lengths) = &decode_context_lengths {
                    let context_slots = context_slots
                        .view(&[rows, context_bucket])
                        .map_err(|error| cuda_error("view packed context slots", error))?;
                    let gathered_shape =
                        [rows * cfg.num_key_value_heads, context_bucket, cfg.head_dim];
                    let gathered_key = api::zeros::<bf16>(&gathered_shape)
                        .partition([1, 1, cfg.head_dim])
                        .sync_on(stream)
                        .map_err(|error| cuda_error("allocate packed gathered key", error))?;
                    let gathered_value = api::zeros::<bf16>(&gathered_shape)
                        .partition([1, 1, cfg.head_dim])
                        .sync_on(stream)
                        .map_err(|error| cuda_error("allocate packed gathered value", error))?;
                    let (_, _, gathered_key) = kernels::gather_flat_kv_decode_batch_bf16(
                        &context_slots,
                        &self.key_cache[layer_index],
                        gathered_key,
                    )
                    .generics(vec![
                        cfg.head_dim.to_string(),
                        cfg.num_key_value_heads.to_string(),
                    ])
                    .sync_on(stream)
                    .map_err(|error| cuda_error("gather packed key cache", error))?;
                    let (_, _, gathered_value) = kernels::gather_flat_kv_decode_batch_bf16(
                        &context_slots,
                        &self.value_cache[layer_index],
                        gathered_value,
                    )
                    .generics(vec![
                        cfg.head_dim.to_string(),
                        cfg.num_key_value_heads.to_string(),
                    ])
                    .sync_on(stream)
                    .map_err(|error| cuda_error("gather packed value cache", error))?;
                    let gathered_key = gathered_key.unpartition();
                    let gathered_value = gathered_value.unpartition();
                    let attention =
                        api::zeros::<bf16>(&[rows, cfg.num_attention_heads, cfg.head_dim])
                            .partition([1, 1, cfg.head_dim])
                            .sync_on(stream)
                            .map_err(|error| {
                                cuda_error("allocate packed attention output", error)
                            })?;
                    let (_, _, _, _, attention, _, _) = unsafe {
                        kernels::decode_attention_batch_bf16(
                            &rotated_query,
                            &gathered_key,
                            &gathered_value,
                            context_lengths,
                            attention,
                            1.0 / (cfg.head_dim as f32).sqrt(),
                            (cfg.num_attention_heads / cfg.num_key_value_heads) as i32,
                        )
                    }
                    .generics(vec![
                        ATTENTION_KEY_BLOCK.to_string(),
                        cfg.head_dim.to_string(),
                        cfg.num_key_value_heads.to_string(),
                    ])
                    .sync_on(stream)
                    .map_err(|error| cuda_error("packed decode attention", error))?;
                    attention.unpartition()
                } else {
                    let gathered_key = api::zeros::<bf16>(&[
                        cfg.num_key_value_heads,
                        context_bucket,
                        cfg.head_dim,
                    ])
                    .partition([1, 1, cfg.head_dim])
                    .sync_on(stream)
                    .map_err(|error| cuda_error("allocate gathered key", error))?;
                    let gathered_value = api::zeros::<bf16>(&[
                        cfg.num_key_value_heads,
                        context_bucket,
                        cfg.head_dim,
                    ])
                    .partition([1, 1, cfg.head_dim])
                    .sync_on(stream)
                    .map_err(|error| cuda_error("allocate gathered value", error))?;
                    let (_, _, gathered_key) = kernels::gather_flat_kv_bf16(
                        &context_slots,
                        &self.key_cache[layer_index],
                        gathered_key,
                    )
                    .generics(vec![cfg.head_dim.to_string()])
                    .sync_on(stream)
                    .map_err(|error| cuda_error("gather key cache", error))?;
                    let (_, _, gathered_value) = kernels::gather_flat_kv_bf16(
                        &context_slots,
                        &self.value_cache[layer_index],
                        gathered_value,
                    )
                    .generics(vec![cfg.head_dim.to_string()])
                    .sync_on(stream)
                    .map_err(|error| cuda_error("gather value cache", error))?;
                    let gathered_key = gathered_key.unpartition();
                    let gathered_value = gathered_value.unpartition();
                    let attention =
                        api::zeros::<bf16>(&[rows, cfg.num_attention_heads, cfg.head_dim])
                            .partition([ATTENTION_QUERY_BLOCK, 1, cfg.head_dim])
                            .sync_on(stream)
                            .map_err(|error| cuda_error("allocate attention output", error))?;
                    let metadata = attention_metadata.as_ref().ok_or_else(|| {
                        ModelError::Cuda("causal attention metadata is missing".into())
                    })?;
                    let (_, _, _, _, attention, _, _) = unsafe {
                        kernels::causal_attention_bf16(
                            &rotated_query,
                            &gathered_key,
                            &gathered_value,
                            metadata,
                            attention,
                            1.0 / (cfg.head_dim as f32).sqrt(),
                            (cfg.num_attention_heads / cfg.num_key_value_heads) as i32,
                        )
                    }
                    .generics(vec![
                        ATTENTION_QUERY_BLOCK.to_string(),
                        ATTENTION_KEY_BLOCK.to_string(),
                        cfg.head_dim.to_string(),
                    ])
                    .sync_on(stream)
                    .map_err(|error| cuda_error("causal attention", error))?;
                    attention.unpartition()
                };
                let attention = Arc::new(
                    attention
                        .reshape(&[rows, cfg.hidden_size])
                        .map_err(|error| cuda_error("reshape attention output", error))?,
                );
                let attention_output = self.gemm(
                    layer.output.clone(),
                    attention,
                    cfg.hidden_size,
                    rows,
                    cfg.hidden_size,
                    "attention output projection",
                )?;
                let (mlp_input, hidden_after_attention) =
                    self.add_rms_norm(residual, attention_output, layer.post_norm.clone(), rows)?;
                let gate = self.gemm(
                    layer.gate.clone(),
                    mlp_input.clone(),
                    cfg.intermediate_size,
                    rows,
                    cfg.hidden_size,
                    "MLP gate projection",
                )?;
                let up = self.gemm(
                    layer.up.clone(),
                    mlp_input,
                    cfg.intermediate_size,
                    rows,
                    cfg.hidden_size,
                    "MLP up projection",
                )?;
                let activated = api::zeros::<bf16>(&[rows, cfg.intermediate_size])
                    .partition([1, MLP_BLOCK])
                    .sync_on(stream)
                    .map_err(|error| cuda_error("allocate MLP activation", error))?;
                let (_, _, activated) = kernels::silu_mul_bf16(&gate, &up, activated)
                    .generics(vec![MLP_BLOCK.to_string()])
                    .sync_on(stream)
                    .map_err(|error| cuda_error("SiLU gated activation", error))?;
                let activated = Arc::new(activated.unpartition());
                let down = self.gemm(
                    layer.down.clone(),
                    activated,
                    cfg.hidden_size,
                    rows,
                    cfg.intermediate_size,
                    "MLP down projection",
                )?;
                pending = Some((hidden_after_attention, down));
            }

            if !return_logits {
                return Ok(ForwardOutput::None);
            }

            let (residual, update) = pending
                .ok_or_else(|| ModelError::Cuda("model has no transformer layers".into()))?;
            let (final_hidden, _) =
                self.add_rms_norm(residual, update, self.weights.final_norm.clone(), rows)?;
            if decode_contexts.is_some() {
                let logits = self.gemm(
                    self.weights.lm_head.clone(),
                    final_hidden,
                    cfg.vocab_size,
                    rows,
                    cfg.hidden_size,
                    "packed language-model head",
                )?;
                let argmax_blocks = cfg.vocab_size.div_ceil(ARGMAX_BLOCK);
                let reduce_block = argmax_blocks.next_power_of_two();
                let block_max = api::zeros::<f32>(&[rows, argmax_blocks])
                    .partition([1, 1])
                    .sync_on(stream)
                    .map_err(|error| cuda_error("allocate packed argmax maxima", error))?;
                let block_index = api::zeros::<u32>(&[rows, argmax_blocks])
                    .partition([1, 1])
                    .sync_on(stream)
                    .map_err(|error| cuda_error("allocate packed argmax indices", error))?;
                let (_, block_max, block_index, _) = kernels::argmax_blocks_batch_bf16(
                    &logits,
                    block_max,
                    block_index,
                    cfg.vocab_size as i32,
                )
                .generics(vec![ARGMAX_BLOCK.to_string()])
                .sync_on(stream)
                .map_err(|error| cuda_error("packed logits argmax blocks", error))?;
                let block_max = block_max.unpartition();
                let block_index = block_index.unpartition();
                let sampled = api::zeros::<u32>(&[rows])
                    .partition([1])
                    .sync_on(stream)
                    .map_err(|error| cuda_error("allocate packed sampled tokens", error))?;
                let (_, _, sampled, _) = kernels::argmax_reduce_batch_bf16(
                    &block_max,
                    &block_index,
                    sampled,
                    argmax_blocks as i32,
                )
                .generics(vec![reduce_block.to_string()])
                .sync_on(stream)
                .map_err(|error| cuda_error("reduce packed logits argmax", error))?;
                let sampled = Arc::new(sampled.unpartition());
                let sampled = sampled
                    .clone()
                    .to_host_vec()
                    .sync_on(stream)
                    .map_err(|error| cuda_error("copy packed sampled tokens to host", error))?;
                return Ok(ForwardOutput::Tokens(sampled));
            }
            let last = api::zeros::<bf16>(&[cfg.hidden_size])
                .partition([HIDDEN_BLOCK])
                .sync_on(stream)
                .map_err(|error| cuda_error("allocate final token hidden state", error))?;
            let (_, last, _) = kernels::gather_row_bf16(&final_hidden, last, (rows - 1) as i32)
                .generics(vec![HIDDEN_BLOCK.to_string()])
                .sync_on(stream)
                .map_err(|error| cuda_error("gather final token hidden state", error))?;
            let last = last.unpartition();
            let logits = self.gemm(
                self.weights.lm_head.clone(),
                Arc::new(last),
                cfg.vocab_size,
                1,
                cfg.hidden_size,
                "language-model head",
            )?;
            let logits: Vec<bf16> = logits
                .clone()
                .to_host_vec()
                .sync_on(stream)
                .map_err(|error| cuda_error("copy logits to host", error))?;
            Ok(ForwardOutput::Logits(
                logits.into_iter().map(bf16::to_f32).collect(),
            ))
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
            let bucket = contexts
                .iter()
                .map(Vec::len)
                .max()
                .unwrap_or(1)
                .next_power_of_two()
                .max(16);
            let key = (batch_size, bucket);
            if !self.decode_graphs.contains_key(&key) {
                let graph = DecodeGraph::capture(self, batch_size, bucket)?;
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

        fn warm_decode_graphs(&mut self) -> Result<(), ModelError> {
            if self.capacity == 0 {
                return Err(ModelError::Cuda(
                    "KV capacity must be positive before graph warmup".into(),
                ));
            }
            let max_bucket = self.capacity.next_power_of_two().max(16);
            let mut bucket = 16usize;
            while bucket <= max_bucket {
                // Compile every shape and initialize the cuBLAS handle before
                // capture. Repeating slot zero is safe during warmup and is
                // overwritten when the allocator first assigns that slot.
                let context_slots = vec![0u32; bucket];
                let position =
                    (self.capacity - 1).min(self.model.config.max_position_embeddings - 1) as u32;
                self.forward_eager(
                    &[self.model.config.bos_token_id],
                    &[position],
                    &[0],
                    &context_slots,
                    true,
                )?;
                let graph = DecodeGraph::capture(self, 1, bucket)?;
                self.decode_graphs.insert((1, bucket), graph);
                bucket = bucket.checked_mul(2).ok_or_else(|| {
                    ModelError::Cuda("decode graph bucket size overflowed".into())
                })?;
            }
            self.execution_stats = BackendExecutionStats {
                graph_captures: self.decode_graphs.len() as u64,
                ..BackendExecutionStats::default()
            };
            Ok(())
        }

        fn gemm(
            &self,
            weight: Bf16Tensor,
            input: Bf16Tensor,
            output_size: usize,
            rows: usize,
            input_size: usize,
            operation: &'static str,
        ) -> Result<Bf16Tensor, ModelError> {
            let output = api::zeros::<bf16>(&[rows, output_size])
                .sync_on(&self.stream)
                .map_err(|error| cuda_error("allocate GEMM output", error))?;
            let output = cublas::gemm_bf16(
                weight.clone(),
                input.clone(),
                output,
                output_size,
                rows,
                input_size,
            )
            .map_err(|error| ModelError::Cuda(format!("{operation}: {error}")))?
            .sync_on(&self.stream)
            .map_err(|error| cuda_error(operation, error))?
            .map_err(|error| ModelError::Cuda(format!("{operation}: {error}")))?;
            Ok(Arc::new(output))
        }

        fn rms_norm(
            &self,
            input: Bf16Tensor,
            weight: Bf16Tensor,
            rows: usize,
        ) -> Result<Bf16Tensor, ModelError> {
            let output = api::zeros::<bf16>(&[rows, self.model.config.hidden_size])
                .partition([1, self.model.config.hidden_size])
                .sync_on(&self.stream)
                .map_err(|error| cuda_error("allocate RMSNorm output", error))?;
            let (_, _, output, _) = unsafe {
                kernels::rms_norm_bf16(&input, &weight, output, self.model.config.rms_norm_eps)
            }
            .generics(vec![
                self.model.config.hidden_size.to_string(),
                HIDDEN_BLOCK.to_string(),
            ])
            .sync_on(&self.stream)
            .map_err(|error| cuda_error("RMSNorm", error))?;
            Ok(Arc::new(output.unpartition()))
        }

        fn add_rms_norm(
            &self,
            residual: Bf16Tensor,
            update: Bf16Tensor,
            weight: Bf16Tensor,
            rows: usize,
        ) -> Result<(Bf16Tensor, Bf16Tensor), ModelError> {
            let normalized = api::zeros::<bf16>(&[rows, self.model.config.hidden_size])
                .partition([1, self.model.config.hidden_size])
                .sync_on(&self.stream)
                .map_err(|error| cuda_error("allocate fused RMSNorm output", error))?;
            let combined = api::zeros::<bf16>(&[rows, self.model.config.hidden_size])
                .partition([1, self.model.config.hidden_size])
                .sync_on(&self.stream)
                .map_err(|error| cuda_error("allocate residual output", error))?;
            let (_, _, _, normalized, combined, _) = unsafe {
                kernels::add_rms_norm_bf16(
                    &residual,
                    &update,
                    &weight,
                    normalized,
                    combined,
                    self.model.config.rms_norm_eps,
                )
            }
            .generics(vec![
                self.model.config.hidden_size.to_string(),
                HIDDEN_BLOCK.to_string(),
            ])
            .sync_on(&self.stream)
            .map_err(|error| cuda_error("fused residual RMSNorm", error))?;
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
        gathered_key: Tensor<bf16>,
        gathered_value: Tensor<bf16>,
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
            cfg: &super::Config,
            batch_size: usize,
            context_bucket: usize,
            stream: &Arc<Stream>,
        ) -> Result<Self, ModelError> {
            Ok(Self {
                attention_input: zeros_bf16(&[batch_size, cfg.hidden_size], stream)?,
                residual: zeros_bf16(&[batch_size, cfg.hidden_size], stream)?,
                query: zeros_bf16(&[batch_size, cfg.q_width()], stream)?,
                key: zeros_bf16(&[batch_size, cfg.kv_width()], stream)?,
                value: zeros_bf16(&[batch_size, cfg.kv_width()], stream)?,
                rotated_query: zeros_bf16(
                    &[batch_size, cfg.num_attention_heads, cfg.head_dim],
                    stream,
                )?,
                rotated_key: zeros_bf16(
                    &[batch_size, cfg.num_key_value_heads, cfg.head_dim],
                    stream,
                )?,
                gathered_key: zeros_bf16(
                    &[
                        batch_size * cfg.num_key_value_heads,
                        context_bucket,
                        cfg.head_dim,
                    ],
                    stream,
                )?,
                gathered_value: zeros_bf16(
                    &[
                        batch_size * cfg.num_key_value_heads,
                        context_bucket,
                        cfg.head_dim,
                    ],
                    stream,
                )?,
                attention: zeros_bf16(
                    &[batch_size, cfg.num_attention_heads, cfg.head_dim],
                    stream,
                )?,
                attention_flat: zeros_bf16(&[batch_size, cfg.hidden_size], stream)?,
                attention_output: zeros_bf16(&[batch_size, cfg.hidden_size], stream)?,
                mlp_input: zeros_bf16(&[batch_size, cfg.hidden_size], stream)?,
                hidden_after_attention: zeros_bf16(&[batch_size, cfg.hidden_size], stream)?,
                gate: zeros_bf16(&[batch_size, cfg.intermediate_size], stream)?,
                up: zeros_bf16(&[batch_size, cfg.intermediate_size], stream)?,
                activated: zeros_bf16(&[batch_size, cfg.intermediate_size], stream)?,
                down: zeros_bf16(&[batch_size, cfg.hidden_size], stream)?,
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
        context_slots: Tensor<u32>,
        context_lengths: Tensor<i32>,
        logits: Arc<Tensor<bf16>>,
        sampled_token: Arc<Tensor<u32>>,
        batch_size: usize,
        context_bucket: usize,
        _storage: DecodeGraphStorage,
    }

    impl DecodeGraph {
        fn capture(
            runtime: &CudaLlama,
            batch_size: usize,
            context_bucket: usize,
        ) -> Result<Self, ModelError> {
            let cfg = &runtime.model.config;
            let stream = runtime.stream.clone();
            let token_ids = copy_u32(&vec![0; batch_size], &stream, "graph token IDs")?;
            let positions = copy_u32(&vec![0; batch_size], &stream, "graph positions")?;
            let current_slots = copy_u32(
                &vec![runtime.capacity as u32; batch_size],
                &stream,
                "graph current slots",
            )?;
            let context_slots = copy_u32(
                &vec![runtime.capacity as u32; batch_size * context_bucket],
                &stream,
                "graph context slots",
            )?;
            let context_lengths = copy_i32(&vec![1; batch_size], &stream, "graph context lengths")?;
            let mut embedding_hidden = zeros_bf16(&[batch_size, cfg.hidden_size], &stream)?;
            let mut layers: Vec<_> = (0..cfg.num_hidden_layers)
                .map(|_| DecodeLayerBuffers::allocate(cfg, batch_size, context_bucket, &stream))
                .collect::<Result<_, _>>()?;
            let mut final_hidden = zeros_bf16(&[batch_size, cfg.hidden_size], &stream)?;
            let mut final_residual = zeros_bf16(&[batch_size, cfg.hidden_size], &stream)?;
            let logits = Arc::new(zeros_bf16(&[batch_size, cfg.vocab_size], &stream)?);
            let argmax_blocks = cfg.vocab_size.div_ceil(ARGMAX_BLOCK);
            let argmax_reduce_block = argmax_blocks.next_power_of_two();
            let mut argmax_block_max = api::zeros::<f32>(&[batch_size, argmax_blocks])
                .sync_on(&stream)
                .map_err(|error| cuda_error("allocate graph argmax maxima", error))?;
            let mut argmax_block_index = api::zeros::<u32>(&[batch_size, argmax_blocks])
                .sync_on(&stream)
                .map_err(|error| cuda_error("allocate graph argmax indices", error))?;
            let mut sampled_token = api::zeros::<u32>(&[batch_size])
                .sync_on(&stream)
                .map_err(|error| cuda_error("allocate graph sampled token", error))?;

            let graph = CudaGraph::scope(&stream, |scope| {
                scope.record(
                    kernels::embedding_bf16(
                        &token_ids,
                        &runtime.weights.embedding,
                        (&mut embedding_hidden).partition([1, HIDDEN_BLOCK]),
                    )
                    .generics(vec![cfg.hidden_size.to_string(), HIDDEN_BLOCK.to_string()]),
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
                            .generics(vec![cfg.hidden_size.to_string(), HIDDEN_BLOCK.to_string()]),
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
                            .generics(vec![cfg.hidden_size.to_string(), HIDDEN_BLOCK.to_string()]),
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
                    scope.record(
                        kernels::rope_q_bf16(
                            &query,
                            &positions,
                            &runtime.cosine,
                            &runtime.sine,
                            (&mut layer.rotated_query).partition([1, 1, cfg.head_dim]),
                        )
                        .generics(vec![
                            cfg.head_dim.to_string(),
                            (cfg.head_dim / 2).to_string(),
                        ]),
                    )?;
                    scope.record(
                        unsafe {
                            kernels::rope_kv_write_bf16(
                                &key,
                                &value,
                                &positions,
                                &current_slots,
                                &runtime.cosine,
                                &runtime.sine,
                                runtime.key_cache[layer_index].device_pointer(),
                                runtime.value_cache[layer_index].device_pointer(),
                                (&mut layer.rotated_key).partition([1, 1, cfg.head_dim]),
                            )
                        }
                        .generics(vec![
                            cfg.head_dim.to_string(),
                            (cfg.head_dim / 2).to_string(),
                            cfg.num_key_value_heads.to_string(),
                        ]),
                    )?;
                    let context_slots_view = context_slots
                        .view(&[batch_size, context_bucket])
                        .map_err(device_error)?;
                    scope.record(
                        kernels::gather_flat_kv_decode_batch_bf16(
                            &context_slots_view,
                            &runtime.key_cache[layer_index],
                            (&mut layer.gathered_key).partition([1, 1, cfg.head_dim]),
                        )
                        .generics(vec![
                            cfg.head_dim.to_string(),
                            cfg.num_key_value_heads.to_string(),
                        ]),
                    )?;
                    scope.record(
                        kernels::gather_flat_kv_decode_batch_bf16(
                            &context_slots_view,
                            &runtime.value_cache[layer_index],
                            (&mut layer.gathered_value).partition([1, 1, cfg.head_dim]),
                        )
                        .generics(vec![
                            cfg.head_dim.to_string(),
                            cfg.num_key_value_heads.to_string(),
                        ]),
                    )?;
                    scope.record(
                        unsafe {
                            kernels::decode_attention_batch_bf16(
                                &layer.rotated_query,
                                &layer.gathered_key,
                                &layer.gathered_value,
                                &context_lengths,
                                (&mut layer.attention).partition([1, 1, cfg.head_dim]),
                                1.0 / (cfg.head_dim as f32).sqrt(),
                                (cfg.num_attention_heads / cfg.num_key_value_heads) as i32,
                            )
                        }
                        .generics(vec![
                            ATTENTION_KEY_BLOCK.to_string(),
                            cfg.head_dim.to_string(),
                            cfg.num_key_value_heads.to_string(),
                        ]),
                    )?;
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
                        .generics(vec![cfg.hidden_size.to_string(), HIDDEN_BLOCK.to_string()]),
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
                            (&mut layer.activated).partition([1, MLP_BLOCK]),
                        )
                        .generics(vec![MLP_BLOCK.to_string()]),
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
                    .generics(vec![cfg.hidden_size.to_string(), HIDDEN_BLOCK.to_string()]),
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
                    .generics(vec![ARGMAX_BLOCK.to_string()]),
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
            if token_ids.len() != self.batch_size
                || positions.len() != self.batch_size
                || current_slots.len() != self.batch_size
                || contexts.len() != self.batch_size
                || contexts
                    .iter()
                    .any(|context| context.is_empty() || context.len() > self.context_bucket)
                || (!greedy && self.batch_size != 1)
            {
                return Err(ModelError::Cuda(
                    "invalid decode graph batch metadata".into(),
                ));
            }
            let token_ids = copy_u32(token_ids, &self.stream, "decode graph tokens")?;
            let positions = copy_u32(positions, &self.stream, "decode graph positions")?;
            let current_slots = copy_u32(current_slots, &self.stream, "decode graph KV slots")?;
            let mut padded_context_slots =
                Vec::with_capacity(self.batch_size * self.context_bucket);
            let mut context_lengths = Vec::with_capacity(self.batch_size);
            for context in contexts {
                context_lengths
                    .push(i32::try_from(context.len()).map_err(|_| {
                        ModelError::Cuda("decode graph context exceeds i32".into())
                    })?);
                padded_context_slots.extend_from_slice(context);
                padded_context_slots.resize(
                    padded_context_slots.len() + self.context_bucket - context.len(),
                    sentinel_slot,
                );
            }
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
                if self.batch_size == 1 {
                    return Ok(ForwardOutput::Token(sampled[0]));
                }
                return Ok(ForwardOutput::Tokens(sampled));
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

    fn zeros_bf16(shape: &[usize], stream: &Arc<Stream>) -> Result<Tensor<bf16>, ModelError> {
        api::zeros::<bf16>(shape)
            .sync_on(stream)
            .map_err(|error| cuda_error("allocate decode graph buffer", error))
    }

    fn device_error(error: impl std::fmt::Debug) -> DeviceError {
        DeviceError::Internal(format!("decode graph tensor view: {error:?}"))
    }

    struct LlamaCudaBackend {
        runtime: CudaLlama,
        requests: HashMap<RequestId, CudaRequest>,
    }

    struct CudaRequest {
        prompt: Vec<u32>,
        generated: Vec<u32>,
        slots: Vec<u32>,
        decoder: Box<dyn IncrementalDecoder>,
        temperature: f32,
        top_p: f32,
        rng: SplitMix64,
    }

    impl Backend for LlamaCudaBackend {
        fn model(&self) -> Arc<dyn Model> {
            self.runtime.model.clone()
        }

        fn add_request(
            &mut self,
            request: &GenerateRequest,
        ) -> Result<PreparedRequest, BackendError> {
            if self.requests.contains_key(&request.id) {
                return Err(BackendError::InvalidRequest(format!(
                    "duplicate request {}",
                    request.id
                )));
            }
            let prompt = self
                .runtime
                .model
                .encode(&request.prompt)
                .map_err(invalid_request)?;
            if prompt.is_empty() {
                return Err(BackendError::InvalidRequest(
                    "prompt must encode to at least one token".into(),
                ));
            }
            let prompt_tokens = prompt.len();
            self.requests.insert(
                request.id,
                CudaRequest {
                    prompt,
                    generated: Vec::with_capacity(request.params.max_tokens),
                    slots: Vec::with_capacity(prompt_tokens + request.params.max_tokens),
                    decoder: self.runtime.model.decoder(),
                    temperature: request.params.temperature,
                    top_p: request.params.top_p,
                    rng: SplitMix64::new(request.params.seed),
                },
            );
            Ok(PreparedRequest { prompt_tokens })
        }

        fn step(&mut self, batch: &[ScheduledWork]) -> Result<Vec<StepOutput>, BackendError> {
            let mut outputs = Vec::with_capacity(batch.len());
            let packed_ids: HashSet<_> = batch
                .iter()
                .filter_map(|work| {
                    let request = self.requests.get(&work.request_id)?;
                    (work.sample
                        && work.num_tokens == 1
                        && work.position > 0
                        && request.temperature == 0.0)
                        .then_some(work.request_id)
                })
                .collect();
            if packed_ids.len() >= 2 {
                let mut request_ids = Vec::with_capacity(packed_ids.len());
                let mut token_ids = Vec::with_capacity(packed_ids.len());
                let mut positions = Vec::with_capacity(packed_ids.len());
                let mut current_slots = Vec::with_capacity(packed_ids.len());
                let mut contexts = Vec::with_capacity(packed_ids.len());
                for work in batch
                    .iter()
                    .filter(|work| packed_ids.contains(&work.request_id))
                {
                    let request = self.requests.get_mut(&work.request_id).ok_or_else(|| {
                        BackendError::Execution(format!("unknown request {}", work.request_id))
                    })?;
                    if work.kv_slots.len() != 1 || work.position != request.slots.len() {
                        return Err(BackendError::Execution(format!(
                            "invalid packed schedule metadata for request {}",
                            work.request_id
                        )));
                    }
                    let available = request.prompt.len() + request.generated.len();
                    if work.position >= available {
                        return Err(BackendError::Execution(format!(
                            "packed token position {} exceeds request token state {available}",
                            work.position
                        )));
                    }
                    let token_id = if work.position < request.prompt.len() {
                        request.prompt[work.position]
                    } else {
                        request.generated[work.position - request.prompt.len()]
                    };
                    request.slots.push(work.kv_slots[0]);
                    request_ids.push(work.request_id);
                    token_ids.push(token_id);
                    positions.push(u32::try_from(work.position).map_err(|_| {
                        BackendError::Execution("packed token position exceeds u32".into())
                    })?);
                    current_slots.push(work.kv_slots[0]);
                    contexts.push(request.slots.clone());
                }
                let sampled = self
                    .runtime
                    .forward_decode_batch(&token_ids, &positions, &current_slots, &contexts)
                    .map_err(execution)?;
                self.runtime.execution_stats.packed_decode_forwards += 1;
                self.runtime.execution_stats.packed_decode_requests += request_ids.len() as u64;
                if sampled.len() != request_ids.len() {
                    return Err(BackendError::Execution(format!(
                        "packed decode returned {} tokens for {} requests",
                        sampled.len(),
                        request_ids.len()
                    )));
                }
                for (request_id, token_id) in request_ids.into_iter().zip(sampled) {
                    let request = self.requests.get_mut(&request_id).ok_or_else(|| {
                        BackendError::Execution(format!("unknown request {request_id}"))
                    })?;
                    request.generated.push(token_id);
                    let text = request.decoder.push(token_id).map_err(execution)?;
                    outputs.push(StepOutput {
                        request_id,
                        token_id: Some(token_id),
                        text,
                        is_eos: self.runtime.model.eos_token_ids().contains(&token_id),
                    });
                }
            }
            for work in batch {
                if packed_ids.len() >= 2 && packed_ids.contains(&work.request_id) {
                    continue;
                }
                let request = self.requests.get_mut(&work.request_id).ok_or_else(|| {
                    BackendError::Execution(format!("unknown request {}", work.request_id))
                })?;
                if work.num_tokens == 0
                    || work.kv_slots.len() != work.num_tokens
                    || work.position != request.slots.len()
                {
                    return Err(BackendError::Execution(format!(
                        "invalid schedule metadata for request {}",
                        work.request_id
                    )));
                }

                let end = work
                    .position
                    .checked_add(work.num_tokens)
                    .ok_or_else(|| BackendError::Execution("token position overflowed".into()))?;
                let available = request.prompt.len() + request.generated.len();
                if end > available {
                    return Err(BackendError::Execution(format!(
                        "scheduled tokens {end} exceed request token state {available}"
                    )));
                }
                let token_ids: Vec<u32> = (work.position..end)
                    .map(|position| {
                        if position < request.prompt.len() {
                            request.prompt[position]
                        } else {
                            request.generated[position - request.prompt.len()]
                        }
                    })
                    .collect();
                let positions: Vec<u32> = (work.position..end)
                    .map(|position| {
                        u32::try_from(position).map_err(|_| {
                            BackendError::Execution("token position exceeds u32".into())
                        })
                    })
                    .collect::<Result<_, _>>()?;
                request.slots.extend_from_slice(&work.kv_slots);

                let forward = self
                    .runtime
                    .forward(
                        &token_ids,
                        &positions,
                        &work.kv_slots,
                        &request.slots,
                        work.sample,
                        request.temperature == 0.0,
                    )
                    .map_err(execution)?;
                if !work.sample {
                    continue;
                }
                let token_id = match forward {
                    ForwardOutput::Token(token_id) => token_id,
                    ForwardOutput::Logits(logits) => sample_token(
                        &logits,
                        request.temperature,
                        request.top_p,
                        &mut request.rng,
                    )?,
                    ForwardOutput::None | ForwardOutput::Tokens(_) => {
                        return Err(BackendError::Execution(
                            "sampled forward omitted logits and token".into(),
                        ));
                    }
                };
                request.generated.push(token_id);
                let text = request.decoder.push(token_id).map_err(execution)?;
                outputs.push(StepOutput {
                    request_id: work.request_id,
                    token_id: Some(token_id),
                    text,
                    is_eos: self.runtime.model.eos_token_ids().contains(&token_id),
                });
            }
            Ok(outputs)
        }

        fn remove_request(&mut self, request_id: RequestId) {
            self.requests.remove(&request_id);
        }

        fn take_execution_stats(&mut self) -> BackendExecutionStats {
            std::mem::take(&mut self.runtime.execution_stats)
        }
    }

    #[derive(Debug)]
    struct SplitMix64 {
        state: u64,
    }

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn unit_f64(&mut self) -> f64 {
            self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut value = self.state;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^= value >> 31;
            ((value >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
        }
    }

    fn sample_token(
        logits: &[f32],
        temperature: f32,
        top_p: f32,
        rng: &mut SplitMix64,
    ) -> Result<u32, BackendError> {
        if logits.is_empty() || logits.iter().any(|logit| !logit.is_finite()) {
            return Err(BackendError::Execution(
                "sampler received empty or non-finite logits".into(),
            ));
        }
        if temperature == 0.0 {
            return logits
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(token, _)| token as u32)
                .ok_or_else(|| BackendError::Execution("sampler received no logits".into()));
        }

        let inverse_temperature = 1.0 / temperature;
        let max = logits
            .iter()
            .copied()
            .map(|logit| logit * inverse_temperature)
            .max_by(f32::total_cmp)
            .ok_or_else(|| BackendError::Execution("sampler received no logits".into()))?;
        let mut candidates: Vec<(u32, f64)> = logits
            .iter()
            .enumerate()
            .map(|(token, logit)| {
                (
                    token as u32,
                    ((*logit * inverse_temperature - max) as f64).exp(),
                )
            })
            .collect();
        candidates.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
        let total: f64 = candidates.iter().map(|(_, weight)| weight).sum();
        if !total.is_finite() || total <= 0.0 {
            return Err(BackendError::Execution(
                "sampler probability mass is invalid".into(),
            ));
        }

        let cutoff = total * top_p as f64;
        let mut retained_mass = 0.0;
        let mut retained = 0usize;
        for (_, weight) in &candidates {
            retained_mass += weight;
            retained += 1;
            if retained_mass >= cutoff {
                break;
            }
        }
        let draw = rng.unit_f64() * retained_mass;
        let mut cumulative = 0.0;
        for (token, weight) in candidates.into_iter().take(retained) {
            cumulative += weight;
            if draw < cumulative {
                return Ok(token);
            }
        }
        Err(BackendError::Execution(
            "sampler failed to select a token".into(),
        ))
    }

    fn invalid_request(error: ModelError) -> BackendError {
        BackendError::InvalidRequest(error.to_string())
    }

    fn execution(error: ModelError) -> BackendError {
        BackendError::Execution(error.to_string())
    }

    pub(super) fn load_backend(
        model_id: &str,
        model_dir: &Path,
        device_id: usize,
        kv_capacity_tokens: usize,
    ) -> Result<Box<dyn Backend>, ModelError> {
        let model = Arc::new(Llama32::load(model_id, model_dir)?);
        let mut runtime = CudaLlama::load(model, device_id, kv_capacity_tokens)?;
        runtime.warm_decode_graphs()?;
        tracing::info!(
            graph_buckets = runtime.decode_graphs.len(),
            kv_capacity_tokens,
            "Llama CUDA backend loaded and decode graphs warmed"
        );
        Ok(Box::new(LlamaCudaBackend {
            runtime,
            requests: HashMap::new(),
        }))
    }

    pub(super) fn validate(
        model_id: &str,
        model_dir: &Path,
        device_id: usize,
    ) -> Result<CudaModelReport, ModelError> {
        let model = Llama32::load(model_id, model_dir)?;
        let device = Device::new(device_id).map_err(|error| {
            ModelError::Cuda(format!("initialize device {device_id}: {error:?}"))
        })?;
        let stream = device
            .new_stream()
            .map_err(|error| ModelError::Cuda(format!("create stream: {error:?}")))?;
        let weights = DeviceWeights::load(&model.weights, &stream)?;

        let name = "model.norm.weight";
        let expected = model.weights.tensor(name)?;
        let actual: Vec<bf16> = weights
            .get(name)?
            .to_host_vec()
            .sync_on(&stream)
            .map_err(|error| ModelError::Cuda(format!("verify `{name}`: {error:?}")))?;
        let matches = actual
            .iter()
            .zip(expected.data().chunks_exact(2))
            .all(|(actual, bytes)| actual.to_bits() == u16::from_le_bytes([bytes[0], bytes[1]]));
        if !matches || actual.len() * 2 != expected.data().len() {
            return Err(ModelError::Cuda(format!(
                "BF16 round-trip mismatch for `{name}`"
            )));
        }
        Ok(CudaModelReport {
            model_id: model.id,
            device_id,
            tensors: weights.tensors.len(),
            bytes: weights.bytes,
        })
    }

    pub(super) fn validate_next_token(
        model_id: &str,
        model_dir: &Path,
        device_id: usize,
        prompt: &str,
    ) -> Result<CudaForwardReport, ModelError> {
        let model = Llama32::load(model_id, model_dir)?;
        let token_ids = model.tokenizer.encode(prompt)?;
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
        let mut runtime = CudaLlama::load(Arc::new(model), device_id, capacity)?;
        let logits = match runtime.forward(&token_ids, &positions, &slots, &slots, true, false)? {
            ForwardOutput::Logits(logits) => logits,
            ForwardOutput::None | ForwardOutput::Token(_) | ForwardOutput::Tokens(_) => {
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
        let next_token_text = runtime.model.tokenizer.decode(&[next_token_id])?;
        Ok(CudaForwardReport {
            model_id: runtime.model.id.clone(),
            prompt_tokens: token_ids.len(),
            next_token_id,
            next_token_text,
            top_logits: ranked,
        })
    }

    fn rope_tables(config: &super::Config, stream: &Arc<Stream>) -> Result<RopeTables, ModelError> {
        let half = config.head_dim / 2;
        let mut inverse_frequency = Vec::with_capacity(half);
        for index in 0..half {
            let exponent = (2 * index) as f32 / config.head_dim as f32;
            let base_frequency = 1.0 / config.rope_theta.powf(exponent);
            let wavelength = TAU / base_frequency;
            let low_wavelength = config.rope_scaling.original_max_position_embeddings as f32
                / config.rope_scaling.low_freq_factor;
            let high_wavelength = config.rope_scaling.original_max_position_embeddings as f32
                / config.rope_scaling.high_freq_factor;
            let frequency = if wavelength < high_wavelength {
                base_frequency
            } else if wavelength > low_wavelength {
                base_frequency / config.rope_scaling.factor
            } else {
                let smooth = (config.rope_scaling.original_max_position_embeddings as f32
                    / wavelength
                    - config.rope_scaling.low_freq_factor)
                    / (config.rope_scaling.high_freq_factor - config.rope_scaling.low_freq_factor);
                (1.0 - smooth) * base_frequency / config.rope_scaling.factor
                    + smooth * base_frequency
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
}

fn render_chat(messages: &[ChatMessage<'_>]) -> Result<String, ModelError> {
    if messages.is_empty() {
        return Err(ModelError::InvalidInput(
            "messages must contain at least one item".into(),
        ));
    }
    if messages.iter().any(|message| message.content.is_empty()) {
        return Err(ModelError::InvalidInput(
            "message content must not be empty".into(),
        ));
    }

    let mut prompt = String::from("<|begin_of_text|>");
    for message in messages {
        let role = match message.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        };
        prompt.push_str("<|start_header_id|>");
        prompt.push_str(role);
        prompt.push_str("<|end_header_id|>\n\n");
        prompt.push_str(message.content);
        prompt.push_str("<|eot_id|>");
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    Ok(prompt)
}

struct Decoder {
    tokenizer: Tokenizer,
    token_ids: Vec<u32>,
    decoded: String,
}

impl IncrementalDecoder for Decoder {
    fn push(&mut self, token_id: u32) -> Result<String, ModelError> {
        self.token_ids.push(token_id);
        let decoded = self.tokenizer.decode(&self.token_ids)?;
        let delta = decoded
            .strip_prefix(&self.decoded)
            .unwrap_or(&decoded)
            .to_owned();
        self.decoded = decoded;
        Ok(delta)
    }
}

fn validate_weights(weights: &WeightStore, config: &Config) -> Result<(), ModelError> {
    weights.validate_bf16(
        "model.embed_tokens.weight",
        &[config.vocab_size, config.hidden_size],
    )?;
    weights.validate_bf16("model.norm.weight", &[config.hidden_size])?;
    if !config.tie_word_embeddings {
        weights.validate_bf16("lm_head.weight", &[config.vocab_size, config.hidden_size])?;
    }

    for layer in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        weights.validate_bf16(
            &format!("{prefix}.input_layernorm.weight"),
            &[config.hidden_size],
        )?;
        weights.validate_bf16(
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[config.hidden_size],
        )?;
        weights.validate_bf16(
            &format!("{prefix}.self_attn.q_proj.weight"),
            &[config.q_width(), config.hidden_size],
        )?;
        weights.validate_bf16(
            &format!("{prefix}.self_attn.k_proj.weight"),
            &[config.kv_width(), config.hidden_size],
        )?;
        weights.validate_bf16(
            &format!("{prefix}.self_attn.v_proj.weight"),
            &[config.kv_width(), config.hidden_size],
        )?;
        weights.validate_bf16(
            &format!("{prefix}.self_attn.o_proj.weight"),
            &[config.hidden_size, config.q_width()],
        )?;
        weights.validate_bf16(
            &format!("{prefix}.mlp.gate_proj.weight"),
            &[config.intermediate_size, config.hidden_size],
        )?;
        weights.validate_bf16(
            &format!("{prefix}.mlp.up_proj.weight"),
            &[config.intermediate_size, config.hidden_size],
        )?;
        weights.validate_bf16(
            &format!("{prefix}.mlp.down_proj.weight"),
            &[config.hidden_size, config.intermediate_size],
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_json(dtype: &str) -> String {
        format!(
            r#"{{
                "architectures":["LlamaForCausalLM"],
                "attention_bias":false,
                "bos_token_id":128000,
                "eos_token_id":[128001,128008,128009],
                "head_dim":64,
                "hidden_act":"silu",
                "hidden_size":2048,
                "intermediate_size":8192,
                "max_position_embeddings":131072,
                "mlp_bias":false,
                "model_type":"llama",
                "num_attention_heads":32,
                "num_hidden_layers":16,
                "num_key_value_heads":8,
                "rms_norm_eps":0.00001,
                "rope_scaling":{{"factor":32.0,"high_freq_factor":4.0,"low_freq_factor":1.0,"original_max_position_embeddings":8192,"rope_type":"llama3"}},
                "rope_theta":500000.0,
                "tie_word_embeddings":true,
                "torch_dtype":"{dtype}",
                "vocab_size":128256
            }}"#
        )
    }

    #[test]
    fn accepts_target_config() {
        let config: Config = serde_json::from_str(&config_json("bfloat16")).unwrap();
        config.validate().unwrap();
        assert_eq!(config.q_width(), 2048);
        assert_eq!(config.kv_width(), 512);
    }

    #[test]
    fn rejects_non_bf16_model() {
        let config: Config = serde_json::from_str(&config_json("float16")).unwrap();
        assert!(matches!(
            config.validate(),
            Err(ModelError::InvalidConfig(message)) if message.contains("bfloat16")
        ));
    }

    #[test]
    fn owns_its_chat_template() {
        let prompt = render_chat(&[
            ChatMessage {
                role: ChatRole::System,
                content: "Be terse.",
            },
            ChatMessage {
                role: ChatRole::User,
                content: "Hello",
            },
        ])
        .unwrap();
        assert_eq!(
            prompt,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe terse.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn rejects_empty_chat() {
        assert!(matches!(
            render_chat(&[]),
            Err(ModelError::InvalidInput(message))
                if message.contains("messages must contain at least one item")
        ));
    }
}
