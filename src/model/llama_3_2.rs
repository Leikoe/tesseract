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
    use std::{collections::HashMap, f32::consts::TAU, path::Path, sync::Arc};

    use cuda_async::device_operation::DeviceOp;
    use cuda_core::{Device, Stream};
    use cutile::{
        api,
        core::bf16,
        tensor::{Reshape, Tensor, ToHostVec},
        tile_kernel::{PartitionOp, TileKernel},
    };

    use crate::{
        cuda::{cublas, kernels},
        engine::{
            Backend, BackendError, GenerateRequest, PreparedRequest, RequestId, ScheduledWork,
            StepOutput,
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
                capacity,
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
            })
        }

        fn forward(
            &self,
            token_ids: &[u32],
            positions: &[u32],
            current_slots: &[u32],
            context_slots: &[u32],
            return_logits: bool,
        ) -> Result<Option<Vec<f32>>, ModelError> {
            let rows = token_ids.len();
            if rows == 0
                || positions.len() != rows
                || current_slots.len() != rows
                || context_slots.is_empty()
                || context_slots
                    .iter()
                    .chain(current_slots)
                    .any(|slot| *slot as usize >= self.capacity)
            {
                return Err(ModelError::Cuda("invalid forward batch metadata".into()));
            }
            let cfg = &self.model.config;
            let stream = &self.stream;
            let query_start = positions[0] as i32;
            let token_ids = copy_u32(token_ids, stream, "token IDs")?;
            let positions = copy_u32(positions, stream, "positions")?;
            let current_slots = copy_u32(current_slots, stream, "current KV slots")?;
            let context_slots = copy_u32(context_slots, stream, "context KV slots")?;

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

                let context_len = context_slots.shape()[0] as usize;
                let gathered_key =
                    api::zeros::<bf16>(&[cfg.num_key_value_heads, context_len, cfg.head_dim])
                        .partition([1, 1, cfg.head_dim])
                        .sync_on(stream)
                        .map_err(|error| cuda_error("allocate gathered key", error))?;
                let gathered_value =
                    api::zeros::<bf16>(&[cfg.num_key_value_heads, context_len, cfg.head_dim])
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

                let attention = api::zeros::<bf16>(&[rows, cfg.num_attention_heads, cfg.head_dim])
                    .partition([ATTENTION_QUERY_BLOCK, 1, cfg.head_dim])
                    .sync_on(stream)
                    .map_err(|error| cuda_error("allocate attention output", error))?;
                let (_, _, _, attention, _, _, _, _) = unsafe {
                    kernels::causal_attention_bf16(
                        &rotated_query,
                        &gathered_key,
                        &gathered_value,
                        attention,
                        1.0 / (cfg.head_dim as f32).sqrt(),
                        (cfg.num_attention_heads / cfg.num_key_value_heads) as i32,
                        context_len as i32,
                        query_start,
                    )
                }
                .generics(vec![
                    ATTENTION_QUERY_BLOCK.to_string(),
                    ATTENTION_KEY_BLOCK.to_string(),
                    cfg.head_dim.to_string(),
                ])
                .sync_on(stream)
                .map_err(|error| cuda_error("causal attention", error))?;
                let attention = Arc::new(
                    attention
                        .unpartition()
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
                return Ok(None);
            }

            let (residual, update) = pending
                .ok_or_else(|| ModelError::Cuda("model has no transformer layers".into()))?;
            let (final_hidden, _) =
                self.add_rms_norm(residual, update, self.weights.final_norm.clone(), rows)?;
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
            Ok(Some(logits.into_iter().map(bf16::to_f32).collect()))
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
            let output = cublas::gemm_bf16(weight, input, output, output_size, rows, input_size)
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
            for work in batch {
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

                let logits = self
                    .runtime
                    .forward(
                        &token_ids,
                        &positions,
                        &work.kv_slots,
                        &request.slots,
                        work.sample,
                    )
                    .map_err(execution)?;
                if !work.sample {
                    continue;
                }
                let logits = logits.ok_or_else(|| {
                    BackendError::Execution("sampled forward omitted logits".into())
                })?;
                let token_id = sample_token(
                    &logits,
                    request.temperature,
                    request.top_p,
                    &mut request.rng,
                )?;
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
        let runtime = CudaLlama::load(model, device_id, kv_capacity_tokens)?;
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
        let runtime = CudaLlama::load(Arc::new(model), device_id, capacity)?;
        let logits = runtime
            .forward(&token_ids, &positions, &slots, &slots, true)?
            .ok_or_else(|| ModelError::Cuda("forward omitted requested logits".into()))?;
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
}
