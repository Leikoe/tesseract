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
pub(super) fn load_cuda_executor(
    model_id: &str,
    model_dir: &Path,
    device_id: usize,
    kv_capacity_tokens: usize,
    max_batch_tokens: usize,
    max_running: usize,
) -> Result<Box<dyn crate::engine::ModelExecutor>, ModelError> {
    cuda_impl::load_executor(
        model_id,
        model_dir,
        device_id,
        kv_capacity_tokens,
        max_batch_tokens,
        max_running,
    )
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

const TOKENIZER_WARMUP_TEXT: &str = concat!(
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
    "You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
    "Warm tokenizer paths: abc XYZ 0123456789, punctuation !?; café 東京 😀\n",
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
);

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

    fn warm(&self) -> Result<(), ModelError> {
        let token_ids = self.encode(TOKENIZER_WARMUP_TEXT)?;
        if token_ids.is_empty() {
            return Err(ModelError::Tokenizer(
                "tokenizer warmup produced no tokens".into(),
            ));
        }
        Ok(())
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
        // The tokenizers regex/BPE machinery initializes lazily. Since prompt
        // preparation runs on the single executor thread, paying that cost on
        // the first admitted request leaves the A100 idle for hundreds of
        // milliseconds. Readiness includes representative tokenizer paths so
        // the first user request sees steady-state preprocessing latency.
        tokenizer.warm()?;
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
            kernels,
        },
        engine::{ExecutionError, ExecutionStats, ModelExecutor, TokenId},
        model::Model,
    };

    use super::{
        CudaForwardReport, CudaModelReport, CudaTokenLogit, Llama32, ModelError, WeightStore,
        execution_bucket, warmup_logical_sizes,
    };

    const HIDDEN_BLOCK: usize = 512;
    const MLP_BLOCK: usize = 512;
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

    struct CudaLlama<A> {
        model: Arc<Llama32>,
        stream: Arc<Stream>,
        weights: RuntimeWeights,
        attention: A,
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
        num_key_value_heads: usize,
        head_dim: usize,
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
            let valid =
                rows > 0
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
                        !context.is_empty()
                            && context.iter().all(|slot| (*slot as usize) < capacity)
                    })
                    && self
                        .request_indices
                        .iter()
                        .all(|index| (*index as usize) < self.contexts.len())
                    && self.context_lengths.iter().all(|length| *length > 0)
                    && self.request_indices.iter().zip(self.context_lengths).all(
                        |(index, length)| *length as usize <= self.contexts[*index as usize].len(),
                    )
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
            config: &super::Config,
            stream: &Arc<Stream>,
            capacity: usize,
            scratch_slots: usize,
            cosine: Arc<Tensor<f32>>,
            sine: Arc<Tensor<f32>>,
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
                let key_cache =
                    api::zeros::<bf16>(&cache_shape)
                        .sync_on(stream)
                        .map_err(|error| {
                            ModelError::Cuda(format!("allocate layer {layer} key cache: {error:?}"))
                        })?;
                let value_cache =
                    api::zeros::<bf16>(&cache_shape)
                        .sync_on(stream)
                        .map_err(|error| {
                            ModelError::Cuda(format!(
                                "allocate layer {layer} value cache: {error:?}"
                            ))
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
                num_key_value_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
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
            let rotated_query = output_buffer_async::<bf16>(
                &[input.rows, self.num_attention_heads, self.head_dim],
                input.stream,
                "rotated query",
            )?
            .partition([1, 1, self.head_dim]);
            let (_, _, _, _, rotated_query) = enqueue(
                kernels::rope_q_bf16(
                    input.query,
                    input.positions,
                    &self.cosine,
                    &self.sine,
                    rotated_query,
                )
                .generics(vec![
                    self.head_dim.to_string(),
                    (self.head_dim / 2).to_string(),
                ]),
                input.stream,
                "query RoPE",
            )?;
            let rotated_query = Arc::new(rotated_query.unpartition());

            let rotated_key = output_buffer_async::<bf16>(
                &[input.rows, self.num_key_value_heads, self.head_dim],
                input.stream,
                "rotated key",
            )?
            .partition([1, 1, self.head_dim]);
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
                    self.head_dim.to_string(),
                    (self.head_dim / 2).to_string(),
                    self.num_key_value_heads.to_string(),
                ]),
                input.stream,
                "key RoPE and flat KV write",
            )?;

            let attention = output_buffer_async::<bf16>(
                &[input.rows, self.num_attention_heads, self.head_dim],
                input.stream,
                "ragged attention output",
            )?
            .partition([1, 1, self.head_dim]);
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
                        1.0 / (self.head_dim as f32).sqrt(),
                        (self.num_attention_heads / self.num_key_value_heads) as i32,
                    )
                }
                .generics(vec![
                    ATTENTION_KEY_BLOCK.to_string(),
                    self.head_dim.to_string(),
                    self.num_key_value_heads.to_string(),
                ]),
                input.stream,
                "ragged flat-KV attention",
            )?;
            Ok(attention.unpartition())
        }

        fn record_decode(&self, input: DecodeGraphAttention<'_>) -> Result<(), Self::Error> {
            let state = self.layer_state(input.layer)?;
            input.scope.record(
                kernels::rope_q_bf16(
                    input.query,
                    input.positions,
                    &self.cosine,
                    &self.sine,
                    input.rotated_query.partition([1, 1, self.head_dim]),
                )
                .generics(vec![
                    self.head_dim.to_string(),
                    (self.head_dim / 2).to_string(),
                ]),
            )?;
            input.scope.record(
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
                        input.rotated_key.partition([1, 1, self.head_dim]),
                    )
                }
                .generics(vec![
                    self.head_dim.to_string(),
                    (self.head_dim / 2).to_string(),
                    self.num_key_value_heads.to_string(),
                ]),
            )?;
            input.scope.record(
                unsafe {
                    kernels::ragged_attention_bf16(
                        input.rotated_query,
                        input.request_indices,
                        input.context_slots,
                        input.context_lengths,
                        state.key_cache.device_pointer(),
                        state.value_cache.device_pointer(),
                        input.attention.partition([1, 1, self.head_dim]),
                        1.0 / (self.head_dim as f32).sqrt(),
                        (self.num_attention_heads / self.num_key_value_heads) as i32,
                    )
                }
                .generics(vec![
                    ATTENTION_KEY_BLOCK.to_string(),
                    self.head_dim.to_string(),
                    self.num_key_value_heads.to_string(),
                ]),
            )?;
            Ok(())
        }
    }

    impl CudaLlama<DirectFlatKvAttention> {
        fn load(
            model: Arc<Llama32>,
            device_id: usize,
            capacity: usize,
            max_running: usize,
            max_batch_tokens: usize,
        ) -> Result<Self, ModelError> {
            let device = Device::new(device_id).map_err(|error| {
                ModelError::Cuda(format!("initialize device {device_id}: {error:?}"))
            })?;
            let stream = device
                .new_stream()
                .map_err(|error| ModelError::Cuda(format!("create stream: {error:?}")))?;
            let weights = RuntimeWeights::load(&model, &stream)?;
            let (cosine, sine) = rope_tables(&model.config, &stream)?;
            let max_decode_batch_bucket = execution_bucket(max_running, 1)
                .ok_or_else(|| ModelError::Cuda("max running batch bucket overflowed".into()))?;
            let max_query_bucket = execution_bucket(max_batch_tokens, 1)
                .ok_or_else(|| ModelError::Cuda("max query bucket overflowed".into()))?;
            let attention = DirectFlatKvAttention::load(
                &model.config,
                &stream,
                capacity,
                max_decode_batch_bucket.max(max_query_bucket),
                cosine,
                sine,
            )?;
            Ok(Self {
                model,
                stream,
                weights,
                attention,
                capacity,
                max_decode_batch_bucket,
                max_query_bucket,
                decode_graphs: HashMap::new(),
                failed_decode_graphs: HashSet::new(),
                execution_stats: ExecutionStats::default(),
            })
        }
    }

    impl<A: AttentionBackend<Error = ModelError>> CudaLlama<A> {
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
                let context_bucket = execution_bucket(context_slots.len(), 16)
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
            let batch_bucket = execution_bucket(token_ids.len(), 1)
                .ok_or_else(|| ModelError::Cuda("decode batch bucket overflowed".into()))?;
            let context_bucket =
                execution_bucket(contexts.iter().map(Vec::len).max().unwrap_or(0), 16)
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
                ForwardOutput::None | ForwardOutput::Logits(_) | ForwardOutput::BatchLogits(_) => {
                    Err(ModelError::Cuda(
                        "packed decode omitted sampled tokens".into(),
                    ))
                }
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
            let cfg = &self.model.config;
            let stream = &self.stream;
            let rows = execution_bucket(logical_rows, 1)
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
            let context_bucket = execution_bucket(self.capacity, 16)
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
                    .partition([1, HIDDEN_BLOCK]);
            let (_, _, hidden) = enqueue(
                kernels::embedding_bf16(&token_ids, &*self.weights.embedding, hidden)
                    .generics(vec![cfg.hidden_size.to_string(), HIDDEN_BLOCK.to_string()]),
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
                .partition([1, MLP_BLOCK]);
                let (_, _, activated) = enqueue(
                    kernels::silu_mul_bf16(&gate, &up, activated)
                        .generics(vec![MLP_BLOCK.to_string()]),
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
                drop(rotated_key);
            }

            if sample_rows.is_empty() {
                completion.synchronize("complete ragged forward")?;
                return Ok(ForwardOutput::None);
            }

            let (residual, update) = pending
                .ok_or_else(|| ModelError::Cuda("model has no transformer layers".into()))?;
            let (final_hidden, _final_residual) =
                self.add_rms_norm(&residual, &update, &self.weights.final_norm, rows)?;
            let samples = sample_rows.len();
            let sample_bucket = execution_bucket(samples, 1)
                .ok_or_else(|| ModelError::Cuda("sample row bucket size overflowed".into()))?;
            let mut padded_sample_rows = sample_rows.to_vec();
            padded_sample_rows.resize(sample_bucket, sample_rows[0]);
            let sample_rows_device = copy_u32(&padded_sample_rows, stream, "sample rows")?;
            let sampled_hidden = output_buffer_async::<bf16>(
                &[sample_bucket, cfg.hidden_size],
                stream,
                "sampled hidden states",
            )?
            .partition([1, HIDDEN_BLOCK]);
            let (_, _, sampled_hidden) = enqueue(
                unsafe {
                    kernels::gather_rows_bf16(
                        final_hidden.device_pointer(),
                        &sample_rows_device,
                        sampled_hidden,
                    )
                }
                .generics(vec![cfg.hidden_size.to_string(), HIDDEN_BLOCK.to_string()]),
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
                let argmax_blocks = cfg.vocab_size.div_ceil(ARGMAX_BLOCK);
                let reduce_block = execution_bucket(argmax_blocks, 1)
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
                    .generics(vec![ARGMAX_BLOCK.to_string()]),
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
            let batch_bucket = execution_bucket(batch_size, 1)
                .filter(|bucket| *bucket <= self.max_decode_batch_bucket)
                .ok_or_else(|| ModelError::Cuda("decode graph batch bucket overflowed".into()))?;
            let context_bucket =
                execution_bucket(contexts.iter().map(Vec::len).max().unwrap_or(0), 16).ok_or_else(
                    || ModelError::Cuda("decode graph context bucket overflowed".into()),
                )?;
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
                .min(self.model.config.max_position_embeddings);
            let max_logical_samples = max_running.min(self.capacity);
            let max_sample_bucket = execution_bucket(max_logical_samples, 1)
                .ok_or_else(|| ModelError::Cuda("sample warmup bucket overflowed".into()))?;
            let decode_batch_warmup_sizes = warmup_logical_sizes(max_logical_samples, 1)
                .ok_or_else(|| ModelError::Cuda("decode batch warmup sizes overflowed".into()))?;
            let query_warmup_sizes = warmup_logical_sizes(max_logical_queries, 1)
                .ok_or_else(|| ModelError::Cuda("query warmup sizes overflowed".into()))?;
            for logical_queries in query_warmup_sizes {
                let query_bucket = execution_bucket(logical_queries, 1)
                    .ok_or_else(|| ModelError::Cuda("query warmup bucket overflowed".into()))?;
                let token_ids = vec![self.model.config.bos_token_id; logical_queries];
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
                        u32::try_from(row % warm_requests).map_err(|_| {
                            ModelError::Cuda("warmup request index exceeds u32".into())
                        })
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
                let bucket = execution_bucket(logical_context, 16)
                    .ok_or_else(|| ModelError::Cuda("context warmup bucket overflowed".into()))?;
                // Compile every shape and initialize the cuBLAS handle before
                // capture. Repeating slot zero is safe during warmup and is
                // overwritten when the allocator first assigns that slot.
                let context_slots = vec![0u32; bucket];
                let position = (bucket - 1)
                    .min(self.capacity - 1)
                    .min(self.model.config.max_position_embeddings - 1)
                    as u32;
                self.forward_eager(
                    &[self.model.config.bos_token_id],
                    &[position],
                    &[0],
                    &context_slots,
                    true,
                    true,
                )?;
                for logical_batch in &decode_batch_warmup_sizes {
                    let batch_bucket = execution_bucket(*logical_batch, 1).ok_or_else(|| {
                        ModelError::Cuda("decode batch warmup bucket overflowed".into())
                    })?;
                    let mut graph = DecodeGraph::capture(self, batch_bucket, bucket)?;
                    // Capture records work but does not pay every first-launch
                    // cost. Replay each graph once before readiness using one
                    // real warmup row and isolated padding slots so user
                    // requests never trigger module/graph initialization.
                    graph.forward(
                        &[self.model.config.bos_token_id],
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
                &[rows, self.model.config.hidden_size],
                &self.stream,
                "RMSNorm output",
            )?
            .partition([1, self.model.config.hidden_size]);
            let (_, _, output, _) = enqueue(
                unsafe {
                    kernels::rms_norm_bf16(input, weight, output, self.model.config.rms_norm_eps)
                }
                .generics(vec![
                    self.model.config.hidden_size.to_string(),
                    HIDDEN_BLOCK.to_string(),
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
                &[rows, self.model.config.hidden_size],
                &self.stream,
                "fused RMSNorm output",
            )?
            .partition([1, self.model.config.hidden_size]);
            let combined = output_buffer_async::<bf16>(
                &[rows, self.model.config.hidden_size],
                &self.stream,
                "residual output",
            )?
            .partition([1, self.model.config.hidden_size]);
            let (_, _, _, normalized, combined, _) = enqueue(
                unsafe {
                    kernels::add_rms_norm_bf16(
                        residual,
                        update,
                        weight,
                        normalized,
                        combined,
                        self.model.config.rms_norm_eps,
                    )
                }
                .generics(vec![
                    self.model.config.hidden_size.to_string(),
                    HIDDEN_BLOCK.to_string(),
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
            cfg: &super::Config,
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
                rotated_key: output_bf16(
                    &[batch_size, cfg.num_key_value_heads, cfg.head_dim],
                    stream,
                )?,
                attention: output_bf16(
                    &[batch_size, cfg.num_attention_heads, cfg.head_dim],
                    stream,
                )?,
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
            runtime: &CudaLlama<A>,
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
            let argmax_blocks = cfg.vocab_size.div_ceil(ARGMAX_BLOCK);
            let argmax_reduce_block = execution_bucket(argmax_blocks, 1)
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
            let current_slots =
                copy_u32(&padded_current_slots, &self.stream, "decode graph KV slots")?;
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
        // SAFETY: CudaLlama is thread-confined and all eager work uses this one
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

    impl<A: AttentionBackend<Error = ModelError>> ModelProgram for CudaLlama<A> {
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
                        vocab_size: self.model.config.vocab_size,
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

    pub(super) fn load_executor(
        model_id: &str,
        model_dir: &Path,
        device_id: usize,
        kv_capacity_tokens: usize,
        max_batch_tokens: usize,
        max_running: usize,
    ) -> Result<Box<dyn ModelExecutor>, ModelError> {
        let model = Arc::new(Llama32::load(model_id, model_dir)?);
        let mut runtime = CudaLlama::load(
            model,
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
            "Llama CUDA executor loaded and execution buckets warmed"
        );
        Ok(Box::new(CudaExecutor::new(runtime)))
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
        let mut runtime =
            CudaLlama::load(Arc::new(model), device_id, capacity, 1, token_ids.len())?;
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

#[cfg(any(feature = "cuda", test))]
fn execution_bucket(logical_size: usize, minimum: usize) -> Option<usize> {
    if logical_size == 0 || minimum == 0 {
        return None;
    }
    logical_size
        .checked_next_power_of_two()
        .map(|bucket| bucket.max(minimum))
}

#[cfg(any(feature = "cuda", test))]
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
    use proptest::prelude::*;
    use std::collections::BTreeSet;

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

    #[test]
    fn execution_buckets_reject_empty_dimensions() {
        assert_eq!(execution_bucket(0, 1), None);
        assert_eq!(execution_bucket(1, 0), None);
        assert_eq!(warmup_logical_sizes(0, 1), None);
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
