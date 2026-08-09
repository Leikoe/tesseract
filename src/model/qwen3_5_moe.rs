use std::{collections::HashSet, path::Path, sync::Arc};

use serde::Deserialize;

use super::{
    ArchitectureFactory, ChatMessage, ChatRole, IncrementalDecoder, Model, ModelError,
    ModelManifest, ModelSummary, read_file,
    tokenizer::Tokenizer,
    weights::{SafeTensorSource, WeightSource},
};

#[cfg(feature = "cuda")]
use super::{CudaForwardReport, CudaModelReport};

pub(super) static FACTORY: Factory = Factory;

pub(super) struct Factory;

impl ArchitectureFactory for Factory {
    fn name(&self) -> &'static str {
        "qwen3_5_moe_text"
    }

    fn probe(&self, manifest: &ModelManifest) -> bool {
        manifest.model_type == "qwen3_5_moe"
            && manifest
                .architectures
                .iter()
                .any(|architecture| architecture == "Qwen3_5MoeForConditionalGeneration")
    }

    fn load(&self, model_id: &str, model_dir: &Path) -> Result<Arc<dyn Model>, ModelError> {
        Qwen35MoeText::load(model_id, model_dir).map(|model| Arc::new(model) as Arc<dyn Model>)
    }

    #[cfg(feature = "cuda")]
    fn load_cuda_executor(
        &self,
        _model_id: &str,
        _model_dir: &Path,
        _device_id: usize,
        _kv_capacity_tokens: usize,
        _max_batch_tokens: usize,
        _max_running: usize,
    ) -> Result<Box<dyn crate::engine::ModelExecutor>, ModelError> {
        Err(runtime_pending())
    }

    #[cfg(feature = "cuda")]
    fn validate_cuda_model(
        &self,
        _model_id: &str,
        _model_dir: &Path,
        _device_id: usize,
    ) -> Result<CudaModelReport, ModelError> {
        Err(runtime_pending())
    }

    #[cfg(feature = "cuda")]
    fn validate_cuda_next_token(
        &self,
        _model_id: &str,
        _model_dir: &Path,
        _device_id: usize,
        _prompt: &str,
    ) -> Result<CudaForwardReport, ModelError> {
        Err(runtime_pending())
    }
}

#[cfg(feature = "cuda")]
fn runtime_pending() -> ModelError {
    ModelError::UnsupportedExecution(
        "Qwen3.5/3.6 MoE text checkpoint validation is implemented, but its hybrid CUDA program is not yet implemented"
            .into(),
    )
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum LayerKind {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Deserialize)]
struct RopeParameters {
    mrope_interleaved: bool,
    mrope_section: Vec<usize>,
    partial_rotary_factor: f32,
    rope_theta: f32,
    rope_type: String,
}

#[derive(Debug, Deserialize)]
struct TextConfig {
    attention_bias: bool,
    attn_output_gate: bool,
    bos_token_id: u32,
    dtype: String,
    eos_token_id: u32,
    full_attention_interval: usize,
    head_dim: usize,
    hidden_act: String,
    hidden_size: usize,
    layer_types: Vec<LayerKind>,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_value_head_dim: usize,
    mamba_ssm_dtype: String,
    max_position_embeddings: usize,
    model_type: String,
    moe_intermediate_size: usize,
    num_attention_heads: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    partial_rotary_factor: f32,
    rms_norm_eps: f32,
    rope_parameters: RopeParameters,
    shared_expert_intermediate_size: usize,
    tie_word_embeddings: bool,
    use_cache: bool,
    vocab_size: usize,
}

#[derive(Debug, Deserialize)]
struct QuantSpec {
    dynamic: bool,
    num_bits: usize,
    #[serde(rename = "type")]
    kind: String,
    group_size: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct QuantGroup {
    input_activations: QuantSpec,
    weights: QuantSpec,
    targets: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct QuantizationConfig {
    config_groups: std::collections::HashMap<String, QuantGroup>,
    quant_method: String,
}

#[derive(Debug, Deserialize)]
struct Config {
    architectures: Vec<String>,
    dtype: String,
    model_type: String,
    quantization_config: QuantizationConfig,
    text_config: TextConfig,
    tie_word_embeddings: bool,
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
        let text = &self.text_config;
        if self.model_type != "qwen3_5_moe"
            || !self
                .architectures
                .iter()
                .any(|architecture| architecture == "Qwen3_5MoeForConditionalGeneration")
            || text.model_type != "qwen3_5_moe_text"
        {
            return invalid("Qwen3.6 text requires Qwen3_5MoeForConditionalGeneration");
        }
        if self.dtype != "bfloat16" || text.dtype != "bfloat16" {
            return invalid("Qwen3.6 text residual dtype must be bfloat16");
        }
        if self.tie_word_embeddings || text.tie_word_embeddings {
            return invalid("Qwen3.6 text requires an untied LM head");
        }
        if text.hidden_act != "silu" || text.attention_bias || !text.attn_output_gate {
            return invalid("Qwen3.6 text requires bias-free gated attention and SiLU MoE");
        }
        if !text.use_cache
            || text.hidden_size == 0
            || text.head_dim == 0
            || text.num_hidden_layers == 0
            || text.vocab_size == 0
        {
            return invalid("Qwen3.6 text dimensions and cache configuration are invalid");
        }
        if text.num_attention_heads == 0
            || text.num_key_value_heads == 0
            || !text
                .num_attention_heads
                .is_multiple_of(text.num_key_value_heads)
            || text.hidden_size != text.num_attention_heads * text.linear_key_head_dim
        {
            return invalid("Qwen3.6 attention head geometry is inconsistent");
        }
        if text.layer_types.len() != text.num_hidden_layers || text.full_attention_interval == 0 {
            return invalid("Qwen3.6 layer schedule length is inconsistent");
        }
        for (layer, kind) in text.layer_types.iter().enumerate() {
            let expected = if (layer + 1).is_multiple_of(text.full_attention_interval) {
                LayerKind::FullAttention
            } else {
                LayerKind::LinearAttention
            };
            if *kind != expected {
                return Err(ModelError::InvalidConfig(format!(
                    "layer {layer} violates the hybrid attention schedule"
                )));
            }
        }
        if text.linear_conv_kernel_dim == 0
            || text.linear_num_key_heads == 0
            || text.linear_num_value_heads == 0
            || text.linear_key_head_dim == 0
            || text.linear_value_head_dim == 0
            || text.mamba_ssm_dtype != "float32"
        {
            return invalid("Qwen3.6 linear-attention geometry is invalid");
        }
        if text.num_experts == 0
            || text.num_experts_per_tok == 0
            || text.num_experts_per_tok > text.num_experts
            || text.moe_intermediate_size == 0
            || text.shared_expert_intermediate_size == 0
        {
            return invalid("Qwen3.6 MoE geometry is invalid");
        }
        if text.rms_norm_eps <= 0.0 || !text.rms_norm_eps.is_finite() {
            return invalid("Qwen3.6 RMSNorm epsilon must be finite and positive");
        }
        let rotary = text.partial_rotary_factor * text.head_dim as f32;
        let rope = &text.rope_parameters;
        if rope.rope_type != "default"
            || !rope.mrope_interleaved
            || rope.partial_rotary_factor != text.partial_rotary_factor
            || rope.rope_theta <= 0.0
            || !rope.rope_theta.is_finite()
            || rotary.fract() != 0.0
            || rope.mrope_section.iter().sum::<usize>() * 2 != rotary as usize
        {
            return invalid("Qwen3.6 partial interleaved RoPE configuration is invalid");
        }
        if text.max_position_embeddings == 0
            || text.bos_token_id as usize >= text.vocab_size
            || text.eos_token_id as usize >= text.vocab_size
        {
            return invalid("Qwen3.6 context or special-token configuration is invalid");
        }
        self.quantization_config.parse_targets()
    }
}

impl QuantizationConfig {
    fn parse_targets(&self) -> Result<(), ModelError> {
        if self.quant_method != "modelopt" {
            return Err(ModelError::InvalidConfig(
                "Qwen3.6 requires a ModelOpt mixed-precision export".into(),
            ));
        }
        let mut saw_fp8 = false;
        let mut saw_nvfp4 = false;
        let mut targets = HashSet::new();
        for group in self.config_groups.values() {
            let weights = &group.weights;
            let activations = &group.input_activations;
            let is_fp8 = weights.kind == "float"
                && activations.kind == "float"
                && !weights.dynamic
                && !activations.dynamic
                && weights.num_bits == 8
                && activations.num_bits == 8
                && weights.group_size.is_none()
                && activations.group_size.is_none();
            let is_nvfp4 = weights.kind == "float"
                && activations.kind == "float"
                && !weights.dynamic
                && !activations.dynamic
                && weights.num_bits == 4
                && activations.num_bits == 4
                && weights.group_size == Some(16)
                && activations.group_size == Some(16);
            if !is_fp8 && !is_nvfp4 {
                return Err(ModelError::InvalidConfig(
                    "Qwen3.6 contains an unsupported ModelOpt quantization group".into(),
                ));
            }
            if group.targets.is_empty()
                || group
                    .targets
                    .iter()
                    .any(|target| target.is_empty() || !targets.insert(target.clone()))
            {
                return Err(ModelError::InvalidConfig(
                    "Qwen3.6 quantization targets are empty or duplicated".into(),
                ));
            }
            saw_fp8 |= is_fp8;
            saw_nvfp4 |= is_nvfp4;
        }
        if !saw_fp8 || !saw_nvfp4 {
            return Err(ModelError::InvalidConfig(
                "Qwen3.6 requires both FP8 and NVFP4 quantization groups".into(),
            ));
        }
        // Quantized layer artifacts parse their own target entry and tensor
        // representation at construction. Do not reconstruct the producer's
        // full manifest here merely to compare it back to the checkpoint.
        Ok(())
    }
}

const TOKENIZER_WARMUP_TEXT: &str = concat!(
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
    "<|im_start|>user\nWarm tokenizer paths: abc XYZ 0123456789, café 東京 😀<|im_end|>\n",
    "<|im_start|>assistant\n<think>\n",
);

struct Qwen35MoeText {
    id: String,
    config: Config,
    tokenizer: Tokenizer,
    weights: Arc<dyn WeightSource>,
    eos_token_ids: [u32; 1],
}

impl Qwen35MoeText {
    fn load(model_id: &str, model_dir: &Path) -> Result<Self, ModelError> {
        let config = Config::load(model_dir)?;
        let weights: Arc<dyn WeightSource> = Arc::new(SafeTensorSource::open(model_dir)?);
        let tokenizer = Tokenizer::load(model_dir)?;
        tokenizer.warm(TOKENIZER_WARMUP_TEXT)?;
        let eos_token_ids = [config.text_config.eos_token_id];
        Ok(Self {
            id: model_id.into(),
            config,
            tokenizer,
            weights,
            eos_token_ids,
        })
    }
}

impl Model for Qwen35MoeText {
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
        self.tokenizer.incremental_decoder()
    }

    fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    fn summary(&self) -> ModelSummary {
        let text = &self.config.text_config;
        ModelSummary {
            id: self.id.clone(),
            architecture: self.config.architectures[0].clone(),
            dtype: text.dtype.clone(),
            layers: text.num_hidden_layers,
            hidden_size: text.hidden_size,
            attention_heads: text.num_attention_heads,
            kv_heads: text.num_key_value_heads,
            vocab_size: text.vocab_size,
            tensors: self.weights.tensor_count(),
        }
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
    if messages
        .iter()
        .enumerate()
        .any(|(index, message)| index != 0 && message.role == ChatRole::System)
    {
        return Err(ModelError::InvalidInput(
            "system message must be the first message".into(),
        ));
    }

    let mut prompt = String::new();
    for message in messages {
        let role = match message.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        };
        prompt.push_str("<|im_start|>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(message.content.trim());
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n<think>\n");
    Ok(prompt)
}

#[cfg(test)]
mod tests;
