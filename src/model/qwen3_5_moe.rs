use std::{collections::HashMap, path::Path, sync::Arc};

use serde::Deserialize;

use super::{
    ChatMessage, ChatRole, IncrementalDecoder, Model, ModelError, ModelSummary, read_file,
    tokenizer::Tokenizer,
    weights::{SafeTensorSource, WeightSource},
};

#[cfg(feature = "cuda")]
use super::{CudaForwardReport, CudaModelReport};

#[cfg(feature = "cuda")]
pub(super) fn load_cuda_executor(
    _model_dir: &Path,
    _device_id: usize,
    _kv_capacity_tokens: usize,
    _max_batch_tokens: usize,
    _max_running: usize,
) -> Result<Box<dyn crate::engine::ModelExecutor>, ModelError> {
    Err(runtime_pending())
}

#[cfg(feature = "cuda")]
pub(super) fn validate_cuda_model(
    _model_id: &str,
    _model_dir: &Path,
    _device_id: usize,
) -> Result<CudaModelReport, ModelError> {
    Err(runtime_pending())
}

#[cfg(feature = "cuda")]
pub(super) fn validate_cuda_next_token(
    _model_id: &str,
    _model_dir: &Path,
    _device_id: usize,
    _prompt: &str,
) -> Result<CudaForwardReport, ModelError> {
    Err(runtime_pending())
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

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
enum LayerQuantization {
    #[serde(rename = "FP8")]
    Fp8,
    #[serde(rename = "MXFP8")]
    Mxfp8,
    #[serde(rename = "NVFP4")]
    Nvfp4,
    #[serde(rename = "W4A16_NVFP4")]
    W4A16Nvfp4,
}

#[derive(Debug, Deserialize)]
struct QuantizedLayer {
    quant_algo: LayerQuantization,
    group_size: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct QuantizationConfig {
    quant_algo: String,
    quantized_layers: HashMap<String, QuantizedLayer>,
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
        if self.quant_method != "modelopt" || self.quant_algo != "MIXED_PRECISION" {
            return Err(ModelError::InvalidConfig(
                "Qwen3.6 requires a ModelOpt mixed-precision export".into(),
            ));
        }
        if self.quantized_layers.is_empty()
            || self.quantized_layers.iter().any(|(name, layer)| {
                name.is_empty()
                    || matches!(
                        layer.quant_algo,
                        LayerQuantization::Nvfp4 | LayerQuantization::W4A16Nvfp4
                    ) && layer.group_size != Some(16)
            })
        {
            return Err(ModelError::InvalidConfig(
                "Qwen3.6 contains empty or unrepresentable per-layer quantization metadata".into(),
            ));
        }
        Ok(())
    }
}

const TOKENIZER_WARMUP_TEXT: &str = concat!(
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
    "<|im_start|>user\nWarm tokenizer paths: abc XYZ 0123456789, café 東京 😀<|im_end|>\n",
    "<|im_start|>assistant\n<think>\n",
);

pub(super) struct Qwen35MoeText {
    config: Config,
    tokenizer: Tokenizer,
    weights: Arc<dyn WeightSource>,
    eos_token_ids: [u32; 1],
}

impl Qwen35MoeText {
    pub(super) const ARCH_NAME: &'static str = "Qwen3_5MoeForConditionalGeneration";

    pub(super) fn load(model_dir: &Path) -> Result<Self, ModelError> {
        let config = Config::load(model_dir)?;
        let weights: Arc<dyn WeightSource> = Arc::new(SafeTensorSource::open(model_dir)?);
        let tokenizer = Tokenizer::load(model_dir)?;
        tokenizer.warm(TOKENIZER_WARMUP_TEXT)?;
        let eos_token_ids = [config.text_config.eos_token_id];
        Ok(Self {
            config,
            tokenizer,
            weights,
            eos_token_ids,
        })
    }
}

impl Model for Qwen35MoeText {
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
