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
use crate::cuda::qwen3_5_moe::{self as cuda_program, Artifact, Config as CudaConfig};

#[cfg(feature = "cuda")]
pub(super) fn load_cuda_executor(
    model_dir: &Path,
    device_id: usize,
    kv_capacity_tokens: usize,
    max_batch_tokens: usize,
    max_running: usize,
) -> Result<Box<dyn crate::engine::ModelExecutor>, ModelError> {
    let model = Arc::new(Qwen35MoeText::load(model_dir)?);
    cuda_program::load_executor(
        model.cuda_artifact(),
        device_id,
        kv_capacity_tokens,
        max_batch_tokens,
        max_running,
    )
}

#[cfg(feature = "cuda")]
pub(super) fn cuda_model_report(
    model_id: &str,
    model_dir: &Path,
    device_id: usize,
) -> Result<CudaModelReport, ModelError> {
    let model = Arc::new(Qwen35MoeText::load(model_dir)?);
    cuda_program::checkpoint_report(model_id, model.cuda_artifact(), device_id)
}

#[cfg(feature = "cuda")]
pub(super) fn cuda_forward_report(
    model_id: &str,
    model_dir: &Path,
    device_id: usize,
    prompt: &str,
) -> Result<CudaForwardReport, ModelError> {
    let model = Arc::new(Qwen35MoeText::load(model_dir)?);
    cuda_program::forward_report(model_id, model.cuda_artifact(), device_id, prompt)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Debug, Deserialize)]
struct TextConfig {
    attn_output_gate: bool,
    dtype: String,
    eos_token_id: u32,
    head_dim: usize,
    hidden_act: String,
    hidden_size: usize,
    #[cfg(feature = "cuda")]
    layer_types: Vec<LayerKind>,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_value_head_dim: usize,
    mamba_ssm_dtype: String,
    max_position_embeddings: usize,
    moe_intermediate_size: usize,
    num_attention_heads: usize,
    #[cfg(feature = "cuda")]
    num_experts: usize,
    num_experts_per_tok: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    partial_rotary_factor: f32,
    rms_norm_eps: f32,
    rope_parameters: RopeParameters,
    shared_expert_intermediate_size: usize,
    vocab_size: usize,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Debug, Deserialize)]
struct RopeParameters {
    mrope_interleaved: bool,
    mrope_section: Vec<usize>,
    rope_theta: f32,
    rope_type: String,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum LayerKind {
    LinearAttention,
    FullAttention,
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
    quantization_config: QuantizationConfig,
    text_config: TextConfig,
}

impl Config {
    fn load(model_dir: &Path) -> Result<Self, ModelError> {
        let path = model_dir.join("config.json");
        let text = read_file(&path)?;
        serde_json::from_str(&text).map_err(|source| ModelError::Json {
            path: path.clone(),
            source,
        })
    }
}

impl QuantizationConfig {
    fn target_counts(&self) -> ([usize; 4], usize) {
        let mut counts = [0; 4];
        let mut grouped = 0;
        for target in self.quantized_layers.values() {
            let bucket = match target.quant_algo {
                LayerQuantization::Fp8 => 0,
                LayerQuantization::Mxfp8 => 1,
                LayerQuantization::Nvfp4 => 2,
                LayerQuantization::W4A16Nvfp4 => 3,
            };
            counts[bucket] += 1;
            grouped += usize::from(target.group_size.is_some());
        }
        (counts, grouped)
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
        let ([fp8, mxfp8, nvfp4, w4a16_nvfp4], grouped) =
            config.quantization_config.target_counts();
        tracing::info!(
            method = %config.quantization_config.quant_method,
            algorithm = %config.quantization_config.quant_algo,
            targets = config.quantization_config.quantized_layers.len(),
            fp8,
            mxfp8,
            nvfp4,
            w4a16_nvfp4,
            grouped,
            "parsed checkpoint quantization metadata"
        );
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

#[cfg(feature = "cuda")]
impl Qwen35MoeText {
    fn cuda_artifact(self: &Arc<Self>) -> Artifact {
        Artifact {
            model: self.clone(),
            config: CudaConfig {
                attn_output_gate: self.config.text_config.attn_output_gate,
                head_dim: self.config.text_config.head_dim,
                hidden_act: self.config.text_config.hidden_act.clone(),
                hidden_size: self.config.text_config.hidden_size,
                layers: self
                    .config
                    .text_config
                    .layer_types
                    .iter()
                    .map(|kind| match kind {
                        LayerKind::LinearAttention => cuda_program::LayerKind::LinearAttention,
                        LayerKind::FullAttention => cuda_program::LayerKind::FullAttention,
                    })
                    .collect(),
                linear_conv_kernel_dim: self.config.text_config.linear_conv_kernel_dim,
                linear_key_head_dim: self.config.text_config.linear_key_head_dim,
                linear_num_key_heads: self.config.text_config.linear_num_key_heads,
                linear_num_value_heads: self.config.text_config.linear_num_value_heads,
                linear_value_head_dim: self.config.text_config.linear_value_head_dim,
                mamba_ssm_dtype: self.config.text_config.mamba_ssm_dtype.clone(),
                max_position_embeddings: self.config.text_config.max_position_embeddings,
                moe_intermediate_size: self.config.text_config.moe_intermediate_size,
                num_experts: self.config.text_config.num_experts,
                num_experts_per_tok: self.config.text_config.num_experts_per_tok,
                num_attention_heads: self.config.text_config.num_attention_heads,
                num_key_value_heads: self.config.text_config.num_key_value_heads,
                partial_rotary_factor: self.config.text_config.partial_rotary_factor,
                rms_norm_eps: self.config.text_config.rms_norm_eps,
                rope_interleaved: self.config.text_config.rope_parameters.mrope_interleaved,
                rope_section: self
                    .config
                    .text_config
                    .rope_parameters
                    .mrope_section
                    .clone(),
                rope_theta: self.config.text_config.rope_parameters.rope_theta,
                rope_type: self.config.text_config.rope_parameters.rope_type.clone(),
                shared_expert_intermediate_size: self
                    .config
                    .text_config
                    .shared_expert_intermediate_size,
                vocab_size: self.config.text_config.vocab_size,
            },
            weights: self.weights.clone(),
        }
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
