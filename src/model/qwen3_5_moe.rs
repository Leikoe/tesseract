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
    _model_dir: &Path,
    _device_id: usize,
    _kv_capacity_tokens: usize,
    _max_batch_tokens: usize,
    _max_running: usize,
) -> Result<Box<dyn crate::engine::ModelExecutor>, ModelError> {
    Err(runtime_pending())
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
        "Qwen3.5/3.6 MoE text checkpoint parsing is implemented, but its hybrid CUDA program is not yet implemented"
            .into(),
    )
}

#[derive(Debug, Deserialize)]
struct TextConfig {
    dtype: String,
    eos_token_id: u32,
    hidden_size: usize,
    #[cfg(feature = "cuda")]
    layer_types: Vec<LayerKind>,
    num_attention_heads: usize,
    #[cfg(feature = "cuda")]
    num_experts: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    vocab_size: usize,
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
                num_experts: self.config.text_config.num_experts,
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
