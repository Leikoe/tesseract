use std::{path::Path, sync::Arc};

use serde::Deserialize;

use super::{
    ChatMessage, ChatRole, IncrementalDecoder, Model, ModelError, ModelSummary, read_file,
    tokenizer::Tokenizer,
    weights::{SafeTensorSource, WeightSource},
};

#[cfg(feature = "cuda")]
use super::{CudaForwardReport, CudaModelReport};
#[cfg(feature = "cuda")]
use crate::cuda::dense_decoder::{
    self, DenseDecoderArtifact, DenseDecoderConfig, DenseDecoderWeightNames, DenseLayerWeightNames,
    Llama3RopeConfig,
};

#[cfg(feature = "cuda")]
pub(super) fn load_cuda_executor(
    model_dir: &Path,
    device_id: usize,
    kv_capacity_tokens: usize,
    max_batch_tokens: usize,
    max_running: usize,
) -> Result<Box<dyn crate::engine::ModelExecutor>, ModelError> {
    let model = Arc::new(Llama32::load(model_dir)?);
    dense_decoder::load_executor(
        model.dense_decoder_artifact()?,
        device_id,
        kv_capacity_tokens,
        max_batch_tokens,
        max_running,
    )
}

#[cfg(feature = "cuda")]
pub(super) fn validate_cuda_model(
    model_id: &str,
    model_dir: &Path,
    device_id: usize,
) -> Result<CudaModelReport, ModelError> {
    let model = Llama32::load(model_dir)?;
    dense_decoder::validate(
        model_id,
        model.weights.as_ref(),
        "model.norm.weight",
        device_id,
    )
}

#[cfg(feature = "cuda")]
pub(super) fn validate_cuda_next_token(
    model_id: &str,
    model_dir: &Path,
    device_id: usize,
    prompt: &str,
) -> Result<CudaForwardReport, ModelError> {
    let model = Arc::new(Llama32::load(model_dir)?);
    dense_decoder::validate_next_token(model_id, model.dense_decoder_artifact()?, device_id, prompt)
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

const TOKENIZER_WARMUP_TEXT: &str = concat!(
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
    "You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
    "Warm tokenizer paths: abc XYZ 0123456789, punctuation !?; café 東京 😀\n",
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
);

pub(super) struct Llama32 {
    config: Config,
    tokenizer: Tokenizer,
    weights: Arc<dyn WeightSource>,
}

impl Llama32 {
    pub(super) const ARCH_NAME: &'static str = "LlamaForCausalLM";

    pub(super) fn load(model_dir: &Path) -> Result<Self, ModelError> {
        let config = Config::load(model_dir)?;
        let weights: Arc<dyn WeightSource> = Arc::new(SafeTensorSource::open(model_dir)?);
        validate_weights(weights.as_ref(), &config)?;
        let tokenizer = Tokenizer::load(model_dir)?;
        // The tokenizers regex/BPE machinery initializes lazily. Since prompt
        // preparation runs on the single executor thread, paying that cost on
        // the first admitted request leaves the A100 idle for hundreds of
        // milliseconds. Readiness includes representative tokenizer paths so
        // the first user request sees steady-state preprocessing latency.
        tokenizer.warm(TOKENIZER_WARMUP_TEXT)?;
        Ok(Self {
            config,
            tokenizer,
            weights,
        })
    }
}

#[cfg(feature = "cuda")]
impl Llama32 {
    fn dense_decoder_artifact(self: &Arc<Self>) -> Result<DenseDecoderArtifact, ModelError> {
        let config = &self.config;
        let layers = (0..config.num_hidden_layers)
            .map(|layer| {
                let prefix = format!("model.layers.{layer}");
                DenseLayerWeightNames {
                    input_norm: format!("{prefix}.input_layernorm.weight"),
                    post_norm: format!("{prefix}.post_attention_layernorm.weight"),
                    query: format!("{prefix}.self_attn.q_proj.weight"),
                    key: format!("{prefix}.self_attn.k_proj.weight"),
                    value: format!("{prefix}.self_attn.v_proj.weight"),
                    output: format!("{prefix}.self_attn.o_proj.weight"),
                    gate: format!("{prefix}.mlp.gate_proj.weight"),
                    up: format!("{prefix}.mlp.up_proj.weight"),
                    down: format!("{prefix}.mlp.down_proj.weight"),
                }
            })
            .collect();
        DenseDecoderArtifact::try_new(
            self.clone(),
            DenseDecoderConfig {
                bos_token_id: config.bos_token_id,
                head_dim: config.head_dim,
                hidden_size: config.hidden_size,
                intermediate_size: config.intermediate_size,
                max_position_embeddings: config.max_position_embeddings,
                num_attention_heads: config.num_attention_heads,
                num_hidden_layers: config.num_hidden_layers,
                num_key_value_heads: config.num_key_value_heads,
                rms_norm_eps: config.rms_norm_eps,
                rope: Llama3RopeConfig {
                    factor: config.rope_scaling.factor,
                    high_frequency_factor: config.rope_scaling.high_freq_factor,
                    low_frequency_factor: config.rope_scaling.low_freq_factor,
                    original_max_positions: config.rope_scaling.original_max_position_embeddings,
                    theta: config.rope_theta,
                },
                vocab_size: config.vocab_size,
            },
            self.weights.clone(),
            DenseDecoderWeightNames {
                embedding: "model.embed_tokens.weight".into(),
                final_norm: "model.norm.weight".into(),
                lm_head: (!config.tie_word_embeddings).then(|| "lm_head.weight".into()),
                layers,
            },
        )
    }
}

impl Model for Llama32 {
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
        &self.config.eos_token_id
    }

    fn summary(&self) -> ModelSummary {
        ModelSummary {
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

fn validate_weights(weights: &dyn WeightSource, config: &Config) -> Result<(), ModelError> {
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
