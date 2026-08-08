use std::path::Path;

use serde::Deserialize;

use super::{ModelError, read_file};

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    pub factor: f32,
    pub high_freq_factor: f32,
    pub low_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    pub architectures: Vec<String>,
    pub attention_bias: bool,
    pub bos_token_id: u32,
    pub eos_token_id: Vec<u32>,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub mlp_bias: bool,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_scaling: RopeScaling,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub vocab_size: usize,
}

impl LlamaConfig {
    pub fn from_model_dir(model_dir: &Path) -> Result<Self, ModelError> {
        let path = model_dir.join("config.json");
        let text = read_file(&path)?;
        let config: Self = serde_json::from_str(&text).map_err(|source| ModelError::Json {
            path: path.clone(),
            source,
        })?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        let invalid = |message: &str| Err(ModelError::InvalidConfig(message.into()));
        if self.model_type != "llama"
            || !self
                .architectures
                .iter()
                .any(|architecture| architecture == "LlamaForCausalLM")
        {
            return invalid("v1 supports only LlamaForCausalLM");
        }
        if self.torch_dtype != "bfloat16" {
            return invalid("v1 requires torch_dtype=bfloat16");
        }
        if self.hidden_act != "silu" {
            return invalid("v1 requires the SiLU gated MLP");
        }
        if self.attention_bias || self.mlp_bias {
            return invalid("v1 does not support attention or MLP bias tensors");
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
            return invalid("v1 requires a valid Llama 3 RoPE scaling configuration");
        }
        if self.max_position_embeddings < self.rope_scaling.original_max_position_embeddings {
            return invalid("max_position_embeddings is smaller than the original RoPE context");
        }
        if self.eos_token_id.is_empty() {
            return invalid("at least one EOS token is required");
        }
        Ok(())
    }

    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn q_width(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
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
    fn accepts_llama_32_1b_shape() {
        let config: LlamaConfig = serde_json::from_str(&config_json("bfloat16")).unwrap();
        config.validate().unwrap();
        assert_eq!(config.num_kv_groups(), 4);
        assert_eq!(config.q_width(), 2048);
        assert_eq!(config.kv_width(), 512);
    }

    #[test]
    fn rejects_non_bf16_model() {
        let config: LlamaConfig = serde_json::from_str(&config_json("float16")).unwrap();
        assert!(matches!(
            config.validate(),
            Err(ModelError::InvalidConfig(message)) if message.contains("bfloat16")
        ));
    }
}
