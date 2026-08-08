use serde_json::json;

use super::*;

fn target_config() -> Config {
    let layer_types: Vec<_> = (0..40)
        .map(|layer| {
            if (layer + 1) % 4 == 0 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    let text_for_targets = TextConfig {
        attention_bias: false,
        attn_output_gate: true,
        bos_token_id: 248044,
        dtype: "bfloat16".into(),
        eos_token_id: 248044,
        full_attention_interval: 4,
        head_dim: 256,
        hidden_act: "silu".into(),
        hidden_size: 2048,
        layer_types: (0..40)
            .map(|layer| {
                if (layer + 1) % 4 == 0 {
                    LayerKind::FullAttention
                } else {
                    LayerKind::LinearAttention
                }
            })
            .collect(),
        linear_conv_kernel_dim: 4,
        linear_key_head_dim: 128,
        linear_num_key_heads: 16,
        linear_num_value_heads: 32,
        linear_value_head_dim: 128,
        mamba_ssm_dtype: "float32".into(),
        max_position_embeddings: 262144,
        model_type: "qwen3_5_moe_text".into(),
        moe_intermediate_size: 512,
        num_attention_heads: 16,
        num_experts: 256,
        num_experts_per_tok: 8,
        num_hidden_layers: 40,
        num_key_value_heads: 2,
        partial_rotary_factor: 0.25,
        rms_norm_eps: 1e-6,
        rope_parameters: RopeParameters {
            mrope_interleaved: true,
            mrope_section: vec![11, 11, 10],
            partial_rotary_factor: 0.25,
            rope_theta: 10_000_000.0,
            rope_type: "default".into(),
        },
        shared_expert_intermediate_size: 512,
        tie_word_embeddings: false,
        use_cache: true,
        vocab_size: 248320,
    };
    let fp8: Vec<_> = expected_fp8_targets(&text_for_targets)
        .into_iter()
        .collect();
    let nvfp4: Vec<_> = expected_nvfp4_targets(&text_for_targets)
        .into_iter()
        .collect();
    let value = json!({
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "dtype": "bfloat16",
        "model_type": "qwen3_5_moe",
        "tie_word_embeddings": false,
        "text_config": {
            "attention_bias": false,
            "attn_output_gate": true,
            "bos_token_id": 248044,
            "dtype": "bfloat16",
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "layer_types": layer_types,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_moe_text",
            "moe_intermediate_size": 512,
            "num_attention_heads": 16,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 40,
            "num_key_value_heads": 2,
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"mrope_interleaved":true,"mrope_section":[11,11,10],"partial_rotary_factor":0.25,"rope_theta":10000000.0,"rope_type":"default"},
            "shared_expert_intermediate_size": 512,
            "tie_word_embeddings": false,
            "use_cache": true,
            "vocab_size": 248320
        },
        "quantization_config": {
            "quant_method": "modelopt",
            "config_groups": {
                "group_0": {"input_activations":{"dynamic":false,"num_bits":8,"type":"float"},"weights":{"dynamic":false,"num_bits":8,"type":"float"},"targets":fp8},
                "group_1": {"input_activations":{"dynamic":false,"num_bits":4,"type":"float","group_size":16},"weights":{"dynamic":false,"num_bits":4,"type":"float","group_size":16},"targets":nvfp4}
            }
        }
    });
    serde_json::from_value(value).unwrap()
}

#[test]
fn accepts_pinned_text_config_and_exact_tensor_manifest() {
    let config = target_config();
    config.validate().unwrap();
    let tensors = text_tensor_contracts(&config.text_config).unwrap();
    assert_eq!(tensors.len(), 124_116);
    assert_eq!(
        tensors
            .iter()
            .find(|tensor| tensor.name.ends_with("layers.3.self_attn.q_proj.weight"))
            .unwrap()
            .shape,
        [8192, 2048]
    );
}

#[test]
fn rejects_a_layer_schedule_that_only_has_the_right_counts() {
    let mut config = target_config();
    config.text_config.layer_types.swap(0, 3);
    assert!(matches!(
        config.validate(),
        Err(ModelError::InvalidConfig(message)) if message.contains("schedule")
    ));
}

#[test]
fn rejects_incomplete_quantization_targets() {
    let mut config = target_config();
    config
        .quantization_config
        .config_groups
        .get_mut("group_0")
        .unwrap()
        .targets
        .pop();
    assert!(matches!(
        config.validate(),
        Err(ModelError::InvalidConfig(message)) if message.contains("target set")
    ));
}

#[test]
fn renders_text_only_qwen_generation_prompt() {
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
        "<|im_start|>system\nBe terse.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
}
