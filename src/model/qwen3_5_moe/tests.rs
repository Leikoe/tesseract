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
    let fp8 = [
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
    ];
    let nvfp4 = ["model.language_model.layers.0.mlp.experts"];
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
fn rejects_a_layer_schedule_that_only_has_the_right_counts() {
    let mut config = target_config();
    config.text_config.layer_types.swap(0, 3);
    assert!(matches!(
        config.validate(),
        Err(ModelError::InvalidConfig(message)) if message.contains("schedule")
    ));
}

#[test]
fn parses_quantization_targets_without_reconstructing_the_manifest() {
    let mut config = target_config();
    config
        .quantization_config
        .config_groups
        .get_mut("group_0")
        .unwrap()
        .targets
        .pop();
    config.validate().unwrap();
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
