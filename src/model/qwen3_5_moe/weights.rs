use std::collections::HashSet;

use super::{LayerKind, TextConfig};
use crate::model::{
    ModelError,
    weights::{WeightDtype, WeightSource},
};

pub(super) struct TensorContract {
    pub(super) name: String,
    dtype: WeightDtype,
    pub(super) shape: Vec<usize>,
}

impl TensorContract {
    fn new(name: impl Into<String>, dtype: WeightDtype, shape: impl Into<Vec<usize>>) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape: shape.into(),
        }
    }
}

pub(super) fn text_tensor_contracts(
    config: &TextConfig,
) -> Result<Vec<TensorContract>, ModelError> {
    let fp8_targets = config
        .layer_types
        .iter()
        .map(|kind| match kind {
            LayerKind::LinearAttention => 3,
            LayerKind::FullAttention => 4,
        })
        .sum::<usize>();
    let linear_layers = config
        .layer_types
        .iter()
        .filter(|kind| **kind == LayerKind::LinearAttention)
        .count();
    let full_layers = config.num_hidden_layers - linear_layers;
    let expected_count =
        6usize
            .checked_add(config.num_hidden_layers.checked_mul(16).ok_or_else(|| {
                ModelError::InvalidConfig("Qwen3.6 tensor count overflowed".into())
            })?)
            .and_then(|count| {
                count.checked_add(
                    config
                        .num_hidden_layers
                        .checked_mul(config.num_experts)?
                        .checked_mul(12)?,
                )
            })
            .and_then(|count| count.checked_add(fp8_targets * 3))
            .and_then(|count| count.checked_add(linear_layers * 6))
            .and_then(|count| count.checked_add(full_layers * 2))
            .ok_or_else(|| ModelError::InvalidConfig("Qwen3.6 tensor count overflowed".into()))?;
    let mut tensors = Vec::with_capacity(expected_count);
    tensors.push(TensorContract::new(
        "model.language_model.embed_tokens.weight",
        WeightDtype::Bf16,
        [config.vocab_size, config.hidden_size],
    ));
    tensors.push(TensorContract::new(
        "model.language_model.norm.weight",
        WeightDtype::Bf16,
        [config.hidden_size],
    ));
    push_nvfp4(
        &mut tensors,
        "lm_head",
        config.vocab_size,
        config.hidden_size,
    )?;

    for (layer, kind) in config.layer_types.iter().enumerate() {
        let prefix = format!("model.language_model.layers.{layer}");
        for suffix in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            tensors.push(TensorContract::new(
                format!("{prefix}.{suffix}"),
                WeightDtype::Bf16,
                [config.hidden_size],
            ));
        }
        match kind {
            LayerKind::LinearAttention => push_linear_attention(&mut tensors, &prefix, config),
            LayerKind::FullAttention => push_full_attention(&mut tensors, &prefix, config),
        }
        push_moe(&mut tensors, &prefix, config)?;
    }
    debug_assert_eq!(tensors.len(), expected_count);
    Ok(tensors)
}

fn push_fp8(tensors: &mut Vec<TensorContract>, name: &str, shape: [usize; 2]) {
    tensors.push(TensorContract::new(
        format!("{name}.weight"),
        WeightDtype::F8E4M3,
        shape,
    ));
    for suffix in ["input_scale", "weight_scale"] {
        tensors.push(TensorContract::new(
            format!("{name}.{suffix}"),
            WeightDtype::F32,
            [],
        ));
    }
}

fn push_nvfp4(
    tensors: &mut Vec<TensorContract>,
    name: &str,
    output: usize,
    input: usize,
) -> Result<(), ModelError> {
    if !input.is_multiple_of(16) {
        return Err(ModelError::InvalidConfig(format!(
            "NVFP4 input width {input} for `{name}` is not divisible by 16"
        )));
    }
    tensors.push(TensorContract::new(
        format!("{name}.weight"),
        WeightDtype::U8,
        [output, input / 2],
    ));
    tensors.push(TensorContract::new(
        format!("{name}.weight_scale"),
        WeightDtype::F8E4M3,
        [output, input / 16],
    ));
    for suffix in ["input_scale", "weight_scale_2"] {
        tensors.push(TensorContract::new(
            format!("{name}.{suffix}"),
            WeightDtype::F32,
            [],
        ));
    }
    Ok(())
}

fn push_linear_attention(tensors: &mut Vec<TensorContract>, prefix: &str, config: &TextConfig) {
    let name = format!("{prefix}.linear_attn");
    let key_width = config.linear_num_key_heads * config.linear_key_head_dim;
    let value_width = config.linear_num_value_heads * config.linear_value_head_dim;
    let qkv_width = key_width * 2 + value_width;
    for (suffix, shape) in [
        ("A_log", vec![config.linear_num_value_heads]),
        ("dt_bias", vec![config.linear_num_value_heads]),
        (
            "in_proj_a.weight",
            vec![config.linear_num_value_heads, config.hidden_size],
        ),
        (
            "in_proj_b.weight",
            vec![config.linear_num_value_heads, config.hidden_size],
        ),
        ("norm.weight", vec![config.linear_value_head_dim]),
        (
            "conv1d.weight",
            vec![qkv_width, 1, config.linear_conv_kernel_dim],
        ),
    ] {
        tensors.push(TensorContract::new(
            format!("{name}.{suffix}"),
            WeightDtype::Bf16,
            shape,
        ));
    }
    push_fp8(
        tensors,
        &format!("{name}.in_proj_qkv"),
        [qkv_width, config.hidden_size],
    );
    push_fp8(
        tensors,
        &format!("{name}.in_proj_z"),
        [value_width, config.hidden_size],
    );
    push_fp8(
        tensors,
        &format!("{name}.out_proj"),
        [config.hidden_size, value_width],
    );
}

fn push_full_attention(tensors: &mut Vec<TensorContract>, prefix: &str, config: &TextConfig) {
    let name = format!("{prefix}.self_attn");
    for suffix in ["q_norm.weight", "k_norm.weight"] {
        tensors.push(TensorContract::new(
            format!("{name}.{suffix}"),
            WeightDtype::Bf16,
            [config.head_dim],
        ));
    }
    let q_width = config.num_attention_heads * config.head_dim;
    let kv_width = config.num_key_value_heads * config.head_dim;
    push_fp8(
        tensors,
        &format!("{name}.q_proj"),
        [q_width * 2, config.hidden_size],
    );
    push_fp8(
        tensors,
        &format!("{name}.k_proj"),
        [kv_width, config.hidden_size],
    );
    push_fp8(
        tensors,
        &format!("{name}.v_proj"),
        [kv_width, config.hidden_size],
    );
    push_fp8(
        tensors,
        &format!("{name}.o_proj"),
        [config.hidden_size, q_width],
    );
}

fn push_moe(
    tensors: &mut Vec<TensorContract>,
    prefix: &str,
    config: &TextConfig,
) -> Result<(), ModelError> {
    let name = format!("{prefix}.mlp");
    tensors.push(TensorContract::new(
        format!("{name}.gate.weight"),
        WeightDtype::Bf16,
        [config.num_experts, config.hidden_size],
    ));
    tensors.push(TensorContract::new(
        format!("{name}.shared_expert_gate.weight"),
        WeightDtype::Bf16,
        [1, config.hidden_size],
    ));
    for expert in 0..config.num_experts {
        let expert = format!("{name}.experts.{expert}");
        push_swiglu(
            tensors,
            &expert,
            config.moe_intermediate_size,
            config.hidden_size,
        )?;
    }
    push_swiglu(
        tensors,
        &format!("{name}.shared_expert"),
        config.shared_expert_intermediate_size,
        config.hidden_size,
    )
}

fn push_swiglu(
    tensors: &mut Vec<TensorContract>,
    prefix: &str,
    intermediate: usize,
    hidden: usize,
) -> Result<(), ModelError> {
    push_nvfp4(
        tensors,
        &format!("{prefix}.gate_proj"),
        intermediate,
        hidden,
    )?;
    push_nvfp4(tensors, &format!("{prefix}.up_proj"), intermediate, hidden)?;
    push_nvfp4(
        tensors,
        &format!("{prefix}.down_proj"),
        hidden,
        intermediate,
    )
}

pub(super) fn validate_text_weights(
    weights: &dyn WeightSource,
    config: &TextConfig,
) -> Result<(), ModelError> {
    let contracts = text_tensor_contracts(config)?;
    for contract in &contracts {
        weights.validate_tensor(&contract.name, &contract.dtype, &contract.shape)?;
    }
    let actual: HashSet<_> = weights
        .names()
        .into_iter()
        .filter(|name| {
            name == "lm_head"
                || name.starts_with("lm_head.")
                || name.starts_with("model.language_model.")
        })
        .collect();
    let expected: HashSet<_> = contracts
        .into_iter()
        .map(|contract| contract.name)
        .collect();
    if actual != expected {
        return Err(ModelError::InvalidConfig(format!(
            "Qwen3.6 text tensor manifest differs from the expected contract (actual {}, expected {})",
            actual.len(),
            expected.len()
        )));
    }
    Ok(())
}
