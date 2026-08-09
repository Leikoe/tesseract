use std::sync::Arc;

use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream};
use cutile::{
    api,
    core::bf16,
    tensor::{Reshape, Tensor},
};

use crate::model::{
    CudaModelReport, Model, ModelError,
    weights::{WeightDtype, WeightSource},
};

use super::linear::{ExpertProjection, Fp8W8A16Linear, GroupedNvfp4W4A16, Nvfp4W4A16Linear};

type Bf16Tensor = Arc<Tensor<bf16>>;
type F32Tensor = Arc<Tensor<f32>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerKind {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone)]
pub(crate) struct Config {
    pub(crate) layers: Vec<LayerKind>,
    pub(crate) num_experts: usize,
}

pub(crate) struct Artifact {
    pub(crate) model: Arc<dyn Model>,
    pub(crate) config: Config,
    pub(crate) weights: Arc<dyn WeightSource>,
}

pub(crate) struct Checkpoint {
    embedding: Bf16Tensor,
    final_norm: Bf16Tensor,
    lm_head: Nvfp4W4A16Linear,
    layers: Vec<Layer>,
}

struct Layer {
    input_norm: Bf16Tensor,
    post_attention_norm: Bf16Tensor,
    attention: Attention,
    moe: Moe,
}

enum Attention {
    Linear(LinearAttention),
    Full(FullAttention),
}

struct LinearAttention {
    a_log: F32Tensor,
    conv1d: Bf16Tensor,
    dt_bias: F32Tensor,
    input_a: Bf16Tensor,
    input_b: Bf16Tensor,
    input_qkv: Fp8W8A16Linear,
    input_z: Fp8W8A16Linear,
    norm: Bf16Tensor,
    output: Fp8W8A16Linear,
}

struct FullAttention {
    key: Fp8W8A16Linear,
    key_norm: Bf16Tensor,
    output: Fp8W8A16Linear,
    query: Fp8W8A16Linear,
    query_norm: Bf16Tensor,
    value: Fp8W8A16Linear,
}

struct Moe {
    router: Bf16Tensor,
    routed_gate: GroupedNvfp4W4A16,
    routed_up: GroupedNvfp4W4A16,
    routed_down: GroupedNvfp4W4A16,
    shared_gate: Nvfp4W4A16Linear,
    shared_up: Nvfp4W4A16Linear,
    shared_down: Nvfp4W4A16Linear,
    shared_router: Bf16Tensor,
}

impl Checkpoint {
    fn load(artifact: &Artifact, stream: &Arc<Stream>) -> Result<Self, ModelError> {
        let source = artifact.weights.as_ref();
        let embedding = load_bf16(source, "model.language_model.embed_tokens.weight", stream)?;
        let final_norm = load_bf16(source, "model.language_model.norm.weight", stream)?;
        let lm_head = Nvfp4W4A16Linear::load(source, "lm_head", stream)?;
        let mut layers = Vec::with_capacity(artifact.config.layers.len());
        for (index, kind) in artifact.config.layers.iter().copied().enumerate() {
            let prefix = format!("model.language_model.layers.{index}");
            let input_norm =
                load_bf16(source, &format!("{prefix}.input_layernorm.weight"), stream)?;
            let post_attention_norm = load_bf16(
                source,
                &format!("{prefix}.post_attention_layernorm.weight"),
                stream,
            )?;
            let attention = match kind {
                LayerKind::LinearAttention => {
                    let prefix = format!("{prefix}.linear_attn");
                    Attention::Linear(LinearAttention {
                        a_log: load_f32(source, &format!("{prefix}.A_log"), stream)?,
                        conv1d: load_bf16(source, &format!("{prefix}.conv1d.weight"), stream)?,
                        dt_bias: load_f32(source, &format!("{prefix}.dt_bias"), stream)?,
                        input_a: load_bf16(source, &format!("{prefix}.in_proj_a.weight"), stream)?,
                        input_b: load_bf16(source, &format!("{prefix}.in_proj_b.weight"), stream)?,
                        input_qkv: Fp8W8A16Linear::load(
                            source,
                            &format!("{prefix}.in_proj_qkv"),
                            stream,
                        )?,
                        input_z: Fp8W8A16Linear::load(
                            source,
                            &format!("{prefix}.in_proj_z"),
                            stream,
                        )?,
                        norm: load_bf16(source, &format!("{prefix}.norm.weight"), stream)?,
                        output: Fp8W8A16Linear::load(
                            source,
                            &format!("{prefix}.out_proj"),
                            stream,
                        )?,
                    })
                }
                LayerKind::FullAttention => {
                    let prefix = format!("{prefix}.self_attn");
                    Attention::Full(FullAttention {
                        key: Fp8W8A16Linear::load(source, &format!("{prefix}.k_proj"), stream)?,
                        key_norm: load_bf16(source, &format!("{prefix}.k_norm.weight"), stream)?,
                        output: Fp8W8A16Linear::load(source, &format!("{prefix}.o_proj"), stream)?,
                        query: Fp8W8A16Linear::load(source, &format!("{prefix}.q_proj"), stream)?,
                        query_norm: load_bf16(source, &format!("{prefix}.q_norm.weight"), stream)?,
                        value: Fp8W8A16Linear::load(source, &format!("{prefix}.v_proj"), stream)?,
                    })
                }
            };
            let moe_prefix = format!("{prefix}.mlp");
            let experts_prefix = format!("{moe_prefix}.experts");
            let shared_prefix = format!("{moe_prefix}.shared_expert");
            let moe = Moe {
                router: load_bf16(source, &format!("{moe_prefix}.gate.weight"), stream)?,
                routed_gate: GroupedNvfp4W4A16::load(
                    source,
                    &experts_prefix,
                    ExpertProjection::Gate,
                    artifact.config.num_experts,
                    stream,
                )?,
                routed_up: GroupedNvfp4W4A16::load(
                    source,
                    &experts_prefix,
                    ExpertProjection::Up,
                    artifact.config.num_experts,
                    stream,
                )?,
                routed_down: GroupedNvfp4W4A16::load(
                    source,
                    &experts_prefix,
                    ExpertProjection::Down,
                    artifact.config.num_experts,
                    stream,
                )?,
                shared_gate: Nvfp4W4A16Linear::load(
                    source,
                    &format!("{shared_prefix}.gate_proj"),
                    stream,
                )?,
                shared_up: Nvfp4W4A16Linear::load(
                    source,
                    &format!("{shared_prefix}.up_proj"),
                    stream,
                )?,
                shared_down: Nvfp4W4A16Linear::load(
                    source,
                    &format!("{shared_prefix}.down_proj"),
                    stream,
                )?,
                shared_router: load_bf16(
                    source,
                    &format!("{moe_prefix}.shared_expert_gate.weight"),
                    stream,
                )?,
            };
            layers.push(Layer {
                input_norm,
                post_attention_norm,
                attention,
                moe,
            });
        }
        Ok(Self {
            embedding,
            final_norm,
            lm_head,
            layers,
        })
    }

    fn device_bytes(&self) -> usize {
        self.embedding.num_bytes()
            + self.final_norm.num_bytes()
            + self.lm_head.device_bytes()
            + self.layers.iter().map(Layer::device_bytes).sum::<usize>()
    }
}

impl Layer {
    fn device_bytes(&self) -> usize {
        self.input_norm.num_bytes()
            + self.post_attention_norm.num_bytes()
            + self.attention.device_bytes()
            + self.moe.device_bytes()
    }
}

impl Attention {
    fn device_bytes(&self) -> usize {
        match self {
            Self::Linear(weights) => weights.device_bytes(),
            Self::Full(weights) => weights.device_bytes(),
        }
    }
}

impl LinearAttention {
    fn device_bytes(&self) -> usize {
        self.a_log.num_bytes()
            + self.conv1d.num_bytes()
            + self.dt_bias.num_bytes()
            + self.input_a.num_bytes()
            + self.input_b.num_bytes()
            + self.input_qkv.device_bytes()
            + self.input_z.device_bytes()
            + self.norm.num_bytes()
            + self.output.device_bytes()
    }
}

impl FullAttention {
    fn device_bytes(&self) -> usize {
        self.key.device_bytes()
            + self.key_norm.num_bytes()
            + self.output.device_bytes()
            + self.query.device_bytes()
            + self.query_norm.num_bytes()
            + self.value.device_bytes()
    }
}

impl Moe {
    fn device_bytes(&self) -> usize {
        self.router.num_bytes()
            + self.routed_gate.device_bytes()
            + self.routed_up.device_bytes()
            + self.routed_down.device_bytes()
            + self.shared_gate.device_bytes()
            + self.shared_up.device_bytes()
            + self.shared_down.device_bytes()
            + self.shared_router.num_bytes()
    }
}

pub(crate) fn checkpoint_report(
    model_id: &str,
    artifact: Artifact,
    device_id: usize,
) -> Result<CudaModelReport, ModelError> {
    let device = Device::new(device_id)
        .map_err(|error| ModelError::Cuda(format!("initialize device {device_id}: {error:?}")))?;
    let stream = device
        .new_stream()
        .map_err(|error| ModelError::Cuda(format!("create stream: {error:?}")))?;
    let checkpoint = Checkpoint::load(&artifact, &stream)?;
    let bytes = checkpoint.device_bytes();
    drop(checkpoint);
    Ok(CudaModelReport {
        model_id: model_id.into(),
        device_id,
        tensors: artifact.weights.tensor_count(),
        bytes,
    })
}

fn load_bf16(
    source: &dyn WeightSource,
    name: &str,
    stream: &Arc<Stream>,
) -> Result<Bf16Tensor, ModelError> {
    let tensor = source.tensor(name)?;
    if tensor.dtype() != &WeightDtype::Bf16 {
        return Err(ModelError::WrongDtype {
            name: name.into(),
            expected: WeightDtype::Bf16.to_string(),
            actual: tensor.dtype().to_string(),
        });
    }
    let host = Arc::new(
        tensor
            .bytes()
            .chunks_exact(2)
            .map(|bytes| bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect::<Vec<_>>(),
    );
    let device = api::copy_host_vec_to_device(&host)
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("upload `{name}`: {error:?}")))?
        .reshape(tensor.shape())
        .map_err(|error| ModelError::Cuda(format!("reshape `{name}`: {error:?}")))?;
    Ok(Arc::new(device))
}

fn load_f32(
    source: &dyn WeightSource,
    name: &str,
    stream: &Arc<Stream>,
) -> Result<F32Tensor, ModelError> {
    let tensor = source.tensor(name)?;
    if tensor.dtype() != &WeightDtype::F32 {
        return Err(ModelError::WrongDtype {
            name: name.into(),
            expected: WeightDtype::F32.to_string(),
            actual: tensor.dtype().to_string(),
        });
    }
    let host = Arc::new(
        tensor
            .bytes()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect::<Vec<_>>(),
    );
    let device = api::copy_host_vec_to_device(&host)
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("upload `{name}`: {error:?}")))?
        .reshape(tensor.shape())
        .map_err(|error| ModelError::Cuda(format!("reshape `{name}`: {error:?}")))?;
    Ok(Arc::new(device))
}
