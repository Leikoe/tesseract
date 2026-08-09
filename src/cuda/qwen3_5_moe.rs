use std::sync::Arc;

use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream};
use cutile::{
    api,
    core::bf16,
    tensor::{PartitionMut, Reshape, Tensor, TensorView},
    tile_kernel::TileKernel,
};

use crate::model::{
    CudaModelReport, ModelError,
    weights::{WeightDtype, WeightSource},
};

use super::{
    cublas,
    gdn::{self as gdn_backend, GdnPrefillPlan, GdnState},
    kernels,
    linear::{ExpertProjection, Fp8W8A16Linear, GroupedNvfp4W4A16, Nvfp4W4A16Linear},
    moe::{self as moe_backend, RoutingPlan},
    qwen_attention::{AttentionInput, QwenFlatKvAttention},
};

type Bf16Tensor = Arc<Tensor<bf16>>;
type F32Tensor = Arc<Tensor<f32>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerKind {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone)]
pub(crate) struct Config {
    pub(crate) attn_output_gate: bool,
    pub(crate) head_dim: usize,
    pub(crate) hidden_act: String,
    pub(crate) hidden_size: usize,
    pub(crate) layers: Vec<LayerKind>,
    pub(crate) linear_conv_kernel_dim: usize,
    pub(crate) linear_key_head_dim: usize,
    pub(crate) linear_num_key_heads: usize,
    pub(crate) linear_num_value_heads: usize,
    pub(crate) linear_value_head_dim: usize,
    pub(crate) mamba_ssm_dtype: String,
    pub(crate) max_position_embeddings: usize,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) num_experts: usize,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) partial_rotary_factor: f32,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_interleaved: bool,
    pub(crate) rope_section: Vec<usize>,
    pub(crate) rope_theta: f32,
    pub(crate) rope_type: String,
    pub(crate) shared_expert_intermediate_size: usize,
    pub(crate) vocab_size: usize,
}

pub(crate) struct Artifact {
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
                        conv1d: load_bf16_as(
                            source,
                            &format!("{prefix}.conv1d.weight"),
                            &[8192, 4],
                            stream,
                        )?,
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

    fn forward_linear(
        &self,
        residual: Bf16Tensor,
        update: Option<Bf16Tensor>,
        state: &mut GdnState,
        state_slots: Arc<Tensor<i32>>,
        prefill: Option<&GdnPrefillPlan>,
        rows: usize,
        epsilon: f32,
        stream: &Arc<Stream>,
    ) -> Result<(Bf16Tensor, Bf16Tensor), ModelError> {
        let (attention_input, residual) = match update {
            Some(update) => gemma_add_rms_norm(
                residual,
                update,
                self.input_norm.clone(),
                rows,
                epsilon,
                stream,
            )?,
            None => (
                gemma_rms_norm(
                    residual.clone(),
                    self.input_norm.clone(),
                    rows,
                    epsilon,
                    stream,
                )?,
                residual,
            ),
        };
        let Attention::Linear(attention) = &self.attention else {
            return Err(ModelError::Cuda(
                "linear decode called for a full-attention layer".into(),
            ));
        };
        let attention_output = attention.forward(
            attention_input,
            state,
            state_slots,
            prefill,
            rows,
            epsilon,
            stream,
        )?;
        let (moe_input, residual) = gemma_add_rms_norm(
            residual,
            Arc::new(attention_output),
            self.post_attention_norm.clone(),
            rows,
            epsilon,
            stream,
        )?;
        let moe_output = self.moe.forward(moe_input, rows, stream)?;
        Ok((residual, Arc::new(moe_output)))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_full(
        &self,
        residual: Bf16Tensor,
        update: Option<Bf16Tensor>,
        backend: &QwenFlatKvAttention,
        full_layer: usize,
        positions: Arc<Tensor<u32>>,
        current_slots: Arc<Tensor<u32>>,
        request_indices: Arc<Tensor<u32>>,
        context_slots: &TensorView<'_, u32>,
        context_lengths: Arc<Tensor<i32>>,
        rows: usize,
        epsilon: f32,
        stream: &Arc<Stream>,
    ) -> Result<(Bf16Tensor, Bf16Tensor), ModelError> {
        let (attention_input, residual) = match update {
            Some(update) => gemma_add_rms_norm(
                residual,
                update,
                self.input_norm.clone(),
                rows,
                epsilon,
                stream,
            )?,
            None => (
                gemma_rms_norm(
                    residual.clone(),
                    self.input_norm.clone(),
                    rows,
                    epsilon,
                    stream,
                )?,
                residual,
            ),
        };
        let Attention::Full(attention) = &self.attention else {
            return Err(ModelError::Cuda(
                "full-attention forward called for a linear-attention layer".into(),
            ));
        };
        let attention_output = attention.forward(
            attention_input,
            backend,
            full_layer,
            positions,
            current_slots,
            request_indices,
            context_slots,
            context_lengths,
            rows,
            epsilon,
            stream,
        )?;
        let (moe_input, residual) = gemma_add_rms_norm(
            residual,
            Arc::new(attention_output),
            self.post_attention_norm.clone(),
            rows,
            epsilon,
            stream,
        )?;
        let moe_output = self.moe.forward(moe_input, rows, stream)?;
        Ok((residual, Arc::new(moe_output)))
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

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: Bf16Tensor,
        state: &mut GdnState,
        state_slots: Arc<Tensor<i32>>,
        prefill: Option<&GdnPrefillPlan>,
        rows: usize,
        epsilon: f32,
        stream: &Arc<Stream>,
    ) -> Result<Tensor<bf16>, ModelError> {
        const HIDDEN_SIZE: usize = 2048;
        const VALUE_SIZE: usize = 4096;
        if !rows.is_multiple_of(16) || hidden.shape() != [rows as i32, HIDDEN_SIZE as i32] {
            return Err(ModelError::Cuda("invalid Qwen GDN input geometry".into()));
        }
        let mixed_qkv = Arc::new(self.input_qkv.enqueue(hidden.clone(), rows, stream)?);
        let mixed_qkv = Arc::new(match prefill {
            Some(plan) => state.prefill_conv(
                mixed_qkv,
                self.conv1d.clone(),
                plan.query_start_offsets(),
                state_slots.clone(),
                rows,
                plan.requests(),
                stream,
            )?,
            None => state.decode_conv(
                mixed_qkv,
                self.conv1d.clone(),
                state_slots.clone(),
                rows,
                stream,
            )?,
        });
        let a = bf16_gemm(
            self.input_a.clone(),
            hidden.clone(),
            32,
            rows,
            HIDDEN_SIZE,
            "Qwen GDN a projection",
            stream,
        )?;
        let b = bf16_gemm(
            self.input_b.clone(),
            hidden.clone(),
            32,
            rows,
            HIDDEN_SIZE,
            "Qwen GDN b projection",
            stream,
        )?;
        let recurrent = Arc::new(match prefill {
            Some(plan) => state.prefill(
                mixed_qkv,
                a,
                b,
                self.a_log.clone(),
                self.dt_bias.clone(),
                state_slots,
                plan,
                stream,
            )?,
            None => state.decode(
                mixed_qkv,
                a,
                b,
                self.a_log.clone(),
                self.dt_bias.clone(),
                state_slots,
                rows,
                stream,
            )?,
        });
        let gate = self
            .input_z
            .enqueue(hidden, rows, stream)?
            .reshape(&[rows, 32, 128])
            .map_err(|error| ModelError::Cuda(format!("reshape Qwen GDN z gate: {error:?}")))?;
        let gated = gdn_backend::output_gate(
            recurrent,
            Arc::new(gate),
            self.norm.clone(),
            epsilon,
            rows,
            stream,
        )?
        .reshape(&[rows, VALUE_SIZE])
        .map_err(|error| ModelError::Cuda(format!("reshape Qwen GDN output: {error:?}")))?;
        self.output.enqueue(Arc::new(gated), rows, stream)
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

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: Bf16Tensor,
        backend: &QwenFlatKvAttention,
        layer: usize,
        positions: Arc<Tensor<u32>>,
        current_slots: Arc<Tensor<u32>>,
        request_indices: Arc<Tensor<u32>>,
        context_slots: &TensorView<'_, u32>,
        context_lengths: Arc<Tensor<i32>>,
        rows: usize,
        epsilon: f32,
        stream: &Arc<Stream>,
    ) -> Result<Tensor<bf16>, ModelError> {
        let q_gate = Arc::new(self.query.enqueue(hidden.clone(), rows, stream)?);
        let key = Arc::new(self.key.enqueue(hidden.clone(), rows, stream)?);
        let value = Arc::new(self.value.enqueue(hidden, rows, stream)?);
        let attention = backend.enqueue(AttentionInput {
            layer,
            q_gate,
            key,
            value,
            q_weight_delta: self.query_norm.clone(),
            k_weight_delta: self.key_norm.clone(),
            positions,
            current_slots,
            request_indices,
            context_slots,
            context_lengths,
            rows,
            epsilon,
            stream,
        })?;
        self.output.enqueue(Arc::new(attention), rows, stream)
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

    fn forward(
        &self,
        hidden: Bf16Tensor,
        rows: usize,
        stream: &Arc<Stream>,
    ) -> Result<Tensor<bf16>, ModelError> {
        const HIDDEN_SIZE: usize = 2048;
        if !rows.is_multiple_of(16) || hidden.shape() != [rows as i32, HIDDEN_SIZE as i32] {
            return Err(ModelError::Cuda("invalid Qwen MoE input geometry".into()));
        }

        let router_logits = bf16_gemm(
            self.router.clone(),
            hidden.clone(),
            256,
            rows,
            HIDDEN_SIZE,
            "Qwen routed-expert logits",
            stream,
        )?;
        let routing = RoutingPlan::build(router_logits, rows, stream)?;
        let dispatched = routing.dispatch(hidden.clone(), rows, HIDDEN_SIZE, stream)?;
        let dispatched_rows = routing.max_dispatched_rows;
        let routed_gate = self.routed_gate.enqueue_device_plan(
            dispatched.hidden.clone(),
            dispatched_rows,
            dispatched.expert_by_row_tile.clone(),
            stream,
        )?;
        let routed_up = self.routed_up.enqueue_device_plan(
            dispatched.hidden,
            dispatched_rows,
            dispatched.expert_by_row_tile.clone(),
            stream,
        )?;
        let routed_activated = silu_mul(
            Arc::new(routed_gate),
            Arc::new(routed_up),
            dispatched_rows,
            self.routed_gate.output_size(),
            stream,
        )?;
        let routed_down = self.routed_down.enqueue_device_plan(
            Arc::new(routed_activated),
            dispatched_rows,
            dispatched.expert_by_row_tile,
            stream,
        )?;
        let routed = routing.combine(Arc::new(routed_down), rows, HIDDEN_SIZE, stream)?;

        let shared_gate = self.shared_gate.enqueue(hidden.clone(), rows, stream)?;
        let shared_up = self.shared_up.enqueue(hidden.clone(), rows, stream)?;
        let shared_activated = silu_mul(
            Arc::new(shared_gate),
            Arc::new(shared_up),
            rows,
            self.shared_gate.output_size(),
            stream,
        )?;
        let shared = self
            .shared_down
            .enqueue(Arc::new(shared_activated), rows, stream)?;
        let shared_logits = bf16_gemm(
            self.shared_router.clone(),
            hidden,
            1,
            rows,
            HIDDEN_SIZE,
            "Qwen shared-expert gate",
            stream,
        )?;
        moe_backend::combine_shared(
            Arc::new(routed),
            Arc::new(shared),
            shared_logits,
            rows,
            HIDDEN_SIZE,
            stream,
        )
    }
}

fn bf16_gemm(
    weight: Bf16Tensor,
    input: Bf16Tensor,
    output_size: usize,
    rows: usize,
    input_size: usize,
    operation: &'static str,
    stream: &Arc<Stream>,
) -> Result<Bf16Tensor, ModelError> {
    let output = api::zeros::<bf16>(&[rows, output_size])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate {operation} output: {error:?}")))?;
    let output = cublas::gemm_bf16(weight, input, output, output_size, rows, input_size)
        .map_err(|error| ModelError::Cuda(format!("prepare {operation}: {error}")))?
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("schedule {operation}: {error:?}")))?
        .map_err(|error| ModelError::Cuda(format!("execute {operation}: {error}")))?;
    Ok(Arc::new(output))
}

fn silu_mul(
    gate: Bf16Tensor,
    up: Bf16Tensor,
    rows: usize,
    width: usize,
    stream: &Arc<Stream>,
) -> Result<Tensor<bf16>, ModelError> {
    const BLOCK: usize = 256;
    if width == 0
        || !width.is_multiple_of(BLOCK)
        || gate.shape() != [rows as i32, width as i32]
        || up.shape() != gate.shape()
    {
        return Err(ModelError::Cuda("invalid Qwen SiLU gate geometry".into()));
    }
    let mut output = api::zeros::<bf16>(&[rows, width])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate Qwen SiLU output: {error:?}")))?;
    let (_, _, output_partition) =
        kernels::silu_mul_bf16(gate, up, (&mut output).partition([1, BLOCK]))
            .generics(vec![BLOCK.to_string()])
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("execute Qwen SiLU gate: {error:?}")))?;
    drop(output_partition);
    Ok(output)
}

fn gemma_rms_norm(
    input: Bf16Tensor,
    weight_delta: Bf16Tensor,
    rows: usize,
    epsilon: f32,
    stream: &Arc<Stream>,
) -> Result<Bf16Tensor, ModelError> {
    const HIDDEN_SIZE: usize = 2048;
    const BLOCK: usize = 256;
    if input.shape() != [rows as i32, HIDDEN_SIZE as i32]
        || weight_delta.shape() != [HIDDEN_SIZE as i32]
    {
        return Err(ModelError::Cuda(
            "invalid Qwen Gemma RMSNorm geometry".into(),
        ));
    }
    let mut output = api::zeros::<bf16>(&[rows, HIDDEN_SIZE])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate Qwen normalized output: {error:?}")))?;
    let (_, _, output_partition, _) = unsafe {
        kernels::gemma_rms_norm_bf16(
            input,
            weight_delta,
            (&mut output).partition([1, HIDDEN_SIZE]),
            epsilon,
        )
    }
    .generics(vec![HIDDEN_SIZE.to_string(), BLOCK.to_string()])
    .sync_on(stream)
    .map_err(|error| ModelError::Cuda(format!("execute Qwen Gemma RMSNorm: {error:?}")))?;
    drop(output_partition);
    Ok(Arc::new(output))
}

fn gemma_add_rms_norm(
    residual: Bf16Tensor,
    update: Bf16Tensor,
    weight_delta: Bf16Tensor,
    rows: usize,
    epsilon: f32,
    stream: &Arc<Stream>,
) -> Result<(Bf16Tensor, Bf16Tensor), ModelError> {
    const HIDDEN_SIZE: usize = 2048;
    const BLOCK: usize = 256;
    if residual.shape() != [rows as i32, HIDDEN_SIZE as i32]
        || update.shape() != residual.shape()
        || weight_delta.shape() != [HIDDEN_SIZE as i32]
    {
        return Err(ModelError::Cuda(
            "invalid Qwen fused add Gemma RMSNorm geometry".into(),
        ));
    }
    let mut normalized = api::zeros::<bf16>(&[rows, HIDDEN_SIZE])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate Qwen normalized output: {error:?}")))?;
    let mut combined = api::zeros::<bf16>(&[rows, HIDDEN_SIZE])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate Qwen residual output: {error:?}")))?;
    let (_, _, _, normalized_partition, combined_partition, _) = unsafe {
        kernels::gemma_add_rms_norm_bf16(
            residual,
            update,
            weight_delta,
            (&mut normalized).partition([1, HIDDEN_SIZE]),
            (&mut combined).partition([1, HIDDEN_SIZE]),
            epsilon,
        )
    }
    .generics(vec![HIDDEN_SIZE.to_string(), BLOCK.to_string()])
    .sync_on(stream)
    .map_err(|error| {
        ModelError::Cuda(format!("execute Qwen fused add Gemma RMSNorm: {error:?}"))
    })?;
    drop(normalized_partition);
    drop(combined_partition);
    Ok((Arc::new(normalized), Arc::new(combined)))
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
    let shape = source.tensor(name)?.shape().to_vec();
    load_bf16_as(source, name, &shape, stream)
}

fn load_bf16_as(
    source: &dyn WeightSource,
    name: &str,
    shape: &[usize],
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
        .reshape(shape)
        .map_err(|error| ModelError::Cuda(format!("reshape `{name}`: {error:?}")))?;
    Ok(Arc::new(device))
}

fn load_f32(
    source: &dyn WeightSource,
    name: &str,
    stream: &Arc<Stream>,
) -> Result<F32Tensor, ModelError> {
    let tensor = source.tensor(name)?;
    let host = Arc::new(match tensor.dtype() {
        WeightDtype::F32 => tensor
            .bytes()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect::<Vec<_>>(),
        WeightDtype::Bf16 => tensor
            .bytes()
            .chunks_exact(2)
            .map(|bytes| bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
            .collect::<Vec<_>>(),
        actual => {
            return Err(ModelError::WrongDtype {
                name: name.into(),
                expected: format!("{} or {}", WeightDtype::F32, WeightDtype::Bf16),
                actual: actual.to_string(),
            });
        }
    });
    let device = api::copy_host_vec_to_device(&host)
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("upload `{name}`: {error:?}")))?
        .reshape(tensor.shape())
        .map_err(|error| ModelError::Cuda(format!("reshape `{name}`: {error:?}")))?;
    Ok(Arc::new(device))
}
