use std::sync::Arc;

use cuda_async::device_operation::DeviceOp;
use cuda_core::Stream;
use cutile::{
    api,
    core::bf16,
    tensor::{IntoPartition, PartitionMut, Tensor},
    tile_kernel::TileKernel,
};

use crate::{
    cuda::{
        attention::{AttentionBackend, DecodeGraphAttention, EagerAttention},
        kernel_plan::AttentionKernelPlan,
        kernels,
    },
    model::ModelError,
};

use super::{Bf16Tensor, DenseDecoderConfig, cuda_error, enqueue, output_buffer_async};

pub(super) struct FlatKvLayerState {
    key_cache: Bf16Tensor,
    value_cache: Bf16Tensor,
}

pub(super) struct DirectFlatKvAttention {
    layers: Vec<FlatKvLayerState>,
    cosine: Arc<Tensor<f32>>,
    sine: Arc<Tensor<f32>>,
    num_attention_heads: usize,
    plan: AttentionKernelPlan,
}

impl DirectFlatKvAttention {
    pub(super) fn load(
        config: &DenseDecoderConfig,
        stream: &Arc<Stream>,
        capacity: usize,
        scratch_slots: usize,
        cosine: Arc<Tensor<f32>>,
        sine: Arc<Tensor<f32>>,
        plan: AttentionKernelPlan,
    ) -> Result<Self, ModelError> {
        let cache_shape = [
            capacity
                .checked_add(1)
                .and_then(|slots| slots.checked_add(scratch_slots))
                .ok_or_else(|| {
                    ModelError::Cuda("KV cache sentinel allocation overflowed".into())
                })?,
            config.num_key_value_heads,
            config.head_dim,
        ];
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let key_cache = api::zeros::<bf16>(&cache_shape)
                .sync_on(stream)
                .map_err(|error| {
                    ModelError::Cuda(format!("allocate layer {layer} key cache: {error:?}"))
                })?;
            let value_cache =
                api::zeros::<bf16>(&cache_shape)
                    .sync_on(stream)
                    .map_err(|error| {
                        ModelError::Cuda(format!("allocate layer {layer} value cache: {error:?}"))
                    })?;
            layers.push(FlatKvLayerState {
                key_cache: Arc::new(key_cache),
                value_cache: Arc::new(value_cache),
            });
        }
        Ok(Self {
            layers,
            cosine,
            sine,
            num_attention_heads: config.num_attention_heads,
            plan,
        })
    }
}

impl AttentionBackend for DirectFlatKvAttention {
    type LayerState = FlatKvLayerState;
    type Error = ModelError;

    fn layer_state(&self, layer: usize) -> Result<&Self::LayerState, Self::Error> {
        self.layers
            .get(layer)
            .ok_or_else(|| ModelError::Cuda(format!("attention layer {layer} is out of range")))
    }

    fn enqueue_eager(&self, input: EagerAttention<'_>) -> Result<Tensor<bf16>, Self::Error> {
        let state = self.layer_state(input.layer)?;
        let head_dim = self.plan.head_dim.get();
        let rotated_query = output_buffer_async::<bf16>(
            &[input.rows, self.num_attention_heads, head_dim],
            input.stream,
            "rotated query",
        )?
        .partition([1, 1, head_dim]);
        let (_, _, _, _, rotated_query) = enqueue(
            kernels::rope_q_bf16(
                input.query,
                input.positions,
                &self.cosine,
                &self.sine,
                rotated_query,
            )
            .generics(vec![
                head_dim.to_string(),
                self.plan.rope.rotary_dim.get().to_string(),
            ]),
            input.stream,
            "query RoPE",
        )?;
        let rotated_query = Arc::new(rotated_query.unpartition());

        let rotated_key = output_buffer_async::<bf16>(
            &[input.rows, self.plan.kv_heads.get(), head_dim],
            input.stream,
            "rotated key",
        )?
        .partition([1, 1, head_dim]);
        let (_, _, _, _, _, _, _, _, _rotated_key) = enqueue(
            unsafe {
                kernels::rope_kv_write_bf16(
                    input.key,
                    input.value,
                    input.positions,
                    input.current_slots,
                    &self.cosine,
                    &self.sine,
                    state.key_cache.device_pointer(),
                    state.value_cache.device_pointer(),
                    rotated_key,
                )
            }
            .generics(vec![
                head_dim.to_string(),
                self.plan.rope.rotary_dim.get().to_string(),
                self.plan.kv_heads.get().to_string(),
            ]),
            input.stream,
            "key RoPE and flat KV write",
        )?;

        let attention = output_buffer_async::<bf16>(
            &[input.rows, self.num_attention_heads, head_dim],
            input.stream,
            "ragged attention output",
        )?
        .partition([1, 1, head_dim]);
        let (_, _, _, _, _, _, attention, _, _) = enqueue(
            unsafe {
                kernels::ragged_attention_bf16(
                    &rotated_query,
                    input.request_indices,
                    input.context_slots,
                    input.context_lengths,
                    state.key_cache.device_pointer(),
                    state.value_cache.device_pointer(),
                    attention,
                    1.0 / (head_dim as f32).sqrt(),
                    self.plan.query_heads_per_kv.get() as i32,
                )
            }
            .generics(vec![
                self.plan.prefill_mixed.key_block.get().to_string(),
                head_dim.to_string(),
                self.plan.kv_heads.get().to_string(),
            ]),
            input.stream,
            "ragged flat-KV attention",
        )?;
        Ok(attention.unpartition())
    }

    fn record_decode(&self, input: DecodeGraphAttention<'_>) -> Result<(), Self::Error> {
        let state = self.layer_state(input.layer)?;
        let head_dim = self.plan.head_dim.get();
        input
            .scope
            .record(
                kernels::rope_q_bf16(
                    input.query,
                    input.positions,
                    &self.cosine,
                    &self.sine,
                    input.rotated_query.partition([1, 1, head_dim]),
                )
                .generics(vec![
                    head_dim.to_string(),
                    self.plan.rope.rotary_dim.get().to_string(),
                ]),
            )
            .map_err(|error| cuda_error("record query RoPE", error))?;
        input
            .scope
            .record(
                unsafe {
                    kernels::rope_kv_write_bf16(
                        input.key,
                        input.value,
                        input.positions,
                        input.current_slots,
                        &self.cosine,
                        &self.sine,
                        state.key_cache.device_pointer(),
                        state.value_cache.device_pointer(),
                        input.rotated_key.partition([1, 1, head_dim]),
                    )
                }
                .generics(vec![
                    head_dim.to_string(),
                    self.plan.rope.rotary_dim.get().to_string(),
                    self.plan.kv_heads.get().to_string(),
                ]),
            )
            .map_err(|error| cuda_error("record key RoPE and flat KV write", error))?;
        input
            .scope
            .record(
                unsafe {
                    kernels::ragged_attention_bf16(
                        &*input.rotated_query,
                        input.request_indices,
                        input.context_slots,
                        input.context_lengths,
                        state.key_cache.device_pointer(),
                        state.value_cache.device_pointer(),
                        input.attention.partition([1, 1, head_dim]),
                        1.0 / (head_dim as f32).sqrt(),
                        self.plan.query_heads_per_kv.get() as i32,
                    )
                }
                .generics(vec![
                    self.plan.decode.key_block.get().to_string(),
                    head_dim.to_string(),
                    self.plan.kv_heads.get().to_string(),
                ]),
            )
            .map_err(|error| cuda_error("record ragged flat-KV attention", error))?;
        Ok(())
    }
}
