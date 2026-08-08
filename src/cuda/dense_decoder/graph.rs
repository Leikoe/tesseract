use std::sync::Arc;

use cuda_async::{
    cuda_graph::{CudaGraph, Scope},
    device_operation::DeviceOp,
    error::DeviceError,
};
use cuda_core::Stream;
use cutile::{
    api,
    core::bf16,
    tensor::{PartitionMut, Tensor, ToHostVec},
    tile_kernel::TileKernel,
};

use crate::{
    cuda::{
        attention::{AttentionBackend, DecodeGraphAttention},
        cublas, kernels,
    },
    model::ModelError,
};

use super::{
    DenseDecoder, DenseDecoderConfig, ForwardOutput, copy_i32, copy_u32, cuda_error, device_error,
    output_bf16, output_buffer,
};

struct DecodeLayerBuffers {
    attention_input: Tensor<bf16>,
    residual: Tensor<bf16>,
    query: Tensor<bf16>,
    key: Tensor<bf16>,
    value: Tensor<bf16>,
    rotated_query: Tensor<bf16>,
    rotated_key: Tensor<bf16>,
    attention: Tensor<bf16>,
    attention_flat: Tensor<bf16>,
    attention_output: Tensor<bf16>,
    mlp_input: Tensor<bf16>,
    hidden_after_attention: Tensor<bf16>,
    gate: Tensor<bf16>,
    up: Tensor<bf16>,
    activated: Tensor<bf16>,
    down: Tensor<bf16>,
}

impl DecodeLayerBuffers {
    fn allocate(
        cfg: &DenseDecoderConfig,
        batch_size: usize,
        stream: &Arc<Stream>,
    ) -> Result<Self, ModelError> {
        Ok(Self {
            attention_input: output_bf16(&[batch_size, cfg.hidden_size], stream)?,
            // Layer zero does not consume this field, so unlike the other
            // graph buffers it is not guaranteed to be overwritten.
            residual: api::zeros::<bf16>(&[batch_size, cfg.hidden_size])
                .sync_on(stream)
                .map_err(|error| cuda_error("initialize graph residual", error))?,
            query: output_bf16(&[batch_size, cfg.q_width()], stream)?,
            key: output_bf16(&[batch_size, cfg.kv_width()], stream)?,
            value: output_bf16(&[batch_size, cfg.kv_width()], stream)?,
            rotated_query: output_bf16(
                &[batch_size, cfg.num_attention_heads, cfg.head_dim],
                stream,
            )?,
            rotated_key: output_bf16(&[batch_size, cfg.num_key_value_heads, cfg.head_dim], stream)?,
            attention: output_bf16(&[batch_size, cfg.num_attention_heads, cfg.head_dim], stream)?,
            attention_flat: output_bf16(&[batch_size, cfg.hidden_size], stream)?,
            attention_output: output_bf16(&[batch_size, cfg.hidden_size], stream)?,
            mlp_input: output_bf16(&[batch_size, cfg.hidden_size], stream)?,
            hidden_after_attention: output_bf16(&[batch_size, cfg.hidden_size], stream)?,
            gate: output_bf16(&[batch_size, cfg.intermediate_size], stream)?,
            up: output_bf16(&[batch_size, cfg.intermediate_size], stream)?,
            activated: output_bf16(&[batch_size, cfg.intermediate_size], stream)?,
            down: output_bf16(&[batch_size, cfg.hidden_size], stream)?,
        })
    }
}

struct DecodeGraphStorage {
    _embedding_hidden: Tensor<bf16>,
    _layers: Vec<DecodeLayerBuffers>,
    _final_hidden: Tensor<bf16>,
    _final_residual: Tensor<bf16>,
    _argmax_block_max: Tensor<f32>,
    _argmax_block_index: Tensor<u32>,
}

pub(super) struct DecodeGraph {
    graph: CudaGraph<()>,
    stream: Arc<Stream>,
    token_ids: Tensor<u32>,
    positions: Tensor<u32>,
    current_slots: Tensor<u32>,
    _request_indices: Tensor<u32>,
    context_slots: Tensor<u32>,
    context_lengths: Tensor<i32>,
    logits: Arc<Tensor<bf16>>,
    sampled_token: Arc<Tensor<u32>>,
    batch_size: usize,
    context_bucket: usize,
    _storage: DecodeGraphStorage,
}

impl DecodeGraph {
    pub(super) fn capture<A: AttentionBackend<Error = ModelError>>(
        runtime: &DenseDecoder<A>,
        batch_size: usize,
        context_bucket: usize,
    ) -> Result<Self, ModelError> {
        let cfg = &runtime.config;
        let stream = runtime.stream.clone();
        let token_ids = copy_u32(&vec![0; batch_size], &stream, "graph token IDs")?;
        let positions = copy_u32(&vec![0; batch_size], &stream, "graph positions")?;
        let current_slots = copy_u32(
            &vec![runtime.capacity as u32; batch_size],
            &stream,
            "graph current slots",
        )?;
        let request_indices = copy_u32(
            &(0..batch_size)
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| ModelError::Cuda("decode graph batch exceeds u32".into()))?,
            &stream,
            "graph request indices",
        )?;
        let context_slots = copy_u32(
            &vec![runtime.capacity as u32; batch_size * context_bucket],
            &stream,
            "graph context slots",
        )?;
        let context_lengths = copy_i32(&vec![1; batch_size], &stream, "graph context lengths")?;
        let mut embedding_hidden = output_bf16(&[batch_size, cfg.hidden_size], &stream)?;
        let mut layers: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|_| DecodeLayerBuffers::allocate(cfg, batch_size, &stream))
            .collect::<Result<_, _>>()?;
        let mut final_hidden = output_bf16(&[batch_size, cfg.hidden_size], &stream)?;
        let mut final_residual = output_bf16(&[batch_size, cfg.hidden_size], &stream)?;
        let logits = Arc::new(output_bf16(&[batch_size, cfg.vocab_size], &stream)?);
        let hidden_block = runtime.kernel_plan.dense.embedding.block.get();
        let rms_norm_block = runtime.kernel_plan.dense.rms_norm.block.get();
        let add_rms_norm_block = runtime.kernel_plan.dense.add_rms_norm.block.get();
        let mlp_block = runtime.kernel_plan.dense.silu_mul.block.get();
        let argmax_block = runtime.kernel_plan.sampling.argmax.block.get();
        let argmax_blocks = cfg.vocab_size.div_ceil(argmax_block);
        let argmax_reduce_block = runtime
            .kernel_plan
            .shapes
            .query_bucket(argmax_blocks)
            .ok_or_else(|| ModelError::Cuda("graph argmax bucket overflowed".into()))?;
        let mut argmax_block_max =
            output_buffer::<f32>(&[batch_size, argmax_blocks], &stream, "graph argmax maxima")?;
        let mut argmax_block_index = output_buffer::<u32>(
            &[batch_size, argmax_blocks],
            &stream,
            "graph argmax indices",
        )?;
        let mut sampled_token =
            output_buffer::<u32>(&[batch_size], &stream, "graph sampled token")?;

        let graph = CudaGraph::scope(&stream, |scope| {
            scope.record(
                kernels::embedding_bf16(
                    &token_ids,
                    &runtime.weights.embedding,
                    (&mut embedding_hidden).partition([1, hidden_block]),
                )
                .generics(vec![cfg.hidden_size.to_string(), hidden_block.to_string()]),
            )?;

            for layer_index in 0..cfg.num_hidden_layers {
                if layer_index == 0 {
                    let layer = &mut layers[0];
                    scope.record(
                        unsafe {
                            kernels::rms_norm_bf16(
                                &embedding_hidden,
                                &runtime.weights.layers[0].input_norm,
                                (&mut layer.attention_input).partition([1, cfg.hidden_size]),
                                cfg.rms_norm_eps,
                            )
                        }
                        .generics(vec![
                            cfg.hidden_size.to_string(),
                            rms_norm_block.to_string(),
                        ]),
                    )?;
                } else {
                    let (previous, current) = layers.split_at_mut(layer_index);
                    let previous = &previous[layer_index - 1];
                    let current = &mut current[0];
                    scope.record(
                        unsafe {
                            kernels::add_rms_norm_bf16(
                                &previous.hidden_after_attention,
                                &previous.down,
                                &runtime.weights.layers[layer_index].input_norm,
                                (&mut current.attention_input).partition([1, cfg.hidden_size]),
                                (&mut current.residual).partition([1, cfg.hidden_size]),
                                cfg.rms_norm_eps,
                            )
                        }
                        .generics(vec![
                            cfg.hidden_size.to_string(),
                            add_rms_norm_block.to_string(),
                        ]),
                    )?;
                }

                let layer_weights = &runtime.weights.layers[layer_index];
                let layer = &mut layers[layer_index];
                record_gemm(
                    scope,
                    &layer_weights.query,
                    &layer.attention_input,
                    &layer.query,
                    cfg.q_width(),
                    batch_size,
                    cfg.hidden_size,
                )?;
                record_gemm(
                    scope,
                    &layer_weights.key,
                    &layer.attention_input,
                    &layer.key,
                    cfg.kv_width(),
                    batch_size,
                    cfg.hidden_size,
                )?;
                record_gemm(
                    scope,
                    &layer_weights.value,
                    &layer.attention_input,
                    &layer.value,
                    cfg.kv_width(),
                    batch_size,
                    cfg.hidden_size,
                )?;

                let query = layer
                    .query
                    .view(&[batch_size, cfg.num_attention_heads, cfg.head_dim])
                    .map_err(device_error)?;
                let key = layer
                    .key
                    .view(&[batch_size, cfg.num_key_value_heads, cfg.head_dim])
                    .map_err(device_error)?;
                let value = layer
                    .value
                    .view(&[batch_size, cfg.num_key_value_heads, cfg.head_dim])
                    .map_err(device_error)?;
                let context_slots_view = context_slots
                    .view(&[batch_size, context_bucket])
                    .map_err(device_error)?;
                runtime
                    .attention
                    .record_decode(DecodeGraphAttention {
                        scope,
                        layer: layer_index,
                        query: &query,
                        key: &key,
                        value: &value,
                        positions: &positions,
                        current_slots: &current_slots,
                        request_indices: &request_indices,
                        context_slots: &context_slots_view,
                        context_lengths: &context_lengths,
                        rotated_query: &mut layer.rotated_query,
                        rotated_key: &mut layer.rotated_key,
                        attention: &mut layer.attention,
                    })
                    .map_err(|error| DeviceError::Internal(error.to_string()))?;
                scope.record(api::memcpy(&mut layer.attention_flat, &layer.attention))?;
                record_gemm(
                    scope,
                    &layer_weights.output,
                    &layer.attention_flat,
                    &layer.attention_output,
                    cfg.hidden_size,
                    batch_size,
                    cfg.hidden_size,
                )?;
                let residual = if layer_index == 0 {
                    &embedding_hidden
                } else {
                    &layer.residual
                };
                scope.record(
                    unsafe {
                        kernels::add_rms_norm_bf16(
                            residual,
                            &layer.attention_output,
                            &layer_weights.post_norm,
                            (&mut layer.mlp_input).partition([1, cfg.hidden_size]),
                            (&mut layer.hidden_after_attention).partition([1, cfg.hidden_size]),
                            cfg.rms_norm_eps,
                        )
                    }
                    .generics(vec![
                        cfg.hidden_size.to_string(),
                        add_rms_norm_block.to_string(),
                    ]),
                )?;
                record_gemm(
                    scope,
                    &layer_weights.gate,
                    &layer.mlp_input,
                    &layer.gate,
                    cfg.intermediate_size,
                    batch_size,
                    cfg.hidden_size,
                )?;
                record_gemm(
                    scope,
                    &layer_weights.up,
                    &layer.mlp_input,
                    &layer.up,
                    cfg.intermediate_size,
                    batch_size,
                    cfg.hidden_size,
                )?;
                scope.record(
                    kernels::silu_mul_bf16(
                        &layer.gate,
                        &layer.up,
                        (&mut layer.activated).partition([1, mlp_block]),
                    )
                    .generics(vec![mlp_block.to_string()]),
                )?;
                record_gemm(
                    scope,
                    &layer_weights.down,
                    &layer.activated,
                    &layer.down,
                    cfg.hidden_size,
                    batch_size,
                    cfg.intermediate_size,
                )?;
            }

            let last = layers
                .last()
                .ok_or_else(|| DeviceError::Internal("model has no layers".into()))?;
            scope.record(
                unsafe {
                    kernels::add_rms_norm_bf16(
                        &last.hidden_after_attention,
                        &last.down,
                        &runtime.weights.final_norm,
                        (&mut final_hidden).partition([1, cfg.hidden_size]),
                        (&mut final_residual).partition([1, cfg.hidden_size]),
                        cfg.rms_norm_eps,
                    )
                }
                .generics(vec![
                    cfg.hidden_size.to_string(),
                    add_rms_norm_block.to_string(),
                ]),
            )?;
            record_gemm(
                scope,
                &runtime.weights.lm_head,
                &final_hidden,
                &logits,
                cfg.vocab_size,
                batch_size,
                cfg.hidden_size,
            )?;
            scope.record(
                kernels::argmax_blocks_batch_bf16(
                    &logits,
                    (&mut argmax_block_max).partition([1, 1]),
                    (&mut argmax_block_index).partition([1, 1]),
                    cfg.vocab_size as i32,
                )
                .generics(vec![argmax_block.to_string()]),
            )?;
            scope.record(
                kernels::argmax_reduce_batch_bf16(
                    &argmax_block_max,
                    &argmax_block_index,
                    (&mut sampled_token).partition([1]),
                    argmax_blocks as i32,
                )
                .generics(vec![argmax_reduce_block.to_string()]),
            )?;
            Ok(())
        })
        .map_err(|error| cuda_error("capture decode CUDA graph", error))?;

        Ok(Self {
            graph,
            stream,
            token_ids,
            positions,
            current_slots,
            _request_indices: request_indices,
            context_slots,
            context_lengths,
            logits,
            sampled_token: Arc::new(sampled_token),
            batch_size,
            context_bucket,
            _storage: DecodeGraphStorage {
                _embedding_hidden: embedding_hidden,
                _layers: layers,
                _final_hidden: final_hidden,
                _final_residual: final_residual,
                _argmax_block_max: argmax_block_max,
                _argmax_block_index: argmax_block_index,
            },
        })
    }

    pub(super) fn forward(
        &mut self,
        token_ids: &[u32],
        positions: &[u32],
        current_slots: &[u32],
        contexts: &[Vec<u32>],
        sentinel_slot: u32,
        greedy: bool,
    ) -> Result<ForwardOutput, ModelError> {
        let logical_batch = token_ids.len();
        if logical_batch == 0
            || logical_batch > self.batch_size
            || positions.len() != logical_batch
            || current_slots.len() != logical_batch
            || contexts.len() != logical_batch
            || contexts
                .iter()
                .any(|context| context.is_empty() || context.len() > self.context_bucket)
            || (!greedy && logical_batch != 1)
        {
            return Err(ModelError::Cuda(
                "invalid decode graph batch metadata".into(),
            ));
        }
        let mut padded_token_ids = token_ids.to_vec();
        padded_token_ids.resize(self.batch_size, token_ids[0]);
        let mut padded_positions = positions.to_vec();
        padded_positions.resize(self.batch_size, 0);
        let mut padded_current_slots = current_slots.to_vec();
        let mut padded_context_slots = Vec::with_capacity(self.batch_size * self.context_bucket);
        let mut context_lengths = Vec::with_capacity(self.batch_size);
        for context in contexts {
            context_lengths.push(
                i32::try_from(context.len())
                    .map_err(|_| ModelError::Cuda("decode graph context exceeds i32".into()))?,
            );
            padded_context_slots.extend_from_slice(context);
            padded_context_slots.resize(
                padded_context_slots.len() + self.context_bucket - context.len(),
                sentinel_slot,
            );
        }
        for padding_index in logical_batch..self.batch_size {
            let offset = u32::try_from(padding_index - logical_batch + 1)
                .map_err(|_| ModelError::Cuda("decode padding slot exceeds u32".into()))?;
            let padding_slot = sentinel_slot
                .checked_add(offset)
                .ok_or_else(|| ModelError::Cuda("decode padding slot overflowed".into()))?;
            padded_current_slots.push(padding_slot);
            context_lengths.push(1);
            padded_context_slots.push(padding_slot);
            padded_context_slots.resize(
                padded_context_slots.len() + self.context_bucket - 1,
                sentinel_slot,
            );
        }
        let token_ids = copy_u32(&padded_token_ids, &self.stream, "decode graph tokens")?;
        let positions = copy_u32(&padded_positions, &self.stream, "decode graph positions")?;
        let current_slots = copy_u32(&padded_current_slots, &self.stream, "decode graph KV slots")?;
        let context_slots = copy_u32(
            &padded_context_slots,
            &self.stream,
            "decode graph context slots",
        )?;
        let context_lengths = copy_i32(
            &context_lengths,
            &self.stream,
            "decode graph context lengths",
        )?;

        self.graph
            .update(api::memcpy(&mut self.token_ids, &token_ids))
            .map_err(|error| cuda_error("update graph token", error))?;
        self.graph
            .update(api::memcpy(&mut self.positions, &positions))
            .map_err(|error| cuda_error("update graph position", error))?;
        self.graph
            .update(api::memcpy(&mut self.current_slots, &current_slots))
            .map_err(|error| cuda_error("update graph KV slot", error))?;
        self.graph
            .update(api::memcpy(&mut self.context_slots, &context_slots))
            .map_err(|error| cuda_error("update graph context slots", error))?;
        self.graph
            .update(api::memcpy(&mut self.context_lengths, &context_lengths))
            .map_err(|error| cuda_error("update graph context lengths", error))?;
        self.graph
            .launch()
            .sync_on(&self.stream)
            .map_err(|error| cuda_error("replay decode CUDA graph", error))?;
        if greedy {
            let sampled: Vec<u32> = self
                .sampled_token
                .clone()
                .to_host_vec()
                .sync_on(&self.stream)
                .map_err(|error| cuda_error("copy graph sampled token to host", error))?;
            if sampled.len() != self.batch_size {
                return Err(ModelError::Cuda(format!(
                    "graph argmax returned {} tokens for batch {}",
                    sampled.len(),
                    self.batch_size
                )));
            }
            if logical_batch == 1 {
                return Ok(ForwardOutput::Token(sampled[0]));
            }
            return Ok(ForwardOutput::Tokens(
                sampled.into_iter().take(logical_batch).collect(),
            ));
        }
        let logits: Vec<bf16> = self
            .logits
            .clone()
            .to_host_vec()
            .sync_on(&self.stream)
            .map_err(|error| cuda_error("copy graph logits to host", error))?;
        Ok(ForwardOutput::Logits(
            logits.into_iter().map(bf16::to_f32).collect(),
        ))
    }
}

fn record_gemm(
    scope: &Scope,
    matrix: &Tensor<bf16>,
    rhs: &Tensor<bf16>,
    out: &Tensor<bf16>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), DeviceError> {
    scope
        .record(
            cublas::gemm_bf16_into(matrix, rhs, out, m, n, k)
                .map_err(|error| DeviceError::Internal(error.to_string()))?,
        )?
        .map_err(|error| DeviceError::Internal(error.to_string()))
}
