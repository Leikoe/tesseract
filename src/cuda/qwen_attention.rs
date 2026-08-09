//! Qwen3.5/3.6 full-attention preparation and flat-KV execution.

use std::sync::Arc;

use cuda_async::device_operation::DeviceOp;
use cuda_core::Stream;
use cutile::{
    api,
    core::bf16,
    tensor::{PartitionMut, Reshape, Tensor, TensorView, ToHostVec},
    tile_kernel::TileKernel,
};

use crate::model::ModelError;

use super::kernels;

const QUERY_HEADS: usize = 16;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const ROTARY_DIM: usize = 64;
const HALF_ROTARY: usize = ROTARY_DIM / 2;
const HEAD_BLOCK: usize = 32;
const QUERY_SIZE: usize = QUERY_HEADS * HEAD_DIM;

#[cutile::module]
mod qwen_kernels {
    use cutile::core::*;

    #[allow(clippy::too_many_arguments)]
    #[cutile::entry()]
    unsafe fn prepare_qkvgate(
        q_gate: &Tensor<bf16, { [-1, 8192] }>,
        key: &Tensor<bf16, { [-1, 512] }>,
        value: &Tensor<bf16, { [-1, 512] }>,
        q_weight_delta: &Tensor<bf16, { [256] }>,
        k_weight_delta: &Tensor<bf16, { [256] }>,
        positions: &Tensor<u32, { [-1] }>,
        current_slots: &Tensor<u32, { [-1] }>,
        cosine: &Tensor<f32, { [-1, 32] }>,
        sine: &Tensor<f32, { [-1, 32] }>,
        key_cache_ptr: *mut bf16,
        value_cache_ptr: *mut bf16,
        epsilon: f32,
        query_out: &mut Tensor<bf16, { [1, 1, 32] }>,
        gate_out: &mut Tensor<bf16, { [1, 1, 32] }>,
    ) {
        let pid = get_tile_block_id();
        let row = pid.0;
        let head = pid.1;
        let block = pid.2;
        let q_gate = q_gate.partition(const_shape![1, 32]);
        let q_head_base = head * 16i32;
        const ZERO: f32 = 0.0;
        let mut q_squares: Tile<f32, { [1, 32] }> = broadcast_scalar(ZERO, const_shape![1, 32]);
        for q_block in 0i32..8i32 {
            let q: Tile<f32, { [1, 32] }> = convert_tile(q_gate.load([row, q_head_base + q_block]));
            q_squares = q_squares + q * q;
        }
        let q_sum: Tile<f32, { [1] }> = reduce_sum(q_squares, 1i32);
        let q_sum: Tile<f32, { [] }> = q_sum.reshape(const_shape![]);
        const HEAD_WIDTH: f32 = 256.0;
        let head_width: Tile<f32, { [] }> = broadcast_scalar(HEAD_WIDTH, const_shape![]);
        let epsilon_tile: Tile<f32, { [] }> = scalar_to_tile(epsilon);
        let q_inverse: Tile<f32, { [] }> =
            rsqrt(true_div(q_sum, head_width) + epsilon_tile, ftz::Disabled);
        let q_inverse: Tile<f32, { [1, 32] }> = q_inverse
            .reshape(const_shape![1, 1])
            .broadcast(const_shape![1, 32]);
        let q_delta = q_weight_delta.partition(const_shape![32]);
        const ONE: f32 = 1.0;
        let one: Tile<f32, { [1, 32] }> = broadcast_scalar(ONE, const_shape![1, 32]);

        let q: Tile<f32, { [1, 32] }> = convert_tile(q_gate.load([row, q_head_base + block]));
        let q_scale: Tile<f32, { [1, 32] }> =
            one + convert_tile(q_delta.load([block]).reshape(const_shape![1, 32]));
        let mut q = q * q_inverse * q_scale;
        if block < 2i32 {
            let q_lo: Tile<f32, { [1, 32] }> = convert_tile(q_gate.load([row, q_head_base]));
            let q_hi: Tile<f32, { [1, 32] }> = convert_tile(q_gate.load([row, q_head_base + 1i32]));
            let q_lo_scale: Tile<f32, { [1, 32] }> =
                one + convert_tile(q_delta.load([0i32]).reshape(const_shape![1, 32]));
            let q_hi_scale: Tile<f32, { [1, 32] }> =
                one + convert_tile(q_delta.load([1i32]).reshape(const_shape![1, 32]));
            let q_lo = q_lo * q_inverse * q_lo_scale;
            let q_hi = q_hi * q_inverse * q_hi_scale;
            let position = positions.partition(const_shape![1]).load([row]);
            let position: Tile<i32, { [1] }> = bitcast(position);
            let position: i32 = tile_to_scalar(position.reshape(const_shape![]));
            let cos = cosine.partition(const_shape![1, 32]).load([position, 0i32]);
            let sin = sine.partition(const_shape![1, 32]).load([position, 0i32]);
            if block == 0i32 {
                q = q_lo * cos - q_hi * sin;
            } else {
                q = q_hi * cos + q_lo * sin;
            }
        }
        let q: Tile<bf16, { [1, 1, 32] }> =
            ftof(q.reshape(const_shape![1, 1, 32]), rounding::NearestEven);
        query_out.store(q);
        let gate: Tile<bf16, { [1, 1, 32] }> = q_gate
            .load([row, q_head_base + 8i32 + block])
            .reshape(const_shape![1, 1, 32]);
        gate_out.store(gate);

        if head < 2i32 {
            let key = key.partition(const_shape![1, 32]);
            let value = value.partition(const_shape![1, 32]);
            let head_base = head * 8i32;
            let mut k_squares: Tile<f32, { [1, 32] }> = broadcast_scalar(ZERO, const_shape![1, 32]);
            for k_block in 0i32..8i32 {
                let k: Tile<f32, { [1, 32] }> = convert_tile(key.load([row, head_base + k_block]));
                k_squares = k_squares + k * k;
            }
            let k_sum: Tile<f32, { [1] }> = reduce_sum(k_squares, 1i32);
            let k_sum: Tile<f32, { [] }> = k_sum.reshape(const_shape![]);
            let k_inverse: Tile<f32, { [] }> =
                rsqrt(true_div(k_sum, head_width) + epsilon_tile, ftz::Disabled);
            let k_inverse: Tile<f32, { [1, 32] }> = k_inverse
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![1, 32]);
            let k_delta = k_weight_delta.partition(const_shape![32]);
            let k: Tile<f32, { [1, 32] }> = convert_tile(key.load([row, head_base + block]));
            let k_scale: Tile<f32, { [1, 32] }> =
                one + convert_tile(k_delta.load([block]).reshape(const_shape![1, 32]));
            let mut k = k * k_inverse * k_scale;
            if block < 2i32 {
                let k_lo: Tile<f32, { [1, 32] }> = convert_tile(key.load([row, head_base]));
                let k_hi: Tile<f32, { [1, 32] }> = convert_tile(key.load([row, head_base + 1i32]));
                let k_lo_scale: Tile<f32, { [1, 32] }> =
                    one + convert_tile(k_delta.load([0i32]).reshape(const_shape![1, 32]));
                let k_hi_scale: Tile<f32, { [1, 32] }> =
                    one + convert_tile(k_delta.load([1i32]).reshape(const_shape![1, 32]));
                let k_lo = k_lo * k_inverse * k_lo_scale;
                let k_hi = k_hi * k_inverse * k_hi_scale;
                let position = positions.partition(const_shape![1]).load([row]);
                let position: Tile<i32, { [1] }> = bitcast(position);
                let position: i32 = tile_to_scalar(position.reshape(const_shape![]));
                let cos = cosine.partition(const_shape![1, 32]).load([position, 0i32]);
                let sin = sine.partition(const_shape![1, 32]).load([position, 0i32]);
                if block == 0i32 {
                    k = k_lo * cos - k_hi * sin;
                } else {
                    k = k_hi * cos + k_lo * sin;
                }
            }
            let k: Tile<bf16, { [1, 32] }> = ftof(k, rounding::NearestEven);
            let v: Tile<bf16, { [1, 32] }> = value.load([row, head_base + block]);
            let slot = current_slots.partition(const_shape![1]).load([row]);
            let slot: Tile<i32, { [1] }> = bitcast(slot);
            let slot: i32 = tile_to_scalar(slot.reshape(const_shape![]));
            let cache_offset = (slot * 2i32 + head) * 256i32 + block * 32i32;
            let lane: Tile<i32, { [32] }> = iota(const_shape![32]);
            let offsets: Tile<i32, { [1, 32] }> =
                cache_offset.broadcast(const_shape![1, 32]) + lane.reshape(const_shape![1, 32]);
            let key_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(key_cache_ptr);
            let key_base: PointerTile<*mut bf16, { [1, 1] }> = key_base.reshape(const_shape![1, 1]);
            let key_base: PointerTile<*mut bf16, { [1, 32] }> =
                key_base.broadcast(const_shape![1, 32]);
            let value_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(value_cache_ptr);
            let value_base: PointerTile<*mut bf16, { [1, 1] }> =
                value_base.reshape(const_shape![1, 1]);
            let value_base: PointerTile<*mut bf16, { [1, 32] }> =
                value_base.broadcast(const_shape![1, 32]);
            let key_pointer: PointerTile<*mut bf16, { [1, 32] }> = key_base.offset_tile(offsets);
            let value_pointer: PointerTile<*mut bf16, { [1, 32] }> =
                value_base.offset_tile(offsets);
            let _key_store = store_ptr_tko(
                key_pointer,
                k,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                Latency::<0>,
            );
            let _value_store = store_ptr_tko(
                value_pointer,
                v,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                Latency::<0>,
            );
        }
    }

    #[cutile::entry()]
    fn sigmoid_gate_attention(
        attention: &Tensor<bf16, { [-1, 16, 256] }>,
        gate: &Tensor<bf16, { [-1, 16, 256] }>,
        output: &mut Tensor<bf16, { [1, 1, 32] }>,
    ) {
        let pid = get_tile_block_id();
        let attention: Tile<f32, { [1, 1, 32] }> = convert_tile(
            attention
                .partition(const_shape![1, 1, 32])
                .load([pid.0, pid.1, pid.2]),
        );
        let gate: Tile<f32, { [1, 1, 32] }> = convert_tile(
            gate.partition(const_shape![1, 1, 32])
                .load([pid.0, pid.1, pid.2]),
        );
        const ONE: f32 = 1.0;
        const ZERO: f32 = 0.0;
        let one: Tile<f32, { [1, 1, 32] }> = broadcast_scalar(ONE, const_shape![1, 1, 32]);
        let zero: Tile<f32, { [1, 1, 32] }> = broadcast_scalar(ZERO, const_shape![1, 1, 32]);
        let gated: Tile<bf16, { [1, 1, 32] }> = ftof(
            attention * true_div(one, one + exp(zero - gate)),
            rounding::NearestEven,
        );
        output.store(gated);
    }
}

use qwen_kernels::{prepare_qkvgate, sigmoid_gate_attention};

struct LayerState {
    key: Arc<Tensor<bf16>>,
    value: Arc<Tensor<bf16>>,
}

pub(crate) struct QwenFlatKvAttention {
    layers: Vec<LayerState>,
    cosine: Arc<Tensor<f32>>,
    sine: Arc<Tensor<f32>>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) struct AttentionInput<'a> {
    pub(crate) layer: usize,
    pub(crate) q_gate: Arc<Tensor<bf16>>,
    pub(crate) key: Arc<Tensor<bf16>>,
    pub(crate) value: Arc<Tensor<bf16>>,
    pub(crate) q_weight_delta: Arc<Tensor<bf16>>,
    pub(crate) k_weight_delta: Arc<Tensor<bf16>>,
    pub(crate) positions: Arc<Tensor<u32>>,
    pub(crate) current_slots: Arc<Tensor<u32>>,
    pub(crate) request_indices: Arc<Tensor<u32>>,
    pub(crate) context_slots: &'a TensorView<'a, u32>,
    pub(crate) context_lengths: Arc<Tensor<i32>>,
    pub(crate) rows: usize,
    pub(crate) epsilon: f32,
    pub(crate) stream: &'a Arc<Stream>,
}

impl QwenFlatKvAttention {
    pub(crate) fn load(
        layers: usize,
        capacity: usize,
        max_positions: usize,
        rope_theta: f32,
        stream: &Arc<Stream>,
    ) -> Result<Self, ModelError> {
        let cache_shape = [capacity, KV_HEADS, HEAD_DIM];
        let mut layer_states = Vec::with_capacity(layers);
        for layer in 0..layers {
            let key = api::zeros::<bf16>(&cache_shape)
                .sync_on(stream)
                .map_err(|error| {
                    ModelError::Cuda(format!("allocate Qwen layer {layer} key cache: {error:?}"))
                })?;
            let value = api::zeros::<bf16>(&cache_shape)
                .sync_on(stream)
                .map_err(|error| {
                    ModelError::Cuda(format!(
                        "allocate Qwen layer {layer} value cache: {error:?}"
                    ))
                })?;
            layer_states.push(LayerState {
                key: Arc::new(key),
                value: Arc::new(value),
            });
        }
        let (cosine, sine) = rope_tables(max_positions, rope_theta, stream)?;
        Ok(Self {
            layers: layer_states,
            cosine,
            sine,
        })
    }

    pub(crate) fn enqueue(&self, input: AttentionInput<'_>) -> Result<Tensor<bf16>, ModelError> {
        let state = self.layers.get(input.layer).ok_or_else(|| {
            ModelError::Cuda(format!(
                "Qwen attention layer {} is out of range",
                input.layer
            ))
        })?;
        let rows = input.rows;
        if input.q_gate.shape() != [rows as i32, 8192]
            || input.key.shape() != [rows as i32, 512]
            || input.value.shape() != [rows as i32, 512]
            || input.q_weight_delta.shape() != [HEAD_DIM as i32]
            || input.k_weight_delta.shape() != [HEAD_DIM as i32]
        {
            return Err(ModelError::Cuda(
                "invalid Qwen attention projection geometry".into(),
            ));
        }
        let mut query = api::zeros::<bf16>(&[rows, QUERY_HEADS, HEAD_DIM])
            .sync_on(input.stream)
            .map_err(|error| ModelError::Cuda(format!("allocate Qwen query: {error:?}")))?;
        let mut gate = api::zeros::<bf16>(&[rows, QUERY_HEADS, HEAD_DIM])
            .sync_on(input.stream)
            .map_err(|error| {
                ModelError::Cuda(format!("allocate Qwen attention gate: {error:?}"))
            })?;
        let (_, _, _, _, _, _, _, _, _, _, _, _, query_partition, gate_partition) = unsafe {
            prepare_qkvgate(
                input.q_gate,
                input.key,
                input.value,
                input.q_weight_delta,
                input.k_weight_delta,
                input.positions,
                input.current_slots,
                self.cosine.clone(),
                self.sine.clone(),
                state.key.device_pointer(),
                state.value.device_pointer(),
                input.epsilon,
                (&mut query).partition([1, 1, HEAD_BLOCK]),
                (&mut gate).partition([1, 1, HEAD_BLOCK]),
            )
        }
        .sync_on(input.stream)
        .map_err(|error| ModelError::Cuda(format!("prepare Qwen Q/K/V/gate: {error:?}")))?;
        drop(query_partition);
        drop(gate_partition);

        let mut attention = api::zeros::<bf16>(&[rows, QUERY_HEADS, HEAD_DIM])
            .sync_on(input.stream)
            .map_err(|error| {
                ModelError::Cuda(format!("allocate Qwen attention output: {error:?}"))
            })?;
        const KEY_BLOCK: usize = 16;
        let (_, _, _, _, _, _, attention_partition, _, _) = unsafe {
            kernels::ragged_attention_bf16(
                Arc::new(query),
                input.request_indices,
                input.context_slots,
                input.context_lengths,
                state.key.device_pointer(),
                state.value.device_pointer(),
                (&mut attention).partition([1, 1, HEAD_DIM]),
                1.0 / (HEAD_DIM as f32).sqrt(),
                (QUERY_HEADS / KV_HEADS) as i32,
            )
        }
        .generics(vec![
            KEY_BLOCK.to_string(),
            HEAD_DIM.to_string(),
            KV_HEADS.to_string(),
        ])
        .sync_on(input.stream)
        .map_err(|error| ModelError::Cuda(format!("execute Qwen ragged attention: {error:?}")))?;
        drop(attention_partition);

        let mut gated = api::zeros::<bf16>(&[rows, QUERY_HEADS, HEAD_DIM])
            .sync_on(input.stream)
            .map_err(|error| {
                ModelError::Cuda(format!("allocate Qwen gated attention: {error:?}"))
            })?;
        let (_, _, gated_partition) = sigmoid_gate_attention(
            Arc::new(attention),
            Arc::new(gate),
            (&mut gated).partition([1, 1, HEAD_BLOCK]),
        )
        .sync_on(input.stream)
        .map_err(|error| ModelError::Cuda(format!("gate Qwen attention output: {error:?}")))?;
        drop(gated_partition);
        gated
            .reshape(&[rows, QUERY_SIZE])
            .map_err(|error| ModelError::Cuda(format!("reshape Qwen attention output: {error:?}")))
    }
}

fn rope_tables(
    max_positions: usize,
    theta: f32,
    stream: &Arc<Stream>,
) -> Result<(Arc<Tensor<f32>>, Arc<Tensor<f32>>), ModelError> {
    let elements = max_positions
        .checked_mul(HALF_ROTARY)
        .ok_or_else(|| ModelError::Cuda("Qwen RoPE table size overflowed".into()))?;
    let mut cosine = Vec::with_capacity(elements);
    let mut sine = Vec::with_capacity(elements);
    for position in 0..max_positions {
        for index in 0..HALF_ROTARY {
            let exponent = (2 * index) as f32 / ROTARY_DIM as f32;
            let frequency = 1.0 / theta.powf(exponent);
            let angle = position as f32 * frequency;
            cosine.push(angle.cos());
            sine.push(angle.sin());
        }
    }
    let cosine = api::copy_host_vec_to_device(&Arc::new(cosine))
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("upload Qwen RoPE cosine: {error:?}")))?
        .reshape(&[max_positions, HALF_ROTARY])
        .map_err(|error| ModelError::Cuda(format!("reshape Qwen RoPE cosine: {error:?}")))?;
    let sine = api::copy_host_vec_to_device(&Arc::new(sine))
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("upload Qwen RoPE sine: {error:?}")))?
        .reshape(&[max_positions, HALF_ROTARY])
        .map_err(|error| ModelError::Cuda(format!("reshape Qwen RoPE sine: {error:?}")))?;
    Ok((Arc::new(cosine), Arc::new(sine)))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct AttentionProbe {
    pub(crate) max_abs_error: f32,
}

pub(crate) fn probe(stream: &Arc<Stream>) -> Result<AttentionProbe, ModelError> {
    let rows = 2usize;
    let q_gate = upload_bf16(
        &vec![bf16::from_f32(0.0); rows * 8192],
        &[rows, 8192],
        stream,
    )?;
    let key_host = (0..rows * KV_HEADS * HEAD_DIM)
        .map(|index| bf16::from_f32((index % 29) as f32 / 17.0 - 0.75))
        .collect::<Vec<_>>();
    let value_host = (0..rows * KV_HEADS * HEAD_DIM)
        .map(|index| bf16::from_f32((index % 31) as f32 / 19.0 - 0.5))
        .collect::<Vec<_>>();
    let key = upload_bf16(&key_host, &[rows, 512], stream)?;
    let value = upload_bf16(&value_host, &[rows, 512], stream)?;
    let norm = upload_bf16(&vec![bf16::from_f32(0.0); HEAD_DIM], &[HEAD_DIM], stream)?;
    let positions = upload_u32(&[0, 1], &[rows], stream)?;
    let slots = upload_u32(&[0, 1], &[rows], stream)?;
    let request_indices = upload_u32(&[0, 1], &[rows], stream)?;
    let context_lengths = upload_i32(&[1, 1], &[rows], stream)?;
    let context_slots = upload_u32(&[0, 1], &[rows, 1], stream)?;
    let context_view = context_slots.view(&[rows, 1]).map_err(|error| {
        ModelError::Cuda(format!("view Qwen attention context probe: {error:?}"))
    })?;
    let backend = QwenFlatKvAttention::load(1, rows, 4, 10_000_000.0, stream)?;
    let output = backend.enqueue(AttentionInput {
        layer: 0,
        q_gate,
        key,
        value,
        q_weight_delta: norm.clone(),
        k_weight_delta: norm,
        positions,
        current_slots: slots,
        request_indices,
        context_slots: &context_view,
        context_lengths,
        rows,
        epsilon: 1.0e-6,
        stream,
    })?;
    let actual: Vec<bf16> = output
        .to_host_vec()
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("download Qwen attention probe: {error:?}")))?;
    let mut max_abs_error = 0.0f32;
    for row in 0..rows {
        for query_head in 0..QUERY_HEADS {
            let kv_head = query_head / (QUERY_HEADS / KV_HEADS);
            for element in 0..HEAD_DIM {
                let index = (row * QUERY_HEADS + query_head) * HEAD_DIM + element;
                let expected = bf16::from_f32(
                    value_host[(row * KV_HEADS + kv_head) * HEAD_DIM + element].to_f32() * 0.5,
                )
                .to_f32();
                let error = (actual[index].to_f32() - expected).abs();
                max_abs_error = max_abs_error.max(error);
                if error > 0.01 || !actual[index].to_f32().is_finite() {
                    return Err(ModelError::Cuda(format!(
                        "Qwen full-attention mismatch at {index}: {} != {expected}",
                        actual[index].to_f32()
                    )));
                }
            }
        }
    }
    Ok(AttentionProbe { max_abs_error })
}

fn upload_bf16(
    values: &[bf16],
    shape: &[usize],
    stream: &Arc<Stream>,
) -> Result<Arc<Tensor<bf16>>, ModelError> {
    let tensor = api::copy_host_vec_to_device(&Arc::new(values.to_vec()))
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("upload Qwen attention BF16 probe: {error:?}")))?
        .reshape(shape)
        .map_err(|error| {
            ModelError::Cuda(format!("reshape Qwen attention BF16 probe: {error:?}"))
        })?;
    Ok(Arc::new(tensor))
}

fn upload_u32(
    values: &[u32],
    shape: &[usize],
    stream: &Arc<Stream>,
) -> Result<Arc<Tensor<u32>>, ModelError> {
    let tensor = api::copy_host_vec_to_device(&Arc::new(values.to_vec()))
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("upload Qwen attention U32 probe: {error:?}")))?
        .reshape(shape)
        .map_err(|error| {
            ModelError::Cuda(format!("reshape Qwen attention U32 probe: {error:?}"))
        })?;
    Ok(Arc::new(tensor))
}

fn upload_i32(
    values: &[i32],
    shape: &[usize],
    stream: &Arc<Stream>,
) -> Result<Arc<Tensor<i32>>, ModelError> {
    let tensor = api::copy_host_vec_to_device(&Arc::new(values.to_vec()))
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("upload Qwen attention I32 probe: {error:?}")))?
        .reshape(shape)
        .map_err(|error| {
            ModelError::Cuda(format!("reshape Qwen attention I32 probe: {error:?}"))
        })?;
    Ok(Arc::new(tensor))
}
