//! Qwen3.5/3.6 Gated DeltaNet recurrent execution.

use std::sync::Arc;

use cuda_async::device_operation::DeviceOp;
use cuda_core::Stream;
use cutile::{
    api,
    core::bf16,
    tensor::{PartitionMut, Reshape, Tensor, ToHostVec},
};

use crate::model::ModelError;

const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const VALUE_BLOCK: usize = 32;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    unsafe fn qwen_gdn_decode(
        query: &Tensor<bf16, { [-1, 16, 128] }>,
        key: &Tensor<bf16, { [-1, 16, 128] }>,
        value: &Tensor<bf16, { [-1, 32, 128] }>,
        a: &Tensor<bf16, { [-1, 32] }>,
        b: &Tensor<bf16, { [-1, 32] }>,
        a_log: &Tensor<f32, { [32] }>,
        dt_bias: &Tensor<f32, { [32] }>,
        state_slots: &Tensor<i32, { [-1] }>,
        state_ptr: *mut f32,
        output: &mut Tensor<bf16, { [1, 1, 32] }>,
    ) {
        const EPSILON: f32 = 1.0e-6;
        const SOFTPLUS_THRESHOLD: f32 = 20.0;
        const ONE: f32 = 1.0;
        const ZERO: f32 = 0.0;
        const QUERY_SCALE: f32 = 1.0 / 11.313_708;

        let pid = get_tile_block_id();
        let row = pid.0;
        let value_head = pid.1;
        let value_block = pid.2;
        let key_head = value_head / 2i32;
        let value_offset = value_block * 32i32;

        let query: Tile<f32, { [128] }> = convert_tile(
            query
                .partition(const_shape![1, 1, 128])
                .load([row, key_head, 0i32])
                .reshape(const_shape![128]),
        );
        let key: Tile<f32, { [128] }> = convert_tile(
            key.partition(const_shape![1, 1, 128])
                .load([row, key_head, 0i32])
                .reshape(const_shape![128]),
        );
        let query_norm: Tile<f32, { [] }> = reduce_sum(query * query, 0i32).reshape(const_shape![]);
        let key_norm: Tile<f32, { [] }> = reduce_sum(key * key, 0i32).reshape(const_shape![]);
        let epsilon: Tile<f32, { [] }> = broadcast_scalar(EPSILON, const_shape![]);
        let query_inverse = rsqrt(query_norm + epsilon, ftz::Disabled);
        let key_inverse = rsqrt(key_norm + epsilon, ftz::Disabled);
        let query_scale: Tile<f32, { [] }> = broadcast_scalar(QUERY_SCALE, const_shape![]);
        let query = query
            * (query_inverse * query_scale)
                .reshape(const_shape![1])
                .broadcast(const_shape![128]);
        let key = key
            * key_inverse
                .reshape(const_shape![1])
                .broadcast(const_shape![128]);

        let value: Tile<f32, { [32] }> = convert_tile(
            value
                .partition(const_shape![1, 1, 32])
                .load([row, value_head, value_block])
                .reshape(const_shape![32]),
        );
        let a_value: Tile<f32, { [] }> = convert_tile(
            a.partition(const_shape![1, 1])
                .load([row, value_head])
                .reshape(const_shape![]),
        );
        let b_value: Tile<f32, { [] }> = convert_tile(
            b.partition(const_shape![1, 1])
                .load([row, value_head])
                .reshape(const_shape![]),
        );
        let a_log: Tile<f32, { [] }> = a_log
            .partition(const_shape![1])
            .load([value_head])
            .reshape(const_shape![]);
        let dt_bias: Tile<f32, { [] }> = dt_bias
            .partition(const_shape![1])
            .load([value_head])
            .reshape(const_shape![]);
        let gate_input = a_value + dt_bias;
        let threshold: Tile<f32, { [] }> = broadcast_scalar(SOFTPLUS_THRESHOLD, const_shape![]);
        let one: Tile<f32, { [] }> = broadcast_scalar(ONE, const_shape![]);
        let zero: Tile<f32, { [] }> = broadcast_scalar(ZERO, const_shape![]);
        let softplus = select(
            le_tile(gate_input, threshold),
            log(one + exp(gate_input)),
            gate_input,
        );
        let decay = exp(zero - exp(a_log) * softplus);
        let beta = true_div(one, one + exp(zero - b_value));
        let beta: Tile<bf16, { [] }> = ftof(beta, rounding::NearestEven);
        let beta: Tile<f32, { [] }> = convert_tile(beta);

        let slot = state_slots.partition(const_shape![1]).load([row]);
        let slot: i32 = tile_to_scalar(slot.reshape(const_shape![]));
        let v_lane: Tile<i32, { [32] }> = iota(const_shape![32]);
        let v_lane: Tile<i32, { [32, 1] }> = (v_lane + value_offset).reshape(const_shape![32, 1]);
        let k_lane: Tile<i32, { [128] }> = iota(const_shape![128]);
        let k_lane: Tile<i32, { [1, 128] }> = k_lane.reshape(const_shape![1, 128]);
        let state_base = (slot * 32i32 + value_head) * 128i32 * 128i32;
        let state_offset = state_base.broadcast(const_shape![32, 128])
            + v_lane.broadcast(const_shape![32, 128]) * 128i32.broadcast(const_shape![32, 128])
            + k_lane.broadcast(const_shape![32, 128]);
        let state_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(state_ptr);
        let state_base: PointerTile<*mut f32, { [1, 1] }> = state_base.reshape(const_shape![1, 1]);
        let state_base: PointerTile<*mut f32, { [32, 128] }> =
            state_base.broadcast(const_shape![32, 128]);
        let state_pointer = state_base.offset_tile(state_offset);
        let (state, load_token): (Tile<f32, { [32, 128] }>, Token) = load_ptr_tko(
            state_pointer,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        let decay = decay
            .reshape(const_shape![1, 1])
            .broadcast(const_shape![32, 128]);
        let mut state = state * decay;
        let key_matrix = key
            .reshape(const_shape![1, 128])
            .broadcast(const_shape![32, 128]);
        let predicted: Tile<f32, { [32] }> = reduce_sum(state * key_matrix, 1i32);
        let delta = (value - predicted) * beta.broadcast(const_shape![32]);
        state = state
            + delta
                .reshape(const_shape![32, 1])
                .broadcast(const_shape![32, 128])
                * key_matrix;
        let query_matrix = query
            .reshape(const_shape![1, 128])
            .broadcast(const_shape![32, 128]);
        let result: Tile<f32, { [32] }> = reduce_sum(state * query_matrix, 1i32);
        let result: Tile<bf16, { [1, 1, 32] }> = ftof(
            result.reshape(const_shape![1, 1, 32]),
            rounding::NearestEven,
        );
        output.store(result);
        let _store_token = store_ptr_tko(
            state_pointer,
            state,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            Some(load_token),
            Latency::<0>,
        );
    }
}

use kernels::qwen_gdn_decode;

pub(crate) struct RecurrentState {
    tensor: Tensor<f32>,
    slots: usize,
}

impl RecurrentState {
    pub(crate) fn zeros(slots: usize, stream: &Arc<Stream>) -> Result<Self, ModelError> {
        let tensor = api::zeros::<f32>(&[slots, VALUE_HEADS, HEAD_DIM, HEAD_DIM])
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("allocate GDN state: {error:?}")))?;
        Ok(Self { tensor, slots })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode(
        &mut self,
        query: Arc<Tensor<bf16>>,
        key: Arc<Tensor<bf16>>,
        value: Arc<Tensor<bf16>>,
        a: Arc<Tensor<bf16>>,
        b: Arc<Tensor<bf16>>,
        a_log: Arc<Tensor<f32>>,
        dt_bias: Arc<Tensor<f32>>,
        state_slots: Arc<Tensor<i32>>,
        rows: usize,
        stream: &Arc<Stream>,
    ) -> Result<Tensor<bf16>, ModelError> {
        if self.slots == 0 {
            return Err(ModelError::Cuda("GDN state has no slots".into()));
        }
        if query.shape() != [rows as i32, KEY_HEADS as i32, HEAD_DIM as i32]
            || key.shape() != query.shape()
            || value.shape() != [rows as i32, VALUE_HEADS as i32, HEAD_DIM as i32]
            || a.shape() != [rows as i32, VALUE_HEADS as i32]
            || b.shape() != a.shape()
            || a_log.shape() != [VALUE_HEADS as i32]
            || dt_bias.shape() != [VALUE_HEADS as i32]
            || state_slots.shape() != [rows as i32]
        {
            return Err(ModelError::Cuda("invalid GDN decode geometry".into()));
        }
        let mut output = api::zeros::<bf16>(&[rows, VALUE_HEADS, HEAD_DIM])
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("allocate GDN output: {error:?}")))?;
        let (_, _, _, _, _, _, _, _, _, output_partition) = unsafe {
            qwen_gdn_decode(
                query,
                key,
                value,
                a,
                b,
                a_log,
                dt_bias,
                state_slots,
                self.tensor.device_pointer(),
                (&mut output).partition([1, 1, VALUE_BLOCK]),
            )
        }
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("execute GDN decode: {error:?}")))?;
        drop(output_partition);
        Ok(output)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GdnProbe {
    pub(crate) max_abs_error: f32,
}

pub(crate) fn probe(stream: &Arc<Stream>) -> Result<GdnProbe, ModelError> {
    let rows = 2usize;
    let query = host_bf16(rows * KEY_HEADS * HEAD_DIM, 17, 8.0);
    let key = host_bf16(rows * KEY_HEADS * HEAD_DIM, 19, 9.0);
    let value = host_bf16(rows * VALUE_HEADS * HEAD_DIM, 23, 11.0);
    let a = host_bf16(rows * VALUE_HEADS, 7, 5.0);
    let b = host_bf16(rows * VALUE_HEADS, 11, 6.0);
    let a_log = (0..VALUE_HEADS)
        .map(|index| -2.0 + index as f32 / 64.0)
        .collect::<Vec<_>>();
    let dt_bias = (0..VALUE_HEADS)
        .map(|index| -0.5 + index as f32 / 64.0)
        .collect::<Vec<_>>();
    let state_slots = vec![1i32, 0i32];
    let mut state = RecurrentState::zeros(2, stream)?;
    let query = upload_bf16(&query, &[rows, KEY_HEADS, HEAD_DIM], stream)?;
    let key = upload_bf16(&key, &[rows, KEY_HEADS, HEAD_DIM], stream)?;
    let value = upload_bf16(&value, &[rows, VALUE_HEADS, HEAD_DIM], stream)?;
    let a = upload_bf16(&a, &[rows, VALUE_HEADS], stream)?;
    let b = upload_bf16(&b, &[rows, VALUE_HEADS], stream)?;
    let a_log_device = upload_f32(&a_log, &[VALUE_HEADS], stream)?;
    let dt_bias_device = upload_f32(&dt_bias, &[VALUE_HEADS], stream)?;
    let state_slots = upload_i32(&state_slots, stream)?;
    let _first = state.decode(
        query.clone(),
        key.clone(),
        value.clone(),
        a.clone(),
        b.clone(),
        a_log_device.clone(),
        dt_bias_device.clone(),
        state_slots.clone(),
        rows,
        stream,
    )?;
    let output = state.decode(
        query,
        key,
        value,
        a,
        b,
        a_log_device,
        dt_bias_device,
        state_slots,
        rows,
        stream,
    )?;
    let actual: Vec<bf16> = output
        .to_host_vec()
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("download GDN probe: {error:?}")))?;
    let mut max_abs_error = 0.0f32;
    for row in 0..rows {
        for value_head in 0..VALUE_HEADS {
            let key_head = value_head / 2;
            let q = &query[(row * KEY_HEADS + key_head) * HEAD_DIM
                ..(row * KEY_HEADS + key_head + 1) * HEAD_DIM];
            let k = &key[(row * KEY_HEADS + key_head) * HEAD_DIM
                ..(row * KEY_HEADS + key_head + 1) * HEAD_DIM];
            let q_norm = (q.iter().map(|x| x.to_f32().powi(2)).sum::<f32>() + 1.0e-6).sqrt();
            let k_norm = (k.iter().map(|x| x.to_f32().powi(2)).sum::<f32>() + 1.0e-6).sqrt();
            let gate = a[row * VALUE_HEADS + value_head].to_f32() + dt_bias[value_head];
            let softplus = if gate <= 20.0 {
                (1.0 + gate.exp()).ln()
            } else {
                gate
            };
            let decay = (-a_log[value_head].exp() * softplus).exp();
            let beta =
                bf16::from_f32(1.0 / (1.0 + (-b[row * VALUE_HEADS + value_head].to_f32()).exp()))
                    .to_f32();
            let v = &value[(row * VALUE_HEADS + value_head) * HEAD_DIM
                ..(row * VALUE_HEADS + value_head + 1) * HEAD_DIM];
            let normalized_key_squared = k
                .iter()
                .map(|element| (element.to_f32() / k_norm).powi(2))
                .sum::<f32>();
            let state_coefficient = decay * beta + beta * (1.0 - beta * normalized_key_squared);
            for value_index in 0..HEAD_DIM {
                let mut expected = 0.0f32;
                for key_index in 0..HEAD_DIM {
                    let state_value = state_coefficient
                        * v[value_index].to_f32()
                        * (k[key_index].to_f32() / k_norm);
                    expected += state_value * (q[key_index].to_f32() / q_norm) / 128.0f32.sqrt();
                }
                let expected = bf16::from_f32(expected).to_f32();
                let index = (row * VALUE_HEADS + value_head) * HEAD_DIM + value_index;
                let error = (actual[index].to_f32() - expected).abs();
                max_abs_error = max_abs_error.max(error);
                if error > 0.01 || !actual[index].to_f32().is_finite() || decay <= 0.0 {
                    return Err(ModelError::Cuda(format!(
                        "GDN differential mismatch at {index}: {} != {expected}",
                        actual[index].to_f32()
                    )));
                }
            }
        }
    }
    Ok(GdnProbe { max_abs_error })
}

fn host_bf16(len: usize, modulus: usize, divisor: f32) -> Vec<bf16> {
    (0..len)
        .map(|index| bf16::from_f32((index % modulus) as f32 / divisor - 1.0))
        .collect()
}

fn upload_bf16(
    values: &[bf16],
    shape: &[usize],
    stream: &Arc<Stream>,
) -> Result<Arc<Tensor<bf16>>, ModelError> {
    let tensor = api::copy_host_vec_to_device(&Arc::new(values.to_vec()))
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("upload GDN BF16 probe: {error:?}")))?
        .reshape(shape)
        .map_err(|error| ModelError::Cuda(format!("reshape GDN BF16 probe: {error:?}")))?;
    Ok(Arc::new(tensor))
}

fn upload_f32(
    values: &[f32],
    shape: &[usize],
    stream: &Arc<Stream>,
) -> Result<Arc<Tensor<f32>>, ModelError> {
    let tensor = api::copy_host_vec_to_device(&Arc::new(values.to_vec()))
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("upload GDN F32 probe: {error:?}")))?
        .reshape(shape)
        .map_err(|error| ModelError::Cuda(format!("reshape GDN F32 probe: {error:?}")))?;
    Ok(Arc::new(tensor))
}

fn upload_i32(values: &[i32], stream: &Arc<Stream>) -> Result<Arc<Tensor<i32>>, ModelError> {
    api::copy_host_vec_to_device(&Arc::new(values.to_vec()))
        .sync_on(stream)
        .map(Arc::new)
        .map_err(|error| ModelError::Cuda(format!("upload GDN state slots: {error:?}")))
}
