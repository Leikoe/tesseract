//! Qwen3.5/3.6 Gated DeltaNet recurrent execution.

use std::sync::Arc;

use cuda_async::device_operation::DeviceOp;
use cuda_core::Stream;
use cutile::{
    api,
    core::bf16,
    tensor::{PartitionMut, Reshape, Tensor, ToHostVec},
};

use crate::{chunking::plan_packed_queries, model::ModelError};

const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const VALUE_BLOCK: usize = 32;
const CONV_FEATURES: usize = 8192;
const CONV_WIDTH: usize = 4;
const CONV_STATE_WIDTH: usize = CONV_WIDTH - 1;
const CONV_BLOCK: usize = 256;
const PREFILL_CHUNK: usize = 16;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    unsafe fn qwen_gdn_conv_decode(
        input: &Tensor<bf16, { [-1, 8192] }>,
        weight: &Tensor<bf16, { [8192, 4] }>,
        state_slots: &Tensor<i32, { [-1] }>,
        state_ptr: *mut bf16,
        output: &mut Tensor<bf16, { [1, 256] }>,
    ) {
        let pid = get_tile_block_id();
        let row = pid.0;
        let feature_block = pid.1;
        let slot = state_slots.partition(const_shape![1]).load([row]);
        let slot: i32 = tile_to_scalar(slot.reshape(const_shape![]));

        let lane: Tile<i32, { [256] }> = iota(const_shape![256]);
        let feature_offset = feature_block * 256i32;
        let feature_offset: Tile<i32, { [256] }> = feature_offset.broadcast(const_shape![256]);
        let feature = lane + feature_offset;
        let state_offset = (slot * 8192i32 * 3i32).broadcast(const_shape![256])
            + feature * 3i32.broadcast(const_shape![256]);

        let state_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(state_ptr);
        let state_base: PointerTile<*mut bf16, { [1] }> = state_base.reshape(const_shape![1]);
        let state_base: PointerTile<*mut bf16, { [256] }> = state_base.broadcast(const_shape![256]);
        let state_0_pointer = state_base.offset_tile(state_offset);
        let state_1_pointer =
            state_base.offset_tile(state_offset + 1i32.broadcast(const_shape![256]));
        let state_2_pointer =
            state_base.offset_tile(state_offset + 2i32.broadcast(const_shape![256]));
        let (state_0, state_0_token): (Tile<bf16, { [256] }>, Token) = load_ptr_tko(
            state_0_pointer,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        let (state_1, state_1_token): (Tile<bf16, { [256] }>, Token) = load_ptr_tko(
            state_1_pointer,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        let (state_2, state_2_token): (Tile<bf16, { [256] }>, Token) = load_ptr_tko(
            state_2_pointer,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );

        let input: Tile<bf16, { [1, 256] }> = input
            .partition(const_shape![1, 256])
            .load([row, feature_block]);
        let weight = weight.partition(const_shape![256, 1]);
        let weight_0: Tile<bf16, { [1, 256] }> = weight
            .load([feature_block, 0i32])
            .reshape(const_shape![1, 256]);
        let weight_1: Tile<bf16, { [1, 256] }> = weight
            .load([feature_block, 1i32])
            .reshape(const_shape![1, 256]);
        let weight_2: Tile<bf16, { [1, 256] }> = weight
            .load([feature_block, 2i32])
            .reshape(const_shape![1, 256]);
        let weight_3: Tile<bf16, { [1, 256] }> = weight
            .load([feature_block, 3i32])
            .reshape(const_shape![1, 256]);
        let state_0: Tile<f32, { [1, 256] }> = convert_tile(state_0.reshape(const_shape![1, 256]));
        let state_1: Tile<f32, { [1, 256] }> = convert_tile(state_1.reshape(const_shape![1, 256]));
        let state_2: Tile<f32, { [1, 256] }> = convert_tile(state_2.reshape(const_shape![1, 256]));
        let input_f32: Tile<f32, { [1, 256] }> = convert_tile(input);
        let weight_0: Tile<f32, { [1, 256] }> = convert_tile(weight_0);
        let weight_1: Tile<f32, { [1, 256] }> = convert_tile(weight_1);
        let weight_2: Tile<f32, { [1, 256] }> = convert_tile(weight_2);
        let weight_3: Tile<f32, { [1, 256] }> = convert_tile(weight_3);
        let convolved =
            state_0 * weight_0 + state_1 * weight_1 + state_2 * weight_2 + input_f32 * weight_3;
        const ONE: f32 = 1.0;
        const ZERO: f32 = 0.0;
        let one: Tile<f32, { [1, 256] }> = broadcast_scalar(ONE, const_shape![1, 256]);
        let zero: Tile<f32, { [1, 256] }> = broadcast_scalar(ZERO, const_shape![1, 256]);
        let activated: Tile<bf16, { [1, 256] }> = ftof(
            convolved * true_div(one, one + exp(zero - convolved)),
            rounding::NearestEven,
        );
        output.store(activated);

        let input_state: Tile<bf16, { [256] }> = input.reshape(const_shape![256]);
        let next_state_0: Tile<bf16, { [256] }> =
            ftof(state_1.reshape(const_shape![256]), rounding::NearestEven);
        let next_state_1: Tile<bf16, { [256] }> =
            ftof(state_2.reshape(const_shape![256]), rounding::NearestEven);
        let _state_0_store = store_ptr_tko(
            state_0_pointer,
            next_state_0,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            Some(state_0_token),
            Latency::<0>,
        );
        let _state_1_store = store_ptr_tko(
            state_1_pointer,
            next_state_1,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            Some(state_1_token),
            Latency::<0>,
        );
        let _state_2_store = store_ptr_tko(
            state_2_pointer,
            input_state,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            Some(state_2_token),
            Latency::<0>,
        );
    }

    /// Runs the width-four causal convolution for every packed request while
    /// keeping the three-token tail resident in registers. Requests are
    /// independent tile blocks; channels are parallel within each request.
    #[cutile::entry()]
    unsafe fn qwen_gdn_conv_prefill(
        input: &Tensor<bf16, { [-1, 8192] }>,
        weight: &Tensor<bf16, { [8192, 4] }>,
        query_start_offsets: &Tensor<i32, { [-1] }>,
        state_slots: &Tensor<i32, { [-1] }>,
        state_ptr: *mut bf16,
        output_ptr: *mut bf16,
        completion: &mut Tensor<i32, { [1, 1] }>,
    ) {
        const ONE: f32 = 1.0;
        const ZERO: f32 = 0.0;
        const ZERO_I32: i32 = 0;

        let pid = get_tile_block_id();
        let request = pid.0;
        let feature_block = pid.1;
        let offsets = query_start_offsets.partition(const_shape![1]);
        let start: i32 = tile_to_scalar(offsets.load([request]).reshape(const_shape![]));
        let end: i32 = tile_to_scalar(offsets.load([request + 1i32]).reshape(const_shape![]));
        let slot = state_slots.partition(const_shape![1]).load([request]);
        let slot: i32 = tile_to_scalar(slot.reshape(const_shape![]));

        let lane: Tile<i32, { [256] }> = iota(const_shape![256]);
        let feature_offset = feature_block * 256i32;
        let feature_offset: Tile<i32, { [256] }> = feature_offset.broadcast(const_shape![256]);
        let feature = lane + feature_offset;
        let state_offset = (slot * 8192i32 * 3i32).broadcast(const_shape![256])
            + feature * 3i32.broadcast(const_shape![256]);

        let state_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(state_ptr);
        let state_base: PointerTile<*mut bf16, { [1] }> = state_base.reshape(const_shape![1]);
        let state_base: PointerTile<*mut bf16, { [256] }> = state_base.broadcast(const_shape![256]);
        let state_0_pointer = state_base.offset_tile(state_offset);
        let state_1_pointer =
            state_base.offset_tile(state_offset + 1i32.broadcast(const_shape![256]));
        let state_2_pointer =
            state_base.offset_tile(state_offset + 2i32.broadcast(const_shape![256]));
        let (state_0, state_0_token): (Tile<bf16, { [256] }>, Token) = load_ptr_tko(
            state_0_pointer,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        let (state_1, state_1_token): (Tile<bf16, { [256] }>, Token) = load_ptr_tko(
            state_1_pointer,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        let (state_2, state_2_token): (Tile<bf16, { [256] }>, Token) = load_ptr_tko(
            state_2_pointer,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        let mut state_0: Tile<f32, { [1, 256] }> =
            convert_tile(state_0.reshape(const_shape![1, 256]));
        let mut state_1: Tile<f32, { [1, 256] }> =
            convert_tile(state_1.reshape(const_shape![1, 256]));
        let mut state_2: Tile<f32, { [1, 256] }> =
            convert_tile(state_2.reshape(const_shape![1, 256]));

        let weight = weight.partition(const_shape![256, 1]);
        let weight_0: Tile<f32, { [1, 256] }> = convert_tile(
            weight
                .load([feature_block, 0i32])
                .reshape(const_shape![1, 256]),
        );
        let weight_1: Tile<f32, { [1, 256] }> = convert_tile(
            weight
                .load([feature_block, 1i32])
                .reshape(const_shape![1, 256]),
        );
        let weight_2: Tile<f32, { [1, 256] }> = convert_tile(
            weight
                .load([feature_block, 2i32])
                .reshape(const_shape![1, 256]),
        );
        let weight_3: Tile<f32, { [1, 256] }> = convert_tile(
            weight
                .load([feature_block, 3i32])
                .reshape(const_shape![1, 256]),
        );
        let one: Tile<f32, { [1, 256] }> = broadcast_scalar(ONE, const_shape![1, 256]);
        let zero: Tile<f32, { [1, 256] }> = broadcast_scalar(ZERO, const_shape![1, 256]);
        let input = input.partition(const_shape![1, 256]);
        let output_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(output_ptr);
        let output_base: PointerTile<*mut bf16, { [1] }> = output_base.reshape(const_shape![1]);
        let output_base: PointerTile<*mut bf16, { [256] }> =
            output_base.broadcast(const_shape![256]);

        for row in start..end {
            let value: Tile<bf16, { [1, 256] }> = input.load([row, feature_block]);
            let value_f32: Tile<f32, { [1, 256] }> = convert_tile(value);
            let convolved =
                state_0 * weight_0 + state_1 * weight_1 + state_2 * weight_2 + value_f32 * weight_3;
            let activated: Tile<bf16, { [256] }> = ftof(
                (convolved * true_div(one, one + exp(zero - convolved))).reshape(const_shape![256]),
                rounding::NearestEven,
            );
            let output_offset =
                (row * 8192i32 + feature_block * 256i32).broadcast(const_shape![256]) + lane;
            let output_pointer = output_base.offset_tile(output_offset);
            let _output_token = store_ptr_tko(
                output_pointer,
                activated,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                Latency::<0>,
            );
            state_0 = state_1;
            state_1 = state_2;
            state_2 = value_f32;
        }

        let next_state_0: Tile<bf16, { [256] }> =
            ftof(state_0.reshape(const_shape![256]), rounding::NearestEven);
        let next_state_1: Tile<bf16, { [256] }> =
            ftof(state_1.reshape(const_shape![256]), rounding::NearestEven);
        let next_state_2: Tile<bf16, { [256] }> =
            ftof(state_2.reshape(const_shape![256]), rounding::NearestEven);
        let _state_0_store = store_ptr_tko(
            state_0_pointer,
            next_state_0,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            Some(state_0_token),
            Latency::<0>,
        );
        let _state_1_store = store_ptr_tko(
            state_1_pointer,
            next_state_1,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            Some(state_1_token),
            Latency::<0>,
        );
        let _state_2_store = store_ptr_tko(
            state_2_pointer,
            next_state_2,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            Some(state_2_token),
            Latency::<0>,
        );
        completion.store(broadcast_scalar(ZERO_I32, const_shape![1, 1]));
    }

    /// Computes the chunk-local cumulative log decay and BF16-rounded update
    /// rate used by both the triangular solve and recurrent state update.
    #[cutile::entry()]
    unsafe fn qwen_gdn_prefill_gates(
        a: &Tensor<bf16, { [-1, 32] }>,
        b: &Tensor<bf16, { [-1, 32] }>,
        a_log: &Tensor<f32, { [32] }>,
        dt_bias: &Tensor<f32, { [32] }>,
        chunk_starts: &Tensor<i32, { [-1] }>,
        chunk_lengths: &Tensor<i32, { [-1] }>,
        log_decay_cumsum_ptr: *mut f32,
        beta_ptr: *mut bf16,
        completion: &mut Tensor<i32, { [1, 1] }>,
    ) {
        const SOFTPLUS_THRESHOLD: f32 = 20.0;
        const ONE: f32 = 1.0;
        const ZERO: f32 = 0.0;
        const ZERO_I32: i32 = 0;

        let pid = get_tile_block_id();
        let chunk = pid.0;
        let head = pid.1;
        let start: i32 = tile_to_scalar(
            chunk_starts
                .partition(const_shape![1])
                .load([chunk])
                .reshape(const_shape![]),
        );
        let len: i32 = tile_to_scalar(
            chunk_lengths
                .partition(const_shape![1])
                .load([chunk])
                .reshape(const_shape![]),
        );
        let a_log: Tile<f32, { [] }> = a_log
            .partition(const_shape![1])
            .load([head])
            .reshape(const_shape![]);
        let dt_bias: Tile<f32, { [] }> = dt_bias
            .partition(const_shape![1])
            .load([head])
            .reshape(const_shape![]);
        let threshold: Tile<f32, { [] }> = broadcast_scalar(SOFTPLUS_THRESHOLD, const_shape![]);
        let one: Tile<f32, { [] }> = broadcast_scalar(ONE, const_shape![]);
        let zero: Tile<f32, { [] }> = broadcast_scalar(ZERO, const_shape![]);
        let a = a.partition(const_shape![1, 1]);
        let b = b.partition(const_shape![1, 1]);
        let log_decay_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(log_decay_cumsum_ptr);
        let beta_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(beta_ptr);
        let mut cumulative_log_decay: Tile<f32, { [] }> = broadcast_scalar(ZERO, const_shape![]);

        for local_row in 0i32..len {
            let row = start + local_row;
            let gate_input: Tile<f32, { [] }> =
                convert_tile(a.load([row, head]).reshape(const_shape![])) + dt_bias;
            let softplus = select(
                le_tile(gate_input, threshold),
                log(one + exp(gate_input)),
                gate_input,
            );
            cumulative_log_decay = cumulative_log_decay - exp(a_log) * softplus;
            let beta: Tile<f32, { [] }> = convert_tile(b.load([row, head]).reshape(const_shape![]));
            let beta = true_div(one, one + exp(zero - beta));
            let beta: Tile<bf16, { [] }> = ftof(beta, rounding::NearestEven);
            let offset: Tile<i32, { [] }> = (row * 32i32 + head).broadcast(const_shape![]);
            let log_decay_pointer: PointerTile<*mut f32, { [] }> =
                log_decay_base.offset_tile(offset);
            let beta_pointer: PointerTile<*mut bf16, { [] }> = beta_base.offset_tile(offset);
            let _log_decay_store = store_ptr_tko(
                log_decay_pointer,
                cumulative_log_decay,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                Latency::<0>,
            );
            let _beta_store = store_ptr_tko(
                beta_pointer,
                beta,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                Latency::<0>,
            );
        }
        completion.store(broadcast_scalar(ZERO_I32, const_shape![1, 1]));
    }

    /// Forms the gated KKT matrix and solves its unit-lower-triangular system
    /// in registers. The one resulting inverse is shared by W and U.
    #[cutile::entry()]
    unsafe fn qwen_gdn_prefill_kkt_solve(
        mixed_qkv_ptr: *mut bf16,
        log_decay_cumsum_ptr: *mut f32,
        beta_ptr: *mut bf16,
        chunk_starts: &Tensor<i32, { [-1] }>,
        chunk_lengths: &Tensor<i32, { [-1] }>,
        inverse: &mut Tensor<bf16, { [1, 1, 256] }>,
    ) {
        const EPSILON: f32 = 1.0e-6;
        const ONE: f32 = 1.0;
        const ZERO: f32 = 0.0;

        let pid = get_tile_block_id();
        let chunk = pid.0;
        let value_head = pid.1;
        let key_head = value_head / 2i32;
        let start: i32 = tile_to_scalar(
            chunk_starts
                .partition(const_shape![1])
                .load([chunk])
                .reshape(const_shape![]),
        );
        let len: i32 = tile_to_scalar(
            chunk_lengths
                .partition(const_shape![1])
                .load([chunk])
                .reshape(const_shape![]),
        );

        let row: Tile<i32, { [16] }> = iota(const_shape![16]);
        let row: Tile<i32, { [16, 1] }> = row.reshape(const_shape![16, 1]);
        let column: Tile<i32, { [128] }> = iota(const_shape![128]);
        let column: Tile<i32, { [1, 128] }> = column.reshape(const_shape![1, 128]);
        let row_shape = const_shape![16, 1];
        let key_offset = ((start.broadcast(row_shape) + row) * 8192i32.broadcast(row_shape)
            + ((key_head + 16i32) * 128i32).broadcast(row_shape))
        .broadcast(const_shape![16, 128])
            + column.broadcast(const_shape![16, 128]);
        let key_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(mixed_qkv_ptr);
        let key_base: PointerTile<*mut bf16, { [1, 1] }> = key_base.reshape(const_shape![1, 1]);
        let key_base: PointerTile<*mut bf16, { [16, 128] }> =
            key_base.broadcast(const_shape![16, 128]);
        let key_pointer: PointerTile<*mut bf16, { [16, 128] }> = key_base.offset_tile(key_offset);
        let valid_row: Tile<bool, { [16, 1] }> = lt_tile(row, len.broadcast(const_shape![16, 1]));
        let valid_key: Tile<bool, { [16, 128] }> = valid_row.broadcast(const_shape![16, 128]);
        let (key, _): (Tile<bf16, { [16, 128] }>, Token) = load_ptr_tko(
            key_pointer,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(valid_key),
            Some(bf16::ZERO),
            None,
            Latency::<0>,
        );
        let key_f32: Tile<f32, { [16, 128] }> = convert_tile(key);
        let key_square_sum: Tile<f32, { [16, 1] }> = reduce_sum(key_f32 * key_f32, 1i32);
        let epsilon: Tile<f32, { [16, 1] }> = broadcast_scalar(EPSILON, const_shape![16, 1]);
        let key_inverse = rsqrt(key_square_sum + epsilon, ftz::Disabled);
        let key: Tile<bf16, { [16, 128] }> = ftof(
            key_f32 * key_inverse.broadcast(const_shape![16, 128]),
            rounding::NearestEven,
        );
        let zero_matrix: Tile<f32, { [16, 16] }> = broadcast_scalar(ZERO, const_shape![16, 16]);
        let kkt: Tile<f32, { [16, 16] }> = mma(key, key.transpose(), zero_matrix);

        let head_offset = (start.broadcast(row_shape) + row) * 32i32.broadcast(row_shape)
            + value_head.broadcast(row_shape);
        let log_base: PointerTile<*mut f32, { [] }> = pointer_to_tile(log_decay_cumsum_ptr);
        let log_base: PointerTile<*mut f32, { [1, 1] }> = log_base.reshape(const_shape![1, 1]);
        let log_base: PointerTile<*mut f32, { [16, 1] }> = log_base.broadcast(const_shape![16, 1]);
        let log_pointer: PointerTile<*mut f32, { [16, 1] }> = log_base.offset_tile(head_offset);
        let (log_decay, _): (Tile<f32, { [16, 1] }>, Token) = load_ptr_tko(
            log_pointer,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(valid_row),
            Some(ZERO),
            None,
            Latency::<0>,
        );
        let beta_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(beta_ptr);
        let beta_base: PointerTile<*mut bf16, { [1, 1] }> = beta_base.reshape(const_shape![1, 1]);
        let beta_base: PointerTile<*mut bf16, { [16, 1] }> =
            beta_base.broadcast(const_shape![16, 1]);
        let beta_pointer: PointerTile<*mut bf16, { [16, 1] }> = beta_base.offset_tile(head_offset);
        let (beta, _): (Tile<bf16, { [16, 1] }>, Token) = load_ptr_tko(
            beta_pointer,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(valid_row),
            Some(bf16::ZERO),
            None,
            Latency::<0>,
        );
        let beta: Tile<f32, { [16, 1] }> = convert_tile(beta);

        let causal_column: Tile<i32, { [16] }> = iota(const_shape![16]);
        let causal_column: Tile<i32, { [1, 16] }> = causal_column.reshape(const_shape![1, 16]);
        let valid_column: Tile<bool, { [1, 16] }> =
            lt_tile(causal_column, len.broadcast(const_shape![1, 16]));
        let strict_lower = gt_tile(
            row.broadcast(const_shape![16, 16]),
            causal_column.broadcast(const_shape![16, 16]),
        );
        let valid_matrix = valid_row.broadcast(const_shape![16, 16])
            & valid_column.broadcast(const_shape![16, 16]);
        let decay_ratio = exp(log_decay.broadcast(const_shape![16, 16])
            - log_decay.transpose().broadcast(const_shape![16, 16]));
        let system = select(
            strict_lower & valid_matrix,
            kkt * decay_ratio * beta.broadcast(const_shape![16, 16]),
            zero_matrix,
        );
        let mut solved = zero_matrix - system;

        for solved_row in 2i32..16i32 {
            let selected_row = eq_tile(row, solved_row.broadcast(const_shape![16, 1]))
                .broadcast(const_shape![16, 16]);
            let mut next_row: Tile<f32, { [1, 16] }> = reduce_sum(
                select(selected_row, zero_matrix - system, zero_matrix),
                0i32,
            );
            let before_row = lt_tile(causal_column, solved_row.broadcast(const_shape![1, 16]));
            let row_is_valid: Tile<bool, { [1, 16] }> = lt_tile(
                solved_row.broadcast(const_shape![1, 16]),
                len.broadcast(const_shape![1, 16]),
            );
            next_row = select(
                before_row & row_is_valid,
                next_row,
                broadcast_scalar(ZERO, const_shape![1, 16]),
            );
            let correction: Tile<f32, { [1, 16] }> = reduce_sum(
                next_row.transpose().broadcast(const_shape![16, 16]) * solved,
                0i32,
            );
            next_row = next_row + correction;
            solved = select(
                selected_row,
                next_row.broadcast(const_shape![16, 16]),
                solved,
            );
        }

        let diagonal = eq_tile(
            row.broadcast(const_shape![16, 16]),
            causal_column.broadcast(const_shape![16, 16]),
        );
        let one_matrix: Tile<f32, { [16, 16] }> = broadcast_scalar(ONE, const_shape![16, 16]);
        solved = select(
            valid_matrix,
            solved + select(diagonal, one_matrix, zero_matrix),
            zero_matrix,
        );
        let solved: Tile<bf16, { [1, 1, 256] }> = ftof(
            solved.reshape(const_shape![1, 1, 256]),
            rounding::NearestEven,
        );
        inverse.store(solved);
    }

    #[cutile::entry()]
    unsafe fn qwen_gdn_decode(
        mixed_qkv: &Tensor<bf16, { [-1, 8192] }>,
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

        let mixed_qkv_128 = mixed_qkv.partition(const_shape![1, 128]);
        let query: Tile<f32, { [128] }> = convert_tile(
            mixed_qkv_128
                .load([row, key_head])
                .reshape(const_shape![128]),
        );
        let key_block = key_head + 16i32;
        let key: Tile<f32, { [128] }> = convert_tile(
            mixed_qkv_128
                .load([row, key_block])
                .reshape(const_shape![128]),
        );
        let query_norm: Tile<f32, { [1] }> = reduce_sum(query * query, 0i32);
        let query_norm: Tile<f32, { [] }> = query_norm.reshape(const_shape![]);
        let key_norm: Tile<f32, { [1] }> = reduce_sum(key * key, 0i32);
        let key_norm: Tile<f32, { [] }> = key_norm.reshape(const_shape![]);
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

        let packed_value_block = 128i32 + value_head * 4i32 + value_block;
        let value: Tile<f32, { [32] }> = convert_tile(
            mixed_qkv
                .partition(const_shape![1, 32])
                .load([row, packed_value_block])
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
        let value_offset: Tile<i32, { [32] }> = value_offset.broadcast(const_shape![32]);
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
        let beta: Tile<f32, { [1] }> = beta.reshape(const_shape![1]);
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

    #[cutile::entry()]
    fn qwen_gdn_output_gate(
        input: &Tensor<bf16, { [-1, 32, 128] }>,
        gate: &Tensor<bf16, { [-1, 32, 128] }>,
        weight: &Tensor<bf16, { [128] }>,
        epsilon: f32,
        output: &mut Tensor<bf16, { [1, 1, 128] }>,
    ) {
        let pid = get_tile_block_id();
        let input: Tile<f32, { [128] }> = convert_tile(
            input
                .partition(const_shape![1, 1, 128])
                .load([pid.0, pid.1, 0i32])
                .reshape(const_shape![128]),
        );
        let gate: Tile<f32, { [128] }> = convert_tile(
            gate.partition(const_shape![1, 1, 128])
                .load([pid.0, pid.1, 0i32])
                .reshape(const_shape![128]),
        );
        let weight: Tile<f32, { [128] }> =
            convert_tile(weight.partition(const_shape![128]).load([0i32]));
        let square_sum: Tile<f32, { [1] }> = reduce_sum(input * input, 0i32);
        let square_sum: Tile<f32, { [] }> = square_sum.reshape(const_shape![]);
        const WIDTH: f32 = 128.0;
        let width: Tile<f32, { [] }> = broadcast_scalar(WIDTH, const_shape![]);
        let epsilon: Tile<f32, { [] }> = scalar_to_tile(epsilon);
        let inverse: Tile<f32, { [] }> =
            rsqrt(true_div(square_sum, width) + epsilon, ftz::Disabled);
        let inverse: Tile<f32, { [1] }> = inverse.reshape(const_shape![1]);
        let inverse: Tile<f32, { [128] }> = inverse.broadcast(const_shape![128]);
        const ONE: f32 = 1.0;
        const ZERO: f32 = 0.0;
        let one: Tile<f32, { [128] }> = broadcast_scalar(ONE, const_shape![128]);
        let zero: Tile<f32, { [128] }> = broadcast_scalar(ZERO, const_shape![128]);
        let gated = input * inverse * weight * gate * true_div(one, one + exp(zero - gate));
        let gated: Tile<bf16, { [1, 1, 128] }> = ftof(
            gated.reshape(const_shape![1, 1, 128]),
            rounding::NearestEven,
        );
        output.store(gated);
    }
}

use kernels::{
    qwen_gdn_conv_decode, qwen_gdn_conv_prefill, qwen_gdn_decode, qwen_gdn_output_gate,
    qwen_gdn_prefill_gates, qwen_gdn_prefill_kkt_solve,
};

pub(crate) struct GdnPrefillPlan {
    chunk_starts: Arc<Tensor<i32>>,
    chunk_lengths: Arc<Tensor<i32>>,
    chunks: usize,
    rows: usize,
}

impl GdnPrefillPlan {
    pub(crate) fn from_offsets(offsets: &[u32], stream: &Arc<Stream>) -> Result<Self, ModelError> {
        let chunks = plan_packed_queries(offsets, PREFILL_CHUNK as u32)
            .map_err(|error| ModelError::Cuda(format!("plan packed GDN prefill: {error}")))?;
        let starts = chunks
            .iter()
            .map(|chunk| i32::try_from(chunk.start()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| ModelError::Cuda("GDN prefill row exceeds i32".into()))?;
        let lengths = chunks
            .iter()
            .map(|chunk| i32::try_from(chunk.len()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| ModelError::Cuda("GDN prefill chunk length exceeds i32".into()))?;
        let rows = offsets.last().copied().unwrap_or(0) as usize;
        Ok(Self {
            chunk_starts: upload_i32(&starts, stream)?,
            chunk_lengths: upload_i32(&lengths, stream)?,
            chunks: chunks.len(),
            rows,
        })
    }

    pub(crate) const fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) const fn chunks(&self) -> usize {
        self.chunks
    }
}

struct GdnPrefillGates {
    log_decay_cumsum: Tensor<f32>,
    beta: Tensor<bf16>,
}

fn prepare_prefill_gates(
    a: Arc<Tensor<bf16>>,
    b: Arc<Tensor<bf16>>,
    a_log: Arc<Tensor<f32>>,
    dt_bias: Arc<Tensor<f32>>,
    plan: &GdnPrefillPlan,
    stream: &Arc<Stream>,
) -> Result<GdnPrefillGates, ModelError> {
    let rows = plan.rows();
    if plan.chunks() == 0
        || a.shape() != [rows as i32, VALUE_HEADS as i32]
        || b.shape() != a.shape()
        || a_log.shape() != [VALUE_HEADS as i32]
        || dt_bias.shape() != [VALUE_HEADS as i32]
    {
        return Err(ModelError::Cuda("invalid GDN prefill gate geometry".into()));
    }
    let log_decay_cumsum = api::zeros::<f32>(&[rows, VALUE_HEADS])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate GDN prefill log gates: {error:?}")))?;
    let beta = api::zeros::<bf16>(&[rows, VALUE_HEADS])
        .sync_on(stream)
        .map_err(|error| {
            ModelError::Cuda(format!("allocate GDN prefill update gates: {error:?}"))
        })?;
    let mut completion = api::zeros::<i32>(&[plan.chunks(), VALUE_HEADS])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate GDN gate launch grid: {error:?}")))?;
    let (_, _, _, _, _, _, _, _, completion_partition) = unsafe {
        qwen_gdn_prefill_gates(
            a,
            b,
            a_log,
            dt_bias,
            plan.chunk_starts.clone(),
            plan.chunk_lengths.clone(),
            log_decay_cumsum.device_pointer(),
            beta.device_pointer(),
            (&mut completion).partition([1, 1]),
        )
    }
    .sync_on(stream)
    .map_err(|error| ModelError::Cuda(format!("execute GDN prefill gates: {error:?}")))?;
    drop(completion_partition);
    Ok(GdnPrefillGates {
        log_decay_cumsum,
        beta,
    })
}

fn solve_prefill_kkt(
    mixed_qkv: Arc<Tensor<bf16>>,
    gates: &GdnPrefillGates,
    plan: &GdnPrefillPlan,
    stream: &Arc<Stream>,
) -> Result<Tensor<bf16>, ModelError> {
    if mixed_qkv.shape() != [plan.rows() as i32, CONV_FEATURES as i32] {
        return Err(ModelError::Cuda("invalid GDN prefill KKT geometry".into()));
    }
    let mut inverse =
        api::zeros::<bf16>(&[plan.chunks(), VALUE_HEADS, PREFILL_CHUNK * PREFILL_CHUNK])
            .sync_on(stream)
            .map_err(|error| {
                ModelError::Cuda(format!("allocate GDN prefill inverse: {error:?}"))
            })?;
    let (_, _, _, _, _, inverse_partition) = unsafe {
        qwen_gdn_prefill_kkt_solve(
            mixed_qkv.device_pointer(),
            gates.log_decay_cumsum.device_pointer(),
            gates.beta.device_pointer(),
            plan.chunk_starts.clone(),
            plan.chunk_lengths.clone(),
            (&mut inverse).partition([1, 1, PREFILL_CHUNK * PREFILL_CHUNK]),
        )
    }
    .sync_on(stream)
    .map_err(|error| ModelError::Cuda(format!("execute GDN prefill KKT solve: {error:?}")))?;
    drop(inverse_partition);
    Ok(inverse)
}

pub(crate) fn output_gate(
    input: Arc<Tensor<bf16>>,
    gate: Arc<Tensor<bf16>>,
    weight: Arc<Tensor<bf16>>,
    epsilon: f32,
    rows: usize,
    stream: &Arc<Stream>,
) -> Result<Tensor<bf16>, ModelError> {
    if input.shape() != [rows as i32, VALUE_HEADS as i32, HEAD_DIM as i32]
        || gate.shape() != input.shape()
        || weight.shape() != [HEAD_DIM as i32]
        || !epsilon.is_finite()
        || epsilon <= 0.0
    {
        return Err(ModelError::Cuda("invalid GDN output gate geometry".into()));
    }
    let mut output = api::zeros::<bf16>(&[rows, VALUE_HEADS, HEAD_DIM])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate GDN gated output: {error:?}")))?;
    let (_, _, _, _, output_partition) = qwen_gdn_output_gate(
        input,
        gate,
        weight,
        epsilon,
        (&mut output).partition([1, 1, HEAD_DIM]),
    )
    .sync_on(stream)
    .map_err(|error| ModelError::Cuda(format!("execute GDN output gate: {error:?}")))?;
    drop(output_partition);
    Ok(output)
}

pub(crate) struct GdnState {
    conv: Tensor<bf16>,
    tensor: Tensor<f32>,
    slots: usize,
}

impl GdnState {
    pub(crate) fn zeros(slots: usize, stream: &Arc<Stream>) -> Result<Self, ModelError> {
        let conv = api::zeros::<bf16>(&[slots, CONV_FEATURES, CONV_STATE_WIDTH])
            .sync_on(stream)
            .map_err(|error| {
                ModelError::Cuda(format!("allocate GDN convolution state: {error:?}"))
            })?;
        let tensor = api::zeros::<f32>(&[slots, VALUE_HEADS, HEAD_DIM, HEAD_DIM])
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("allocate GDN state: {error:?}")))?;
        Ok(Self {
            conv,
            tensor,
            slots,
        })
    }

    pub(crate) fn decode_conv(
        &mut self,
        input: Arc<Tensor<bf16>>,
        weight: Arc<Tensor<bf16>>,
        state_slots: Arc<Tensor<i32>>,
        rows: usize,
        stream: &Arc<Stream>,
    ) -> Result<Tensor<bf16>, ModelError> {
        if self.slots == 0
            || input.shape() != [rows as i32, CONV_FEATURES as i32]
            || weight.shape() != [CONV_FEATURES as i32, CONV_WIDTH as i32]
            || state_slots.shape() != [rows as i32]
        {
            return Err(ModelError::Cuda("invalid GDN convolution geometry".into()));
        }
        let mut output = api::zeros::<bf16>(&[rows, CONV_FEATURES])
            .sync_on(stream)
            .map_err(|error| {
                ModelError::Cuda(format!("allocate GDN convolution output: {error:?}"))
            })?;
        let (_, _, _, _, output_partition) = unsafe {
            qwen_gdn_conv_decode(
                input,
                weight,
                state_slots,
                self.conv.device_pointer(),
                (&mut output).partition([1, CONV_BLOCK]),
            )
        }
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("execute GDN convolution decode: {error:?}")))?;
        drop(output_partition);
        Ok(output)
    }

    pub(crate) fn prefill_conv(
        &mut self,
        input: Arc<Tensor<bf16>>,
        weight: Arc<Tensor<bf16>>,
        query_start_offsets: Arc<Tensor<i32>>,
        state_slots: Arc<Tensor<i32>>,
        rows: usize,
        requests: usize,
        stream: &Arc<Stream>,
    ) -> Result<Tensor<bf16>, ModelError> {
        if self.slots == 0
            || requests == 0
            || input.shape() != [rows as i32, CONV_FEATURES as i32]
            || weight.shape() != [CONV_FEATURES as i32, CONV_WIDTH as i32]
            || query_start_offsets.shape() != [(requests + 1) as i32]
            || state_slots.shape() != [requests as i32]
        {
            return Err(ModelError::Cuda(
                "invalid GDN convolution prefill geometry".into(),
            ));
        }
        let output = api::zeros::<bf16>(&[rows, CONV_FEATURES])
            .sync_on(stream)
            .map_err(|error| {
                ModelError::Cuda(format!(
                    "allocate GDN convolution prefill output: {error:?}"
                ))
            })?;
        // cuTile derives the launch grid from tensor partitions. This tiny
        // completion matrix provides exactly one block per request/channel tile
        // while the packed-token output is written through its device pointer.
        let mut completion = api::zeros::<i32>(&[requests, CONV_FEATURES / CONV_BLOCK])
            .sync_on(stream)
            .map_err(|error| {
                ModelError::Cuda(format!("allocate GDN convolution launch grid: {error:?}"))
            })?;
        let (_, _, _, _, _, _, completion_partition) = unsafe {
            qwen_gdn_conv_prefill(
                input,
                weight,
                query_start_offsets,
                state_slots,
                self.conv.device_pointer(),
                output.device_pointer(),
                (&mut completion).partition([1, 1]),
            )
        }
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("execute GDN convolution prefill: {error:?}")))?;
        drop(completion_partition);
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode(
        &mut self,
        mixed_qkv: Arc<Tensor<bf16>>,
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
        if mixed_qkv.shape() != [rows as i32, CONV_FEATURES as i32]
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
        let (_, _, _, _, _, _, _, output_partition) = unsafe {
            qwen_gdn_decode(
                mixed_qkv,
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
    let conv_max_abs_error = probe_conv(stream)?;
    let prefill_gate_max_abs_error = probe_prefill_gates(stream)?;
    let prefill_kkt_max_abs_error = probe_prefill_kkt(stream)?;
    let output_gate_max_abs_error = probe_output_gate(stream)?;
    let rows = 2usize;
    let query_host = host_bf16(rows * KEY_HEADS * HEAD_DIM, 17, 8.0);
    let key_host = host_bf16(rows * KEY_HEADS * HEAD_DIM, 19, 9.0);
    let value_host = host_bf16(rows * VALUE_HEADS * HEAD_DIM, 23, 11.0);
    let a_host = host_bf16(rows * VALUE_HEADS, 7, 5.0);
    let b_host = host_bf16(rows * VALUE_HEADS, 11, 6.0);
    let a_log = (0..VALUE_HEADS)
        .map(|index| -2.0 + index as f32 / 64.0)
        .collect::<Vec<_>>();
    let dt_bias = (0..VALUE_HEADS)
        .map(|index| -0.5 + index as f32 / 64.0)
        .collect::<Vec<_>>();
    let mut mixed_qkv_host = Vec::with_capacity(rows * CONV_FEATURES);
    for row in 0..rows {
        mixed_qkv_host.extend_from_slice(
            &query_host[row * KEY_HEADS * HEAD_DIM..(row + 1) * KEY_HEADS * HEAD_DIM],
        );
        mixed_qkv_host.extend_from_slice(
            &key_host[row * KEY_HEADS * HEAD_DIM..(row + 1) * KEY_HEADS * HEAD_DIM],
        );
        mixed_qkv_host.extend_from_slice(
            &value_host[row * VALUE_HEADS * HEAD_DIM..(row + 1) * VALUE_HEADS * HEAD_DIM],
        );
    }
    let state_slots = vec![1i32, 0i32];
    let mut state = GdnState::zeros(2, stream)?;
    let mixed_qkv = upload_bf16(&mixed_qkv_host, &[rows, CONV_FEATURES], stream)?;
    let a = upload_bf16(&a_host, &[rows, VALUE_HEADS], stream)?;
    let b = upload_bf16(&b_host, &[rows, VALUE_HEADS], stream)?;
    let a_log_device = upload_f32(&a_log, &[VALUE_HEADS], stream)?;
    let dt_bias_device = upload_f32(&dt_bias, &[VALUE_HEADS], stream)?;
    let state_slots = upload_i32(&state_slots, stream)?;
    let _first = state.decode(
        mixed_qkv.clone(),
        a.clone(),
        b.clone(),
        a_log_device.clone(),
        dt_bias_device.clone(),
        state_slots.clone(),
        rows,
        stream,
    )?;
    let output = state.decode(
        mixed_qkv,
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
    let mut max_abs_error = conv_max_abs_error
        .max(prefill_gate_max_abs_error)
        .max(prefill_kkt_max_abs_error)
        .max(output_gate_max_abs_error);
    for row in 0..rows {
        for value_head in 0..VALUE_HEADS {
            let key_head = value_head / 2;
            let q = &query_host[(row * KEY_HEADS + key_head) * HEAD_DIM
                ..(row * KEY_HEADS + key_head + 1) * HEAD_DIM];
            let k = &key_host[(row * KEY_HEADS + key_head) * HEAD_DIM
                ..(row * KEY_HEADS + key_head + 1) * HEAD_DIM];
            let q_norm = (q.iter().map(|x| x.to_f32().powi(2)).sum::<f32>() + 1.0e-6).sqrt();
            let k_norm = (k.iter().map(|x| x.to_f32().powi(2)).sum::<f32>() + 1.0e-6).sqrt();
            let gate = a_host[row * VALUE_HEADS + value_head].to_f32() + dt_bias[value_head];
            let softplus = if gate <= 20.0 {
                (1.0 + gate.exp()).ln()
            } else {
                gate
            };
            let decay = (-a_log[value_head].exp() * softplus).exp();
            let beta = bf16::from_f32(
                1.0 / (1.0 + (-b_host[row * VALUE_HEADS + value_head].to_f32()).exp()),
            )
            .to_f32();
            let v = &value_host[(row * VALUE_HEADS + value_head) * HEAD_DIM
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

fn probe_prefill_kkt(stream: &Arc<Stream>) -> Result<f32, ModelError> {
    const EPSILON: f32 = 1.0e-6;
    let offsets = [0u32, 17, 38];
    let rows = *offsets.last().unwrap() as usize;
    let mixed_qkv_host = host_bf16(rows * CONV_FEATURES, 31, 19.0);
    let a_host = host_bf16(rows * VALUE_HEADS, 23, 13.0);
    let b_host = host_bf16(rows * VALUE_HEADS, 29, 17.0);
    let a_log = (0..VALUE_HEADS)
        .map(|index| -2.0 + index as f32 / 64.0)
        .collect::<Vec<_>>();
    let dt_bias = (0..VALUE_HEADS)
        .map(|index| -0.5 + index as f32 / 64.0)
        .collect::<Vec<_>>();
    let plan = GdnPrefillPlan::from_offsets(&offsets, stream)?;
    let mixed_qkv = upload_bf16(&mixed_qkv_host, &[rows, CONV_FEATURES], stream)?;
    let gates = prepare_prefill_gates(
        upload_bf16(&a_host, &[rows, VALUE_HEADS], stream)?,
        upload_bf16(&b_host, &[rows, VALUE_HEADS], stream)?,
        upload_f32(&a_log, &[VALUE_HEADS], stream)?,
        upload_f32(&dt_bias, &[VALUE_HEADS], stream)?,
        &plan,
        stream,
    )?;
    let inverse = solve_prefill_kkt(mixed_qkv, &gates, &plan, stream)?;
    let actual: Vec<bf16> = inverse
        .to_host_vec()
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("download GDN prefill inverse: {error:?}")))?;
    let log_decay: Vec<f32> = gates
        .log_decay_cumsum
        .to_host_vec()
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("download GDN KKT log gates: {error:?}")))?;
    let beta: Vec<bf16> =
        gates.beta.to_host_vec().sync_on(stream).map_err(|error| {
            ModelError::Cuda(format!("download GDN KKT update gates: {error:?}"))
        })?;
    let chunks = plan_packed_queries(&offsets, PREFILL_CHUNK as u32)
        .map_err(|error| ModelError::Cuda(format!("plan GDN KKT reference: {error}")))?;

    let mut max_abs_error = 0.0f32;
    for (chunk_index, chunk) in chunks.iter().enumerate() {
        let len = chunk.len() as usize;
        for value_head in 0..VALUE_HEADS {
            let key_head = value_head / 2;
            let mut keys = vec![[0.0f32; HEAD_DIM]; len];
            for (local_row, key) in keys.iter_mut().enumerate() {
                let row = chunk.start() as usize + local_row;
                let start = row * CONV_FEATURES + (KEY_HEADS + key_head) * HEAD_DIM;
                let norm = (mixed_qkv_host[start..start + HEAD_DIM]
                    .iter()
                    .map(|element| element.to_f32().powi(2))
                    .sum::<f32>()
                    + EPSILON)
                    .sqrt();
                for element in 0..HEAD_DIM {
                    key[element] =
                        bf16::from_f32(mixed_qkv_host[start + element].to_f32() / norm).to_f32();
                }
            }
            let mut system = [[0.0f32; PREFILL_CHUNK]; PREFILL_CHUNK];
            for row in 0..len {
                let packed_row = chunk.start() as usize + row;
                for column in 0..row {
                    let packed_column = chunk.start() as usize + column;
                    let dot = keys[row]
                        .iter()
                        .zip(&keys[column])
                        .map(|(lhs, rhs)| lhs * rhs)
                        .sum::<f32>();
                    system[row][column] = beta[packed_row * VALUE_HEADS + value_head].to_f32()
                        * (log_decay[packed_row * VALUE_HEADS + value_head]
                            - log_decay[packed_column * VALUE_HEADS + value_head])
                            .exp()
                        * dot;
                }
            }
            let mut expected = [[0.0f32; PREFILL_CHUNK]; PREFILL_CHUNK];
            for row in 0..len {
                expected[row][row] = 1.0;
                for column in 0..row {
                    let mut value = 0.0f32;
                    for inner in column..row {
                        value += system[row][inner] * expected[inner][column];
                    }
                    expected[row][column] = -value;
                }
            }
            for row in 0..PREFILL_CHUNK {
                for column in 0..PREFILL_CHUNK {
                    let index = (((chunk_index * VALUE_HEADS + value_head) * PREFILL_CHUNK + row)
                        * PREFILL_CHUNK)
                        + column;
                    let expected = bf16::from_f32(expected[row][column]).to_f32();
                    let error = (actual[index].to_f32() - expected).abs();
                    max_abs_error = max_abs_error.max(error);
                    if error > 0.02 || !actual[index].to_f32().is_finite() {
                        return Err(ModelError::Cuda(format!(
                            "GDN prefill inverse mismatch at {index}: {} != {expected}",
                            actual[index].to_f32()
                        )));
                    }
                }
            }
        }
    }
    Ok(max_abs_error)
}

fn probe_prefill_gates(stream: &Arc<Stream>) -> Result<f32, ModelError> {
    let offsets = [0u32, 17, 38];
    let rows = *offsets.last().unwrap() as usize;
    let a_host = host_bf16(rows * VALUE_HEADS, 23, 13.0);
    let b_host = host_bf16(rows * VALUE_HEADS, 29, 17.0);
    let a_log = (0..VALUE_HEADS)
        .map(|index| -2.0 + index as f32 / 64.0)
        .collect::<Vec<_>>();
    let dt_bias = (0..VALUE_HEADS)
        .map(|index| -0.5 + index as f32 / 64.0)
        .collect::<Vec<_>>();
    let plan = GdnPrefillPlan::from_offsets(&offsets, stream)?;
    let gates = prepare_prefill_gates(
        upload_bf16(&a_host, &[rows, VALUE_HEADS], stream)?,
        upload_bf16(&b_host, &[rows, VALUE_HEADS], stream)?,
        upload_f32(&a_log, &[VALUE_HEADS], stream)?,
        upload_f32(&dt_bias, &[VALUE_HEADS], stream)?,
        &plan,
        stream,
    )?;
    let actual_log_decay: Vec<f32> = gates
        .log_decay_cumsum
        .to_host_vec()
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("download GDN prefill log gates: {error:?}")))?;
    let actual_beta: Vec<bf16> = gates.beta.to_host_vec().sync_on(stream).map_err(|error| {
        ModelError::Cuda(format!("download GDN prefill update gates: {error:?}"))
    })?;

    let chunks = plan_packed_queries(&offsets, PREFILL_CHUNK as u32)
        .map_err(|error| ModelError::Cuda(format!("plan GDN gate reference: {error}")))?;
    let mut max_abs_error = 0.0f32;
    for chunk in chunks {
        for head in 0..VALUE_HEADS {
            let mut cumulative = 0.0f32;
            for row in chunk.start() as usize..(chunk.start() + chunk.len()) as usize {
                let index = row * VALUE_HEADS + head;
                let gate = a_host[index].to_f32() + dt_bias[head];
                let softplus = if gate <= 20.0 {
                    (1.0 + gate.exp()).ln()
                } else {
                    gate
                };
                cumulative -= a_log[head].exp() * softplus;
                let expected_beta =
                    bf16::from_f32(1.0 / (1.0 + (-b_host[index].to_f32()).exp())).to_f32();
                let log_error = (actual_log_decay[index] - cumulative).abs();
                let beta_error = (actual_beta[index].to_f32() - expected_beta).abs();
                max_abs_error = max_abs_error.max(log_error).max(beta_error);
                if log_error > 1.0e-5 || beta_error > 1.0e-5 || !actual_log_decay[index].is_finite()
                {
                    return Err(ModelError::Cuda(format!(
                        "GDN prefill gate mismatch at {index}: ({}, {}) != ({cumulative}, {expected_beta})",
                        actual_log_decay[index],
                        actual_beta[index].to_f32(),
                    )));
                }
            }
        }
    }
    Ok(max_abs_error)
}

fn probe_conv(stream: &Arc<Stream>) -> Result<f32, ModelError> {
    let prefill_max_abs_error = probe_conv_prefill(stream)?;
    let rows = 2usize;
    let first_host = host_bf16(rows * CONV_FEATURES, 23, 17.0);
    let second_host = host_bf16(rows * CONV_FEATURES, 29, 19.0);
    let weight_host = host_bf16(CONV_FEATURES * CONV_WIDTH, 13, 23.0);
    let state_slots = upload_i32(&[1i32, 0i32], stream)?;
    let weight = upload_bf16(&weight_host, &[CONV_FEATURES, CONV_WIDTH], stream)?;
    let mut state = GdnState::zeros(rows, stream)?;
    let _first = state.decode_conv(
        upload_bf16(&first_host, &[rows, CONV_FEATURES], stream)?,
        weight.clone(),
        state_slots.clone(),
        rows,
        stream,
    )?;
    let second = state.decode_conv(
        upload_bf16(&second_host, &[rows, CONV_FEATURES], stream)?,
        weight,
        state_slots,
        rows,
        stream,
    )?;
    let actual: Vec<bf16> = second
        .to_host_vec()
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("download GDN convolution probe: {error:?}")))?;
    let mut max_abs_error = 0.0f32;
    for index in 0..actual.len() {
        let feature = index % CONV_FEATURES;
        let first = first_host[index].to_f32();
        let second = second_host[index].to_f32();
        let weight_2 = weight_host[feature * CONV_WIDTH + 2].to_f32();
        let weight_3 = weight_host[feature * CONV_WIDTH + 3].to_f32();
        let convolved = first * weight_2 + second * weight_3;
        let expected = bf16::from_f32(convolved / (1.0 + (-convolved).exp())).to_f32();
        let error = (actual[index].to_f32() - expected).abs();
        max_abs_error = max_abs_error.max(error);
        if error > 0.01 || !actual[index].to_f32().is_finite() {
            return Err(ModelError::Cuda(format!(
                "GDN convolution differential mismatch at {index}: {} != {expected}",
                actual[index].to_f32()
            )));
        }
    }
    Ok(max_abs_error.max(prefill_max_abs_error))
}

fn probe_conv_prefill(stream: &Arc<Stream>) -> Result<f32, ModelError> {
    let requests = 2usize;
    let offsets = [0i32, 3, 8];
    let rows = 8usize;
    let input_host = host_bf16(rows * CONV_FEATURES, 31, 29.0);
    let continuation_host = host_bf16(requests * CONV_FEATURES, 37, 31.0);
    let weight_host = host_bf16(CONV_FEATURES * CONV_WIDTH, 17, 27.0);
    let slots = [1i32, 0i32];
    let mut state = GdnState::zeros(requests, stream)?;
    let weight = upload_bf16(&weight_host, &[CONV_FEATURES, CONV_WIDTH], stream)?;
    let state_slots = upload_i32(&slots, stream)?;
    let output = state.prefill_conv(
        upload_bf16(&input_host, &[rows, CONV_FEATURES], stream)?,
        weight.clone(),
        upload_i32(&offsets, stream)?,
        state_slots.clone(),
        rows,
        requests,
        stream,
    )?;
    let actual: Vec<bf16> = output.to_host_vec().sync_on(stream).map_err(|error| {
        ModelError::Cuda(format!("download GDN convolution prefill probe: {error:?}"))
    })?;
    let continuation = state.decode_conv(
        upload_bf16(&continuation_host, &[requests, CONV_FEATURES], stream)?,
        weight,
        state_slots,
        requests,
        stream,
    )?;
    let continuation_actual: Vec<bf16> =
        continuation
            .to_host_vec()
            .sync_on(stream)
            .map_err(|error| {
                ModelError::Cuda(format!(
                    "download GDN convolution continuation probe: {error:?}"
                ))
            })?;

    let mut max_abs_error = 0.0f32;
    for request in 0..requests {
        for feature in 0..CONV_FEATURES {
            let mut tail = [0.0f32; CONV_STATE_WIDTH];
            for row in offsets[request] as usize..offsets[request + 1] as usize {
                let input = input_host[row * CONV_FEATURES + feature].to_f32();
                let convolved = tail[0] * weight_host[feature * CONV_WIDTH].to_f32()
                    + tail[1] * weight_host[feature * CONV_WIDTH + 1].to_f32()
                    + tail[2] * weight_host[feature * CONV_WIDTH + 2].to_f32()
                    + input * weight_host[feature * CONV_WIDTH + 3].to_f32();
                let expected = bf16::from_f32(convolved / (1.0 + (-convolved).exp())).to_f32();
                let index = row * CONV_FEATURES + feature;
                let error = (actual[index].to_f32() - expected).abs();
                max_abs_error = max_abs_error.max(error);
                if error > 0.01 || !actual[index].to_f32().is_finite() {
                    return Err(ModelError::Cuda(format!(
                        "GDN convolution prefill mismatch at {index}: {} != {expected}",
                        actual[index].to_f32()
                    )));
                }
                tail = [tail[1], tail[2], input];
            }

            let input = continuation_host[request * CONV_FEATURES + feature].to_f32();
            let convolved = tail[0] * weight_host[feature * CONV_WIDTH].to_f32()
                + tail[1] * weight_host[feature * CONV_WIDTH + 1].to_f32()
                + tail[2] * weight_host[feature * CONV_WIDTH + 2].to_f32()
                + input * weight_host[feature * CONV_WIDTH + 3].to_f32();
            let expected = bf16::from_f32(convolved / (1.0 + (-convolved).exp())).to_f32();
            let index = request * CONV_FEATURES + feature;
            let error = (continuation_actual[index].to_f32() - expected).abs();
            max_abs_error = max_abs_error.max(error);
            if error > 0.01 || !continuation_actual[index].to_f32().is_finite() {
                return Err(ModelError::Cuda(format!(
                    "GDN convolution continuation mismatch at {index}: {} != {expected}",
                    continuation_actual[index].to_f32()
                )));
            }
        }
    }
    Ok(max_abs_error)
}

fn probe_output_gate(stream: &Arc<Stream>) -> Result<f32, ModelError> {
    let rows = 2usize;
    let input_host = host_bf16(rows * VALUE_HEADS * HEAD_DIM, 31, 13.0);
    let gate_host = host_bf16(rows * VALUE_HEADS * HEAD_DIM, 37, 17.0);
    let weight_host = host_bf16(HEAD_DIM, 19, 11.0);
    const EPSILON: f32 = 1.0e-6;
    let output = output_gate(
        upload_bf16(&input_host, &[rows, VALUE_HEADS, HEAD_DIM], stream)?,
        upload_bf16(&gate_host, &[rows, VALUE_HEADS, HEAD_DIM], stream)?,
        upload_bf16(&weight_host, &[HEAD_DIM], stream)?,
        EPSILON,
        rows,
        stream,
    )?;
    let actual: Vec<bf16> = output
        .to_host_vec()
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("download GDN output gate probe: {error:?}")))?;
    let mut max_abs_error = 0.0f32;
    for row_head in 0..rows * VALUE_HEADS {
        let start = row_head * HEAD_DIM;
        let square_mean = input_host[start..start + HEAD_DIM]
            .iter()
            .map(|element| element.to_f32().powi(2))
            .sum::<f32>()
            / HEAD_DIM as f32;
        let inverse = 1.0 / (square_mean + EPSILON).sqrt();
        for element in 0..HEAD_DIM {
            let index = start + element;
            let input = input_host[index].to_f32();
            let gate = gate_host[index].to_f32();
            let expected = bf16::from_f32(
                input * inverse * weight_host[element].to_f32() * gate / (1.0 + (-gate).exp()),
            )
            .to_f32();
            let error = (actual[index].to_f32() - expected).abs();
            max_abs_error = max_abs_error.max(error);
            if error > 0.01 || !actual[index].to_f32().is_finite() {
                return Err(ModelError::Cuda(format!(
                    "GDN output gate differential mismatch at {index}: {} != {expected}",
                    actual[index].to_f32()
                )));
            }
        }
    }
    Ok(max_abs_error)
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
