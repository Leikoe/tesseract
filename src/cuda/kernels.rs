//! Model-neutral BF16 transformer primitives implemented with cuTile.

#![allow(clippy::too_many_arguments)]

#[allow(clippy::too_many_arguments)]
#[cutile::module]
mod tile {
    use cutile::core::*;

    #[cutile::entry()]
    fn embedding_bf16<const D: i32, const BLOCK: i32>(
        token_ids: &Tensor<u32, { [-1] }>,
        table: &Tensor<bf16, { [-1, D] }>,
        out: &mut Tensor<bf16, { [1, BLOCK] }>,
    ) {
        let pid = get_tile_block_id();
        let row = pid.0;
        let block = pid.1;
        let token = token_ids.partition(const_shape![1]).load([row]);
        let token: Tile<i32, { [1] }> = bitcast(token);
        let token: i32 = tile_to_scalar(token.reshape(const_shape![]));
        let table = table.partition(const_shape![1, BLOCK]);
        out.store(table.load([token, block]));
    }

    #[cutile::entry()]
    unsafe fn rms_norm_bf16<const N: i32, const BLOCK: i32>(
        input: &Tensor<bf16, { [-1, N] }>,
        weight: &Tensor<bf16, { [N] }>,
        out: &mut Tensor<bf16, { [1, N] }>,
        epsilon: f32,
    ) {
        let shape = const_shape![1, BLOCK];
        let row = get_tile_block_id().0;
        let input = input.partition(shape);
        let mut squares: Tile<f32, { [1, BLOCK] }> = constant(0.0f32, shape);
        for block in 0i32..(N / BLOCK) {
            let values: Tile<f32, { [1, BLOCK] }> = convert_tile(input.load([row, block]));
            squares = squares + values * values;
        }
        let sum: Tile<f32, { [1] }> = reduce_sum(squares, 1i32);
        let sum: Tile<f32, { [] }> = sum.reshape(const_shape![]);
        let sum: f32 = tile_to_scalar(sum);
        let n: f32 = convert_scalar(N);
        let inverse: Tile<f32, { [] }> = rsqrt(scalar_to_tile(sum / n + epsilon), ftz::Disabled);
        let inverse: f32 = tile_to_scalar(inverse);
        let inverse: Tile<f32, { [1, BLOCK] }> = inverse.broadcast(shape);
        let weight = weight.partition(const_shape![BLOCK]);
        let mut out = unsafe { out.partition_mut(shape) };
        for block in 0i32..(N / BLOCK) {
            let values: Tile<f32, { [1, BLOCK] }> = convert_tile(input.load([row, block]));
            let scale: Tile<f32, { [1, BLOCK] }> =
                convert_tile(weight.load([block]).reshape(shape));
            let normalized: Tile<bf16, { [1, BLOCK] }> = convert_tile(values * inverse * scale);
            unsafe { out.store(normalized, [0i32, block]) };
        }
    }

    #[cutile::entry()]
    unsafe fn add_rms_norm_bf16<const N: i32, const BLOCK: i32>(
        residual: &Tensor<bf16, { [-1, N] }>,
        update: &Tensor<bf16, { [-1, N] }>,
        weight: &Tensor<bf16, { [N] }>,
        normalized: &mut Tensor<bf16, { [1, N] }>,
        combined_out: &mut Tensor<bf16, { [1, N] }>,
        epsilon: f32,
    ) {
        let shape = const_shape![1, BLOCK];
        let row = get_tile_block_id().0;
        let residual = residual.partition(shape);
        let update = update.partition(shape);
        let mut squares: Tile<f32, { [1, BLOCK] }> = constant(0.0f32, shape);
        for block in 0i32..(N / BLOCK) {
            let residual: Tile<f32, { [1, BLOCK] }> = convert_tile(residual.load([row, block]));
            let update: Tile<f32, { [1, BLOCK] }> = convert_tile(update.load([row, block]));
            let combined = residual + update;
            squares = squares + combined * combined;
        }
        let sum: Tile<f32, { [1] }> = reduce_sum(squares, 1i32);
        let sum: Tile<f32, { [] }> = sum.reshape(const_shape![]);
        let sum: f32 = tile_to_scalar(sum);
        let n: f32 = convert_scalar(N);
        let inverse: Tile<f32, { [] }> = rsqrt(scalar_to_tile(sum / n + epsilon), ftz::Disabled);
        let inverse: f32 = tile_to_scalar(inverse);
        let inverse: Tile<f32, { [1, BLOCK] }> = inverse.broadcast(shape);
        let weight = weight.partition(const_shape![BLOCK]);
        let mut normalized = unsafe { normalized.partition_mut(shape) };
        let mut combined_out = unsafe { combined_out.partition_mut(shape) };
        for block in 0i32..(N / BLOCK) {
            let residual: Tile<f32, { [1, BLOCK] }> = convert_tile(residual.load([row, block]));
            let update: Tile<f32, { [1, BLOCK] }> = convert_tile(update.load([row, block]));
            let combined = residual + update;
            let scale: Tile<f32, { [1, BLOCK] }> =
                convert_tile(weight.load([block]).reshape(shape));
            let combined_bf16: Tile<bf16, { [1, BLOCK] }> = convert_tile(combined);
            let normalized_bf16: Tile<bf16, { [1, BLOCK] }> =
                convert_tile(combined * inverse * scale);
            unsafe {
                combined_out.store(combined_bf16, [0i32, block]);
                normalized.store(normalized_bf16, [0i32, block]);
            }
        }
    }

    #[cutile::entry()]
    fn silu_mul_bf16<const BLOCK: i32>(
        gate: &Tensor<bf16, { [-1, -1] }>,
        up: &Tensor<bf16, { [-1, -1] }>,
        out: &mut Tensor<bf16, { [1, BLOCK] }>,
    ) {
        let pid = get_tile_block_id();
        let gate: Tile<f32, { [1, BLOCK] }> =
            convert_tile(gate.partition(const_shape![1, BLOCK]).load([pid.0, pid.1]));
        let up: Tile<f32, { [1, BLOCK] }> =
            convert_tile(up.partition(const_shape![1, BLOCK]).load([pid.0, pid.1]));
        let one: Tile<f32, { [1, BLOCK] }> = constant(1.0f32, const_shape![1, BLOCK]);
        let zero: Tile<f32, { [1, BLOCK] }> = constant(0.0f32, const_shape![1, BLOCK]);
        let activated: Tile<f32, { [1, BLOCK] }> =
            gate * true_div(one, one + exp(zero - gate)) * up;
        let activated: Tile<bf16, { [1, BLOCK] }> = convert_tile(activated);
        out.store(activated);
    }

    #[cutile::entry()]
    fn rope_q_bf16<const D: i32, const HALF: i32>(
        input: &Tensor<bf16, { [-1, -1, D] }>,
        positions: &Tensor<u32, { [-1] }>,
        cos: &Tensor<f32, { [-1, HALF] }>,
        sin: &Tensor<f32, { [-1, HALF] }>,
        out: &mut Tensor<bf16, { [1, 1, D] }>,
    ) {
        let pid = get_tile_block_id();
        let row = pid.0;
        let head = pid.1;
        let position = positions.partition(const_shape![1]).load([row]);
        let position: Tile<i32, { [1] }> = bitcast(position);
        let position: i32 = tile_to_scalar(position.reshape(const_shape![]));
        let input = input.partition(const_shape![1, 1, HALF]);
        let lo: Tile<f32, { [1, HALF] }> =
            convert_tile(input.load([row, head, 0i32]).reshape(const_shape![1, HALF]));
        let hi: Tile<f32, { [1, HALF] }> =
            convert_tile(input.load([row, head, 1i32]).reshape(const_shape![1, HALF]));
        let cos: Tile<f32, { [1, HALF] }> =
            cos.partition(const_shape![1, HALF]).load([position, 0i32]);
        let sin: Tile<f32, { [1, HALF] }> =
            sin.partition(const_shape![1, HALF]).load([position, 0i32]);
        let mut out = unsafe { out.partition_mut(const_shape![1, 1, HALF]) };
        let rotated_lo: Tile<bf16, { [1, 1, HALF] }> =
            convert_tile((lo * cos - hi * sin).reshape(const_shape![1, 1, HALF]));
        let rotated_hi: Tile<bf16, { [1, 1, HALF] }> =
            convert_tile((hi * cos + lo * sin).reshape(const_shape![1, 1, HALF]));
        unsafe {
            out.store(rotated_lo, [0i32, 0i32, 0i32]);
            out.store(rotated_hi, [0i32, 0i32, 1i32]);
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[cutile::entry()]
    unsafe fn rope_kv_write_bf16<const D: i32, const HALF: i32, const KV_HEADS: i32>(
        key: &Tensor<bf16, { [-1, KV_HEADS, D] }>,
        value: &Tensor<bf16, { [-1, KV_HEADS, D] }>,
        positions: &Tensor<u32, { [-1] }>,
        slots: &Tensor<u32, { [-1] }>,
        cos: &Tensor<f32, { [-1, HALF] }>,
        sin: &Tensor<f32, { [-1, HALF] }>,
        key_cache_ptr: *mut bf16,
        value_cache_ptr: *mut bf16,
        key_out: &mut Tensor<bf16, { [1, 1, D] }>,
    ) {
        let pid = get_tile_block_id();
        let row = pid.0;
        let head = pid.1;
        let position = positions.partition(const_shape![1]).load([row]);
        let position: Tile<i32, { [1] }> = bitcast(position);
        let position: i32 = tile_to_scalar(position.reshape(const_shape![]));
        let slot = slots.partition(const_shape![1]).load([row]);
        let slot: Tile<i32, { [1] }> = bitcast(slot);
        let slot: i32 = tile_to_scalar(slot.reshape(const_shape![]));

        let key = key.partition(const_shape![1, 1, HALF]);
        let lo: Tile<f32, { [1, HALF] }> =
            convert_tile(key.load([row, head, 0i32]).reshape(const_shape![1, HALF]));
        let hi: Tile<f32, { [1, HALF] }> =
            convert_tile(key.load([row, head, 1i32]).reshape(const_shape![1, HALF]));
        let cos: Tile<f32, { [1, HALF] }> =
            cos.partition(const_shape![1, HALF]).load([position, 0i32]);
        let sin: Tile<f32, { [1, HALF] }> =
            sin.partition(const_shape![1, HALF]).load([position, 0i32]);
        let key_lo: Tile<bf16, { [1, HALF] }> = convert_tile(lo * cos - hi * sin);
        let key_hi: Tile<bf16, { [1, HALF] }> = convert_tile(hi * cos + lo * sin);
        let mut key_out = unsafe { key_out.partition_mut(const_shape![1, 1, HALF]) };
        unsafe {
            key_out.store(key_lo.reshape(const_shape![1, 1, HALF]), [0i32, 0i32, 0i32]);
            key_out.store(key_hi.reshape(const_shape![1, 1, HALF]), [0i32, 0i32, 1i32]);
        }

        let base: i32 = (slot * KV_HEADS + head) * D;
        let half_offsets: Tile<i32, { [HALF] }> = iota(const_shape![HALF]);
        let half_offsets: Tile<i32, { [1, HALF] }> = half_offsets.reshape(const_shape![1, HALF]);
        let key_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(key_cache_ptr);
        let key_base: PointerTile<*mut bf16, { [1, 1] }> = key_base.reshape(const_shape![1, 1]);
        let key_base: PointerTile<*mut bf16, { [1, HALF] }> =
            key_base.broadcast(const_shape![1, HALF]);
        let base_offsets: Tile<i32, { [1, HALF] }> =
            base.broadcast(const_shape![1, HALF]) + half_offsets;
        let high_half: Tile<i32, { [1, HALF] }> = HALF.broadcast(const_shape![1, HALF]);
        let key_lo_ptr: PointerTile<*mut bf16, { [1, HALF] }> = key_base.offset_tile(base_offsets);
        let _key_lo_store: Token = store_ptr_tko(
            key_lo_ptr,
            key_lo,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<0>,
        );
        let key_hi_ptr: PointerTile<*mut bf16, { [1, HALF] }> =
            key_base.offset_tile(base_offsets + high_half);
        let _key_hi_store: Token = store_ptr_tko(
            key_hi_ptr,
            key_hi,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<0>,
        );

        let value: Tile<bf16, { [1, 1, D] }> = value
            .partition(const_shape![1, 1, D])
            .load([row, head, 0i32]);
        let value_offsets: Tile<i32, { [D] }> = iota(const_shape![D]);
        let value_offsets: Tile<i32, { [1, 1, D] }> = value_offsets.reshape(const_shape![1, 1, D]);
        let value_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(value_cache_ptr);
        let value_base: PointerTile<*mut bf16, { [1, 1, 1] }> =
            value_base.reshape(const_shape![1, 1, 1]);
        let value_base: PointerTile<*mut bf16, { [1, 1, D] }> =
            value_base.broadcast(const_shape![1, 1, D]);
        let value_base_offset: Tile<i32, { [1, 1, D] }> =
            base.broadcast(const_shape![1, 1, D]) + value_offsets;
        let value_ptr: PointerTile<*mut bf16, { [1, 1, D] }> =
            value_base.offset_tile(value_base_offset);
        let _value_store: Token = store_ptr_tko(
            value_ptr,
            value,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<0>,
        );
    }

    #[cutile::entry()]
    fn gather_flat_kv_bf16<const D: i32>(
        slots: &Tensor<u32, { [-1] }>,
        cache: &Tensor<bf16, { [-1, -1, D] }>,
        out: &mut Tensor<bf16, { [1, 1, D] }>,
    ) {
        let pid = get_tile_block_id();
        let head = pid.0;
        let position = pid.1;
        let slot = slots.partition(const_shape![1]).load([position]);
        let slot: Tile<i32, { [1] }> = bitcast(slot);
        let slot: i32 = tile_to_scalar(slot.reshape(const_shape![]));
        let cache = cache.partition(const_shape![1, 1, D]);
        out.store(cache.load([slot, head, 0i32]));
    }

    #[cutile::entry()]
    fn gather_row_bf16<const BLOCK: i32>(
        input: &Tensor<bf16, { [-1, -1] }>,
        out: &mut Tensor<bf16, { [BLOCK] }>,
        row: i32,
    ) {
        let block = get_tile_block_id().0;
        let input = input.partition(const_shape![1, BLOCK]);
        out.store(input.load([row, block]).reshape(const_shape![BLOCK]));
    }

    #[allow(clippy::too_many_arguments)]
    #[cutile::entry()]
    unsafe fn causal_attention_bf16<const BM: i32, const BN: i32, const D: i32>(
        query: &Tensor<bf16, { [-1, -1, D] }>,
        key: &Tensor<bf16, { [-1, -1, D] }>,
        value: &Tensor<bf16, { [-1, -1, D] }>,
        metadata: &Tensor<i32, { [2] }>,
        out: &mut Tensor<bf16, { [BM, 1, D] }>,
        scale: f32,
        group_size: i32,
    ) {
        let pid = get_tile_block_id();
        let query_block = pid.0;
        let query_head = pid.1;
        let kv_head = query_head / group_size;
        let metadata = metadata.partition(const_shape![1]);
        let context_len: i32 = tile_to_scalar(metadata.load([0i32]).reshape(const_shape![]));
        let query_start: i32 = tile_to_scalar(metadata.load([1i32]).reshape(const_shape![]));
        let query = query.partition(const_shape![BM, 1, D]);
        let query: Tile<f32, { [BM, D] }> = convert_tile(
            query
                .load([query_block, query_head, 0i32])
                .reshape(const_shape![BM, D]),
        );
        let key = key.partition(const_shape![1, BN, D]);
        let value = value.partition(const_shape![1, BN, D]);
        let mut row_max: Tile<f32, { [BM, 1] }> = constant(-1.0e30f32, const_shape![BM, 1]);
        let mut row_sum: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]);
        let mut accumulator: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);
        let lane: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let lane: Tile<i32, { [BM, BN] }> = lane
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![BM, BN]);
        let query_lane: Tile<i32, { [BM] }> = iota(const_shape![BM]);
        let query_position: i32 = query_start + query_block * BM;
        let query_position: Tile<i32, { [BM] }> =
            query_position.broadcast(const_shape![BM]) + query_lane;
        let query_position: Tile<i32, { [BM, BN] }> = query_position
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, BN]);
        for block in 0i32..((context_len + BN - 1i32) / BN) {
            let key_tile = key
                .load([kv_head, block, 0i32])
                .reshape(const_shape![BN, D]);
            let key_tile: Tile<f32, { [BN, D] }> = convert_tile(key_tile);
            let key_tile: Tile<f32, { [D, BN] }> = key_tile.transpose();
            let scores_zero: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            let scores: Tile<f32, { [BM, BN] }> = mma(query, key_tile, scores_zero);
            let key_position: Tile<i32, { [BM, BN] }> =
                (block * BN).broadcast(const_shape![BM, BN]) + lane;
            let valid: Tile<bool, { [BM, BN] }> =
                lt_tile(key_position, context_len.broadcast(const_shape![BM, BN]))
                    & ge_tile(query_position, key_position);
            let scale: Tile<f32, { [BM, BN] }> = scale.broadcast(const_shape![BM, BN]);
            let negative_infinity: Tile<f32, { [BM, BN] }> =
                constant(-1.0e30f32, const_shape![BM, BN]);
            let scores: Tile<f32, { [BM, BN] }> = select(valid, scores * scale, negative_infinity);
            let block_max: Tile<f32, { [BM] }> = reduce_max(scores, 1i32);
            let block_max: Tile<f32, { [BM, 1] }> = block_max.reshape(const_shape![BM, 1]);
            let next_max: Tile<f32, { [BM, 1] }> = max_tile(row_max, block_max);
            let probabilities: Tile<f32, { [BM, BN] }> =
                exp(scores - next_max.broadcast(const_shape![BM, BN]));
            let block_sum: Tile<f32, { [BM] }> = reduce_sum(probabilities, 1i32);
            let block_sum: Tile<f32, { [BM, 1] }> = block_sum.reshape(const_shape![BM, 1]);
            let correction: Tile<f32, { [BM, 1] }> = exp(row_max - next_max);
            row_sum = row_sum * correction + block_sum;
            accumulator = accumulator * correction.broadcast(const_shape![BM, D]);
            let value_tile = value
                .load([kv_head, block, 0i32])
                .reshape(const_shape![BN, D]);
            let probabilities: Tile<bf16, { [BM, BN] }> = convert_tile(probabilities);
            accumulator = mma(probabilities, value_tile, accumulator);
            row_max = next_max;
        }
        let epsilon: Tile<f32, { [BM, 1] }> = constant(1.0e-8f32, const_shape![BM, 1]);
        let denominator: Tile<f32, { [BM, D] }> =
            max_tile(row_sum, epsilon).broadcast(const_shape![BM, D]);
        let output: Tile<bf16, { [BM, 1, D] }> =
            convert_tile(true_div(accumulator, denominator).reshape(const_shape![BM, 1, D]));
        out.store(output);
    }
}

#[allow(unused_imports)]
pub(crate) use tile::{
    add_rms_norm_bf16, causal_attention_bf16, embedding_bf16, gather_flat_kv_bf16, gather_row_bf16,
    rms_norm_bf16, rope_kv_write_bf16, rope_q_bf16, silu_mul_bf16,
};
