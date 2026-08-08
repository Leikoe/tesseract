//! Model-neutral BF16 transformer primitives implemented with cuTile.

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
        capacity: i32,
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

        let cache_shape = Shape::<{ [-1, KV_HEADS, D] }> { dims: &[capacity] };
        let cache_strides = Array::<{ [-1, D, 1] }> {
            dims: &[KV_HEADS * D],
        };
        let token = new_token_unordered();
        let mut key_cache = unsafe {
            make_tensor_view(
                pointer_to_tile(key_cache_ptr),
                cache_shape,
                cache_strides,
                token,
            )
        };
        let mut value_cache = unsafe {
            make_tensor_view(
                pointer_to_tile(value_cache_ptr),
                cache_shape,
                cache_strides,
                token,
            )
        };
        let mut key_cache = unsafe { key_cache.partition_mut(const_shape![1, 1, HALF]) };
        let mut value_cache = unsafe { value_cache.partition_mut(const_shape![1, 1, D]) };
        let value = value
            .partition(const_shape![1, 1, D])
            .load([row, head, 0i32]);
        unsafe {
            key_cache.store(key_lo.reshape(const_shape![1, 1, HALF]), [slot, head, 0i32]);
            key_cache.store(key_hi.reshape(const_shape![1, 1, HALF]), [slot, head, 1i32]);
            value_cache.store(value, [slot, head, 0i32]);
        }
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
    unsafe fn causal_attention_bf16<const BM: i32, const BN: i32, const D: i32>(
        query: &Tensor<bf16, { [-1, -1, D] }>,
        key: &Tensor<bf16, { [-1, -1, D] }>,
        value: &Tensor<bf16, { [-1, -1, D] }>,
        out: &mut Tensor<bf16, { [BM, 1, D] }>,
        scale: f32,
        group_size: i32,
        context_len: i32,
        query_start: i32,
    ) {
        let pid = get_tile_block_id();
        let query_block = pid.0;
        let query_head = pid.1;
        let kv_head = query_head / group_size;
        let query = query.partition(const_shape![BM, 1, D]);
        let query: Tile<f32, { [BM, D] }> = convert_tile(
            query
                .load([query_block, query_head, 0i32])
                .reshape(const_shape![BM, D]),
        );
        let key = key.partition(const_shape![1, BN, D]);
        let value = value.partition(const_shape![1, BN, D]);
        let mut row_max = constant(-1.0e30f32, const_shape![BM, 1]);
        let mut row_sum = constant(0.0f32, const_shape![BM, 1]);
        let mut accumulator = constant(0.0f32, const_shape![BM, D]);
        let lane = iota(const_shape![BN]);
        let lane = lane
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![BM, BN]);
        let query_lane = iota(const_shape![BM]);
        let query_position = query_start + query_block * BM;
        let query_position = query_position.broadcast(const_shape![BM]) + query_lane;
        let query_position = query_position
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, BN]);
        for block in 0i32..((context_len + BN - 1i32) / BN) {
            let key_tile = key
                .load([kv_head, block, 0i32])
                .reshape(const_shape![BN, D]);
            let key_tile: Tile<f32, { [BN, D] }> = convert_tile(key_tile);
            let key_tile = key_tile.transpose();
            let scores = mma(query, key_tile, constant(0.0f32, const_shape![BM, BN]));
            let key_position = (block * BN).broadcast(const_shape![BM, BN]) + lane;
            let valid = lt_tile(key_position, context_len.broadcast(const_shape![BM, BN]))
                & ge_tile(query_position, key_position);
            let scale: Tile<f32, { [BM, BN] }> = scale.broadcast(const_shape![BM, BN]);
            let scores = select(
                valid,
                scores * scale,
                constant(-1.0e30f32, const_shape![BM, BN]),
            );
            let block_max: Tile<f32, { [BM] }> = reduce_max(scores, 1i32);
            let block_max: Tile<f32, { [BM, 1] }> = block_max.reshape(const_shape![BM, 1]);
            let next_max = max_tile(row_max, block_max);
            let probabilities = exp(scores - next_max.broadcast(const_shape![BM, BN]));
            let block_sum: Tile<f32, { [BM] }> = reduce_sum(probabilities, 1i32);
            let block_sum: Tile<f32, { [BM, 1] }> = block_sum.reshape(const_shape![BM, 1]);
            let correction = exp(row_max - next_max);
            row_sum = row_sum * correction + block_sum;
            accumulator = accumulator * correction.broadcast(const_shape![BM, D]);
            let value_tile = value
                .load([kv_head, block, 0i32])
                .reshape(const_shape![BN, D]);
            let probabilities: Tile<bf16, { [BM, BN] }> = convert_tile(probabilities);
            accumulator = mma(probabilities, value_tile, accumulator);
            row_max = next_max;
        }
        let epsilon = constant(1.0e-8f32, const_shape![BM, 1]);
        let denominator = max_tile(row_sum, epsilon).broadcast(const_shape![BM, D]);
        let output: Tile<bf16, { [BM, 1, D] }> =
            convert_tile(true_div(accumulator, denominator).reshape(const_shape![BM, 1, D]));
        out.store(output);
    }
}

#[allow(unused_imports)]
pub(crate) use tile::{
    add_rms_norm_bf16, causal_attention_bf16, embedding_bf16, gather_flat_kv_bf16, rms_norm_bf16,
    rope_kv_write_bf16, rope_q_bf16, silu_mul_bf16,
};
