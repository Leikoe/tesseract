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
    unsafe fn gemma_rms_norm_bf16<const N: i32, const BLOCK: i32>(
        input: &Tensor<bf16, { [-1, N] }>,
        weight_delta: &Tensor<bf16, { [N] }>,
        out: &mut Tensor<bf16, { [1, N] }>,
        epsilon: f32,
    ) {
        let shape = const_shape![1, BLOCK];
        let row = get_tile_block_id().0;
        let input = input.partition(shape);
        const ZERO: f32 = 0.0;
        let mut squares: Tile<f32, { [1, BLOCK] }> = broadcast_scalar(ZERO, shape);
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
        let weight_delta = weight_delta.partition(const_shape![BLOCK]);
        let mut out = unsafe { out.partition_mut(shape) };
        const ONE: f32 = 1.0;
        let one: Tile<f32, { [1, BLOCK] }> = broadcast_scalar(ONE, shape);
        for block in 0i32..(N / BLOCK) {
            let values: Tile<f32, { [1, BLOCK] }> = convert_tile(input.load([row, block]));
            let delta: Tile<f32, { [1, BLOCK] }> =
                convert_tile(weight_delta.load([block]).reshape(shape));
            let normalized: Tile<bf16, { [1, BLOCK] }> =
                ftof(values * inverse * (one + delta), rounding::NearestEven);
            unsafe { out.store(normalized, [0i32, block]) };
        }
    }

    #[cutile::entry()]
    unsafe fn gemma_add_rms_norm_bf16<const N: i32, const BLOCK: i32>(
        residual: &Tensor<bf16, { [-1, N] }>,
        update: &Tensor<bf16, { [-1, N] }>,
        weight_delta: &Tensor<bf16, { [N] }>,
        normalized: &mut Tensor<bf16, { [1, N] }>,
        combined_out: &mut Tensor<bf16, { [1, N] }>,
        epsilon: f32,
    ) {
        let shape = const_shape![1, BLOCK];
        let row = get_tile_block_id().0;
        let residual = residual.partition(shape);
        let update = update.partition(shape);
        const ZERO: f32 = 0.0;
        let mut squares: Tile<f32, { [1, BLOCK] }> = broadcast_scalar(ZERO, shape);
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
        let weight_delta = weight_delta.partition(const_shape![BLOCK]);
        let mut normalized = unsafe { normalized.partition_mut(shape) };
        let mut combined_out = unsafe { combined_out.partition_mut(shape) };
        const ONE: f32 = 1.0;
        let one: Tile<f32, { [1, BLOCK] }> = broadcast_scalar(ONE, shape);
        for block in 0i32..(N / BLOCK) {
            let residual: Tile<f32, { [1, BLOCK] }> = convert_tile(residual.load([row, block]));
            let update: Tile<f32, { [1, BLOCK] }> = convert_tile(update.load([row, block]));
            let combined = residual + update;
            let delta: Tile<f32, { [1, BLOCK] }> =
                convert_tile(weight_delta.load([block]).reshape(shape));
            let combined_bf16: Tile<bf16, { [1, BLOCK] }> = ftof(combined, rounding::NearestEven);
            let normalized_bf16: Tile<bf16, { [1, BLOCK] }> =
                ftof(combined * inverse * (one + delta), rounding::NearestEven);
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

    /// Ragged attention over the flat physical KV cache.
    ///
    /// Query rows are flattened across requests. `request_indices[row]`
    /// selects one row of the padded logical-to-physical slot table, while
    /// `context_lengths[row]` applies the causal boundary for that individual
    /// query token. K/V are loaded directly through the physical slot map; no
    /// per-layer gathered cache is materialized.
    #[allow(clippy::too_many_arguments)]
    #[cutile::entry()]
    unsafe fn ragged_attention_bf16<const BN: i32, const D: i32, const KV_HEADS: i32>(
        query: &Tensor<bf16, { [-1, -1, D] }>,
        request_indices: &Tensor<u32, { [-1] }>,
        context_slots: &Tensor<u32, { [-1, -1] }>,
        context_lengths: &Tensor<i32, { [-1] }>,
        key_cache_ptr: *mut bf16,
        value_cache_ptr: *mut bf16,
        out: &mut Tensor<bf16, { [1, 1, D] }>,
        scale: f32,
        group_size: i32,
    ) {
        let pid = get_tile_block_id();
        let row = pid.0;
        let query_head = pid.1;
        let kv_head = query_head / group_size;
        let request_index: Tile<u32, { [1] }> =
            request_indices.partition(const_shape![1]).load([row]);
        let request_index: Tile<i32, { [1] }> = bitcast(request_index);
        let request_index: i32 = tile_to_scalar(request_index.reshape(const_shape![]));
        let context_len: i32 = tile_to_scalar(
            context_lengths
                .partition(const_shape![1])
                .load([row])
                .reshape(const_shape![]),
        );
        let query: Tile<f32, { [1, D] }> = convert_tile(
            query
                .partition(const_shape![1, 1, D])
                .load([row, query_head, 0i32])
                .reshape(const_shape![1, D]),
        );
        let context_slots = context_slots.partition(const_shape![1, BN]);
        let mut row_max: Tile<f32, { [1, 1] }> = constant(-1.0e30f32, const_shape![1, 1]);
        let mut row_sum: Tile<f32, { [1, 1] }> = constant(0.0f32, const_shape![1, 1]);
        let mut accumulator: Tile<f32, { [1, D] }> = constant(0.0f32, const_shape![1, D]);
        let lane: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let lane: Tile<i32, { [1, BN] }> = lane.reshape(const_shape![1, BN]);
        let element: Tile<i32, { [D] }> = iota(const_shape![D]);
        let element: Tile<i32, { [1, 1, D] }> = element.reshape(const_shape![1, 1, D]);
        let key_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(key_cache_ptr);
        let key_base: PointerTile<*mut bf16, { [1, 1, 1] }> =
            key_base.reshape(const_shape![1, 1, 1]);
        let key_base: PointerTile<*mut bf16, { [1, BN, D] }> =
            key_base.broadcast(const_shape![1, BN, D]);
        let value_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(value_cache_ptr);
        let value_base: PointerTile<*mut bf16, { [1, 1, 1] }> =
            value_base.reshape(const_shape![1, 1, 1]);
        let value_base: PointerTile<*mut bf16, { [1, BN, D] }> =
            value_base.broadcast(const_shape![1, BN, D]);

        for block in 0i32..((context_len + BN - 1i32) / BN) {
            let slots: Tile<u32, { [1, BN] }> = context_slots.load([request_index, block]);
            let slots: Tile<i32, { [1, BN] }> = bitcast(slots);
            let shape = const_shape![1, BN];
            let cache_row: Tile<i32, { [1, BN] }> =
                (slots * KV_HEADS.broadcast(shape) + kv_head.broadcast(shape)) * D.broadcast(shape);
            let offsets: Tile<i32, { [1, BN, D] }> = cache_row
                .reshape(const_shape![1, BN, 1])
                .broadcast(const_shape![1, BN, D])
                + element.broadcast(const_shape![1, BN, D]);
            let key_ptrs: PointerTile<*mut bf16, { [1, BN, D] }> = key_base.offset_tile(offsets);
            let (key_tile, _): (Tile<bf16, { [1, BN, D] }>, Token) = load_ptr_tko(
                key_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let key_tile: Tile<f32, { [BN, D] }> =
                convert_tile(key_tile.reshape(const_shape![BN, D]));
            let key_tile: Tile<f32, { [D, BN] }> = key_tile.transpose();
            let scores_zero: Tile<f32, { [1, BN] }> = constant(0.0f32, const_shape![1, BN]);
            let scores: Tile<f32, { [1, BN] }> = mma(query, key_tile, scores_zero);
            let positions: Tile<i32, { [1, BN] }> =
                (block * BN).broadcast(const_shape![1, BN]) + lane;
            let valid: Tile<bool, { [1, BN] }> =
                lt_tile(positions, context_len.broadcast(const_shape![1, BN]));
            let scale: Tile<f32, { [1, BN] }> = scale.broadcast(const_shape![1, BN]);
            let negative_infinity: Tile<f32, { [1, BN] }> =
                constant(-1.0e30f32, const_shape![1, BN]);
            let scores: Tile<f32, { [1, BN] }> = select(valid, scores * scale, negative_infinity);
            let block_max: Tile<f32, { [1] }> = reduce_max(scores, 1i32);
            let block_max: Tile<f32, { [1, 1] }> = block_max.reshape(const_shape![1, 1]);
            let next_max: Tile<f32, { [1, 1] }> = max_tile(row_max, block_max);
            let probabilities: Tile<f32, { [1, BN] }> =
                exp(scores - next_max.broadcast(const_shape![1, BN]));
            let block_sum: Tile<f32, { [1] }> = reduce_sum(probabilities, 1i32);
            let block_sum: Tile<f32, { [1, 1] }> = block_sum.reshape(const_shape![1, 1]);
            let correction: Tile<f32, { [1, 1] }> = exp(row_max - next_max);
            row_sum = row_sum * correction + block_sum;
            accumulator = accumulator * correction.broadcast(const_shape![1, D]);
            let value_ptrs: PointerTile<*mut bf16, { [1, BN, D] }> =
                value_base.offset_tile(offsets);
            let (value_tile, _): (Tile<bf16, { [1, BN, D] }>, Token) = load_ptr_tko(
                value_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let value_tile: Tile<bf16, { [BN, D] }> = value_tile.reshape(const_shape![BN, D]);
            let probabilities: Tile<bf16, { [1, BN] }> = convert_tile(probabilities);
            accumulator = mma(probabilities, value_tile, accumulator);
            row_max = next_max;
        }
        let epsilon: Tile<f32, { [1, 1] }> = constant(1.0e-8f32, const_shape![1, 1]);
        let denominator: Tile<f32, { [1, D] }> =
            max_tile(row_sum, epsilon).broadcast(const_shape![1, D]);
        let output: Tile<bf16, { [1, 1, D] }> =
            convert_tile(true_div(accumulator, denominator).reshape(const_shape![1, 1, D]));
        out.store(output);
    }

    /// Request-tiled causal prefill attention over the flat physical KV cache.
    ///
    /// Each tile block owns `BM` consecutive query rows from one request and
    /// one query head. Keeping the request dimension explicit is important:
    /// flattened query rows from adjacent requests must never share a causal
    /// tile. The launch grid is `(requests, query_heads, query_blocks)`.
    #[allow(clippy::too_many_arguments)]
    #[cutile::entry()]
    unsafe fn ragged_prefill_attention_bf16<
        const BM: i32,
        const BN: i32,
        const D: i32,
        const QUERY_HEADS: i32,
        const KV_HEADS: i32,
    >(
        query_ptr: *mut bf16,
        query_start_offsets: &Tensor<u32, { [-1] }>,
        context_slots: &Tensor<u32, { [-1, -1] }>,
        context_lengths_ptr: *mut i32,
        key_cache_ptr: *mut bf16,
        value_cache_ptr: *mut bf16,
        out_ptr: *mut bf16,
        scale: f32,
        group_size: i32,
    ) {
        let pid = get_tile_block_id();
        let request = pid.0;
        let query_head = pid.1;
        let query_block = pid.2;
        let kv_head = query_head / group_size;
        let offsets = query_start_offsets.partition(const_shape![1]);
        let query_start: Tile<i32, { [1] }> = bitcast(offsets.load([request]));
        let query_start: i32 = tile_to_scalar(query_start.reshape(const_shape![]));
        let query_end: Tile<i32, { [1] }> = bitcast(offsets.load([request + 1i32]));
        let query_end: i32 = tile_to_scalar(query_end.reshape(const_shape![]));
        let block_start = query_start + query_block * BM;
        if block_start < query_end {
            let query_lane: Tile<i32, { [BM] }> = iota(const_shape![BM]);
            let query_rows: Tile<i32, { [BM] }> =
                block_start.broadcast(const_shape![BM]) + query_lane;
            let valid_query: Tile<bool, { [BM] }> =
                lt_tile(query_rows, query_end.broadcast(const_shape![BM]));
            let feature: Tile<i32, { [D] }> = iota(const_shape![D]);
            let query_offsets: Tile<i32, { [BM, D] }> = (query_rows
                * QUERY_HEADS.broadcast(const_shape![BM])
                + query_head.broadcast(const_shape![BM]))
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, D])
                * D.broadcast(const_shape![BM, D])
                + feature
                    .reshape(const_shape![1, D])
                    .broadcast(const_shape![BM, D]);
            let query_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(query_ptr);
            let query_base: PointerTile<*mut bf16, { [1, 1] }> =
                query_base.reshape(const_shape![1, 1]);
            let query_base: PointerTile<*mut bf16, { [BM, D] }> =
                query_base.broadcast(const_shape![BM, D]);
            let query_pointers = query_base.offset_tile(query_offsets);
            let query_mask: Tile<bool, { [BM, D] }> = valid_query
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, D]);
            let (query, _): (Tile<bf16, { [BM, D] }>, Token) = load_ptr_tko(
                query_pointers,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(query_mask),
                Some(0.0),
                None,
                Latency::<0>,
            );

            let lengths_base: PointerTile<*mut i32, { [] }> = pointer_to_tile(context_lengths_ptr);
            let lengths_base: PointerTile<*mut i32, { [1] }> =
                lengths_base.reshape(const_shape![1]);
            let lengths_base: PointerTile<*mut i32, { [BM] }> =
                lengths_base.broadcast(const_shape![BM]);
            let length_pointers = lengths_base.offset_tile(query_rows);
            let (query_context_lengths, _): (Tile<i32, { [BM] }>, Token) = load_ptr_tko(
                length_pointers,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid_query),
                Some(0i32),
                None,
                Latency::<0>,
            );
            let max_context_len: Tile<i32, { [1] }> = reduce_max(query_context_lengths, 0i32);
            let max_context_len: i32 = tile_to_scalar(max_context_len.reshape(const_shape![]));

            const NEGATIVE_INFINITY: f32 = -1.0e30f32;
            let mut row_max: Tile<f32, { [BM, 1] }> =
                constant(NEGATIVE_INFINITY, const_shape![BM, 1]);
            const ZERO: f32 = 0.0f32;
            let mut row_sum: Tile<f32, { [BM, 1] }> = constant(ZERO, const_shape![BM, 1]);
            let mut accumulator: Tile<f32, { [BM, D] }> = constant(ZERO, const_shape![BM, D]);
            let key_lane: Tile<i32, { [BN] }> = iota(const_shape![BN]);
            let context_slots = context_slots.partition(const_shape![1, BN]);
            let cache_feature: Tile<i32, { [D] }> = iota(const_shape![D]);
            let key_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(key_cache_ptr);
            let key_base: PointerTile<*mut bf16, { [1, 1] }> = key_base.reshape(const_shape![1, 1]);
            let key_base: PointerTile<*mut bf16, { [BN, D] }> =
                key_base.broadcast(const_shape![BN, D]);
            let value_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(value_cache_ptr);
            let value_base: PointerTile<*mut bf16, { [1, 1] }> =
                value_base.reshape(const_shape![1, 1]);
            let value_base: PointerTile<*mut bf16, { [BN, D] }> =
                value_base.broadcast(const_shape![BN, D]);

            for block in 0i32..((max_context_len + BN - 1i32) / BN) {
                let slots: Tile<u32, { [1, BN] }> = context_slots.load([request, block]);
                let slots: Tile<i32, { [BN] }> = bitcast(slots.reshape(const_shape![BN]));
                let key_positions: Tile<i32, { [BN] }> =
                    (block * BN).broadcast(const_shape![BN]) + key_lane;
                let valid_key: Tile<bool, { [BN] }> =
                    lt_tile(key_positions, max_context_len.broadcast(const_shape![BN]));
                let cache_rows: Tile<i32, { [BN] }> = slots * KV_HEADS.broadcast(const_shape![BN])
                    + kv_head.broadcast(const_shape![BN]);
                let cache_offsets: Tile<i32, { [BN, D] }> = cache_rows
                    .reshape(const_shape![BN, 1])
                    .broadcast(const_shape![BN, D])
                    * D.broadcast(const_shape![BN, D])
                    + cache_feature
                        .reshape(const_shape![1, D])
                        .broadcast(const_shape![BN, D]);
                let cache_mask: Tile<bool, { [BN, D] }> = valid_key
                    .reshape(const_shape![BN, 1])
                    .broadcast(const_shape![BN, D]);
                let key_pointers = key_base.offset_tile(cache_offsets);
                let (key, _): (Tile<bf16, { [BN, D] }>, Token) = load_ptr_tko(
                    key_pointers,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(cache_mask),
                    Some(0.0),
                    None,
                    Latency::<0>,
                );
                let scores_zero: Tile<f32, { [BM, BN] }> = constant(ZERO, const_shape![BM, BN]);
                let scores: Tile<f32, { [BM, BN] }> = mma(query, key.transpose(), scores_zero);
                let causal: Tile<bool, { [BM, BN] }> = lt_tile(
                    key_positions
                        .reshape(const_shape![1, BN])
                        .broadcast(const_shape![BM, BN]),
                    query_context_lengths
                        .reshape(const_shape![BM, 1])
                        .broadcast(const_shape![BM, BN]),
                ) & valid_query
                    .reshape(const_shape![BM, 1])
                    .broadcast(const_shape![BM, BN]);
                let scaled: Tile<f32, { [BM, BN] }> =
                    scores * scale.broadcast(const_shape![BM, BN]);
                let negative_infinity: Tile<f32, { [BM, BN] }> =
                    constant(NEGATIVE_INFINITY, const_shape![BM, BN]);
                let scores = select(causal, scaled, negative_infinity);
                let block_max: Tile<f32, { [BM] }> = reduce_max(scores, 1i32);
                let block_max: Tile<f32, { [BM, 1] }> = block_max.reshape(const_shape![BM, 1]);
                let next_max = max_tile(row_max, block_max);
                let probabilities: Tile<f32, { [BM, BN] }> =
                    exp(scores - next_max.broadcast(const_shape![BM, BN]));
                let block_sum: Tile<f32, { [BM] }> = reduce_sum(probabilities, 1i32);
                let block_sum: Tile<f32, { [BM, 1] }> = block_sum.reshape(const_shape![BM, 1]);
                let correction: Tile<f32, { [BM, 1] }> = exp(row_max - next_max);
                row_sum = row_sum * correction + block_sum;
                accumulator = accumulator * correction.broadcast(const_shape![BM, D]);
                let value_pointers = value_base.offset_tile(cache_offsets);
                let (value, _): (Tile<bf16, { [BN, D] }>, Token) = load_ptr_tko(
                    value_pointers,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(cache_mask),
                    Some(0.0),
                    None,
                    Latency::<0>,
                );
                let probabilities: Tile<bf16, { [BM, BN] }> = convert_tile(probabilities);
                accumulator = mma(probabilities, value, accumulator);
                row_max = next_max;
            }

            const EPSILON: f32 = 1.0e-8f32;
            let epsilon: Tile<f32, { [BM, 1] }> = constant(EPSILON, const_shape![BM, 1]);
            let denominator: Tile<f32, { [BM, D] }> =
                max_tile(row_sum, epsilon).broadcast(const_shape![BM, D]);
            let output: Tile<bf16, { [BM, D] }> = convert_tile(true_div(accumulator, denominator));
            let output_base: PointerTile<*mut bf16, { [] }> = pointer_to_tile(out_ptr);
            let output_base: PointerTile<*mut bf16, { [1, 1] }> =
                output_base.reshape(const_shape![1, 1]);
            let output_base: PointerTile<*mut bf16, { [BM, D] }> =
                output_base.broadcast(const_shape![BM, D]);
            let output_pointers = output_base.offset_tile(query_offsets);
            let _output_store: Token = store_ptr_tko(
                output_pointers,
                output,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(query_mask),
                None,
                Latency::<0>,
            );
        }
    }

    #[cutile::entry()]
    unsafe fn gather_rows_bf16<const WIDTH: i32, const BLOCK: i32>(
        input_ptr: *mut bf16,
        rows: &Tensor<u32, { [-1] }>,
        out: &mut Tensor<bf16, { [1, BLOCK] }>,
    ) {
        let pid = get_tile_block_id();
        let row: Tile<u32, { [1] }> = rows.partition(const_shape![1]).load([pid.0]);
        let row: Tile<i32, { [1] }> = bitcast(row);
        let row: i32 = tile_to_scalar(row.reshape(const_shape![]));
        let offsets: Tile<i32, { [BLOCK] }> = (row * WIDTH + pid.1 * BLOCK)
            .broadcast(const_shape![BLOCK])
            + iota(const_shape![BLOCK]);
        let input: PointerTile<*mut bf16, { [] }> = pointer_to_tile(input_ptr);
        let input: PointerTile<*mut bf16, { [1] }> = input.reshape(const_shape![1]);
        let input: PointerTile<*mut bf16, { [BLOCK] }> = input.broadcast(const_shape![BLOCK]);
        let input = input.offset_tile(offsets);
        let (values, _): (Tile<bf16, { [BLOCK] }>, Token) = load_ptr_tko(
            input,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        out.store(values.reshape(const_shape![1, BLOCK]));
    }

    #[cutile::entry()]
    fn argmax_blocks_batch_bf16<const BLOCK: i32>(
        logits: &Tensor<bf16, { [-1, -1] }>,
        block_max: &mut Tensor<f32, { [1, 1] }>,
        block_index: &mut Tensor<u32, { [1, 1] }>,
        length: i32,
    ) {
        let pid = get_tile_block_id();
        let row = pid.0;
        let block = pid.1;
        let logits = logits.partition(const_shape![1, BLOCK]);
        let values_bf16: Tile<bf16, { [1, BLOCK] }> = logits.load([row, block]);
        let values: Tile<f32, { [BLOCK] }> = convert_tile(values_bf16.reshape(const_shape![BLOCK]));
        let base: i32 = block * BLOCK;
        let base: Tile<i32, { [BLOCK] }> = base.broadcast(const_shape![BLOCK]);
        let indices: Tile<i32, { [BLOCK] }> = base + iota(const_shape![BLOCK]);
        let valid: Tile<bool, { [BLOCK] }> =
            lt_tile(indices, length.broadcast(const_shape![BLOCK]));
        let magnitude: Tile<f32, { [BLOCK] }> = constant(1.0e30f32, const_shape![BLOCK]);
        let zero: Tile<f32, { [BLOCK] }> = constant(0.0f32, const_shape![BLOCK]);
        let values: Tile<f32, { [BLOCK] }> = select(valid, values, zero - magnitude);
        let maximum: Tile<f32, { [1] }> = reduce_max(values, 0i32);
        let maximum_scalar: f32 = tile_to_scalar(maximum.reshape(const_shape![]));
        let is_maximum: Tile<bool, { [BLOCK] }> =
            eq_tile(values, maximum_scalar.broadcast(const_shape![BLOCK]));
        let invalid_index: Tile<i32, { [BLOCK] }> = constant(2147483647i32, const_shape![BLOCK]);
        let candidates: Tile<i32, { [BLOCK] }> = select(is_maximum, indices, invalid_index);
        let winner: Tile<i32, { [1] }> = reduce_min(candidates, 0i32);
        let winner: i32 = tile_to_scalar(winner.reshape(const_shape![]));
        let maximum: Tile<f32, { [1, 1] }> =
            scalar_to_tile(maximum_scalar).reshape(const_shape![1, 1]);
        let winner: Tile<i32, { [1, 1] }> = scalar_to_tile(winner).reshape(const_shape![1, 1]);
        let winner: Tile<u32, { [1, 1] }> = bitcast(winner);
        block_max.store(maximum);
        block_index.store(winner);
    }

    #[cutile::entry()]
    fn argmax_reduce_batch_bf16<const BLOCK: i32>(
        block_max: &Tensor<f32, { [-1, -1] }>,
        block_index: &Tensor<u32, { [-1, -1] }>,
        out: &mut Tensor<u32, { [1] }>,
        num_blocks: i32,
    ) {
        let row = get_tile_block_id().0;
        let maxima: Tile<f32, { [1, BLOCK] }> = block_max
            .partition(const_shape![1, BLOCK])
            .load([row, 0i32]);
        let maxima: Tile<f32, { [BLOCK] }> = maxima.reshape(const_shape![BLOCK]);
        let indices: Tile<u32, { [1, BLOCK] }> = block_index
            .partition(const_shape![1, BLOCK])
            .load([row, 0i32]);
        let indices: Tile<i32, { [BLOCK] }> = bitcast(indices.reshape(const_shape![BLOCK]));
        let offsets: Tile<i32, { [BLOCK] }> = iota(const_shape![BLOCK]);
        let valid: Tile<bool, { [BLOCK] }> =
            lt_tile(offsets, num_blocks.broadcast(const_shape![BLOCK]));
        let magnitude: Tile<f32, { [BLOCK] }> = constant(1.0e30f32, const_shape![BLOCK]);
        let zero: Tile<f32, { [BLOCK] }> = constant(0.0f32, const_shape![BLOCK]);
        let masked: Tile<f32, { [BLOCK] }> = select(valid, maxima, zero - magnitude);
        let maximum: Tile<f32, { [1] }> = reduce_max(masked, 0i32);
        let maximum_scalar: f32 = tile_to_scalar(maximum.reshape(const_shape![]));
        let is_maximum: Tile<bool, { [BLOCK] }> =
            eq_tile(masked, maximum_scalar.broadcast(const_shape![BLOCK]));
        let invalid_index: Tile<i32, { [BLOCK] }> = constant(2147483647i32, const_shape![BLOCK]);
        let candidates: Tile<i32, { [BLOCK] }> = select(is_maximum, indices, invalid_index);
        let winner: Tile<i32, { [1] }> = reduce_min(candidates, 0i32);
        let winner: i32 = tile_to_scalar(winner.reshape(const_shape![]));
        let winner: Tile<i32, { [1] }> = scalar_to_tile(winner).reshape(const_shape![1]);
        let winner: Tile<u32, { [1] }> = bitcast(winner);
        out.store(winner);
    }

    #[cutile::entry()]
    fn argmax_blocks_bf16<const BLOCK: i32>(
        logits: &Tensor<bf16, { [-1] }>,
        block_max: &mut Tensor<f32, { [1] }>,
        block_index: &mut Tensor<u32, { [1] }>,
        length: i32,
    ) {
        let block: i32 = get_tile_block_id().0;
        let logits = logits.partition(const_shape![BLOCK]);
        let values_bf16: Tile<bf16, { [BLOCK] }> = logits.load([block]);
        let values: Tile<f32, { [BLOCK] }> = convert_tile(values_bf16);
        let base: i32 = block * BLOCK;
        let base: Tile<i32, { [BLOCK] }> = base.broadcast(const_shape![BLOCK]);
        let offsets: Tile<i32, { [BLOCK] }> = iota(const_shape![BLOCK]);
        let indices: Tile<i32, { [BLOCK] }> = base + offsets;
        let length: Tile<i32, { [BLOCK] }> = length.broadcast(const_shape![BLOCK]);
        let valid: Tile<bool, { [BLOCK] }> = lt_tile(indices, length);
        let magnitude: Tile<f32, { [BLOCK] }> = constant(1.0e30f32, const_shape![BLOCK]);
        let zero: Tile<f32, { [BLOCK] }> = constant(0.0f32, const_shape![BLOCK]);
        let negative_infinity: Tile<f32, { [BLOCK] }> = zero - magnitude;
        let values: Tile<f32, { [BLOCK] }> = select(valid, values, negative_infinity);
        let maximum: Tile<f32, { [1] }> = reduce_max(values, 0i32);
        let maximum_scalar: f32 = tile_to_scalar(maximum.reshape(const_shape![]));
        let maximum: Tile<f32, { [BLOCK] }> = maximum_scalar.broadcast(const_shape![BLOCK]);
        let is_maximum: Tile<bool, { [BLOCK] }> = eq_tile(values, maximum);
        let invalid_index: Tile<i32, { [BLOCK] }> = constant(2147483647i32, const_shape![BLOCK]);
        let candidates: Tile<i32, { [BLOCK] }> = select(is_maximum, indices, invalid_index);
        let winner: Tile<i32, { [1] }> = reduce_min(candidates, 0i32);
        let winner: i32 = tile_to_scalar(winner.reshape(const_shape![]));
        let maximum: Tile<f32, { [1] }> = scalar_to_tile(maximum_scalar).reshape(const_shape![1]);
        let winner: Tile<i32, { [1] }> = scalar_to_tile(winner).reshape(const_shape![1]);
        let winner: Tile<u32, { [1] }> = bitcast(winner);
        block_max.store(maximum);
        block_index.store(winner);
    }

    #[cutile::entry()]
    fn argmax_reduce_bf16<const BLOCK: i32>(
        block_max: &Tensor<f32, { [-1] }>,
        block_index: &Tensor<u32, { [-1] }>,
        out: &mut Tensor<u32, { [1] }>,
        num_blocks: i32,
    ) {
        let maxima: Tile<f32, { [BLOCK] }> = block_max.partition(const_shape![BLOCK]).load([0i32]);
        let indices: Tile<u32, { [BLOCK] }> =
            block_index.partition(const_shape![BLOCK]).load([0i32]);
        let indices: Tile<i32, { [BLOCK] }> = bitcast(indices);
        let offsets: Tile<i32, { [BLOCK] }> = iota(const_shape![BLOCK]);
        let num_blocks: Tile<i32, { [BLOCK] }> = num_blocks.broadcast(const_shape![BLOCK]);
        let valid: Tile<bool, { [BLOCK] }> = lt_tile(offsets, num_blocks);
        let magnitude: Tile<f32, { [BLOCK] }> = constant(1.0e30f32, const_shape![BLOCK]);
        let zero: Tile<f32, { [BLOCK] }> = constant(0.0f32, const_shape![BLOCK]);
        let negative_infinity: Tile<f32, { [BLOCK] }> = zero - magnitude;
        let masked: Tile<f32, { [BLOCK] }> = select(valid, maxima, negative_infinity);
        let maximum: Tile<f32, { [1] }> = reduce_max(masked, 0i32);
        let maximum_scalar: f32 = tile_to_scalar(maximum.reshape(const_shape![]));
        let maximum: Tile<f32, { [BLOCK] }> = maximum_scalar.broadcast(const_shape![BLOCK]);
        let is_maximum: Tile<bool, { [BLOCK] }> = eq_tile(masked, maximum);
        let invalid_index: Tile<i32, { [BLOCK] }> = constant(2147483647i32, const_shape![BLOCK]);
        let candidates: Tile<i32, { [BLOCK] }> = select(is_maximum, indices, invalid_index);
        let winner: Tile<i32, { [1] }> = reduce_min(candidates, 0i32);
        let winner: i32 = tile_to_scalar(winner.reshape(const_shape![]));
        let winner: Tile<i32, { [1] }> = scalar_to_tile(winner).reshape(const_shape![1]);
        let winner: Tile<u32, { [1] }> = bitcast(winner);
        out.store(winner);
    }
}

#[allow(unused_imports)]
pub(crate) use tile::{
    add_rms_norm_bf16, argmax_blocks_batch_bf16, argmax_blocks_bf16, argmax_reduce_batch_bf16,
    argmax_reduce_bf16, causal_attention_bf16, embedding_bf16, gather_flat_kv_bf16,
    gather_rows_bf16, gemma_add_rms_norm_bf16, gemma_rms_norm_bf16, ragged_attention_bf16,
    ragged_prefill_attention_bf16, rms_norm_bf16, rope_kv_write_bf16, rope_q_bf16, silu_mul_bf16,
};
