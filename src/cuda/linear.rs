//! Construction-time selected quantized linear implementations.
//!
//! These are leaf kernels, not runtime plugins. Model programs store concrete
//! typed projection artifacts and invoke them without inspecting checkpoint
//! formats or device capabilities in the token hot path.

use std::sync::Arc;

use cuda_async::device_operation::DeviceOp;
use cuda_core::{IntoResult, Stream};
use cutile::{
    api,
    core::bf16,
    tensor::{PartitionMut, Reshape, Tensor, ToHostVec},
    tile_kernel::{CompileOptions, TileKernel},
};

use crate::{
    cuda::execution::StreamExecution,
    model::{
        ModelError,
        weights::{WeightDtype, WeightSource, WeightTensor},
    },
    quantization::{decode_e2m1, decode_e4m3fn},
};

const TILE_M: usize = 16;
const TILE_N: usize = 16;
const DENSE_TILE_N: usize = 64;
const GROUP_K: usize = 16;
const GROUPED_SMALL_TILE_M: usize = 16;
const GROUPED_LARGE_TILE_M: usize = 64;
const GROUPED_TILE_N: usize = 64;
const GROUPED_TILE_K: usize = GROUP_K;
const FP8_SUBNORMAL_SCALE: f32 = 1.0 / 512.0;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn fp8_w8a16<const K_TILES: i32>(
        mut output: MappedPartitionMut<bf16, { [16, 64] }, { [8, 1] }>,
        input: &Tensor<bf16, { [-1, -1] }>,
        weight: &Tensor<u8, { [-1, -1] }>,
        weight_scale: f32,
        subnormal_scale: f32,
    ) {
        let k_tiles = Dim::new(K_TILES);
        for out_idx in output.iter_indices() {
            let (row_tile, column_tile) = out_idx.components();
            const ZERO_F32: f32 = 0.0;
            let mut accumulator = broadcast_scalar(ZERO_F32, const_shape![16, 64]);

            for k_tile in k_tiles {
                let activation = input.load_tile(const_shape![16, 16], [row_tile, k_tile]);
                let encoded: Tile<i32, { [64, 16] }> =
                    exti(weight.load_tile(const_shape![64, 16], [column_tile, k_tile]));
                let decoded = decode_fp8_wide(encoded, subnormal_scale);
                let scale = broadcast_scalar(weight_scale, const_shape![64, 16]);
                let weight: Tile<bf16, { [64, 16] }> = ftof(decoded * scale, rounding::NearestEven);
                accumulator = mma(activation, weight.transpose(), accumulator);
            }
            let output_tile: Tile<bf16, { [16, 64] }> = ftof(accumulator, rounding::NearestEven);
            output.store(output_tile, out_idx);
        }
    }

    #[cutile::entry()]
    fn nvfp4_w4a16<const K_TILES: i32>(
        mut output: MappedPartitionMut<bf16, { [16, 64] }, { [8, 1] }>,
        input: &Tensor<bf16, { [-1, -1] }>,
        packed_weight: &Tensor<u8, { [-1, -1] }>,
        weight_scale: &Tensor<bf16, { [-1, -1] }>,
        weight_global_scale: f32,
    ) {
        let k_tiles = Dim::new(K_TILES);
        const LOW_MASK: i32 = 0x0f;
        let low_mask = broadcast_scalar(LOW_MASK, const_shape![64, 8]);
        const NIBBLE_SHIFT: i32 = 4;
        let nibble_shift = broadcast_scalar(NIBBLE_SHIFT, const_shape![64, 8]);
        const ZERO_I32: i32 = 0;
        let zero_i32 = broadcast_scalar(ZERO_I32, const_shape![64, 8]);
        const BYTE_MODULUS: i32 = 256;
        let byte_modulus = broadcast_scalar(BYTE_MODULUS, const_shape![64, 8]);

        for out_idx in output.iter_indices() {
            let (row_tile, column_tile) = out_idx.components();
            const ZERO_F32: f32 = 0.0;
            let mut accumulator = broadcast_scalar(ZERO_F32, const_shape![16, 64]);

            for k_tile in k_tiles {
                let activation = input.load_tile(const_shape![16, 16], [row_tile, k_tile]);
                let packed: Tile<i32, { [64, 8] }> =
                    exti(packed_weight.load_tile(const_shape![64, 8], [column_tile, k_tile]));
                let packed = select(lt_tile(packed, zero_i32), packed + byte_modulus, packed);
                let low = andi(packed, low_mask).reshape(const_shape![64, 8, 1]);
                let high = shri(packed, nibble_shift).reshape(const_shape![64, 8, 1]);
                let nibbles: Tile<i32, { [64, 8, 2] }> = cat(low, high, 2);
                let decoded =
                    decode_fp4_grouped(nibbles.reshape(const_shape![64, 16]), const_shape![64, 16]);
                let scale: Tile<f32, { [64, 16] }> = convert_tile(
                    weight_scale
                        .load_tile(const_shape![64, 1], [column_tile, k_tile])
                        .broadcast(const_shape![64, 16]),
                );
                let global = broadcast_scalar(weight_global_scale, const_shape![64, 16]);
                let weight: Tile<bf16, { [64, 16] }> =
                    ftof(decoded * scale * global, rounding::NearestEven);
                accumulator = mma(activation, weight.transpose(), accumulator);
            }
            let output_tile: Tile<bf16, { [16, 64] }> = ftof(accumulator, rounding::NearestEven);
            output.store(output_tile, out_idx);
        }
    }

    /// One launch processes all routed expert segments. Dispatch pads each
    /// expert segment to 16 rows and supplies one expert id per row tile, so
    /// no output tile crosses an expert boundary and no per-expert launch is
    /// required.
    #[cutile::entry(unchecked_accesses = false)]
    fn grouped_nvfp4_w4a16<
        const K_TILES: i32,
        const ROW_TILE: i32,
        const N: i32,
        const K_PACKED: i32,
        const K_SCALES: i32,
    >(
        mut output: MappedPartitionMut<bf16, { [ROW_TILE, 64] }, { [8, 1] }>,
        dispatched: &Tensor<bf16, { [-1, -1] }>,
        expert_by_row_tile: &Tensor<i32, { [-1] }>,
        packed_weight: &Tensor<u8, { [-1, N, K_PACKED] }>,
        weight_scale: &Tensor<u8, { [-1, N, K_SCALES] }>,
        weight_global_scale: &Tensor<f32, { [-1] }>,
    ) {
        const LOW_MASK: i32 = 0x0f;
        let low_mask: Tile<i32, { [1, 64, 8] }> =
            broadcast_scalar(LOW_MASK, const_shape![1, 64, 8]);
        const NIBBLE_SHIFT: i32 = 4;
        let nibble_shift: Tile<i32, { [1, 64, 8] }> =
            broadcast_scalar(NIBBLE_SHIFT, const_shape![1, 64, 8]);
        const ZERO_I32: i32 = 0;
        let zero_i32: Tile<i32, { [1, 64, 8] }> =
            broadcast_scalar(ZERO_I32, const_shape![1, 64, 8]);
        const BYTE_MODULUS: i32 = 256;
        let byte_modulus: Tile<i32, { [1, 64, 8] }> =
            broadcast_scalar(BYTE_MODULUS, const_shape![1, 64, 8]);
        let k_tiles = Dim::new(K_TILES);

        // `iter_indices` maps the full logical output grid onto a physical
        // grid capped at the device's SM count. It mints proof-carrying,
        // disjoint indices, so persistent stores remain checked and safe.
        for out_idx in output.iter_indices() {
            let (row_tile, column_tile) = out_idx.components();
            let expert_tile = expert_by_row_tile.load_tile(const_shape![1], [row_tile]);
            let expert: i32 = tile_to_scalar(expert_tile.reshape(const_shape![]));
            let global = weight_global_scale
                .load_tile(const_shape![1], [expert])
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![64, 16]);
            const ZERO_F32: f32 = 0.0;
            let mut accumulator = broadcast_scalar(ZERO_F32, const_shape![ROW_TILE, 64]);

            for k_tile in k_tiles {
                let activation =
                    dispatched.load_tile(const_shape![ROW_TILE, 16], [row_tile, k_tile]);
                let packed: Tile<i32, { [1, 64, 8] }> = exti(
                    packed_weight.load_tile(const_shape![1, 64, 8], [expert, column_tile, k_tile]),
                );
                let packed = select(lt_tile(packed, zero_i32), packed + byte_modulus, packed);
                let low = andi(packed, low_mask).reshape(const_shape![64, 8, 1]);
                let high = shri(packed, nibble_shift).reshape(const_shape![64, 8, 1]);
                let nibbles: Tile<i32, { [64, 8, 2] }> = cat(low, high, 2);
                let weight =
                    decode_fp4_grouped(nibbles.reshape(const_shape![64, 16]), const_shape![64, 16]);
                let scale_bits: Tile<i32, { [64, 1] }> = exti(
                    weight_scale
                        .load_tile(const_shape![1, 64, 1], [expert, column_tile, k_tile])
                        .reshape(const_shape![64, 1]),
                );
                let scale = decode_fp8_shape(scale_bits, const_shape![64, 1])
                    .broadcast(const_shape![64, 16]);
                let weight: Tile<bf16, { [64, 16] }> =
                    ftof(weight * scale * global, rounding::NearestEven);
                accumulator = mma(activation, weight.transpose(), accumulator);
            }
            let output_tile: Tile<bf16, { [ROW_TILE, 64] }> =
                ftof(accumulator, rounding::NearestEven);
            output.store(output_tile, out_idx);
        }
    }

    /// Ampere W4A16 candidate with a 32-wide reduction tile. Two adjacent
    /// NVFP4 scale groups are decoded together, cutting loop/control overhead
    /// in half while preserving the checkpoint's packed FP4/FP8 storage.
    #[cutile::entry(unchecked_accesses = false)]
    fn grouped_nvfp4_w4a16_k32<
        const K_TILES: i32,
        const ROW_TILE: i32,
        const TILE_N: i32,
        const N: i32,
        const K_PACKED: i32,
        const K_SCALES: i32,
    >(
        mut output: MappedPartitionMut<bf16, { [ROW_TILE, TILE_N] }, { [8, 1] }>,
        dispatched: &Tensor<bf16, { [-1, -1] }>,
        expert_by_row_tile: &Tensor<i32, { [-1] }>,
        packed_weight: &Tensor<u8, { [-1, N, K_PACKED] }>,
        weight_scale: &Tensor<u8, { [-1, N, K_SCALES] }>,
        weight_global_scale: &Tensor<f32, { [-1] }>,
    ) {
        const LOW_MASK: i32 = 0x0f;
        let low_mask: Tile<i32, { [1, TILE_N, 16] }> =
            broadcast_scalar(LOW_MASK, const_shape![1, TILE_N, 16]);
        const NIBBLE_SHIFT: i32 = 4;
        let nibble_shift: Tile<i32, { [1, TILE_N, 16] }> =
            broadcast_scalar(NIBBLE_SHIFT, const_shape![1, TILE_N, 16]);
        const ZERO_I32: i32 = 0;
        let zero_i32: Tile<i32, { [1, TILE_N, 16] }> =
            broadcast_scalar(ZERO_I32, const_shape![1, TILE_N, 16]);
        const BYTE_MODULUS: i32 = 256;
        let byte_modulus: Tile<i32, { [1, TILE_N, 16] }> =
            broadcast_scalar(BYTE_MODULUS, const_shape![1, TILE_N, 16]);
        let k_tiles = Dim::new(K_TILES);

        for out_idx in output.iter_indices() {
            let (row_tile, column_tile) = out_idx.components();
            let expert_tile = expert_by_row_tile.load_tile(const_shape![1], [row_tile]);
            let expert: i32 = tile_to_scalar(expert_tile.reshape(const_shape![]));
            let global = weight_global_scale
                .load_tile(const_shape![1], [expert])
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![TILE_N, 32]);
            const ZERO_F32: f32 = 0.0;
            let mut accumulator = broadcast_scalar(ZERO_F32, const_shape![ROW_TILE, TILE_N]);

            for k_tile in k_tiles {
                let activation =
                    dispatched.load_tile(const_shape![ROW_TILE, 32], [row_tile, k_tile]);
                let packed: Tile<i32, { [1, TILE_N, 16] }> = exti(
                    packed_weight
                        .load_tile(const_shape![1, TILE_N, 16], [expert, column_tile, k_tile]),
                );
                let packed = select(lt_tile(packed, zero_i32), packed + byte_modulus, packed);
                let low = andi(packed, low_mask).reshape(const_shape![TILE_N, 16, 1]);
                let high = shri(packed, nibble_shift).reshape(const_shape![TILE_N, 16, 1]);
                let nibbles: Tile<i32, { [TILE_N, 16, 2] }> = cat(low, high, 2);
                let weight = decode_fp4_grouped(
                    nibbles.reshape(const_shape![TILE_N, 32]),
                    const_shape![TILE_N, 32],
                );
                let scale_bits: Tile<i32, { [TILE_N, 2] }> = exti(
                    weight_scale
                        .load_tile(const_shape![1, TILE_N, 2], [expert, column_tile, k_tile])
                        .reshape(const_shape![TILE_N, 2]),
                );
                let scale = decode_fp8_shape(scale_bits, const_shape![TILE_N, 2])
                    .reshape(const_shape![TILE_N, 2, 1])
                    .broadcast(const_shape![TILE_N, 2, 16])
                    .reshape(const_shape![TILE_N, 32]);
                let weight: Tile<bf16, { [TILE_N, 32] }> =
                    ftof(weight * scale * global, rounding::NearestEven);
                accumulator = mma(activation, weight.transpose(), accumulator);
            }
            let output_tile: Tile<bf16, { [ROW_TILE, TILE_N] }> =
                ftof(accumulator, rounding::NearestEven);
            output.store(output_tile, out_idx);
        }
    }

    /// Fused routed-expert FC1. Gate and up projections share the dispatched
    /// activation load, retain independent ModelOpt scales, and feed a
    /// register-resident SwiGLU epilogue. Only the activated BF16 down-input
    /// reaches global memory.
    #[cutile::entry(unchecked_accesses = false)]
    fn grouped_nvfp4_w4a16_silu_mul<
        const K_TILES: i32,
        const ROW_TILE: i32,
        const N: i32,
        const K_PACKED: i32,
        const K_SCALES: i32,
    >(
        mut output: MappedPartitionMut<bf16, { [ROW_TILE, 64] }, { [8, 1] }>,
        dispatched: &Tensor<bf16, { [-1, -1] }>,
        expert_by_row_tile: &Tensor<i32, { [-1] }>,
        gate_packed_weight: &Tensor<u8, { [-1, N, K_PACKED] }>,
        gate_weight_scale: &Tensor<u8, { [-1, N, K_SCALES] }>,
        gate_weight_global_scale: &Tensor<f32, { [-1] }>,
        up_packed_weight: &Tensor<u8, { [-1, N, K_PACKED] }>,
        up_weight_scale: &Tensor<u8, { [-1, N, K_SCALES] }>,
        up_weight_global_scale: &Tensor<f32, { [-1] }>,
    ) {
        const LOW_MASK: i32 = 0x0f;
        let low_mask: Tile<i32, { [1, 64, 8] }> =
            broadcast_scalar(LOW_MASK, const_shape![1, 64, 8]);
        const NIBBLE_SHIFT: i32 = 4;
        let nibble_shift: Tile<i32, { [1, 64, 8] }> =
            broadcast_scalar(NIBBLE_SHIFT, const_shape![1, 64, 8]);
        const ZERO_I32: i32 = 0;
        let zero_i32: Tile<i32, { [1, 64, 8] }> =
            broadcast_scalar(ZERO_I32, const_shape![1, 64, 8]);
        const BYTE_MODULUS: i32 = 256;
        let byte_modulus: Tile<i32, { [1, 64, 8] }> =
            broadcast_scalar(BYTE_MODULUS, const_shape![1, 64, 8]);
        let k_tiles = Dim::new(K_TILES);

        for out_idx in output.iter_indices() {
            let (row_tile, column_tile) = out_idx.components();
            let expert_tile = expert_by_row_tile.load_tile(const_shape![1], [row_tile]);
            let expert: i32 = tile_to_scalar(expert_tile.reshape(const_shape![]));
            let gate_global = gate_weight_global_scale
                .load_tile(const_shape![1], [expert])
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![64, 16]);
            let up_global = up_weight_global_scale
                .load_tile(const_shape![1], [expert])
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![64, 16]);
            const ZERO_F32: f32 = 0.0;
            let mut gate_accumulator = broadcast_scalar(ZERO_F32, const_shape![ROW_TILE, 64]);
            let mut up_accumulator = broadcast_scalar(ZERO_F32, const_shape![ROW_TILE, 64]);

            for k_tile in k_tiles {
                let activation =
                    dispatched.load_tile(const_shape![ROW_TILE, 16], [row_tile, k_tile]);

                let gate_packed: Tile<i32, { [1, 64, 8] }> = exti(
                    gate_packed_weight
                        .load_tile(const_shape![1, 64, 8], [expert, column_tile, k_tile]),
                );
                let gate_packed = select(
                    lt_tile(gate_packed, zero_i32),
                    gate_packed + byte_modulus,
                    gate_packed,
                );
                let gate_low = andi(gate_packed, low_mask).reshape(const_shape![64, 8, 1]);
                let gate_high = shri(gate_packed, nibble_shift).reshape(const_shape![64, 8, 1]);
                let gate_nibbles: Tile<i32, { [64, 8, 2] }> = cat(gate_low, gate_high, 2);
                let gate_weight = decode_fp4_grouped(
                    gate_nibbles.reshape(const_shape![64, 16]),
                    const_shape![64, 16],
                );
                let gate_scale_bits: Tile<i32, { [64, 1] }> = exti(
                    gate_weight_scale
                        .load_tile(const_shape![1, 64, 1], [expert, column_tile, k_tile])
                        .reshape(const_shape![64, 1]),
                );
                let gate_scale = decode_fp8_shape(gate_scale_bits, const_shape![64, 1])
                    .broadcast(const_shape![64, 16]);
                let gate_weight: Tile<bf16, { [64, 16] }> = ftof(
                    gate_weight * gate_scale * gate_global,
                    rounding::NearestEven,
                );
                gate_accumulator = mma(activation, gate_weight.transpose(), gate_accumulator);

                let up_packed: Tile<i32, { [1, 64, 8] }> = exti(
                    up_packed_weight
                        .load_tile(const_shape![1, 64, 8], [expert, column_tile, k_tile]),
                );
                let up_packed = select(
                    lt_tile(up_packed, zero_i32),
                    up_packed + byte_modulus,
                    up_packed,
                );
                let up_low = andi(up_packed, low_mask).reshape(const_shape![64, 8, 1]);
                let up_high = shri(up_packed, nibble_shift).reshape(const_shape![64, 8, 1]);
                let up_nibbles: Tile<i32, { [64, 8, 2] }> = cat(up_low, up_high, 2);
                let up_weight = decode_fp4_grouped(
                    up_nibbles.reshape(const_shape![64, 16]),
                    const_shape![64, 16],
                );
                let up_scale_bits: Tile<i32, { [64, 1] }> = exti(
                    up_weight_scale
                        .load_tile(const_shape![1, 64, 1], [expert, column_tile, k_tile])
                        .reshape(const_shape![64, 1]),
                );
                let up_scale = decode_fp8_shape(up_scale_bits, const_shape![64, 1])
                    .broadcast(const_shape![64, 16]);
                let up_weight: Tile<bf16, { [64, 16] }> =
                    ftof(up_weight * up_scale * up_global, rounding::NearestEven);
                up_accumulator = mma(activation, up_weight.transpose(), up_accumulator);
            }

            const ONE_F32: f32 = 1.0;
            let one = broadcast_scalar(ONE_F32, const_shape![ROW_TILE, 64]);
            let activated = gate_accumulator
                * true_div(
                    one,
                    one + exp(
                        broadcast_scalar(ZERO_F32, const_shape![ROW_TILE, 64]) - gate_accumulator
                    ),
                )
                * up_accumulator;
            let output_tile: Tile<bf16, { [ROW_TILE, 64] }> =
                ftof(activated, rounding::NearestEven);
            output.store(output_tile, out_idx);
        }
    }

    fn decode_fp4_grouped<const S: [i32; 2]>(
        nibbles: Tile<i32, S>,
        shape: Shape<S>,
    ) -> Tile<f32, S> {
        // E2M1's normal exponent and mantissa bits map directly into BF16.
        // Constructing the encoding needs one subnormal select instead of the
        // seven value comparisons used by a scalar lookup table. This mirrors
        // Marlin's bit-level dequantization while retaining a tile expression
        // the cuTile compiler can schedule around MMA.
        const MAGNITUDE_MASK: i32 = 7;
        let magnitude = andi(nibbles, broadcast_scalar(MAGNITUDE_MASK, shape));
        const EXPONENT_SHIFT: i32 = 1;
        let exponent = shri(magnitude, broadcast_scalar(EXPONENT_SHIFT, shape));
        const MANTISSA_MASK: i32 = 1;
        let mantissa = andi(magnitude, broadcast_scalar(MANTISSA_MASK, shape));
        const BF16_EXPONENT_REBIAS: i32 = 126;
        const BF16_EXPONENT_SCALE: i32 = 128;
        const BF16_MANTISSA_SCALE: i32 = 64;
        let normal_bits = (exponent + broadcast_scalar(BF16_EXPONENT_REBIAS, shape))
            * broadcast_scalar(BF16_EXPONENT_SCALE, shape)
            + mantissa * broadcast_scalar(BF16_MANTISSA_SCALE, shape);
        const ZERO_I32: i32 = 0;
        let zero_i32 = broadcast_scalar(ZERO_I32, shape);
        const HALF_BF16_BITS: i32 = 0x3f00;
        let subnormal_bits = magnitude * broadcast_scalar(HALF_BF16_BITS, shape);
        let magnitude_bits = select(eq_tile(exponent, zero_i32), subnormal_bits, normal_bits);
        const SIGN_SHIFT: i32 = 3;
        let sign = shri(nibbles, broadcast_scalar(SIGN_SHIFT, shape));
        const BF16_SIGN_BIT: i32 = 0x8000;
        let bits = magnitude_bits + sign * broadcast_scalar(BF16_SIGN_BIT, shape);
        let bits: Tile<u16, S> = trunci(bits, overflow::NoUnsignedWrap);
        let decoded: Tile<bf16, S> = bitcast(bits);
        convert_tile(decoded)
    }

    /// Decode an E4M3 byte tile by constructing the exact BF16 encoding.
    /// Keeping this shape-generic lets grouped kernels retain checkpoint FP8
    /// scale storage and decode only the scale fragment needed by an MMA tile.
    fn decode_fp8_shape<const S: [i32; 2]>(encoded: Tile<i32, S>, shape: Shape<S>) -> Tile<f32, S> {
        const BYTE_MODULUS: i32 = 256;
        let byte_modulus = broadcast_scalar(BYTE_MODULUS, shape);
        const ZERO_I32: i32 = 0;
        let zero_i32 = broadcast_scalar(ZERO_I32, shape);
        let encoded = select(lt_tile(encoded, zero_i32), encoded + byte_modulus, encoded);
        const SIGN_SHIFT: i32 = 7;
        let sign = shri(encoded, broadcast_scalar(SIGN_SHIFT, shape));
        const MAGNITUDE_MASK: i32 = 0x7f;
        let magnitude = andi(encoded, broadcast_scalar(MAGNITUDE_MASK, shape));
        const EXPONENT_SHIFT: i32 = 3;
        let exponent = shri(magnitude, broadcast_scalar(EXPONENT_SHIFT, shape));
        const MANTISSA_MASK: i32 = 7;
        let mantissa = andi(magnitude, broadcast_scalar(MANTISSA_MASK, shape));

        const BF16_EXPONENT_REBIAS: i32 = 120;
        const BF16_EXPONENT_SHIFT: i32 = 128;
        const BF16_MANTISSA_SHIFT: i32 = 16;
        let normal_bits = (exponent + broadcast_scalar(BF16_EXPONENT_REBIAS, shape))
            * broadcast_scalar(BF16_EXPONENT_SHIFT, shape)
            + mantissa * broadcast_scalar(BF16_MANTISSA_SHIFT, shape);

        const SUBNORMAL_ONE_BITS: i32 = 0x3b00;
        let subnormal_low = mantissa * broadcast_scalar(SUBNORMAL_ONE_BITS, shape);
        const SUBNORMAL_TWO_BITS: i32 = 0x3b80;
        const SUBNORMAL_MID_STEP: i32 = 0x40;
        let subnormal_mid = broadcast_scalar(SUBNORMAL_TWO_BITS, shape)
            + (mantissa - broadcast_scalar(2i32, shape))
                * broadcast_scalar(SUBNORMAL_MID_STEP, shape);
        const SUBNORMAL_FOUR_BITS: i32 = 0x3c00;
        const SUBNORMAL_HIGH_STEP: i32 = 0x20;
        let subnormal_high = broadcast_scalar(SUBNORMAL_FOUR_BITS, shape)
            + (mantissa - broadcast_scalar(4i32, shape))
                * broadcast_scalar(SUBNORMAL_HIGH_STEP, shape);
        let below_four = lt_tile(mantissa, broadcast_scalar(4i32, shape));
        let below_two = lt_tile(mantissa, broadcast_scalar(2i32, shape));
        let subnormal_bits = select(
            below_four,
            select(below_two, subnormal_low, subnormal_mid),
            subnormal_high,
        );
        let magnitude_bits = select(eq_tile(exponent, zero_i32), subnormal_bits, normal_bits);
        const BF16_SIGN_BIT: i32 = 0x8000;
        let bits = magnitude_bits + sign * broadcast_scalar(BF16_SIGN_BIT, shape);
        let bits: Tile<u16, S> = trunci(bits, overflow::NoUnsignedWrap);
        let decoded: Tile<bf16, S> = bitcast(bits);
        let decoded: Tile<f32, S> = convert_tile(decoded);
        decoded
    }

    fn decode_fp4(nibbles: Tile<i32, { [16, 16] }>) -> Tile<f32, { [16, 16] }> {
        let eight: Tile<i32, { [16, 16] }> = constant(8i32, const_shape![16, 16]);
        let magnitude = nibbles % eight;
        let sign = nibbles / eight;
        let one: Tile<i32, { [16, 16] }> = constant(1i32, const_shape![16, 16]);
        let two: Tile<i32, { [16, 16] }> = constant(2i32, const_shape![16, 16]);
        let three: Tile<i32, { [16, 16] }> = constant(3i32, const_shape![16, 16]);
        let four: Tile<i32, { [16, 16] }> = constant(4i32, const_shape![16, 16]);
        let five: Tile<i32, { [16, 16] }> = constant(5i32, const_shape![16, 16]);
        let six: Tile<i32, { [16, 16] }> = constant(6i32, const_shape![16, 16]);
        let seven: Tile<i32, { [16, 16] }> = constant(7i32, const_shape![16, 16]);
        let zero_f: Tile<f32, { [16, 16] }> = constant(0.0f32, const_shape![16, 16]);
        let half_f: Tile<f32, { [16, 16] }> = constant(0.5f32, const_shape![16, 16]);
        let one_f: Tile<f32, { [16, 16] }> = constant(1.0f32, const_shape![16, 16]);
        let one_half_f: Tile<f32, { [16, 16] }> = constant(1.5f32, const_shape![16, 16]);
        let two_f: Tile<f32, { [16, 16] }> = constant(2.0f32, const_shape![16, 16]);
        let three_f: Tile<f32, { [16, 16] }> = constant(3.0f32, const_shape![16, 16]);
        let four_f: Tile<f32, { [16, 16] }> = constant(4.0f32, const_shape![16, 16]);
        let six_f: Tile<f32, { [16, 16] }> = constant(6.0f32, const_shape![16, 16]);
        let mut value = zero_f;
        value = select(eq_tile(magnitude, one), half_f, value);
        value = select(eq_tile(magnitude, two), one_f, value);
        value = select(eq_tile(magnitude, three), one_half_f, value);
        value = select(eq_tile(magnitude, four), two_f, value);
        value = select(eq_tile(magnitude, five), three_f, value);
        value = select(eq_tile(magnitude, six), four_f, value);
        value = select(eq_tile(magnitude, seven), six_f, value);
        select(eq_tile(sign, one), zero_f - value, value)
    }

    fn decode_fp8(
        encoded: Tile<i32, { [16, 16] }>,
        subnormal_scale: f32,
    ) -> Tile<f32, { [16, 16] }> {
        let shape = const_shape![16, 16];
        let byte_modulus: Tile<i32, { [16, 16] }> = constant(256i32, shape);
        let zero_i32: Tile<i32, { [16, 16] }> = constant(0i32, shape);
        let encoded = select(lt_tile(encoded, zero_i32), encoded + byte_modulus, encoded);
        let sign = encoded / constant(128i32, shape);
        let magnitude = encoded % constant(128i32, shape);
        let exponent = magnitude / constant(8i32, shape);
        let mantissa = magnitude % constant(8i32, shape);
        let exponent_f: Tile<f32, { [16, 16] }> = convert_tile(exponent);
        let mantissa_f: Tile<f32, { [16, 16] }> = convert_tile(mantissa);
        let one_f: Tile<f32, { [16, 16] }> = constant(1.0f32, shape);
        let eight_f: Tile<f32, { [16, 16] }> = constant(8.0f32, shape);
        let seven_f: Tile<f32, { [16, 16] }> = constant(7.0f32, shape);
        let subnormal_scale: Tile<f32, { [16, 16] }> = broadcast_scalar(subnormal_scale, shape);
        let normal =
            (one_f + true_div(mantissa_f, eight_f)) * exp2(exponent_f - seven_f, ftz::Disabled);
        let subnormal = mantissa_f * subnormal_scale;
        let magnitude = select(eq_tile(exponent, zero_i32), subnormal, normal);
        let zero_f: Tile<f32, { [16, 16] }> = constant(0.0f32, shape);
        let negative = zero_f - magnitude;
        let one_i32: Tile<i32, { [16, 16] }> = constant(1i32, shape);
        select(eq_tile(sign, one_i32), negative, magnitude)
    }

    fn decode_fp8_wide(
        encoded: Tile<i32, { [64, 16] }>,
        _subnormal_scale: f32,
    ) -> Tile<f32, { [64, 16] }> {
        let shape = const_shape![64, 16];
        const BYTE_MODULUS: i32 = 256;
        let byte_modulus = broadcast_scalar(BYTE_MODULUS, shape);
        const ZERO_I32: i32 = 0;
        let zero_i32 = broadcast_scalar(ZERO_I32, shape);
        let encoded = select(lt_tile(encoded, zero_i32), encoded + byte_modulus, encoded);
        const SIGN_DIVISOR: i32 = 128;
        let sign = encoded / broadcast_scalar(SIGN_DIVISOR, shape);
        let magnitude = encoded % broadcast_scalar(SIGN_DIVISOR, shape);
        const MANTISSA_DIVISOR: i32 = 8;
        let divisor_i32 = broadcast_scalar(MANTISSA_DIVISOR, shape);
        let exponent = magnitude / divisor_i32;
        let mantissa = magnitude % divisor_i32;

        // A normal E4M3 value has the same explicit mantissa bits as BF16.
        // Rebias its exponent from 7 to 127 and shift the three mantissa bits
        // into BF16's seven-bit mantissa field. This avoids an `exp2` per
        // weight while preserving the exact BF16 value consumed by MMA.
        const BF16_EXPONENT_REBIAS: i32 = 120;
        const BF16_EXPONENT_SHIFT: i32 = 128;
        const BF16_MANTISSA_SHIFT: i32 = 16;
        let normal_bits = (exponent + broadcast_scalar(BF16_EXPONENT_REBIAS, shape))
            * broadcast_scalar(BF16_EXPONENT_SHIFT, shape)
            + mantissa * broadcast_scalar(BF16_MANTISSA_SHIFT, shape);

        // E4M3 subnormals are m * 2^-9 for m in 0..=7. Their normalized BF16
        // encodings are small enough to select directly and exactly.
        const SUBNORMAL_1: i32 = 0x3b00;
        const SUBNORMAL_2: i32 = 0x3b80;
        const SUBNORMAL_3: i32 = 0x3bc0;
        const SUBNORMAL_4: i32 = 0x3c00;
        const SUBNORMAL_5: i32 = 0x3c20;
        const SUBNORMAL_6: i32 = 0x3c40;
        const SUBNORMAL_7: i32 = 0x3c60;
        let mut subnormal_bits = zero_i32;
        subnormal_bits = select(
            eq_tile(mantissa, broadcast_scalar(1i32, shape)),
            broadcast_scalar(SUBNORMAL_1, shape),
            subnormal_bits,
        );
        subnormal_bits = select(
            eq_tile(mantissa, broadcast_scalar(2i32, shape)),
            broadcast_scalar(SUBNORMAL_2, shape),
            subnormal_bits,
        );
        subnormal_bits = select(
            eq_tile(mantissa, broadcast_scalar(3i32, shape)),
            broadcast_scalar(SUBNORMAL_3, shape),
            subnormal_bits,
        );
        subnormal_bits = select(
            eq_tile(mantissa, broadcast_scalar(4i32, shape)),
            broadcast_scalar(SUBNORMAL_4, shape),
            subnormal_bits,
        );
        subnormal_bits = select(
            eq_tile(mantissa, broadcast_scalar(5i32, shape)),
            broadcast_scalar(SUBNORMAL_5, shape),
            subnormal_bits,
        );
        subnormal_bits = select(
            eq_tile(mantissa, broadcast_scalar(6i32, shape)),
            broadcast_scalar(SUBNORMAL_6, shape),
            subnormal_bits,
        );
        subnormal_bits = select(
            eq_tile(mantissa, broadcast_scalar(7i32, shape)),
            broadcast_scalar(SUBNORMAL_7, shape),
            subnormal_bits,
        );
        let magnitude_bits = select(eq_tile(exponent, zero_i32), subnormal_bits, normal_bits);
        const BF16_SIGN_BIT: i32 = 0x8000;
        let bits = magnitude_bits + sign * broadcast_scalar(BF16_SIGN_BIT, shape);
        let bits: Tile<u16, { [64, 16] }> = trunci(bits, overflow::NoUnsignedWrap);
        let decoded: Tile<bf16, { [64, 16] }> = bitcast(bits);
        convert_tile(decoded)
    }
}

use kernels::{
    fp8_w8a16, grouped_nvfp4_w4a16, grouped_nvfp4_w4a16_k32, grouped_nvfp4_w4a16_silu_mul,
    nvfp4_w4a16,
};

/// Scalar-scaled ModelOpt FP8 projection executed as W8A16 on SM80. The
/// checkpoint's E4M3 bytes remain FP8 on device and are decoded in the MMA
/// pipeline without changing the storage contract declared by the manifest.
pub(crate) struct Fp8W8A16Linear {
    input_size: usize,
    output_size: usize,
    weight: Arc<Tensor<u8>>,
    weight_scale: f32,
    device_bytes: usize,
}

impl Fp8W8A16Linear {
    pub(crate) fn load(
        source: &dyn WeightSource,
        prefix: &str,
        stream: &Arc<Stream>,
    ) -> Result<Self, ModelError> {
        let weight_name = format!("{prefix}.weight");
        let weight_scale_name = format!("{prefix}.weight_scale");
        let input_scale_name = format!("{prefix}.input_scale");
        let weight = source.tensor(&weight_name)?;
        let weight_scale = source.tensor(&weight_scale_name)?;
        let input_scale = source.tensor(&input_scale_name)?;
        if weight.dtype() != &WeightDtype::F8E4M3 || weight.shape().len() != 2 {
            return invalid_tensor(&weight_name, "expected rank-2 E4M3 storage");
        }
        let output_size = weight.shape()[0];
        let input_size = weight.shape()[1];
        if input_size == 0
            || output_size == 0
            || !input_size.is_multiple_of(TILE_N)
            || !output_size.is_multiple_of(DENSE_TILE_N)
            || weight.byte_len() != output_size.saturating_mul(input_size)
        {
            return invalid_tensor(&weight_name, "unrepresentable W8A16 storage geometry");
        }
        if weight_scale.dtype() != &WeightDtype::F32 || !weight_scale.shape().is_empty() {
            return invalid_tensor(&weight_scale_name, "expected an F32 scalar");
        }
        if input_scale.dtype() != &WeightDtype::F32 || !input_scale.shape().is_empty() {
            return invalid_tensor(&input_scale_name, "expected an F32 scalar");
        }
        let weight_scale = scalar_f32(weight_scale.bytes(), &weight_scale_name)?;
        if !weight_scale.is_finite() || weight_scale <= 0.0 {
            return invalid_tensor(&weight_scale_name, "scale must be finite and positive");
        }
        Self::from_host(
            weight.bytes(),
            input_size,
            output_size,
            weight_scale,
            stream,
        )
    }

    pub(super) fn from_host(
        encoded: &[u8],
        input_size: usize,
        output_size: usize,
        weight_scale: f32,
        stream: &Arc<Stream>,
    ) -> Result<Self, ModelError> {
        if encoded.len() != input_size.saturating_mul(output_size) {
            return invalid_tensor("fp8", "invalid W8A16 host artifact length");
        }
        let weight = api::copy_host_vec_to_device(&Arc::new(encoded.to_vec()))
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("upload FP8 weight: {error:?}")))?
            .reshape(&[output_size, input_size])
            .map_err(|error| ModelError::Cuda(format!("reshape FP8 weight: {error:?}")))?;
        let device_bytes = weight.num_bytes();
        Ok(Self {
            input_size,
            output_size,
            weight: Arc::new(weight),
            weight_scale,
            device_bytes,
        })
    }

    pub(crate) const fn input_size(&self) -> usize {
        self.input_size
    }

    pub(crate) const fn output_size(&self) -> usize {
        self.output_size
    }

    pub(crate) const fn device_bytes(&self) -> usize {
        self.device_bytes
    }

    pub(crate) fn enqueue(
        &self,
        input: Arc<Tensor<bf16>>,
        rows: usize,
        execution: &mut StreamExecution<'_>,
    ) -> Result<Tensor<bf16>, ModelError> {
        if rows == 0 || !rows.is_multiple_of(TILE_M) {
            return Err(ModelError::Cuda(format!(
                "FP8 W8A16 rows must be a positive multiple of {TILE_M}; got {rows}"
            )));
        }
        let expected_shape = [rows as i32, self.input_size as i32];
        if input.shape() != expected_shape {
            return Err(ModelError::Cuda(format!(
                "FP8 W8A16 input shape {:?}; expected {expected_shape:?}",
                input.shape()
            )));
        }
        let output = execution.enqueue(
            api::zeros::<bf16>(&[rows, self.output_size]),
            "allocate FP8 output",
        )?;
        self.enqueue_into(input, rows, output, execution)
    }

    pub(crate) fn enqueue_into(
        &self,
        input: Arc<Tensor<bf16>>,
        rows: usize,
        mut output: Tensor<bf16>,
        execution: &mut StreamExecution<'_>,
    ) -> Result<Tensor<bf16>, ModelError> {
        if rows == 0 || !rows.is_multiple_of(TILE_M) {
            return Err(ModelError::Cuda(format!(
                "FP8 W8A16 rows must be a positive multiple of {TILE_M}; got {rows}"
            )));
        }
        let expected_shape = [rows as i32, self.input_size as i32];
        if input.shape() != expected_shape
            || output.shape() != [rows as i32, self.output_size as i32]
        {
            return Err(ModelError::Cuda(format!(
                "invalid FP8 W8A16 input/output geometry: input={:?}, output={:?}",
                input.shape(),
                output.shape()
            )));
        }
        let logical_tiles = (rows / TILE_M)
            .checked_mul(self.output_size / DENSE_TILE_N)
            .ok_or_else(|| ModelError::Cuda("FP8 output tile count overflowed".into()))?;
        let workers = logical_tiles.min(device_sm_count(execution.stream())?);
        let output_partition = (&mut output).partition([TILE_M, DENSE_TILE_N]).map(
            [8, 1],
            u32::try_from(workers)
                .map_err(|_| ModelError::Cuda("persistent FP8 worker count overflowed".into()))?,
        );
        let (output_partition, ..) = execution.enqueue(
            fp8_w8a16(
                output_partition,
                input,
                self.weight.clone(),
                self.weight_scale,
                FP8_SUBNORMAL_SCALE,
            )
            .generics(vec![(self.input_size / TILE_N).to_string()]),
            "execute FP8 W8A16",
        )?;
        drop(output_partition);
        Ok(output)
    }
}

#[derive(Clone, Copy)]
pub(super) struct ParsedNvfp4Projection<'a> {
    pub(super) input_size: usize,
    pub(super) output_size: usize,
    pub(super) packed_weight: &'a [u8],
    pub(super) scale_bytes: &'a [u8],
    pub(super) weight_global_scale: f32,
}

/// Parses ModelOpt's four-tensor W4A16 representation. Geometry comes from
/// the checkpoint; these checks only establish that the bytes can be safely
/// interpreted by the kernel.
pub(super) fn parse_nvfp4_projection<'a>(
    source: &'a dyn WeightSource,
    prefix: &str,
) -> Result<ParsedNvfp4Projection<'a>, ModelError> {
    let weight_name = format!("{prefix}.weight");
    let scale_name = format!("{prefix}.weight_scale");
    let global_name = format!("{prefix}.weight_scale_2");
    let input_scale_name = format!("{prefix}.input_scale");
    let weight = source.tensor(&weight_name)?;
    let scale = source.tensor(&scale_name)?;
    let global = source.tensor(&global_name)?;
    let input_scale = source.tensor(&input_scale_name)?;

    if weight.dtype() != &WeightDtype::U8 || weight.shape().len() != 2 {
        return invalid_tensor(&weight_name, "expected rank-2 packed U8 storage");
    }
    let output_size = weight.shape()[0];
    let packed_input_size = weight.shape()[1];
    let input_size = packed_input_size
        .checked_mul(2)
        .ok_or_else(|| ModelError::InvalidTensor {
            name: weight_name.clone(),
            message: "logical input width overflowed".into(),
        })?;
    if input_size == 0
        || output_size == 0
        || !input_size.is_multiple_of(GROUP_K)
        || !output_size.is_multiple_of(DENSE_TILE_N)
        || weight.byte_len() != output_size.saturating_mul(packed_input_size)
    {
        return invalid_tensor(&weight_name, "unrepresentable W4A16 storage geometry");
    }
    let expected_scale_shape = [output_size, input_size / GROUP_K];
    if scale.dtype() != &WeightDtype::F8E4M3
        || scale.shape() != expected_scale_shape
        || scale.byte_len() != output_size.saturating_mul(input_size / GROUP_K)
    {
        return invalid_tensor(
            &scale_name,
            "scale storage does not cover packed weight groups",
        );
    }
    if global.dtype() != &WeightDtype::F32 || !global.shape().is_empty() {
        return invalid_tensor(&global_name, "expected an F32 scalar");
    }
    if input_scale.dtype() != &WeightDtype::F32 || !input_scale.shape().is_empty() {
        return invalid_tensor(&input_scale_name, "expected an F32 scalar");
    }
    let weight_global_scale = scalar_f32(global.bytes(), &global_name)?;
    let input_scale = scalar_f32(input_scale.bytes(), &input_scale_name)?;
    if !weight_global_scale.is_finite()
        || weight_global_scale <= 0.0
        || !input_scale.is_finite()
        || input_scale <= 0.0
    {
        return invalid_tensor(prefix, "quantization scales must be finite and positive");
    }
    Ok(ParsedNvfp4Projection {
        input_size,
        output_size,
        packed_weight: weight.bytes(),
        scale_bytes: scale.bytes(),
        weight_global_scale,
    })
}

/// Packed ModelOpt W4A16 projection selected for SM80.
pub(crate) struct Nvfp4W4A16Linear {
    input_size: usize,
    output_size: usize,
    packed_weight: Arc<Tensor<u8>>,
    weight_scale: Arc<Tensor<bf16>>,
    weight_global_scale: f32,
    device_bytes: usize,
}

impl Nvfp4W4A16Linear {
    pub(crate) fn load(
        source: &dyn WeightSource,
        prefix: &str,
        stream: &Arc<Stream>,
    ) -> Result<Self, ModelError> {
        let parsed = parse_nvfp4_projection(source, prefix)?;
        Self::from_host(
            parsed.input_size,
            parsed.output_size,
            parsed.packed_weight,
            parsed.scale_bytes,
            parsed.weight_global_scale,
            stream,
        )
    }

    pub(super) fn from_host(
        input_size: usize,
        output_size: usize,
        packed_weight: &[u8],
        scale_bytes: &[u8],
        weight_global_scale: f32,
        stream: &Arc<Stream>,
    ) -> Result<Self, ModelError> {
        let packed_weight = api::copy_host_vec_to_device(&Arc::new(packed_weight.to_vec()))
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("upload NVFP4 weight: {error:?}")))?
            .reshape(&[output_size, input_size / 2])
            .map_err(|error| ModelError::Cuda(format!("reshape NVFP4 weight: {error:?}")))?;
        let scales = scale_bytes
            .iter()
            .map(|bits| decode_e4m3fn(*bits))
            .map(|value| {
                if value.is_finite() {
                    Ok(bf16::from_f32(value))
                } else {
                    Err(ModelError::InvalidTensor {
                        name: "weight_scale".into(),
                        message: "contains E4M3 NaN".into(),
                    })
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let weight_scale = api::copy_host_vec_to_device(&Arc::new(scales))
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("upload NVFP4 scales: {error:?}")))?
            .reshape(&[output_size, input_size / GROUP_K])
            .map_err(|error| ModelError::Cuda(format!("reshape NVFP4 scales: {error:?}")))?;
        let device_bytes = packed_weight
            .num_bytes()
            .checked_add(weight_scale.num_bytes())
            .ok_or_else(|| ModelError::Cuda("NVFP4 device byte count overflowed".into()))?;
        Ok(Self {
            input_size,
            output_size,
            packed_weight: Arc::new(packed_weight),
            weight_scale: Arc::new(weight_scale),
            weight_global_scale,
            device_bytes,
        })
    }

    pub(crate) const fn input_size(&self) -> usize {
        self.input_size
    }

    pub(crate) const fn output_size(&self) -> usize {
        self.output_size
    }

    pub(crate) const fn device_bytes(&self) -> usize {
        self.device_bytes
    }

    pub(crate) fn enqueue(
        &self,
        input: Arc<Tensor<bf16>>,
        rows: usize,
        execution: &mut StreamExecution<'_>,
    ) -> Result<Tensor<bf16>, ModelError> {
        if rows == 0 || !rows.is_multiple_of(TILE_M) {
            return Err(ModelError::Cuda(format!(
                "NVFP4 W4A16 rows must be a positive multiple of {TILE_M}; got {rows}"
            )));
        }
        let expected_shape = [rows as i32, self.input_size as i32];
        if input.shape() != expected_shape {
            return Err(ModelError::Cuda(format!(
                "NVFP4 W4A16 input shape {:?}; expected {expected_shape:?}",
                input.shape()
            )));
        }
        let output = execution.enqueue(
            api::zeros::<bf16>(&[rows, self.output_size]),
            "allocate NVFP4 output",
        )?;
        self.enqueue_into(input, rows, output, execution)
    }

    pub(crate) fn enqueue_into(
        &self,
        input: Arc<Tensor<bf16>>,
        rows: usize,
        mut output: Tensor<bf16>,
        execution: &mut StreamExecution<'_>,
    ) -> Result<Tensor<bf16>, ModelError> {
        if rows == 0 || !rows.is_multiple_of(TILE_M) {
            return Err(ModelError::Cuda(format!(
                "NVFP4 W4A16 rows must be a positive multiple of {TILE_M}; got {rows}"
            )));
        }
        let expected_shape = [rows as i32, self.input_size as i32];
        if input.shape() != expected_shape
            || output.shape() != [rows as i32, self.output_size as i32]
        {
            return Err(ModelError::Cuda(format!(
                "invalid NVFP4 W4A16 input/output geometry: input={:?}, output={:?}",
                input.shape(),
                output.shape()
            )));
        }
        let logical_tiles = (rows / TILE_M)
            .checked_mul(self.output_size / DENSE_TILE_N)
            .ok_or_else(|| ModelError::Cuda("NVFP4 output tile count overflowed".into()))?;
        let workers = logical_tiles.min(device_sm_count(execution.stream())?);
        let output_partition = (&mut output).partition([TILE_M, DENSE_TILE_N]).map(
            [8, 1],
            u32::try_from(workers)
                .map_err(|_| ModelError::Cuda("persistent NVFP4 worker count overflowed".into()))?,
        );
        let (output_partition, ..) = execution.enqueue(
            nvfp4_w4a16(
                output_partition,
                input,
                self.packed_weight.clone(),
                self.weight_scale.clone(),
                self.weight_global_scale,
            )
            .generics(vec![(self.input_size / GROUP_K).to_string()]),
            "execute NVFP4 W4A16",
        )?;
        drop(output_partition);
        Ok(output)
    }
}

/// Packed expert bank for a single-launch grouped W4A16 GEMM.
pub(crate) struct GroupedNvfp4W4A16 {
    input_size: usize,
    output_size: usize,
    num_experts: usize,
    packed_weight: Arc<Tensor<u8>>,
    weight_scale: Arc<Tensor<u8>>,
    weight_global_scale: Arc<Tensor<f32>>,
    device_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GroupedCutileSchedule {
    K16,
    K32N64 { occupancy: i32 },
    K32N128 { occupancy: i32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertProjection {
    Gate,
    Up,
    Down,
}

impl ExpertProjection {
    pub(super) const fn suffix(self) -> &'static str {
        match self {
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }
}

impl GroupedNvfp4W4A16 {
    pub(crate) const fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub(crate) const fn output_size(&self) -> usize {
        self.output_size
    }

    /// Loads one projection from every individually named checkpoint expert
    /// into a single packed device bank. The temporary host staging vectors
    /// are exact-sized and released after upload; packed weights are never
    /// expanded to BF16.
    pub(crate) fn load(
        source: &dyn WeightSource,
        experts_prefix: &str,
        projection: ExpertProjection,
        num_experts: usize,
        stream: &Arc<Stream>,
    ) -> Result<Self, ModelError> {
        if num_experts == 0 {
            return invalid_tensor(experts_prefix, "expert bank is empty");
        }
        let first_prefix = format!("{experts_prefix}.0.{}", projection.suffix());
        let first = parse_nvfp4_projection(source, &first_prefix)?;
        let input_size = first.input_size;
        let output_size = first.output_size;
        let packed_len = num_experts
            .checked_mul(output_size)
            .and_then(|size| size.checked_mul(input_size / 2))
            .ok_or_else(|| ModelError::InvalidTensor {
                name: experts_prefix.into(),
                message: "grouped packed-weight size overflowed".into(),
            })?;
        let scale_len = num_experts
            .checked_mul(output_size)
            .and_then(|size| size.checked_mul(input_size / GROUP_K))
            .ok_or_else(|| ModelError::InvalidTensor {
                name: experts_prefix.into(),
                message: "grouped scale size overflowed".into(),
            })?;
        let mut packed_weight = Vec::with_capacity(packed_len);
        let mut scale_bytes = Vec::with_capacity(scale_len);
        let mut global_scales = Vec::with_capacity(num_experts);

        for expert in 0..num_experts {
            let prefix = format!("{experts_prefix}.{expert}.{}", projection.suffix());
            let parsed = if expert == 0 {
                first
            } else {
                parse_nvfp4_projection(source, &prefix)?
            };
            if parsed.input_size != input_size || parsed.output_size != output_size {
                return invalid_tensor(&prefix, "expert projection geometry differs from expert 0");
            }
            packed_weight.extend_from_slice(parsed.packed_weight);
            scale_bytes.extend_from_slice(parsed.scale_bytes);
            global_scales.push(parsed.weight_global_scale);
        }
        debug_assert_eq!(packed_weight.len(), packed_len);
        debug_assert_eq!(scale_bytes.len(), scale_len);
        Self::from_host_owned(
            num_experts,
            input_size,
            output_size,
            packed_weight,
            scale_bytes,
            global_scales,
            stream,
        )
    }

    pub(super) fn from_host_owned(
        num_experts: usize,
        input_size: usize,
        output_size: usize,
        packed_weight: Vec<u8>,
        scale_bytes: Vec<u8>,
        weight_global_scale: Vec<f32>,
        stream: &Arc<Stream>,
    ) -> Result<Self, ModelError> {
        let expected_weights = num_experts
            .checked_mul(output_size)
            .and_then(|size| size.checked_mul(input_size / 2));
        let expected_scales = num_experts
            .checked_mul(output_size)
            .and_then(|size| size.checked_mul(input_size / GROUP_K));
        if num_experts == 0
            || input_size == 0
            || output_size == 0
            || !input_size.is_multiple_of(GROUPED_TILE_K)
            || !output_size.is_multiple_of(GROUPED_TILE_N)
            || expected_weights != Some(packed_weight.len())
            || expected_scales != Some(scale_bytes.len())
            || weight_global_scale.len() != num_experts
            || weight_global_scale
                .iter()
                .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            return invalid_tensor("grouped_nvfp4", "invalid grouped W4A16 artifact");
        }
        let device_bytes = packed_weight
            .len()
            .checked_add(scale_bytes.len())
            .and_then(|bytes| {
                bytes.checked_add(weight_global_scale.len() * std::mem::size_of::<f32>())
            })
            .ok_or_else(|| ModelError::Cuda("grouped NVFP4 byte count overflowed".into()))?;

        let packed_weight = api::copy_host_vec_to_device(&Arc::new(packed_weight))
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("upload grouped NVFP4 weight: {error:?}")))?
            .reshape(&[num_experts, output_size, input_size / 2])
            .map_err(|error| {
                ModelError::Cuda(format!("reshape grouped NVFP4 weight: {error:?}"))
            })?;
        if scale_bytes
            .iter()
            .any(|bits| !decode_e4m3fn(*bits).is_finite())
        {
            return invalid_tensor("grouped_nvfp4.weight_scale", "contains E4M3 NaN");
        }
        let weight_scale = api::copy_host_vec_to_device(&Arc::new(scale_bytes))
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("upload grouped NVFP4 scales: {error:?}")))?
            .reshape(&[num_experts, output_size, input_size / GROUP_K])
            .map_err(|error| {
                ModelError::Cuda(format!("reshape grouped NVFP4 scales: {error:?}"))
            })?;
        let weight_global_scale = api::copy_host_vec_to_device(&Arc::new(weight_global_scale))
            .sync_on(stream)
            .map_err(|error| {
                ModelError::Cuda(format!("upload grouped NVFP4 global scales: {error:?}"))
            })?;
        Ok(Self {
            input_size,
            output_size,
            num_experts,
            packed_weight: Arc::new(packed_weight),
            weight_scale: Arc::new(weight_scale),
            weight_global_scale: Arc::new(weight_global_scale),
            device_bytes,
        })
    }

    pub(crate) const fn device_bytes(&self) -> usize {
        self.device_bytes
    }

    /// Host-supplied expert maps use the small 16-row dispatch geometry. The
    /// production device-resident path additionally supports 64-row prefill
    /// tiles.
    pub(crate) fn enqueue(
        &self,
        dispatched: Arc<Tensor<bf16>>,
        rows: usize,
        expert_by_row_tile: &[i32],
        execution: &mut StreamExecution<'_>,
    ) -> Result<Tensor<bf16>, ModelError> {
        if rows == 0
            || !rows.is_multiple_of(TILE_M)
            || expert_by_row_tile.len() != rows / TILE_M
            || expert_by_row_tile
                .iter()
                .any(|expert| *expert < 0 || *expert as usize >= self.num_experts)
        {
            return Err(ModelError::Cuda(
                "invalid grouped NVFP4 dispatch row/expert map".into(),
            ));
        }
        let expected_shape = [rows as i32, self.input_size as i32];
        if dispatched.shape() != expected_shape {
            return Err(ModelError::Cuda(format!(
                "grouped NVFP4 input shape {:?}; expected {expected_shape:?}",
                dispatched.shape()
            )));
        }
        let expert_by_row_tile =
            api::copy_host_vec_to_device(&Arc::new(expert_by_row_tile.to_vec()))
                .sync_on(execution.stream())
                .map_err(|error| {
                    ModelError::Cuda(format!("upload grouped expert map: {error:?}"))
                })?;
        let mut output = execution.enqueue(
            api::zeros::<bf16>(&[rows, self.output_size]),
            "allocate grouped output",
        )?;
        let logical_tiles = (rows / TILE_M)
            .checked_mul(self.output_size / GROUPED_TILE_N)
            .ok_or_else(|| ModelError::Cuda("grouped output tile count overflowed".into()))?;
        let workers = logical_tiles.min(device_sm_count(execution.stream())?);
        let output_partition = (&mut output).partition([TILE_M, GROUPED_TILE_N]).map(
            [8, 1],
            u32::try_from(workers).map_err(|_| {
                ModelError::Cuda("persistent grouped worker count overflowed u32".into())
            })?,
        );
        execution.enqueue(
            grouped_nvfp4_w4a16(
                output_partition,
                dispatched,
                Arc::new(expert_by_row_tile),
                self.packed_weight.clone(),
                self.weight_scale.clone(),
                self.weight_global_scale.clone(),
            )
            .generics(vec![
                (self.input_size / GROUPED_TILE_K).to_string(),
                GROUPED_SMALL_TILE_M.to_string(),
                self.output_size.to_string(),
                (self.input_size / 2).to_string(),
                (self.input_size / GROUP_K).to_string(),
            ]),
            "execute grouped NVFP4 W4A16",
        )?;
        Ok(output)
    }

    pub(crate) fn enqueue_device_plan_into(
        &self,
        dispatched: Arc<Tensor<bf16>>,
        rows: usize,
        expert_by_row_tile: Arc<Tensor<i32>>,
        output: Tensor<bf16>,
        execution: &mut StreamExecution<'_>,
    ) -> Result<Tensor<bf16>, ModelError> {
        self.enqueue_device_plan_into_with_schedule(
            dispatched,
            rows,
            expert_by_row_tile,
            output,
            GroupedCutileSchedule::K16,
            execution,
        )
    }

    pub(super) fn enqueue_device_plan_into_with_schedule(
        &self,
        dispatched: Arc<Tensor<bf16>>,
        rows: usize,
        expert_by_row_tile: Arc<Tensor<i32>>,
        mut output: Tensor<bf16>,
        schedule: GroupedCutileSchedule,
        execution: &mut StreamExecution<'_>,
    ) -> Result<Tensor<bf16>, ModelError> {
        let map_rows = expert_by_row_tile
            .shape()
            .first()
            .copied()
            .and_then(|rows| usize::try_from(rows).ok())
            .ok_or_else(|| ModelError::Cuda("invalid grouped expert row-map shape".into()))?;
        let tile_rows = grouped_tile_rows(rows, map_rows)?;
        if rows == 0
            || dispatched.shape() != [rows as i32, self.input_size as i32]
            || expert_by_row_tile.shape() != [map_rows as i32]
            || output.shape() != [rows as i32, self.output_size as i32]
        {
            return Err(ModelError::Cuda(
                "invalid device-resident grouped NVFP4 dispatch/output plan".into(),
            ));
        }
        let tile_columns = match schedule {
            GroupedCutileSchedule::K16 | GroupedCutileSchedule::K32N64 { .. } => GROUPED_TILE_N,
            GroupedCutileSchedule::K32N128 { .. } => 128,
        };
        if !self.output_size.is_multiple_of(tile_columns) {
            return Err(ModelError::Cuda(format!(
                "grouped NVFP4 output size {} is not divisible by schedule tile width {tile_columns}",
                self.output_size
            )));
        }
        let logical_tiles = map_rows
            .checked_mul(self.output_size / tile_columns)
            .ok_or_else(|| ModelError::Cuda("grouped output tile count overflowed".into()))?;
        let target_occupancy = match schedule {
            GroupedCutileSchedule::K16 => 1usize,
            GroupedCutileSchedule::K32N64 { occupancy }
            | GroupedCutileSchedule::K32N128 { occupancy } => usize::try_from(occupancy)
                .ok()
                .filter(|occupancy| *occupancy > 0)
                .ok_or_else(|| {
                    ModelError::Cuda("grouped cuTile occupancy must be positive".into())
                })?,
        };
        // The compiler occupancy hint controls resources per CTA; it does not
        // launch the matching number of CTAs. Follow cuTile's static-persistent
        // schedule and launch up to `SMs * occupancy` workers so every intended
        // resident slot has work when the logical grid is large enough.
        let worker_capacity = device_sm_count(execution.stream())?
            .checked_mul(target_occupancy)
            .ok_or_else(|| ModelError::Cuda("grouped worker capacity overflowed".into()))?;
        let workers = logical_tiles.min(worker_capacity);
        let workers = u32::try_from(workers).map_err(|_| {
            ModelError::Cuda("persistent grouped worker count overflowed u32".into())
        })?;
        if let GroupedCutileSchedule::K32N64 { occupancy } = schedule {
            let options = CompileOptions::default()
                .occupancy(occupancy)
                .num_worker_warps_per_cta(8)
                .max_divisibility(16);
            if tile_rows == GROUPED_LARGE_TILE_M {
                let output_partition = (&mut output)
                    .partition([GROUPED_LARGE_TILE_M, GROUPED_TILE_N])
                    .map([8, 1], workers);
                let (output_partition, ..) = execution.enqueue(
                    grouped_nvfp4_w4a16_k32(
                        output_partition,
                        dispatched,
                        expert_by_row_tile,
                        self.packed_weight.clone(),
                        self.weight_scale.clone(),
                        self.weight_global_scale.clone(),
                    )
                    .generics(vec![
                        (self.input_size / 32).to_string(),
                        GROUPED_LARGE_TILE_M.to_string(),
                        GROUPED_TILE_N.to_string(),
                        self.output_size.to_string(),
                        (self.input_size / 2).to_string(),
                        (self.input_size / GROUP_K).to_string(),
                    ])
                    .compile_options(options),
                    "execute large-tile K32 grouped NVFP4 W4A16",
                )?;
                drop(output_partition);
            } else {
                let output_partition = (&mut output)
                    .partition([GROUPED_SMALL_TILE_M, GROUPED_TILE_N])
                    .map([8, 1], workers);
                let (output_partition, ..) = execution.enqueue(
                    grouped_nvfp4_w4a16_k32(
                        output_partition,
                        dispatched,
                        expert_by_row_tile,
                        self.packed_weight.clone(),
                        self.weight_scale.clone(),
                        self.weight_global_scale.clone(),
                    )
                    .generics(vec![
                        (self.input_size / 32).to_string(),
                        GROUPED_SMALL_TILE_M.to_string(),
                        GROUPED_TILE_N.to_string(),
                        self.output_size.to_string(),
                        (self.input_size / 2).to_string(),
                        (self.input_size / GROUP_K).to_string(),
                    ])
                    .compile_options(options),
                    "execute small-tile K32 grouped NVFP4 W4A16",
                )?;
                drop(output_partition);
            }
        } else if let GroupedCutileSchedule::K32N128 { occupancy } = schedule {
            let options = CompileOptions::default()
                .occupancy(occupancy)
                .num_worker_warps_per_cta(8)
                .max_divisibility(16);
            if tile_rows == GROUPED_LARGE_TILE_M {
                let output_partition = (&mut output)
                    .partition([GROUPED_LARGE_TILE_M, 128])
                    .map([8, 1], workers);
                let (output_partition, ..) = execution.enqueue(
                    grouped_nvfp4_w4a16_k32(
                        output_partition,
                        dispatched,
                        expert_by_row_tile,
                        self.packed_weight.clone(),
                        self.weight_scale.clone(),
                        self.weight_global_scale.clone(),
                    )
                    .generics(vec![
                        (self.input_size / 32).to_string(),
                        GROUPED_LARGE_TILE_M.to_string(),
                        128.to_string(),
                        self.output_size.to_string(),
                        (self.input_size / 2).to_string(),
                        (self.input_size / GROUP_K).to_string(),
                    ])
                    .compile_options(options),
                    "execute large-tile N128 K32 grouped NVFP4 W4A16",
                )?;
                drop(output_partition);
            } else {
                let output_partition = (&mut output)
                    .partition([GROUPED_SMALL_TILE_M, 128])
                    .map([8, 1], workers);
                let (output_partition, ..) = execution.enqueue(
                    grouped_nvfp4_w4a16_k32(
                        output_partition,
                        dispatched,
                        expert_by_row_tile,
                        self.packed_weight.clone(),
                        self.weight_scale.clone(),
                        self.weight_global_scale.clone(),
                    )
                    .generics(vec![
                        (self.input_size / 32).to_string(),
                        GROUPED_SMALL_TILE_M.to_string(),
                        128.to_string(),
                        self.output_size.to_string(),
                        (self.input_size / 2).to_string(),
                        (self.input_size / GROUP_K).to_string(),
                    ])
                    .compile_options(options),
                    "execute small-tile N128 K32 grouped NVFP4 W4A16",
                )?;
                drop(output_partition);
            }
        } else if tile_rows == GROUPED_LARGE_TILE_M {
            let output_partition = (&mut output)
                .partition([GROUPED_LARGE_TILE_M, GROUPED_TILE_N])
                .map([8, 1], workers);
            let (output_partition, ..) = execution.enqueue(
                grouped_nvfp4_w4a16(
                    output_partition,
                    dispatched,
                    expert_by_row_tile,
                    self.packed_weight.clone(),
                    self.weight_scale.clone(),
                    self.weight_global_scale.clone(),
                )
                .generics(vec![
                    (self.input_size / GROUPED_TILE_K).to_string(),
                    GROUPED_LARGE_TILE_M.to_string(),
                    self.output_size.to_string(),
                    (self.input_size / 2).to_string(),
                    (self.input_size / GROUP_K).to_string(),
                ]),
                "execute large-tile grouped NVFP4 W4A16",
            )?;
            drop(output_partition);
        } else {
            let output_partition = (&mut output)
                .partition([GROUPED_SMALL_TILE_M, GROUPED_TILE_N])
                .map([8, 1], workers);
            let (output_partition, ..) = execution.enqueue(
                grouped_nvfp4_w4a16(
                    output_partition,
                    dispatched,
                    expert_by_row_tile,
                    self.packed_weight.clone(),
                    self.weight_scale.clone(),
                    self.weight_global_scale.clone(),
                )
                .generics(vec![
                    (self.input_size / GROUPED_TILE_K).to_string(),
                    GROUPED_SMALL_TILE_M.to_string(),
                    self.output_size.to_string(),
                    (self.input_size / 2).to_string(),
                    (self.input_size / GROUP_K).to_string(),
                ]),
                "execute small-tile grouped NVFP4 W4A16",
            )?;
            drop(output_partition);
        }
        Ok(output)
    }

    /// Executes gate and up expert banks together and writes
    /// `silu(gate) * up` directly into `output`.
    pub(crate) fn enqueue_silu_mul_device_plan_into(
        &self,
        up: &Self,
        dispatched: Arc<Tensor<bf16>>,
        rows: usize,
        expert_by_row_tile: Arc<Tensor<i32>>,
        mut output: Tensor<bf16>,
        execution: &mut StreamExecution<'_>,
    ) -> Result<Tensor<bf16>, ModelError> {
        if self.input_size != up.input_size
            || self.output_size != up.output_size
            || self.num_experts != up.num_experts
        {
            return Err(ModelError::Cuda(
                "fused grouped NVFP4 gate/up banks have incompatible geometry".into(),
            ));
        }
        let map_rows = expert_by_row_tile
            .shape()
            .first()
            .copied()
            .and_then(|rows| usize::try_from(rows).ok())
            .ok_or_else(|| ModelError::Cuda("invalid fused expert row-map shape".into()))?;
        let tile_rows = grouped_tile_rows(rows, map_rows)?;
        if rows == 0
            || dispatched.shape() != [rows as i32, self.input_size as i32]
            || expert_by_row_tile.shape() != [map_rows as i32]
            || output.shape() != [rows as i32, self.output_size as i32]
        {
            return Err(ModelError::Cuda(
                "invalid fused grouped NVFP4 gate/up dispatch/output plan".into(),
            ));
        }
        let logical_tiles = map_rows
            .checked_mul(self.output_size / GROUPED_TILE_N)
            .ok_or_else(|| ModelError::Cuda("grouped output tile count overflowed".into()))?;
        let workers = logical_tiles.min(device_sm_count(execution.stream())?);
        let workers = u32::try_from(workers).map_err(|_| {
            ModelError::Cuda("persistent grouped worker count overflowed u32".into())
        })?;
        if tile_rows == GROUPED_LARGE_TILE_M {
            let output_partition = (&mut output)
                .partition([GROUPED_LARGE_TILE_M, GROUPED_TILE_N])
                .map([8, 1], workers);
            let (output_partition, ..) = execution.enqueue(
                grouped_nvfp4_w4a16_silu_mul(
                    output_partition,
                    dispatched,
                    expert_by_row_tile,
                    self.packed_weight.clone(),
                    self.weight_scale.clone(),
                    self.weight_global_scale.clone(),
                    up.packed_weight.clone(),
                    up.weight_scale.clone(),
                    up.weight_global_scale.clone(),
                )
                .generics(vec![
                    (self.input_size / GROUPED_TILE_K).to_string(),
                    GROUPED_LARGE_TILE_M.to_string(),
                    self.output_size.to_string(),
                    (self.input_size / 2).to_string(),
                    (self.input_size / GROUP_K).to_string(),
                ]),
                "execute large-tile fused grouped NVFP4 gate/up SwiGLU",
            )?;
            drop(output_partition);
        } else {
            let output_partition = (&mut output)
                .partition([GROUPED_SMALL_TILE_M, GROUPED_TILE_N])
                .map([8, 1], workers);
            let (output_partition, ..) = execution.enqueue(
                grouped_nvfp4_w4a16_silu_mul(
                    output_partition,
                    dispatched,
                    expert_by_row_tile,
                    self.packed_weight.clone(),
                    self.weight_scale.clone(),
                    self.weight_global_scale.clone(),
                    up.packed_weight.clone(),
                    up.weight_scale.clone(),
                    up.weight_global_scale.clone(),
                )
                .generics(vec![
                    (self.input_size / GROUPED_TILE_K).to_string(),
                    GROUPED_SMALL_TILE_M.to_string(),
                    self.output_size.to_string(),
                    (self.input_size / 2).to_string(),
                    (self.input_size / GROUP_K).to_string(),
                ]),
                "execute small-tile fused grouped NVFP4 gate/up SwiGLU",
            )?;
            drop(output_partition);
        }
        Ok(output)
    }
}

fn grouped_tile_rows(rows: usize, map_rows: usize) -> Result<usize, ModelError> {
    let tile_rows = rows
        .checked_div(map_rows)
        .filter(|_| rows % map_rows == 0)
        .ok_or_else(|| ModelError::Cuda("invalid grouped expert row-map geometry".into()))?;
    if matches!(tile_rows, GROUPED_SMALL_TILE_M | GROUPED_LARGE_TILE_M) {
        Ok(tile_rows)
    } else {
        Err(ModelError::Cuda(format!(
            "unsupported grouped row tile {tile_rows}"
        )))
    }
}

fn device_sm_count(stream: &Arc<Stream>) -> Result<usize, ModelError> {
    let mut count = 0i32;
    unsafe {
        cuda_core::sys::cuDeviceGetAttribute(
            &mut count,
            cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            stream.device().cu_device(),
        )
        .result()
        .map_err(|error| ModelError::Cuda(format!("query CUDA SM count: {error:?}")))?;
    }
    usize::try_from(count)
        .ok()
        .filter(|count| *count > 0)
        .ok_or_else(|| ModelError::Cuda(format!("CUDA reported invalid SM count {count}")))
}

fn scalar_f32(bytes: &[u8], name: &str) -> Result<f32, ModelError> {
    let bytes: [u8; 4] = bytes.try_into().map_err(|_| ModelError::InvalidTensor {
        name: name.into(),
        message: "scalar F32 storage is not four bytes".into(),
    })?;
    Ok(f32::from_le_bytes(bytes))
}

fn invalid_tensor<T>(name: &str, message: &str) -> Result<T, ModelError> {
    Err(ModelError::InvalidTensor {
        name: name.into(),
        message: message.into(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct QuantizedLinearProbe {
    pub(crate) fp8_max_abs_error: f32,
    pub(crate) max_abs_error: f32,
    pub(crate) grouped_max_abs_error: f32,
}

pub(crate) fn probe_quantized_linears(
    stream: &Arc<Stream>,
) -> Result<QuantizedLinearProbe, ModelError> {
    let fp8_max_abs_error = probe_fp8_w8a16(stream)?;
    let (max_abs_error, grouped_max_abs_error) = probe_nvfp4_w4a16(stream)?;
    Ok(QuantizedLinearProbe {
        fp8_max_abs_error,
        max_abs_error,
        grouped_max_abs_error,
    })
}

fn probe_fp8_w8a16(stream: &Arc<Stream>) -> Result<f32, ModelError> {
    let input_size = TILE_N;
    let output_size = DENSE_TILE_N;
    let encoded = (0..output_size * input_size)
        .map(|index| {
            const VALUES: [u8; 12] = [
                0x00, 0x01, 0x20, 0x38, 0x3c, 0x40, 0x60, 0x7e, 0x80, 0xb8, 0xc0, 0xfe,
            ];
            VALUES[index % VALUES.len()]
        })
        .collect::<Vec<_>>();
    let weight_scale = 0.25f32;
    let source = Fp8ProbeSource {
        prefix: "probe",
        input_size,
        output_size,
        encoded: &encoded,
        weight_scale: weight_scale.to_le_bytes(),
        input_scale: 1.0f32.to_le_bytes(),
    };
    let linear = Fp8W8A16Linear::load(&source, "probe", stream)?;
    let mut execution = StreamExecution::new(stream);
    if linear.input_size() != input_size
        || linear.output_size() != output_size
        || linear.device_bytes() != encoded.len()
    {
        return Err(ModelError::Cuda(
            "FP8 loader did not preserve geometry/device-byte accounting".into(),
        ));
    }
    let input_host = (0..TILE_M * input_size)
        .map(|index| bf16::from_f32((index % 7) as f32 - 3.0))
        .collect::<Vec<_>>();
    let input = api::copy_host_vec_to_device(&Arc::new(input_host.clone()))
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("upload FP8 probe input: {error:?}")))?
        .reshape(&[TILE_M, input_size])
        .map_err(|error| ModelError::Cuda(format!("reshape FP8 probe input: {error:?}")))?;
    let output = linear.enqueue(Arc::new(input), TILE_M, &mut execution)?;
    let actual: Vec<bf16> = output
        .to_host_vec()
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("copy FP8 probe output: {error:?}")))?;
    execution.mark_synchronized();
    let mut max_abs_error = 0.0f32;
    for row in 0..TILE_M {
        for column in 0..output_size {
            let mut expected = 0.0f32;
            for k in 0..input_size {
                let weight =
                    bf16::from_f32(decode_e4m3fn(encoded[column * input_size + k]) * weight_scale)
                        .to_f32();
                expected += input_host[row * input_size + k].to_f32() * weight;
            }
            let expected = bf16::from_f32(expected).to_f32();
            let actual = actual[row * output_size + column].to_f32();
            let error = (actual - expected).abs();
            max_abs_error = max_abs_error.max(error);
            if !actual.is_finite() || error > 0.25 {
                return Err(ModelError::Cuda(format!(
                    "FP8 differential mismatch at ({row}, {column}): {actual} != {expected}"
                )));
            }
        }
    }
    Ok(max_abs_error)
}

/// Probes the production W4A16 kernels with every FP4 code, non-unit E4M3
/// scales, a non-unit global scale, and signed BF16 activations.
fn probe_nvfp4_w4a16(stream: &Arc<Stream>) -> Result<(f32, f32), ModelError> {
    let input_size = GROUP_K;
    let output_size = DENSE_TILE_N;
    let packed = (0..output_size * input_size / 2)
        .map(|index| {
            let low = (index * 2 % 16) as u8;
            let high = ((index * 2 + 1) % 16) as u8;
            low | (high << 4)
        })
        .collect::<Vec<_>>();
    let scale_bytes = (0..output_size)
        .map(|index| if index % 2 == 0 { 0x38 } else { 0x40 })
        .collect::<Vec<_>>();
    let global_scale = 0.5f32;
    let source = Nvfp4ValidationSource {
        prefix: "probe",
        input_size,
        output_size,
        packed: &packed,
        scales: &scale_bytes,
        global_scale: global_scale.to_le_bytes(),
        input_scale: 1.0f32.to_le_bytes(),
    };
    let linear = Nvfp4W4A16Linear::load(&source, "probe", stream)?;
    let mut execution = StreamExecution::new(stream);
    let expected_device_bytes = packed
        .len()
        .checked_add(scale_bytes.len() * std::mem::size_of::<bf16>())
        .ok_or_else(|| ModelError::Cuda("validation byte count overflowed".into()))?;
    if linear.input_size() != input_size
        || linear.output_size() != output_size
        || linear.device_bytes() != expected_device_bytes
    {
        return Err(ModelError::Cuda(
            "NVFP4 loader did not preserve geometry/device-byte accounting".into(),
        ));
    }
    let input_host = (0..TILE_M * input_size)
        .map(|index| bf16::from_f32((index % 7) as f32 - 3.0))
        .collect::<Vec<_>>();
    let input = api::copy_host_vec_to_device(&Arc::new(input_host.clone()))
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("upload NVFP4 test input: {error:?}")))?
        .reshape(&[TILE_M, input_size])
        .map_err(|error| ModelError::Cuda(format!("reshape NVFP4 test input: {error:?}")))?;
    let output = Arc::new(linear.enqueue(Arc::new(input), TILE_M, &mut execution)?);
    let actual_values: Vec<bf16> = output
        .to_host_vec()
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("copy NVFP4 test output: {error:?}")))?;
    execution.mark_synchronized();

    let mut max_abs_error = 0.0f32;
    for row in 0..TILE_M {
        for column in 0..output_size {
            let scale = decode_e4m3fn(scale_bytes[column]) * global_scale;
            let mut expected = 0.0f32;
            for k in 0..input_size {
                let byte = packed[column * (input_size / 2) + k / 2];
                let nibble = if k % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                let weight = bf16::from_f32(decode_e2m1(nibble) * scale).to_f32();
                expected += input_host[row * input_size + k].to_f32() * weight;
            }
            let expected = bf16::from_f32(expected).to_f32();
            let actual = actual_values[row * output_size + column].to_f32();
            max_abs_error = max_abs_error.max((actual - expected).abs());
            if !actual.is_finite() || (actual - expected).abs() > 0.25 {
                let actual_row = actual_values[row * output_size..(row + 1) * output_size]
                    .iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<_>>();
                return Err(ModelError::Cuda(format!(
                    "NVFP4 differential mismatch at ({row}, {column}): {actual} != {expected}; actual row {actual_row:?}"
                )));
            }
        }
    }
    let grouped_max_abs_error = probe_grouped_nvfp4_w4a16(stream)?;
    Ok((max_abs_error, grouped_max_abs_error))
}

fn probe_grouped_nvfp4_w4a16(stream: &Arc<Stream>) -> Result<f32, ModelError> {
    const EXPERTS: usize = 2;
    let input_size = GROUPED_TILE_K;
    let output_size = GROUPED_TILE_N;
    let packed = (0..EXPERTS * output_size * input_size / 2)
        .map(|index| {
            let expert = index / (output_size * input_size / 2);
            let low = ((index * 2 + expert) % 16) as u8;
            let high = ((index * 2 + 1 + expert) % 16) as u8;
            low | (high << 4)
        })
        .collect::<Vec<_>>();
    let scale_bytes = (0..EXPERTS * output_size * (input_size / GROUP_K))
        .map(|index| if index % 3 == 0 { 0x38 } else { 0x40 })
        .collect::<Vec<_>>();
    let global_scales = [0.5f32, 0.25f32];
    let global_scale_bytes = global_scales.map(f32::to_le_bytes);
    let input_scale_bytes = [1.0f32.to_le_bytes(); EXPERTS];
    let source = GroupedNvfp4ValidationSource {
        prefix: "experts",
        input_size,
        output_size,
        packed: &packed,
        scales: &scale_bytes,
        global_scales: &global_scale_bytes,
        input_scales: &input_scale_bytes,
    };
    let grouped =
        GroupedNvfp4W4A16::load(&source, "experts", ExpertProjection::Gate, EXPERTS, stream)?;
    let up_packed = packed
        .iter()
        .enumerate()
        .map(|(index, byte)| byte.rotate_left((index % 7) as u32))
        .collect::<Vec<_>>();
    let up_scale_bytes = scale_bytes
        .iter()
        .map(|scale| if *scale == 0x38 { 0x40 } else { 0x38 })
        .collect::<Vec<_>>();
    let up_global_scales = [0.125f32, 0.75f32];
    let up_global_scale_bytes = up_global_scales.map(f32::to_le_bytes);
    let up_source = GroupedNvfp4ValidationSource {
        prefix: "experts",
        input_size,
        output_size,
        packed: &up_packed,
        scales: &up_scale_bytes,
        global_scales: &up_global_scale_bytes,
        input_scales: &input_scale_bytes,
    };
    let up_grouped =
        GroupedNvfp4W4A16::load(&up_source, "experts", ExpertProjection::Up, EXPERTS, stream)?;
    let mut execution = StreamExecution::new(stream);
    for projection in [ExpertProjection::Up, ExpertProjection::Down] {
        let parsed = GroupedNvfp4W4A16::load(&source, "experts", projection, EXPERTS, stream)?;
        if parsed.device_bytes() != grouped.device_bytes() {
            return Err(ModelError::Cuda(
                "grouped projection parser produced inconsistent storage".into(),
            ));
        }
    }
    let expected_device_bytes = packed
        .len()
        .checked_add(scale_bytes.len())
        .and_then(|bytes| bytes.checked_add(global_scales.len() * std::mem::size_of::<f32>()))
        .ok_or_else(|| ModelError::Cuda("grouped validation byte count overflowed".into()))?;
    if grouped.device_bytes() != expected_device_bytes {
        return Err(ModelError::Cuda(format!(
            "grouped NVFP4 device-byte accounting mismatch: got {}, expected {expected_device_bytes}",
            grouped.device_bytes(),
        )));
    }
    let rows = GROUPED_LARGE_TILE_M * EXPERTS;
    let input_host = (0..rows * input_size)
        .map(|index| bf16::from_f32((index % 9) as f32 - 4.0))
        .collect::<Vec<_>>();
    let input = api::copy_host_vec_to_device(&Arc::new(input_host.clone()))
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("upload grouped test input: {error:?}")))?
        .reshape(&[rows, input_size])
        .map_err(|error| ModelError::Cuda(format!("reshape grouped test input: {error:?}")))?;
    let expert_by_row_tile = [1i32, 0i32];
    let input = Arc::new(input);
    let expert_by_row_tile_device =
        api::copy_host_vec_to_device(&Arc::new(expert_by_row_tile.to_vec()))
            .sync_on(stream)
            .map_err(|error| {
                ModelError::Cuda(format!("upload fused grouped expert map: {error:?}"))
            })?;
    let expert_by_row_tile_device = Arc::new(expert_by_row_tile_device);
    let output_buffer = execution.enqueue(
        api::zeros::<bf16>(&[rows, output_size]),
        "allocate large-tile grouped probe output",
    )?;
    let output = Arc::new(grouped.enqueue_device_plan_into(
        input.clone(),
        rows,
        expert_by_row_tile_device.clone(),
        output_buffer,
        &mut execution,
    )?);
    let fused_output = execution.enqueue(
        api::zeros::<bf16>(&[rows, output_size]),
        "allocate fused grouped probe output",
    )?;
    let fused_output = Arc::new(grouped.enqueue_silu_mul_device_plan_into(
        &up_grouped,
        input,
        rows,
        expert_by_row_tile_device,
        fused_output,
        &mut execution,
    )?);
    let actual: Vec<bf16> = output
        .to_host_vec()
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("copy grouped test output: {error:?}")))?;
    let fused_actual: Vec<bf16> = fused_output
        .to_host_vec()
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("copy fused grouped test output: {error:?}")))?;
    execution.mark_synchronized();

    let mut max_abs_error = 0.0f32;
    for row in 0..rows {
        let expert = expert_by_row_tile[row / GROUPED_LARGE_TILE_M] as usize;
        for column in 0..output_size {
            let scale_index = (expert * output_size + column) * (input_size / GROUP_K);
            let scale = decode_e4m3fn(scale_bytes[scale_index]) * global_scales[expert];
            let mut expected = 0.0f32;
            for k in 0..input_size {
                let byte_index = (expert * output_size + column) * (input_size / 2) + k / 2;
                let byte = packed[byte_index];
                let nibble = if k % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                let weight = bf16::from_f32(decode_e2m1(nibble) * scale).to_f32();
                expected += input_host[row * input_size + k].to_f32() * weight;
            }
            let expected_accumulator = expected;
            let expected = bf16::from_f32(expected).to_f32();
            let actual = actual[row * output_size + column].to_f32();
            max_abs_error = max_abs_error.max((actual - expected).abs());
            if !actual.is_finite() || (actual - expected).abs() > 0.25 {
                return Err(ModelError::Cuda(format!(
                    "grouped NVFP4 mismatch at ({row}, {column}), expert {expert}: {actual} != {expected}"
                )));
            }

            let up_scale_index = (expert * output_size + column) * (input_size / GROUP_K);
            let up_scale = decode_e4m3fn(up_scale_bytes[up_scale_index]) * up_global_scales[expert];
            let mut expected_up = 0.0f32;
            for k in 0..input_size {
                let byte_index = (expert * output_size + column) * (input_size / 2) + k / 2;
                let byte = up_packed[byte_index];
                let nibble = if k % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                let weight = bf16::from_f32(decode_e2m1(nibble) * up_scale).to_f32();
                expected_up += input_host[row * input_size + k].to_f32() * weight;
            }
            let expected_fused =
                expected_accumulator * (1.0 / (1.0 + (-expected_accumulator).exp())) * expected_up;
            let expected_fused = bf16::from_f32(expected_fused).to_f32();
            let fused_actual = fused_actual[row * output_size + column].to_f32();
            max_abs_error = max_abs_error.max((fused_actual - expected_fused).abs());
            let fused_tolerance = 1.0f32.max(expected_fused.abs() * 0.02);
            if !fused_actual.is_finite() || (fused_actual - expected_fused).abs() > fused_tolerance
            {
                return Err(ModelError::Cuda(format!(
                    "fused grouped NVFP4 SwiGLU mismatch at ({row}, {column}), expert {expert}: {fused_actual} != {expected_fused}"
                )));
            }
        }
    }
    Ok(max_abs_error)
}

struct Fp8ProbeSource<'a> {
    prefix: &'a str,
    input_size: usize,
    output_size: usize,
    encoded: &'a [u8],
    weight_scale: [u8; 4],
    input_scale: [u8; 4],
}

impl WeightSource for Fp8ProbeSource<'_> {
    fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, ModelError> {
        let suffix = name
            .strip_prefix(self.prefix)
            .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
        match suffix {
            ".weight" => Ok(WeightTensor::new(
                WeightDtype::F8E4M3,
                vec![self.output_size, self.input_size],
                self.encoded,
            )),
            ".weight_scale" => Ok(WeightTensor::new(
                WeightDtype::F32,
                vec![],
                &self.weight_scale,
            )),
            ".input_scale" => Ok(WeightTensor::new(
                WeightDtype::F32,
                vec![],
                &self.input_scale,
            )),
            _ => Err(ModelError::MissingTensor(name.into())),
        }
    }

    fn names(&self) -> Vec<String> {
        ["weight", "weight_scale", "input_scale"]
            .into_iter()
            .map(|suffix| format!("{}.{suffix}", self.prefix))
            .collect()
    }
}

struct Nvfp4ValidationSource<'a> {
    prefix: &'a str,
    input_size: usize,
    output_size: usize,
    packed: &'a [u8],
    scales: &'a [u8],
    global_scale: [u8; 4],
    input_scale: [u8; 4],
}

impl WeightSource for Nvfp4ValidationSource<'_> {
    fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, ModelError> {
        let suffix = name
            .strip_prefix(self.prefix)
            .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
        match suffix {
            ".weight" => Ok(WeightTensor::new(
                WeightDtype::U8,
                vec![self.output_size, self.input_size / 2],
                self.packed,
            )),
            ".weight_scale" => Ok(WeightTensor::new(
                WeightDtype::F8E4M3,
                vec![self.output_size, self.input_size / GROUP_K],
                self.scales,
            )),
            ".weight_scale_2" => Ok(WeightTensor::new(
                WeightDtype::F32,
                vec![],
                &self.global_scale,
            )),
            ".input_scale" => Ok(WeightTensor::new(
                WeightDtype::F32,
                vec![],
                &self.input_scale,
            )),
            _ => Err(ModelError::MissingTensor(name.into())),
        }
    }

    fn names(&self) -> Vec<String> {
        ["weight", "weight_scale", "weight_scale_2", "input_scale"]
            .into_iter()
            .map(|suffix| format!("{}.{suffix}", self.prefix))
            .collect()
    }
}

struct GroupedNvfp4ValidationSource<'a> {
    prefix: &'a str,
    input_size: usize,
    output_size: usize,
    packed: &'a [u8],
    scales: &'a [u8],
    global_scales: &'a [[u8; 4]],
    input_scales: &'a [[u8; 4]],
}

impl WeightSource for GroupedNvfp4ValidationSource<'_> {
    fn tensor(&self, name: &str) -> Result<WeightTensor<'_>, ModelError> {
        let remainder = name
            .strip_prefix(self.prefix)
            .and_then(|name| name.strip_prefix('.'))
            .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
        let (expert, suffix) = remainder
            .split_once('.')
            .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
        let expert = expert
            .parse::<usize>()
            .ok()
            .filter(|expert| *expert < self.global_scales.len())
            .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
        let packed_stride = self.output_size * (self.input_size / 2);
        let scale_stride = self.output_size * (self.input_size / GROUP_K);
        let suffix = ["gate_proj.", "up_proj.", "down_proj."]
            .into_iter()
            .find_map(|projection| suffix.strip_prefix(projection))
            .ok_or_else(|| ModelError::MissingTensor(name.into()))?;
        match suffix {
            "weight" => Ok(WeightTensor::new(
                WeightDtype::U8,
                vec![self.output_size, self.input_size / 2],
                &self.packed[expert * packed_stride..(expert + 1) * packed_stride],
            )),
            "weight_scale" => Ok(WeightTensor::new(
                WeightDtype::F8E4M3,
                vec![self.output_size, self.input_size / GROUP_K],
                &self.scales[expert * scale_stride..(expert + 1) * scale_stride],
            )),
            "weight_scale_2" => Ok(WeightTensor::new(
                WeightDtype::F32,
                vec![],
                &self.global_scales[expert],
            )),
            "input_scale" => Ok(WeightTensor::new(
                WeightDtype::F32,
                vec![],
                &self.input_scales[expert],
            )),
            _ => Err(ModelError::MissingTensor(name.into())),
        }
    }

    fn names(&self) -> Vec<String> {
        (0..self.global_scales.len())
            .flat_map(|expert| {
                ["gate_proj", "up_proj", "down_proj"]
                    .into_iter()
                    .flat_map(move |projection| {
                        ["weight", "weight_scale", "weight_scale_2", "input_scale"].map(
                            move |suffix| format!("{}.{expert}.{projection}.{suffix}", self.prefix),
                        )
                    })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use cuda_core::Device;

    #[test]
    #[ignore = "requires an NVIDIA GPU and CUDA 13.2+"]
    fn dense_quantized_linears_match_the_reference() {
        let device = Device::new(0).expect("initialize CUDA device");
        let stream = device.new_stream().expect("create CUDA stream");
        let probe = super::probe_quantized_linears(&stream).expect("run quantized linear probes");
        assert!(probe.fp8_max_abs_error.is_finite());
        assert!(probe.max_abs_error.is_finite());
        assert!(probe.grouped_max_abs_error.is_finite());
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU and CUDA 13.2+"]
    fn fused_grouped_gate_up_matches_the_reference() {
        let device = Device::new(0).expect("initialize CUDA device");
        let stream = device.new_stream().expect("create CUDA stream");
        let error =
            super::probe_grouped_nvfp4_w4a16(&stream).expect("run fused grouped NVFP4 probe");
        assert!(error.is_finite(), "probe returned non-finite error");
    }
}
