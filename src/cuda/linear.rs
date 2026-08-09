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
    tile_kernel::TileKernel,
};

use crate::{
    model::{
        ModelError,
        weights::{WeightDtype, WeightSource, WeightTensor},
    },
    quantization::{decode_e2m1, decode_e4m3fn},
};

const TILE_M: usize = 16;
const TILE_N: usize = 16;
const GROUP_K: usize = 16;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn fp8_w8a16<const K_TILES: i32>(
        output: &mut Tensor<bf16, { [16, 16] }>,
        input: &Tensor<bf16, { [-1, -1] }>,
        weight: &Tensor<u8, { [-1, -1] }>,
        weight_scale: f32,
    ) {
        let pid = get_tile_block_id();
        let k_tiles = Dim::new(K_TILES);
        let mut accumulator = constant(0.0f32, const_shape![16, 16]);

        for k_tile in k_tiles {
            let activation = input.load_tile(const_shape![16, 16], [pid.0, k_tile]);
            let encoded: Tile<i32, { [16, 16] }> =
                exti(weight.load_tile(const_shape![16, 16], [pid.1, k_tile]));
            let decoded = decode_fp8(encoded);
            let scale = broadcast_scalar(weight_scale, const_shape![16, 16]);
            let weight: Tile<bf16, { [16, 16] }> = ftof(decoded * scale, rounding::NearestEven);
            accumulator = mma(activation, weight.transpose(), accumulator);
        }
        let output_tile: Tile<bf16, { [16, 16] }> = ftof(accumulator, rounding::NearestEven);
        output.store(output_tile);
    }

    #[cutile::entry()]
    fn nvfp4_w4a16<const K_TILES: i32>(
        output: &mut Tensor<bf16, { [16, 16] }>,
        input: &Tensor<bf16, { [-1, -1] }>,
        packed_weight: &Tensor<u8, { [-1, -1] }>,
        weight_scale: &Tensor<bf16, { [-1, -1] }>,
        weight_global_scale: f32,
    ) {
        let pid = get_tile_block_id();
        let k_tiles = Dim::new(K_TILES);
        let low_mask: Tile<i32, { [16, 8] }> = constant(0x0fi32, const_shape![16, 8]);
        let nibble_shift: Tile<i32, { [16, 8] }> = constant(4i32, const_shape![16, 8]);
        let zero_i32: Tile<i32, { [16, 8] }> = constant(0i32, const_shape![16, 8]);
        let byte_modulus: Tile<i32, { [16, 8] }> = constant(256i32, const_shape![16, 8]);
        let mut accumulator = constant(0.0f32, const_shape![16, 16]);

        for k_tile in k_tiles {
            let activation = input.load_tile(const_shape![16, 16], [pid.0, k_tile]);
            let packed: Tile<i32, { [16, 8] }> =
                exti(packed_weight.load_tile(const_shape![16, 8], [pid.1, k_tile]));
            let packed = select(lt_tile(packed, zero_i32), packed + byte_modulus, packed);
            let low = andi(packed, low_mask).reshape(const_shape![16, 8, 1]);
            let high = shri(packed, nibble_shift).reshape(const_shape![16, 8, 1]);
            let nibbles: Tile<i32, { [16, 8, 2] }> = cat(low, high, 2);
            let weight = decode_fp4(nibbles.reshape(const_shape![16, 16]));
            let scale: Tile<f32, { [16, 16] }> = convert_tile(
                weight_scale
                    .load_tile(const_shape![16, 1], [pid.1, k_tile])
                    .broadcast(const_shape![16, 16]),
            );
            let global = broadcast_scalar(weight_global_scale, const_shape![16, 16]);
            let weight: Tile<bf16, { [16, 16] }> =
                ftof(weight * scale * global, rounding::NearestEven);
            accumulator = mma(activation, weight.transpose(), accumulator);
        }
        let output_tile: Tile<bf16, { [16, 16] }> = ftof(accumulator, rounding::NearestEven);
        output.store(output_tile);
    }

    /// One launch processes all routed expert segments. Dispatch pads each
    /// expert segment to 16 rows and supplies one expert id per row tile, so
    /// no output tile crosses an expert boundary and no per-expert launch is
    /// required.
    #[cutile::entry(unchecked_accesses = false)]
    fn grouped_nvfp4_w4a16<const K_TILES: i32>(
        mut output: MappedPartitionMut<bf16, { [16, 16] }, { [8, 1] }>,
        dispatched: &Tensor<bf16, { [-1, -1] }>,
        expert_by_row_tile: &Tensor<i32, { [-1] }>,
        packed_weight: &Tensor<u8, { [-1, -1, -1] }>,
        weight_scale: &Tensor<bf16, { [-1, -1, -1] }>,
        weight_global_scale: &Tensor<f32, { [-1] }>,
    ) {
        let low_mask: Tile<i32, { [1, 16, 8] }> = constant(0x0fi32, const_shape![1, 16, 8]);
        let nibble_shift: Tile<i32, { [1, 16, 8] }> = constant(4i32, const_shape![1, 16, 8]);
        let zero_i32: Tile<i32, { [1, 16, 8] }> = constant(0i32, const_shape![1, 16, 8]);
        let byte_modulus: Tile<i32, { [1, 16, 8] }> = constant(256i32, const_shape![1, 16, 8]);
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
                .broadcast(const_shape![16, 16]);
            let mut accumulator = constant(0.0f32, const_shape![16, 16]);

            for k_tile in k_tiles {
                let activation = dispatched.load_tile(const_shape![16, 16], [row_tile, k_tile]);
                let packed: Tile<i32, { [1, 16, 8] }> = exti(
                    packed_weight.load_tile(const_shape![1, 16, 8], [expert, column_tile, k_tile]),
                );
                let packed = select(lt_tile(packed, zero_i32), packed + byte_modulus, packed);
                let low = andi(packed, low_mask).reshape(const_shape![16, 8, 1]);
                let high = shri(packed, nibble_shift).reshape(const_shape![16, 8, 1]);
                let nibbles: Tile<i32, { [16, 8, 2] }> = cat(low, high, 2);
                let weight = decode_fp4(nibbles.reshape(const_shape![16, 16]));
                let scale: Tile<f32, { [16, 16] }> = convert_tile(
                    weight_scale
                        .load_tile(const_shape![1, 16, 1], [expert, column_tile, k_tile])
                        .reshape(const_shape![16, 1])
                        .broadcast(const_shape![16, 16]),
                );
                let weight: Tile<bf16, { [16, 16] }> =
                    ftof(weight * scale * global, rounding::NearestEven);
                accumulator = mma(activation, weight.transpose(), accumulator);
            }
            let output_tile: Tile<bf16, { [16, 16] }> = ftof(accumulator, rounding::NearestEven);
            output.store(output_tile, out_idx);
        }
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

    fn decode_fp8(encoded: Tile<i32, { [16, 16] }>) -> Tile<f32, { [16, 16] }> {
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
        const SUBNORMAL_SCALE: f32 = 1.0 / 512.0;
        let subnormal_scale: Tile<f32, { [16, 16] }> = constant(SUBNORMAL_SCALE, shape);
        let normal =
            (one_f + true_div(mantissa_f, eight_f)) * exp2(exponent_f - seven_f, ftz::Disabled);
        let subnormal = mantissa_f * subnormal_scale;
        let magnitude = select(eq_tile(exponent, zero_i32), subnormal, normal);
        let zero_f: Tile<f32, { [16, 16] }> = constant(0.0f32, shape);
        let negative = zero_f - magnitude;
        let one_i32: Tile<i32, { [16, 16] }> = constant(1i32, shape);
        select(eq_tile(sign, one_i32), negative, magnitude)
    }
}

use kernels::{fp8_w8a16, grouped_nvfp4_w4a16, nvfp4_w4a16};

/// Scalar-scaled ModelOpt FP8 projection executed as W8A16 on SM80. The
/// checkpoint's E4M3 bytes stay packed on device; the kernel decodes one tile,
/// casts it to BF16, and accumulates with BF16 activations in FP32.
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
            || !output_size.is_multiple_of(TILE_N)
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
        let weight = api::copy_host_vec_to_device(&Arc::new(weight.bytes().to_vec()))
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
        stream: &Arc<Stream>,
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
        let mut output = api::zeros::<bf16>(&[rows, self.output_size])
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("allocate FP8 output: {error:?}")))?;
        fp8_w8a16(
            (&mut output).partition([TILE_M, TILE_N]),
            input,
            self.weight.clone(),
            self.weight_scale,
        )
        .generics(vec![(self.input_size / TILE_N).to_string()])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("execute FP8 W8A16: {error:?}")))?;
        Ok(output)
    }
}

#[derive(Clone, Copy)]
struct ParsedNvfp4Projection<'a> {
    input_size: usize,
    output_size: usize,
    packed_weight: &'a [u8],
    scale_bytes: &'a [u8],
    weight_global_scale: f32,
}

/// Parses ModelOpt's four-tensor W4A16 representation. Geometry comes from
/// the checkpoint; these checks only establish that the bytes can be safely
/// interpreted by the kernel.
fn parse_nvfp4_projection<'a>(
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
        || !output_size.is_multiple_of(TILE_N)
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

    fn from_host(
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
        stream: &Arc<Stream>,
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
        let mut output = api::zeros::<bf16>(&[rows, self.output_size])
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("allocate NVFP4 output: {error:?}")))?;
        nvfp4_w4a16(
            (&mut output).partition([TILE_M, TILE_N]),
            input,
            self.packed_weight.clone(),
            self.weight_scale.clone(),
            self.weight_global_scale,
        )
        .generics(vec![(self.input_size / GROUP_K).to_string()])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("execute NVFP4 W4A16: {error:?}")))?;
        Ok(output)
    }
}

/// Packed expert bank for a single-launch grouped W4A16 GEMM.
pub(crate) struct GroupedNvfp4W4A16 {
    input_size: usize,
    output_size: usize,
    num_experts: usize,
    packed_weight: Arc<Tensor<u8>>,
    weight_scale: Arc<Tensor<bf16>>,
    weight_global_scale: Arc<Tensor<f32>>,
    device_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertProjection {
    Gate,
    Up,
    Down,
}

impl ExpertProjection {
    const fn suffix(self) -> &'static str {
        match self {
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }
}

impl GroupedNvfp4W4A16 {
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

    fn from_host_owned(
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
            || !input_size.is_multiple_of(GROUP_K)
            || !output_size.is_multiple_of(TILE_N)
            || expected_weights != Some(packed_weight.len())
            || expected_scales != Some(scale_bytes.len())
            || weight_global_scale.len() != num_experts
            || weight_global_scale
                .iter()
                .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            return invalid_tensor("grouped_nvfp4", "invalid grouped W4A16 artifact");
        }

        let packed_weight = api::copy_host_vec_to_device(&Arc::new(packed_weight))
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("upload grouped NVFP4 weight: {error:?}")))?
            .reshape(&[num_experts, output_size, input_size / 2])
            .map_err(|error| {
                ModelError::Cuda(format!("reshape grouped NVFP4 weight: {error:?}"))
            })?;
        let scales = scale_bytes
            .iter()
            .map(|bits| decode_e4m3fn(*bits))
            .map(|value| {
                if value.is_finite() {
                    Ok(bf16::from_f32(value))
                } else {
                    invalid_tensor("grouped_nvfp4.weight_scale", "contains E4M3 NaN")
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let weight_scale = api::copy_host_vec_to_device(&Arc::new(scales))
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
        let device_bytes = packed_weight
            .num_bytes()
            .checked_add(weight_scale.num_bytes())
            .and_then(|bytes| bytes.checked_add(weight_global_scale.num_bytes()))
            .ok_or_else(|| ModelError::Cuda("grouped NVFP4 byte count overflowed".into()))?;
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

    /// `expert_by_row_tile` has one entry per 16-row dispatched tile. Routing
    /// must pad each expert segment to that boundary before calling this leaf.
    pub(crate) fn enqueue(
        &self,
        dispatched: Arc<Tensor<bf16>>,
        rows: usize,
        expert_by_row_tile: &[i32],
        stream: &Arc<Stream>,
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
                .sync_on(stream)
                .map_err(|error| {
                    ModelError::Cuda(format!("upload grouped expert map: {error:?}"))
                })?;
        let mut output = api::zeros::<bf16>(&[rows, self.output_size])
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("allocate grouped output: {error:?}")))?;
        let logical_tiles = (rows / TILE_M)
            .checked_mul(self.output_size / TILE_N)
            .ok_or_else(|| ModelError::Cuda("grouped output tile count overflowed".into()))?;
        let workers = logical_tiles.min(device_sm_count(stream)?);
        let output_partition = (&mut output).partition([TILE_M, TILE_N]).map(
            [8, 1],
            u32::try_from(workers).map_err(|_| {
                ModelError::Cuda("persistent grouped worker count overflowed u32".into())
            })?,
        );
        grouped_nvfp4_w4a16(
            output_partition,
            dispatched,
            Arc::new(expert_by_row_tile),
            self.packed_weight.clone(),
            self.weight_scale.clone(),
            self.weight_global_scale.clone(),
        )
        .generics(vec![(self.input_size / GROUP_K).to_string()])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("execute grouped NVFP4 W4A16: {error:?}")))?;
        Ok(output)
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
    let output_size = TILE_N;
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
    let output = linear.enqueue(Arc::new(input), TILE_M, stream)?;
    let actual: Vec<bf16> = output
        .to_host_vec()
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("copy FP8 probe output: {error:?}")))?;
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
    let output_size = TILE_N;
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
    let output = Arc::new(linear.enqueue(Arc::new(input), TILE_M, stream)?);
    let actual_values: Vec<bf16> = output
        .to_host_vec()
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("copy NVFP4 test output: {error:?}")))?;

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
    let input_size = GROUP_K;
    let output_size = TILE_N;
    let packed = (0..EXPERTS * output_size * input_size / 2)
        .map(|index| {
            let expert = index / (output_size * input_size / 2);
            let low = ((index * 2 + expert) % 16) as u8;
            let high = ((index * 2 + 1 + expert) % 16) as u8;
            low | (high << 4)
        })
        .collect::<Vec<_>>();
    let scale_bytes = (0..EXPERTS * output_size)
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
        .checked_add(scale_bytes.len() * std::mem::size_of::<bf16>())
        .and_then(|bytes| bytes.checked_add(global_scales.len() * std::mem::size_of::<f32>()))
        .ok_or_else(|| ModelError::Cuda("grouped validation byte count overflowed".into()))?;
    if grouped.device_bytes() != expected_device_bytes {
        return Err(ModelError::Cuda(
            "grouped NVFP4 device-byte accounting mismatch".into(),
        ));
    }
    let rows = TILE_M * EXPERTS;
    let input_host = (0..rows * input_size)
        .map(|index| bf16::from_f32((index % 9) as f32 - 4.0))
        .collect::<Vec<_>>();
    let input = api::copy_host_vec_to_device(&Arc::new(input_host.clone()))
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("upload grouped test input: {error:?}")))?
        .reshape(&[rows, input_size])
        .map_err(|error| ModelError::Cuda(format!("reshape grouped test input: {error:?}")))?;
    let expert_by_row_tile = [1i32, 0i32];
    let output = Arc::new(grouped.enqueue(Arc::new(input), rows, &expert_by_row_tile, stream)?);
    let actual: Vec<bf16> = output
        .to_host_vec()
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("copy grouped test output: {error:?}")))?;

    let mut max_abs_error = 0.0f32;
    for row in 0..rows {
        let expert = expert_by_row_tile[row / TILE_M] as usize;
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
            let expected = bf16::from_f32(expected).to_f32();
            let actual = actual[row * output_size + column].to_f32();
            max_abs_error = max_abs_error.max((actual - expected).abs());
            if !actual.is_finite() || (actual - expected).abs() > 0.25 {
                return Err(ModelError::Cuda(format!(
                    "grouped NVFP4 mismatch at ({row}, {column}), expert {expert}: {actual} != {expected}"
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
