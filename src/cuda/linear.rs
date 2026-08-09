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
        weights::{WeightDtype, WeightSource},
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
    fn nvfp4_w4a16<const K_TILES: i32>(
        output: &mut Tensor<bf16, { [16, 16] }>,
        input: &Tensor<bf16, { [-1, -1] }>,
        packed_weight: &Tensor<u8, { [-1, -1] }>,
        weight_scale: &Tensor<bf16, { [-1, -1] }>,
        weight_global_scale: f32,
    ) {
        let pid = get_tile_block_id();
        let k_tiles = Dim::new(K_TILES);
        let low_mask: Tile<u8, { [16, 8] }> = constant(0x0fu8, const_shape![16, 8]);
        let nibble_shift: Tile<u8, { [16, 8] }> = constant(4u8, const_shape![16, 8]);
        let mut accumulator = constant(0.0f32, const_shape![16, 16]);

        for k_tile in k_tiles {
            let activation = input.load_tile(const_shape![16, 16], [pid.0, k_tile]);
            let packed = packed_weight.load_tile(const_shape![16, 8], [pid.1, k_tile]);
            let low = andi(packed, low_mask).reshape(const_shape![16, 8, 1]);
            let high = shri(packed, nibble_shift).reshape(const_shape![16, 8, 1]);
            let nibbles: Tile<u8, { [16, 8, 2] }> = cat(low, high, 2);
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
        let low_mask: Tile<u8, { [1, 16, 8] }> = constant(0x0fu8, const_shape![1, 16, 8]);
        let nibble_shift: Tile<u8, { [1, 16, 8] }> = constant(4u8, const_shape![1, 16, 8]);
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
                let packed =
                    packed_weight.load_tile(const_shape![1, 16, 8], [expert, column_tile, k_tile]);
                let low = andi(packed, low_mask).reshape(const_shape![16, 8, 1]);
                let high = shri(packed, nibble_shift).reshape(const_shape![16, 8, 1]);
                let nibbles: Tile<u8, { [16, 8, 2] }> = cat(low, high, 2);
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

    fn decode_fp4(nibbles: Tile<u8, { [16, 16] }>) -> Tile<f32, { [16, 16] }> {
        let eight: Tile<u8, { [16, 16] }> = constant(8u8, const_shape![16, 16]);
        let magnitude = nibbles % eight;
        let sign = nibbles / eight;
        let one: Tile<u8, { [16, 16] }> = constant(1u8, const_shape![16, 16]);
        let two: Tile<u8, { [16, 16] }> = constant(2u8, const_shape![16, 16]);
        let three: Tile<u8, { [16, 16] }> = constant(3u8, const_shape![16, 16]);
        let four: Tile<u8, { [16, 16] }> = constant(4u8, const_shape![16, 16]);
        let five: Tile<u8, { [16, 16] }> = constant(5u8, const_shape![16, 16]);
        let six: Tile<u8, { [16, 16] }> = constant(6u8, const_shape![16, 16]);
        let seven: Tile<u8, { [16, 16] }> = constant(7u8, const_shape![16, 16]);
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
}

use kernels::{grouped_nvfp4_w4a16, nvfp4_w4a16};

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
        let weight_name = format!("{prefix}.weight");
        let scale_name = format!("{prefix}.weight_scale");
        let global_name = format!("{prefix}.weight_scale_2");
        let input_scale_name = format!("{prefix}.input_scale");
        let weight = source.tensor(&weight_name)?;
        let scale = source.tensor(&scale_name)?;
        let global = source.tensor(&global_name)?;
        let input_scale = source.tensor(&input_scale_name)?;

        if weight.dtype() != &WeightDtype::U8 || weight.shape().len() != 2 {
            return invalid_tensor(&weight_name, "expected rank-2 packed U8 weight");
        }
        let output_size = weight.shape()[0];
        let input_size =
            weight.shape()[1]
                .checked_mul(2)
                .ok_or_else(|| ModelError::InvalidTensor {
                    name: weight_name.clone(),
                    message: "logical input width overflowed".into(),
                })?;
        if input_size == 0
            || output_size == 0
            || !input_size.is_multiple_of(GROUP_K)
            || !output_size.is_multiple_of(TILE_N)
        {
            return invalid_tensor(&weight_name, "unsupported W4A16 projection geometry");
        }
        source.validate_tensor(
            &scale_name,
            &WeightDtype::F8E4M3,
            &[output_size, input_size / GROUP_K],
        )?;
        source.validate_tensor(&global_name, &WeightDtype::F32, &[])?;
        source.validate_tensor(&input_scale_name, &WeightDtype::F32, &[])?;

        let weight_global_scale = scalar_f32(global.bytes(), &global_name)?;
        // W4A16 deliberately does not quantize activations, but validating the
        // exported placeholder catches corrupt or mismatched projection sets.
        let input_scale = scalar_f32(input_scale.bytes(), &input_scale_name)?;
        if !weight_global_scale.is_finite()
            || weight_global_scale <= 0.0
            || !input_scale.is_finite()
            || input_scale <= 0.0
        {
            return invalid_tensor(
                prefix,
                "global quantization scales must be finite and positive",
            );
        }

        Self::from_host(
            input_size,
            output_size,
            weight.bytes(),
            scale.bytes(),
            weight_global_scale,
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

impl GroupedNvfp4W4A16 {
    fn from_host(
        num_experts: usize,
        input_size: usize,
        output_size: usize,
        packed_weight: &[u8],
        scale_bytes: &[u8],
        weight_global_scale: &[f32],
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

        let packed_weight = api::copy_host_vec_to_device(&Arc::new(packed_weight.to_vec()))
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
        let weight_global_scale =
            api::copy_host_vec_to_device(&Arc::new(weight_global_scale.to_vec()))
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
pub(crate) struct Nvfp4LinearValidation {
    pub(crate) max_abs_error: f32,
    pub(crate) grouped_max_abs_error: f32,
}

/// Differentially validates the production W4A16 kernel with every FP4 code,
/// non-unit E4M3 scales, a non-unit global scale, and signed BF16 activations.
pub(crate) fn validate_nvfp4_w4a16(
    stream: &Arc<Stream>,
) -> Result<Nvfp4LinearValidation, ModelError> {
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
    let linear = Nvfp4W4A16Linear::from_host(
        input_size,
        output_size,
        &packed,
        &scale_bytes,
        global_scale,
        stream,
    )?;
    let input_host = (0..TILE_M * input_size)
        .map(|index| bf16::from_f32((index % 7) as f32 - 3.0))
        .collect::<Vec<_>>();
    let input = api::copy_host_vec_to_device(&Arc::new(input_host.clone()))
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("upload NVFP4 test input: {error:?}")))?
        .reshape(&[TILE_M, input_size])
        .map_err(|error| ModelError::Cuda(format!("reshape NVFP4 test input: {error:?}")))?;
    let output = Arc::new(linear.enqueue(Arc::new(input), TILE_M, stream)?);
    let actual: Vec<bf16> = output
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
            let actual = actual[row * output_size + column].to_f32();
            max_abs_error = max_abs_error.max((actual - expected).abs());
            if !actual.is_finite() || (actual - expected).abs() > 0.25 {
                return Err(ModelError::Cuda(format!(
                    "NVFP4 differential mismatch at ({row}, {column}): {actual} != {expected}"
                )));
            }
        }
    }
    let grouped_max_abs_error = validate_grouped_nvfp4_w4a16(stream)?;
    Ok(Nvfp4LinearValidation {
        max_abs_error,
        grouped_max_abs_error,
    })
}

fn validate_grouped_nvfp4_w4a16(stream: &Arc<Stream>) -> Result<f32, ModelError> {
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
    let grouped = GroupedNvfp4W4A16::from_host(
        EXPERTS,
        input_size,
        output_size,
        &packed,
        &scale_bytes,
        &global_scales,
        stream,
    )?;
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
