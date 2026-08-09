//! Model-neutral CUDA infrastructure and kernel validation.
//!
//! Architecture-specific dimensions, tensor names, prompt behavior, and
//! forward-pass composition belong to the corresponding model module.

use std::sync::Arc;

use cuda_async::device_operation::DeviceOp;
use cuda_async::{cuda_graph::CudaGraph, error::DeviceError};
use cuda_core::{Device, f4e2m1fnx2, f8e4m3fn};
use cutile::{
    api::{self, DeviceOpReshape},
    core::bf16,
    tensor::{PartitionMut, Reshape, Tensor, ToHostVec},
};
use thiserror::Error;

pub(crate) mod attention;
pub(crate) mod batch;
pub(crate) mod cublas;
pub(crate) mod dense_decoder;
pub(crate) mod executor;
pub(crate) mod kernel_plan;
pub(crate) mod kernels;
pub(crate) mod linear;

#[cutile::module]
mod smoke_kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn add_bf16<const BLOCK: i32>(
        out: &mut Tensor<bf16, { [BLOCK] }>,
        lhs: &Tensor<bf16, { [-1] }>,
        rhs: &Tensor<bf16, { [-1] }>,
    ) {
        let block = get_tile_block_id().0;
        let lhs = lhs.load_tile(const_shape![BLOCK], [block]);
        let rhs = rhs.load_tile(const_shape![BLOCK], [block]);
        out.store(lhs + rhs);
    }
}

#[cutile::module]
mod nvfp4_probe_kernels {
    use cutile::core::*;

    /// Exercises cuTile's scaled FP4 matrix multiply exactly as an inference
    /// kernel would. The host deliberately applies no architecture-name gate:
    /// successful compilation and numerically correct execution are the
    /// capability test, whether cuTile uses hardware instructions or a lower
    /// level fallback for the target architecture.
    #[cutile::entry()]
    fn scaled_mma<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const BK_PACKED: i32,
        const BK_SCALES: i32,
    >(
        out: &mut Tensor<f32, { [BM, BN] }>,
        lhs: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        rhs: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        lhs_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        rhs_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
    ) {
        let pid = get_tile_block_id();
        let k_tiles = Dim::new(lhs.shape()[1] / BK_PACKED);
        let lhs = lhs.partition(const_shape![BM, BK_PACKED]);
        let rhs = rhs.partition(const_shape![BN, BK_PACKED]);
        let lhs_scales = lhs_scales.partition(const_shape![BM, BK_SCALES]);
        let rhs_scales = rhs_scales.partition(const_shape![BN, BK_SCALES]);
        let mut accumulator = constant(0.0f32, const_shape![BM, BN]);

        for k_tile in k_tiles {
            let lhs = lhs.load([pid.0, k_tile]).unpack(const_shape![BM, BK]);
            let rhs = rhs
                .load([pid.1, k_tile])
                .unpack(const_shape![BN, BK])
                .transpose();
            let lhs_scales = lhs_scales.load([pid.0, k_tile]);
            let rhs_scales = rhs_scales.load([pid.1, k_tile]).transpose();
            accumulator = mmaf_scaled(lhs, rhs, accumulator, lhs_scales, rhs_scales);
        }
        out.store(accumulator);
    }

    /// SM80-compatible fallback: keep checkpoint storage byte-addressable,
    /// decode FP4 and E4M3 in Tile IR, then use ordinary BF16 tensor-core MMA.
    /// This fixed validation tile exercises every primitive needed by the
    /// shape-generic production kernel without claiming performance parity.
    #[cutile::entry()]
    fn byte_decode_mma(
        out: &mut Tensor<f32, { [16, 16] }>,
        lhs_packed: &Tensor<u8, { [-1, -1] }>,
        rhs_packed: &Tensor<u8, { [-1, -1] }>,
        lhs_scale_bytes: &Tensor<u8, { [-1, -1] }>,
        rhs_scale_bytes: &Tensor<u8, { [-1, -1] }>,
    ) {
        let lhs_packed = lhs_packed.load_tile(const_shape![16, 8], [0, 0]);
        let rhs_packed = rhs_packed.load_tile(const_shape![16, 8], [0, 0]);
        let lhs_scale_bytes = lhs_scale_bytes.load_tile(const_shape![16, 1], [0, 0]);
        let rhs_scale_bytes = rhs_scale_bytes.load_tile(const_shape![16, 1], [0, 0]);
        let sixteen: Tile<u8, { [16, 8] }> = constant(16u8, const_shape![16, 8]);
        let lhs_low = (lhs_packed % sixteen).reshape(const_shape![16, 8, 1]);
        let lhs_high = (lhs_packed / sixteen).reshape(const_shape![16, 8, 1]);
        let lhs_nibbles: Tile<u8, { [16, 8, 2] }> = cat(lhs_low, lhs_high, 2);
        let rhs_low = (rhs_packed % sixteen).reshape(const_shape![16, 8, 1]);
        let rhs_high = (rhs_packed / sixteen).reshape(const_shape![16, 8, 1]);
        let rhs_nibbles: Tile<u8, { [16, 8, 2] }> = cat(rhs_low, rhs_high, 2);

        let lhs = decode_fp4(lhs_nibbles.reshape(const_shape![16, 16]));
        let rhs = decode_fp4(rhs_nibbles.reshape(const_shape![16, 16]));
        let lhs_scales = decode_e4m3(lhs_scale_bytes).broadcast(const_shape![16, 16]);
        let rhs_scales = decode_e4m3(rhs_scale_bytes).broadcast(const_shape![16, 16]);
        let lhs: Tile<bf16, { [16, 16] }> = ftof(lhs * lhs_scales, rounding::NearestEven);
        let rhs: Tile<bf16, { [16, 16] }> = ftof(rhs * rhs_scales, rounding::NearestEven);
        let accumulator = constant(0.0f32, const_shape![16, 16]);
        out.store(mma(lhs, rhs.transpose(), accumulator));
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

    fn decode_e4m3(bytes: Tile<u8, { [16, 1] }>) -> Tile<f32, { [16, 1] }> {
        let eight: Tile<u8, { [16, 1] }> = constant(8u8, const_shape![16, 1]);
        let sixteen: Tile<u8, { [16, 1] }> = constant(16u8, const_shape![16, 1]);
        let exponent = (bytes / eight) % sixteen;
        let mantissa = bytes % eight;
        let zero_u8: Tile<u8, { [16, 1] }> = constant(0u8, const_shape![16, 1]);
        let one_u8: Tile<u8, { [16, 1] }> = constant(1u8, const_shape![16, 1]);
        let exponent_f: Tile<f32, { [16, 1] }> = convert_tile(exponent);
        let mantissa_f: Tile<f32, { [16, 1] }> = convert_tile(mantissa);
        let zero_f: Tile<f32, { [16, 1] }> = constant(0.0f32, const_shape![16, 1]);
        let one_f: Tile<f32, { [16, 1] }> = constant(1.0f32, const_shape![16, 1]);
        let two_f: Tile<f32, { [16, 1] }> = constant(2.0f32, const_shape![16, 1]);
        let seven_f: Tile<f32, { [16, 1] }> = constant(7.0f32, const_shape![16, 1]);
        let eight_f: Tile<f32, { [16, 1] }> = constant(8.0f32, const_shape![16, 1]);
        let five_twelve_f: Tile<f32, { [16, 1] }> = constant(512.0f32, const_shape![16, 1]);
        let normal = (one_f + mantissa_f / eight_f) * pow(two_f, exponent_f - seven_f);
        let subnormal = mantissa_f / five_twelve_f;
        let unsigned = select(eq_tile(exponent, zero_u8), subnormal, normal);
        let one_twenty_eight: Tile<u8, { [16, 1] }> = constant(128u8, const_shape![16, 1]);
        let sign = bytes / one_twenty_eight;
        select(eq_tile(sign, one_u8), zero_f - unsigned, unsigned)
    }
}

use nvfp4_probe_kernels::{byte_decode_mma, scaled_mma};
use smoke_kernels::add_bf16;

const SMOKE_ELEMENTS: usize = 4096;
const SMOKE_BLOCK: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16SmokeReport {
    pub device_id: usize,
    pub elements: usize,
    pub gemm_rows: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CudaKernelCapability {
    Available { max_abs_error: f32 },
    Unavailable { detail: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Nvfp4CapabilityReport {
    pub device_id: usize,
    pub scaled_mma: CudaKernelCapability,
    pub byte_decode_mma: CudaKernelCapability,
    pub w4a16_linear: CudaKernelCapability,
    pub grouped_w4a16: CudaKernelCapability,
}

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("failed to enable the persistent cuTile CUBIN cache: {0}")]
    CubinCache(#[source] std::io::Error),
    #[error("failed to initialize CUDA device {device_id}: {message}")]
    Device { device_id: usize, message: String },
    #[error("BF16 cuTile validation failed during {operation}: {message}")]
    Bf16Kernel {
        operation: &'static str,
        message: String,
    },
    #[error("BF16 cuTile validation produced {actual} at element {index}; expected 2")]
    WrongValue { index: usize, actual: f32 },
    #[error(
        "cuTile NVFP4 validation failed in {operation}: produced {actual} at element {index}; expected {expected}"
    )]
    WrongNvfp4Value {
        operation: &'static str,
        index: usize,
        actual: f32,
        expected: f32,
    },
    #[error("NVFP4 linear validation failed: {message}")]
    QuantizedLinear { message: String },
    #[error(transparent)]
    Cublas(#[from] cublas::CublasError),
}

/// Attempts both cuTile's scaled NVFP4 MMA and its byte-decoded BF16 fallback,
/// validating every output numerically. Unsupported compiler/runtime paths are
/// reported as capability results rather than inferred from the GPU name.
pub fn probe_nvfp4(device_id: usize) -> Result<Nvfp4CapabilityReport, CudaError> {
    use cutile::tile_kernel::TileKernel;

    const TILE: usize = 16;
    const K: usize = 128;
    const K_TILE: usize = 64;

    enable_persistent_cubin_cache()?;
    let device = Device::new(device_id).map_err(|error| CudaError::Device {
        device_id,
        message: format!("{error:?}"),
    })?;
    let stream = device.new_stream().map_err(|error| CudaError::Device {
        device_id,
        message: format!("failed to create stream: {error:?}"),
    })?;

    // 0x2 is FP4 +1.0 and 0x38 is E4M3 +1.0. Thus every result must be K.
    let fp4_one = f4e2m1fnx2::from_nibbles(0x2, 0x2);
    let scale_one = f8e4m3fn(0x38);
    let lhs: Arc<Tensor<f4e2m1fnx2>> =
        api::copy_host_vec_to_device(&Arc::new(vec![fp4_one; TILE * K / 2]))
            .reshape(&[TILE, K / 2])
            .sync_on(&stream)
            .map_err(|error| nvfp4_kernel_error("upload lhs", error))?
            .into();
    let rhs: Arc<Tensor<f4e2m1fnx2>> =
        api::copy_host_vec_to_device(&Arc::new(vec![fp4_one; TILE * K / 2]))
            .reshape(&[TILE, K / 2])
            .sync_on(&stream)
            .map_err(|error| nvfp4_kernel_error("upload rhs", error))?
            .into();
    let lhs_scales: Arc<Tensor<f8e4m3fn>> =
        api::copy_host_vec_to_device(&Arc::new(vec![scale_one; TILE * K / 16]))
            .reshape(&[TILE, K / 16])
            .sync_on(&stream)
            .map_err(|error| nvfp4_kernel_error("upload lhs scales", error))?
            .into();
    let rhs_scales: Arc<Tensor<f8e4m3fn>> =
        api::copy_host_vec_to_device(&Arc::new(vec![scale_one; TILE * K / 16]))
            .reshape(&[TILE, K / 16])
            .sync_on(&stream)
            .map_err(|error| nvfp4_kernel_error("upload rhs scales", error))?
            .into();
    let mut out_tensor = api::zeros::<f32>(&[TILE, TILE])
        .sync_on(&stream)
        .map_err(|error| nvfp4_kernel_error("allocate output", error))?;
    let out = (&mut out_tensor).partition([TILE, TILE]);

    let launch = scaled_mma(out, lhs, rhs, lhs_scales, rhs_scales)
        .generics(vec![
            TILE.to_string(),
            TILE.to_string(),
            K_TILE.to_string(),
            (K_TILE / 2).to_string(),
            (K_TILE / 16).to_string(),
        ])
        .sync_on(&stream);
    let scaled_mma = match launch {
        Ok((out, ..)) => {
            drop(out);
            validate_nvfp4_output(out_tensor, &stream, "scaled MMA", K as f32)?
        }
        Err(error) => CudaKernelCapability::Unavailable {
            detail: format!("{error:?}"),
        },
    };

    let packed_one = 0x22u8;
    let scale_one = 0x38u8;
    let lhs: Arc<Tensor<u8>> =
        api::copy_host_vec_to_device(&Arc::new(vec![packed_one; TILE * K / 2]))
            .reshape(&[TILE, K / 2])
            .sync_on(&stream)
            .map_err(|error| nvfp4_kernel_error("upload byte lhs", error))?
            .into();
    let rhs: Arc<Tensor<u8>> =
        api::copy_host_vec_to_device(&Arc::new(vec![packed_one; TILE * K / 2]))
            .reshape(&[TILE, K / 2])
            .sync_on(&stream)
            .map_err(|error| nvfp4_kernel_error("upload byte rhs", error))?
            .into();
    let lhs_scales: Arc<Tensor<u8>> =
        api::copy_host_vec_to_device(&Arc::new(vec![scale_one; TILE * K / 16]))
            .reshape(&[TILE, K / 16])
            .sync_on(&stream)
            .map_err(|error| nvfp4_kernel_error("upload byte lhs scales", error))?
            .into();
    let rhs_scales: Arc<Tensor<u8>> =
        api::copy_host_vec_to_device(&Arc::new(vec![scale_one; TILE * K / 16]))
            .reshape(&[TILE, K / 16])
            .sync_on(&stream)
            .map_err(|error| nvfp4_kernel_error("upload byte rhs scales", error))?
            .into();
    let mut out_tensor = api::zeros::<f32>(&[TILE, TILE])
        .sync_on(&stream)
        .map_err(|error| nvfp4_kernel_error("allocate byte fallback output", error))?;
    let out = (&mut out_tensor).partition([TILE, TILE]);
    let byte_decode_mma =
        match byte_decode_mma(out, lhs, rhs, lhs_scales, rhs_scales).sync_on(&stream) {
            Ok((out, ..)) => {
                drop(out);
                validate_nvfp4_output(out_tensor, &stream, "byte-decode MMA", 16.0)?
            }
            Err(error) => CudaKernelCapability::Unavailable {
                detail: format!("{error:?}"),
            },
        };

    // These are the production W4A16 contracts used by dense projections and
    // grouped MoE projections. Unlike the capability probes above, a numerical
    // mismatch is a hard validation failure rather than an unavailable optional
    // fast path.
    let linear =
        linear::validate_nvfp4_w4a16(&stream).map_err(|error| CudaError::QuantizedLinear {
            message: error.to_string(),
        })?;

    Ok(Nvfp4CapabilityReport {
        device_id,
        scaled_mma,
        byte_decode_mma,
        w4a16_linear: CudaKernelCapability::Available {
            max_abs_error: linear.max_abs_error,
        },
        grouped_w4a16: CudaKernelCapability::Available {
            max_abs_error: linear.grouped_max_abs_error,
        },
    })
}

fn validate_nvfp4_output(
    out: Tensor<f32>,
    stream: &Arc<cuda_core::Stream>,
    operation: &'static str,
    expected: f32,
) -> Result<CudaKernelCapability, CudaError> {
    let host: Vec<f32> = out
        .to_host_vec()
        .sync_on(stream)
        .map_err(|error| nvfp4_kernel_error("copy output to host", error))?;
    let mut max_abs_error = 0.0f32;
    for (index, actual) in host.into_iter().enumerate() {
        let error = (actual - expected).abs();
        max_abs_error = max_abs_error.max(error);
        if !actual.is_finite() || error > 1.0e-3 {
            return Err(CudaError::WrongNvfp4Value {
                operation,
                index,
                actual,
                expected,
            });
        }
    }

    Ok(CudaKernelCapability::Available { max_abs_error })
}

/// Enables cuTile's process-wide persistent CUBIN cache.
///
/// The cache contains executable device code, so cuTile deliberately uses a
/// private per-user directory (`$XDG_CACHE_HOME/cutile/kernels`, falling back
/// to `~/.cache/cutile/kernels`) and rejects unsafe permissions. Calling this
/// before any kernel launch lets a replacement spot worker reuse warmed
/// kernels when its home cache is restored or mounted persistently.
pub fn enable_persistent_cubin_cache() -> Result<(), CudaError> {
    cutile::jit_cache::enable_default().map_err(CudaError::CubinCache)
}

/// Compile and execute a Tesseract-owned BF16 cuTile kernel on the requested
/// CUDA device, then copy FP32-converted results back for validation.
pub fn probe_bf16_cutile(device_id: usize) -> Result<Bf16SmokeReport, CudaError> {
    enable_persistent_cubin_cache()?;
    let device = Device::new(device_id).map_err(|error| CudaError::Device {
        device_id,
        message: format!("{error:?}"),
    })?;
    let stream = device.new_stream().map_err(|error| CudaError::Device {
        device_id,
        message: format!("failed to create stream: {error:?}"),
    })?;

    let lhs = api::ones::<bf16>(&[SMOKE_ELEMENTS])
        .sync_on(&stream)
        .map_err(|error| kernel_error("allocate lhs", error))?;
    let rhs = api::ones::<bf16>(&[SMOKE_ELEMENTS])
        .sync_on(&stream)
        .map_err(|error| kernel_error("allocate rhs", error))?;
    let mut out = api::zeros::<bf16>(&[SMOKE_ELEMENTS])
        .sync_on(&stream)
        .map_err(|error| kernel_error("allocate output", error))?;

    add_bf16((&mut out).partition([SMOKE_BLOCK]), &lhs, &rhs)
        .sync_on(&stream)
        .map_err(|error| kernel_error("compile/launch add", error))?;

    let out = Arc::new(out);
    let converted = api::convert::<bf16, f32>(out.clone())
        .sync_on(&stream)
        .map_err(|error| kernel_error("convert output to FP32", error))?;
    let converted = Arc::new(converted);
    let host: Vec<f32> = converted
        .clone()
        .to_host_vec()
        .sync_on(&stream)
        .map_err(|error| kernel_error("copy output to host", error))?;

    for (index, actual) in host.into_iter().enumerate() {
        if actual != 2.0 {
            return Err(CudaError::WrongValue { index, actual });
        }
    }

    let matrix = api::copy_host_vec_to_device(&std::sync::Arc::new(vec![
        bf16::from_f32(1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(3.0),
        bf16::from_f32(4.0),
        bf16::from_f32(5.0),
        bf16::from_f32(6.0),
    ]))
    .sync_on(&stream)
    .map_err(|error| kernel_error("copy GEMM matrix", error))?
    .reshape(&[2, 3])
    .map_err(|error| kernel_error("reshape GEMM matrix", error))?;
    let rhs = api::copy_host_vec_to_device(&std::sync::Arc::new(vec![
        bf16::from_f32(1.0),
        bf16::from_f32(0.0),
        bf16::from_f32(-1.0),
    ]))
    .sync_on(&stream)
    .map_err(|error| kernel_error("copy GEMM rhs", error))?
    .reshape(&[1, 3])
    .map_err(|error| kernel_error("reshape GEMM rhs", error))?;
    let gemm_out = api::zeros::<bf16>(&[1, 2])
        .sync_on(&stream)
        .map_err(|error| kernel_error("allocate GEMM output", error))?;
    let matrix = std::sync::Arc::new(matrix);
    let rhs = std::sync::Arc::new(rhs);
    let gemm_out = cublas::gemm_bf16(matrix.clone(), rhs.clone(), gemm_out, 2, 1, 3)?
        .sync_on(&stream)
        .map_err(|error| kernel_error("execute BF16 cuBLAS GEMM", error))??;
    let gemm_out = Arc::new(gemm_out);
    let gemm_host: Vec<bf16> = gemm_out
        .clone()
        .to_host_vec()
        .sync_on(&stream)
        .map_err(|error| kernel_error("copy GEMM result to host", error))?;
    let expected = [-2.0f32, -2.0f32];
    for (index, (actual, expected)) in gemm_host.into_iter().zip(expected).enumerate() {
        let actual = actual.to_f32();
        if (actual - expected).abs() > 0.01 {
            return Err(CudaError::WrongValue {
                index: SMOKE_ELEMENTS + index,
                actual,
            });
        }
    }

    let graph_out = Arc::new(
        api::zeros::<bf16>(&[1, 2])
            .sync_on(&stream)
            .map_err(|error| kernel_error("allocate graph GEMM output", error))?,
    );
    let graph = CudaGraph::scope(&stream, |scope| {
        let result = scope.record(
            cublas::gemm_bf16_into(&matrix, &rhs, &graph_out, 2, 1, 3)
                .map_err(|error| DeviceError::Internal(error.to_string()))?,
        )?;
        result.map_err(|error| DeviceError::Internal(error.to_string()))
    })
    .map_err(|error| kernel_error("capture BF16 cuBLAS graph", error))?;
    graph
        .launch()
        .sync_on(&stream)
        .map_err(|error| kernel_error("replay BF16 cuBLAS graph", error))?;
    let graph_host: Vec<bf16> = graph_out
        .clone()
        .to_host_vec()
        .sync_on(&stream)
        .map_err(|error| kernel_error("copy graph GEMM result", error))?;
    if graph_host
        .into_iter()
        .zip(expected)
        .any(|(actual, expected)| (actual.to_f32() - expected).abs() > 0.01)
    {
        return Err(CudaError::Bf16Kernel {
            operation: "validate cuBLAS graph replay",
            message: "captured GEMM result differs from eager GEMM".into(),
        });
    }

    validate_transformer_primitives(&stream)?;

    Ok(Bf16SmokeReport {
        device_id,
        elements: SMOKE_ELEMENTS,
        gemm_rows: 2,
    })
}

fn validate_transformer_primitives(
    stream: &std::sync::Arc<cuda_core::Stream>,
) -> Result<(), CudaError> {
    use cutile::tile_kernel::{PartitionOp, TileKernel};

    let token_ids = api::copy_host_vec_to_device(&std::sync::Arc::new(vec![0u32, 1u32]))
        .sync_on(stream)
        .map_err(|error| kernel_error("copy transformer token IDs", error))?;
    let table = api::ones::<bf16>(&[2, 128])
        .sync_on(stream)
        .map_err(|error| kernel_error("allocate transformer embedding", error))?;
    let hidden = api::zeros::<bf16>(&[2, 128])
        .partition([1, 64])
        .sync_on(stream)
        .map_err(|error| kernel_error("allocate transformer hidden state", error))?;
    let (_, _, hidden) = kernels::embedding_bf16(&token_ids, &table, hidden)
        .generics(vec!["128".into(), "64".into()])
        .sync_on(stream)
        .map_err(|error| kernel_error("execute embedding", error))?;
    let hidden = hidden.unpartition();

    let norm_weight = api::ones::<bf16>(&[128])
        .sync_on(stream)
        .map_err(|error| kernel_error("allocate RMSNorm weight", error))?;
    let normalized = api::zeros::<bf16>(&[2, 128])
        .partition([1, 128])
        .sync_on(stream)
        .map_err(|error| kernel_error("allocate RMSNorm output", error))?;
    let (_, _, normalized, _) =
        unsafe { kernels::rms_norm_bf16(&hidden, &norm_weight, normalized, 1.0e-5) }
            .generics(vec!["128".into(), "64".into()])
            .sync_on(stream)
            .map_err(|error| kernel_error("execute RMSNorm", error))?;
    let normalized = normalized.unpartition();

    let up = api::ones::<bf16>(&[2, 128])
        .sync_on(stream)
        .map_err(|error| kernel_error("allocate SiLU input", error))?;
    let activated = api::zeros::<bf16>(&[2, 128])
        .partition([1, 64])
        .sync_on(stream)
        .map_err(|error| kernel_error("allocate SiLU output", error))?;
    let (_, _, activated) = kernels::silu_mul_bf16(&normalized, &up, activated)
        .generics(vec!["64".into()])
        .sync_on(stream)
        .map_err(|error| kernel_error("execute SiLU", error))?;
    let activated = Arc::new(activated.unpartition());
    let activated_host: Vec<bf16> = activated
        .clone()
        .to_host_vec()
        .sync_on(stream)
        .map_err(|error| kernel_error("copy SiLU result", error))?;
    if activated_host
        .iter()
        .any(|value| !(0.72..=0.75).contains(&value.to_f32()))
    {
        return Err(CudaError::Bf16Kernel {
            operation: "validate SiLU result",
            message: "unexpected activation value".into(),
        });
    }

    let query = api::ones::<bf16>(&[1, 4, 64])
        .sync_on(stream)
        .map_err(|error| kernel_error("allocate RoPE query", error))?;
    let positions = api::copy_host_vec_to_device(&std::sync::Arc::new(vec![0u32]))
        .sync_on(stream)
        .map_err(|error| kernel_error("copy RoPE positions", error))?;
    let cos = api::ones::<f32>(&[1, 32])
        .sync_on(stream)
        .map_err(|error| kernel_error("allocate RoPE cosine", error))?;
    let sin = api::zeros::<f32>(&[1, 32])
        .sync_on(stream)
        .map_err(|error| kernel_error("allocate RoPE sine", error))?;
    let rotated = api::zeros::<bf16>(&[1, 4, 64])
        .partition([1, 1, 64])
        .sync_on(stream)
        .map_err(|error| kernel_error("allocate RoPE output", error))?;
    let (_, _, _, _, rotated) = kernels::rope_q_bf16(&query, &positions, &cos, &sin, rotated)
        .generics(vec!["64".into(), "32".into()])
        .sync_on(stream)
        .map_err(|error| kernel_error("execute RoPE", error))?;
    let rotated = rotated.unpartition();

    let flat_key = api::zeros::<bf16>(&[4, 2, 64])
        .sync_on(stream)
        .map_err(|error| kernel_error("allocate flat key cache", error))?;
    let flat_value = api::ones::<bf16>(&[4, 2, 64])
        .sync_on(stream)
        .map_err(|error| kernel_error("allocate flat value cache", error))?;
    let slots = api::copy_host_vec_to_device(&std::sync::Arc::new(vec![3u32, 1u32]))
        .sync_on(stream)
        .map_err(|error| kernel_error("copy flat KV slots", error))?;
    let key = api::zeros::<bf16>(&[2, 2, 64])
        .partition([1, 1, 64])
        .sync_on(stream)
        .map_err(|error| kernel_error("allocate gathered key", error))?;
    let value = api::zeros::<bf16>(&[2, 2, 64])
        .partition([1, 1, 64])
        .sync_on(stream)
        .map_err(|error| kernel_error("allocate gathered value", error))?;
    let (_, _, key) = kernels::gather_flat_kv_bf16(&slots, &flat_key, key)
        .generics(vec!["64".into()])
        .sync_on(stream)
        .map_err(|error| kernel_error("gather flat key cache", error))?;
    let (_, _, value) = kernels::gather_flat_kv_bf16(&slots, &flat_value, value)
        .generics(vec!["64".into()])
        .sync_on(stream)
        .map_err(|error| kernel_error("gather flat value cache", error))?;
    let key = key.unpartition();
    let value = value.unpartition();
    let attention = api::zeros::<bf16>(&[1, 4, 64])
        .partition([1, 1, 64])
        .sync_on(stream)
        .map_err(|error| kernel_error("allocate attention output", error))?;
    let metadata = api::copy_host_vec_to_device(&std::sync::Arc::new(vec![2i32, 0i32]))
        .sync_on(stream)
        .map_err(|error| kernel_error("upload attention metadata", error))?;
    let (_, _, _, _, attention, _, _) = unsafe {
        kernels::causal_attention_bf16(&rotated, &key, &value, &metadata, attention, 0.125, 2)
    }
    .generics(vec!["1".into(), "16".into(), "64".into()])
    .sync_on(stream)
    .map_err(|error| kernel_error("execute causal attention", error))?;
    let attention = Arc::new(attention.unpartition());
    let attention_host: Vec<bf16> = attention
        .clone()
        .to_host_vec()
        .sync_on(stream)
        .map_err(|error| kernel_error("copy attention result", error))?;
    if attention_host.iter().any(|value| value.to_f32() != 1.0) {
        return Err(CudaError::Bf16Kernel {
            operation: "validate causal attention result",
            message: "uniform value cache did not produce ones".into(),
        });
    }
    Ok(())
}

fn kernel_error(operation: &'static str, error: impl std::fmt::Debug) -> CudaError {
    CudaError::Bf16Kernel {
        operation,
        message: format!("{error:?}"),
    }
}

fn nvfp4_kernel_error(operation: &'static str, error: impl std::fmt::Debug) -> CudaError {
    CudaError::Bf16Kernel {
        operation,
        message: format!("NVFP4 probe: {error:?}"),
    }
}
