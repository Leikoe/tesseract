//! Framework-independent Marlin weight preparation, execution, and timing.

use std::{ffi::c_void, sync::Arc};

use cuda_async::device_operation::DeviceOp;
use cuda_core::{IntoResult, Stream, sys};
use cutile::{
    DType, api,
    core::{bf16, f16},
    tensor::{PartitionMut, Reshape, Tensor, ToHostVec},
    tile_kernel::TileKernel,
};
use serde::Serialize;

use crate::cuda::{
    execution::StreamExecution,
    kernels,
    linear::{
        ExpertProjection, Fp8W8A16Linear, GroupedNvfp4W4A16, Nvfp4W4A16Linear,
        parse_nvfp4_projection,
    },
};
use crate::{
    model::{ModelError, weights::WeightSource},
    quantization::decode_e4m3fn,
};

unsafe extern "C" {
    fn tesseract_marlin_repack(
        input: *const c_void,
        output: *mut c_void,
        size_k: i32,
        size_n: i32,
        num_bits: i32,
        stream: *mut c_void,
    ) -> i32;

    fn tesseract_marlin_repack_experts(
        input: *const c_void,
        output: *mut c_void,
        experts: i32,
        size_k: i32,
        size_n: i32,
        num_bits: i32,
        stream: *mut c_void,
    ) -> i32;

    fn tesseract_marlin_gemm_bf16(
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        c_tmp: *mut c_void,
        scales: *const c_void,
        global_scale: *const f32,
        workspace: *mut i32,
        m: i32,
        n: i32,
        k: i32,
        quant_bits: i32,
        stream: *mut c_void,
    ) -> i32;

    fn tesseract_marlin_moe_gemm_bf16(
        activations: *const c_void,
        packed_expert_weights: *const c_void,
        output: *mut c_void,
        fp32_temporary: *mut c_void,
        block_scales: *const c_void,
        expert_global_scales_bf16: *const c_void,
        workspace: *mut i32,
        expert_ids: *const i32,
        rows: i32,
        output_size: i32,
        input_size: i32,
        moe_block_size: i32,
        stream: *mut c_void,
    ) -> i32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MarlinQuantization {
    Fp8,
    Nvfp4,
}

impl MarlinQuantization {
    const fn bits(self) -> usize {
        match self {
            Self::Fp8 => 8,
            Self::Nvfp4 => 4,
        }
    }
}

enum MarlinScales {
    Bf16(Arc<Tensor<bf16>>),
    Fp8(Arc<Tensor<u8>>),
}

impl MarlinScales {
    fn device_pointer(&self) -> *const c_void {
        match self {
            Self::Bf16(tensor) => tensor.device_pointer().cu_deviceptr() as usize as *const c_void,
            Self::Fp8(tensor) => tensor.device_pointer().cu_deviceptr() as usize as *const c_void,
        }
    }
}

pub struct MarlinLinear {
    quantization: MarlinQuantization,
    input_size: usize,
    output_size: usize,
    packed_weight: Arc<Tensor<u32>>,
    scales: MarlinScales,
    global_scale: Option<Arc<Tensor<f32>>>,
    workspace: Arc<Tensor<i32>>,
}

impl MarlinLinear {
    pub fn fp8(
        encoded_nk: &[u8],
        input_size: usize,
        output_size: usize,
        weight_scale: f32,
        stream: &Arc<Stream>,
    ) -> Result<Self, ModelError> {
        if encoded_nk.len() != input_size.saturating_mul(output_size)
            || input_size % 16 != 0
            || output_size % 64 != 0
            || !weight_scale.is_finite()
            || weight_scale <= 0.0
        {
            return Err(ModelError::Cuda("invalid Marlin FP8 artifact".into()));
        }
        let packed_kn = pack_k_major(encoded_nk, input_size, output_size, 8)?;
        let packed_weight = repack(&packed_kn, input_size, output_size, 8, stream)?;

        // Marlin's register decoder injects E4M3 bits into BF16. Folding the
        // exponent-bias difference into the channel scale removes a per-value
        // correction from the hot loop.
        const FP8_TO_BF16_EXPONENT_BIAS: f32 = 1.329_228e36; // 2^120
        let scale = bf16::from_f32(weight_scale * FP8_TO_BF16_EXPONENT_BIAS);
        let scales = permute_channel_scales(&vec![scale; output_size], output_size)?;
        let scales = upload(scales, stream, "upload Marlin FP8 scales")?;
        Ok(Self {
            quantization: MarlinQuantization::Fp8,
            input_size,
            output_size,
            packed_weight,
            scales: MarlinScales::Bf16(scales),
            global_scale: None,
            workspace: workspace(stream)?,
        })
    }

    pub fn nvfp4(
        packed_nk: &[u8],
        scale_bytes_ng: &[u8],
        input_size: usize,
        output_size: usize,
        weight_global_scale: f32,
        stream: &Arc<Stream>,
    ) -> Result<Self, ModelError> {
        let groups = input_size / 16;
        if input_size % 16 != 0
            || output_size % 64 != 0
            || packed_nk.len() != output_size.saturating_mul(input_size / 2)
            || scale_bytes_ng.len() != output_size.saturating_mul(groups)
            || !weight_global_scale.is_finite()
            || weight_global_scale <= 0.0
        {
            return Err(ModelError::Cuda("invalid Marlin NVFP4 artifact".into()));
        }
        let packed_kn = pack_k_major(packed_nk, input_size, output_size, 4)?;
        let packed_weight = repack(&packed_kn, input_size, output_size, 4, stream)?;
        let (scales, adjusted_global_scale) =
            prepare_nvfp4_scales(scale_bytes_ng, input_size, output_size, weight_global_scale)?;
        let scales = upload(scales, stream, "upload Marlin NVFP4 scales")?;
        let global_scale = upload(
            vec![adjusted_global_scale],
            stream,
            "upload Marlin NVFP4 global scale",
        )?;
        Ok(Self {
            quantization: MarlinQuantization::Nvfp4,
            input_size,
            output_size,
            packed_weight,
            scales: MarlinScales::Fp8(scales),
            global_scale: Some(global_scale),
            workspace: workspace(stream)?,
        })
    }

    pub fn execute(
        &self,
        input: &Tensor<bf16>,
        output: &Tensor<bf16>,
        temporary: &Tensor<f32>,
        rows: usize,
        stream: &Arc<Stream>,
    ) -> Result<(), ModelError> {
        if rows == 0
            || input.shape() != [rows as i32, self.input_size as i32]
            || output.shape() != [rows as i32, self.output_size as i32]
            || temporary.shape() != [rows as i32, self.output_size as i32]
        {
            return Err(ModelError::Cuda("invalid Marlin GEMM geometry".into()));
        }
        stream
            .device()
            .bind_to_thread()
            .map_err(|error| ModelError::Cuda(format!("bind Marlin device: {error:?}")))?;
        let status = unsafe {
            tesseract_marlin_gemm_bf16(
                input.device_pointer().cu_deviceptr() as usize as *const c_void,
                self.packed_weight.device_pointer().cu_deviceptr() as usize as *const c_void,
                output.device_pointer().cu_deviceptr() as usize as *mut c_void,
                temporary.device_pointer().cu_deviceptr() as usize as *mut c_void,
                self.scales.device_pointer(),
                self.global_scale
                    .as_ref()
                    .map_or(std::ptr::null(), |scale| {
                        scale.device_pointer().cu_deviceptr() as usize as *const f32
                    }),
                self.workspace.device_pointer().cu_deviceptr() as usize as *mut i32,
                rows as i32,
                self.output_size as i32,
                self.input_size as i32,
                self.quantization.bits() as i32,
                stream.cu_stream().cast(),
            )
        };
        status_result(status, "launch Marlin GEMM")
    }
}

pub struct MarlinMoe {
    input_size: usize,
    output_size: usize,
    num_experts: usize,
    packed_weight: Arc<Tensor<u32>>,
    scales: Arc<Tensor<u8>>,
    global_scales: Arc<Tensor<bf16>>,
    workspace: Arc<Tensor<i32>>,
}

impl MarlinMoe {
    pub(crate) fn load(
        source: &dyn WeightSource,
        experts_prefix: &str,
        projection: ExpertProjection,
        num_experts: usize,
        stream: &Arc<Stream>,
    ) -> Result<Self, ModelError> {
        if num_experts == 0 {
            return Err(ModelError::InvalidTensor {
                name: experts_prefix.into(),
                message: "expert bank is empty".into(),
            });
        }
        let mut packed = Vec::new();
        let mut scales = Vec::new();
        let mut globals = Vec::with_capacity(num_experts);
        let mut geometry = None;
        for expert in 0..num_experts {
            let prefix = format!("{experts_prefix}.{expert}.{}", projection.suffix());
            let parsed = parse_nvfp4_projection(source, &prefix)?;
            match geometry {
                None => geometry = Some((parsed.input_size, parsed.output_size)),
                Some(expected) if expected != (parsed.input_size, parsed.output_size) => {
                    return Err(ModelError::InvalidTensor {
                        name: prefix,
                        message: "expert projection geometry differs from expert 0".into(),
                    });
                }
                Some(_) => {}
            }
            packed.extend_from_slice(parsed.packed_weight);
            scales.extend_from_slice(parsed.scale_bytes);
            globals.push(parsed.weight_global_scale);
        }
        let (input_size, output_size) = geometry.expect("non-empty expert bank has geometry");
        Self::nvfp4(&packed, &scales, &globals, input_size, output_size, stream)
    }

    pub fn nvfp4(
        packed_enk: &[u8],
        scale_bytes_eng: &[u8],
        weight_global_scales: &[f32],
        input_size: usize,
        output_size: usize,
        stream: &Arc<Stream>,
    ) -> Result<Self, ModelError> {
        let num_experts = weight_global_scales.len();
        let weight_bytes_per_expert = output_size.saturating_mul(input_size / 2);
        let scale_bytes_per_expert = output_size.saturating_mul(input_size / 16);
        if num_experts == 0
            || input_size % 16 != 0
            || output_size % 64 != 0
            || packed_enk.len() != num_experts.saturating_mul(weight_bytes_per_expert)
            || scale_bytes_eng.len() != num_experts.saturating_mul(scale_bytes_per_expert)
            || weight_global_scales
                .iter()
                .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            return Err(ModelError::Cuda(
                "invalid grouped Marlin NVFP4 artifact".into(),
            ));
        }

        let mut packed_kn = Vec::with_capacity(num_experts * input_size * output_size / 8);
        let mut scales = Vec::with_capacity(num_experts * scale_bytes_per_expert);
        let mut global_scales = Vec::with_capacity(num_experts);
        for expert in 0..num_experts {
            let weight_start = expert * weight_bytes_per_expert;
            packed_kn.extend(pack_k_major(
                &packed_enk[weight_start..weight_start + weight_bytes_per_expert],
                input_size,
                output_size,
                4,
            )?);
            let scale_start = expert * scale_bytes_per_expert;
            let (expert_scales, adjusted_global_scale) = prepare_nvfp4_scales(
                &scale_bytes_eng[scale_start..scale_start + scale_bytes_per_expert],
                input_size,
                output_size,
                weight_global_scales[expert],
            )?;
            scales.extend(expert_scales);
            global_scales.push(bf16::from_f32(adjusted_global_scale));
        }

        let source = upload(packed_kn, stream, "upload grouped Marlin source weights")?;
        let packed_weight = api::zeros::<u32>(&[source.shape()[0] as usize])
            .sync_on(stream)
            .map_err(|error| {
                ModelError::Cuda(format!("allocate grouped Marlin weights: {error:?}"))
            })?;
        let status = unsafe {
            tesseract_marlin_repack_experts(
                source.device_pointer().cu_deviceptr() as usize as *const c_void,
                packed_weight.device_pointer().cu_deviceptr() as usize as *mut c_void,
                num_experts as i32,
                input_size as i32,
                output_size as i32,
                4,
                stream.cu_stream().cast(),
            )
        };
        status_result(status, "repack grouped Marlin weights")?;
        unsafe { stream.synchronize() }.map_err(|error| {
            ModelError::Cuda(format!("synchronize grouped Marlin repack: {error:?}"))
        })?;
        Ok(Self {
            input_size,
            output_size,
            num_experts,
            packed_weight: Arc::new(packed_weight),
            scales: upload(scales, stream, "upload grouped Marlin scales")?,
            global_scales: upload(global_scales, stream, "upload grouped Marlin global scales")?,
            workspace: workspace_len(stream, sm_count(stream)?.saturating_mul(4))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        input: &Tensor<bf16>,
        output: &Tensor<bf16>,
        temporary: &Tensor<f32>,
        expert_ids: &Tensor<i32>,
        rows: usize,
        block_rows: usize,
        stream: &Arc<Stream>,
    ) -> Result<(), ModelError> {
        if rows == 0
            || !rows.is_multiple_of(block_rows)
            || !matches!(block_rows, 8 | 16 | 32 | 48 | 64)
            || input.shape() != [rows as i32, self.input_size as i32]
            || output.shape() != [rows as i32, self.output_size as i32]
            || temporary.shape() != [rows as i32, self.output_size as i32]
            || expert_ids.shape() != [(rows / block_rows) as i32]
        {
            return Err(ModelError::Cuda(
                "invalid grouped Marlin execution geometry".into(),
            ));
        }
        stream
            .device()
            .bind_to_thread()
            .map_err(|error| ModelError::Cuda(format!("bind Marlin device: {error:?}")))?;
        let status = unsafe {
            tesseract_marlin_moe_gemm_bf16(
                input.device_pointer().cu_deviceptr() as usize as *const c_void,
                self.packed_weight.device_pointer().cu_deviceptr() as usize as *const c_void,
                output.device_pointer().cu_deviceptr() as usize as *mut c_void,
                temporary.device_pointer().cu_deviceptr() as usize as *mut c_void,
                self.scales.device_pointer().cu_deviceptr() as usize as *const c_void,
                self.global_scales.device_pointer().cu_deviceptr() as usize as *const c_void,
                self.workspace.device_pointer().cu_deviceptr() as usize as *mut i32,
                expert_ids.device_pointer().cu_deviceptr() as usize as *const i32,
                rows as i32,
                self.output_size as i32,
                self.input_size as i32,
                block_rows as i32,
                stream.cu_stream().cast(),
            )
        };
        status_result(status, "launch grouped Marlin GEMM")
    }

    pub const fn num_experts(&self) -> usize {
        self.num_experts
    }
}

fn pack_k_major(
    encoded_nk: &[u8],
    input_size: usize,
    output_size: usize,
    bits: usize,
) -> Result<Vec<u32>, ModelError> {
    let values_per_byte = 8 / bits;
    let values_per_word = 32 / bits;
    let source_row_bytes = input_size / values_per_byte;
    let mut packed = vec![0u32; (input_size / values_per_word) * output_size];
    for packed_k in 0..input_size / values_per_word {
        for column in 0..output_size {
            let mut word = 0u32;
            for offset in 0..values_per_word {
                let k = packed_k * values_per_word + offset;
                let byte = encoded_nk[column * source_row_bytes + k / values_per_byte];
                let value = if bits == 8 {
                    byte
                } else if k % 2 == 0 {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                word |= u32::from(value) << (offset * bits);
            }
            packed[packed_k * output_size + column] = word;
        }
    }
    Ok(packed)
}

fn repack(
    packed_kn: &[u32],
    input_size: usize,
    output_size: usize,
    bits: usize,
    stream: &Arc<Stream>,
) -> Result<Arc<Tensor<u32>>, ModelError> {
    let input = upload(packed_kn.to_vec(), stream, "upload Marlin source weights")?;
    let output = api::zeros::<u32>(&[packed_kn.len()])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate Marlin weights: {error:?}")))?;
    let status = unsafe {
        tesseract_marlin_repack(
            input.device_pointer().cu_deviceptr() as usize as *const c_void,
            output.device_pointer().cu_deviceptr() as usize as *mut c_void,
            input_size as i32,
            output_size as i32,
            bits as i32,
            stream.cu_stream().cast(),
        )
    };
    status_result(status, "repack Marlin weights")?;
    unsafe { stream.synchronize() }
        .map_err(|error| ModelError::Cuda(format!("synchronize Marlin repack: {error:?}")))?;
    Ok(Arc::new(output))
}

fn workspace(stream: &Arc<Stream>) -> Result<Arc<Tensor<i32>>, ModelError> {
    workspace_len(stream, sm_count(stream)?)
}

fn sm_count(stream: &Arc<Stream>) -> Result<usize, ModelError> {
    let mut sms = 0i32;
    unsafe {
        sys::cuDeviceGetAttribute(
            &mut sms,
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            stream.device().cu_device(),
        )
        .result()
        .map_err(|error| ModelError::Cuda(format!("query Marlin SM count: {error:?}")))?;
    }
    usize::try_from(sms)
        .ok()
        .filter(|sms| *sms > 0)
        .ok_or_else(|| ModelError::Cuda(format!("invalid Marlin SM count {sms}")))
}

fn workspace_len(stream: &Arc<Stream>, length: usize) -> Result<Arc<Tensor<i32>>, ModelError> {
    let workspace = api::zeros::<i32>(&[length])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate Marlin workspace: {error:?}")))?;
    Ok(Arc::new(workspace))
}

fn upload<T: DType + Send + Sync + 'static>(
    values: Vec<T>,
    stream: &Arc<Stream>,
    operation: &str,
) -> Result<Arc<Tensor<T>>, ModelError> {
    let tensor = api::copy_host_vec_to_device(&Arc::new(values))
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("{operation}: {error:?}")))?;
    Ok(Arc::new(tensor))
}

fn permute_channel_scales(scales: &[bf16], output_size: usize) -> Result<Vec<bf16>, ModelError> {
    const PERM: [usize; 32] = [
        0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27, 4, 5, 12, 13, 20, 21, 28, 29, 6,
        7, 14, 15, 22, 23, 30, 31,
    ];
    if scales.len() != output_size || output_size % PERM.len() != 0 {
        return Err(ModelError::Cuda("invalid Marlin channel scales".into()));
    }
    let mut result = Vec::with_capacity(scales.len());
    for chunk in scales.chunks_exact(PERM.len()) {
        result.extend(PERM.map(|index| chunk[index]));
    }
    Ok(result)
}

fn prepare_nvfp4_scales(
    scale_bytes_ng: &[u8],
    input_size: usize,
    output_size: usize,
    global_scale: f32,
) -> Result<(Vec<u8>, f32), ModelError> {
    const PERM: [usize; 64] = [
        0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50, 58,
        3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53,
        61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63,
    ];
    let groups = input_size / 16;
    let mut transposed = vec![0.0f32; groups * output_size];
    for group in 0..groups {
        for column in 0..output_size {
            let value = decode_e4m3fn(scale_bytes_ng[column * groups + group]);
            if !value.is_finite() || value < 0.0 {
                return Err(ModelError::Cuda(
                    "Marlin NVFP4 scales must be finite and non-negative".into(),
                ));
            }
            transposed[group * output_size + column] = bf16::from_f32(value).to_f32();
        }
    }
    let max_scale = transposed.iter().copied().fold(0.0f32, f32::max);
    let scaled_max = max_scale * 128.0;
    let factor = if scaled_max > 0.0 && scaled_max < 448.0 * 128.0 {
        2.0f32.powf((448.0 * 128.0 / scaled_max).log2().floor())
    } else {
        1.0
    };

    let mut permuted = Vec::with_capacity(transposed.len());
    for chunk in transposed.chunks_exact(PERM.len()) {
        permuted.extend(PERM.map(|index| chunk[index]));
    }
    for chunk in permuted.chunks_exact_mut(4) {
        chunk.swap(1, 2);
    }
    let encoded = permuted
        .into_iter()
        .map(|scale| {
            let scaled = scale * factor * 128.0;
            if scaled < 2.0 {
                0
            } else {
                ((f16::from_f32(scaled).to_bits() << 1) >> 8) as u8
            }
        })
        .collect();
    const FP4_TO_BF16_GLOBAL_BIAS: f32 = 6.646_14e35; // 2^119
    Ok((encoded, global_scale * FP4_TO_BF16_GLOBAL_BIAS / factor))
}

fn status_result(status: i32, operation: &str) -> Result<(), ModelError> {
    if status == 0 {
        Ok(())
    } else {
        Err(ModelError::Cuda(format!(
            "{operation} failed with CUDA/native status {status}"
        )))
    }
}

#[derive(Debug, Serialize)]
pub struct MarlinProbe {
    pub fp8_max_abs_error: f32,
    pub nvfp4_max_abs_error: f32,
    pub grouped_nvfp4_max_abs_error: f32,
}

#[derive(Debug, Serialize)]
pub struct KernelBenchmarkReport {
    pub device_id: usize,
    pub warmup_iterations: usize,
    pub timed_iterations: usize,
    pub cutile_fp8_max_abs_error: f32,
    pub cutile_nvfp4_max_abs_error: f32,
    pub marlin_fp8_max_abs_error: f32,
    pub marlin_nvfp4_max_abs_error: f32,
    pub marlin_grouped_nvfp4_max_abs_error: f32,
    pub samples: Vec<KernelBenchmarkSample>,
}

#[derive(Debug, Serialize)]
pub struct KernelBenchmarkSample {
    pub implementation: &'static str,
    pub quantization: MarlinQuantization,
    pub rows: usize,
    pub output_size: usize,
    pub input_size: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routing_pattern: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_experts: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_rows: Option<usize>,
    pub raw_ms: Vec<f32>,
    pub minimum_ms: f32,
    pub median_ms: f32,
    pub p90_ms: f32,
    pub logical_tflops: f64,
    pub packed_weight_gb_per_second: f64,
}

pub fn benchmark(
    device_id: usize,
    rows: &[usize],
    warmup_iterations: usize,
    timed_iterations: usize,
) -> Result<KernelBenchmarkReport, ModelError> {
    if rows.is_empty()
        || rows
            .iter()
            .any(|rows| *rows == 0 || !rows.is_multiple_of(16))
        || timed_iterations == 0
    {
        return Err(ModelError::Cuda(
            "invalid Marlin benchmark arguments".into(),
        ));
    }
    let device = cuda_core::Device::new(device_id)
        .map_err(|error| ModelError::Cuda(format!("initialize benchmark device: {error:?}")))?;
    let stream = device
        .new_stream()
        .map_err(|error| ModelError::Cuda(format!("create benchmark stream: {error:?}")))?;
    let cutile_probe = super::linear::probe_quantized_linears(&stream)?;
    let marlin_probe = probe(&stream)?;
    let mut samples = Vec::new();
    for &m in rows {
        benchmark_fp8_case(
            &stream,
            m,
            2048,
            8192,
            warmup_iterations,
            timed_iterations,
            &mut samples,
        )?;
        benchmark_nvfp4_case(
            &stream,
            m,
            2048,
            512,
            warmup_iterations,
            timed_iterations,
            &mut samples,
        )?;
        benchmark_nvfp4_case(
            &stream,
            m,
            512,
            2048,
            warmup_iterations,
            timed_iterations,
            &mut samples,
        )?;
    }
    benchmark_grouped_nvfp4_shape(
        &stream,
        rows,
        2048,
        512,
        warmup_iterations,
        timed_iterations,
        &mut samples,
    )?;
    benchmark_grouped_nvfp4_shape(
        &stream,
        rows,
        512,
        2048,
        warmup_iterations,
        timed_iterations,
        &mut samples,
    )?;
    Ok(KernelBenchmarkReport {
        device_id,
        warmup_iterations,
        timed_iterations,
        cutile_fp8_max_abs_error: cutile_probe.fp8_max_abs_error,
        cutile_nvfp4_max_abs_error: cutile_probe.max_abs_error,
        marlin_fp8_max_abs_error: marlin_probe.fp8_max_abs_error,
        marlin_nvfp4_max_abs_error: marlin_probe.nvfp4_max_abs_error,
        marlin_grouped_nvfp4_max_abs_error: marlin_probe.grouped_nvfp4_max_abs_error,
        samples,
    })
}

#[allow(clippy::too_many_arguments)]
fn benchmark_fp8_case(
    stream: &Arc<Stream>,
    rows: usize,
    input_size: usize,
    output_size: usize,
    warmup: usize,
    iterations: usize,
    samples: &mut Vec<KernelBenchmarkSample>,
) -> Result<(), ModelError> {
    const SCALE: f32 = 0.25;
    let encoded = (0..output_size * input_size)
        .map(|index| [0x20, 0x38, 0x3c, 0x40, 0x60, 0xb8, 0xc0, 0xe0][index % 8])
        .collect::<Vec<_>>();
    let input = benchmark_input(rows, input_size, stream)?;
    let cutile = Fp8W8A16Linear::from_host(&encoded, input_size, output_size, SCALE, stream)?;
    let marlin = MarlinLinear::fp8(&encoded, input_size, output_size, SCALE, stream)?;
    samples.push(time_cutile_fp8(
        &cutile,
        input.clone(),
        rows,
        input_size,
        output_size,
        warmup,
        iterations,
        stream,
    )?);
    samples.push(time_marlin(
        &marlin,
        input,
        rows,
        input_size,
        output_size,
        warmup,
        iterations,
        stream,
    )?);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn benchmark_nvfp4_case(
    stream: &Arc<Stream>,
    rows: usize,
    input_size: usize,
    output_size: usize,
    warmup: usize,
    iterations: usize,
    samples: &mut Vec<KernelBenchmarkSample>,
) -> Result<(), ModelError> {
    const GLOBAL: f32 = 0.5;
    let packed = (0..output_size * input_size / 2)
        .map(|index| ((index * 2) % 16) as u8 | ((((index * 2 + 1) % 16) as u8) << 4))
        .collect::<Vec<_>>();
    let scales = (0..output_size * input_size / 16)
        .map(|index| [0x30, 0x38, 0x40, 0x44][index % 4])
        .collect::<Vec<_>>();
    let input = benchmark_input(rows, input_size, stream)?;
    let cutile =
        Nvfp4W4A16Linear::from_host(input_size, output_size, &packed, &scales, GLOBAL, stream)?;
    let marlin = MarlinLinear::nvfp4(&packed, &scales, input_size, output_size, GLOBAL, stream)?;
    samples.push(time_cutile_nvfp4(
        &cutile,
        input.clone(),
        rows,
        input_size,
        output_size,
        warmup,
        iterations,
        stream,
    )?);
    samples.push(time_marlin(
        &marlin,
        input,
        rows,
        input_size,
        output_size,
        warmup,
        iterations,
        stream,
    )?);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn benchmark_grouped_nvfp4_shape(
    stream: &Arc<Stream>,
    row_counts: &[usize],
    input_size: usize,
    output_size: usize,
    warmup: usize,
    iterations: usize,
    samples: &mut Vec<KernelBenchmarkSample>,
) -> Result<(), ModelError> {
    const EXPERTS: usize = 256;
    let packed = (0..EXPERTS * output_size * input_size / 2)
        .map(|index| {
            let expert = index / (output_size * input_size / 2);
            let low = ((index * 2 + expert) % 16) as u8;
            let high = ((index * 2 + 1 + expert * 3) % 16) as u8;
            low | (high << 4)
        })
        .collect::<Vec<_>>();
    let scales = (0..EXPERTS * output_size * (input_size / 16))
        .map(|index| [0x30, 0x38, 0x40, 0x44][index % 4])
        .collect::<Vec<_>>();
    let globals = (0..EXPERTS)
        .map(|expert| [0.25, 0.5, 0.75, 1.0][expert % 4])
        .collect::<Vec<_>>();
    let cutile = GroupedNvfp4W4A16::from_host_owned(
        EXPERTS,
        input_size,
        output_size,
        packed.clone(),
        scales.clone(),
        globals.clone(),
        stream,
    )?;
    let marlin = MarlinMoe::nvfp4(&packed, &scales, &globals, input_size, output_size, stream)?;
    let gate_up = if input_size == 2048 && output_size == 512 {
        let up_packed = packed
            .iter()
            .map(|byte| byte.rotate_left(4))
            .collect::<Vec<_>>();
        let mut up_scales = scales.clone();
        up_scales.reverse();
        let up_globals = globals
            .iter()
            .copied()
            .cycle()
            .skip(1)
            .take(EXPERTS)
            .collect::<Vec<_>>();
        Some((
            GroupedNvfp4W4A16::from_host_owned(
                EXPERTS,
                input_size,
                output_size,
                up_packed.clone(),
                up_scales.clone(),
                up_globals.clone(),
                stream,
            )?,
            MarlinMoe::nvfp4(
                &up_packed,
                &up_scales,
                &up_globals,
                input_size,
                output_size,
                stream,
            )?,
        ))
    } else {
        None
    };
    for &rows in row_counts {
        let block_rows = if rows >= 512 && rows.is_multiple_of(64) {
            64
        } else {
            16
        };
        for pattern in ["uniform", "skewed"] {
            let expert_map = grouped_expert_map(rows / block_rows, EXPERTS, pattern);
            if pattern == "skewed" && expert_map.len() < 2 {
                continue;
            }
            let expert_map = upload(expert_map, stream, "upload grouped benchmark expert map")?;
            let input = benchmark_input(rows, input_size, stream)?;
            samples.push(time_cutile_grouped(
                &cutile,
                input.clone(),
                expert_map.clone(),
                rows,
                input_size,
                output_size,
                block_rows,
                pattern,
                warmup,
                iterations,
                stream,
            )?);
            samples.push(time_marlin_grouped(
                &marlin,
                input.clone(),
                expert_map.clone(),
                rows,
                input_size,
                output_size,
                block_rows,
                pattern,
                warmup,
                iterations,
                stream,
            )?);
            if let Some((cutile_up, marlin_up)) = &gate_up {
                samples.push(time_cutile_grouped_gate_up(
                    &cutile,
                    cutile_up,
                    input.clone(),
                    expert_map.clone(),
                    rows,
                    input_size,
                    output_size,
                    block_rows,
                    pattern,
                    warmup,
                    iterations,
                    stream,
                )?);
                samples.push(time_marlin_grouped_gate_up(
                    &marlin,
                    marlin_up,
                    input.clone(),
                    expert_map.clone(),
                    rows,
                    input_size,
                    output_size,
                    block_rows,
                    pattern,
                    warmup,
                    iterations,
                    stream,
                )?);
            }
        }
    }
    Ok(())
}

fn grouped_expert_map(blocks: usize, experts: usize, pattern: &str) -> Vec<i32> {
    match pattern {
        "uniform" => (0..blocks)
            .map(|block| ((block * experts / blocks.max(1)).min(experts - 1)) as i32)
            .collect(),
        "skewed" => {
            let hot_blocks = (blocks * 3 / 4).max(1);
            (0..blocks)
                .map(|block| {
                    if block < hot_blocks {
                        0
                    } else {
                        1 + ((block - hot_blocks) * (experts - 1) / (blocks - hot_blocks).max(1))
                            as i32
                    }
                })
                .collect()
        }
        _ => unreachable!("benchmark routing patterns are closed"),
    }
}

fn benchmark_input(
    rows: usize,
    input_size: usize,
    stream: &Arc<Stream>,
) -> Result<Arc<Tensor<bf16>>, ModelError> {
    let values = (0..rows * input_size)
        .map(|index| bf16::from_f32((index % 17) as f32 / 8.0 - 1.0))
        .collect::<Vec<_>>();
    upload(values, stream, "upload kernel benchmark input")?
        .reshape(&[rows, input_size])
        .map_err(|error| ModelError::Cuda(format!("reshape benchmark input: {error:?}")))
}

#[allow(clippy::too_many_arguments)]
fn time_cutile_fp8(
    linear: &Fp8W8A16Linear,
    input: Arc<Tensor<bf16>>,
    rows: usize,
    input_size: usize,
    output_size: usize,
    warmup: usize,
    iterations: usize,
    stream: &Arc<Stream>,
) -> Result<KernelBenchmarkSample, ModelError> {
    let mut output = api::zeros::<bf16>(&[rows, output_size])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate benchmark output: {error:?}")))?;
    for _ in 0..warmup {
        let mut execution = StreamExecution::new(stream);
        output = linear.enqueue_into(input.clone(), rows, output, &mut execution)?;
        execution.synchronize("warm Marlin comparison FP8")?;
    }
    let mut raw_ms = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = TimingEvent::new(stream)?;
        let end = TimingEvent::new(stream)?;
        start.record(stream)?;
        let mut execution = StreamExecution::new(stream);
        output = linear.enqueue_into(input.clone(), rows, output, &mut execution)?;
        end.record(stream)?;
        raw_ms.push(start.elapsed_ms(&end, stream)?);
        execution.mark_synchronized();
    }
    Ok(summarize(
        "cutile_packed",
        MarlinQuantization::Fp8,
        rows,
        output_size,
        input_size,
        raw_ms,
    ))
}

#[allow(clippy::too_many_arguments)]
fn time_cutile_nvfp4(
    linear: &Nvfp4W4A16Linear,
    input: Arc<Tensor<bf16>>,
    rows: usize,
    input_size: usize,
    output_size: usize,
    warmup: usize,
    iterations: usize,
    stream: &Arc<Stream>,
) -> Result<KernelBenchmarkSample, ModelError> {
    let mut output = api::zeros::<bf16>(&[rows, output_size])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate benchmark output: {error:?}")))?;
    for _ in 0..warmup {
        let mut execution = StreamExecution::new(stream);
        output = linear.enqueue_into(input.clone(), rows, output, &mut execution)?;
        execution.synchronize("warm Marlin comparison NVFP4")?;
    }
    let mut raw_ms = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = TimingEvent::new(stream)?;
        let end = TimingEvent::new(stream)?;
        start.record(stream)?;
        let mut execution = StreamExecution::new(stream);
        output = linear.enqueue_into(input.clone(), rows, output, &mut execution)?;
        end.record(stream)?;
        raw_ms.push(start.elapsed_ms(&end, stream)?);
        execution.mark_synchronized();
    }
    Ok(summarize(
        "cutile_packed",
        MarlinQuantization::Nvfp4,
        rows,
        output_size,
        input_size,
        raw_ms,
    ))
}

#[allow(clippy::too_many_arguments)]
fn time_cutile_grouped(
    linear: &GroupedNvfp4W4A16,
    input: Arc<Tensor<bf16>>,
    expert_map: Arc<Tensor<i32>>,
    rows: usize,
    input_size: usize,
    output_size: usize,
    block_rows: usize,
    routing_pattern: &'static str,
    warmup: usize,
    iterations: usize,
    stream: &Arc<Stream>,
) -> Result<KernelBenchmarkSample, ModelError> {
    let mut output = api::zeros::<bf16>(&[rows, output_size])
        .sync_on(stream)
        .map_err(|error| {
            ModelError::Cuda(format!("allocate grouped benchmark output: {error:?}"))
        })?;
    for _ in 0..warmup {
        let mut execution = StreamExecution::new(stream);
        output = linear.enqueue_device_plan_into(
            input.clone(),
            rows,
            expert_map.clone(),
            output,
            &mut execution,
        )?;
        execution.synchronize("warm grouped cuTile comparison")?;
    }
    let mut raw_ms = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = TimingEvent::new(stream)?;
        let end = TimingEvent::new(stream)?;
        start.record(stream)?;
        let mut execution = StreamExecution::new(stream);
        output = linear.enqueue_device_plan_into(
            input.clone(),
            rows,
            expert_map.clone(),
            output,
            &mut execution,
        )?;
        end.record(stream)?;
        raw_ms.push(start.elapsed_ms(&end, stream)?);
        execution.mark_synchronized();
    }
    Ok(summarize_grouped(
        "cutile_grouped",
        rows,
        output_size,
        input_size,
        routing_pattern,
        linear.num_experts(),
        block_rows,
        raw_ms,
    ))
}

#[allow(clippy::too_many_arguments)]
fn time_marlin_grouped(
    linear: &MarlinMoe,
    input: Arc<Tensor<bf16>>,
    expert_map: Arc<Tensor<i32>>,
    rows: usize,
    input_size: usize,
    output_size: usize,
    block_rows: usize,
    routing_pattern: &'static str,
    warmup: usize,
    iterations: usize,
    stream: &Arc<Stream>,
) -> Result<KernelBenchmarkSample, ModelError> {
    let output = api::zeros::<bf16>(&[rows, output_size])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate grouped Marlin output: {error:?}")))?;
    let temporary = api::zeros::<f32>(&[rows, output_size])
        .sync_on(stream)
        .map_err(|error| {
            ModelError::Cuda(format!("allocate grouped Marlin temporary: {error:?}"))
        })?;
    for _ in 0..warmup {
        linear.execute(
            &input,
            &output,
            &temporary,
            &expert_map,
            rows,
            block_rows,
            stream,
        )?;
    }
    unsafe { stream.synchronize() }
        .map_err(|error| ModelError::Cuda(format!("warm grouped Marlin kernel: {error:?}")))?;
    let mut raw_ms = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = TimingEvent::new(stream)?;
        let end = TimingEvent::new(stream)?;
        start.record(stream)?;
        linear.execute(
            &input,
            &output,
            &temporary,
            &expert_map,
            rows,
            block_rows,
            stream,
        )?;
        end.record(stream)?;
        raw_ms.push(start.elapsed_ms(&end, stream)?);
    }
    Ok(summarize_grouped(
        "marlin_grouped",
        rows,
        output_size,
        input_size,
        routing_pattern,
        linear.num_experts(),
        block_rows,
        raw_ms,
    ))
}

#[allow(clippy::too_many_arguments)]
fn time_cutile_grouped_gate_up(
    gate: &GroupedNvfp4W4A16,
    up: &GroupedNvfp4W4A16,
    input: Arc<Tensor<bf16>>,
    expert_map: Arc<Tensor<i32>>,
    rows: usize,
    input_size: usize,
    output_size: usize,
    block_rows: usize,
    routing_pattern: &'static str,
    warmup: usize,
    iterations: usize,
    stream: &Arc<Stream>,
) -> Result<KernelBenchmarkSample, ModelError> {
    let mut activated = api::zeros::<bf16>(&[rows, output_size])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate fused gate/up output: {error:?}")))?;
    for _ in 0..warmup {
        let mut execution = StreamExecution::new(stream);
        activated = gate.enqueue_silu_mul_device_plan_into(
            up,
            input.clone(),
            rows,
            expert_map.clone(),
            activated,
            &mut execution,
        )?;
        execution.synchronize("warm fused grouped cuTile gate/up")?;
    }
    let mut raw_ms = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = TimingEvent::new(stream)?;
        let end = TimingEvent::new(stream)?;
        start.record(stream)?;
        let mut execution = StreamExecution::new(stream);
        activated = gate.enqueue_silu_mul_device_plan_into(
            up,
            input.clone(),
            rows,
            expert_map.clone(),
            activated,
            &mut execution,
        )?;
        end.record(stream)?;
        raw_ms.push(start.elapsed_ms(&end, stream)?);
        execution.mark_synchronized();
    }
    Ok(summarize_gate_up(
        "cutile_grouped_gate_up_silu",
        rows,
        output_size,
        input_size,
        routing_pattern,
        gate.num_experts(),
        block_rows,
        raw_ms,
    ))
}

#[allow(clippy::too_many_arguments)]
fn time_marlin_grouped_gate_up(
    gate: &MarlinMoe,
    up: &MarlinMoe,
    input: Arc<Tensor<bf16>>,
    expert_map: Arc<Tensor<i32>>,
    rows: usize,
    input_size: usize,
    output_size: usize,
    block_rows: usize,
    routing_pattern: &'static str,
    warmup: usize,
    iterations: usize,
    stream: &Arc<Stream>,
) -> Result<KernelBenchmarkSample, ModelError> {
    const BLOCK: usize = 256;
    let gate_output = Arc::new(
        api::zeros::<bf16>(&[rows, output_size])
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("allocate Marlin gate output: {error:?}")))?,
    );
    let up_output = Arc::new(
        api::zeros::<bf16>(&[rows, output_size])
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("allocate Marlin up output: {error:?}")))?,
    );
    let gate_temporary = api::zeros::<f32>(&[rows, output_size])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate Marlin gate temporary: {error:?}")))?;
    let up_temporary = api::zeros::<f32>(&[rows, output_size])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate Marlin up temporary: {error:?}")))?;
    let mut activated = api::zeros::<bf16>(&[rows, output_size])
        .sync_on(stream)
        .map_err(|error| {
            ModelError::Cuda(format!("allocate Marlin activated output: {error:?}"))
        })?;
    let launch = |activated: &mut Tensor<bf16>,
                  execution: &mut StreamExecution<'_>|
     -> Result<(), ModelError> {
        gate.execute(
            &input,
            &gate_output,
            &gate_temporary,
            &expert_map,
            rows,
            block_rows,
            stream,
        )?;
        up.execute(
            &input,
            &up_output,
            &up_temporary,
            &expert_map,
            rows,
            block_rows,
            stream,
        )?;
        let (_, _, output_partition) = execution.enqueue(
            kernels::silu_mul_bf16(
                gate_output.clone(),
                up_output.clone(),
                activated.partition([1, BLOCK]),
            )
            .generics(vec![BLOCK.to_string()]),
            "execute Marlin comparison SiLU",
        )?;
        drop(output_partition);
        Ok(())
    };
    for _ in 0..warmup {
        let mut execution = StreamExecution::new(stream);
        launch(&mut activated, &mut execution)?;
        execution.synchronize("warm grouped Marlin gate/up")?;
    }
    let mut raw_ms = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = TimingEvent::new(stream)?;
        let end = TimingEvent::new(stream)?;
        start.record(stream)?;
        let mut execution = StreamExecution::new(stream);
        launch(&mut activated, &mut execution)?;
        end.record(stream)?;
        raw_ms.push(start.elapsed_ms(&end, stream)?);
        execution.mark_synchronized();
    }
    Ok(summarize_gate_up(
        "marlin_grouped_gate_up_silu",
        rows,
        output_size,
        input_size,
        routing_pattern,
        gate.num_experts(),
        block_rows,
        raw_ms,
    ))
}

#[allow(clippy::too_many_arguments)]
fn time_marlin(
    linear: &MarlinLinear,
    input: Arc<Tensor<bf16>>,
    rows: usize,
    input_size: usize,
    output_size: usize,
    warmup: usize,
    iterations: usize,
    stream: &Arc<Stream>,
) -> Result<KernelBenchmarkSample, ModelError> {
    let output = api::zeros::<bf16>(&[rows, output_size])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate Marlin output: {error:?}")))?;
    let temporary = api::zeros::<f32>(&[rows, output_size])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate Marlin temporary: {error:?}")))?;
    for _ in 0..warmup {
        linear.execute(&input, &output, &temporary, rows, stream)?;
    }
    unsafe { stream.synchronize() }
        .map_err(|error| ModelError::Cuda(format!("warm Marlin kernel: {error:?}")))?;
    let mut raw_ms = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = TimingEvent::new(stream)?;
        let end = TimingEvent::new(stream)?;
        start.record(stream)?;
        linear.execute(&input, &output, &temporary, rows, stream)?;
        end.record(stream)?;
        raw_ms.push(start.elapsed_ms(&end, stream)?);
    }
    Ok(summarize(
        "marlin",
        linear.quantization,
        rows,
        output_size,
        input_size,
        raw_ms,
    ))
}

fn summarize(
    implementation: &'static str,
    quantization: MarlinQuantization,
    rows: usize,
    output_size: usize,
    input_size: usize,
    raw_ms: Vec<f32>,
) -> KernelBenchmarkSample {
    let mut sorted = raw_ms.clone();
    sorted.sort_by(f32::total_cmp);
    let minimum_ms = sorted[0];
    let median_ms = sorted[sorted.len() / 2];
    let p90_ms = sorted[((sorted.len() - 1) as f32 * 0.9).ceil() as usize];
    let seconds = f64::from(median_ms) / 1000.0;
    let logical_tflops =
        2.0 * rows as f64 * output_size as f64 * input_size as f64 / seconds / 1.0e12;
    let packed_bytes = output_size as f64 * input_size as f64 * quantization.bits() as f64 / 8.0;
    KernelBenchmarkSample {
        implementation,
        quantization,
        rows,
        output_size,
        input_size,
        routing_pattern: None,
        num_experts: None,
        block_rows: None,
        raw_ms,
        minimum_ms,
        median_ms,
        p90_ms,
        logical_tflops,
        packed_weight_gb_per_second: packed_bytes / seconds / 1.0e9,
    }
}

#[allow(clippy::too_many_arguments)]
fn summarize_grouped(
    implementation: &'static str,
    rows: usize,
    output_size: usize,
    input_size: usize,
    routing_pattern: &'static str,
    num_experts: usize,
    block_rows: usize,
    raw_ms: Vec<f32>,
) -> KernelBenchmarkSample {
    let mut sample = summarize(
        implementation,
        MarlinQuantization::Nvfp4,
        rows,
        output_size,
        input_size,
        raw_ms,
    );
    sample.routing_pattern = Some(routing_pattern);
    sample.num_experts = Some(num_experts);
    sample.block_rows = Some(block_rows);
    sample
}

#[allow(clippy::too_many_arguments)]
fn summarize_gate_up(
    implementation: &'static str,
    rows: usize,
    output_size: usize,
    input_size: usize,
    routing_pattern: &'static str,
    num_experts: usize,
    block_rows: usize,
    raw_ms: Vec<f32>,
) -> KernelBenchmarkSample {
    let mut sample = summarize_grouped(
        implementation,
        rows,
        output_size,
        input_size,
        routing_pattern,
        num_experts,
        block_rows,
        raw_ms,
    );
    // Gate and up are two logical GEMMs over independent packed expert banks.
    sample.logical_tflops *= 2.0;
    sample.packed_weight_gb_per_second *= 2.0;
    sample
}

struct TimingEvent(sys::CUevent);

impl TimingEvent {
    fn new(stream: &Arc<Stream>) -> Result<Self, ModelError> {
        stream
            .device()
            .bind_to_thread()
            .map_err(|error| ModelError::Cuda(format!("bind timing device: {error:?}")))?;
        let mut event = std::mem::MaybeUninit::uninit();
        unsafe {
            sys::cuEventCreate(
                event.as_mut_ptr(),
                sys::CUevent_flags_enum_CU_EVENT_DEFAULT as u32,
            )
            .result()
            .map_err(|error| ModelError::Cuda(format!("create timing event: {error:?}")))?;
            Ok(Self(event.assume_init()))
        }
    }

    fn record(&self, stream: &Arc<Stream>) -> Result<(), ModelError> {
        unsafe { sys::cuEventRecord(self.0, stream.cu_stream()) }
            .result()
            .map_err(|error| ModelError::Cuda(format!("record timing event: {error:?}")))
    }

    fn elapsed_ms(&self, end: &Self, stream: &Arc<Stream>) -> Result<f32, ModelError> {
        stream
            .device()
            .bind_to_thread()
            .map_err(|error| ModelError::Cuda(format!("bind timing device: {error:?}")))?;
        unsafe { sys::cuEventSynchronize(end.0) }
            .result()
            .map_err(|error| ModelError::Cuda(format!("synchronize timing event: {error:?}")))?;
        let mut milliseconds = 0.0f32;
        unsafe { sys::cuEventElapsedTime_v2(&mut milliseconds, self.0, end.0) }
            .result()
            .map_err(|error| ModelError::Cuda(format!("read timing event: {error:?}")))?;
        Ok(milliseconds)
    }
}

impl Drop for TimingEvent {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuEventDestroy_v2(self.0);
        }
    }
}

pub fn probe(stream: &Arc<Stream>) -> Result<MarlinProbe, ModelError> {
    Ok(MarlinProbe {
        fp8_max_abs_error: probe_fp8(stream)?,
        nvfp4_max_abs_error: probe_nvfp4(stream)?,
        grouped_nvfp4_max_abs_error: probe_grouped_nvfp4(stream)?,
    })
}

fn probe_fp8(stream: &Arc<Stream>) -> Result<f32, ModelError> {
    const M: usize = 16;
    const N: usize = 128;
    const K: usize = 128;
    const SCALE: f32 = 0.25;
    let encoded = (0..N * K)
        .map(|index| [0x01, 0x20, 0x38, 0x3c, 0x40, 0x60, 0x80, 0xb8][index % 8])
        .collect::<Vec<_>>();
    let linear = MarlinLinear::fp8(&encoded, K, N, SCALE, stream)?;
    let input_host = (0..M * K)
        .map(|index| bf16::from_f32((index % 7) as f32 - 3.0))
        .collect::<Vec<_>>();
    let input = upload(input_host.clone(), stream, "upload Marlin FP8 probe input")?;
    let input = input
        .reshape(&[M, K])
        .map_err(|error| ModelError::Cuda(format!("reshape Marlin input: {error:?}")))?;
    let output = api::zeros::<bf16>(&[M, N])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate Marlin output: {error:?}")))?;
    let temporary = api::zeros::<f32>(&[M, N])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate Marlin temporary: {error:?}")))?;
    linear.execute(&input, &output, &temporary, M, stream)?;
    verify(output, stream, M, N, K, &input_host, |column, k| {
        decode_e4m3fn(encoded[column * K + k]) * SCALE
    })
}

fn probe_nvfp4(stream: &Arc<Stream>) -> Result<f32, ModelError> {
    const M: usize = 16;
    const N: usize = 128;
    const K: usize = 128;
    const GLOBAL: f32 = 0.5;
    let packed = (0..N * K / 2)
        .map(|index| ((index * 2) % 16) as u8 | ((((index * 2 + 1) % 16) as u8) << 4))
        .collect::<Vec<_>>();
    let scales = (0..N * (K / 16))
        .map(|index| if index % 2 == 0 { 0x38 } else { 0x40 })
        .collect::<Vec<_>>();
    let linear = MarlinLinear::nvfp4(&packed, &scales, K, N, GLOBAL, stream)?;
    let input_host = (0..M * K)
        .map(|index| bf16::from_f32((index % 7) as f32 - 3.0))
        .collect::<Vec<_>>();
    let input = upload(input_host.clone(), stream, "upload Marlin FP4 probe input")?;
    let input = input
        .reshape(&[M, K])
        .map_err(|error| ModelError::Cuda(format!("reshape Marlin input: {error:?}")))?;
    let output = api::zeros::<bf16>(&[M, N])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate Marlin output: {error:?}")))?;
    let temporary = api::zeros::<f32>(&[M, N])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate Marlin temporary: {error:?}")))?;
    linear.execute(&input, &output, &temporary, M, stream)?;
    verify(output, stream, M, N, K, &input_host, |column, k| {
        let byte = packed[column * (K / 2) + k / 2];
        let nibble = if k % 2 == 0 { byte & 0x0f } else { byte >> 4 };
        crate::quantization::decode_e2m1(nibble)
            * decode_e4m3fn(scales[column * (K / 16) + k / 16])
            * GLOBAL
    })
}

fn probe_grouped_nvfp4(stream: &Arc<Stream>) -> Result<f32, ModelError> {
    const EXPERTS: usize = 2;
    const BLOCK_ROWS: usize = 16;
    const M: usize = EXPERTS * BLOCK_ROWS;
    const N: usize = 128;
    const K: usize = 128;
    const GLOBALS: [f32; EXPERTS] = [0.25, 0.75];
    let packed = (0..EXPERTS * N * K / 2)
        .map(|index| {
            let expert = index / (N * K / 2);
            let low = ((index * 2 + expert * 3) % 16) as u8;
            let high = ((index * 2 + 1 + expert * 5) % 16) as u8;
            low | (high << 4)
        })
        .collect::<Vec<_>>();
    let scales = (0..EXPERTS * N * (K / 16))
        .map(|index| {
            let expert = index / (N * (K / 16));
            if (index + expert) % 2 == 0 {
                0x38
            } else {
                0x40
            }
        })
        .collect::<Vec<_>>();
    let linear = MarlinMoe::nvfp4(&packed, &scales, &GLOBALS, K, N, stream)?;
    if linear.num_experts() != EXPERTS {
        return Err(ModelError::Cuda(
            "grouped Marlin expert count changed".into(),
        ));
    }
    let input_host = (0..M * K)
        .map(|index| bf16::from_f32((index % 7) as f32 - 3.0))
        .collect::<Vec<_>>();
    let input = upload(
        input_host.clone(),
        stream,
        "upload grouped Marlin probe input",
    )?
    .reshape(&[M, K])
    .map_err(|error| ModelError::Cuda(format!("reshape grouped Marlin input: {error:?}")))?;
    let output = api::zeros::<bf16>(&[M, N])
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("allocate grouped Marlin output: {error:?}")))?;
    let temporary = api::zeros::<f32>(&[M, N])
        .sync_on(stream)
        .map_err(|error| {
            ModelError::Cuda(format!("allocate grouped Marlin temporary: {error:?}"))
        })?;
    let expert_ids = upload(
        (0..EXPERTS as i32).collect(),
        stream,
        "upload grouped Marlin expert IDs",
    )?;
    linear.execute(
        &input,
        &output,
        &temporary,
        &expert_ids,
        M,
        BLOCK_ROWS,
        stream,
    )?;
    let actual: Vec<bf16> = output
        .to_host_vec()
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("download grouped Marlin probe: {error:?}")))?;
    let mut max_error = 0.0f32;
    for row in 0..M {
        let expert = row / BLOCK_ROWS;
        for column in 0..N {
            let mut expected = 0.0f32;
            for k in 0..K {
                let weight_base = expert * N * K / 2 + column * K / 2;
                let byte = packed[weight_base + k / 2];
                let nibble = if k % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                let scale_base = expert * N * (K / 16) + column * (K / 16);
                let weight = crate::quantization::decode_e2m1(nibble)
                    * decode_e4m3fn(scales[scale_base + k / 16])
                    * GLOBALS[expert];
                expected += input_host[row * K + k].to_f32() * bf16::from_f32(weight).to_f32();
            }
            let expected = bf16::from_f32(expected).to_f32();
            let value = actual[row * N + column].to_f32();
            let error = (value - expected).abs();
            max_error = max_error.max(error);
            if !value.is_finite() || error > 0.5 {
                return Err(ModelError::Cuda(format!(
                    "grouped Marlin mismatch at expert {expert}, ({row}, {column}): {value} != {expected}"
                )));
            }
        }
    }
    Ok(max_error)
}

#[allow(clippy::too_many_arguments)]
fn verify(
    output: Tensor<bf16>,
    stream: &Arc<Stream>,
    rows: usize,
    columns: usize,
    reduction: usize,
    input: &[bf16],
    weight: impl Fn(usize, usize) -> f32,
) -> Result<f32, ModelError> {
    let actual: Vec<bf16> = output
        .to_host_vec()
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("download Marlin probe: {error:?}")))?;
    let mut max_error = 0.0f32;
    for row in 0..rows {
        for column in 0..columns {
            let mut expected = 0.0f32;
            for k in 0..reduction {
                expected += input[row * reduction + k].to_f32()
                    * bf16::from_f32(weight(column, k)).to_f32();
            }
            let expected = bf16::from_f32(expected).to_f32();
            let value = actual[row * columns + column].to_f32();
            let error = (value - expected).abs();
            max_error = max_error.max(error);
            if !value.is_finite() || error > 0.5 {
                return Err(ModelError::Cuda(format!(
                    "Marlin mismatch at ({row}, {column}): {value} != {expected}"
                )));
            }
        }
    }
    Ok(max_error)
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "requires an SM80+ CUDA device"]
    fn native_grouped_marlin_matches_cpu_oracle() {
        let device = cuda_core::Device::new(0).expect("initialize CUDA device");
        let stream = device.new_stream().expect("create CUDA stream");
        let max_error = super::probe_grouped_nvfp4(&stream).expect("run grouped Marlin probe");
        assert!(max_error <= 0.5, "grouped Marlin max error {max_error}");
    }
}
