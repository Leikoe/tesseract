//! Framework-independent Marlin weight preparation, execution, and timing.

use std::{ffi::c_void, sync::Arc};

use cuda_async::device_operation::DeviceOp;
use cuda_core::{IntoResult, Stream, sys};
use cutile::{
    DType, api,
    core::{bf16, f16},
    tensor::{Reshape, Tensor, ToHostVec},
};
use serde::Serialize;

use crate::{model::ModelError, quantization::decode_e4m3fn};

unsafe extern "C" {
    fn tesseract_marlin_repack(
        input: *const c_void,
        output: *mut c_void,
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
    let workspace = api::zeros::<i32>(&[sms as usize])
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
}

pub fn probe(stream: &Arc<Stream>) -> Result<MarlinProbe, ModelError> {
    Ok(MarlinProbe {
        fp8_max_abs_error: probe_fp8(stream)?,
        nvfp4_max_abs_error: probe_nvfp4(stream)?,
    })
}

fn probe_fp8(stream: &Arc<Stream>) -> Result<f32, ModelError> {
    const M: usize = 16;
    const N: usize = 64;
    const K: usize = 64;
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
    const N: usize = 64;
    const K: usize = 64;
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
