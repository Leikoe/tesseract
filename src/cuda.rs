//! Model-neutral CUDA infrastructure and kernel validation.
//!
//! Architecture-specific dimensions, tensor names, prompt behavior, and
//! forward-pass composition belong to the corresponding model module.

use cuda_async::device_operation::DeviceOp;
use cuda_async::{cuda_graph::CudaGraph, error::DeviceError};
use cuda_core::Device;
use cutile::{
    api,
    core::bf16,
    tensor::{PartitionMut, Reshape, ToHostVec},
    tile_kernel::ToHostVecOp,
};
use thiserror::Error;

pub(crate) mod cublas;
pub(crate) mod kernels;

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

use smoke_kernels::add_bf16;

const SMOKE_ELEMENTS: usize = 4096;
const SMOKE_BLOCK: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16SmokeReport {
    pub device_id: usize,
    pub elements: usize,
    pub gemm_rows: usize,
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
    #[error(transparent)]
    Cublas(#[from] cublas::CublasError),
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
pub fn validate_bf16_cutile(device_id: usize) -> Result<Bf16SmokeReport, CudaError> {
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

    let host: Vec<f32> = api::convert::<bf16, f32>(out.into())
        .sync_on(&stream)
        .map_err(|error| kernel_error("convert output to FP32", error))?
        .dup()
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
    let gemm_host: Vec<bf16> = gemm_out
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

    let graph_out = api::zeros::<bf16>(&[1, 2])
        .sync_on(&stream)
        .map_err(|error| kernel_error("allocate graph GEMM output", error))?;
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
    let activated_host: Vec<bf16> = activated
        .unpartition()
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
    let attention_host: Vec<bf16> = attention
        .unpartition()
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
