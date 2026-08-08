//! Model-neutral CUDA infrastructure and kernel validation.
//!
//! Architecture-specific dimensions, tensor names, prompt behavior, and
//! forward-pass composition belong to the corresponding model module.

use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::{
    api,
    core::bf16,
    tensor::{PartitionMut, ToHostVec},
    tile_kernel::ToHostVecOp,
};
use thiserror::Error;

#[cutile::module]
mod kernels {
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

use kernels::add_bf16;

const SMOKE_ELEMENTS: usize = 4096;
const SMOKE_BLOCK: i32 = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16SmokeReport {
    pub device_id: usize,
    pub elements: usize,
}

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("failed to initialize CUDA device {device_id}: {message}")]
    Device { device_id: usize, message: String },
    #[error("BF16 cuTile validation failed during {operation}: {message}")]
    Bf16Kernel {
        operation: &'static str,
        message: String,
    },
    #[error("BF16 cuTile validation produced {actual} at element {index}; expected 2")]
    WrongValue { index: usize, actual: f32 },
}

/// Compile and execute a Tesseract-owned BF16 cuTile kernel on the requested
/// CUDA device, then copy FP32-converted results back for validation.
pub fn validate_bf16_cutile(device_id: usize) -> Result<Bf16SmokeReport, CudaError> {
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

    Ok(Bf16SmokeReport {
        device_id,
        elements: SMOKE_ELEMENTS,
    })
}

fn kernel_error(operation: &'static str, error: impl std::fmt::Debug) -> CudaError {
    CudaError::Bf16Kernel {
        operation,
        message: format!("{error:?}"),
    }
}
