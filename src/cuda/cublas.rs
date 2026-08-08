use std::{cell::RefCell, ffi::c_void, sync::Arc};

use cuda_async::device_operation::{DeviceOp, value, with_context};
use cuda_core::IntoResult;
use cudarc::cublas::{result as cublas_result, sys as cublas_sys};
use cutile::{core::bf16, tensor::Tensor};
use thiserror::Error;

type HandleKey = (usize, usize);
type Handle = usize;

thread_local! {
    static HANDLE: RefCell<Option<(HandleKey, Handle)>> = const { RefCell::new(None) };
}

#[derive(Debug, Error)]
pub enum CublasError {
    #[error("invalid BF16 GEMM dimensions m={m}, n={n}, k={k}")]
    InvalidDimensions { m: usize, n: usize, k: usize },
    #[error("BF16 cuBLAS operation failed: {0}")]
    Operation(String),
}

unsafe fn handle(
    device_id: usize,
    stream: cublas_sys::cudaStream_t,
) -> Result<cublas_sys::cublasHandle_t, CublasError> {
    let key = (device_id, stream as usize);
    HANDLE.with(|cached| {
        if let Some((cached_key, handle)) = *cached.borrow()
            && cached_key == key
        {
            return Ok(handle as cublas_sys::cublasHandle_t);
        }
        let handle = cublas_result::create_handle()
            .map_err(|error| CublasError::Operation(format!("create handle: {error:?}")))?;
        cublas_result::set_stream(handle, stream)
            .map_err(|error| CublasError::Operation(format!("set stream: {error:?}")))?;
        *cached.borrow_mut() = Some((key, handle as usize));
        Ok(handle)
    })
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch(
    device_id: usize,
    stream: cublas_sys::cudaStream_t,
    matrix: &Tensor<bf16>,
    rhs: &Tensor<bf16>,
    out: &Tensor<bf16>,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), CublasError> {
    let handle = unsafe { handle(device_id, stream)? };
    let alpha = 1.0f32;
    let beta = 0.0f32;
    unsafe {
        cublas_sys::cublasSetPointerMode_v2(
            handle,
            cublas_sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
        )
        .result()
        .map_err(|error| CublasError::Operation(format!("set pointer mode: {error:?}")))?;
        cublas_result::gemm_ex(
            handle,
            cublas_sys::cublasOperation_t::CUBLAS_OP_T,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            k,
            (&alpha as *const f32).cast::<c_void>(),
            matrix.device_pointer().cu_deviceptr() as usize as *const c_void,
            cublas_sys::cudaDataType_t::CUDA_R_16BF,
            k,
            rhs.device_pointer().cu_deviceptr() as usize as *const c_void,
            cublas_sys::cudaDataType_t::CUDA_R_16BF,
            k,
            (&beta as *const f32).cast::<c_void>(),
            out.device_pointer().cu_deviceptr() as usize as *mut c_void,
            cublas_sys::cudaDataType_t::CUDA_R_16BF,
            m,
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
        .map_err(|error| CublasError::Operation(format!("gemm_ex: {error:?}")))
    }
}

pub fn gemm_bf16(
    matrix: Arc<Tensor<bf16>>,
    rhs: Arc<Tensor<bf16>>,
    out: Tensor<bf16>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<impl DeviceOp<Output = Result<Tensor<bf16>, CublasError>>, CublasError> {
    if m == 0
        || n == 0
        || k == 0
        || m > i32::MAX as usize
        || n > i32::MAX as usize
        || k > i32::MAX as usize
    {
        return Err(CublasError::InvalidDimensions { m, n, k });
    }

    Ok(with_context(move |context| {
        let result = (|| {
            context
                .device()
                .bind_to_thread()
                .map_err(|error| CublasError::Operation(format!("bind context: {error:?}")))?;
            let stream = context.get_cuda_stream().cu_stream() as cublas_sys::cudaStream_t;
            unsafe {
                launch(
                    context.get_device_id(),
                    stream,
                    &matrix,
                    &rhs,
                    &out,
                    m as i32,
                    n as i32,
                    k as i32,
                )?;
            }
            Ok(out)
        })();
        value(result)
    }))
}
