//! Reusable, stream-safe tensor storage for synchronous CUDA model programs.
//!
//! A workspace has three states for each allocation:
//! available, checked out by the current forward, or retired while kernels may
//! still reference it. Retired storage only becomes available again after the
//! execution scope proves that its stream is synchronized. This prevents a
//! fast producer from overwriting an input that a queued consumer has not read.

use std::{collections::VecDeque, sync::Arc};

use cuda_core::DType;
use cutile::{api, tensor::Tensor};

use crate::{cuda::execution::StreamExecution, model::ModelError};

struct TensorPool<T: DType> {
    available: VecDeque<Tensor<T>>,
    retired: Vec<Tensor<T>>,
    cached_bytes: usize,
    byte_limit: usize,
}

impl<T: DType> TensorPool<T> {
    const fn new(byte_limit: usize) -> Self {
        Self {
            available: VecDeque::new(),
            retired: Vec::new(),
            cached_bytes: 0,
            byte_limit,
        }
    }

    fn take(
        &mut self,
        shape: &[usize],
        execution: &mut StreamExecution<'_>,
        allocation_name: &'static str,
    ) -> Result<Tensor<T>, ModelError> {
        let requested_shape = shape
            .iter()
            .map(|&dimension| {
                i32::try_from(dimension)
                    .map_err(|_| ModelError::Cuda("workspace shape exceeds i32".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(index) = self
            .available
            .iter()
            .position(|tensor| tensor.shape() == requested_shape)
        {
            let tensor = self
                .available
                .remove(index)
                .expect("workspace index came from the same queue");
            self.cached_bytes -= tensor.num_bytes();
            return Ok(tensor);
        }
        execution.enqueue(api::zeros::<T>(shape), allocation_name)
    }

    fn take_zeroed(
        &mut self,
        shape: &[usize],
        execution: &mut StreamExecution<'_>,
        allocation_name: &'static str,
        clear_name: &'static str,
    ) -> Result<Tensor<T>, ModelError> {
        let had_matching_tensor = self.available.iter().any(|tensor| {
            tensor.shape().len() == shape.len()
                && tensor
                    .shape()
                    .iter()
                    .zip(shape)
                    .all(|(&actual, &expected)| usize::try_from(actual) == Ok(expected))
        });
        let tensor = self.take(shape, execution, allocation_name)?;
        if had_matching_tensor {
            execution.enqueue(api::fill(tensor, T::zero()), clear_name)
        } else {
            Ok(tensor)
        }
    }

    fn retire(&mut self, tensor: Tensor<T>) {
        self.retired.push(tensor);
    }

    fn retire_shared(
        &mut self,
        tensor: Arc<Tensor<T>>,
        name: &'static str,
    ) -> Result<(), ModelError> {
        let tensor = Arc::try_unwrap(tensor).map_err(|_| {
            ModelError::Cuda(format!(
                "cannot retire {name}: workspace tensor still has host aliases"
            ))
        })?;
        self.retire(tensor);
        Ok(())
    }

    fn reclaim(&mut self) {
        for tensor in self.retired.drain(..) {
            let bytes = tensor.num_bytes();
            if bytes > self.byte_limit {
                continue;
            }
            while self.cached_bytes.saturating_add(bytes) > self.byte_limit {
                let Some(evicted) = self.available.pop_front() else {
                    break;
                };
                self.cached_bytes -= evicted.num_bytes();
            }
            self.cached_bytes += bytes;
            self.available.push_back(tensor);
        }
    }
}

/// Shape-keyed scratch storage owned by one model executor.
///
/// The executor is single-threaded, so the workspace requires no locking. The
/// byte limit is shared proportionally across supported activation dtypes and
/// bounds retained storage when serving a changing mix of batch shapes.
pub(crate) struct ExecutionWorkspace {
    bf16: TensorPool<cuda_core::bf16>,
    f32: TensorPool<f32>,
    i32: TensorPool<i32>,
    u32: TensorPool<u32>,
}

impl ExecutionWorkspace {
    pub(crate) fn new(byte_limit: usize) -> Self {
        let metadata_limit = byte_limit / 16;
        Self {
            bf16: TensorPool::new(byte_limit.saturating_sub(metadata_limit * 3)),
            f32: TensorPool::new(metadata_limit),
            i32: TensorPool::new(metadata_limit),
            u32: TensorPool::new(metadata_limit),
        }
    }

    pub(crate) fn take_bf16(
        &mut self,
        shape: &[usize],
        execution: &mut StreamExecution<'_>,
        name: &'static str,
    ) -> Result<Tensor<cuda_core::bf16>, ModelError> {
        self.bf16.take(shape, execution, name)
    }

    pub(crate) fn take_zeroed_bf16(
        &mut self,
        shape: &[usize],
        execution: &mut StreamExecution<'_>,
        allocation_name: &'static str,
        clear_name: &'static str,
    ) -> Result<Tensor<cuda_core::bf16>, ModelError> {
        self.bf16
            .take_zeroed(shape, execution, allocation_name, clear_name)
    }

    pub(crate) fn take_f32(
        &mut self,
        shape: &[usize],
        execution: &mut StreamExecution<'_>,
        name: &'static str,
    ) -> Result<Tensor<f32>, ModelError> {
        self.f32.take(shape, execution, name)
    }

    pub(crate) fn take_zeroed_i32(
        &mut self,
        shape: &[usize],
        execution: &mut StreamExecution<'_>,
        allocation_name: &'static str,
        clear_name: &'static str,
    ) -> Result<Tensor<i32>, ModelError> {
        self.i32
            .take_zeroed(shape, execution, allocation_name, clear_name)
    }

    pub(crate) fn take_u32(
        &mut self,
        shape: &[usize],
        execution: &mut StreamExecution<'_>,
        name: &'static str,
    ) -> Result<Tensor<u32>, ModelError> {
        self.u32.take(shape, execution, name)
    }

    pub(crate) fn retire_bf16(&mut self, tensor: Tensor<cuda_core::bf16>) {
        self.bf16.retire(tensor);
    }

    pub(crate) fn retire_shared_bf16(
        &mut self,
        tensor: Arc<Tensor<cuda_core::bf16>>,
        name: &'static str,
    ) -> Result<(), ModelError> {
        self.bf16.retire_shared(tensor, name)
    }

    pub(crate) fn retire_f32(&mut self, tensor: Tensor<f32>) {
        self.f32.retire(tensor);
    }

    pub(crate) fn retire_i32(&mut self, tensor: Tensor<i32>) {
        self.i32.retire(tensor);
    }

    pub(crate) fn retire_u32(&mut self, tensor: Tensor<u32>) {
        self.u32.retire(tensor);
    }

    pub(crate) fn reclaim(&mut self, execution: &StreamExecution<'_>) -> Result<(), ModelError> {
        if !execution.is_synchronized() {
            return Err(ModelError::Cuda(
                "cannot reclaim CUDA workspace before stream completion".into(),
            ));
        }
        self.bf16.reclaim();
        self.f32.reclaim();
        self.i32.reclaim();
        self.u32.reclaim();
        Ok(())
    }
}
