//! Stream-ordered execution scope for synchronous model programs.
//!
//! Leaf operations may enqueue without blocking, but scheduler-visible
//! completion is only truthful after this scope has synchronized. Dropping an
//! unfinished scope drains the stream so error paths cannot recycle KV or
//! recurrent-state slots while kernels still reference them.

use std::{sync::Arc, time::Instant};

use cuda_async::device_operation::DeviceOp;
use cuda_core::Stream;

use crate::model::ModelError;

pub(crate) struct StreamExecution<'a> {
    stream: &'a Arc<Stream>,
    synchronized: bool,
}

impl<'a> StreamExecution<'a> {
    pub(crate) fn new(stream: &'a Arc<Stream>) -> Self {
        Self {
            stream,
            synchronized: true,
        }
    }

    pub(crate) const fn stream(&self) -> &'a Arc<Stream> {
        self.stream
    }

    pub(crate) const fn is_synchronized(&self) -> bool {
        self.synchronized
    }

    pub(crate) fn mark_pending(&mut self) {
        self.synchronized = false;
    }

    pub(crate) fn enqueue<T>(
        &mut self,
        operation: impl DeviceOp<Output = T>,
        name: &'static str,
    ) -> Result<T, ModelError> {
        self.synchronized = false;
        let started = Instant::now();
        // SAFETY: the scope owns submission order for this model stream and
        // drains it before scheduler-visible completion or on every drop path.
        let output = unsafe { operation.async_on(self.stream) }
            .map_err(|error| ModelError::Cuda(format!("{name}: {error:?}")));
        let elapsed = started.elapsed();
        if elapsed.as_millis() >= 5 {
            tracing::debug!(
                name,
                elapsed_ms = elapsed.as_secs_f64() * 1_000.0,
                "slow CUDA enqueue"
            );
        }
        output
    }

    pub(crate) fn synchronize(&mut self, name: &'static str) -> Result<(), ModelError> {
        if self.synchronized {
            return Ok(());
        }
        let started = Instant::now();
        // SAFETY: this executor is the sole owner of the stream.
        let result = unsafe { self.stream.synchronize() }
            .map_err(|error| ModelError::Cuda(format!("{name}: {error:?}")));
        let elapsed = started.elapsed();
        if elapsed.as_millis() >= 5 {
            tracing::debug!(
                name,
                elapsed_ms = elapsed.as_secs_f64() * 1_000.0,
                "slow CUDA synchronization"
            );
        }
        if result.is_ok() {
            self.synchronized = true;
        }
        result
    }

    /// Records a synchronization performed by a host transfer on this stream.
    pub(crate) fn mark_synchronized(&mut self) {
        self.synchronized = true;
    }
}

impl Drop for StreamExecution<'_> {
    fn drop(&mut self) {
        if !self.synchronized {
            // Preserve the original error on unwind/early return. A later CUDA
            // operation will surface a drain failure and poison the executor.
            let _ = unsafe { self.stream.synchronize() };
        }
    }
}
