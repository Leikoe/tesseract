use std::sync::Arc;

use thiserror::Error;

use crate::model::Model;

use super::{ForwardBatch, RequestId, TokenId};

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("model execution failed: {0}")]
    Execution(String),
    #[error("backend is unavailable: {0}")]
    Unavailable(String),
}

#[derive(Debug, Clone)]
pub struct StepOutput {
    pub request_id: RequestId,
    pub token_id: TokenId,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BackendExecutionStats {
    pub eager_forwards: u64,
    pub graph_replays: u64,
    pub graph_captures: u64,
    pub packed_decode_forwards: u64,
    pub packed_decode_requests: u64,
}

/// Thread-confined model backend. Production backends are constructed and used
/// on the dedicated engine thread, so CUDA handles do not need to be movable.
pub trait Backend: 'static {
    fn model(&self) -> Arc<dyn Model>;

    fn step(&mut self, batch: &ForwardBatch) -> Result<Vec<StepOutput>, BackendError>;

    fn take_execution_stats(&mut self) -> BackendExecutionStats {
        BackendExecutionStats::default()
    }

    fn shutdown(&mut self) -> Result<(), BackendError> {
        Ok(())
    }
}

impl<T: Backend + ?Sized> Backend for Box<T> {
    fn model(&self) -> Arc<dyn Model> {
        (**self).model()
    }

    fn step(&mut self, batch: &ForwardBatch) -> Result<Vec<StepOutput>, BackendError> {
        (**self).step(batch)
    }

    fn take_execution_stats(&mut self) -> BackendExecutionStats {
        (**self).take_execution_stats()
    }

    fn shutdown(&mut self) -> Result<(), BackendError> {
        (**self).shutdown()
    }
}
