use std::sync::Arc;

use thiserror::Error;

use crate::model::Model;

use super::{GenerateRequest, RequestId};

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("model execution failed: {0}")]
    Execution(String),
    #[error("backend is unavailable: {0}")]
    Unavailable(String),
}

#[derive(Debug, Clone, Copy)]
pub struct PreparedRequest {
    pub prompt_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct ScheduledWork {
    pub request_id: RequestId,
    pub position: usize,
    pub num_tokens: usize,
    pub kv_slots: Vec<u32>,
    pub sample: bool,
}

#[derive(Debug, Clone)]
pub struct StepOutput {
    pub request_id: RequestId,
    pub token_id: Option<u32>,
    pub text: String,
    pub is_eos: bool,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BackendExecutionStats {
    pub eager_forwards: u64,
    pub graph_replays: u64,
    pub graph_captures: u64,
}

/// Thread-confined model backend. Production backends are constructed and used
/// on the dedicated engine thread, so CUDA handles do not need to be movable.
pub trait Backend: 'static {
    fn model(&self) -> Arc<dyn Model>;

    fn add_request(&mut self, request: &GenerateRequest) -> Result<PreparedRequest, BackendError>;

    fn step(&mut self, batch: &[ScheduledWork]) -> Result<Vec<StepOutput>, BackendError>;

    fn remove_request(&mut self, request_id: RequestId);

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

    fn add_request(&mut self, request: &GenerateRequest) -> Result<PreparedRequest, BackendError> {
        (**self).add_request(request)
    }

    fn step(&mut self, batch: &[ScheduledWork]) -> Result<Vec<StepOutput>, BackendError> {
        (**self).step(batch)
    }

    fn remove_request(&mut self, request_id: RequestId) {
        (**self).remove_request(request_id);
    }

    fn take_execution_stats(&mut self) -> BackendExecutionStats {
        (**self).take_execution_stats()
    }

    fn shutdown(&mut self) -> Result<(), BackendError> {
        (**self).shutdown()
    }
}
