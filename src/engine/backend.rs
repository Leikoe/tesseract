use thiserror::Error;

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

pub trait Backend: Send + 'static {
    fn model_id(&self) -> &str;

    fn add_request(&mut self, request: &GenerateRequest) -> Result<PreparedRequest, BackendError>;

    fn step(&mut self, batch: &[ScheduledWork]) -> Result<Vec<StepOutput>, BackendError>;

    fn remove_request(&mut self, request_id: RequestId);

    fn shutdown(&mut self) -> Result<(), BackendError> {
        Ok(())
    }
}
