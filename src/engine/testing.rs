use std::collections::HashMap;

use super::{
    Backend, BackendError, GenerateRequest, PreparedRequest, RequestId, ScheduledWork, StepOutput,
};

/// Deterministic token backend used only by API and scheduler tests.
pub struct DeterministicBackend {
    model_id: String,
    requests: HashMap<RequestId, TestRequest>,
}

struct TestRequest {
    generated: usize,
}

impl DeterministicBackend {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            requests: HashMap::new(),
        }
    }
}

impl Backend for DeterministicBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn add_request(&mut self, request: &GenerateRequest) -> Result<PreparedRequest, BackendError> {
        let prompt_tokens = request.prompt.split_whitespace().count().max(1);
        self.requests
            .insert(request.id, TestRequest { generated: 0 });
        Ok(PreparedRequest { prompt_tokens })
    }

    fn step(&mut self, batch: &[ScheduledWork]) -> Result<Vec<StepOutput>, BackendError> {
        let mut outputs = Vec::new();
        for work in batch {
            if !work.sample {
                continue;
            }
            let request = self.requests.get_mut(&work.request_id).ok_or_else(|| {
                BackendError::Execution(format!("unknown request {}", work.request_id))
            })?;
            let index = request.generated;
            request.generated += 1;
            outputs.push(StepOutput {
                request_id: work.request_id,
                token_id: Some(1000 + index as u32),
                text: format!(" token{index}"),
                is_eos: false,
            });
        }
        Ok(outputs)
    }

    fn remove_request(&mut self, request_id: RequestId) {
        self.requests.remove(&request_id);
    }
}
