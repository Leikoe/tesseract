use std::{collections::HashMap, sync::Arc, time::Duration};

use crate::model::{ChatMessage, IncrementalDecoder, Model, ModelError, ModelSummary};

use super::{
    Backend, BackendError, GenerateRequest, PreparedRequest, RequestId, ScheduledBatch, StepOutput,
};

/// Deterministic token backend used only by API and scheduler tests.
pub struct DeterministicBackend {
    model: Arc<TestModel>,
    requests: HashMap<RequestId, TestRequest>,
    step_delay: Duration,
}

struct TestRequest {
    generated: usize,
}

impl DeterministicBackend {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model: Arc::new(TestModel {
                id: model_id.into(),
            }),
            requests: HashMap::new(),
            step_delay: Duration::ZERO,
        }
    }

    pub fn with_step_delay(mut self, step_delay: Duration) -> Self {
        self.step_delay = step_delay;
        self
    }
}

impl Backend for DeterministicBackend {
    fn model(&self) -> Arc<dyn Model> {
        self.model.clone()
    }

    fn add_request(&mut self, request: &GenerateRequest) -> Result<PreparedRequest, BackendError> {
        let prompt_tokens = request.prompt.split_whitespace().count().max(1);
        self.requests
            .insert(request.id, TestRequest { generated: 0 });
        Ok(PreparedRequest { prompt_tokens })
    }

    fn step(&mut self, batch: &ScheduledBatch) -> Result<Vec<StepOutput>, BackendError> {
        if !self.step_delay.is_zero() {
            std::thread::sleep(self.step_delay);
        }
        let mut outputs = Vec::new();
        for work in batch.work() {
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

struct TestModel {
    id: String,
}

impl Model for TestModel {
    fn id(&self) -> &str {
        &self.id
    }

    fn render_chat(&self, messages: &[ChatMessage<'_>]) -> Result<String, ModelError> {
        if messages.is_empty() {
            return Err(ModelError::InvalidInput(
                "messages must not be empty".into(),
            ));
        }
        Ok(messages
            .iter()
            .map(|message| message.content)
            .collect::<Vec<_>>()
            .join("\n"))
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, ModelError> {
        Ok((0..text.split_whitespace().count().max(1) as u32).collect())
    }

    fn decoder(&self) -> Box<dyn IncrementalDecoder> {
        Box::new(TestDecoder)
    }

    fn eos_token_ids(&self) -> &[u32] {
        &[]
    }

    fn summary(&self) -> ModelSummary {
        ModelSummary {
            id: self.id.clone(),
            architecture: "deterministic-test-model".into(),
            dtype: "none".into(),
            layers: 0,
            hidden_size: 0,
            attention_heads: 0,
            kv_heads: 0,
            vocab_size: 0,
            tensors: 0,
        }
    }
}

struct TestDecoder;

impl IncrementalDecoder for TestDecoder {
    fn push(&mut self, token_id: u32) -> Result<String, ModelError> {
        Ok(format!(" token{token_id}"))
    }
}
