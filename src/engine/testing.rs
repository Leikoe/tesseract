use std::{sync::Arc, time::Duration};

use crate::model::{ChatMessage, IncrementalDecoder, Model, ModelError, ModelSummary};

use super::{Backend, BackendError, ForwardBatch, StepOutput, TokenId};

/// Deterministic token backend used only by API and scheduler tests.
pub struct DeterministicBackend {
    model: Arc<TestModel>,
    step_delay: Duration,
    fail_next_step: bool,
}

impl DeterministicBackend {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model: Arc::new(TestModel {
                id: model_id.into(),
            }),
            step_delay: Duration::ZERO,
            fail_next_step: false,
        }
    }

    pub fn with_step_delay(mut self, step_delay: Duration) -> Self {
        self.step_delay = step_delay;
        self
    }

    pub fn failing_next_step(mut self) -> Self {
        self.fail_next_step = true;
        self
    }
}

impl Backend for DeterministicBackend {
    fn model(&self) -> Arc<dyn Model> {
        self.model.clone()
    }

    fn step(&mut self, batch: &ForwardBatch) -> Result<Vec<StepOutput>, BackendError> {
        if std::mem::take(&mut self.fail_next_step) {
            return Err(BackendError::Execution("injected step failure".into()));
        }
        if !self.step_delay.is_zero() {
            std::thread::sleep(self.step_delay);
        }
        let mut outputs = Vec::new();
        for sequence in batch.sequences() {
            if !sequence.should_sample() {
                continue;
            }
            let previous =
                sequence.token_ids().last().copied().ok_or_else(|| {
                    BackendError::Execution("sampled sequence has no token".into())
                })?;
            let token_id = if previous.get() >= 1000 {
                previous
                    .get()
                    .checked_add(1)
                    .map(TokenId::new)
                    .ok_or_else(|| BackendError::Execution("test token overflowed".into()))?
            } else {
                TokenId::new(1000)
            };
            outputs.push(StepOutput {
                request_id: sequence.request_id(),
                token_id,
            });
        }
        Ok(outputs)
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
        Ok(format!(" token{}", token_id.saturating_sub(1000)))
    }
}
