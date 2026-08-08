use std::{sync::Arc, time::Duration};

use crate::model::{ChatMessage, IncrementalDecoder, Model, ModelError, ModelSummary};

use super::{
    BatchTicket, CompletionId, ExecutionError, ExecutionOutput, ForwardBatch, GeneratedTokens,
    ImmediateCompletion, ModelExecutor, StateSchema, TokenId,
};

/// Deterministic model executor used only by API and scheduler tests.
pub struct DeterministicExecutor {
    model: Arc<TestModel>,
    submission_delay: Duration,
    fail_next_submission: bool,
    completions: ImmediateCompletion,
    state_schema: StateSchema,
}

impl DeterministicExecutor {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model: Arc::new(TestModel {
                id: model_id.into(),
            }),
            submission_delay: Duration::ZERO,
            fail_next_submission: false,
            completions: ImmediateCompletion::default(),
            state_schema: StateSchema::try_flat_kv(u32::MAX as usize).unwrap(),
        }
    }

    pub fn with_submission_delay(mut self, submission_delay: Duration) -> Self {
        self.submission_delay = submission_delay;
        self
    }

    pub fn with_state_capacity(mut self, capacity: usize) -> Self {
        self.state_schema = StateSchema::try_flat_kv(capacity).unwrap();
        self
    }

    pub fn failing_next_submission(mut self) -> Self {
        self.fail_next_submission = true;
        self
    }
}

impl ModelExecutor for DeterministicExecutor {
    fn model(&self) -> Arc<dyn Model> {
        self.model.clone()
    }

    fn state_schema(&self) -> &StateSchema {
        &self.state_schema
    }

    fn submit(&mut self, batch: &ForwardBatch) -> Result<BatchTicket, ExecutionError> {
        if batch.arena_id() != self.state_schema.arena_id() {
            return Err(ExecutionError::StateArenaMismatch {
                batch: batch.arena_id(),
                executor: self.state_schema.arena_id(),
            });
        }
        self.completions.ensure_available()?;
        if std::mem::take(&mut self.fail_next_submission) {
            return Err(ExecutionError::Execution(
                "injected submission failure".into(),
            ));
        }
        if !self.submission_delay.is_zero() {
            std::thread::sleep(self.submission_delay);
        }
        let mut tokens = Vec::new();
        for sequence in batch.sequences() {
            if !sequence.should_sample() {
                continue;
            }
            let previous =
                sequence.token_ids().last().copied().ok_or_else(|| {
                    ExecutionError::Execution("sampled sequence has no token".into())
                })?;
            let token_id = if previous.get() >= 1000 {
                previous
                    .get()
                    .checked_add(1)
                    .map(TokenId::new)
                    .ok_or_else(|| ExecutionError::Execution("test token overflowed".into()))?
            } else {
                TokenId::new(1000)
            };
            tokens.push(GeneratedTokens::one(sequence.request_id(), token_id));
        }
        self.completions
            .submit(ExecutionOutput::Generation { requests: tokens })
    }

    fn poll(
        &mut self,
        completion: CompletionId,
    ) -> Result<Option<ExecutionOutput>, ExecutionError> {
        self.completions.poll(completion)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{ForwardPhase, ForwardSequence, KvSlot, Position};

    #[test]
    fn executor_rejects_slots_from_another_arena() {
        let mut executor = DeterministicExecutor::new("test-model").with_state_capacity(8);
        let foreign = StateSchema::try_flat_kv(8).unwrap();
        let request_id = crate::engine::RequestId::now_v7();
        let sequence = ForwardSequence::try_new(
            request_id,
            ForwardPhase::Prefill,
            Position::new(0),
            vec![TokenId::new(1)],
            vec![KvSlot::new(0)],
            vec![KvSlot::new(0)],
            None,
        )
        .unwrap();
        let batch = ForwardBatch::try_from_sequences(foreign.arena_id(), vec![sequence]).unwrap();

        assert!(matches!(
            executor.submit(&batch),
            Err(ExecutionError::StateArenaMismatch { .. })
        ));
    }
}
