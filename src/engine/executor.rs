use std::sync::Arc;

use thiserror::Error;

use crate::model::Model;

use super::{ForwardBatch, RequestId, TokenId};

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ExecutionError {
    #[error("model execution failed: {0}")]
    Execution(String),
    #[error("executor is unavailable: {0}")]
    Unavailable(String),
    #[error("completion {0} is not pending")]
    InvalidCompletion(u64),
}

#[derive(Debug, Clone)]
pub struct GeneratedToken {
    pub request_id: RequestId,
    pub token_id: TokenId,
}

/// Mode-aware result of one submitted batch.
///
/// Generation is the only mode implemented today. Keeping the mode tag at the
/// executor boundary avoids baking one-token generation into the protocol when
/// pooling, prompt logprobs, or speculative multi-token acceptance are added.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ExecutionOutput {
    Generation { tokens: Vec<GeneratedToken> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompletionId(u64);

impl CompletionId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchTicket {
    completion: CompletionId,
}

impl BatchTicket {
    pub const fn new(completion: CompletionId) -> Self {
        Self { completion }
    }

    pub const fn completion(self) -> CompletionId {
        self.completion
    }
}

/// Reusable completion storage for executors whose device call is currently
/// synchronous. It implements the same ticket protocol as a future event-based
/// executor and enforces the initial one-in-flight contract.
#[derive(Debug, Default)]
#[cfg(any(test, feature = "cuda", feature = "test-backend"))]
pub(crate) struct ImmediateCompletion {
    next: u64,
    pending: Option<(CompletionId, ExecutionOutput)>,
}

#[cfg(any(test, feature = "cuda", feature = "test-backend"))]
impl ImmediateCompletion {
    pub(crate) fn ensure_available(&self) -> Result<(), ExecutionError> {
        if self.pending.is_some() {
            return Err(ExecutionError::Unavailable(
                "the one-in-flight completion slot is occupied".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn submit(
        &mut self,
        output: ExecutionOutput,
    ) -> Result<BatchTicket, ExecutionError> {
        self.ensure_available()?;
        let completion = CompletionId::new(self.next);
        self.next = self.next.wrapping_add(1);
        self.pending = Some((completion, output));
        Ok(BatchTicket::new(completion))
    }

    pub(crate) fn poll(
        &mut self,
        completion: CompletionId,
    ) -> Result<Option<ExecutionOutput>, ExecutionError> {
        match self.pending.as_ref() {
            Some((pending, _)) if *pending == completion => {}
            _ => return Err(ExecutionError::InvalidCompletion(completion.get())),
        }
        Ok(self.pending.take().map(|(_, output)| output))
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ExecutionStats {
    pub eager_forwards: u64,
    pub graph_replays: u64,
    pub graph_captures: u64,
    pub packed_decode_forwards: u64,
    pub packed_decode_requests: u64,
}

/// Thread-confined model executor. Production executors are constructed and
/// used on the dedicated engine thread, so CUDA handles do not need to be
/// movable. `poll` is non-blocking; physical request state remains live until
/// the returned completion is observed, even after cancellation.
pub trait ModelExecutor: 'static {
    fn model(&self) -> Arc<dyn Model>;

    fn submit(&mut self, batch: &ForwardBatch) -> Result<BatchTicket, ExecutionError>;

    fn poll(&mut self, completion: CompletionId)
    -> Result<Option<ExecutionOutput>, ExecutionError>;

    fn take_execution_stats(&mut self) -> ExecutionStats {
        ExecutionStats::default()
    }

    fn shutdown(&mut self) -> Result<(), ExecutionError> {
        Ok(())
    }
}

impl<T: ModelExecutor + ?Sized> ModelExecutor for Box<T> {
    fn model(&self) -> Arc<dyn Model> {
        (**self).model()
    }

    fn submit(&mut self, batch: &ForwardBatch) -> Result<BatchTicket, ExecutionError> {
        (**self).submit(batch)
    }

    fn poll(
        &mut self,
        completion: CompletionId,
    ) -> Result<Option<ExecutionOutput>, ExecutionError> {
        (**self).poll(completion)
    }

    fn take_execution_stats(&mut self) -> ExecutionStats {
        (**self).take_execution_stats()
    }

    fn shutdown(&mut self) -> Result<(), ExecutionError> {
        (**self).shutdown()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immediate_completion_enforces_ticket_identity_and_one_in_flight() {
        let mut completions = ImmediateCompletion::default();
        let first = completions
            .submit(ExecutionOutput::Generation { tokens: Vec::new() })
            .unwrap();
        assert!(matches!(
            completions.submit(ExecutionOutput::Generation { tokens: Vec::new() }),
            Err(ExecutionError::Unavailable(_))
        ));
        assert!(matches!(
            completions.poll(CompletionId::new(first.completion().get() + 1)),
            Err(ExecutionError::InvalidCompletion(_))
        ));
        assert!(completions.poll(first.completion()).unwrap().is_some());
        assert!(matches!(
            completions.poll(first.completion()),
            Err(ExecutionError::InvalidCompletion(_))
        ));
        assert!(
            completions
                .submit(ExecutionOutput::Generation { tokens: Vec::new() })
                .is_ok()
        );
    }
}
