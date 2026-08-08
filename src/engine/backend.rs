use std::{collections::HashSet, sync::Arc};

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
    pub phase: WorkPhase,
    pub position: usize,
    pub num_tokens: usize,
    pub kv_slots: Vec<u32>,
    pub sample: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkPhase {
    Prefill,
    Decode,
}

/// One scheduler decision presented to the model backend.
///
/// A batch contains at most one contiguous token range per request. Backends
/// may flatten every range into one ragged model forward; they must not assume
/// all ranges have the same length or that prefill and decode are separated.
#[derive(Debug, Clone)]
pub struct ScheduledBatch {
    work: Vec<ScheduledWork>,
    query_start_offsets: Vec<usize>,
    num_tokens: usize,
    num_decode_tokens: usize,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ScheduledBatchError {
    #[error("scheduled work for request {request_id} contains zero tokens")]
    EmptyWork { request_id: RequestId },
    #[error("scheduled work for request {request_id} has {slots} KV slots for {tokens} tokens")]
    WrongSlotCount {
        request_id: RequestId,
        tokens: usize,
        slots: usize,
    },
    #[error("decode work for request {request_id} must contain exactly one token, got {tokens}")]
    WrongDecodeTokenCount {
        request_id: RequestId,
        tokens: usize,
    },
    #[error("decode work for request {request_id} must sample its scheduled token")]
    DecodeMustSample { request_id: RequestId },
    #[error("request {request_id} appears more than once in one scheduler batch")]
    DuplicateRequest { request_id: RequestId },
    #[error("physical KV slot {slot} is assigned more than once in one scheduler batch")]
    DuplicateKvSlot { slot: u32 },
    #[error("scheduled token count overflowed")]
    TokenCountOverflow,
    #[error("scheduled range for request {request_id} overflowed")]
    PositionOverflow { request_id: RequestId },
}

impl ScheduledBatch {
    pub fn try_from_work(work: Vec<ScheduledWork>) -> Result<Self, ScheduledBatchError> {
        let mut request_ids = HashSet::with_capacity(work.len());
        let mut kv_slots = HashSet::new();
        let mut num_tokens = 0usize;
        let mut num_decode_tokens = 0usize;
        let mut query_start_offsets = Vec::with_capacity(work.len() + 1);
        query_start_offsets.push(0);

        for item in &work {
            if item.num_tokens == 0 {
                return Err(ScheduledBatchError::EmptyWork {
                    request_id: item.request_id,
                });
            }
            if item.kv_slots.len() != item.num_tokens {
                return Err(ScheduledBatchError::WrongSlotCount {
                    request_id: item.request_id,
                    tokens: item.num_tokens,
                    slots: item.kv_slots.len(),
                });
            }
            if item.phase == WorkPhase::Decode && item.num_tokens != 1 {
                return Err(ScheduledBatchError::WrongDecodeTokenCount {
                    request_id: item.request_id,
                    tokens: item.num_tokens,
                });
            }
            if item.phase == WorkPhase::Decode && !item.sample {
                return Err(ScheduledBatchError::DecodeMustSample {
                    request_id: item.request_id,
                });
            }
            if !request_ids.insert(item.request_id) {
                return Err(ScheduledBatchError::DuplicateRequest {
                    request_id: item.request_id,
                });
            }
            for &slot in &item.kv_slots {
                if !kv_slots.insert(slot) {
                    return Err(ScheduledBatchError::DuplicateKvSlot { slot });
                }
            }
            item.position.checked_add(item.num_tokens).ok_or(
                ScheduledBatchError::PositionOverflow {
                    request_id: item.request_id,
                },
            )?;
            num_tokens = num_tokens
                .checked_add(item.num_tokens)
                .ok_or(ScheduledBatchError::TokenCountOverflow)?;
            query_start_offsets.push(num_tokens);
            if item.phase == WorkPhase::Decode {
                num_decode_tokens += item.num_tokens;
            }
        }

        debug_assert_eq!(query_start_offsets.len(), work.len() + 1);
        debug_assert_eq!(query_start_offsets.last().copied(), Some(num_tokens));
        debug_assert!(query_start_offsets.windows(2).all(|pair| pair[0] < pair[1]));
        debug_assert!(num_decode_tokens <= num_tokens);

        Ok(Self {
            work,
            query_start_offsets,
            num_tokens,
            num_decode_tokens,
        })
    }

    pub fn work(&self) -> &[ScheduledWork] {
        &self.work
    }

    pub fn len(&self) -> usize {
        self.work.len()
    }

    pub fn is_empty(&self) -> bool {
        self.work.is_empty()
    }

    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    pub fn num_decode_tokens(&self) -> usize {
        self.num_decode_tokens
    }

    pub fn num_prefill_tokens(&self) -> usize {
        self.num_tokens - self.num_decode_tokens
    }

    /// Prefix sum of scheduled query tokens. Request `i` owns rows
    /// `query_start_offsets[i]..query_start_offsets[i + 1]` in a flattened
    /// model batch.
    pub fn query_start_offsets(&self) -> &[usize] {
        &self.query_start_offsets
    }

    pub fn query_range(&self, request_index: usize) -> std::ops::Range<usize> {
        self.query_start_offsets[request_index]..self.query_start_offsets[request_index + 1]
    }
}

impl std::ops::Deref for ScheduledBatch {
    type Target = [ScheduledWork];

    fn deref(&self) -> &Self::Target {
        self.work()
    }
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
    pub packed_decode_forwards: u64,
    pub packed_decode_requests: u64,
}

/// Thread-confined model backend. Production backends are constructed and used
/// on the dedicated engine thread, so CUDA handles do not need to be movable.
pub trait Backend: 'static {
    fn model(&self) -> Arc<dyn Model>;

    fn add_request(&mut self, request: &GenerateRequest) -> Result<PreparedRequest, BackendError>;

    fn step(&mut self, batch: &ScheduledBatch) -> Result<Vec<StepOutput>, BackendError>;

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

    fn step(&mut self, batch: &ScheduledBatch) -> Result<Vec<StepOutput>, BackendError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn work(
        request_id: RequestId,
        phase: WorkPhase,
        position: usize,
        slots: &[u32],
    ) -> ScheduledWork {
        ScheduledWork {
            request_id,
            phase,
            position,
            num_tokens: slots.len(),
            kv_slots: slots.to_vec(),
            sample: phase == WorkPhase::Decode,
        }
    }

    #[test]
    fn validates_and_summarizes_ragged_mixed_batch() {
        let prefill = RequestId::now_v7();
        let decode = RequestId::now_v7();
        let batch = ScheduledBatch::try_from_work(vec![
            work(prefill, WorkPhase::Prefill, 0, &[4, 5, 6]),
            work(decode, WorkPhase::Decode, 17, &[9]),
        ])
        .unwrap();

        assert_eq!(batch.len(), 2);
        assert_eq!(batch.num_tokens(), 4);
        assert_eq!(batch.num_prefill_tokens(), 3);
        assert_eq!(batch.num_decode_tokens(), 1);
        assert_eq!(batch.query_start_offsets(), &[0, 3, 4]);
        assert_eq!(batch.query_range(0), 0..3);
        assert_eq!(batch.query_range(1), 3..4);
    }

    #[test]
    fn rejects_duplicate_requests_and_slots() {
        let first = RequestId::now_v7();
        let second = RequestId::now_v7();
        assert_eq!(
            ScheduledBatch::try_from_work(vec![
                work(first, WorkPhase::Prefill, 0, &[1]),
                work(first, WorkPhase::Decode, 1, &[2]),
            ])
            .unwrap_err(),
            ScheduledBatchError::DuplicateRequest { request_id: first }
        );
        assert_eq!(
            ScheduledBatch::try_from_work(vec![
                work(first, WorkPhase::Prefill, 0, &[1]),
                work(second, WorkPhase::Prefill, 0, &[1]),
            ])
            .unwrap_err(),
            ScheduledBatchError::DuplicateKvSlot { slot: 1 }
        );
    }

    #[test]
    fn rejects_malformed_ranges() {
        let request_id = RequestId::now_v7();
        let mut empty = work(request_id, WorkPhase::Prefill, 0, &[]);
        assert_eq!(
            ScheduledBatch::try_from_work(vec![empty.clone()]).unwrap_err(),
            ScheduledBatchError::EmptyWork { request_id }
        );
        empty.num_tokens = 1;
        assert_eq!(
            ScheduledBatch::try_from_work(vec![empty]).unwrap_err(),
            ScheduledBatchError::WrongSlotCount {
                request_id,
                tokens: 1,
                slots: 0,
            }
        );
        assert_eq!(
            ScheduledBatch::try_from_work(vec![work(
                request_id,
                WorkPhase::Decode,
                usize::MAX,
                &[3],
            )])
            .unwrap_err(),
            ScheduledBatchError::PositionOverflow { request_id }
        );
        assert_eq!(
            ScheduledBatch::try_from_work(vec![work(request_id, WorkPhase::Decode, 1, &[3, 4],)])
                .unwrap_err(),
            ScheduledBatchError::WrongDecodeTokenCount {
                request_id,
                tokens: 2,
            }
        );
        let mut decode_without_sample = work(request_id, WorkPhase::Decode, 1, &[3]);
        decode_without_sample.sample = false;
        assert_eq!(
            ScheduledBatch::try_from_work(vec![decode_without_sample]).unwrap_err(),
            ScheduledBatchError::DecodeMustSample { request_id }
        );
    }
}
