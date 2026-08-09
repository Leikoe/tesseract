use crate::engine::{BatchLoweringError, ForwardBatch, RequestId, SamplingInput, SequenceIndex};

#[derive(Debug, Clone, Copy)]
pub(crate) struct SampleTarget {
    pub request_id: RequestId,
    pub sampling: SamplingInput,
}

/// Stable host-side materialization consumed by CUDA model programs.
/// Architecture code never reconstructs scheduler state from these arrays.
#[derive(Debug, Default)]
pub(crate) struct CudaBatch {
    pub token_ids: Vec<u32>,
    pub positions: Vec<u32>,
    pub current_slots: Vec<u32>,
    pub request_indices: Vec<u32>,
    pub recurrent_slots: Vec<Option<u32>>,
    pub context_lengths: Vec<i32>,
    pub(super) context_storage: Vec<Vec<u32>>,
    pub(super) num_requests: usize,
    pub sample_rows: Vec<u32>,
    pub samples: Vec<SampleTarget>,
    pub all_samples_greedy: bool,
    pub num_prefill_tokens: usize,
}

impl CudaBatch {
    pub fn lower_into(&mut self, batch: &ForwardBatch) -> Result<(), BatchLoweringError> {
        self.token_ids.clear();
        self.positions.clear();
        self.current_slots.clear();
        self.request_indices.clear();
        self.recurrent_slots.clear();
        self.context_lengths.clear();
        self.sample_rows.clear();
        self.samples.clear();
        self.token_ids.reserve(batch.num_tokens());
        self.positions.reserve(batch.num_tokens());
        self.current_slots.reserve(batch.num_tokens());
        self.request_indices.reserve(batch.num_tokens());
        self.recurrent_slots.reserve(batch.len());
        self.context_lengths.reserve(batch.num_tokens());
        if self.context_storage.len() < batch.len() {
            self.context_storage.resize_with(batch.len(), Vec::new);
        }
        for context in &mut self.context_storage[..batch.len()] {
            context.clear();
        }
        self.num_requests = batch.len();
        self.all_samples_greedy = true;
        self.num_prefill_tokens = batch.num_prefill_tokens();

        for (request_index, sequence) in batch.sequences().iter().enumerate() {
            let request_id = sequence.request_id();
            let end = sequence
                .position()
                .get()
                .checked_add(sequence.num_tokens())
                .ok_or(BatchLoweringError::PositionOverflow(request_id))?;
            let request_index = SequenceIndex::try_from_usize(request_index)
                .map_err(|_| BatchLoweringError::TooManySequences)?;
            self.recurrent_slots
                .push(sequence.recurrent_slot().map(|slot| slot.get()));
            let query_range = batch.query_range(request_index);
            self.request_indices
                .extend(std::iter::repeat_n(request_index.get(), query_range.len()));
            self.token_ids
                .extend(sequence.token_ids().iter().map(|token| token.get()));
            for position in sequence.position().get()..end {
                self.positions.push(
                    u32::try_from(position)
                        .map_err(|_| BatchLoweringError::PositionOutOfRange(request_id))?,
                );
                self.context_lengths.push(
                    i32::try_from(position + 1)
                        .map_err(|_| BatchLoweringError::ContextLengthOutOfRange(request_id))?,
                );
            }
            self.current_slots
                .extend(sequence.kv_slots().iter().map(|slot| slot.get()));
            self.context_storage[request_index.get() as usize]
                .extend(sequence.context_slots().iter().map(|slot| slot.get()));
            if let Some(sampling) = sequence.sampling() {
                let query_row = u32::try_from(query_range.end - 1)
                    .map_err(|_| BatchLoweringError::SampleRowOutOfRange(request_id))?;
                self.sample_rows.push(query_row);
                self.samples.push(SampleTarget {
                    request_id,
                    sampling,
                });
                self.all_samples_greedy &= sampling.is_greedy();
            }
        }
        Ok(())
    }

    pub fn request_count(&self) -> usize {
        self.num_requests
    }

    pub fn contexts(&self) -> &[Vec<u32>] {
        &self.context_storage[..self.num_requests]
    }

    pub fn num_tokens(&self) -> usize {
        self.token_ids.len()
    }

    pub fn is_packed_greedy_decode(&self) -> bool {
        self.num_prefill_tokens == 0
            && self.all_samples_greedy
            && self.samples.len() == self.request_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{ForwardPhase, ForwardSequence, KvSlot, Position, TokenId};

    #[test]
    fn lowers_mixed_batch_to_aligned_flat_metadata() {
        let prefill_id = RequestId::now_v7();
        let decode_id = RequestId::now_v7();
        let prefill = ForwardSequence::try_new(
            prefill_id,
            ForwardPhase::Prefill,
            Position::new(0),
            vec![TokenId::new(10), TokenId::new(11)],
            vec![KvSlot::new(0), KvSlot::new(1)],
            vec![KvSlot::new(0), KvSlot::new(1)],
            None,
        )
        .unwrap();
        let sampling = SamplingInput::try_new(0.0, 1.0, 0.0).unwrap();
        let decode = ForwardSequence::try_new(
            decode_id,
            ForwardPhase::Decode,
            Position::new(2),
            vec![TokenId::new(12)],
            vec![KvSlot::new(4)],
            vec![KvSlot::new(2), KvSlot::new(3), KvSlot::new(4)],
            Some(sampling),
        )
        .unwrap();
        let arena_id = crate::engine::StateSchema::try_flat_kv(8)
            .unwrap()
            .arena_id();
        let batch = ForwardBatch::try_from_sequences(arena_id, vec![decode, prefill]).unwrap();

        let mut lowered = CudaBatch::default();
        lowered.lower_into(&batch).unwrap();

        assert_eq!(lowered.token_ids, [10, 11, 12]);
        assert_eq!(lowered.positions, [0, 1, 2]);
        assert_eq!(lowered.current_slots, [0, 1, 4]);
        assert_eq!(lowered.request_indices, [0, 0, 1]);
        assert_eq!(lowered.recurrent_slots, [None, None]);
        assert_eq!(lowered.context_lengths, [1, 2, 3]);
        assert_eq!(lowered.contexts(), [vec![0, 1], vec![2, 3, 4]]);
        assert_eq!(lowered.sample_rows, [2]);
        assert_eq!(lowered.samples.len(), 1);
        assert_eq!(lowered.samples[0].request_id, decode_id);
        assert_eq!(lowered.num_prefill_tokens, 2);
        assert!(!lowered.is_packed_greedy_decode());
    }

    #[test]
    fn repeated_lowering_reuses_storage_without_stale_metadata() {
        let arena_id = crate::engine::StateSchema::try_flat_kv(8)
            .unwrap()
            .arena_id();
        let first_id = RequestId::now_v7();
        let first = ForwardSequence::try_new(
            first_id,
            ForwardPhase::Prefill,
            Position::new(0),
            vec![TokenId::new(10), TokenId::new(11), TokenId::new(12)],
            vec![KvSlot::new(0), KvSlot::new(1), KvSlot::new(2)],
            vec![KvSlot::new(0), KvSlot::new(1), KvSlot::new(2)],
            None,
        )
        .unwrap();
        let first_peer = ForwardSequence::try_new(
            RequestId::now_v7(),
            ForwardPhase::Prefill,
            Position::new(0),
            vec![TokenId::new(13)],
            vec![KvSlot::new(4)],
            vec![KvSlot::new(4)],
            None,
        )
        .unwrap();
        let first = ForwardBatch::try_from_sequences(arena_id, vec![first, first_peer]).unwrap();
        let mut lowered = CudaBatch::default();
        lowered.lower_into(&first).unwrap();
        let token_storage = lowered.token_ids.as_ptr();
        let context_storage = lowered.contexts()[0].as_ptr();
        let peer_context_storage = lowered.contexts()[1].as_ptr();

        let second_id = RequestId::now_v7();
        let sampling = SamplingInput::try_new(0.0, 1.0, 0.0).unwrap();
        let second = ForwardSequence::try_new(
            second_id,
            ForwardPhase::Decode,
            Position::new(0),
            vec![TokenId::new(20)],
            vec![KvSlot::new(3)],
            vec![KvSlot::new(3)],
            Some(sampling),
        )
        .unwrap();
        let second = ForwardBatch::try_from_sequences(arena_id, vec![second]).unwrap();
        lowered.lower_into(&second).unwrap();

        assert_eq!(lowered.token_ids.as_ptr(), token_storage);
        assert_eq!(lowered.contexts()[0].as_ptr(), context_storage);
        assert_eq!(lowered.token_ids, [20]);
        assert_eq!(lowered.positions, [0]);
        assert_eq!(lowered.current_slots, [3]);
        assert_eq!(lowered.request_indices, [0]);
        assert_eq!(lowered.recurrent_slots, [None]);
        assert_eq!(lowered.context_lengths, [1]);
        assert_eq!(lowered.contexts(), [vec![3]]);
        assert_eq!(lowered.sample_rows, [0]);
        assert_eq!(lowered.samples.len(), 1);
        assert_eq!(lowered.samples[0].request_id, second_id);
        assert!(lowered.is_packed_greedy_decode());

        lowered.lower_into(&first).unwrap();
        assert_eq!(lowered.contexts()[1].as_ptr(), peer_context_storage);
        assert_eq!(lowered.contexts()[1], [4]);
    }
}
