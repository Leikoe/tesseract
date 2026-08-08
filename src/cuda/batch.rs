use crate::engine::{BatchLoweringError, ForwardBatch, RequestId, SamplingInput, SequenceIndex};

#[derive(Debug, Clone, Copy)]
pub(crate) struct SampleTarget {
    pub request_id: RequestId,
    pub sampling: SamplingInput,
}

/// Stable host-side materialization consumed by CUDA model programs.
/// Architecture code never reconstructs scheduler state from these arrays.
#[derive(Debug)]
pub(crate) struct CudaBatch {
    pub token_ids: Vec<u32>,
    pub positions: Vec<u32>,
    pub current_slots: Vec<u32>,
    pub request_indices: Vec<u32>,
    pub context_lengths: Vec<i32>,
    pub contexts: Vec<Vec<u32>>,
    pub sample_rows: Vec<u32>,
    pub samples: Vec<SampleTarget>,
    pub all_samples_greedy: bool,
    pub num_prefill_tokens: usize,
}

impl CudaBatch {
    pub fn lower(batch: &ForwardBatch) -> Result<Self, BatchLoweringError> {
        let mut lowered = Self {
            token_ids: Vec::with_capacity(batch.num_tokens()),
            positions: Vec::with_capacity(batch.num_tokens()),
            current_slots: Vec::with_capacity(batch.num_tokens()),
            request_indices: Vec::with_capacity(batch.num_tokens()),
            context_lengths: Vec::with_capacity(batch.num_tokens()),
            contexts: Vec::with_capacity(batch.len()),
            sample_rows: Vec::new(),
            samples: Vec::new(),
            all_samples_greedy: true,
            num_prefill_tokens: batch.num_prefill_tokens(),
        };

        for (request_index, sequence) in batch.sequences().iter().enumerate() {
            let request_id = sequence.request_id();
            let end = sequence
                .position()
                .get()
                .checked_add(sequence.num_tokens())
                .ok_or(BatchLoweringError::PositionOverflow(request_id))?;
            let request_index = SequenceIndex::try_from_usize(request_index)
                .map_err(|_| BatchLoweringError::TooManySequences)?;
            let query_range = batch.query_range(request_index);
            lowered
                .request_indices
                .extend(std::iter::repeat_n(request_index.get(), query_range.len()));
            lowered
                .token_ids
                .extend(sequence.token_ids().iter().map(|token| token.get()));
            for position in sequence.position().get()..end {
                lowered.positions.push(
                    u32::try_from(position)
                        .map_err(|_| BatchLoweringError::PositionOutOfRange(request_id))?,
                );
                lowered.context_lengths.push(
                    i32::try_from(position + 1)
                        .map_err(|_| BatchLoweringError::ContextLengthOutOfRange(request_id))?,
                );
            }
            lowered
                .current_slots
                .extend(sequence.kv_slots().iter().map(|slot| slot.get()));
            lowered.contexts.push(
                sequence
                    .context_slots()
                    .iter()
                    .map(|slot| slot.get())
                    .collect(),
            );
            if let Some(sampling) = sequence.sampling() {
                let query_row = u32::try_from(query_range.end - 1)
                    .map_err(|_| BatchLoweringError::SampleRowOutOfRange(request_id))?;
                lowered.sample_rows.push(query_row);
                lowered.samples.push(SampleTarget {
                    request_id,
                    sampling,
                });
                lowered.all_samples_greedy &= sampling.is_greedy();
            }
        }
        Ok(lowered)
    }

    pub fn request_count(&self) -> usize {
        self.contexts.len()
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

        let lowered = CudaBatch::lower(&batch).unwrap();

        assert_eq!(lowered.token_ids, [10, 11, 12]);
        assert_eq!(lowered.positions, [0, 1, 2]);
        assert_eq!(lowered.current_slots, [0, 1, 4]);
        assert_eq!(lowered.request_indices, [0, 0, 1]);
        assert_eq!(lowered.context_lengths, [1, 2, 3]);
        assert_eq!(lowered.contexts, [vec![0, 1], vec![2, 3, 4]]);
        assert_eq!(lowered.sample_rows, [2]);
        assert_eq!(lowered.samples.len(), 1);
        assert_eq!(lowered.samples[0].request_id, decode_id);
        assert_eq!(lowered.num_prefill_tokens, 2);
        assert!(!lowered.is_packed_greedy_decode());
    }
}
