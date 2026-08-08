use std::{collections::HashSet, ops::Range};

use thiserror::Error;

use super::RequestId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Position(usize);

impl Position {
    pub(crate) const fn new(value: usize) -> Self {
        Self(value)
    }

    pub const fn get(self) -> usize {
        self.0
    }

    fn checked_advance(self, tokens: usize) -> Option<Self> {
        self.0.checked_add(tokens).map(Self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct KvSlot(u32);

impl KvSlot {
    pub(crate) const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for KvSlot {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct QueryRow(usize);

impl QueryRow {
    const ZERO: Self = Self(0);

    pub const fn get(self) -> usize {
        self.0
    }

    fn checked_advance(self, tokens: usize) -> Option<Self> {
        self.0.checked_add(tokens).map(Self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct SequenceIndex(u32);

impl SequenceIndex {
    pub fn try_from_usize(value: usize) -> Result<Self, ForwardBatchError> {
        u32::try_from(value)
            .map(Self)
            .map_err(|_| ForwardBatchError::TooManySequences)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardPhase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForwardKind {
    Empty,
    Prefill,
    Decode,
    Mixed {
        prefill_sequences: Range<usize>,
        decode_sequences: Range<usize>,
    },
}

#[derive(Debug, Clone)]
pub struct ForwardSequence {
    request_id: RequestId,
    phase: ForwardPhase,
    position: Position,
    kv_slots: Vec<KvSlot>,
    sample: bool,
}

impl ForwardSequence {
    pub(crate) fn try_new(
        request_id: RequestId,
        phase: ForwardPhase,
        position: Position,
        kv_slots: Vec<KvSlot>,
        sample: bool,
    ) -> Result<Self, ForwardBatchError> {
        if kv_slots.is_empty() {
            return Err(ForwardBatchError::EmptySequence { request_id });
        }
        if phase == ForwardPhase::Decode && kv_slots.len() != 1 {
            return Err(ForwardBatchError::WrongDecodeTokenCount {
                request_id,
                tokens: kv_slots.len(),
            });
        }
        if phase == ForwardPhase::Decode && !sample {
            return Err(ForwardBatchError::DecodeMustSample { request_id });
        }
        position
            .checked_advance(kv_slots.len())
            .ok_or(ForwardBatchError::PositionOverflow { request_id })?;

        Ok(Self {
            request_id,
            phase,
            position,
            kv_slots,
            sample,
        })
    }

    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    pub const fn phase(&self) -> ForwardPhase {
        self.phase
    }

    pub const fn position(&self) -> Position {
        self.position
    }

    pub fn num_tokens(&self) -> usize {
        self.kv_slots.len()
    }

    pub fn kv_slots(&self) -> &[KvSlot] {
        &self.kv_slots
    }

    pub const fn should_sample(&self) -> bool {
        self.sample
    }
}

/// A validated scheduler decision presented to an executor.
///
/// Each request owns one contiguous query range. Mixed batches are stable-
/// partitioned into prefill sequences followed by decode sequences, independent
/// of scheduler selection priority.
#[derive(Debug, Clone)]
pub struct ForwardBatch {
    kind: ForwardKind,
    sequences: Vec<ForwardSequence>,
    query_start_offsets: Vec<QueryRow>,
    num_tokens: usize,
    num_decode_tokens: usize,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ForwardBatchError {
    #[error("forward sequence for request {request_id} contains zero tokens")]
    EmptySequence { request_id: RequestId },
    #[error(
        "decode sequence for request {request_id} must contain exactly one token, got {tokens}"
    )]
    WrongDecodeTokenCount {
        request_id: RequestId,
        tokens: usize,
    },
    #[error("decode sequence for request {request_id} must sample its scheduled token")]
    DecodeMustSample { request_id: RequestId },
    #[error("request {request_id} appears more than once in one forward batch")]
    DuplicateRequest { request_id: RequestId },
    #[error("physical KV slot {slot} is assigned more than once in one forward batch")]
    DuplicateKvSlot { slot: KvSlot },
    #[error("forward token count overflowed")]
    TokenCountOverflow,
    #[error("scheduled range for request {request_id} overflowed")]
    PositionOverflow { request_id: RequestId },
    #[error("forward batch has more sequences than a device sequence index can represent")]
    TooManySequences,
}

impl ForwardBatch {
    pub(crate) fn try_from_sequences(
        mut sequences: Vec<ForwardSequence>,
    ) -> Result<Self, ForwardBatchError> {
        let mut request_ids = HashSet::with_capacity(sequences.len());
        let mut kv_slots = HashSet::new();

        for sequence in &sequences {
            if !request_ids.insert(sequence.request_id) {
                return Err(ForwardBatchError::DuplicateRequest {
                    request_id: sequence.request_id,
                });
            }
            for &slot in &sequence.kv_slots {
                if !kv_slots.insert(slot) {
                    return Err(ForwardBatchError::DuplicateKvSlot { slot });
                }
            }
        }

        sequences.sort_by_key(|sequence| match sequence.phase {
            ForwardPhase::Prefill => 0,
            ForwardPhase::Decode => 1,
        });

        let num_prefill_sequences =
            sequences.partition_point(|sequence| sequence.phase == ForwardPhase::Prefill);
        let kind = match (num_prefill_sequences, sequences.len()) {
            (_, 0) => ForwardKind::Empty,
            (0, _) => ForwardKind::Decode,
            (prefill, total) if prefill == total => ForwardKind::Prefill,
            (prefill, total) => ForwardKind::Mixed {
                prefill_sequences: 0..prefill,
                decode_sequences: prefill..total,
            },
        };

        SequenceIndex::try_from_usize(sequences.len())?;
        let mut query_start_offsets = Vec::with_capacity(sequences.len() + 1);
        let mut next_query_row = QueryRow::ZERO;
        let mut num_decode_tokens = 0usize;
        query_start_offsets.push(next_query_row);
        for sequence in &sequences {
            next_query_row = next_query_row
                .checked_advance(sequence.num_tokens())
                .ok_or(ForwardBatchError::TokenCountOverflow)?;
            query_start_offsets.push(next_query_row);
            if sequence.phase == ForwardPhase::Decode {
                num_decode_tokens += sequence.num_tokens();
            }
        }
        let num_tokens = next_query_row.get();

        debug_assert_eq!(query_start_offsets.len(), sequences.len() + 1);
        debug_assert_eq!(query_start_offsets.last().copied(), Some(next_query_row));
        debug_assert!(query_start_offsets.windows(2).all(|rows| rows[0] < rows[1]));
        debug_assert!(num_decode_tokens <= num_tokens);

        Ok(Self {
            kind,
            sequences,
            query_start_offsets,
            num_tokens,
            num_decode_tokens,
        })
    }

    pub const fn kind(&self) -> &ForwardKind {
        &self.kind
    }

    pub fn sequences(&self) -> &[ForwardSequence] {
        &self.sequences
    }

    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    pub const fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    pub const fn num_decode_tokens(&self) -> usize {
        self.num_decode_tokens
    }

    pub const fn num_prefill_tokens(&self) -> usize {
        self.num_tokens - self.num_decode_tokens
    }

    pub fn query_start_offsets(&self) -> &[QueryRow] {
        &self.query_start_offsets
    }

    pub fn query_range(&self, sequence_index: SequenceIndex) -> Range<usize> {
        let index = sequence_index.get() as usize;
        self.query_start_offsets[index].get()..self.query_start_offsets[index + 1].get()
    }
}

impl std::ops::Deref for ForwardBatch {
    type Target = [ForwardSequence];

    fn deref(&self) -> &Self::Target {
        self.sequences()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slots(values: &[u32]) -> Vec<KvSlot> {
        values.iter().copied().map(KvSlot::new).collect()
    }

    fn sequence(
        request_id: RequestId,
        phase: ForwardPhase,
        position: usize,
        slot_values: &[u32],
    ) -> ForwardSequence {
        ForwardSequence::try_new(
            request_id,
            phase,
            Position::new(position),
            slots(slot_values),
            phase == ForwardPhase::Decode,
        )
        .unwrap()
    }

    #[test]
    fn mixed_batch_has_an_explicit_stable_row_partition() {
        let decode = RequestId::now_v7();
        let first_prefill = RequestId::now_v7();
        let second_prefill = RequestId::now_v7();
        let batch = ForwardBatch::try_from_sequences(vec![
            sequence(decode, ForwardPhase::Decode, 17, &[9]),
            sequence(first_prefill, ForwardPhase::Prefill, 0, &[4, 5]),
            sequence(second_prefill, ForwardPhase::Prefill, 2, &[6]),
        ])
        .unwrap();

        assert_eq!(
            batch.kind(),
            &ForwardKind::Mixed {
                prefill_sequences: 0..2,
                decode_sequences: 2..3,
            }
        );
        assert_eq!(
            batch
                .sequences()
                .iter()
                .map(ForwardSequence::request_id)
                .collect::<Vec<_>>(),
            [first_prefill, second_prefill, decode]
        );
        assert_eq!(
            batch
                .query_start_offsets()
                .iter()
                .copied()
                .map(QueryRow::get)
                .collect::<Vec<_>>(),
            [0, 2, 3, 4]
        );
        assert_eq!(
            batch.query_range(SequenceIndex::try_from_usize(2).unwrap()),
            3..4
        );
    }

    #[test]
    fn rejects_duplicate_requests_and_slots() {
        let first = RequestId::now_v7();
        let second = RequestId::now_v7();
        assert_eq!(
            ForwardBatch::try_from_sequences(vec![
                sequence(first, ForwardPhase::Prefill, 0, &[1]),
                sequence(first, ForwardPhase::Decode, 1, &[2]),
            ])
            .unwrap_err(),
            ForwardBatchError::DuplicateRequest { request_id: first }
        );
        assert_eq!(
            ForwardBatch::try_from_sequences(vec![
                sequence(first, ForwardPhase::Prefill, 0, &[1]),
                sequence(second, ForwardPhase::Prefill, 0, &[1]),
            ])
            .unwrap_err(),
            ForwardBatchError::DuplicateKvSlot {
                slot: KvSlot::new(1),
            }
        );
    }

    #[test]
    fn rejects_invalid_sequence_shapes_before_batch_construction() {
        let request_id = RequestId::now_v7();
        assert_eq!(
            ForwardSequence::try_new(
                request_id,
                ForwardPhase::Prefill,
                Position::new(0),
                vec![],
                false,
            )
            .unwrap_err(),
            ForwardBatchError::EmptySequence { request_id }
        );
        assert_eq!(
            ForwardSequence::try_new(
                request_id,
                ForwardPhase::Decode,
                Position::new(1),
                slots(&[3, 4]),
                true,
            )
            .unwrap_err(),
            ForwardBatchError::WrongDecodeTokenCount {
                request_id,
                tokens: 2,
            }
        );
        assert_eq!(
            ForwardSequence::try_new(
                request_id,
                ForwardPhase::Decode,
                Position::new(1),
                slots(&[3]),
                false,
            )
            .unwrap_err(),
            ForwardBatchError::DecodeMustSample { request_id }
        );
    }
}
