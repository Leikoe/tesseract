use std::{collections::HashSet, ops::Range};

use thiserror::Error;

use super::RequestId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct TokenId(u32);

impl TokenId {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingInput {
    temperature: f32,
    top_p: f32,
    random_sample: f64,
}

impl SamplingInput {
    pub(crate) fn try_new(
        temperature: f32,
        top_p: f32,
        random_sample: f64,
    ) -> Result<Self, ForwardBatchError> {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(ForwardBatchError::InvalidSamplingTemperature);
        }
        if !(top_p.is_finite() && 0.0 < top_p && top_p <= 1.0) {
            return Err(ForwardBatchError::InvalidTopP);
        }
        if !random_sample.is_finite() || !(0.0..1.0).contains(&random_sample) {
            return Err(ForwardBatchError::InvalidRandomSample);
        }
        Ok(Self {
            temperature,
            top_p,
            random_sample,
        })
    }

    pub const fn temperature(self) -> f32 {
        self.temperature
    }

    pub const fn top_p(self) -> f32 {
        self.top_p
    }

    pub const fn random_sample(self) -> f64 {
        self.random_sample
    }

    pub const fn is_greedy(self) -> bool {
        self.temperature == 0.0
    }
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
    token_ids: Vec<TokenId>,
    kv_slots: Vec<KvSlot>,
    context_slots: Vec<KvSlot>,
    sampling: Option<SamplingInput>,
}

impl ForwardSequence {
    pub(crate) fn try_new(
        request_id: RequestId,
        phase: ForwardPhase,
        position: Position,
        token_ids: Vec<TokenId>,
        kv_slots: Vec<KvSlot>,
        context_slots: Vec<KvSlot>,
        sampling: Option<SamplingInput>,
    ) -> Result<Self, ForwardBatchError> {
        if kv_slots.is_empty() {
            return Err(ForwardBatchError::EmptySequence { request_id });
        }
        if token_ids.len() != kv_slots.len() {
            return Err(ForwardBatchError::QueryTokenCountMismatch {
                request_id,
                tokens: token_ids.len(),
                slots: kv_slots.len(),
            });
        }
        if phase == ForwardPhase::Decode && kv_slots.len() != 1 {
            return Err(ForwardBatchError::WrongDecodeTokenCount {
                request_id,
                tokens: kv_slots.len(),
            });
        }
        if phase == ForwardPhase::Decode && sampling.is_none() {
            return Err(ForwardBatchError::DecodeMustSample { request_id });
        }
        let context_len = position
            .checked_advance(kv_slots.len())
            .ok_or(ForwardBatchError::PositionOverflow { request_id })?;
        if context_slots.len() != context_len.get() {
            return Err(ForwardBatchError::ContextLengthMismatch {
                request_id,
                expected: context_len.get(),
                actual: context_slots.len(),
            });
        }
        if !context_slots.ends_with(&kv_slots) {
            return Err(ForwardBatchError::ContextTailMismatch { request_id });
        }

        Ok(Self {
            request_id,
            phase,
            position,
            token_ids,
            kv_slots,
            context_slots,
            sampling,
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

    pub fn token_ids(&self) -> &[TokenId] {
        &self.token_ids
    }

    pub fn kv_slots(&self) -> &[KvSlot] {
        &self.kv_slots
    }

    pub fn context_slots(&self) -> &[KvSlot] {
        &self.context_slots
    }

    pub const fn should_sample(&self) -> bool {
        self.sampling.is_some()
    }

    pub const fn sampling(&self) -> Option<SamplingInput> {
        self.sampling
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
    #[error(
        "forward sequence for request {request_id} has {tokens} query tokens but {slots} destination slots"
    )]
    QueryTokenCountMismatch {
        request_id: RequestId,
        tokens: usize,
        slots: usize,
    },
    #[error("decode sequence for request {request_id} must sample its scheduled token")]
    DecodeMustSample { request_id: RequestId },
    #[error(
        "forward sequence for request {request_id} has context length {actual}, expected {expected}"
    )]
    ContextLengthMismatch {
        request_id: RequestId,
        expected: usize,
        actual: usize,
    },
    #[error("forward sequence for request {request_id} does not end with its destination slots")]
    ContextTailMismatch { request_id: RequestId },
    #[error("sampling temperature must be finite and non-negative")]
    InvalidSamplingTemperature,
    #[error("top_p must be finite and in (0, 1]")]
    InvalidTopP,
    #[error("sampling random value must be finite and in [0, 1)")]
    InvalidRandomSample,
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
        let current_slots = slots(slot_values);
        let mut context_slots = (0..position)
            .map(|index| KvSlot::new(10_000 + index as u32))
            .collect::<Vec<_>>();
        context_slots.extend_from_slice(&current_slots);
        ForwardSequence::try_new(
            request_id,
            phase,
            Position::new(position),
            (0..slot_values.len() as u32).map(TokenId::new).collect(),
            current_slots,
            context_slots,
            (phase == ForwardPhase::Decode).then(|| SamplingInput::try_new(0.0, 1.0, 0.0).unwrap()),
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
                vec![],
                vec![],
                None,
            )
            .unwrap_err(),
            ForwardBatchError::EmptySequence { request_id }
        );
        assert_eq!(
            ForwardSequence::try_new(
                request_id,
                ForwardPhase::Decode,
                Position::new(1),
                vec![TokenId::new(1), TokenId::new(2)],
                slots(&[3, 4]),
                slots(&[9, 3, 4]),
                Some(SamplingInput::try_new(0.0, 1.0, 0.0).unwrap()),
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
                vec![TokenId::new(1)],
                slots(&[3]),
                slots(&[9, 3]),
                None,
            )
            .unwrap_err(),
            ForwardBatchError::DecodeMustSample { request_id }
        );

        assert_eq!(
            ForwardSequence::try_new(
                request_id,
                ForwardPhase::Prefill,
                Position::new(0),
                vec![TokenId::new(1), TokenId::new(2)],
                slots(&[3]),
                slots(&[3]),
                None,
            )
            .unwrap_err(),
            ForwardBatchError::QueryTokenCountMismatch {
                request_id,
                tokens: 2,
                slots: 1,
            }
        );
        assert_eq!(
            ForwardSequence::try_new(
                request_id,
                ForwardPhase::Prefill,
                Position::new(2),
                vec![TokenId::new(1)],
                slots(&[3]),
                slots(&[8, 3]),
                None,
            )
            .unwrap_err(),
            ForwardBatchError::ContextLengthMismatch {
                request_id,
                expected: 3,
                actual: 2,
            }
        );
        assert_eq!(
            ForwardSequence::try_new(
                request_id,
                ForwardPhase::Prefill,
                Position::new(1),
                vec![TokenId::new(1)],
                slots(&[3]),
                slots(&[8, 9]),
                None,
            )
            .unwrap_err(),
            ForwardBatchError::ContextTailMismatch { request_id }
        );
    }

    #[test]
    fn sampling_inputs_reject_values_outside_their_contract() {
        assert_eq!(
            SamplingInput::try_new(f32::NAN, 1.0, 0.0).unwrap_err(),
            ForwardBatchError::InvalidSamplingTemperature
        );
        assert_eq!(
            SamplingInput::try_new(1.0, 0.0, 0.0).unwrap_err(),
            ForwardBatchError::InvalidTopP
        );
        assert_eq!(
            SamplingInput::try_new(1.0, 1.0, 1.0).unwrap_err(),
            ForwardBatchError::InvalidRandomSample
        );
    }
}
