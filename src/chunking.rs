//! Packed-sequence chunk descriptors shared by stateful accelerator programs.

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PackedChunk {
    request: u32,
    start: u32,
    len: u32,
}

impl PackedChunk {
    pub(crate) const fn request(self) -> u32 {
        self.request
    }

    pub(crate) const fn start(self) -> u32 {
        self.start
    }

    pub(crate) const fn len(self) -> u32 {
        self.len
    }
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChunkPlanError {
    #[error("chunk size must be positive")]
    EmptyChunk,
    #[error("packed query offsets must start at zero")]
    NonzeroStart,
    #[error("packed query offsets must be strictly increasing")]
    NonIncreasingOffsets,
    #[error("request index does not fit accelerator metadata")]
    TooManyRequests,
}

pub(crate) fn plan_packed_queries(
    offsets: &[u32],
    max_chunk: u32,
) -> Result<Vec<PackedChunk>, ChunkPlanError> {
    if max_chunk == 0 {
        return Err(ChunkPlanError::EmptyChunk);
    }
    let Some((&first, _)) = offsets.split_first() else {
        return Ok(Vec::new());
    };
    if first != 0 {
        return Err(ChunkPlanError::NonzeroStart);
    }
    let mut chunks = Vec::new();
    for (request, boundary) in offsets.windows(2).enumerate() {
        let [start, end] = boundary else {
            unreachable!("windows of length two always contain two elements")
        };
        if start >= end {
            return Err(ChunkPlanError::NonIncreasingOffsets);
        }
        let request = u32::try_from(request).map_err(|_| ChunkPlanError::TooManyRequests)?;
        let mut cursor = *start;
        while cursor < *end {
            let len = (*end - cursor).min(max_chunk);
            chunks.push(PackedChunk {
                request,
                start: cursor,
                len,
            });
            cursor += len;
        }
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn rejects_malformed_boundaries() {
        assert_eq!(
            plan_packed_queries(&[1, 2], 16),
            Err(ChunkPlanError::NonzeroStart)
        );
        assert_eq!(
            plan_packed_queries(&[0, 2, 2], 16),
            Err(ChunkPlanError::NonIncreasingOffsets)
        );
        assert_eq!(
            plan_packed_queries(&[0, 1], 0),
            Err(ChunkPlanError::EmptyChunk)
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn chunks_cover_each_request_once_without_crossing_boundaries(
            lengths in prop::collection::vec(1u16..1024, 1..64),
            max_chunk in 1u16..128,
        ) {
            let mut offsets = Vec::with_capacity(lengths.len() + 1);
            offsets.push(0u32);
            for length in &lengths {
                offsets.push(offsets.last().copied().unwrap() + u32::from(*length));
            }
            let chunks = plan_packed_queries(&offsets, u32::from(max_chunk)).unwrap();
            let mut cursor = 0u32;
            for chunk in &chunks {
                let request = chunk.request() as usize;
                prop_assert_eq!(chunk.start(), cursor);
                prop_assert!(chunk.len() > 0);
                prop_assert!(chunk.len() <= u32::from(max_chunk));
                prop_assert!(chunk.start() >= offsets[request]);
                prop_assert!(chunk.start() + chunk.len() <= offsets[request + 1]);
                cursor += chunk.len();
            }
            prop_assert_eq!(cursor, *offsets.last().unwrap());
        }
    }
}
