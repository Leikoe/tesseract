use std::sync::atomic::{AtomicU64, Ordering};

use thiserror::Error;

static NEXT_ARENA_ID: AtomicU64 = AtomicU64::new(1);

/// Process-local identity of one matched logical allocator and physical state
/// arena. Numeric slots are meaningful only inside this capability domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateArenaId(u64);

impl StateArenaId {
    fn fresh() -> Result<Self, StateSchemaError> {
        NEXT_ARENA_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map(Self)
            .map_err(|_| StateSchemaError::ArenaIdsExhausted)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for StateArenaId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StateGroupKind {
    FlatKv,
    Recurrent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateGroupSpec {
    kind: StateGroupKind,
    capacity: usize,
}

impl StateGroupSpec {
    pub const fn kind(self) -> StateGroupKind {
        self.kind
    }

    pub const fn capacity(self) -> usize {
        self.capacity
    }
}

/// Immutable physical-state contract selected during executor construction.
/// Group geometry is data; backend-specific layout remains executor-private.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateSchema {
    arena_id: StateArenaId,
    groups: Vec<StateGroupSpec>,
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum StateSchemaError {
    #[error("flat-KV capacity must be in 1..={}", u32::MAX)]
    InvalidFlatKvCapacity,
    #[error("recurrent-state capacity must be in 1..={}", u32::MAX)]
    InvalidRecurrentCapacity,
    #[error("process-local state arena IDs are exhausted")]
    ArenaIdsExhausted,
}

impl StateSchema {
    pub fn try_flat_kv(capacity: usize) -> Result<Self, StateSchemaError> {
        if capacity == 0 || capacity > u32::MAX as usize {
            return Err(StateSchemaError::InvalidFlatKvCapacity);
        }
        Ok(Self {
            arena_id: StateArenaId::fresh()?,
            groups: vec![StateGroupSpec {
                kind: StateGroupKind::FlatKv,
                capacity,
            }],
        })
    }

    pub fn try_hybrid(
        flat_kv_capacity: usize,
        recurrent_capacity: usize,
    ) -> Result<Self, StateSchemaError> {
        if flat_kv_capacity == 0 || flat_kv_capacity > u32::MAX as usize {
            return Err(StateSchemaError::InvalidFlatKvCapacity);
        }
        if recurrent_capacity == 0 || recurrent_capacity > u32::MAX as usize {
            return Err(StateSchemaError::InvalidRecurrentCapacity);
        }
        Ok(Self {
            arena_id: StateArenaId::fresh()?,
            groups: vec![
                StateGroupSpec {
                    kind: StateGroupKind::FlatKv,
                    capacity: flat_kv_capacity,
                },
                StateGroupSpec {
                    kind: StateGroupKind::Recurrent,
                    capacity: recurrent_capacity,
                },
            ],
        })
    }

    pub const fn arena_id(&self) -> StateArenaId {
        self.arena_id
    }

    pub fn groups(&self) -> &[StateGroupSpec] {
        &self.groups
    }

    pub fn flat_kv_capacity(&self) -> Option<usize> {
        self.groups
            .iter()
            .find(|group| group.kind == StateGroupKind::FlatKv)
            .map(|group| group.capacity)
    }

    pub fn recurrent_capacity(&self) -> Option<usize> {
        self.groups
            .iter()
            .find(|group| group.kind == StateGroupKind::Recurrent)
            .map(|group| group.capacity)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn separately_constructed_arenas_never_alias() {
        let first = StateSchema::try_flat_kv(8).unwrap();
        let second = StateSchema::try_flat_kv(8).unwrap();
        assert_ne!(first.arena_id(), second.arena_id());
        assert_eq!(first.flat_kv_capacity(), Some(8));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn flat_kv_schema_preserves_valid_capacity(capacity in 1usize..1_000_000) {
            let schema = StateSchema::try_flat_kv(capacity).unwrap();
            prop_assert_eq!(schema.flat_kv_capacity(), Some(capacity));
            prop_assert_eq!(schema.groups().len(), 1);
            prop_assert_eq!(schema.groups()[0].kind(), StateGroupKind::FlatKv);
        }


        #[test]
        fn hybrid_schema_preserves_both_capacities(
            flat_kv_capacity in 1usize..1_000_000,
            recurrent_capacity in 1usize..10_000,
        ) {
            let schema = StateSchema::try_hybrid(flat_kv_capacity, recurrent_capacity).unwrap();
            prop_assert_eq!(schema.flat_kv_capacity(), Some(flat_kv_capacity));
            prop_assert_eq!(schema.recurrent_capacity(), Some(recurrent_capacity));
            prop_assert_eq!(schema.groups().len(), 2);
            prop_assert_eq!(schema.groups()[1].kind(), StateGroupKind::Recurrent);
        }
    }
}
