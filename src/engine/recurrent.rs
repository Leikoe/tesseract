use std::collections::HashMap;

use super::{RecurrentSlot, RequestId};

#[derive(Debug)]
pub(crate) struct RecurrentSlots {
    free: Vec<RecurrentSlot>,
    allocations: HashMap<RequestId, RecurrentSlot>,
}

impl RecurrentSlots {
    pub(crate) fn new(capacity: usize) -> Self {
        assert!(capacity <= u32::MAX as usize);
        Self {
            free: (0..capacity as u32).rev().map(RecurrentSlot::new).collect(),
            allocations: HashMap::new(),
        }
    }

    pub(crate) fn allocate(&mut self, request_id: RequestId) -> Option<RecurrentSlot> {
        if self.allocations.contains_key(&request_id) {
            return None;
        }
        let slot = self.free.pop()?;
        self.allocations.insert(request_id, slot);
        Some(slot)
    }

    pub(crate) fn get(&self, request_id: RequestId) -> Option<RecurrentSlot> {
        self.allocations.get(&request_id).copied()
    }

    pub(crate) fn release(&mut self, request_id: RequestId) {
        if let Some(slot) = self.allocations.remove(&request_id) {
            self.free.push(slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_are_exclusive_and_reusable_after_release() {
        let first = RequestId::now_v7();
        let second = RequestId::now_v7();
        let third = RequestId::now_v7();
        let mut slots = RecurrentSlots::new(2);
        let first_slot = slots.allocate(first).unwrap();
        let second_slot = slots.allocate(second).unwrap();
        assert_ne!(first_slot, second_slot);
        assert!(slots.allocate(third).is_none());
        slots.release(first);
        assert_eq!(slots.allocate(third), Some(first_slot));
    }
}
