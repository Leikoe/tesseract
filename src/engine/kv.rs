use std::collections::HashMap;

use super::{KvSlot, RequestId};

#[derive(Debug)]
pub struct KvSlots {
    free: Vec<KvSlot>,
    allocations: HashMap<RequestId, Vec<KvSlot>>,
    reserved_remaining: HashMap<RequestId, usize>,
}

impl KvSlots {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity <= u32::MAX as usize);
        Self {
            free: (0..capacity as u32).rev().map(KvSlot::new).collect(),
            allocations: HashMap::new(),
            reserved_remaining: HashMap::new(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.free.len() + self.allocations.values().map(Vec::len).sum::<usize>()
    }

    pub fn used(&self) -> usize {
        self.allocations.values().map(Vec::len).sum()
    }

    pub fn available_to_reserve(&self) -> usize {
        self.free
            .len()
            .saturating_sub(self.reserved_remaining.values().sum())
    }

    pub fn reserve(&mut self, request_id: RequestId, tokens: usize) -> bool {
        if self.reserved_remaining.contains_key(&request_id)
            || self.allocations.contains_key(&request_id)
            || tokens > self.available_to_reserve()
        {
            return false;
        }
        self.reserved_remaining.insert(request_id, tokens);
        true
    }

    pub fn allocate(&mut self, request_id: RequestId, tokens: usize) -> Option<Vec<KvSlot>> {
        let remaining = self.reserved_remaining.get_mut(&request_id)?;
        if tokens > *remaining || tokens > self.free.len() {
            return None;
        }

        let split = self.free.len() - tokens;
        let slots = self.free.split_off(split);
        *remaining -= tokens;
        self.allocations
            .entry(request_id)
            .or_default()
            .extend_from_slice(&slots);
        Some(slots)
    }

    pub fn release(&mut self, request_id: RequestId) {
        self.reserved_remaining.remove(&request_id);
        if let Some(mut slots) = self.allocations.remove(&request_id) {
            self.free.append(&mut slots);
        }
    }

    #[cfg(test)]
    fn request_slots(&self, request_id: RequestId) -> &[KvSlot] {
        self.allocations
            .get(&request_id)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn reservations_prevent_overcommit_and_slots_never_alias() {
        let a = RequestId::now_v7();
        let b = RequestId::now_v7();
        let mut kv = KvSlots::new(8);

        assert!(kv.reserve(a, 5));
        assert!(!kv.reserve(b, 4));
        assert!(kv.reserve(b, 3));
        kv.allocate(a, 3).unwrap();
        kv.allocate(b, 2).unwrap();

        let a_slots: HashSet<_> = kv.request_slots(a).iter().copied().collect();
        let b_slots: HashSet<_> = kv.request_slots(b).iter().copied().collect();
        assert!(a_slots.is_disjoint(&b_slots));
        assert_eq!(kv.used(), 5);

        kv.release(a);
        assert_eq!(kv.used(), 2);
        assert!(kv.reserve(RequestId::now_v7(), 5));
    }

    #[test]
    fn allocation_must_stay_within_a_requests_reservation() {
        let id = RequestId::now_v7();
        let mut kv = KvSlots::new(4);
        assert!(kv.reserve(id, 2));
        assert!(kv.allocate(id, 3).is_none());
        assert_eq!(kv.used(), 0);
    }
}
