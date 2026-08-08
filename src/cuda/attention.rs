use std::sync::Arc;

use cuda_async::cuda_graph::Scope;
use cuda_core::Stream;
use cutile::{core::bf16, tensor::Tensor};

/// Stateful, statically composed attention implementation.
///
/// An implementation owns its physical per-layer state and both eager and
/// graph-recording paths. Model programs own Q/K/V projections and consume the
/// returned attention heads; cache layout and attention kernels stay here.
pub(crate) trait AttentionBackend: 'static {
    type LayerState;
    type Error;

    fn layer_state(&self, layer: usize) -> Result<&Self::LayerState, Self::Error>;

    fn enqueue_eager(&self, input: EagerAttention<'_>) -> Result<Tensor<bf16>, Self::Error>;

    fn record_decode(&self, input: DecodeGraphAttention<'_>) -> Result<(), Self::Error>;
}

pub(crate) struct EagerAttention<'a> {
    pub layer: usize,
    pub query: &'a Tensor<bf16>,
    pub key: &'a Tensor<bf16>,
    pub value: &'a Tensor<bf16>,
    pub positions: &'a Tensor<u32>,
    pub current_slots: &'a Tensor<u32>,
    pub request_indices: &'a Tensor<u32>,
    pub context_slots: &'a Tensor<u32>,
    pub context_lengths: &'a Tensor<i32>,
    pub rows: usize,
    pub stream: &'a Arc<Stream>,
}

pub(crate) struct DecodeGraphAttention<'a> {
    pub scope: &'a Scope,
    pub layer: usize,
    pub query: &'a Tensor<bf16>,
    pub key: &'a Tensor<bf16>,
    pub value: &'a Tensor<bf16>,
    pub positions: &'a Tensor<u32>,
    pub current_slots: &'a Tensor<u32>,
    pub request_indices: &'a Tensor<u32>,
    pub context_slots: &'a Tensor<u32>,
    pub context_lengths: &'a Tensor<i32>,
    pub rotated_query: &'a mut Tensor<bf16>,
    pub rotated_key: &'a mut Tensor<bf16>,
    pub attention: &'a mut Tensor<bf16>,
}
