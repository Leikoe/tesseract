//! Device-resident top-k routing and expert dispatch metadata.

use std::sync::Arc;

use cuda_async::device_operation::DeviceOp;
use cuda_core::Stream;
use cutile::{
    api,
    core::bf16,
    tensor::{PartitionMut, Reshape, Tensor, ToHostVec},
};

use crate::model::ModelError;

const TOP_K: usize = 8;
const EXPERTS: usize = 256;
const TILE_M: usize = 16;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    unsafe fn top8_router_256(
        logits: &Tensor<bf16, { [-1, 256] }>,
        ids: &mut Tensor<i32, { [1, 8] }>,
        weights: &mut Tensor<f32, { [1, 8] }>,
        counts_ptr: *mut i32,
    ) {
        const NEGATIVE_INFINITY: f32 = -1.0e30;
        const INVALID_EXPERT: i32 = 256;
        const ONE_I32: i32 = 1;
        let row = get_tile_block_id().0;
        let mut available: Tile<f32, { [1, 256] }> =
            convert_tile(logits.partition(const_shape![1, 256]).load([row, 0i32]));
        let expert: Tile<i32, { [256] }> = iota(const_shape![256]);
        let expert: Tile<i32, { [1, 256] }> = expert.reshape(const_shape![1, 256]);
        let negative_infinity: Tile<f32, { [1, 256] }> =
            broadcast_scalar(NEGATIVE_INFINITY, const_shape![1, 256]);
        let invalid_expert: Tile<i32, { [1, 256] }> =
            broadcast_scalar(INVALID_EXPERT, const_shape![1, 256]);
        let one: Tile<i32, { [1] }> = broadcast_scalar(ONE_I32, const_shape![1]);
        let counts: PointerTile<*mut i32, { [] }> = pointer_to_tile(counts_ptr);
        let counts: PointerTile<*mut i32, { [1] }> = counts.reshape(const_shape![1]);
        let mut ids = unsafe { ids.partition_mut(const_shape![1, 1]) };
        let mut weights = unsafe { weights.partition_mut(const_shape![1, 1]) };

        for rank in 0i32..8i32 {
            let maximum: Tile<f32, { [1] }> = reduce_max(available, 1i32);
            let maximum_broadcast = maximum
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![1, 256]);
            let candidates = select(
                eq_tile(available, maximum_broadcast),
                expert,
                invalid_expert,
            );
            let selected: Tile<i32, { [1] }> = reduce_min(candidates, 1i32);
            unsafe {
                ids.store(selected.reshape(const_shape![1, 1]), [0i32, rank]);
                weights.store(maximum.reshape(const_shape![1, 1]), [0i32, rank]);
            }
            let selected_ptr = counts.offset_tile(selected);
            let (_previous, _token): (Tile<i32, { [1] }>, Token) = atomic_rmw_tko(
                selected_ptr,
                one,
                atomic::Add,
                ordering::Relaxed,
                scope::Device,
                None,
                None,
            );
            let selected = selected
                .reshape(const_shape![1, 1])
                .broadcast(const_shape![1, 256]);
            available = select(eq_tile(expert, selected), negative_infinity, available);
        }
    }

    #[cutile::entry()]
    fn renormalize_top8(weights: &mut Tensor<f32, { [1, 8] }>) {
        let logits = load_tile_mut(weights);
        let maximum: Tile<f32, { [1] }> = reduce_max(logits, 1i32);
        let maximum = maximum
            .reshape(const_shape![1, 1])
            .broadcast(const_shape![1, 8]);
        let exponentials = exp(logits - maximum);
        let sum: Tile<f32, { [1] }> = reduce_sum(exponentials, 1i32);
        let denominator = sum
            .reshape(const_shape![1, 1])
            .broadcast(const_shape![1, 8]);
        weights.store(true_div(exponentials, denominator));
    }

    #[cutile::entry()]
    fn aligned_expert_prefix(
        counts: &Tensor<i32, { [256] }>,
        starts: &mut Tensor<i32, { [256] }>,
        cursors: &mut Tensor<i32, { [256] }>,
    ) {
        const ALIGNMENT_MINUS_ONE: i32 = 15;
        const ALIGNMENT: i32 = 16;
        const SCAN_IDENTITY: i32 = 0;
        let counts = counts.load_tile(const_shape![256], [0i32]);
        let alignment_minus_one: Tile<i32, { [256] }> =
            broadcast_scalar(ALIGNMENT_MINUS_ONE, const_shape![256]);
        let alignment: Tile<i32, { [256] }> = broadcast_scalar(ALIGNMENT, const_shape![256]);
        let padded = ((counts + alignment_minus_one) / alignment) * alignment;
        let inclusive = scan_sum(padded, 0i32, reverse::Forward, SCAN_IDENTITY);
        let exclusive = inclusive - padded;
        starts.store(exclusive);
        cursors.store(exclusive);
    }

    #[cutile::entry()]
    unsafe fn assign_dispatch_rows(
        ids: &Tensor<i32, { [-1, 8] }>,
        positions: &mut Tensor<i32, { [1, 1] }>,
        cursors_ptr: *mut i32,
    ) {
        const ONE_I32: i32 = 1;
        let pid = get_tile_block_id();
        let expert = ids.partition(const_shape![1, 1]).load([pid.0, pid.1]);
        let cursors: PointerTile<*mut i32, { [] }> = pointer_to_tile(cursors_ptr);
        let cursors: PointerTile<*mut i32, { [1, 1] }> = cursors.reshape(const_shape![1, 1]);
        let one: Tile<i32, { [1, 1] }> = broadcast_scalar(ONE_I32, const_shape![1, 1]);
        let expert_ptr: PointerTile<*mut i32, { [1, 1] }> = cursors.offset_tile(expert);
        let (position, _token): (Tile<i32, { [1, 1] }>, Token) = atomic_rmw_tko(
            expert_ptr,
            one,
            atomic::Add,
            ordering::Relaxed,
            scope::Device,
            None,
            None,
        );
        positions.store(position);
    }
}

use kernels::{aligned_expert_prefix, assign_dispatch_rows, renormalize_top8, top8_router_256};

pub(crate) struct RoutingPlan {
    pub(crate) expert_ids: Arc<Tensor<i32>>,
    pub(crate) weights: Arc<Tensor<f32>>,
    pub(crate) starts: Arc<Tensor<i32>>,
    pub(crate) ends: Arc<Tensor<i32>>,
    pub(crate) positions: Arc<Tensor<i32>>,
    pub(crate) max_dispatched_rows: usize,
}

impl RoutingPlan {
    pub(crate) fn build(
        logits: Arc<Tensor<bf16>>,
        rows: usize,
        stream: &Arc<Stream>,
    ) -> Result<Self, ModelError> {
        if rows == 0 || logits.shape() != [rows as i32, EXPERTS as i32] {
            return Err(ModelError::Cuda(format!(
                "router logits shape {:?}; expected [{rows}, {EXPERTS}]",
                logits.shape()
            )));
        }
        let active_experts = EXPERTS.min(rows.saturating_mul(TOP_K));
        let max_dispatched_rows = rows
            .checked_mul(TOP_K)
            .and_then(|replicas| replicas.checked_add(active_experts * (TILE_M - 1)))
            .ok_or_else(|| ModelError::Cuda("MoE dispatch capacity overflowed".into()))?;
        let counts = api::zeros::<i32>(&[EXPERTS])
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("allocate expert counts: {error:?}")))?;
        let mut expert_ids = api::zeros::<i32>(&[rows, TOP_K])
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("allocate expert IDs: {error:?}")))?;
        let mut weights = api::zeros::<f32>(&[rows, TOP_K])
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("allocate routing weights: {error:?}")))?;
        let (_, expert_ids_partition, weights_partition, _) = unsafe {
            top8_router_256(
                logits,
                (&mut expert_ids).partition([1, TOP_K]),
                (&mut weights).partition([1, TOP_K]),
                counts.device_pointer(),
            )
        }
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("execute top-8 routing: {error:?}")))?;
        drop(expert_ids_partition);
        drop(weights_partition);
        let (weights_partition,) = renormalize_top8((&mut weights).partition([1, TOP_K]))
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("renormalize top-8 routing: {error:?}")))?;
        drop(weights_partition);
        let expert_ids = Arc::new(expert_ids);

        let mut starts = api::zeros::<i32>(&[EXPERTS])
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("allocate expert starts: {error:?}")))?;
        let mut cursors = api::zeros::<i32>(&[EXPERTS])
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("allocate expert cursors: {error:?}")))?;
        let (_, starts_partition, cursors_partition) = aligned_expert_prefix(
            Arc::new(counts),
            (&mut starts).partition([EXPERTS]),
            (&mut cursors).partition([EXPERTS]),
        )
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("scan expert counts: {error:?}")))?;
        drop(starts_partition);
        drop(cursors_partition);
        let starts = Arc::new(starts);
        let mut positions = api::zeros::<i32>(&[rows, TOP_K])
            .sync_on(stream)
            .map_err(|error| ModelError::Cuda(format!("allocate dispatch positions: {error:?}")))?;
        let (_, positions_partition, _) = unsafe {
            assign_dispatch_rows(
                expert_ids.clone(),
                (&mut positions).partition([1, 1]),
                cursors.device_pointer(),
            )
        }
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("assign expert dispatch rows: {error:?}")))?;
        drop(positions_partition);

        Ok(Self {
            expert_ids,
            weights: Arc::new(weights),
            starts,
            ends: Arc::new(cursors),
            positions: Arc::new(positions),
            max_dispatched_rows,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RoutingProbe {
    pub(crate) max_weight_sum_error: f32,
}

pub(crate) fn probe(stream: &Arc<Stream>) -> Result<RoutingProbe, ModelError> {
    let rows = 3usize;
    let host = (0..rows * EXPERTS)
        .map(|index| {
            let row = index / EXPERTS;
            let expert = index % EXPERTS;
            bf16::from_f32((row * 13 + expert) as f32 / 32.0)
        })
        .collect::<Vec<_>>();
    let logits = api::copy_host_vec_to_device(&Arc::new(host))
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("upload router probe: {error:?}")))?
        .reshape(&[rows, EXPERTS])
        .map_err(|error| ModelError::Cuda(format!("reshape router probe: {error:?}")))?;
    let plan = RoutingPlan::build(Arc::new(logits), rows, stream)?;
    let ids: Vec<i32> = download(&plan.expert_ids, stream, "router IDs")?;
    let weights: Vec<f32> = download(&plan.weights, stream, "router weights")?;
    let positions: Vec<i32> = download(&plan.positions, stream, "dispatch positions")?;
    let starts: Vec<i32> = download(&plan.starts, stream, "expert starts")?;
    let ends: Vec<i32> = download(&plan.ends, stream, "expert ends")?;

    let mut max_weight_sum_error = 0.0f32;
    for row in 0..rows {
        let row_ids = &ids[row * TOP_K..(row + 1) * TOP_K];
        let expected = (EXPERTS - TOP_K..EXPERTS).rev().map(|id| id as i32);
        if !row_ids.iter().copied().eq(expected) {
            return Err(ModelError::Cuda(format!(
                "router selected {row_ids:?}; expected descending final eight experts"
            )));
        }
        let sum = weights[row * TOP_K..(row + 1) * TOP_K].iter().sum::<f32>();
        max_weight_sum_error = max_weight_sum_error.max((sum - 1.0).abs());
    }
    let mut unique_positions = std::collections::HashSet::new();
    for ((expert, position), assignment) in ids.iter().zip(&positions).zip(0..) {
        if !unique_positions.insert(*position)
            || *position < starts[*expert as usize]
            || *position >= ends[*expert as usize]
        {
            return Err(ModelError::Cuda(format!(
                "invalid dispatch assignment {assignment}: expert={expert}, position={position}"
            )));
        }
    }
    if max_weight_sum_error > 1.0e-5 {
        return Err(ModelError::Cuda(format!(
            "router weights do not sum to one: max error {max_weight_sum_error}"
        )));
    }
    Ok(RoutingProbe {
        max_weight_sum_error,
    })
}

fn download<T: cutile::DType + Copy>(
    tensor: &Arc<Tensor<T>>,
    stream: &Arc<Stream>,
    name: &str,
) -> Result<Vec<T>, ModelError> {
    tensor
        .to_host_vec()
        .sync_on(stream)
        .map_err(|error| ModelError::Cuda(format!("download {name}: {error:?}")))
}
