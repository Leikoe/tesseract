use std::num::NonZeroUsize;

use cuda_core::Device;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ComputeCapability {
    pub major: u8,
    pub minor: u8,
}

impl ComputeCapability {
    pub fn detect(device_ordinal: usize) -> Result<Self, KernelSelectionError> {
        let raw = Device::raw_device(device_ordinal)
            .map_err(|error| KernelSelectionError::Device(format!("{error:?}")))?;
        let name = unsafe { cuda_core::get_device_sm_name(raw) }
            .map_err(|error| KernelSelectionError::Device(format!("{error:?}")))?;
        let digits = name
            .strip_prefix("sm_")
            .ok_or_else(|| KernelSelectionError::Device(format!("unexpected CUDA SM '{name}'")))?;
        let value = digits
            .parse::<u16>()
            .map_err(|_| KernelSelectionError::Device(format!("unexpected CUDA SM '{name}'")))?;
        Ok(Self {
            major: u8::try_from(value / 10).map_err(|_| {
                KernelSelectionError::Device(format!("unexpected CUDA SM '{name}'"))
            })?,
            minor: u8::try_from(value % 10).map_err(|_| {
                KernelSelectionError::Device(format!("unexpected CUDA SM '{name}'"))
            })?,
        })
    }

    const fn supports_bf16(self) -> bool {
        self.major >= 8
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DecoderKernelRequirement {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub attention_heads: usize,
    pub kv_heads: usize,
    pub vocab_size: usize,
}

/// Stable identity for one selected leaf implementation.
///
/// Names are diagnostic ABI: benchmarks and failures may persist them. A
/// semantic implementation change increments `revision` even when its name is
/// unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KernelDescriptor {
    pub name: &'static str,
    pub revision: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TiledKernelPlan {
    pub implementation: KernelDescriptor,
    pub block: NonZeroUsize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GemmKernelPlan {
    pub implementation: KernelDescriptor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DenseKernelPlan {
    pub embedding: TiledKernelPlan,
    pub gather_rows: TiledKernelPlan,
    pub rms_norm: TiledKernelPlan,
    pub add_rms_norm: TiledKernelPlan,
    pub silu_mul: TiledKernelPlan,
    pub gemm: GemmKernelPlan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RopeKernelPlan {
    pub query: KernelDescriptor,
    pub key_value_write: KernelDescriptor,
    pub head_dim: NonZeroUsize,
    pub rotary_dim: NonZeroUsize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RaggedAttentionKernelPlan {
    pub implementation: KernelDescriptor,
    pub key_block: NonZeroUsize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttentionKernelPlan {
    pub rope: RopeKernelPlan,
    /// Eager prefill and mixed execution currently share one masked ragged leaf.
    pub prefill_mixed: RaggedAttentionKernelPlan,
    /// Captured decode currently uses the same leaf, selected independently so
    /// a specialized implementation can replace it without changing the trait.
    pub decode: RaggedAttentionKernelPlan,
    pub head_dim: NonZeroUsize,
    pub kv_heads: NonZeroUsize,
    pub query_heads_per_kv: NonZeroUsize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SamplingKernelPlan {
    pub argmax: TiledKernelPlan,
    pub argmax_reduce: KernelDescriptor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ShapeDispatchPlan {
    pub query_minimum: NonZeroUsize,
    pub context_minimum: NonZeroUsize,
}

impl ShapeDispatchPlan {
    pub fn query_bucket(self, logical: usize) -> Option<usize> {
        execution_bucket(logical, self.query_minimum.get())
    }

    pub fn context_bucket(self, logical: usize) -> Option<usize> {
        execution_bucket(logical, self.context_minimum.get())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KernelPlan {
    pub compute_capability: ComputeCapability,
    pub catalog_revision: u32,
    pub dense: DenseKernelPlan,
    pub attention: AttentionKernelPlan,
    pub sampling: SamplingKernelPlan,
    pub shapes: ShapeDispatchPlan,
}

impl KernelPlan {
    pub fn diagnostic_summary(self) -> String {
        let descriptor = |kernel: KernelDescriptor| format!("{}@{}", kernel.name, kernel.revision);
        format!(
            "catalog@{} sm_{}{} [{}; {}; {}; {}; {}; {}; {}; {}; {}; {}; {}; {}]",
            self.catalog_revision,
            self.compute_capability.major,
            self.compute_capability.minor,
            descriptor(self.dense.embedding.implementation),
            descriptor(self.dense.gather_rows.implementation),
            descriptor(self.dense.rms_norm.implementation),
            descriptor(self.dense.add_rms_norm.implementation),
            descriptor(self.dense.silu_mul.implementation),
            descriptor(self.dense.gemm.implementation),
            descriptor(self.attention.rope.query),
            descriptor(self.attention.rope.key_value_write),
            descriptor(self.attention.prefill_mixed.implementation),
            descriptor(self.attention.decode.implementation),
            descriptor(self.sampling.argmax.implementation),
            descriptor(self.sampling.argmax_reduce),
        )
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct KernelCatalog;

impl KernelCatalog {
    pub fn resolve(
        self,
        capability: ComputeCapability,
        requirement: DecoderKernelRequirement,
    ) -> Result<KernelPlan, KernelSelectionError> {
        if !capability.supports_bf16() {
            return Err(KernelSelectionError::UnsupportedComputeCapability {
                major: capability.major,
                minor: capability.minor,
            });
        }
        validate_geometry(requirement)?;

        let hidden_block = nonzero(512);
        require_divisible(
            "cuTile BF16 embedding and normalization",
            requirement.hidden_size,
            hidden_block,
        )?;
        let mlp_block = nonzero(512);
        require_divisible(
            "cuTile BF16 SiLU-multiply",
            requirement.intermediate_size,
            mlp_block,
        )?;

        let ragged_attention = RaggedAttentionKernelPlan {
            implementation: descriptor("cutile.ragged_attention.bf16", 1),
            key_block: nonzero(16),
        };
        let head_dim = NonZeroUsize::new(requirement.head_dim).unwrap();

        Ok(KernelPlan {
            compute_capability: capability,
            catalog_revision: 1,
            dense: DenseKernelPlan {
                embedding: tiled("cutile.embedding.bf16", 1, hidden_block),
                gather_rows: tiled("cutile.gather_rows.bf16", 1, hidden_block),
                rms_norm: tiled("cutile.rms_norm.bf16", 1, hidden_block),
                add_rms_norm: tiled("cutile.add_rms_norm.bf16", 1, hidden_block),
                silu_mul: tiled("cutile.silu_mul.bf16", 1, mlp_block),
                gemm: GemmKernelPlan {
                    implementation: descriptor("cublas.gemm.bf16", 1),
                },
            },
            attention: AttentionKernelPlan {
                rope: RopeKernelPlan {
                    query: descriptor("cutile.rope_query.bf16", 1),
                    key_value_write: descriptor("cutile.rope_kv_write.bf16", 1),
                    head_dim,
                    rotary_dim: NonZeroUsize::new(requirement.head_dim / 2).unwrap(),
                },
                prefill_mixed: ragged_attention,
                decode: ragged_attention,
                head_dim,
                kv_heads: NonZeroUsize::new(requirement.kv_heads).unwrap(),
                query_heads_per_kv: NonZeroUsize::new(
                    requirement.attention_heads / requirement.kv_heads,
                )
                .unwrap(),
            },
            sampling: SamplingKernelPlan {
                argmax: tiled("cutile.argmax_stage1.f32", 1, nonzero(256)),
                argmax_reduce: descriptor("cutile.argmax_reduce.f32", 1),
            },
            shapes: ShapeDispatchPlan {
                query_minimum: nonzero(1),
                context_minimum: nonzero(16),
            },
        })
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum KernelSelectionError {
    #[error("failed to inspect CUDA device: {0}")]
    Device(String),
    #[error("BF16 kernels require compute capability 8.0 or newer, got {major}.{minor}")]
    UnsupportedComputeCapability { major: u8, minor: u8 },
    #[error("no compatible kernel plan: {0}")]
    UnsupportedGeometry(String),
}

fn validate_geometry(requirement: DecoderKernelRequirement) -> Result<(), KernelSelectionError> {
    let valid = requirement.hidden_size > 0
        && requirement.intermediate_size > 0
        && requirement.head_dim > 0
        && requirement.head_dim.is_multiple_of(2)
        && requirement.attention_heads > 0
        && requirement.kv_heads > 0
        && requirement.vocab_size > 0
        && requirement
            .attention_heads
            .is_multiple_of(requirement.kv_heads)
        && requirement.hidden_size == requirement.attention_heads * requirement.head_dim;
    if valid {
        Ok(())
    } else {
        Err(KernelSelectionError::UnsupportedGeometry(
            "invalid hidden, rotary, head, KV-head, intermediate, or vocabulary dimensions".into(),
        ))
    }
}

fn require_divisible(
    implementation: &str,
    dimension: usize,
    block: NonZeroUsize,
) -> Result<(), KernelSelectionError> {
    if dimension.is_multiple_of(block.get()) {
        Ok(())
    } else {
        Err(KernelSelectionError::UnsupportedGeometry(format!(
            "{implementation} requires dimension {dimension} to be divisible by block {}",
            block.get()
        )))
    }
}

const fn descriptor(name: &'static str, revision: u32) -> KernelDescriptor {
    KernelDescriptor { name, revision }
}

const fn tiled(name: &'static str, revision: u32, block: NonZeroUsize) -> TiledKernelPlan {
    TiledKernelPlan {
        implementation: descriptor(name, revision),
        block,
    }
}

const fn nonzero(value: usize) -> NonZeroUsize {
    match NonZeroUsize::new(value) {
        Some(value) => value,
        None => unreachable!(),
    }
}

fn execution_bucket(logical_size: usize, minimum: usize) -> Option<usize> {
    if logical_size == 0 || minimum == 0 {
        return None;
    }
    logical_size.max(minimum).checked_next_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn llama_requirement() -> DecoderKernelRequirement {
        DecoderKernelRequirement {
            hidden_size: 2048,
            intermediate_size: 8192,
            head_dim: 64,
            attention_heads: 32,
            kv_heads: 8,
            vocab_size: 128_256,
        }
    }

    #[test]
    fn resolves_the_complete_a100_bf16_plan() {
        let plan = KernelCatalog
            .resolve(
                ComputeCapability { major: 8, minor: 0 },
                llama_requirement(),
            )
            .unwrap();
        assert_eq!(plan.dense.embedding.block.get(), 512);
        assert_eq!(
            plan.dense.embedding.implementation.name,
            "cutile.embedding.bf16"
        );
        assert_eq!(plan.dense.silu_mul.block.get(), 512);
        assert_eq!(plan.attention.prefill_mixed.key_block.get(), 16);
        assert_eq!(plan.attention.decode.key_block.get(), 16);
        assert_eq!(plan.attention.query_heads_per_kv.get(), 4);
        assert_eq!(plan.sampling.argmax.block.get(), 256);
        assert_eq!(plan.shapes.query_bucket(3), Some(4));
        assert_eq!(plan.shapes.context_bucket(17), Some(32));
    }

    #[test]
    fn rejects_unsupported_device_and_operation_geometry_before_capture() {
        assert!(matches!(
            KernelCatalog.resolve(
                ComputeCapability { major: 7, minor: 5 },
                llama_requirement()
            ),
            Err(KernelSelectionError::UnsupportedComputeCapability { .. })
        ));
        let mut invalid = llama_requirement();
        invalid.intermediate_size = 8193;
        assert!(matches!(
            KernelCatalog.resolve(ComputeCapability { major: 8, minor: 0 }, invalid),
            Err(KernelSelectionError::UnsupportedGeometry(message))
                if message.contains("SiLU-multiply")
        ));
    }

    proptest! {
        #[test]
        fn shape_dispatch_is_total_and_bounded_for_valid_logical_sizes(
            logical in 1usize..=1_000_000,
        ) {
            let plan = KernelCatalog
                .resolve(
                    ComputeCapability { major: 8, minor: 0 },
                    llama_requirement(),
                )
                .unwrap();
            let query = plan.shapes.query_bucket(logical).unwrap();
            let context = plan.shapes.context_bucket(logical).unwrap();
            prop_assert!(query >= logical && query.is_power_of_two());
            prop_assert!(context >= logical && context >= 16 && context.is_power_of_two());
            prop_assert!(query < logical.saturating_mul(2).max(2));
            prop_assert!(context < logical.max(16).saturating_mul(2));
        }
    }
}
