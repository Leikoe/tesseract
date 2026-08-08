# Scalable checkpoint metadata validation

The SafeTensors weight source now parses and validates each mapped shard header
once during construction. Tensor lookup uses the cached owned metadata and a
checked byte range into the shard mapping. Sharded index entries are checked
against the actual shard metadata before the source becomes usable.

This removes repeated multi-megabyte JSON-header parsing from every tensor
lookup. The change is required by the Qwen3.6 35B-A3B NVFP4 text checkpoint,
whose index declares 124,468 tensors, compared with 146 tensors in the initial
Llama checkpoint.

The format-neutral dtype contract now has first-class BF16, FP32, FP8 E4M3,
and U8 storage types. `WeightSource::validate_tensor` checks dtype, shape,
overflow, whole-byte representation, and exact byte length; BF16 validation is
a convenience specialization of the same contract. This supports validating
the checkpoint's BF16 tensors, FP8 projections/scales, packed NVFP4 bytes, and
scalar FP32 scales without embedding ModelOpt semantics in the transport.

Host verification on 2026-08-08:

```text
cargo fmt --check
cargo test --all-targets                 51 passed
cargo clippy --all-targets -- -D warnings
```

Regression tests cover indexed byte-range lookup, an index that maps a tensor
to a shard that does not contain it, exact BF16 storage, truncated storage, and
typed U8/F8_E4M3/F32 contracts including scalar shape.
