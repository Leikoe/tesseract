# Shared dense CUDA decoder validation

Revision: `9f3f2917704a0ba68c829fd5fd6dfd2565ebd8ce`.

The CUDA transformer program is now constructed from model-neutral
`DenseDecoderConfig`, `DenseDecoderWeightNames`, and `DenseDecoderArtifact`
values. The artifact has private fields and a fallible constructor validating
head geometry, layer mapping cardinality, vocabulary/BOS bounds, RMSNorm, and
RoPE invariants. CUDA no longer imports `Llama32`, the private Llama config, its
tokenizer, or any Llama tensor name.

`src/model/llama_3_2.rs` fell from 2,801 to 538 lines. It now owns configuration
and checkpoint validation, tokenizer/chat behavior, the Llama tensor-name map,
and construction of the shared decoder artifact. The generic execution body is
in `src/cuda/dense_decoder.rs`; further splits should separate its flat-KV
attention implementation and graph/workspace policy without returning those
concerns to a model module.

Validation on the A100 comprised:

- `cargo check --features cuda --all-targets` with no warnings;
- four focused dense-decoder tests, including two shrinking bucket-coverage
  properties and invalid-geometry rejection;
- the full `scripts/node/verify-a100.sh` gate: 36 ordinary tests, strict ordinary
  and CUDA Clippy, release model/device checks, cuTile smoke coverage, the actual
  forward pass, and the upstream cuTile example;
- the actual forward again predicted token 12366, `" Paris"`;
- two retained benchmark runs; the immediate repeat reached 1,598.17 first-shape
  and 1,590.03 warm-shape completion tokens/s.
