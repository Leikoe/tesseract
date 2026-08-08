# Qwen3.6 text checkpoint contract validation

Tesseract now recognizes `Qwen3_5MoeForConditionalGeneration` with outer model
type `qwen3_5_moe` as the text-only Qwen3.5/3.6 MoE architecture. Recognition
uses `config.json`; it does not depend on the deployment model ID.

The adapter validates before GPU allocation:

- the exact 40-layer `linear, linear, linear, full` hybrid schedule;
- full-attention, gated-delta/linear-attention, partial-RoPE, recurrent-state,
  MoE, special-token, and context geometry;
- the complete ModelOpt FP8 and NVFP4 target sets, rejecting missing,
  duplicated, or unexpected targets; and
- all 124,116 text and LM-head tensor roles, including dtype, stored shape,
  scalar scales, packed NVFP4 dimensions, and exact byte length.

The language manifest deliberately excludes 333 vision tensors and 19 MTP
tensors from the pinned checkpoint. Their presence is accepted in the source,
but they are not loaded into the text program. The CUDA construction methods
currently return an explicit unsupported-execution error until the hybrid
program and SM80 quantized kernels exist; architecture recognition alone is
not presented as runnable model support.

Tokenizer mechanics and incremental decoding now live in the shared private
`model::tokenizer` module. Llama retains its own template and warmup text but no
longer duplicates tokenizer loading or streaming reconstruction. The Llama
adapter is 491 lines. Qwen is separated into a 481-line frontend/config adapter,
a 303-line tensor contract module, and a 177-line test module.

Host verification on 2026-08-08:

```text
cargo fmt --check
cargo test --all-targets                 56 passed
cargo clippy --all-targets -- -D warnings
```

The actual public `config.json` from the model repository passed config and
quantization validation; a metadata-only `model-check` proceeded to opening the
first absent weight shard, proving that architecture/config validation—not a
fixture approximation—accepted the published config.
