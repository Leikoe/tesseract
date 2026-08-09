# Qwen3.6 text checkpoint parsing

Tesseract recognizes `Qwen3_5MoeForConditionalGeneration` with outer model
type `qwen3_5_moe` as the text-only Qwen3.5/3.6 MoE architecture. Recognition
comes from `config.json`, not the deployment model ID.

The adapter parses the nested text configuration, including the 40-layer
`linear, linear, linear, full` hybrid schedule, attention and recurrent-state
geometry, MoE dimensions, tokenizer ids, and context limits. Semantic checks
only reject configurations the current implementation cannot interpret
safely.

Quantization is discovered from the checkpoint's own
`quantization_config`: `quant_method = modelopt` and `quant_algo =
MIXED_PRECISION`. Its authoritative `quantized_layers` map parses to 291
entries: 130 FP8 projections and 161 W4A16-NVFP4 projections with group size
16. Tesseract does not reconstruct the producer's target list or a complete
expected tensor manifest.

`SafeTensorSource` maps the index and parses each shard header once. It checks
that index entries refer to real byte ranges, but model artifacts consume and
interpret only the tensors they need. Vision and MTP tensors can coexist with
the text program without appearing in a hand-maintained exclusion manifest.

On the A100 node, the actual checkpoint parses successfully:

```text
model_id=nvidia/Qwen3.6-35B-A3B-NVFP4
architecture=Qwen3_5MoeForConditionalGeneration
dtype=bfloat16
layers=40
hidden_size=2048
attention_heads=16
kv_heads=2
vocab_size=248320
tensors=124468
model_load=ok
```

The SM80 cuTile probes separately establish exact host-oracle agreement for
packed dense and persistent grouped W4A16 leaves. They do not yet establish a
complete Qwen forward pass or production throughput. See
`2026-08-08-sm80-nvfp4-capabilities.md` and
`../sglang-quantized-loading.md`.
