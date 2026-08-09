# SGLang quantized model loading

This trace uses the local SGLang checkout at
`55f02e68875a27eef0efedfc9d5845e066b184c0`. It records the loading boundary
relevant to `nvidia/Qwen3.6-35B-A3B-NVFP4`; it is not a proposal to copy
SGLang's runtime plugin machinery.

## Discovery

SGLang begins with the Hugging Face `config.json`. `ModelConfig` reads
`quantization_config` (or `compression_config`), with a legacy
`hf_quant_config.json` fallback. See
`python/sglang/srt/configs/model_config.py::_parse_quant_hf_config` and
`::_parse_modelopt_quant_config`.

For this checkpoint, the outer values `quant_method = modelopt` and
`quant_algo = MIXED_PRECISION` are not enough to select one kernel. SGLang
parses `quantized_layers`, detects its NVFP4 entries, and normalizes the model
method to `modelopt_mixed`. The resulting `ModelOptMixedPrecisionConfig` keeps
the per-prefix map and resolves each constructed layer independently to FP8,
MXFP8, NVFP4, W4A16-NVFP4, or unquantized. See
`layers/quantization/modelopt_quant.py::ModelOptMixedPrecisionConfig`.

## Construction and loading

The parsed quantization config is passed into the model constructor before
weights are read. Each logical linear or fused-MoE layer asks the config for a
method using its fully qualified prefix. That method allocates the physical
parameter representation. For W4A16-NVFP4 MoE this is:

- fused gate/up weight `w13_weight`: `[experts, 2 * intermediate, hidden / 2]`
  packed U8;
- down weight `w2_weight`: `[experts, hidden, intermediate / 2]` packed U8;
- matching E4M3 group-scale banks with group size 16; and
- per-expert gate/up and down global scales.

Only then does the generic checkpoint iterator stream `(name, tensor)` pairs.
Qwen's `load_weights` maps separate `experts.N.gate_proj`, `up_proj`, and
`down_proj` tensors into the fused banks using expert and shard ids. A final
backend-specific preparation step permutes layouts or scales once, outside the
token loop. Relevant sources are `model_loader/loader.py`,
`layers/linear.py`, `layers/moe/fused_moe_triton/layer.py`, and
`models/qwen3_5.py::Qwen3_5MoeForCausalLM.load_weights`.

## Tesseract boundary

Tesseract should preserve the same separation with smaller static machinery:

1. parse model and ModelOpt metadata into a typed, per-prefix quantization
   plan;
2. have the model builder choose concrete attention, linear, and MoE artifact
   types from that plan and device capabilities;
3. stream checkpoint tensors through explicit name mappings into those
   artifacts; and
4. perform any cuTile/backend layout transform once during construction.

The plan is data, not a runtime plugin. Kernel dispatch in the token path stays
monomorphic. Missing tensors, inconsistent expert geometry, unsupported packed
storage, and illegal scales are reported when an artifact consumes them; the
loader must not regenerate a 124,116-entry expected manifest and compare it to
the checkpoint.
