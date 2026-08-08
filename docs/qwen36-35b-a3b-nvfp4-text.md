# Qwen3.6 35B-A3B NVFP4 text implementation contract

This document fixes the scope and load/runtime contracts for
`nvidia/Qwen3.6-35B-A3B-NVFP4`. The initial implementation is text-only. It
loads `model.language_model.*` and `lm_head.*`, does not construct the vision
encoder, and rejects image or video content at the API boundary.

The observations below are from NVIDIA's public checkpoint revision
`6c7f09d4036e97393f82e9f9ecd1a5c35ca5ee92`, its `config.json`,
`hf_quant_config.json`, tokenizer configuration, tensor index, and SafeTensors
headers. Runtime behavior must be tested against a pinned revision rather than
assuming that the mutable Hugging Face `main` branch has an unchanged schema.

## Fixed architecture

The outer architecture is `Qwen3_5MoeForConditionalGeneration`; the text
submodel is `qwen3_5_moe_text`:

- hidden size 2,048, vocabulary 248,320, and 40 decoder layers;
- 30 recurrent linear-attention layers and 10 full-attention layers in the
  repeating pattern `linear, linear, linear, full`;
- 16 query heads, 2 KV heads, head dimension 256, and partial rotary dimension
  64 for full attention;
- 256 routed experts, top-8 routing, expert intermediate size 512, plus a
  shared expert of size 512 in every layer;
- maximum configured sequence length 262,144;
- convolution width 4 and recurrent-state computation in FP32;
- untied token embedding and LM-head weights; and
- one MTP layer in the checkpoint, excluded from the first ordinary
  autoregressive program.

The checkpoint index contains 124,468 tensors: 124,112 under the language
model, 4 under the LM head, 333 vision tensors, and 19 MTP tensors. This tensor
count is itself a loader scalability requirement, not incidental metadata.

## Mixed quantization contract

`hf_quant_config.json` declares `MIXED_PRECISION`: 130 linear targets use FP8
and 161 target families use `W4A16_NVFP4` with group size 16. Tensor roles must
be validated together; a raw SafeTensors dtype is not enough to identify a
quantization scheme.

An NVFP4 projection has:

- packed E2M1 values stored as `U8`, with two logical weights per byte;
- `F8_E4M3` block scales for groups of 16 logical weights;
- an FP32 `weight_scale_2` global scale; and
- an FP32 `input_scale` in this static ModelOpt export.

For example, an expert gate/up projection with logical shape `[512, 2048]` is
stored as a `U8 [512, 1024]` tensor and an `F8_E4M3 [512, 128]` block-scale
tensor. The down projection has logical shape `[2048, 512]`, stored as
`U8 [2048, 256]` with `F8_E4M3 [2048, 32]` scales.

The linear-attention projections and full-attention Q/K/V/O projections use
`F8_E4M3` weights with scalar FP32 input and weight scales. Embeddings, norms,
router weights, recurrent parameters, and other explicitly excluded operators
remain BF16 (with recurrent accumulation/state governed separately by the FP32
configuration).

Tesseract therefore needs typed projection artifacts such as `Fp8Linear` and
`Nvfp4W4A16Linear`; it must not expose a generic "quantized tensor" whose
callers manually pair names and scales.

## A100 execution policy

A100 (SM80) has no native FP4 or FP8 Tensor Core path. Correct production
execution on this GPU is still possible without expanding the full checkpoint:

- NVFP4 weights use an SM80 weight-only W4A16 kernel, consuming BF16
  activations and applying the block/global scales during GEMM;
- FP8 weights use an SM80 weight-only W8A16 kernel; and
- recurrent state starts in the configured FP32, while full-attention KV may
  begin in BF16 and later gain an independently selected FP8 cache backend.

This is the same compatibility class that current vLLM calls its Marlin
fallback. It is not equivalent to native Blackwell W4A4 NVFP4 throughput, so
backend selection and benchmark reports must name the selected kernel path.
Loading must fail during construction if a projection has no compatible SM80
implementation; implicit whole-weight dequantization is not an acceptable
fallback.

## Required reusable boundaries

The model is the concrete acceptance test for the engine design:

1. `WeightSource` parses each SafeTensors shard header once and provides typed,
   range-checked views. Repeated header parsing is forbidden at this scale.
2. A model-owned config adapter validates the exact layer schedule and creates
   a model-neutral hybrid decoder artifact.
3. Full attention remains an `AttentionBackend`; gated delta/linear attention
   gets a distinct stateful operation-family backend rather than pretending to
   be KV attention.
4. `StateSchema` describes both recurrent state groups and full-attention KV
   groups. The scheduler allocates semantic request state without knowing their
   tensor layouts.
5. `MoeBackend` owns `route -> dispatch -> experts -> combine -> finalize`, its
   packed expert layout, workspace, and graph/capture capabilities. The model
   only supplies routing semantics and layer artifacts.
6. Quantized linear selection is construction-time, typed, and capability
   checked. It is a leaf-kernel plan used by attention, linear attention, MoE,
   and LM head—not runtime plugin dispatch.
7. Vision and MTP are independent program capabilities. Neither is loaded or
   silently ignored by a request that asks to use it.

## Implementation and validation order

1. scalable indexed checkpoint metadata and typed dtype/shape validation;
2. strict Qwen text config, tokenizer/chat frontend, and text-only tensor
   manifest validation;
3. grouped recurrent/KV state schemas and the hybrid layer program;
4. SM80 W8A16 linear kernels and recurrent linear-attention correctness;
5. SM80 W4A16 NVFP4 linear and MoE pipelines;
6. end-to-end next-token differential checks against the pinned checkpoint;
7. packed prefill/decode, graph legality, Compute Sanitizer, and retained A100
   throughput/latency benchmarks.

Every stage must preserve construction-time rejection, property tests for
shape/state invariants, and reference numerical tests. A server that merely
recognizes the architecture name is not model support.
