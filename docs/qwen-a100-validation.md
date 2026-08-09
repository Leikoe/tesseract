# Qwen3.6 NVFP4 A100 validation

This note records the checkpoint-level correctness comparison for
`nvidia/Qwen3.6-35B-A3B-NVFP4` on an NVIDIA A100-SXM4-80GB (SM80).

## Environment

- Tesseract commit: `c63234c`
- Reference runtime: SGLang `0.5.17`
- Reference quantization: ModelOpt mixed FP8/NVFP4
- SGLang dense FP4 backend: Marlin
- SGLang MoE backend: Marlin
- Full-attention KV cache: BF16 in both runtimes
- GDN recurrent state: FP32 in both runtimes
- CUDA toolkit: 13.3 for Tesseract; SGLang's CUDA 13 environment

SGLang must be launched with `--kv-cache-dtype bf16` on SM80. Its automatic
checkpoint-derived FP8 KV-cache selection reaches the Triton full-attention
kernel, which rejects `fp8e4nv` on this architecture. SGLang's ModelOpt loader
explicitly supports SM80 through `--fp4-gemm-backend marlin` and
`--moe-runner-backend marlin`.

## Reproduction

Tesseract exposes raw checkpoint logits through the existing CUDA forward
report command:

```bash
CUDA_PATH=/usr/local/cuda-13.3 \
  target/release/next-token-check \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --model-path /home/ubuntu/models/Qwen3.6-35B-A3B-NVFP4 \
  --prompt 'The capital of France is' \
  --json
```

The reference server was launched as follows. A longer watchdog is only needed
for SGLang's first Marlin JIT compilation; subsequent launches reuse its cache.

```bash
NVCC_PREPEND_FLAGS='-ccbin=/usr/bin/g++-11' \
  sglang serve \
  --model-path /home/ubuntu/models/Qwen3.6-35B-A3B-NVFP4 \
  --language-only \
  --quantization modelopt_fp4 \
  --fp4-gemm-backend marlin \
  --moe-runner-backend marlin \
  --attention-backend triton \
  --kv-cache-dtype bf16 \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --max-running-requests 1 \
  --max-total-tokens 1024 \
  --watchdog-timeout 1200
```

SGLang `/generate` requests used greedy sampling, one output token,
`return_logprob=true`, and `top_logprobs_num=20`. Log probabilities and logits
differ by the prompt's log-normalizer, so the stable comparison is token
identity, rank, and relative gaps rather than absolute values.

## Prefill results

| Raw prompt | Tokens | Tesseract next token | SGLang next token | Top-20 ID overlap |
| --- | ---: | --- | --- | ---: |
| `Hi` | 1 | `11` (`,`) | `11` (`,`) | 18/20 |
| `The capital of France is` | 5 | `11751` (` Paris`) | `11751` (` Paris`) | 19/20 |
| `Hello, my name is` | 5 | `3629` (` John`) | `3629` (` John`) | 19/20 |
| `This is a deliberately longer prompt that crosses the sixteen token GDN chunk boundary so the recurrent prefill state must continue correctly.` | 25 | `271` (`\n\n`) | `271` (`\n\n`) | 20/20 |

The 25-token case crosses the 16-token GDN chunk boundary and therefore covers
chunk-state carry in addition to the full model's embedding, mixed attention,
MoE, final normalization, and language-model head.

The distributions are close but intentionally not bit-identical: Tesseract's
cuTile W4A16 implementation and SGLang's Marlin path use different operation
ordering and rounding. For `Hi`, for example, the gap between tokens `11` and
`1017` is `0.5` in Tesseract and `0.4375` in SGLang.

## Prefill-to-decode result

Both servers received the same OpenAI chat request:

```json
{
  "messages": [{"role": "user", "content": "Hi"}],
  "temperature": 0,
  "max_tokens": 8
}
```

Both tokenized the prompt to 11 tokens and returned the exact same eight-token
continuation:

```text
Here's a thinking process:

1
```

This exercises GDN recurrent-state continuation and full-attention KV-cache
continuation after packed prefill, not only a standalone first-token forward.
