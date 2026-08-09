# A100 long-prefill performance record

This record preserves the first production comparison for
`nvidia/Qwen3.6-35B-A3B-NVFP4` on one A100 80 GB. Both servers used the same
local checkpoint, BF16 KV, one request, an 8,192-token scheduler prefill chunk,
and no CUDA graphs. TTFT is measured from streamed HTTP responses.

## SGLang reference

- SGLang: 0.5.17
- Backends selected by SGLang: FlashInfer full attention, Triton linear
  attention, Marlin FP4 GEMM and MoE
- Input/output: exactly 131,072 random token IDs and 8 generated tokens
- TTFT: 15.981896454 seconds
- End-to-end latency: 16.372333619 seconds
- Input throughput: 8,000.271949 tokens/second

The saved source artifact on the benchmark host is
`/tmp/tesseract-serious-new-node/sglang-native-half-window.jsonl`.

## Tesseract after removing layer barriers

The measured Tesseract revision is `dd734fe`. Revisions `7110525` through
`dd734fe` removed the GDN and MoE layer synchronizations, introduced
stream-ordered reusable scratch, and made activation ownership explicit.

- Input/output request: 8,192 random text tokens and 1 generated token
- Actual chat-template prompt: 8,202 tokens
- Cold TTFT: 37.00437 seconds
- Warm TTFT: 28.680956 seconds
- Warm input throughput: 285.960576 tokens/second

The warm source artifact is
`/tmp/tesseract-serious-new-node/tesseract-warm-8192.json`.

This is already conclusive: warm Tesseract takes 1.79 times SGLang's 131K
TTFT while processing only one sixteenth as much input. The exact 131K
Tesseract run was started separately and must be recorded below when it
finishes or fails.

## Structural causes established from local source

1. Tesseract full attention assigns one CUDA tile block to one query row and
   head (`BM=1`, `BN=16`). Each block serially traverses the whole visible
   context and independently gathers the same K/V rows. SGLang's SM80 Triton
   fallback uses `BM=64`, `BN=64`; its production run selected FlashInfer.
2. Tesseract executes separate QKV, A, B, and Z projections in every GDN
   layer. SGLang merges QKVZ and BA at load time.
3. Tesseract executes routed gate, routed up, activation, and routed down as
   separate operations and materializes both gate and up tensors. SGLang's
   Marlin MoE path uses merged gate/up weights and fused activation.
4. Tesseract uploads multiple metadata arrays per scheduler forward and its
   full-attention path still allocates and synchronizes intermediate tensors.
   SGLang lowers batch metadata once and uses persistent device buffers.

The implementation order is therefore: tiled request-aware prefill attention;
fused routed gate/up/activation; merged GDN projections; persistent async
full-attention buffers; then graph capture. Graph capture cannot repair the
current kernel geometry.

## Exact Tesseract 131K result

Pending. The built-in command is:

```text
tesseract bench --base-url http://127.0.0.1:18100 \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --tokenizer /home/ubuntu/models/Qwen3.6-35B-A3B-NVFP4/tokenizer.json \
  --num-prompts 1 --warmup-requests 0 --input-len 131072 \
  --output-len 1 --length-variation 0 --max-concurrency 1 \
  --timeout-seconds 1800
```
