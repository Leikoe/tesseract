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
Tesseract run started at this revision was cancelled because CUDA validation
probes from a second process contaminated the measurement. It is not reported
as a result.

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

## Tiled-attention and fused-MoE result

Revisions `771f509` through `dff5143` add a request-aware `BM=64`, `BN=64`
prefill-attention path, retain the `BM=1` path for pure decode, and fuse routed
NVFP4 gate/up grouped GEMM with SiLU multiplication. Both new kernels passed
isolated numerical differential tests on the A100 before benchmarking.

With the server and benchmark client as the only GPU and request processes:

| Prompt | TTFT | Input throughput | Previous TTFT | Improvement |
| --- | ---: | ---: | ---: | ---: |
| 8,202 tokens after templating | 13.26756 s | 618.14 tok/s | 28.680956 s | 2.16x |
| 131,072 tokens after templating | 237.88277 s | 550.99 tok/s | incomplete | n/a |

The raw reports are retained as
[`warm-8192.json`](../benchmarks/2026-08-09-qwen-serving/post-tiled/warm-8192.json)
and
[`half-window.json`](../benchmarks/2026-08-09-qwen-serving/post-tiled/half-window.json).

The exact half-window result is 14.88 times slower than SGLang's TTFT and
14.52 times lower in input-token throughput. Sustained GPU utilization was
99--100%, so the dominant failure mode is no longer host synchronization.

An Nsight Systems capture of one warmup and one measured 8K request ranks the
GPU work as follows:

| Kernel family | GPU time | Share |
| --- | ---: | ---: |
| FP8 W8A16 linear | 16.554 s | 62.2% |
| Grouped NVFP4 W4A16 MoE down | 4.106 s | 15.4% |
| Fused grouped NVFP4 gate/up/SILU | 3.088 s | 11.6% |
| Dense NVFP4 W4A16 linear | 1.566 s | 5.9% |
| GDN state output and convolution | 0.618 s | 2.3% |
| Full prefill attention | 0.252 s | 0.9% |

Quantized linear and MoE kernels therefore account for 95.1% of captured GPU
time; full attention is not the next bottleneck at this prompt size. The next
backend work should target A100 Marlin-class FP8 W8A16 and grouped NVFP4
W4A16 kernels, keeping their selection behind the linear and MoE abstractions.
The raw capture is
[`8k-profile.nsys-rep`](../benchmarks/2026-08-09-qwen-serving/post-tiled/8k-profile.nsys-rep)
and the extracted table is
[`8k-kernel-summary.csv`](../benchmarks/2026-08-09-qwen-serving/post-tiled/8k-kernel-summary.csv).

Revision `157c915` widened the dense FP8 and NVFP4 output tile from 16 to 64
columns and placed it on the same bounded persistent-worker scheme as grouped
MoE. Its isolated differential probe passed on the A100. Warm 8K TTFT improved
from 13.26756 to 13.14342 seconds (0.94%), or 623.98 input tokens/second. This
small result rules out output-tile width alone as the missing optimization:
the next backend needs Marlin-style load-time weight repacking and a staged
dequantize/MMA pipeline. The raw report is
[`dense64-warm-8192.json`](../benchmarks/2026-08-09-qwen-serving/post-tiled/dense64-warm-8192.json).

The built-in half-window command is:

```text
tesseract bench --base-url http://127.0.0.1:18100 \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --tokenizer /home/ubuntu/models/Qwen3.6-35B-A3B-NVFP4/tokenizer.json \
  --num-prompts 1 --warmup-requests 0 --input-len 131062 \
  --output-len 1 --length-variation 0 --max-concurrency 1 \
  --timeout-seconds 1800
```
