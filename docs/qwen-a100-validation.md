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

## Decode performance investigation

Nsight Systems on commit `8142294` attributed 89.8% of GPU kernel time for a
32-token generation to quantized GEMMs: 39.6% to grouped routed-expert NVFP4,
34.0% to dense FP8 projections, and 16.2% to shared-expert NVFP4. GDN decode
accounted for 1.4%; routing, dispatch, attention, and expert combination were
each below 0.5%. The profile used 3,840 grouped-GEMM instances, exactly three
expert projections across 40 layers and 32 forwards.

The first decode fix in `8142294` separates logical token rows from the
16-row outer tensor-core padding. A one-token decode now routes one logical row
and reserves eight aligned expert tiles (128 dispatched rows), rather than
routing all 16 physical rows and reserving up to 2,048 dispatched rows. The
routing plan and expert map remain device-resident.

Commit `fefb38a` widens the persistent grouped NVFP4 output tile from 16 to 64
columns while retaining the checkpoint's native 16-element K scale group. The
A100 `cuda-check` differential probe reports zero grouped NVFP4 error after the
change. A rejected 32-wide K experiment demonstrated that adjacent scale groups
cannot be folded by reshaping the stored scale tensor; the gate caught the
result before checkpoint benchmarking.

All timings below use the same running server, raw HTTP request, 11-token Qwen
chat prompt for `Hi`, greedy sampling, and one running request. “Warm” excludes
cuTile compilation but includes prefill, decode, sampling, and HTTP handling.

| Revision | Change | 8 generated tokens | 32 generated tokens |
| --- | --- | ---: | ---: |
| `8142294` | logical-row routing, 16×16×16 grouped tile | 1.134 s | 3.981 s |
| `fefb38a` | logical-row routing, 16×64×16 grouped tile | 0.921 s | 3.421 s |
| `69e9827` | stream-ordered MoE pipeline, one sync per layer | 0.780 s | 2.866 s |

Both measured revisions returned the same eight-token continuation as SGLang:

```text
Here's a thinking process:

1
```

The wider grouped tile improves the first two warm samples by 18.8% and 14.1%,
respectively. Stream-ordering the complete MoE pipeline then improves them by a
further 15.3% and 16.2%. The final 32-token result is 2.24 ms per
layer-forward end to end, or about 11.2 generated tokens/s after amortizing its
one prefill forward.

The motivation for the stream-ordering change is direct profile evidence, not
an assumed launch-cost model. The `8142294` trace contained 81,255
`cuStreamSynchronize` calls, which occupied 52.6% of traced CUDA API time, plus
45,420 `cuMemAllocAsync` calls. Quantized projections and routing now enqueue
their allocation and kernel dependencies on the executor-owned stream; the
final shared-expert combine synchronizes once per layer, so an executor ticket
still cannot complete while device work is in flight. The next engine-level
step is a reusable workspace and explicit stream execution scope, followed by
decode graph capture. The remaining GPU-kernel target is the dense FP8 leaf.

## Recurrent-slot recycling regression

Commit `cc57c49` fixes a production-only correctness failure exposed by running
more requests than the recurrent-slot capacity. The scheduler recycled a slot,
but the Qwen executor retained the preceding request's convolution tail and GDN
matrix state. A fresh request could therefore depend on which request had most
recently occupied its slot, and serial and packed execution disagreed despite
greedy sampling.

The executor now identifies a request's first prefill chunk by its zero starting
position and clears that request's BF16 convolution tail and FP32 recurrent
matrix on the model stream before the first layer consumes them. Continuation
chunks retain their state. The stream execution scope owns the ordering and
drains pending work on errors, so slot initialization cannot race a layer or be
reported complete while still in flight.

The A100 regression sequence deliberately ran four prompts serially first, so
the slots contained unrelated prior state, and then submitted the same prompts
with concurrency four. All four eight-token greedy continuations were
byte-for-byte identical between serial and packed execution. Before the fix,
all four differed. The complete CUDA differential suite also passed at this
revision: grouped NVFP4 reported zero maximum absolute error, GDN recurrent
decode reported `0.0005493164`, and full attention reported zero.

## Stream-safe execution workspace

Commit `0e02078` introduces an executor-owned, shape-keyed workspace with an
explicit three-state lifetime: available, checked out by the current forward,
or retired while enqueued kernels may still reference the allocation. Retired
storage cannot be reclaimed until `StreamExecution` reports a synchronized
stream. A surviving host alias is an execution error rather than permission to
reuse the underlying pointer.

The first integration covers sampling and the allocation-heavy MoE routing
pipeline: routing IDs and weights, expert prefix metadata, dispatch positions,
zeroed dispatch padding, synchronization tickets, expert tile maps, and routed
combine outputs. Reused tensors that require zero initialization are cleared by
`cuMemsetD8Async` on the model stream, independent of tensor rank. The bounded
LRU pool evicts old shapes instead of retaining memory for every batch geometry
ever observed.

The full-checkpoint serial-reuse/concurrency-four regression remained
byte-for-byte identical for all four eight-token greedy continuations. Two
consecutive fully warm concurrency-four measurements produced 24.639 and
24.542 output tokens/s. The fully warm batch-one measurement produced 10.722
output tokens/s. Earlier passes encountered new compiled shapes and are not
reported as steady state. Raw reports are retained under
[`docs/benchmarks/2026-08-09-qwen-workspace`](benchmarks/2026-08-09-qwen-workspace).
