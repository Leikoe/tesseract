# Execution-ticket validation

Validated on 2026-08-08 at Git revision
`dea5e06a8e4a32be8393f81f692b8c519df1cbff`.

## Change under test

The engine-to-device boundary is now `ModelExecutor::{submit,poll}`. Submission
returns a typed `BatchTicket`/`CompletionId`; polling returns a mode-tagged
`ExecutionOutput`. The initial CUDA executor remains synchronous internally but
uses an `ImmediateCompletion` slot that enforces exactly one outstanding ticket
and rejects stale or mismatched completion IDs.

The scheduler retains the submitted `ForwardBatch` until completion. A cancel
received while a request is in flight removes it from scheduling immediately,
but defers user-visible completion and KV release until the execution completion
has been observed. Shutdown also drains the current ticket before reclaiming
request state. This establishes the safety rule needed for later event-backed
asynchronous execution without claiming that GPU overlap exists today.

## Verification

- Local `cargo test`: 36 passed.
- Local `cargo clippy --all-targets -- -D warnings`: passed.
- Local formatting and whitespace checks: passed.
- Ticket tests reject a second in-flight submission, mismatched completion IDs,
  and polling an already-consumed ticket.
- A focused cancellation test proves KV remains allocated before ticket
  completion and is released afterward.
- Existing Proptest scheduler state machines pass through submit/poll on every
  generated engine step.
- A100 `scripts/node/verify-a100.sh`: passed on an NVIDIA A100-SXM4-80GB with
  CUDA 13.3 and Rust 1.89.0.
- The real BF16 Llama forward check still predicted token 12366 (`" Paris"`).

## Performance

The retained workload uses concurrency 8, 16 requests, at most 16 generated
tokens, and two benchmark warmup requests on the same A100 host.

| Revision/workload | Output tok/s | Mean TTFT | Mean inter-token |
| --- | ---: | ---: | ---: |
| Engine authority, first traffic | 1,654.93 | 12.37 ms | 3.35 ms |
| Execution tickets, first traffic | 1,622.77 | 12.44 ms | 3.43 ms |
| Engine authority, repeated | 1,623.51 | 13.98 ms | 3.34 ms |
| Execution tickets, repeated | 1,687.61 | 10.66 ms | 3.36 ms |

The ticket revision was 1.9% lower on the first pass and 3.9% higher on the
repeated pass. Mean inter-token latency differed by +0.08 ms and +0.02 ms.
Because the actual CUDA work is unchanged and the two repeated measurements
move in opposite directions, this supports no sustained performance regression;
it does not support attributing a speedup to the ticket protocol.

Compact summaries and the exact environment are retained in
[`../benchmarks/2026-08-08-execution-tickets/`](../benchmarks/2026-08-08-execution-tickets/README.md).
