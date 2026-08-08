# Dense decoder module-boundary validation

Revision: `0c6c0dc`.

The shared decoder is no longer another monolith hidden behind a smaller Llama
file. Its concrete flat-KV attention implementation is isolated in
`dense_decoder/flat_kv.rs`, and fixed-shape capture/storage/replay policy is in
`dense_decoder/graph.rs`. The remaining `dense_decoder.rs` owns construction,
eager dense-transformer execution, model-program integration, and shared CUDA
helpers.

Current source sizes are:

- `model/llama_3_2.rs`: 538 lines;
- `cuda/dense_decoder.rs`: 1,700 lines;
- `cuda/dense_decoder/flat_kv.rs`: 247 lines;
- `cuda/dense_decoder/graph.rs`: 555 lines.

Strict CUDA Clippy and the four focused dense-decoder tests passed after each
split. The final pushed layout then passed `scripts/node/verify-a100.sh` on the
A100, including the actual model forward, which again predicted token 12366,
`" Paris"`. The split is source-only: no kernel, launch, graph shape, batch, or
sampling behavior changed, so the preceding retained benchmark remains the
performance artifact for this extraction.
