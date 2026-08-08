# Architecture and weight-loading validation

Revision: `a105369`.

Architecture selection and checkpoint transport are now independent startup
axes:

- the static architecture registry probes `config.json` for `model_type` and
  `architectures`, so serving code contains no repeated model-ID branches;
- the Llama factory owns its private config, frontend, tensor-name mapping, and
  construction of the shared dense decoder;
- `WeightSource` presents immutable dtype, shape, and byte views without
  exposing SafeTensors or CUDA types;
- `SafeTensorSource` handles monolithic and indexed/sharded SafeTensors; and
- the CUDA decoder, rather than the checkpoint source, owns BF16 conversion and
  device upload.

The registry and source are cold load-time boundaries. Once loading completes,
the executor retains the statically composed dense decoder and attention
implementation; no plugin lookup or new dynamic dispatch was added to token
execution.

Local validation passed 40 tests and strict ordinary Clippy. This includes two
128-case property tests over the format-neutral BF16 tensor contract: exact
shape/storage pairs are accepted and truncated storage is rejected.

The full `scripts/node/verify-a100.sh` verifier passed on the A100-SXM4-80GB:

- all 40 ordinary and CUDA-feature tests;
- strict ordinary and CUDA-feature Clippy;
- config, tokenizer, and 146-tensor checkpoint validation;
- 2,471,628,800 bytes uploaded and round-tripped through the independent source;
- cuTile BF16/GEMM smoke tests;
- the real Llama forward, which predicted token 12366, `" Paris"`; and
- the pinned upstream cuTile hello-world check.

This extraction changes startup organization only. Kernel plans, graph shapes,
attention launches, batching, and sampling are unchanged, so it is not treated
as a new throughput benchmark point.
