# Native Marlin source

The CUDA kernel core in this directory is extracted from vLLM commit
`f7ef489e93cf92b8d6ce7403b49f1db867bcc35e`, under Apache-2.0:

- `kernel.h`
- `marlin.cuh`
- `marlin_dtypes.cuh`
- `dequant.h`
- `marlin_mma.h`
- `marlin_template.h`

`tesseract_marlin.cu` uses the framework-independent repack kernel body from
TokenSpeed's vendored Marlin source and replaces its TVM-facing launcher with a
narrow C ABI owned by Tesseract. TokenSpeed is pinned under `references/` at
the revision recorded by the repository. Its repack body retains the included
MIT license notice in the upstream source history; the extracted body is used
with its original algorithm and attribution.

The local `core/scalar_type.hpp` is deliberately not a copy of vLLM's
Torch-dependent descriptor. It supplies only the constexpr identifiers and bit
widths consumed by the CUDA templates.

The production Rust wrapper owns shape validation, allocation, storage-format
repacking, stream ordering, numerical validation, and backend selection. No
Torch, Python, or TVM runtime is part of this backend.
