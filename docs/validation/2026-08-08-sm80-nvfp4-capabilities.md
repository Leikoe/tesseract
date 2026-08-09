# SM80 cuTile NVFP4 capability validation

This validation answers a narrower question than the Qwen forward-pass gate:
what the pinned cuTile/CUDA stack actually compiles and executes on the A100,
independent of GPU-name feature tables or upstream example skips.

Environment:

- NVIDIA A100-SXM4-80GB, `sm_80`;
- CUDA 13.3;
- cuTile revision `9fe5756f861bc40f098e6981ac2dff6cf5d3d0e4`; and
- Tesseract revision `68e44cd`.

`cuda-check` runs two permanent numerical capability probes without inspecting
the device name.

The first reproduces the cuTile NVFP4 tutorial boundary: packed
`f4e2m1fnx2` operands, `f8e4m3fn` group scales, FP32 accumulation, and
`mmaf_scaled`. On SM80, `tileiras` rejects this program before launch:

```text
target gpu: sm_80
error: Incompatibility with architecture 'sm_80': unsupported type 'f8E4M3FN'
nvfp4_scaled_mma=unavailable
```

Therefore the pinned toolchain does not silently emulate the tutorial's typed
scaled-MMA path on A100.

The second probe uses the checkpoint's physical representation directly:
packed FP4 and E4M3 scale bytes are both `u8`. A cuTile kernel separates the low
and high nibbles, maps every E2M1 value, decodes E4M3 including subnormals and
sign, applies one 16-element scale group, converts to BF16, and invokes the
ordinary SM80 tensor-core `mma`. Inputs encode +1 and every one of the 256 FP32
outputs must therefore equal 16. The measured result was:

```text
nvfp4_byte_decode_mma=available
nvfp4_byte_decode_mma_max_abs_error=0
cutile_validation=ok
```

This is positive execution evidence for keeping the SM80 W4A16 fallback in
cuTile.

The next permanent probes exercise the packed dense and persistent grouped
W4A16 leaves with all FP4 codes, signed BF16 activations, non-unit E4M3 group
scales, non-unit global scales, K-group accumulation, and reversed expert
routing. Both match the BF16-rounded host oracle exactly:

```text
nvfp4_w4a16_linear=available
nvfp4_w4a16_linear_max_abs_error=0
nvfp4_grouped_w4a16=available
nvfp4_grouped_w4a16_max_abs_error=0
```

This testing also exposed a cuTile-RS lowering detail on SM80: `Tensor<u8>`
pointer lanes currently arrive as signed `i8`, and `exti` preserves that
signedness. The kernel restores the unsigned byte in `i32` with `x < 0 ? x +
256 : x` before extracting nibbles. Without that step, high-bit packed bytes
silently decode incorrectly. The full A100 verifier, including strict CUDA
Clippy and the permanent numerical probes, passed at `68e44cd`.

These results prove arithmetic correctness for the current 16x16 leaves. They
do not prove production Qwen throughput; target-shape tiling, fused MoE stages,
routing, sanitizer runs, and end-to-end model benchmarks remain required.
