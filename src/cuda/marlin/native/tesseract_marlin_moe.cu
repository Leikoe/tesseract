// TokenSpeed's grouped Marlin device kernel, exposed through a framework-free
// C ABI for differential testing and eventual serving integration.

#include <cuda_runtime.h>

#include "moe/moe_wna16_marlin.cuh"

extern "C" int tesseract_marlin_moe_gemm_bf16(
    const void* activations,
    const void* packed_expert_weights,
    void* output,
    void* fp32_temporary,
    const void* block_scales,
    const void* expert_global_scales_bf16,
    int* workspace,
    const int32_t* expert_ids,
    int rows,
    int output_size,
    int input_size,
    int moe_block_size,
    void* raw_stream) {
  if (activations == nullptr || packed_expert_weights == nullptr ||
      output == nullptr || fp32_temporary == nullptr || block_scales == nullptr ||
      expert_global_scales_bf16 == nullptr || workspace == nullptr ||
      expert_ids == nullptr || rows <= 0 || output_size <= 0 ||
      input_size <= 0 || input_size % 16 != 0 || output_size % 64 != 0 ||
      (moe_block_size != 8 &&
       (moe_block_size < 16 || moe_block_size > 64 || moe_block_size % 16 != 0))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  int device = 0;
  int sms = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return static_cast<int>(status);
  status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) return static_cast<int>(status);

  try {
    device::marlin_moe::marlin_mm<__nv_bfloat16, false, false>(
        activations,
        packed_expert_weights,
        output,
        fp32_temporary,
        nullptr,
        const_cast<void*>(block_scales),
        const_cast<void*>(expert_global_scales_bf16),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        const_cast<int32_t*>(expert_ids),
        nullptr,
        nullptr,
        moe_block_size,
        1,
        false,
        false,
        rows,
        output_size,
        input_size,
        workspace,
        host::kFE2M1f,
        false,
        false,
        true,
        false,
        input_size / 16,
        16,
        device,
        static_cast<cudaStream_t>(raw_stream),
        -1,
        -1,
        sms,
        false,
        false,
        false);
  } catch (...) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaPeekAtLastError());
}
