// Repack kernel derived from TokenSpeed's vendored Marlin implementation.
// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include <algorithm>
#include <cstdint>

#include <cuda_runtime.h>

#include "kernel.h"
#include "marlin_template.h"

namespace marlin {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
template <int const num_threads, int const num_bits, bool const has_perm>
__global__ void gptq_marlin_repack_kernel(
    uint32_t const* __restrict__ b_q_weight_ptr,
    uint32_t const* __restrict__ perm_ptr,
    uint32_t* __restrict__ out_ptr,
    int size_k,
    int size_n) {
  return;
}
#else
template <int const num_threads, int const num_bits, bool const has_perm>
__global__ void gptq_marlin_repack_kernel(
    uint32_t const* __restrict__ b_q_weight_ptr,
    uint32_t const* __restrict__ perm_ptr,
    uint32_t* __restrict__ out_ptr,
    int size_k,
    int size_n) {
  constexpr int pack_factor = 32 / num_bits;

  int k_tiles = size_k / tile_k_size;
  int n_tiles = size_n / tile_n_size;
  int block_k_tiles = div_ceil(k_tiles, gridDim.x);

  auto start_k_tile = blockIdx.x * block_k_tiles;
  if (start_k_tile >= k_tiles) {
    return;
  }

  int finish_k_tile = min(start_k_tile + block_k_tiles, k_tiles);

  auto wait_for_stage = [&]() {
    cp_async_wait<repack_stages - 2>();
    __syncthreads();
  };

  extern __shared__ int4 sh[];

  constexpr int perm_size = tile_k_size / 4;

  int4* sh_perm_ptr = sh;
  int4* sh_pipe_ptr = sh_perm_ptr;
  if constexpr (has_perm) {
    sh_pipe_ptr += perm_size;
  }

  constexpr int tile_ints = tile_k_size / pack_factor;

  constexpr int stage_n_threads = tile_n_size / 4;
  constexpr int stage_k_threads = has_perm ? tile_k_size : tile_ints;
  constexpr int stage_size = stage_k_threads * stage_n_threads;

  auto load_perm_to_shared = [&](int k_tile_id) {
    int first_k_int4 = (k_tile_id * tile_k_size) / 4;

    int4 const* perm_int4_ptr = reinterpret_cast<int4 const*>(perm_ptr);

    if (threadIdx.x < perm_size) {
      sh_perm_ptr[threadIdx.x] = perm_int4_ptr[first_k_int4 + threadIdx.x];
    }
    __syncthreads();
  };

  auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      cp_async_fence();
      return;
    }

    int first_n = n_tile_id * tile_n_size;

    int4* sh_ptr = sh_pipe_ptr + stage_size * pipe;

    if constexpr (has_perm) {
      if (threadIdx.x < stage_size) {
        auto k_id = threadIdx.x / stage_n_threads;
        auto n_id = threadIdx.x % stage_n_threads;

        uint32_t const* sh_perm_int_ptr = reinterpret_cast<uint32_t const*>(sh_perm_ptr);

        int src_k = sh_perm_int_ptr[k_id];
        int src_k_packed = src_k / pack_factor;

        cp_async4(
            &sh_ptr[k_id * stage_n_threads + n_id],
            reinterpret_cast<int4 const*>(&(b_q_weight_ptr[src_k_packed * size_n + first_n + (n_id * 4)])));
      }

    } else {
      if (threadIdx.x < stage_size) {
        auto k_id = threadIdx.x / stage_n_threads;
        auto n_id = threadIdx.x % stage_n_threads;

        int first_k = k_tile_id * tile_k_size;
        int first_k_packed = first_k / pack_factor;

        cp_async4(
            &sh_ptr[k_id * stage_n_threads + n_id],
            reinterpret_cast<int4 const*>(&(b_q_weight_ptr[(first_k_packed + k_id) * size_n + first_n + (n_id * 4)])));
      }
    }

    cp_async_fence();
  };

  auto repack_tile = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      return;
    }

    auto warp_id = threadIdx.x / 32;
    auto th_id = threadIdx.x % 32;

    if (warp_id >= 4) {
      return;
    }

    int tc_col = th_id / 4;
    int tc_row = (th_id % 4) * 2;

    constexpr int tc_offsets[4] = {0, 1, 8, 9};

    int cur_n = warp_id * 16 + tc_col;

    constexpr int sh_stride = 64;
    constexpr uint32_t mask = (1 << num_bits) - 1;

    int4* sh_stage_ptr = sh_pipe_ptr + stage_size * pipe;
    uint32_t* sh_stage_int_ptr = reinterpret_cast<uint32_t*>(sh_stage_ptr);

    uint32_t* sh_perm_int_ptr = reinterpret_cast<uint32_t*>(sh_perm_ptr);

    uint32_t vals[8];

    if constexpr (has_perm) {
      for (int i = 0; i < 4; i++) {
        int k_idx = tc_row + tc_offsets[i];

        uint32_t src_k = sh_perm_int_ptr[k_idx];
        uint32_t src_k_pos = src_k % pack_factor;

        uint32_t b1_val = sh_stage_int_ptr[k_idx * sh_stride + cur_n];
        uint32_t b1_cur_val = (b1_val >> (src_k_pos * num_bits)) & mask;

        uint32_t b2_val = sh_stage_int_ptr[k_idx * sh_stride + cur_n + 8];
        uint32_t b2_cur_val = (b2_val >> (src_k_pos * num_bits)) & mask;

        vals[i] = b1_cur_val;
        vals[4 + i] = b2_cur_val;
      }

    } else {
      uint32_t b1_vals[tile_ints];
      uint32_t b2_vals[tile_ints];

#pragma unroll
      for (int i = 0; i < tile_ints; i++) {
        b1_vals[i] = sh_stage_int_ptr[cur_n + sh_stride * i];
        b2_vals[i] = sh_stage_int_ptr[cur_n + 8 + sh_stride * i];
      }

#pragma unroll
      for (int i = 0; i < 4; i++) {
        int cur_elem = tc_row + tc_offsets[i];
        int cur_int = cur_elem / pack_factor;
        int cur_pos = cur_elem % pack_factor;

        vals[i] = (b1_vals[cur_int] >> (cur_pos * num_bits)) & mask;
        vals[4 + i] = (b2_vals[cur_int] >> (cur_pos * num_bits)) & mask;
      }
    }

    constexpr int tile_size_words = tile_k_size * tile_n_size / pack_factor;
    int out_offset = (k_tile_id * n_tiles + n_tile_id) * tile_size_words;

    if constexpr (num_bits == 4) {
      constexpr int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};

      uint32_t res = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        res |= vals[pack_idx[i]] << (i * 4);
      }

      out_ptr[out_offset + th_id * 4 + warp_id] = res;

    } else {
      constexpr int pack_idx[4] = {0, 2, 1, 3};

      uint32_t res1 = 0;
      uint32_t res2 = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        res1 |= vals[pack_idx[i]] << (i * 8);
        res2 |= vals[4 + pack_idx[i]] << (i * 8);
      }

      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 0] = res1;
      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 1] = res2;
    }
  };

  auto start_pipes = [&](int k_tile_id, int n_tile_id) {
#pragma unroll
    for (int pipe = 0; pipe < repack_stages - 1; pipe++) {
      fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
    }

    wait_for_stage();
  };

#pragma unroll
  for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; k_tile_id++) {
    int n_tile_id = 0;

    if constexpr (has_perm) {
      load_perm_to_shared(k_tile_id);
    }

    start_pipes(k_tile_id, n_tile_id);

    while (n_tile_id < n_tiles) {
#pragma unroll
      for (int pipe = 0; pipe < repack_stages; pipe++) {
        fetch_to_shared((pipe + repack_stages - 1) % repack_stages, k_tile_id, n_tile_id + pipe + repack_stages - 1);
        repack_tile(pipe, k_tile_id, n_tile_id + pipe);
        wait_for_stage();
      }
      n_tile_id += repack_stages;
    }
  }

}
#endif

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS) {}

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

#define MATCH_CONFIG(B_TYPE, S_TYPE, GROUP_BLOCKS, TM, TK, TN, THREADS)        \
  if (b_type == B_TYPE && s_type == S_TYPE &&                                 \
      group_blocks == GROUP_BLOCKS && thread_m_blocks == TM &&                \
      thread_k == TK && thread_n == TN && threads == THREADS) {               \
    return Marlin<vllm::kBFloat16.id(), B_TYPE.id(),                          \
                  vllm::kBFloat16.id(), S_TYPE.id(), THREADS, TM, TN / 16,    \
                  TK / 16, false, 4, GROUP_BLOCKS, false>;                    \
  }

#define MATCH_TYPE_CONFIGS(B_TYPE, S_TYPE, GROUP_BLOCKS)                      \
  MATCH_CONFIG(B_TYPE, S_TYPE, GROUP_BLOCKS, 1, 128, 128, 256)                \
  MATCH_CONFIG(B_TYPE, S_TYPE, GROUP_BLOCKS, 1, 64, 128, 128)                 \
  MATCH_CONFIG(B_TYPE, S_TYPE, GROUP_BLOCKS, 1, 128, 64, 128)                 \
  MATCH_CONFIG(B_TYPE, S_TYPE, GROUP_BLOCKS, 2, 64, 256, 256)                 \
  MATCH_CONFIG(B_TYPE, S_TYPE, GROUP_BLOCKS, 2, 64, 128, 128)                 \
  MATCH_CONFIG(B_TYPE, S_TYPE, GROUP_BLOCKS, 2, 128, 64, 128)                 \
  MATCH_CONFIG(B_TYPE, S_TYPE, GROUP_BLOCKS, 3, 64, 256, 256)                 \
  MATCH_CONFIG(B_TYPE, S_TYPE, GROUP_BLOCKS, 3, 64, 128, 128)                 \
  MATCH_CONFIG(B_TYPE, S_TYPE, GROUP_BLOCKS, 3, 128, 64, 128)                 \
  MATCH_CONFIG(B_TYPE, S_TYPE, GROUP_BLOCKS, 4, 64, 256, 256)                 \
  MATCH_CONFIG(B_TYPE, S_TYPE, GROUP_BLOCKS, 4, 64, 128, 128)                 \
  MATCH_CONFIG(B_TYPE, S_TYPE, GROUP_BLOCKS, 4, 128, 64, 128)

MarlinFuncPtr select_kernel(vllm::ScalarType b_type, vllm::ScalarType s_type,
                            int group_blocks, int thread_m_blocks,
                            int thread_k, int thread_n, int threads) {
  MATCH_TYPE_CONFIGS(vllm::kFE4M3fn, vllm::kBFloat16, -1)
  MATCH_TYPE_CONFIGS(vllm::kFE2M1f, vllm::kFE4M3fn, 1)
  return MarlinDefault;
}

#undef MATCH_TYPE_CONFIGS
#undef MATCH_CONFIG

struct ThreadConfig {
  int k;
  int n;
  int threads;
};

int shared_memory_bytes(const ThreadConfig& config, int thread_m_blocks,
                        int num_bits, int group_size) {
  const int pack_factor = 32 / num_bits;
  const int tile_m = thread_m_blocks * 16;
  const int a = 4 * tile_m * config.k * 2;
  const int b = 4 * config.k * config.n / pack_factor * 4;
  const int reduction = tile_m * (config.n + 8) * 2;
  const int temporary = std::max(b, reduction);
  const int groups = group_size == -1 ? 1 : div_ceil(config.k, group_size);
  const int scales = groups * config.n * 2 * 4;
  return temporary + a + scales;
}

bool valid_config(const ThreadConfig& config, int thread_m_blocks, int k,
                  int n, int num_bits, int group_size, int max_shared_mem) {
  return config.k >= min_thread_k && config.n >= min_thread_n &&
         config.threads >= 128 && k % config.k == 0 && n % config.n == 0 &&
         shared_memory_bytes(config, thread_m_blocks, num_bits, group_size) <=
             max_shared_mem - 512;
}

ThreadConfig choose_config(int thread_m_blocks, int k, int n, int num_bits,
                           int group_size, int max_shared_mem, int sms,
                           int rows) {
  constexpr ThreadConfig small[] = {
      {128, 128, 256}, {64, 128, 128}, {128, 64, 128}};
  constexpr ThreadConfig large[] = {
      {64, 256, 256}, {64, 128, 128}, {128, 64, 128}};
  const ThreadConfig* choices = thread_m_blocks > 1 ? large : small;
  for (int index = 0; index < 3; ++index) {
    ThreadConfig selected = choices[index];
    if (!valid_config(selected, thread_m_blocks, k, n, num_bits, group_size,
                      max_shared_mem)) {
      continue;
    }
    if (n / selected.n * div_ceil(rows, thread_m_blocks * 16) * 4 <= sms) {
      ThreadConfig narrow{128, 64, 128};
      if (valid_config(narrow, thread_m_blocks, k, n, num_bits, group_size,
                       max_shared_mem)) {
        selected = narrow;
      }
    }
    return selected;
  }
  return {-1, -1, -1};
}

int launch_gemm(const void* a, const void* b, void* c, void* c_tmp,
                const void* scales, const float* global_scale,
                int* workspace, int m, int n, int k, int quant_bits,
                cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0 || (quant_bits != 4 && quant_bits != 8)) {
    return -1;
  }
  const auto b_type = quant_bits == 4 ? vllm::kFE2M1f : vllm::kFE4M3fn;
  const auto s_type = quant_bits == 4 ? vllm::kFE4M3fn : vllm::kBFloat16;
  const int group_size = quant_bits == 4 ? 16 : -1;
  const int group_blocks = quant_bits == 4 ? 1 : -1;
  const int num_groups = group_size == -1 ? 1 : k / group_size;

  int device = 0;
  int sms = 0;
  int max_shared_mem = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return static_cast<int>(status);
  status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) return static_cast<int>(status);
  status = cudaDeviceGetAttribute(&max_shared_mem,
                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                  device);
  if (status != cudaSuccess) return static_cast<int>(status);

  auto* a_ptr = static_cast<const int4*>(a);
  auto* c_ptr = static_cast<int4*>(c);
  const auto* b_ptr = static_cast<const int4*>(b);
  auto* c_tmp_ptr = static_cast<int4*>(c_tmp);
  const auto* scales_ptr = static_cast<const int4*>(scales);

  int remaining = m;
  constexpr int max_thread_m_blocks = 4;
  const int max_parallel = n <= 4096 ? 128 : 16;
  while (remaining > 0) {
    int parallel = remaining / (max_thread_m_blocks * 16);
    parallel = std::min(parallel, max_parallel);
    const int split_rows =
        parallel > 0 ? parallel * max_thread_m_blocks * 16 : remaining;
    const int thread_m_blocks =
        std::min(div_ceil(split_rows, 16), max_thread_m_blocks);
    const ThreadConfig config =
        choose_config(thread_m_blocks, k, n, quant_bits, group_size,
                      max_shared_mem, sms, split_rows);
    if (config.k < 0) return -2;
    MarlinFuncPtr kernel =
        select_kernel(b_type, s_type, group_blocks, thread_m_blocks, config.k,
                      config.n, config.threads);
    if (kernel == MarlinDefault) return -3;
    status = cudaFuncSetAttribute(kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  max_shared_mem);
    if (status != cudaSuccess) return static_cast<int>(status);
    kernel<<<sms, config.threads, max_shared_mem, stream>>>(
        a_ptr, b_ptr, c_ptr, c_tmp_ptr, nullptr, nullptr, scales_ptr,
        global_scale, nullptr, nullptr, num_groups, split_rows, n, k, k,
        workspace, false, false, true, max_shared_mem);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess) return static_cast<int>(status);
    a_ptr += split_rows * (k / 8);
    c_ptr += split_rows * (n / 8);
    remaining -= split_rows;
  }
  return 0;
}

}  // namespace marlin

extern "C" int tesseract_marlin_repack(const void* input, void* output,
                                        int size_k, int size_n, int num_bits,
                                        void* raw_stream) {
  if (input == nullptr || output == nullptr ||
      (num_bits != 4 && num_bits != 8) || size_k % marlin::tile_k_size != 0 ||
      size_n % marlin::tile_n_size != 0) {
    return -1;
  }
  int device = 0;
  int sms = 0;
  int max_shared_mem = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return static_cast<int>(status);
  status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) return static_cast<int>(status);
  status = cudaDeviceGetAttribute(&max_shared_mem,
                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                  device);
  if (status != cudaSuccess) return static_cast<int>(status);
  auto stream = static_cast<cudaStream_t>(raw_stream);
  if (num_bits == 4) {
    status = cudaFuncSetAttribute(
        marlin::gptq_marlin_repack_kernel<marlin::repack_threads, 4, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    if (status != cudaSuccess) return static_cast<int>(status);
    marlin::gptq_marlin_repack_kernel<marlin::repack_threads, 4, false>
        <<<sms, marlin::repack_threads, max_shared_mem, stream>>>(
            static_cast<const uint32_t*>(input), nullptr,
            static_cast<uint32_t*>(output), size_k, size_n);
  } else {
    status = cudaFuncSetAttribute(
        marlin::gptq_marlin_repack_kernel<marlin::repack_threads, 8, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    if (status != cudaSuccess) return static_cast<int>(status);
    marlin::gptq_marlin_repack_kernel<marlin::repack_threads, 8, false>
        <<<sms, marlin::repack_threads, max_shared_mem, stream>>>(
            static_cast<const uint32_t*>(input), nullptr,
            static_cast<uint32_t*>(output), size_k, size_n);
  }
  return static_cast<int>(cudaPeekAtLastError());
}

extern "C" int tesseract_marlin_gemm_bf16(
    const void* a, const void* b, void* c, void* c_tmp, const void* scales,
    const float* global_scale, int* workspace, int m, int n, int k,
    int quant_bits, void* raw_stream) {
  return marlin::launch_gemm(a, b, c, c_tmp, scales, global_scale, workspace,
                             m, n, k, quant_bits,
                             static_cast<cudaStream_t>(raw_stream));
}
