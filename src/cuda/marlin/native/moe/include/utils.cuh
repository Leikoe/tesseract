// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <stdexcept>

using fp32_t = float;
using fp16_t = __half;
using bf16_t = __nv_bfloat16;
using fp8_e4m3_t = __nv_fp8_e4m3;
using fp8_e5m2_t = __nv_fp8_e5m2;
using fp32x2_t = float2;
using fp16x2_t = __half2;
using bf16x2_t = __nv_bfloat162;
using fp32x4_t = float4;

namespace device {
template <typename T, typename U>
__host__ __device__ constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}
}  // namespace device

namespace host {
template <typename Condition, typename... Args>
inline void RuntimeCheck(Condition condition, Args&&...) {
  if (!static_cast<bool>(condition)) {
    throw std::runtime_error("invalid Marlin MoE launch contract");
  }
}

inline void RuntimeDeviceCheck(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

template <typename... Args>
[[noreturn]] inline void Panic(Args&&...) {
  throw std::runtime_error("unsupported Marlin MoE launch contract");
}
}  // namespace host
