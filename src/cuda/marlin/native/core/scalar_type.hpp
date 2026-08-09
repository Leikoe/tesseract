// Minimal framework-independent scalar type descriptors required by Marlin.
// The original vLLM descriptor also implements formatting and numeric limits;
// the CUDA templates only need stable type IDs, equality, and bit widths.
#pragma once

#include <cstdint>

namespace vllm {

using ScalarTypeId = std::int64_t;

class ScalarType {
 public:
  constexpr ScalarType(ScalarTypeId id, int bits) : id_(id), bits_(bits) {}

  constexpr ScalarTypeId id() const { return id_; }
  constexpr int size_bits() const { return bits_; }

  static constexpr ScalarType from_id(ScalarTypeId id) {
    return ScalarType(id, static_cast<int>(id & 0xff));
  }

  constexpr bool operator==(const ScalarType& other) const {
    return id_ == other.id_;
  }
  constexpr bool operator!=(const ScalarType& other) const {
    return !(*this == other);
  }

 private:
  ScalarTypeId id_;
  int bits_;
};

// Low eight ID bits encode the storage width so from_id remains constexpr.
static inline constexpr ScalarType kS4{0x0104, 4};
static inline constexpr ScalarType kU4{0x0204, 4};
static inline constexpr ScalarType kU4B8{0x0304, 4};
static inline constexpr ScalarType kS8{0x0408, 8};
static inline constexpr ScalarType kU8{0x0508, 8};
static inline constexpr ScalarType kU8B128{0x0608, 8};
static inline constexpr ScalarType kFE2M1f{0x0704, 4};
static inline constexpr ScalarType kFE4M3fn{0x0808, 8};
static inline constexpr ScalarType kFE8M0fnu{0x0908, 8};
static inline constexpr ScalarType kFloat16{0x0a10, 16};
static inline constexpr ScalarType kBFloat16{0x0b10, 16};

}  // namespace vllm
