
namespace utils {

using TmaDescriptor = CUtensorMap;

template <class T>
inline CUtensorMapDataType to_CUtensorMapDataType() {
  if constexpr (std::is_same<T, int8_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if constexpr (std::is_same<T, uint8_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else
    //   if constexpr (std::is_same<T, float_e4m3_t>::value) { return
    //   CU_TENSOR_MAP_DATA_TYPE_UINT8;    } else if constexpr (std::is_same<T,
    //   float_e5m2_t>::value) { return CU_TENSOR_MAP_DATA_TYPE_UINT8;    } else
    if constexpr (std::is_same<T, uint16_t>::value) {
      return CU_TENSOR_MAP_DATA_TYPE_UINT16;
    } else if constexpr (std::is_same<T, uint32_t>::value) {
      return CU_TENSOR_MAP_DATA_TYPE_UINT32;
    } else if constexpr (std::is_same<T, uint64_t>::value) {
      return CU_TENSOR_MAP_DATA_TYPE_UINT64;
    } else if constexpr (std::is_same<T, int32_t>::value) {
      return CU_TENSOR_MAP_DATA_TYPE_INT32;
    } else if constexpr (std::is_same<T, int64_t>::value) {
      return CU_TENSOR_MAP_DATA_TYPE_INT64;
    } else if constexpr (std::is_same<T, half_t>::value) {
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    } else if constexpr (std::is_same<T, float>::value) {
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    } else if constexpr (std::is_same<T, double>::value) {
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
    } else
    //   if constexpr (std::is_same<T,   bfloat16_t>::value) { return
    //   CU_TENSOR_MAP_DATA_TYPE_BFLOAT16; } else if constexpr (std::is_same<T,
    //   tfloat32_t>::value) { return CU_TENSOR_MAP_DATA_TYPE_TFLOAT32; } else
    {
      static_assert(sizeof(T) < 0, "Unknown TMA Format!");
    }
}

enum class SmemSwizzleBits : uint8_t {
  DISABLE = 0,
  B32 = 1,
  B64 = 2,
  B128 = 3,
};

template <int B, int M, int S>
HOST_DEVICE constexpr SmemSwizzleBits get_tma_swizzle_bits(Swizzle<B, M, S>) {
  if constexpr (M == 4) {
    switch (B) {
      default:
        static_assert(0 <= B && B <= 3,
                      "Expected B = 0,1,2, or 3 when M == 4. Unsupported "
                      "layout swizzle.");
      case 3:
        return SmemSwizzleBits::B128;
      case 2:
        return SmemSwizzleBits::B64;
      case 1:
        return SmemSwizzleBits::B32;
      case 0:
        return SmemSwizzleBits::DISABLE;
    }
  } else {
    static_assert(M < 0, "Unsupported layout swizzle.");
  }
}

inline CUtensorMapSwizzle to_CUtensorMapSwizzle(SmemSwizzleBits const& t) {
  switch (t) {
    default:
      assert(false && "Unknown SmemSwizzleBits!");
    case SmemSwizzleBits::DISABLE:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    case SmemSwizzleBits::B32:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case SmemSwizzleBits::B64:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case SmemSwizzleBits::B128:
      return CU_TENSOR_MAP_SWIZZLE_128B;
  }
}

/// In this function, minor dimension moves faster than major dimension
template <int BlockMajorSize, int BlockMinorSize, int TmaDim, typename DType,
          int B, int M, int S>
TmaDescriptor make_tma_copy_desc(DType* gmem_ptr, int shape_major,
                                 int shape_minor,
                                 Swizzle<B, M, S> const& swizzle,
                                 uint32_t num_multicast) {
  void* gmem_address = (void*)gmem_ptr;
  uint64_t gmem_prob_shape[5] = {(uint64_t)shape_minor, (uint64_t)shape_major,
                                 1, 1, 1};
  uint64_t gmem_prob_stride[5] = {sizeof(DType), sizeof(DType) * shape_minor, 0,
                                  0, 0};

  assert((reinterpret_cast<uint64_t>(gmem_address) & 0b1111) == 0);
  assert(gmem_prob_shape[0] >= (uint64_t(1)));
  assert(gmem_prob_shape[0] <= (uint64_t(1) << 32));
  assert(gmem_prob_shape[1] >= (uint64_t(1)));
  assert(gmem_prob_shape[1] <= (uint64_t(1) << 32));
  assert(gmem_prob_shape[2] >= (uint64_t(1)));
  assert(gmem_prob_shape[2] <= (uint64_t(1) << 32));
  assert(gmem_prob_shape[3] >= (uint64_t(1)));
  assert(gmem_prob_shape[3] <= (uint64_t(1) << 32));
  assert(gmem_prob_shape[4] >= (uint64_t(1)));
  assert(gmem_prob_shape[4] <= (uint64_t(1) << 32));

  assert(gmem_prob_stride[0] == sizeof(DType));
  assert(gmem_prob_stride[1] < (uint64_t(1) << 40));
  assert((gmem_prob_stride[1] & 0b1111) == 0);
  assert(gmem_prob_stride[2] < (uint64_t(1) << 40));
  assert((gmem_prob_stride[2] & 0b1111) == 0);
  assert(gmem_prob_stride[3] < (uint64_t(1) << 40));
  assert((gmem_prob_stride[3] & 0b1111) == 0);
  assert(gmem_prob_stride[4] < (uint64_t(1) << 40));
  assert((gmem_prob_stride[4] & 0b1111) == 0);

  assert(BlockMajorSize % num_multicast == 0);
  uint32_t smem_box_shape[5] = {uint32_t(BlockMinorSize),
                                uint32_t(BlockMajorSize / num_multicast), 1, 1,
                                1};
  uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

  assert(smem_box_shape[0] >= (uint32_t(1)));  // Size must be min 1
  assert(smem_box_shape[0] <=
         (uint32_t(1) << 8));                  // Size must be max 2^8 = 256
  assert(smem_box_shape[1] >= (uint32_t(1)));  // Size must be min 1
  assert(smem_box_shape[1] <=
         (uint32_t(1) << 8));                  // Size must be max 2^8 = 256
  assert(smem_box_shape[2] >= (uint32_t(1)));  // Size must be min 1
  assert(smem_box_shape[2] <=
         (uint32_t(1) << 8));                  // Size must be max 2^8 = 256
  assert(smem_box_shape[3] >= (uint32_t(1)));  // Size must be min 1
  assert(smem_box_shape[3] <=
         (uint32_t(1) << 8));                  // Size must be max 2^8 = 256
  assert(smem_box_shape[4] >= (uint32_t(1)));  // Size must be min 1
  assert(smem_box_shape[4] <=
         (uint32_t(1) << 8));  // Size must be max 2^8 = 256

  assert(smem_box_stride[0] >= (uint32_t(1)));  // Stride must be min 1
  assert(smem_box_stride[0] <= (uint32_t(8)));  // Stride must be max 2^3 = 8
  assert(smem_box_stride[1] >= (uint32_t(1)));  // Stride must be min 1
  assert(smem_box_stride[1] <= (uint32_t(8)));  // Stride must be max 2^3 = 8
  assert(smem_box_stride[2] >= (uint32_t(1)));  // Stride must be min 1
  assert(smem_box_stride[2] <= (uint32_t(8)));  // Stride must be max 2^3 = 8
  assert(smem_box_stride[3] >= (uint32_t(1)));  // Stride must be min 1
  assert(smem_box_stride[3] <= (uint32_t(8)));  // Stride must be max 2^3 = 8
  assert(smem_box_stride[4] >= (uint32_t(1)));  // Stride must be min 1
  assert(smem_box_stride[4] <= (uint32_t(8)));  // Stride must be max 2^3 = 8

  TmaDescriptor tma_desc = {0};

  CUtensorMapDataType tma_format =
      to_CUtensorMapDataType<typename std::remove_cv<DType>::type>();
  CUtensorMapInterleave tma_interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
  CUtensorMapL2promotion tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
  CUtensorMapFloatOOBfill tma_oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  CUtensorMapSwizzle smem_swizzle =
      to_CUtensorMapSwizzle(get_tma_swizzle_bits(swizzle));
  CUresult result = cuTensorMapEncodeTiled(
      &tma_desc, tma_format, TmaDim, gmem_address, gmem_prob_shape,
      gmem_prob_stride + 1, smem_box_shape, smem_box_stride, tma_interleave,
      smem_swizzle, tma_l2Promotion, tma_oobFill);

  if (result != CUDA_SUCCESS) {
    std::cerr << "TMA Desc Addr:   " << &tma_desc << "\nformat         "
              << tma_format << "\ndim            " << TmaDim
              << "\ngmem_address   " << gmem_address << "\nglobalDim      "
              << gmem_prob_shape << "\nglobalStrides  " << gmem_prob_stride
              << "\nboxDim         " << smem_box_shape << "\nelementStrides "
              << smem_box_stride << "\ninterleave     " << tma_interleave
              << "\nswizzle        " << smem_swizzle << "\nl2Promotion    "
              << tma_l2Promotion << "\noobFill        " << tma_oobFill
              << std::endl;
    std::cerr << "Error: Failed to initialize the TMA descriptor " << result
              << std::endl;
    assert(false);
  }

  return tma_desc;
}

HOST_DEVICE
void prefetch_tma_descriptor(TmaDescriptor const* desc_ptr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  // Prefetch TMA Descriptor using generic addressing (i.e. no specific state
  // space: const or param)
  asm volatile("prefetch.tensormap [%0];" : : "l"(gmem_int_desc) : "memory");
}

DEVICE void fence_barrier_init() {
  asm volatile(
      "{\n\t"
      "fence.mbarrier_init.release.cluster; \n"
      "}" ::);
}

DEVICE void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" : :);
}

DEVICE void cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;\n" : :);
}

template <uint32_t RegCount>
DEVICE void warpgroup_reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount>
DEVICE void warpgroup_reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

// DEVICE void tma_copy_2d(TmaDescriptor const* const desc_ptr,
//                         uint64_t& smem_mbar, void const* const smem_ptr,
//                         int32_t const& crd0, int32_t const& crd1) {
//   uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
//   uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
//   uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
//   asm volatile(
//       "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
//       "bytes"
//       " [%0], [%1, {%3, %4}], [%2];"
//       :
//       : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0),
//         "r"(crd1)
//       : "memory");
// }

// DEVICE void tma_copy_2d_multicast(TmaDescriptor const* const desc_ptr,
//                                   uint64_t& smem_mbar, uint16_t multicast_mask,
//                                   void const* const smem_ptr,
//                                   int32_t const& crd0, int32_t const& crd1) {
//   uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
//   uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
//   uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
//   asm volatile(
//       "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
//       "bytes.multicast::cluster"
//       " [%0], [%1, {%4, %5}], [%2], %3;"
//       :
//       : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
//         "h"(multicast_mask), "r"(crd0), "r"(crd1)
//       : "memory");
// }

DEVICE void tma_copy_3d(TmaDescriptor const* const desc_ptr,
                        uint64_t& smem_mbar, void const* const smem_ptr,
                        int32_t const& crd0, int32_t const& crd1,
                        int32_t const& crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes"
      " [%0], [%1, {%3, %4, %5}], [%2];"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(crd0),
        "r"(crd1), "r"(crd2)
      : "memory");
}

DEVICE void tma_copy_3d_multicast(TmaDescriptor const* const desc_ptr,
                                  uint64_t& smem_mbar, uint16_t multicast_mask,
                                  void const* const smem_ptr,
                                  int32_t const& crd0, int32_t const& crd1,
                                  int32_t const& crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  uint32_t smem_int_mbar = cast_smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes.multicast::cluster"
      " [%0], [%1, {%4, %5, %6}], [%2], %3;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "h"(multicast_mask), "r"(crd0), "r"(crd1), "r"(crd2)
      : "memory");
}

template <typename T>
DEVICE void cp_async(void* ptr, const T* gmem_ptr) {
  uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
                   smem_ptr),
               "l"(gmem_ptr), "n"(16), "r"(16));
}

// GMMA 64x128x16 F32+=F16*F16
template <int ScaleA, int ScaleB, int ScaleD, int TransA, int TransB>
struct SM90_64x128x16_F32F16F16_SS {
  DEVICE static void wgmma(
      uint64_t const& desc_a, uint64_t const& desc_b, float& d00, float& d01,
      float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
      float& d08, float& d09, float& d10, float& d11, float& d12, float& d13,
      float& d14, float& d15, float& d16, float& d17, float& d18, float& d19,
      float& d20, float& d21, float& d22, float& d23, float& d24, float& d25,
      float& d26, float& d27, float& d28, float& d29, float& d30, float& d31,
      float& d32, float& d33, float& d34, float& d35, float& d36, float& d37,
      float& d38, float& d39, float& d40, float& d41, float& d42, float& d43,
      float& d44, float& d45, float& d46, float& d47, float& d48, float& d49,
      float& d50, float& d51, float& d52, float& d53, float& d54, float& d55,
      float& d56, float& d57, float& d58, float& d59, float& d60, float& d61,
      float& d62, float& d63) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " p,    %67,  %68,  %69,  %70;\n"
        "}\n"
        : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05),
          "+f"(d06), "+f"(d07), "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11),
          "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17),
          "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
          "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29),
          "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35),
          "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41),
          "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47),
          "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53),
          "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59),
          "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63)
        : "l"(desc_a), "l"(desc_b), "r"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
  }
};

DEVICE void warpgroup_fence_operand(float& reg) {
  asm volatile("" : "+f"(reg)::"memory");
}

DEVICE
void warpgroup_arrive() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

DEVICE
void warpgroup_commit_batch() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
DEVICE void warpgroup_wait() {
  static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

union GmmaDescriptor {
  HOST_DEVICE constexpr GmmaDescriptor() noexcept : desc_(0) {}
  HOST_DEVICE constexpr GmmaDescriptor(uint64_t desc) noexcept : desc_(desc) {}
  HOST_DEVICE constexpr GmmaDescriptor(GmmaDescriptor const& t) noexcept
      : desc_(t.desc_) {}
  HOST_DEVICE constexpr GmmaDescriptor(GmmaDescriptor&& t) noexcept
      : desc_(t.desc_) {}

  HOST_DEVICE constexpr GmmaDescriptor& operator=(
      GmmaDescriptor const& t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  HOST_DEVICE constexpr GmmaDescriptor& operator=(GmmaDescriptor&& t) noexcept {
    desc_ = t.desc_;
    return *this;
  }

  uint64_t desc_;
  uint32_t reg32_[2];
  uint16_t reg16_[4];

  // Bitfield implementation avoids the need for shifts in assignment
  struct {
    // start_address, bit [0,14), 4LSB not included
    uint16_t start_address_ : 14, : 2;  // 14 bits [0,14), 2 bits unused
    // leading dimension byte offset, bit [16,30), 4LSB not included
    // For N: This is the stride from the first col to the second col of the 8x2
    // brick in INTERLEAVED
    //   Unused for all SWIZZLE_* layouts (and assumed to be 1)
    // For T: This is the stride from the first 8 rows to the next 8 rows.
    uint16_t leading_byte_offset_ : 14, : 2;  // 14 bits [0,14), 2 bits unused
    // stride dimension byte offset, bit [32,46), 4LSB not included
    // For N: This is the stride from the first 8 rows to the next 8 rows.
    // For T: This is the stride fro mthe first 8 cols to the next 8 cols.
    uint16_t stride_byte_offset_ : 14, : 2;  // 14 bits [0,14), 2 bits unused
    // base_offset, bit [49,52)
    // Valid only for SWIZZLE_128B and SWIZZLE_64B
    uint8_t : 1,
        base_offset_ : 3, : 4;  // 1 bit unused, 3 bits [1,4), 4 bits unused
    // layout type, bit [62,64)
    // SWIZZLE_NONE = 0, SWIZZLE_32B = 3, SWIZZLE_64B = 2, SWIZZLE_128B = 1
    uint8_t : 6, layout_type_ : 2;  // 6 bits unused, 2 bits [6,8)
  } bitfield;

  // Decay to a uint64_t
  HOST_DEVICE constexpr operator uint64_t() const noexcept { return desc_; }
};

/// make shared memory descriptor
template <class PointerType>
DEVICE GmmaDescriptor make_smem_desc(PointerType smem_ptr) {
  GmmaDescriptor desc;
  uint32_t uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  desc.bitfield.start_address_ = uint_ptr >> 4;
  desc.bitfield.layout_type_ =
      0x1;  /// swizzle 128B because we use Swizzle<3,4,3>
  desc.bitfield.leading_byte_offset_ = 0x1;  /// no use
  desc.bitfield.stride_byte_offset_ =
      64;  /// how many 128bits-rows needed between two core matrices
  desc.bitfield.base_offset_ = 0x0;
  return desc;
}

}  // namespace utils

// Cluster-wide barrier. CUDA barrier doesn't support cluster scope. Have to
// follow CUTLASS. CUDA doesn't support barrier because cluster-wide barrier
// arrive can't return phase token. So CUTLASS doesn't use phase token as return
// value. But wait still need the phase token.
struct Barrier {
  uint64_t barrier_;
  DEVICE Barrier() = delete;

  DEVICE void init(uint32_t arrive_count) const {
    uint64_t const* smem_ptr = &barrier_;
    uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "{\n\t"
        "mbarrier.init.shared.b64 [%1], %0; \n"
        "}"
        :
        : "r"(arrive_count), "r"(smem_addr));
  }

  // local arrive
  DEVICE void arrive() const {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
    asm volatile(
        "{\n\t"
        "mbarrier.arrive.shared.b64 _, [%0];\n\t"
        "}"
        :
        : "r"(smem_addr));
  }

  // remote arrive
  DEVICE void arrive(uint32_t cta_id, uint32_t pred = true) const {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 remAddr32;\n\t"
        "setp.eq.u32 p, %2, 1;\n\t"
        "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
        "@p mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
        "}"
        :
        : "r"(smem_addr), "r"(cta_id), "r"(pred));
  }

  DEVICE void wait(uint32_t phase) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
    // Arbitrarily large timer value after which try-wait expires and re-tries.
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra.uni DONE; \n\t"
        "bra.uni     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}"
        :
        : "r"(smem_addr), "r"(phase), "r"(ticks));
  }

  DEVICE uint32_t try_wait(uint32_t phase) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
    uint32_t waitComplete;

    asm volatile(
        "{\n\t"
        ".reg .pred P1; \n\t"
        "mbarrier.try_wait.parity.shared.b64 P1, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P1; \n\t"
        "}"
        : "=r"(waitComplete)
        : "r"(smem_addr), "r"(phase));

    return waitComplete;
  }

  DEVICE void invalidate() {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
    asm volatile(
        "{\n\t"
        "mbarrier.ival.shared.b64 [%0]; \n\t"
        "}"
        :
        : "r"(smem_addr));
  }

  // These are TMA related barrier methods.
  // CULTASS implements it in another barrier.
  // We put them together.
  DEVICE void arrive_and_expect_tx(uint32_t transaction_bytes) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
    asm volatile(
        "{\n\t"
        "mbarrier.arrive.expect_tx.shared.b64 _, [%1], %0; \n\t"
        "}"
        :
        : "r"(transaction_bytes), "r"(smem_addr));
  }

  DEVICE void arrive_and_expect_tx(uint32_t transaction_bytes, uint32_t cta_id,
                                   uint32_t pred) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 remAddr32;\n\t"
        "setp.eq.u32 p, %2, 1;\n\t"
        "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
        "@p mbarrier.arrive.expect_tx.shared::cluster.b64  _, [remAddr32], "
        "%3;\n\t"
        "}"
        :
        : "r"(smem_addr), "r"(cta_id), "r"(pred), "r"(transaction_bytes));
  }

  DEVICE void expect_transaction(uint32_t transaction_bytes) const {
    uint32_t smem_addr = cast_smem_ptr_to_uint(&barrier_);
    asm volatile(
        "{\n\t"
        "mbarrier.expect_tx.shared.b64 [%1], %0; \n\t"
        "}"
        :
        : "r"(transaction_bytes), "r"(smem_addr));
  }
};

enum class BarrierStatus : uint32_t {
  WaitAgain = 0u,
  WaitDone = 1u,
  WaitOnly = 2u
};

struct ArrivalToken {
  HOST_DEVICE ArrivalToken(BarrierStatus barrier_status)
      : barrier_status(barrier_status) {}

  HOST_DEVICE ArrivalToken() = delete;

  HOST_DEVICE BarrierStatus get() const { return barrier_status; }

  HOST_DEVICE bool operator==(ArrivalToken const& other) const {
    return barrier_status == other.get();
  }

  HOST_DEVICE bool operator!=(ArrivalToken const& other) const {
    return !(*this == other);
  }

  BarrierStatus barrier_status;
};

struct ProducerToken : public ArrivalToken {};

struct ConsumerToken : public ArrivalToken {};