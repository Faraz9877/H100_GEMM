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
