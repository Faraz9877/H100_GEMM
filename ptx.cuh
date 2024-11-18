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