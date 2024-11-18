
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