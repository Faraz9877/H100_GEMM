#include "common.h"
#include "reference.h"

const int testM = 4096;
const int testN = 4096;
const int testK = 4096;
const int iters = 200;
static constexpr int WG_NUMBER = 3;

#include "header.cuh"

template <int Stages>
struct PipelineState {
  int index = 0;
  uint32_t phase = 0;
  uint32_t count = 0;

  DEVICE PipelineState() : index(), phase(), count() {}

  DEVICE PipelineState(int index, uint32_t phase, uint32_t count)
      : index(index), phase(phase), count(count) {}

  DEVICE void operator++() {
    if constexpr (Stages > 0) {
      ++index;
      ++count;
      if (index == Stages) {
        index = 0;
        phase ^= 1;
      }
    }
  }

  DEVICE PipelineState advance(uint32_t num_iterations) {
    if constexpr (Stages > 0) {
      if ((num_iterations < Stages) && (index + num_iterations) >= Stages) {
        phase ^= 1;
      }
      if ((num_iterations >= Stages) &&
          (((index + num_iterations) / Stages) % 2) == 1) {
        phase ^= 1;
      }
      index = (index + num_iterations) % Stages;
      count += num_iterations;
    }
    return *this;
  }

  DEVICE static PipelineState make_pipeline_state(PipelineState start_state,
                                                  uint32_t num_iterations) {
    return start_state.advance(num_iterations);
  }
};

template <int Depth, int Length>
struct OrderedBarrierSharedStorage {
  Barrier barrier[Depth][Length];
};

template <int Depth, int Length>
struct OrderedBarrierParams {
  uint32_t group_id;
  uint32_t group_size;
};

template <int Depth, int Length>
struct OrderedBarrier {
  OrderedBarrierParams<Depth, Length> params;
  Barrier* barrier_ptr;
  PipelineState<Depth> state;

  DEVICE OrderedBarrier() = delete;

  DEVICE OrderedBarrier(OrderedBarrierSharedStorage<Depth, Length>& storage,
                        OrderedBarrierParams<Depth, Length>& params)
      : params(params),
        barrier_ptr(&storage.barrier[0][0]),
        state({0, params.group_id == 0, 0}) {
    int warp_idx = threadIdx.x / WARP_SIZE;
    int lane_predicate = elect_one_sync();
    if (warp_idx == 0 && lane_predicate == 1) {
      for (int i = 0; i < Depth; ++i) {
        for (int j = 0; j < Length; ++j) {
          barrier_ptr[i * Length + j].init(params.group_size);
        }
      }
    }
    utils::fence_barrier_init();
  }

  DEVICE void wait() {
    get_barrier_for_current_stage(params.group_id).wait(state.phase);
  }

  // This will the next slot's barrier. Gurantee -> order.
  DEVICE void arrive() {
    int signaling_id = (params.group_id + 1) % Length;
    get_barrier_for_current_stage(signaling_id).arrive();
    ++state;
  }

  DEVICE void advance() { ++state; }

  DEVICE Barrier& get_barrier_for_current_stage(int group_id) {
    return barrier_ptr[state.index * Length + group_id];
  }
};

template <int Stages>
DEVICE PipelineState<Stages> make_producer_start_state() {
  // start from the next phase, so that the barrier wait doesn't block
  // execution.
  return {0, 1, 0};
}

template <int Stages>
struct TmaPipelineSharedStorage {
  Barrier full_barrier[Stages];
  Barrier empty_barrier[Stages];
};

template <int Stages>
struct TmaPipelineParams {
  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };

  uint32_t transaction_bytes = 0;
  ThreadCategory role = ThreadCategory::NonParticipant;
  uint32_t is_leader = 0;
  uint32_t num_consumers = 0;
};

// TmaPipeline structure. Follow CUTLASS impl.
template <int Stages, int ClusterM, int ClusterN>
struct TmaPipeline {
  uint32_t dst_blockid = 0;
  uint32_t is_signaling_thread = 0;
  Barrier* full_barrier_ptr = nullptr;
  Barrier* empty_barrier_ptr = nullptr;
  TmaPipelineParams<Stages> params;

  DEVICE TmaPipeline(TmaPipelineSharedStorage<Stages>& storage,
                     TmaPipelineParams<Stages> p)
      : full_barrier_ptr(&storage.full_barrier[0]),
        empty_barrier_ptr(&storage.empty_barrier[0]),
        params(p) {
    int warp_idx = threadIdx.x / WARP_SIZE;
    int lane_predicate = elect_one_sync();

    if (warp_idx == 0 && lane_predicate == 1) {
      for (int i = 0; i < Stages; ++i) {
        full_barrier_ptr[i].init(1);
      }
      // Question: why num_consumers = WARP_GROUP_SIZE?
      uint32_t const num_consumer_warpgroups_per_cluster =
          params.num_consumers / WARP_GROUP_SIZE;
      // Question: why this? I guess it's the same row and col.
      uint32_t const multicast_consumer_arrival_count =
          (ClusterM + ClusterN - 1) * num_consumer_warpgroups_per_cluster;
      for (int i = 0; i < Stages; ++i) {
        empty_barrier_ptr[i].init(multicast_consumer_arrival_count);
      }
    }
    utils::fence_barrier_init();

    // CUTLASS says the following logic is used to equally spread the duty of
    // SYNCS Empty Arriveal to 128 threads.
    dim3 block_id = block_id_in_cluster();
    static constexpr uint32_t cluster_size = ClusterM * ClusterN;
    static_assert(cluster_size <= MAX_CLUSTER_SIZE, "Cluster size too large!");
    if (params.num_consumers % WARP_GROUP_SIZE == 0) {
      int thread_idx = threadIdx.x % WARP_GROUP_SIZE;
      is_signaling_thread =
          (thread_idx % (WARP_GROUP_SIZE / MAX_CLUSTER_SIZE)) == 0;
      uint32_t thread_row = warp_idx % 4;
      uint32_t thread_col = (thread_idx / 8) % 4;
      auto swizzle = Swizzle<2, 0, -2>{};
      dst_blockid = swizzle(thread_row * 4 + thread_col);
    } else if (params.num_consumers == 32) {
      int thread_idx = threadIdx.x % 32;
      is_signaling_thread = (thread_idx % (32 / MAX_CLUSTER_SIZE)) == 0;
      uint32_t thread_row = thread_idx / 8;
      uint32_t thread_col = (thread_idx % 8) / 2;
      dst_blockid = thread_row * 4 + thread_col;
    } else {
      is_signaling_thread = 0;
      // Should not arrive there.
      assert(false);
    }

    is_signaling_thread &= dst_blockid < cluster_size;
    is_signaling_thread &= is_same_row_or_col(dst_blockid, block_id);
  }

  DEVICE bool is_same_row_or_col(int dst_block_id, dim3 block_id) {
    return ((dst_block_id % ClusterM) == block_id.x) ||
           ((dst_block_id / ClusterM) == block_id.y);
  }

  DEVICE void producer_acquire(PipelineState<Stages> state,
                               ProducerToken barrier_token = {
                                   BarrierStatus::WaitAgain}) {
    if (barrier_token != BarrierStatus::WaitDone) {
      empty_barrier_ptr[state.index].wait(state.phase);
    }
    if (barrier_token == BarrierStatus::WaitOnly) {
      return;
    }

    if (params.is_leader) {
      full_barrier_ptr[state.index].arrive_and_expect_tx(
          params.transaction_bytes);
    }
  }

  DEVICE void producer_tail(PipelineState<Stages> state) {
    for (int i = 0; i < Stages; ++i) {
      producer_acquire(state, {BarrierStatus::WaitOnly});
      ++state;
    }
  }

  DEVICE uint64_t* producer_get_barrier(PipelineState<Stages> state) {
    return reinterpret_cast<uint64_t*>(&full_barrier_ptr[state.index]);
  }

  DEVICE ConsumerToken consumer_try_wait(PipelineState<Stages> state,
                                         uint32_t skip_wait = false) {
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    uint32_t barrier_status =
        full_barrier_ptr[state.index].try_wait(state.phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  DEVICE void consumer_wait(PipelineState<Stages> state) {
    full_barrier_ptr[state.index].wait(state.phase);
  }

  DEVICE void consumer_wait(PipelineState<Stages> state,
                            ConsumerToken barrier_token) {
    if (barrier_token == BarrierStatus::WaitAgain) {
      full_barrier_ptr[state.index].wait(state.phase);
    }
  }

  DEVICE void consumer_release(PipelineState<Stages> state,
                               uint32_t skip = false) {
    empty_barrier_ptr[state.index].arrive(dst_blockid,
                                          is_signaling_thread & (!skip));
  }
};

template <int Stages>
struct EpilogueLoadPipelineSharedStorage {
  Barrier full_barrier[Stages];
  Barrier empty_barrier[Stages];
};

template <int Stages>
struct EpilgoueLoadPipelineParams {
  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };

  ThreadCategory role = ThreadCategory::NonParticipant;
  uint32_t transaction_bytes = 0;
  uint32_t producer_arv_count = 1;
  uint32_t consumer_arv_count = 1;
  uint32_t dst_blockid = block_rank_in_cluster();
};

template <typename AType, typename BType, typename AccumType, int MTile,
          int NTile, int KTile>
struct WgMMA;

template <int MTile, int NTile, int KTile>
struct WgMMA<half_t, half_t, float, MTile, NTile, KTile> {
  static constexpr int elements_per_thread = 2;
  static constexpr int threads_per_row = 4;
  static constexpr int threads_per_col = WARP_SIZE / threads_per_row;
  static constexpr int warp_elements_per_row =
      elements_per_thread * threads_per_row;
  static constexpr int warp_repeats_per_row = NTile / warp_elements_per_row;
  static constexpr int warp_repeats_per_col = 2;
  static constexpr int num_warps = WARP_NUMBER_IN_WARP_GROUP;
  static constexpr int WGMMA_M = 64;
  static constexpr int WGMMA_K = 16;
  static constexpr int num_warp_groups_m = MTile / WGMMA_M;
  static constexpr int k_iter = KTile / WGMMA_K;

  static constexpr int num_elements_accumulators =
      num_warp_groups_m * warp_repeats_per_row * warp_repeats_per_col *
      elements_per_thread;

  DEVICE WgMMA() {}

  DEVICE static void get_m_n_idx_fragment(int& m, int& n, int thread_id,
                                          int k_wgmma, int row_id, int col_id,
                                          int item_id) {
    int warp_id = thread_id / WARP_SIZE;
    int lane_id = thread_id % WARP_SIZE;
    m = k_wgmma * WGMMA_M + warp_id * threads_per_col * warp_repeats_per_col +
        row_id * threads_per_col + lane_id / threads_per_row;
    n = col_id * warp_elements_per_row +
        lane_id % threads_per_row * elements_per_thread + item_id;
  }

  DEVICE static void get_4d_idx_from_linear(int& k_wgmma, int& row_id,
                                            int& col_id, int& item_id,
                                            int linear_id) {
    item_id = linear_id % elements_per_thread;
    row_id = linear_id / elements_per_thread % warp_repeats_per_col;
    col_id = linear_id / elements_per_thread / warp_repeats_per_col %
             warp_repeats_per_row;
    k_wgmma = linear_id / (num_elements_accumulators / num_warp_groups_m);
  }

  template <int ScaleA, int ScaleB, int ScaleD, int TransA, int TransB>
  DEVICE static void wgmma(half_t* smem_A, half_t* smem_B, float* accum) {
    float* accum_ = accum;
    {
      int k = 0;
      accum = accum_;
      auto desc_b = utils::make_smem_desc(smem_B + k * WGMMA_K);
      for (int m = 0; m < num_warp_groups_m; ++m) {
        accum = accum + m * (num_elements_accumulators / num_warp_groups_m);
        auto desc_a =
            utils::make_smem_desc(smem_A + k * WGMMA_K + m * WGMMA_M * KTile);
        // the first is ScaleD
        utils::SM90_64x128x16_F32F16F16_SS<
            ScaleA, ScaleB, ScaleD, TransA,
            TransB>::wgmma(desc_a, desc_b, accum[0], accum[1], accum[2],
                           accum[3], accum[4], accum[5], accum[6], accum[7],
                           accum[8], accum[9], accum[10], accum[11], accum[12],
                           accum[13], accum[14], accum[15], accum[16],
                           accum[17], accum[18], accum[19], accum[20],
                           accum[21], accum[22], accum[23], accum[24],
                           accum[25], accum[26], accum[27], accum[28],
                           accum[29], accum[30], accum[31], accum[32],
                           accum[33], accum[34], accum[35], accum[36],
                           accum[37], accum[38], accum[39], accum[40],
                           accum[41], accum[42], accum[43], accum[44],
                           accum[45], accum[46], accum[47], accum[48],
                           accum[49], accum[50], accum[51], accum[52],
                           accum[53], accum[54], accum[55], accum[56],
                           accum[57], accum[58], accum[59], accum[60],
                           accum[61], accum[62], accum[63]);
      }
    }
    for (int k = 1; k < k_iter; ++k) {
      accum = accum_;
      auto desc_b = utils::make_smem_desc(smem_B + k * WGMMA_K);
      for (int m = 0; m < num_warp_groups_m; ++m) {
        auto desc_a =
            utils::make_smem_desc(smem_A + k * WGMMA_K + m * WGMMA_M * KTile);
        // the remaining must be ScaleD = 1
        utils::SM90_64x128x16_F32F16F16_SS<ScaleA, ScaleB, 1, TransA, TransB>::
            wgmma(desc_a, desc_b, accum[0], accum[1], accum[2], accum[3],
                  accum[4], accum[5], accum[6], accum[7], accum[8], accum[9],
                  accum[10], accum[11], accum[12], accum[13], accum[14],
                  accum[15], accum[16], accum[17], accum[18], accum[19],
                  accum[20], accum[21], accum[22], accum[23], accum[24],
                  accum[25], accum[26], accum[27], accum[28], accum[29],
                  accum[30], accum[31], accum[32], accum[33], accum[34],
                  accum[35], accum[36], accum[37], accum[38], accum[39],
                  accum[40], accum[41], accum[42], accum[43], accum[44],
                  accum[45], accum[46], accum[47], accum[48], accum[49],
                  accum[50], accum[51], accum[52], accum[53], accum[54],
                  accum[55], accum[56], accum[57], accum[58], accum[59],
                  accum[60], accum[61], accum[62], accum[63]);
        accum = accum + (num_elements_accumulators / num_warp_groups_m);
      }
    }
  }
};

// A simplified tile scheduler that always takes AlongN and non swizzle
template <int BlockM, int BlockN, int ClusterM, int ClusterN>
struct TileScheduler {
  int linear_idx;
  int m_blocks;
  int n_blocks;

  struct WorkInfo {
    int m_idx;
    int n_idx;
    bool valid;
  };

  DEVICE TileScheduler(int M, int N) { init(M, N); }

  DEVICE void init(int M, int N) {
    linear_idx = blockIdx.x + blockIdx.y * gridDim.x;
    get_blocks_m_n(M, N);
  }

  DEVICE WorkInfo get_current_work_info() {
    int m_idx, n_idx;
    get_current_m_n_idx(m_idx, n_idx, m_blocks, n_blocks);
    return {m_idx, n_idx, linear_idx < m_blocks * n_blocks};
  }

  DEVICE void advance(int number = 1) {
    linear_idx += number * gridDim.x * gridDim.y;
  }

  DEVICE void get_current_m_n_idx(int& m_idx, int& n_idx, int m_blocks,
                                  int n_blocks) {
    int div_cluster_x = linear_idx / ClusterM;
    int mod_cluster_x = linear_idx % ClusterM;
    int div_cluster_xy = div_cluster_x / ClusterN;
    int mod_cluster_xy = div_cluster_x % ClusterN;
    int clusters_per_row = n_blocks / ClusterN;
    int cluster_row = div_cluster_xy / clusters_per_row;
    int cluster_col = div_cluster_xy % clusters_per_row;
    m_idx = cluster_row * ClusterM + mod_cluster_x;
    n_idx = cluster_col * ClusterN + mod_cluster_xy;
  }

  DEVICE void get_blocks_m_n(int M, int N) {
    m_blocks = ((M + BlockM - 1) / BlockM + ClusterM - 1) / ClusterM * ClusterM;
    n_blocks = ((N + BlockN - 1) / BlockN + ClusterN - 1) / ClusterN * ClusterN;
  }
};

template <class AType, class BType, class CType, class AccumType, int BlockM,
          int BlockN, int BlockK, int ClusterM, int ClusterN, int Stages>
struct MainloopSharedStorage {
  alignas(128) AType smem_A[BlockM * BlockK * Stages];
  alignas(128) BType smem_B[BlockN * BlockK * Stages];
  TmaPipelineSharedStorage<Stages> pipeline;
};

template <class AType, class BType, class CType, class AccumType, int BlockM,
          int BlockN, int BlockK, int ClusterM, int ClusterN, int Stages>
struct MainloopParams {};

template <class AType, class BType, class CType, class AccumType, int BlockM,
          int BlockN, int BlockK, int ClusterM, int ClusterN, int Stages>
struct Mainloop {
  static_assert(std::is_same<AType, BType>::value);
  static constexpr uint32_t TmaTransactionBytes =
      BlockM * BlockK * sizeof(AType) + BlockN * BlockK * sizeof(BType);

  DEVICE static void prefetch_tma_descriptor(
      const utils::TmaDescriptor* tensormap_a,
      const utils::TmaDescriptor* tensormap_b) {
    utils::prefetch_tma_descriptor(tensormap_a);
    utils::prefetch_tma_descriptor(tensormap_b);
  }

  DEVICE void load(const utils::TmaDescriptor& tensormap_a,
                   const utils::TmaDescriptor& tensormap_b,
                   TmaPipeline<Stages, ClusterM, ClusterN> mainloop_pipeline,
                   PipelineState<Stages> mainloop_pipeline_state, int m_idx,
                   int n_idx, int k_tile_count, uint32_t block_rank_in_cluster,
                   MainloopSharedStorage<AType, BType, CType, AccumType, BlockM,
                                         BlockN, BlockK, ClusterM, ClusterN,
                                         Stages>& shared_storage) {
    int warp_idx = threadIdx.x / WARP_SIZE;
    int warp_idx_in_warp_group = warp_idx % 4;
    int lane_predicate = elect_one_sync();

    if (warp_idx_in_warp_group == 0 && lane_predicate == 1) {
      int block_id_x_in_cluster = block_rank_in_cluster % ClusterM;
      int block_id_y_in_cluster = block_rank_in_cluster / ClusterM;
      uint16_t mcast_mask_a = 0;
      uint16_t mcast_mask_b = 0;
      constexpr int multicast_stride_a = BlockM / ClusterN;
      constexpr int multicast_stride_b = BlockN / ClusterM;
      if constexpr (ClusterM > 1) {
        // multicast B
        for (int i = 0; i < ClusterM; ++i) {
          mcast_mask_b |=
              (uint16_t(1) << (block_id_y_in_cluster * ClusterM + i));
        }
      }
      if constexpr (ClusterN > 1) {
        // multicast A
        for (int i = 0; i < ClusterN; ++i) {
          mcast_mask_a |=
              (uint16_t(1) << (block_id_x_in_cluster + i * ClusterN));
        }
      }

      // mcast_mask_b, multicast_stride_b, block_id_x_in_cluster);

      for (int i = 0; i < k_tile_count; ++i) {
        // mainloop_pipeline_state.index, mainloop_pipeline_state.phase);
        mainloop_pipeline.producer_acquire(mainloop_pipeline_state);

        int stage = mainloop_pipeline_state.index;
        AType* smem_ptr_A = (shared_storage.smem_A + stage * BlockM * BlockK);
        BType* smem_ptr_B = (shared_storage.smem_B + stage * BlockN * BlockK);

        // load A and B using the same barrier
        if constexpr (ClusterN > 1) {
          // multicast copy A
          utils::tma_copy_3d_multicast(
              &tensormap_a,
              *mainloop_pipeline.producer_get_barrier(mainloop_pipeline_state),
              mcast_mask_a,
              reinterpret_cast<void*>(smem_ptr_A + block_id_y_in_cluster *
                                                       multicast_stride_a *
                                                       BlockK),
              i * BlockK,  // innermost dim moves fastest
              m_idx * BlockM + block_id_y_in_cluster * multicast_stride_a, 0);
        } else {
          // normal copy A
          utils::tma_copy_3d(
              &tensormap_a,
              *mainloop_pipeline.producer_get_barrier(mainloop_pipeline_state),
              smem_ptr_A, i * BlockK, m_idx * BlockM, 0);
        }

        if constexpr (ClusterM > 1) {
          // multicast copy B
          utils::tma_copy_3d_multicast(
              &tensormap_b,
              *mainloop_pipeline.producer_get_barrier(mainloop_pipeline_state),
              mcast_mask_b,
              reinterpret_cast<void*>(smem_ptr_B + block_id_x_in_cluster *
                                                       multicast_stride_b *
                                                       BlockK),
              i * BlockK,  // innermost dim moves fastest
              n_idx * BlockN + block_id_x_in_cluster * multicast_stride_b, 0);
          // block_id_x_in_cluster * multicast_stride_b);
        } else {
          // normal copy B
          utils::tma_copy_3d(
              &tensormap_b,
              *mainloop_pipeline.producer_get_barrier(mainloop_pipeline_state),
              smem_ptr_B, i * BlockK, n_idx * BlockN, 0);
        }

        //         mainloop_pipeline_state.index);
        // mainloop_pipeline.params.transaction_bytes);

        // this moves to next stage, but doesn't affect the outer state
        // because this state is passed by copy, not reference.
        ++mainloop_pipeline_state;
      }
    }
  }

  DEVICE void load_tail(
      TmaPipeline<Stages, ClusterM, ClusterN> mainloop_pipeline,
      PipelineState<Stages> mainloop_pipeline_state) {
    int warp_idx = threadIdx.x / WARP_SIZE;
    int warp_idx_in_warp_group = warp_idx % 4;
    int lane_predicate = elect_one_sync();

    if (warp_idx_in_warp_group == 0 && lane_predicate == 1) {
      mainloop_pipeline.producer_tail(mainloop_pipeline_state);
    }
  }

  template <typename WGMMA>
  DEVICE void mma(TmaPipeline<Stages, ClusterM, ClusterN> mainloop_pipeline,
                  PipelineState<Stages> mainloop_pipeline_state, WGMMA wgmma,
                  AccumType* accum, int k_tile_count,
                  MainloopSharedStorage<AType, BType, CType, AccumType, BlockM,
                                        BlockN, BlockK, ClusterM, ClusterN,
                                        Stages>& shared_tensors) {
    PipelineState<Stages> mainloop_pipeline_state_release =
        mainloop_pipeline_state;

    for (int i = 0; i < WGMMA::num_elements_accumulators; ++i) {
      utils::warpgroup_fence_operand(accum[i]);
    }


    auto barrier_token =
        mainloop_pipeline.consumer_try_wait(mainloop_pipeline_state);
    mainloop_pipeline.consumer_wait(mainloop_pipeline_state, barrier_token);


    int read_stage = mainloop_pipeline_state.index;
    utils::warpgroup_arrive();

    WGMMA::wgmma<1, 1, 0, 0, 0>(
        shared_tensors.smem_A +
            read_stage * BlockM * BlockK,  // half_t* smem_A,
        shared_tensors.smem_B +
            read_stage * BlockN * BlockK,  // half_t* smem_B,
        accum                              // float* accum,
    );

    utils::warpgroup_commit_batch();
    // move to the next stage
    ++mainloop_pipeline_state;

    for (int i = 0; i < WGMMA::num_elements_accumulators; ++i) {
      utils::warpgroup_fence_operand(accum[i]);
    }



    // start from 1 because the first wgmma was done
    for (; k_tile_count > 1; --k_tile_count) {
      auto barrier_token =
          mainloop_pipeline.consumer_try_wait(mainloop_pipeline_state);
      mainloop_pipeline.consumer_wait(mainloop_pipeline_state, barrier_token);

      int read_stage = mainloop_pipeline_state.index;
      for (int i = 0; i < WGMMA::num_elements_accumulators; ++i) {
        utils::warpgroup_fence_operand(accum[i]);
      }
      utils::warpgroup_arrive();

      WGMMA::wgmma<1, 1, 1, 0, 0>(
          shared_tensors.smem_A +
              read_stage * BlockM * BlockK,  // half_t* smem_A,
          shared_tensors.smem_B +
              read_stage * BlockN * BlockK,  // half_t* smem_B,
          accum                              // float* accum,
      );

      utils::warpgroup_commit_batch();
      utils::warpgroup_wait<1>();
      for (int i = 0; i < WGMMA::num_elements_accumulators; ++i) {
        utils::warpgroup_fence_operand(accum[i]);
      }


      mainloop_pipeline.consumer_release(mainloop_pipeline_state_release);

      // mainloop_pipeline_state_release.index);

      ++mainloop_pipeline_state;
      ++mainloop_pipeline_state_release;
    }


    for (int i = 0; i < WGMMA::num_elements_accumulators; ++i) {
      utils::warpgroup_fence_operand(accum[i]);
    }
  }

  DEVICE void mma_tail(
      TmaPipeline<Stages, ClusterM, ClusterN> mainloop_pipeline,
      PipelineState<Stages> mainloop_pipeline_state, int k_tile_count) {
    mainloop_pipeline_state.advance(k_tile_count - 1);
    utils::warpgroup_wait<0>();

    mainloop_pipeline.consumer_release(mainloop_pipeline_state);
    ++mainloop_pipeline_state;
  }
};

template <class AType, class BType, class CType, class AccumType, int BlockM,
          int BlockN, int BlockK, int ClusterM, int ClusterN, int Stages>
struct EpilogueSharedStorage {};

template <class AType, class BType, class CType, class AccumType, int BlockM,
          int BlockN, int BlockK, int ClusterM, int ClusterN, int Stages>
struct EpilogueParams {};

template <class AType, class BType, class CType, class AccumType, int BlockM,
          int BlockN, int BlockK, int ClusterM, int ClusterN, int Stages>
struct Epilogue {
  DEVICE static void prefetch_tma_descriptor(
      [[maybe_unused]] const utils::TmaDescriptor* tensormap_a,
      [[maybe_unused]] const utils::TmaDescriptor* tensormap_b) {}

  template <typename WGMMA>
  DEVICE void store(CType* dst, WGMMA wgmma,
                    const AccumType accum[WGMMA::num_elements_accumulators],
                    int m_idx, int n_idx, int M, int N) {
    // this store is specialized for WgMMA M64NnK16
    int m = m_idx * BlockM;
    int n = n_idx * BlockN;

#pragma unroll
    for (int i = 0; i < WGMMA::num_elements_accumulators; ++i) {
      int m_frag, n_frag, k_wgmma, row_id, col_id, item_id;
      WGMMA::get_4d_idx_from_linear(k_wgmma, row_id, col_id, item_id, i);
      WGMMA::get_m_n_idx_fragment(m_frag, n_frag, threadIdx.x % WARP_GROUP_SIZE,
                                  k_wgmma, row_id, col_id, item_id);
      if ((m + m_frag < M) && (n + n_frag < N)) {
        dst[(m + m_frag) * N + (n + n_frag)] = (CType)accum[i];
      }
    }
  }
};

using LoadWarpOrderBarrier = OrderedBarrier<1, 2>;
using LoadWarpOrderBarrierSharedStorage = OrderedBarrierSharedStorage<1, 2>;
using LoadWarpOrderBarrierParams = OrderedBarrierParams<1, 2>;

using MathWarpGroupOrderBarrier = OrderedBarrier<2, 2>;
using MathWarpGroupOrderBarrierSharedStorage =
    OrderedBarrierSharedStorage<2, 2>;
using MathWarpGroupOrderBarrierParams = OrderedBarrierParams<2, 2>;

template <class AType, class BType, class CType, class AccumType, int BlockM,
          int BlockN, int BlockK, int ClusterM, int ClusterN, int Stages>
struct KernelSharedStorage {
  alignas(128)
      MainloopSharedStorage<AType, BType, CType, AccumType, BlockM, BlockN,
                            BlockK, ClusterM, ClusterN, Stages> mainloop;
  // epilogue: no shared storage
  alignas(16) MathWarpGroupOrderBarrierSharedStorage math_wg_order;
  alignas(16) LoadWarpOrderBarrierSharedStorage load_order;
};

template <class AType, class BType, class CType, class AccumType, int BlockM,
          int BlockN, int BlockK, int ClusterM, int ClusterN, int Stages>
struct GemmKernelParams {
  GemmParams<AType, BType, CType, AccumType> gemm_params;
  MainloopParams<AType, BType, CType, AccumType, BlockM, BlockN, BlockK,
                 ClusterM, ClusterN, Stages>
      mainloop_params;
  EpilogueParams<AType, BType, CType, AccumType, BlockM, BlockN, BlockK,
                 ClusterM, ClusterN, Stages>
      epilogue_params;
};

template <class AType, class BType, class CType, class AccumType, int BlockM,
          int BlockN, int BlockK, int ClusterM, int ClusterN, int Stages>
__global__ __launch_bounds__(384) void gpu_gemm_kernel(
    // GemmParams<AType, BType, CType, AccumType> gemm_params,
    GemmKernelParams<AType, BType, CType, AccumType, BlockM, BlockN, BlockK,
                     ClusterM, ClusterN, Stages>
        kernel_params,
    const __grid_constant__ utils::TmaDescriptor tensormap_a,
    const __grid_constant__ utils::TmaDescriptor tensormap_b) {
  // we follow CUTLASS warp specialization
  enum class WarpGroupRole { Producer = 0, Consumer0 = 1, Consumer1 = 2 };

  enum class ProducerWarpRole {
    Mainloop = 0,
    Warp1 = 1,
    Epilogue = 2,
    Warp3 = 3
  };

  extern __shared__ uint8_t raw_shared_mem[];
  // this is CUTLASS manner shared storage cast
  KernelSharedStorage<AType, BType, CType, AccumType, BlockM, BlockN, BlockK,
                      ClusterM, ClusterN, Stages>& shared_storage =
      *reinterpret_cast<
          KernelSharedStorage<AType, BType, CType, AccumType, BlockM, BlockN,
                              BlockK, ClusterM, ClusterN, Stages>*>(
          raw_shared_mem);

  // get useful ids:
  int warp_idx = threadIdx.x / WARP_SIZE;
  int warp_idx_in_warp_group =
      threadIdx.x / WARP_SIZE % WARP_NUMBER_IN_WARP_GROUP;
  int warp_group_thread_idx = threadIdx.x % WARP_GROUP_SIZE;
  uint32_t block_idx_in_cluster = block_rank_in_cluster();

  // get roles
  auto warp_group_role = WarpGroupRole(threadIdx.x / WARP_GROUP_SIZE);
  auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
  int lane_predicate = elect_one_sync();

  // only the first thread in a block launch tma prefetch
  // if ((warp_idx == 0) && lane_predicate) {
  //   Mainloop<AType, BType, CType, AccumType, BlockM, BlockN, BlockK, ClusterM,
  //            ClusterN, Stages>::prefetch_tma_descriptor(&tensormap_a,
  //                                                       &tensormap_b);
  //   Epilogue<AType, BType, CType, AccumType, BlockM, BlockN, BlockK, ClusterM,
  //            ClusterN, Stages>::prefetch_tma_descriptor(&tensormap_a,
  //                                                       &tensormap_b);
  // }


  // mainloop pipeline
  TmaPipelineParams<Stages> mainloop_pipeline_params;
  if (warp_group_role == WarpGroupRole::Producer &&
      producer_warp_role == ProducerWarpRole::Mainloop) {
    mainloop_pipeline_params.role =
        TmaPipelineParams<Stages>::ThreadCategory::Producer;
  }
  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    mainloop_pipeline_params.role =
        TmaPipelineParams<Stages>::ThreadCategory::Consumer;
  }
  mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
  mainloop_pipeline_params.num_consumers = WARP_GROUP_SIZE;
  mainloop_pipeline_params.transaction_bytes =
      Mainloop<AType, BType, CType, AccumType, BlockM, BlockN, BlockK, ClusterM,
               ClusterN, Stages>::TmaTransactionBytes;
  TmaPipeline<Stages, ClusterM, ClusterN> mainloop_pipeline(
      shared_storage.mainloop.pipeline, mainloop_pipeline_params);

  // epilogue pipeline: load and store
  // seems not necessary in this example.

  // barriers used to control warpgroups
  LoadWarpOrderBarrierParams load_order_params;
  load_order_params.group_id =
      producer_warp_role == ProducerWarpRole::Mainloop ? 0 : 1;
  load_order_params.group_size = WARP_SIZE;
  LoadWarpOrderBarrier load_order(shared_storage.load_order, load_order_params);

  MathWarpGroupOrderBarrierParams math_wg_order_params;
  math_wg_order_params.group_id = threadIdx.x / WARP_GROUP_SIZE - 1;
  math_wg_order_params.group_size = WARP_GROUP_SIZE;
  MathWarpGroupOrderBarrier math_wg_order(shared_storage.math_wg_order,
                                          math_wg_order_params);

  PipelineState<Stages> mainloop_pipeline_consumer_state;
  PipelineState<Stages> mainloop_pipeline_producer_state =
      make_producer_start_state<Stages>();

  Mainloop<AType, BType, CType, AccumType, BlockM, BlockN, BlockK, ClusterM,
           ClusterN, Stages>
      mainloop;

  WgMMA<AType, BType, AccumType, BlockM, BlockN, BlockK> wgmma;
  Epilogue<AType, BType, CType, AccumType, BlockM, BlockN, BlockK, ClusterM,
           ClusterN, Stages>
      epilogue;

  auto cluster_wait_fn = [&]() {
    if constexpr (ClusterM * ClusterN > 1) {
      utils::cluster_arrive_relaxed();
      return []() { utils::cluster_wait(); };
    } else {
      __syncthreads();
      return []() {};
    }
  }();

  TileScheduler<BlockM, BlockN, ClusterM, ClusterN> scheduler(
      kernel_params.gemm_params.M, kernel_params.gemm_params.N);
  int k_tile_count = (kernel_params.gemm_params.K + BlockK - 1) / BlockK;

  if (warp_group_role == WarpGroupRole::Consumer1) {
    scheduler.advance();
    mainloop_pipeline_consumer_state.advance(k_tile_count);
  }

  auto work_tile_info = scheduler.get_current_work_info();

  cluster_wait_fn();


  if (warp_group_role == WarpGroupRole::Producer) {
    // you can't only dealloc without alloc in consumer!
    // Handtune the magic number 40
    utils::warpgroup_reg_dealloc<40>();

    if (producer_warp_role == ProducerWarpRole::Mainloop) {
      bool first_arrive = true;
      while (work_tile_info.valid) {
        // work_tile_info.n_idx);
        mainloop.load(tensormap_a, tensormap_b, mainloop_pipeline,
                      mainloop_pipeline_producer_state, work_tile_info.m_idx,
                      work_tile_info.n_idx, k_tile_count, block_idx_in_cluster,
                      shared_storage.mainloop);


        mainloop_pipeline_producer_state.advance(k_tile_count);
        if (first_arrive) {
          load_order.arrive();
          first_arrive = false;
        }
        scheduler.advance();
        work_tile_info = scheduler.get_current_work_info();
      }

      mainloop.load_tail(mainloop_pipeline, mainloop_pipeline_producer_state);
    }

  } else if (warp_group_role == WarpGroupRole::Consumer0 ||
             warp_group_role == WarpGroupRole::Consumer1) {
    // you can't only alloc without dealloc in producer!
    // Handtune the magic number 232
    utils::warpgroup_reg_alloc<232>();

    while (work_tile_info.valid) {
      // work_tile_info.n_idx);
      AccumType accumulators[WgMMA<AType, BType, AccumType, BlockM, BlockN,
                                   BlockK>::num_elements_accumulators];

      // consuemr 0 doens't block at the first wait
      math_wg_order.wait();

      mainloop.mma(mainloop_pipeline, mainloop_pipeline_consumer_state, wgmma,
                   accumulators, k_tile_count, shared_storage.mainloop);


      math_wg_order.arrive();

      mainloop.mma_tail(mainloop_pipeline, mainloop_pipeline_consumer_state,
                        k_tile_count);
      mainloop_pipeline_consumer_state.advance(k_tile_count * 2);


      math_wg_order.wait();

      epilogue.store(kernel_params.gemm_params.C, wgmma, accumulators,
                     work_tile_info.m_idx, work_tile_info.n_idx,
                     kernel_params.gemm_params.M, kernel_params.gemm_params.N);

      math_wg_order.arrive();

      // do nothing for epilogue store tail


      scheduler.advance(2);
      work_tile_info = scheduler.get_current_work_info();
    }
  }
}

template <class AType, class BType, class CType, class AccumType, int CLUSTER_M, int CLUSTER_N, int BLOCKM, int BLOCKN, int BLOCKK, int STAGES>
void gpu_gemm(GemmParams<AType, BType, CType, AccumType> gemm_params) {
  int sm_number = get_sm_count();
  dim3 grid(CLUSTER_M * CLUSTER_N, sm_number / (CLUSTER_M * CLUSTER_N), 1);
  dim3 block(WARP_GROUP_SIZE * WG_NUMBER, 1, 1);
  dim3 cluster(CLUSTER_M, CLUSTER_N, 1);
  auto* Kernel = gpu_gemm_kernel<AType, BType, CType, AccumType, BLOCKM, BLOCKN,
                                 BLOCKK, CLUSTER_M, CLUSTER_N, STAGES>;
  size_t smemSizeBytes =
      sizeof(KernelSharedStorage<AType, BType, CType, AccumType, BLOCKM, BLOCKN,
                                 BLOCKK, CLUSTER_M, CLUSTER_N, STAGES>);
  if (smemSizeBytes >= (48 << 10)) {
    cudaError_t result = cudaFuncSetAttribute(
        Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smemSizeBytes);
    CUDA_CHECK(result);
  }
  void const* kernel = (void const*)Kernel;

  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  CUDA_CHECK(status);

  // tensor_map
  utils::TmaDescriptor tensormap_a =
      utils::make_tma_copy_desc<BLOCKM, BLOCKK, 3>(
          gemm_params.A, gemm_params.M, gemm_params.K, Swizzle<3, 4, 3>{},
          CLUSTER_N);
  utils::TmaDescriptor tensormap_b =
      utils::make_tma_copy_desc<BLOCKN, BLOCKK, 3>(
          gemm_params.B, gemm_params.N, gemm_params.K, Swizzle<3, 4, 3>{},
          CLUSTER_M);
  MainloopParams<AType, BType, CType, AccumType, BLOCKM, BLOCKN, BLOCKK,
                 CLUSTER_M, CLUSTER_N, STAGES>
      mainloop_params{};
  /// Prepare kernel params
  GemmKernelParams<AType, BType, CType, AccumType, BLOCKM, BLOCKN, BLOCKK,
                   CLUSTER_M, CLUSTER_N, STAGES>
      params{gemm_params, mainloop_params, {}};

  void* kernel_params[] = {&params, &tensormap_a, &tensormap_b};
  cudaLaunchConfig_t launch_config;
  launch_config.gridDim = {grid.x, grid.y, grid.z};
  launch_config.blockDim = {block.x, block.y, block.z};
  launch_config.dynamicSmemBytes = size_t(smemSizeBytes);
  launch_config.stream = nullptr;

  cudaLaunchAttribute launch_attribute[1];
  launch_attribute[0].id = cudaLaunchAttributeClusterDimension;
  launch_attribute[0].val.clusterDim.x = cluster.x;
  launch_attribute[0].val.clusterDim.y = cluster.y;
  launch_attribute[0].val.clusterDim.z = cluster.z;

  launch_config.attrs = launch_attribute;
  launch_config.numAttrs = 1;

  status = cudaLaunchKernelExC(&launch_config, kernel, kernel_params);
  cudaError_t launch_result = cudaGetLastError();
  CUDA_CHECK(launch_result);
}

int main(int argc, char** argv) {
  int M = testM;
  int N = testN;
  int K = testK;
  if (argc > 1) {
    assert((argc - 1) % 2 == 0);
    for (int i = 1; i < argc; i += 2) {
      char* key = argv[i];
      char* value = argv[i + 1];
      std::string keys(key);
      if (keys == "M") {
        M = std::atoi(value);
      }
      else if (keys == "N") {
        N = std::atoi(value);
      }
      else if (keys == "K") {
        K = std::atoi(value);
      }
      
    }
  }
  std::cout << "Shape M=" << M << ", N=" << N << ", K=" << K << "\n";
  using AType = half_t;
  using BType = half_t;
  using CType = half_t;
  using AccumType = float;
  AccumType alpha = 1.0;
  AccumType beta = 0.0;

  std::vector<int> AShape = {M, K};
  std::vector<int> BShape = {N, K};
  std::vector<int> CShape = {M, N};
  auto hA = alloc_cpu_tensor<AType>(AShape);
  random_fill(hA, AShape);
  auto hB = alloc_cpu_tensor<BType>(BShape);
  random_fill(hB, BShape);
  auto hC = alloc_cpu_tensor<CType>(CShape);
  random_fill(hC, CShape);
  auto goldenC = alloc_cpu_tensor<CType>(CShape);
  random_fill(goldenC, CShape);
  auto dA = alloc_gpu_tensor<AType>(AShape);
  auto dB = alloc_gpu_tensor<BType>(BShape);
  auto dgC = alloc_gpu_tensor<CType>(CShape);
  auto dC = alloc_gpu_tensor<CType>(CShape);

  /// timer
  GPUTimer gpu_timer;

  /// copy data
  copy_to_gpu(hA, dA, AShape);
  copy_to_gpu(hB, dB, BShape);
  copy_to_gpu(hC, dC, CShape);
  copy_to_gpu(goldenC, dgC, CShape);

  /// compute gpu reference
  GemmParams gpu_params(M, N, K, dA, dB, dgC, alpha, beta);
  reference_gpu_gemm(gpu_params);

  /// copy results
  copy_to_cpu(goldenC, dgC, CShape);

  static constexpr int CLUSTER_M = 2;
  static constexpr int CLUSTER_N = 1;
  static constexpr int BLOCKM = 128;
  static constexpr int BLOCKN = 128;
  static constexpr int BLOCKK = 64;
  static constexpr int STAGES = 7;

  /// compute gpu kernel
  GemmParams gpu_kernel_params(M, N, K, dA, dB, dC, alpha, beta);
  gpu_gemm<half_t, half_t, half_t, float, CLUSTER_M, CLUSTER_N, BLOCKM, BLOCKN, BLOCKK, STAGES>(gpu_kernel_params);

  /// copy results
  copy_to_cpu(hC, dC, CShape);

  /// compare results
  assert_allclose(hC, goldenC, CShape, /*rtol=*/1e-3, /*dump=*/false);
  std::cout << "Correct!\n";

  /// profile
  gpu_timer.sync_all();
  gpu_timer.tick();
  for (int i = 0; i < iters; ++i) {
    gpu_gemm<half_t, half_t, half_t, float, CLUSTER_M, CLUSTER_N, BLOCKM, BLOCKN, BLOCKK, STAGES>(gpu_params);
  }
  gpu_timer.tick();
  gpu_timer.sync_all();
  float latency = gpu_timer.report_last_ms() / float(iters);
  std::cout << "Average latency (ms) is " << latency << "\n";

  free_cpu_tensor(hA);
  free_cpu_tensor(hB);
  free_cpu_tensor(hC);
  free_cpu_tensor(goldenC);
  free_gpu_tensor(dA);
  free_gpu_tensor(dB);
  free_gpu_tensor(dC);
  return 0;
}
