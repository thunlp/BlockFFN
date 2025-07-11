#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define MAX_NUM_BLOCKS 128

template <typename T>
__global__ void sparse_up_kernel(
    int hidden_size,
    int tile_size,
    const int* nnz,
    const T* nz_val,
    const int* nz_idx,
    const float4* input,
    const float4* weights,
    T* output) {
    
    using T2 = typename TypeTraits<T>::half2;
    
    int tid = threadIdx.x;
    if (blockIdx.x >= nnz[0]) return;

    int row = nz_idx[blockIdx.x] * gridDim.y * tile_size + blockIdx.y * tile_size + threadIdx.y;
    int row_offset = row * hidden_size;
    float4 partial_sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int j = tid; j < hidden_size; j += blockDim.x) {
        float4 a = input[j], b = weights[row_offset + j];
        T2 prodx = __hmul2(*reinterpret_cast<T2*>(&a.x), *reinterpret_cast<T2*>(&b.x));
        T2 prody = __hmul2(*reinterpret_cast<T2*>(&a.y), *reinterpret_cast<T2*>(&b.y));
        T2 prodz = __hmul2(*reinterpret_cast<T2*>(&a.z), *reinterpret_cast<T2*>(&b.z));
        T2 prodw = __hmul2(*reinterpret_cast<T2*>(&a.w), *reinterpret_cast<T2*>(&b.w));
        partial_sum.x += float(prodx.x) + float(prodx.y);
        partial_sum.y += float(prody.x) + float(prody.y);
        partial_sum.z += float(prodz.x) + float(prodz.y);
        partial_sum.w += float(prodw.x) + float(prodw.y);
    }
    float sum = partial_sum.x + partial_sum.y + partial_sum.z + partial_sum.w;
    // sum all
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    if (tid == 0) {
        output[row] = T(sum / (1.0f + expf(-sum))) * nz_val[blockIdx.x];
    }
}

template <typename T>
__global__ void sparse_down_kernel(
    int block_size,
    int intermediate_size,
    int tile_size,
    const int* nnz,
    const int* nz_idx,
    const float4* input,
    const float4* weights,
    T* output) {

    using T2 = typename TypeTraits<T>::half2;
    
    int row = blockIdx.x * tile_size + threadIdx.y;
    int tid = threadIdx.x;
    int num = nnz[0];
    float4 partial_sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = 0; i < num; i++) {
        int offset = nz_idx[i] * block_size;
        float4 a = input[offset + tid], b = weights[row * intermediate_size + offset + tid];
        T2 prodx = __hmul2(*reinterpret_cast<T2*>(&a.x), *reinterpret_cast<T2*>(&b.x));
        T2 prody = __hmul2(*reinterpret_cast<T2*>(&a.y), *reinterpret_cast<T2*>(&b.y));
        T2 prodz = __hmul2(*reinterpret_cast<T2*>(&a.z), *reinterpret_cast<T2*>(&b.z));
        T2 prodw = __hmul2(*reinterpret_cast<T2*>(&a.w), *reinterpret_cast<T2*>(&b.w));
        partial_sum.x += float(prodx.x) + float(prodx.y);
        partial_sum.y += float(prody.x) + float(prody.y);
        partial_sum.z += float(prodz.x) + float(prodz.y);
        partial_sum.w += float(prodw.x) + float(prodw.y);
    }
    float sum = partial_sum.x + partial_sum.y + partial_sum.z + partial_sum.w;
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    if (tid == 0) {
        output[row] = T(sum);
    }
}

template <typename T>
__global__ void nonzero_kernel_64(int num_tokens, int num_blocks, const T* input, int* nnz, T* nz_val, int* nz_idx) {
    __shared__ uint64_t s_nnz_mask[32];
    __shared__ T s_input[4096];

    int col = threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    uint64_t nnz_mask = 0;
    if (col < num_blocks) {
        for (int row = threadIdx.y; row < num_tokens; row += blockDim.y) {
            T val = input[row * num_blocks + col];
            s_input[row * num_blocks + col] = val;
            nnz_mask |= (val > T(0)) ? (1ULL << col) : 0;
        }
    }
    nnz_mask |= __shfl_down_sync(0xffffffff, nnz_mask, 16);
    nnz_mask |= __shfl_down_sync(0xffffffff, nnz_mask, 8);
    nnz_mask |= __shfl_down_sync(0xffffffff, nnz_mask, 4);
    nnz_mask |= __shfl_down_sync(0xffffffff, nnz_mask, 2);
    nnz_mask |= __shfl_down_sync(0xffffffff, nnz_mask, 1);
    if (lane_id == 0) {
        s_nnz_mask[warp_id] = nnz_mask;
    }
    __syncthreads();
    if (warp_id == 0) {
        nnz_mask = s_nnz_mask[lane_id];
        nnz_mask |= __shfl_down_sync(0xffffffff, nnz_mask, 16);
        nnz_mask |= __shfl_down_sync(0xffffffff, nnz_mask, 8);
        nnz_mask |= __shfl_down_sync(0xffffffff, nnz_mask, 4);
        nnz_mask |= __shfl_down_sync(0xffffffff, nnz_mask, 2);
        nnz_mask |= __shfl_down_sync(0xffffffff, nnz_mask, 1);
        if (col == 0) {
            s_nnz_mask[0] = nnz_mask;
        }
    }
    __syncthreads();
    nnz_mask = s_nnz_mask[0];
    if (col == 0) {
        nnz[0] = __popcll(nnz_mask);
    }
    if (col < num_blocks && (nnz_mask >> col & 1)) {
        int pos = __popcll(nnz_mask & ((1ULL << (col)) - 1));
        nz_idx[pos] = col;
        for (int row = threadIdx.y; row < num_tokens; row += blockDim.y) {
            nz_val[row * num_blocks + pos] = s_input[row * num_blocks + col];
        }
    }
}

template <typename T>
__global__ void nonzero_kernel_128(int num_tokens, int num_blocks, const T* input, int* nnz, T* nz_val, int* nz_idx) {
    __shared__ uint64_t s_nnz_mask[64];
    __shared__ T s_input[4096];

    int col = threadIdx.x;
    int col_g = threadIdx.x / 64;
    int col_v = threadIdx.x % 64;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    uint64_t nnz_mask[2] = {0, 0};
    if (col < num_blocks) {
        for (int row = threadIdx.y; row < num_tokens; row += blockDim.y) {
            T val = input[row * num_blocks + col];
            s_input[row * num_blocks + col] = val;
            nnz_mask[col_g] |= (val > T(0)) ? (1ULL << col_v) : 0;
        }
    }
    nnz_mask[0] |= __shfl_down_sync(0xffffffff, nnz_mask[0], 16);
    nnz_mask[0] |= __shfl_down_sync(0xffffffff, nnz_mask[0], 8);
    nnz_mask[0] |= __shfl_down_sync(0xffffffff, nnz_mask[0], 4);
    nnz_mask[0] |= __shfl_down_sync(0xffffffff, nnz_mask[0], 2);
    nnz_mask[0] |= __shfl_down_sync(0xffffffff, nnz_mask[0], 1);
    nnz_mask[1] |= __shfl_down_sync(0xffffffff, nnz_mask[1], 16);
    nnz_mask[1] |= __shfl_down_sync(0xffffffff, nnz_mask[1], 8);
    nnz_mask[1] |= __shfl_down_sync(0xffffffff, nnz_mask[1], 4);
    nnz_mask[1] |= __shfl_down_sync(0xffffffff, nnz_mask[1], 2);
    nnz_mask[1] |= __shfl_down_sync(0xffffffff, nnz_mask[1], 1);
    if (lane_id == 0) {
        s_nnz_mask[warp_id] = nnz_mask[0];
        s_nnz_mask[32 + warp_id] = nnz_mask[1];
    }
    __syncthreads();
    if (warp_id <= 1) {
        nnz_mask[warp_id] = s_nnz_mask[lane_id + warp_id * 32];
        nnz_mask[warp_id] |= __shfl_down_sync(0xffffffff, nnz_mask[warp_id], 16);
        nnz_mask[warp_id] |= __shfl_down_sync(0xffffffff, nnz_mask[warp_id], 8);
        nnz_mask[warp_id] |= __shfl_down_sync(0xffffffff, nnz_mask[warp_id], 4);
        nnz_mask[warp_id] |= __shfl_down_sync(0xffffffff, nnz_mask[warp_id], 2);
        nnz_mask[warp_id] |= __shfl_down_sync(0xffffffff, nnz_mask[warp_id], 1);
        if (lane_id == 0) {
            s_nnz_mask[warp_id] = nnz_mask[warp_id];
        }
    }
    __syncthreads();
    nnz_mask[0] = s_nnz_mask[0];
    nnz_mask[1] = s_nnz_mask[1];
    int nnz_offset;
    nnz_offset = __popcll(nnz_mask[0]);
    if (col == 0) {
        nnz[0] = nnz_offset + __popcll(nnz_mask[1]);
    }
    if (col < num_blocks && (nnz_mask[col_g] >> col_v & 1)) {
        int pos = nnz_offset * col_g + __popcll(nnz_mask[col_g] & ((1ULL << (col_v)) - 1));
        nz_idx[pos] = col;
        for (int row = threadIdx.y; row < num_tokens; row += blockDim.y) {
            nz_val[row * num_blocks + pos] = s_input[row * num_blocks + col];
        }
    }
}

template <bool is_up, int ratio, class T, class ProblemShape, class CtaTiler,
          class AStride, class ASmemLayout, class G2STiledCopyA, class S2RTiledCopyA,
          class BStride, class BSmemLayout, class G2STiledCopyB, class S2RTiledCopyB,
          class CStride, class CSmemLayout, class TiledMma>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            T const* A, AStride dA, ASmemLayout sA_layout, G2STiledCopyA g2s_copy_a, S2RTiledCopyA s2r_copy_a,
            T const* B, BStride dB, BSmemLayout sB_layout, G2STiledCopyB g2s_copy_b, S2RTiledCopyB s2r_copy_b,
            T      * C, CStride dC, CSmemLayout          , TiledMma mma,
            int const* nnz, T const* nz_val, int const* nz_idx, int block_size) {
  using namespace cute;

  constexpr int bM = get<0>(cta_tiler), bN = get<1>(cta_tiler), bK = get<2>(cta_tiler);

  int num_block = nnz[0];
  int limit = num_block * ratio;

  int block_x;
  if constexpr (is_up) {
    if (blockIdx.x >= limit) return;
    block_x = nz_idx[blockIdx.x / ratio] * ratio + blockIdx.x % ratio;
  } else {
    block_x = blockIdx.x;
  }

  __shared__ int k_idx[256]; // TODO must >= num_block * ratio
  if constexpr (!is_up) {
    for (int i = threadIdx.x; i < num_block; i += blockDim.x) {
        int idx = nz_idx[i];
        for (int j = 0; j < ratio; ++j) {
            k_idx[i * ratio + j] = idx * ratio + j;
        }
    }
    __syncthreads();
  }

  extern __shared__ char smem_buf[];
  // __shared__ T smemA[cosize_v<ASmemLayout>];
  // __shared__ T smemB[cosize_v<BSmemLayout>];
  T* smemA = (T*)smem_buf; // smemA[cosize_v<ASmemLayout>];
  T* smemB = smemA + cosize_v<ASmemLayout>; // [cosize_v<BSmemLayout>];

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(block_x, 0, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory buffers
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K,PIPE)

  //
  // Partition the copying of A and B tiles across the threads
  //

  ThrCopy g2s_thr_copy_a = g2s_copy_a.get_slice(threadIdx.x);
  Tensor tAgA = g2s_thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = g2s_thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K,PIPE)

  ThrCopy g2s_thr_copy_b = g2s_copy_b.get_slice(threadIdx.x);
  Tensor tBgB = g2s_thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = g2s_thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K,PIPE)

  //
  // PREFETCH
  //
  auto K_PIPE_MAX = size<3>(tAsA);

  // Total count of tiles
  int k_tile_count;
  if constexpr (is_up) {
    k_tile_count = size<3>(tAgA);
  } else {
    k_tile_count = limit;
  }
  // Current tile index in gmem to read from
  int k_tile_next = 0;

  // Start async loads for all pipes but the last
  CUTE_UNROLL
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
    if constexpr (is_up) {
        copy(g2s_copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
        copy(g2s_copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
    } else {
        copy(g2s_copy_a, tAgA(_,_,_,k_idx[k_tile_next]), tAsA(_,_,_,k_pipe));
        copy(g2s_copy_b, tBgB(_,_,_,k_idx[k_tile_next]), tBsB(_,_,_,k_pipe));
    }
    cp_async_fence();
    --k_tile_count;
    if (k_tile_count > 0) { ++k_tile_next; }
  }

  //
  // Define A/B partitioning and C accumulators
  //

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate registers for pipelining
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));               // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));               // (MMA,MMA_N,MMA_K)
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.partition_fragment_C(gC);                      // (MMA,MMA_M,MMA_N)

  // Copy Atom retiling
  ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tCsA = s2r_thr_copy_a.partition_S(sA);                         // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrA_copy_view = s2r_thr_copy_a.retile_D(tCrA);                // (MMA,MMA_M,MMA_K)

  ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tCsB = s2r_thr_copy_b.partition_S(sB);                         // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCrB_copy_view = s2r_thr_copy_b.retile_D(tCrB);                // (MMA,MMA_N,MMA_K)

  // Clear the accumulators
  clear(tCrC);

  // Current pipe index in smem to read from
  int smem_pipe_read  = 0;
  // Current pipe index in smem to write to
  int smem_pipe_write = K_PIPE_MAX-1;

  // Pipe slice
  Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
  Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

  // Size of the register pipeline
  auto K_BLOCK_MAX = size<2>(tCrA);

  // PREFETCH register pipeline
  if (K_BLOCK_MAX > 1) {
    // Wait until our first prefetched tile is loaded in
    cp_async_wait<K_PIPE_MAX-2>();
    __syncthreads();

    // Prefetch the first rmem from the first k-tile
    copy(s2r_copy_a, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
    copy(s2r_copy_b, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
  }

  //
  // PIPELINED MAIN LOOP
  //

  CUTE_NO_UNROLL
  while (k_tile_count > -(K_PIPE_MAX-1))
  {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
    {
      if (k_block == K_BLOCK_MAX - 1)
      {
        // Slice the smem_pipe_read smem
        tCsA_p = tCsA(_,_,_,smem_pipe_read);
        tCsB_p = tCsB(_,_,_,smem_pipe_read);

        // Commit the smem for smem_pipe_read
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();
      }

      // Load A, B shmem->regs for k_block+1
      auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;      // static
      copy(s2r_copy_a, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
      copy(s2r_copy_b, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
      // Copy gmem to smem before computing gemm on each k-pipe
      if (k_block == 0)
      {
        if (k_tile_count > 0) {
          if constexpr (is_up) {
            copy(g2s_copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
            copy(g2s_copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
          } else {
            copy(g2s_copy_a, tAgA(_,_,_,k_idx[k_tile_next]), tAsA(_,_,_,smem_pipe_write));
            copy(g2s_copy_b, tBgB(_,_,_,k_idx[k_tile_next]), tBsB(_,_,_,smem_pipe_write));
          }
        }
        cp_async_fence();

        // Advance the gmem tile
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }

        // Advance the smem pipe
        smem_pipe_write = smem_pipe_read;
        ++smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX) ? 0 : smem_pipe_read;
      }
      // Thread-level register gemm for k_block
      gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
    }

  }

  //
  // Epilogue
  //
  T** gC_ptr = (T**)smem_buf;
  if (threadIdx.x == 0) { gC_ptr[0] = tCgC.data().get(); }
  __syncthreads();
  T* tCgC_ptr = gC_ptr[0];
  float* tCrC_ptr = tCrC.data();

  int M = get<0>(shape_MNK);
  constexpr int warp_m = 8, warp_n = 4, warp_threads = 32;
  constexpr int mma_m = mma.template tile_size_mnk<0>() / 16;
  constexpr int mma_n = mma.template tile_size_mnk<1>() / 16;
  constexpr int n_mma_m = bM / mma_m / warp_m / 2, n_mma_n = bN / mma_n / warp_n / 2;
  int block_id_i = threadIdx.x % (warp_threads * mma_m) / warp_threads, block_id_j = threadIdx.x / (warp_threads * mma_m);
  int warp_id = threadIdx.x / warp_threads, lane_id = threadIdx.x % warp_threads;
  int is_odd_warp = warp_id % 2;
  int warp_t_id_i = lane_id / warp_m, warp_t_id_j = lane_id % warp_m;

  // smem float[n_mma_n*n_mma_m, num_warps/2, 2*2, warp_threads+1] // set to 2 because warp_n / 2 = 2
  float* smem_ = (float*)(smem_buf + 256);
  constexpr int stride_3 = warp_threads + 1;
  constexpr int stride_1 = 4 * stride_3;
  int stride_0 = blockDim.x / warp_threads * 2 * stride_3;
  float* smem_C = smem_ + warp_id / 2 * stride_1 + is_odd_warp * 2 * stride_3;
  float block[2];
  for (int j = 0; j < n_mma_n; ++j) {
    for (int i = 0; i < n_mma_m; ++i) {
      float* smem = smem_C + (j * n_mma_m + i) * stride_0;
      int idx_offset = j * n_mma_m * 4 + i * 4;
      CUTE_UNROLL
      for (int k = 0; k < 4; ++k) {
        ((T*)(&block))[k] = T(tCrC_ptr[idx_offset + k]);
      }

      smem[lane_id] = block[0];
      smem[stride_3 + lane_id] = block[1];
    }
  }
  __syncthreads();

  smem_C = smem_ + warp_id / 2 * stride_1 + warp_t_id_i * stride_3 + warp_t_id_j * warp_n + is_odd_warp * 2;
  int col_offset = (block_id_i - is_odd_warp) * 2 * warp_m + lane_id;
  for (int j = 0; j < n_mma_n; ++j) {
    int row = j * 2 * warp_n * mma_n + block_id_j * 2 * warp_n;
    T* cur_ptr = tCgC_ptr + (row + is_odd_warp * 4) * M;
    for (int i = 0; i < n_mma_m; ++i) {
      float* smem = smem_C + (j * n_mma_m + i) * stride_0;

      block[0] = smem[0];
      block[1] = smem[1];

      int col = i * 2 * warp_m * mma_m + col_offset;
      if constexpr (is_up) {
        CUTE_UNROLL
        for (int k = 0; k < 4; ++k) {
          float v = ((T*)(&block))[k];
          cur_ptr[k * M + col] = T(v / (1.0f + expf(-v)));
        }
      } else {
        CUTE_UNROLL
        for (int k = 0; k < 4; ++k) {
          cur_ptr[k * M + col] = ((T*)(&block))[k];
        }
      }
    }
  }
}

template <typename T, bool is_up, int _bM, int _bN, int _bK, int _bP>
void gemm_tn(const Stream& stream,
    int num_tokens, int dim_in, int dim_out,
    const T* input, const T* weight, T* output,
    const int* nnz, const T* nz_val, const int* nz_idx, int block_size
) {
    using namespace cute;
    int M = dim_out, N = num_tokens, K = dim_in;
    auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

    // Define TN strides (mixed)
    auto dA = make_stride(K, Int<1>{});                      // (dM, dK)
    auto dB = make_stride(K, Int<1>{});                      // (dN, dK)
    auto dC = make_stride(Int<1>{}, M);                      // (dM, dN)

    // Define CTA tile sizes (static)
    auto bM = Int<_bM>{};
    auto bN = Int<_bN>{};
    auto bK = Int<_bK>{};
    auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
    auto bP = Int<_bP>{};  // Pipeline

    // Define the smem layouts (static)
    auto sA_atom = cute::composition(cute::Swizzle<3, 3, 3>{}, cute::Layout<cute::Shape<_8, _32>, cute::Stride<_32, _1>>{});
    auto sB_atom = cute::composition(cute::Swizzle<3, 3, 3>{}, cute::Layout<cute::Shape<_8, _32>, cute::Stride<_32, _1>>{});

    auto sA = tile_to_shape(sA_atom, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(sB_atom, make_shape(bN, bK, bP));
    auto sC = make_layout(make_shape(bM, bN));                        // (m,n) -> smem_idx

    // Define the thread layouts (static)

    TiledCopy G2ScopyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, T>{},
                                        Layout<Shape<_32,_4>,Stride<_4,_1>>{},
                                        Layout<Shape< _1,_8>>{});
    TiledCopy G2ScopyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, T>{},
                                        Layout<Shape<_32,_4>,Stride<_4,_1>>{},
                                        Layout<Shape< _1,_8>>{});

    TiledMMA mmaC = make_tiled_mma(typename TypeTraits<T>::mma_type{},
                                    Layout<Shape<_2,_2,_1>>{},
                                    Tile<_32,_32,_16>{});

    TiledCopy S2RcopyA = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, T>{}, mmaC);
    TiledCopy S2RcopyB = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, T>{}, mmaC);

    constexpr int ratio = is_up ? (128 / bM) : (128 / bK);

    auto kernel = gemm_device<is_up, ratio, T, decltype(prob_shape), decltype(cta_tiler),
                              decltype(dA), decltype(sA), decltype(G2ScopyA), decltype(S2RcopyA),
                              decltype(dB), decltype(sB), decltype(G2ScopyB), decltype(S2RcopyB),
                              decltype(dC), decltype(sC), decltype(mmaC)>;
    cudaFuncSetAttribute((void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 72*1024);

    dim3 dimBlock(size(mmaC));
    dim3 dimGrid;
    if constexpr (is_up) {
        dimGrid = dim3(MAX_NUM_BLOCKS, size(ceil_div(N, bN)));
    } else {
        dimGrid = dim3(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
    }
    kernel<<<dimGrid, dimBlock, 72*1024, stream.stream>>>(
        prob_shape, cta_tiler,
        weight, dA, sA, G2ScopyA, S2RcopyA,
        input, dB, sB, G2ScopyB, S2RcopyB,
        output, dC, sC, mmaC,
        nnz, nz_val, nz_idx, block_size
    );
}

template <typename T>
void sparse_up(const Stream& stream, int num_tokens, int num_blocks, int block_size, int hidden_size, const int* nnz, const T* nz_val, const int* nz_idx, const T* input, const T* weight, T* output) {
    if (num_tokens == 1) {
        constexpr int tile_size = 8;
        hidden_size /= (16 / sizeof(T));
        sparse_up_kernel<<<dim3(MAX_NUM_BLOCKS, block_size/tile_size), dim3(32, tile_size), 0, stream.stream>>>(hidden_size, tile_size, nnz, nz_val, nz_idx, (float4*)input, (float4*)weight, output);
    } else if (num_tokens == 32) {
        gemm_tn<T, /*is_up=*/true, /*bM=*/128, /*bN=*/32, /*bK=*/64, /*bP=*/3>(stream, num_tokens, hidden_size, num_blocks * block_size, input, weight, output, nnz, nz_val, nz_idx, block_size);
    } else if (num_tokens == 64) {
        gemm_tn<T, /*is_up=*/true, /*bM=*/128, /*bN=*/64, /*bK=*/32, /*bP=*/6>(stream, num_tokens, hidden_size, num_blocks * block_size, input, weight, output, nnz, nz_val, nz_idx, block_size);
    } else {
        throw std::invalid_argument("Unsupported num_tokens " + std::to_string(num_tokens));
    }
}

template <typename T>
void sparse_down(const Stream& stream, int num_tokens, int num_blocks, int block_size, int hidden_size, const int* nnz, const int* nz_idx, const T* input, const T* weight, T* output) {
    if (num_tokens == 1) {
        constexpr int tile_size = 8;
        block_size /= (16 / sizeof(T));
        sparse_down_kernel<<<hidden_size/tile_size, dim3(block_size, tile_size), 0, stream.stream>>>(block_size, num_blocks * block_size, tile_size, nnz, nz_idx, (float4*)input, (float4*)weight, output);
    } else if (num_tokens == 32) {
        gemm_tn<T, /*is_up=*/false, /*bM=*/64, /*bN=*/32, /*bK=*/128, /*bP=*/3>(stream, num_tokens, num_blocks * block_size, hidden_size, input, weight, output, nnz, nullptr, nz_idx, block_size);
    } else if (num_tokens == 64) {
        gemm_tn<T, /*is_up=*/false, /*bM=*/128, /*bN=*/64, /*bK=*/32, /*bP=*/6>(stream, num_tokens, num_blocks * block_size, hidden_size, input, weight, output, nnz, nullptr, nz_idx, block_size);
    } else {
        throw std::invalid_argument("Unsupported num_tokens " + std::to_string(num_tokens));
    }
}

template <typename T>
void nonzero(const Stream& stream, int num_tokens, int num_blocks, const T* input, int* nnz, T* nz_val, int* nz_idx) {
  if (num_blocks == 64) {
    nonzero_kernel_64<<<1, dim3(ROUND_UP(num_blocks, 64), 1024 / ROUND_UP(num_blocks, 64)), 0, stream.stream>>>(num_tokens, num_blocks, input, nnz, nz_val, nz_idx);
  } else if (num_blocks == 128) {
    nonzero_kernel_128<<<1, dim3(ROUND_UP(num_blocks, 64), 1024 / ROUND_UP(num_blocks, 64)), 0, stream.stream>>>(num_tokens, num_blocks, input, nnz, nz_val, nz_idx);
  } else {
    throw std::invalid_argument("Unsupported num_blocks " + std::to_string(num_blocks));
  }
}

