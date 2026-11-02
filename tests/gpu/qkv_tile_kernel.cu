// I1 target kernel: Y[M,N] = X[M,K] * W[N,K]^T  (nn.Linear weight layout)
// fp16 inputs, fp32 accumulation, fp16 output. Hand-written version of what the
// PKIR->CUDA lowering should generate for one QKV-projection region on sm_86.
//
// PKIR view of this kernel:
//   buffers  As[2][BM][BK], Bs[2][BN][BK]  tier=shared, stages=2 (rows padded against bank conflicts)
//            acc[BM][BN] fp32              tier=registers (WMMA accumulator fragments)
//            Cs[BM][BN] fp32               tier=shared (epilogue staging)
//   roles    loader: all 128 threads issue 16-byte cp.async copies     (capability: async copy)
//            mma:    4 warps, each owns a 32x32 sub-tile = 2x2 WMMA tiles (capability: tensor-core mma)
//            storer: all threads convert fp32 -> fp16 and write Y with bounds checks
//   loop     kt over ceil(K / BK) K-tiles; tile regions are affine in (blockIdx.y, blockIdx.x, kt)
//   sync     "stage full": cp.async.wait_group + __syncthreads before the MMAs read a stage
//            "stage free": __syncthreads after the MMAs, before the next prefetch overwrites it
// Ragged edges are zero-filled by cp.async (src-size 0), so M and N are arbitrary; K must be a
// positive multiple of 8.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <climits>
#include <cstdint>

using namespace nvcuda;

namespace {

constexpr int BM = 64, BN = 64, BK = 32;             // block tile
constexpr int WARPS_M = 2, WARPS_N = 2;              // warp grid inside the block
constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;  // 32 x 32 per warp
constexpr int FM = WM / 16, FN = WN / 16;            // 2 x 2 WMMA tiles per warp
constexpr int THREADS = WARPS_M * WARPS_N * 32;      // 128
constexpr int STAGES = 2;
constexpr int LDS = BK + 8;            // padded row stride (halves) of the A/B stage tiles
constexpr int LDC = BN + 4;            // padded row stride (floats) of the epilogue tile
constexpr int CHUNKS_PER_ROW = BK / 8; // 16-byte (8-half) cp.async chunks per tile row

static_assert((BM * CHUNKS_PER_ROW) % THREADS == 0, "A-tile chunks must divide evenly across threads");
static_assert((BN * CHUNKS_PER_ROW) % THREADS == 0, "B-tile chunks must divide evenly across threads");
static_assert(BK % 16 == 0 && WM % 16 == 0 && WN % 16 == 0, "WMMA tiles are 16x16x16");

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem, bool pred) {
  const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  const int src_bytes = pred ? 16 : 0;  // 0 bytes read => the 16-byte destination is zero-filled
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(gmem), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Loader role: copy one BK-wide K-tile of X (rows from m0) and W (rows from n0) into a stage.
__device__ __forceinline__ void load_stage(half* As_st, half* Bs_st, const half* __restrict__ X,
                                           const half* __restrict__ W, int M, int N, int K,
                                           int m0, int n0, int k0, int tid) {
#pragma unroll
  for (int i = 0; i < BM * CHUNKS_PER_ROW / THREADS; ++i) {
    const int c = tid + i * THREADS;
    const int r = c / CHUNKS_PER_ROW, col = (c % CHUNKS_PER_ROW) * 8;
    const bool ok = m0 + r < M && k0 + col < K;
    cp_async_16(As_st + r * LDS + col, ok ? X + static_cast<size_t>(m0 + r) * K + k0 + col : X, ok);
  }
#pragma unroll
  for (int i = 0; i < BN * CHUNKS_PER_ROW / THREADS; ++i) {
    const int c = tid + i * THREADS;
    const int r = c / CHUNKS_PER_ROW, col = (c % CHUNKS_PER_ROW) * 8;
    const bool ok = n0 + r < N && k0 + col < K;
    cp_async_16(Bs_st + r * LDS + col, ok ? W + static_cast<size_t>(n0 + r) * K + k0 + col : W, ok);
  }
}

__global__ void __launch_bounds__(THREADS)
qkv_tile_kernel(const half* __restrict__ X, const half* __restrict__ W, half* __restrict__ Y,
                int M, int N, int K) {
  __shared__ __align__(128) half As[STAGES][BM][LDS];
  __shared__ __align__(128) half Bs[STAGES][BN][LDS];
  __shared__ __align__(128) float Cs[BM][LDC];

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int k_tiles = (K + BK - 1) / BK;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

  load_stage(&As[0][0][0], &Bs[0][0][0], X, W, M, N, K, m0, n0, 0, tid);
  cp_async_commit();

  for (int kt = 0; kt < k_tiles; ++kt) {
    const int s = kt % STAGES;
    if (kt + 1 < k_tiles) {
      const int s_next = (kt + 1) % STAGES;
      load_stage(&As[s_next][0][0], &Bs[s_next][0][0], X, W, M, N, K, m0, n0, (kt + 1) * BK, tid);
      cp_async_commit();
      cp_async_wait<1>();  // this thread's copies for K-tile kt have landed...
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();  // ...and every thread's copies are visible: stage s is full

    // MMA role.
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[FM];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b[FN];  // rows of W = columns of W^T
#pragma unroll
      for (int i = 0; i < FM; ++i) wmma::load_matrix_sync(a[i], &As[s][warp_m * WM + i * 16][kk], LDS);
#pragma unroll
      for (int j = 0; j < FN; ++j) wmma::load_matrix_sync(b[j], &Bs[s][warp_n * WN + j * 16][kk], LDS);
#pragma unroll
      for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
    __syncthreads();  // stage s is free: the next iteration's prefetch may overwrite it
  }

  // Storer role: stage fp32 results in shared memory, then convert and write with bounds checks.
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j)
      wmma::store_matrix_sync(&Cs[warp_m * WM + i * 16][warp_n * WN + j * 16], acc[i][j], LDC,
                              wmma::mem_row_major);
  __syncthreads();
  for (int idx = tid; idx < BM * BN; idx += THREADS) {
    const int r = idx / BN, c = idx % BN;
    if (m0 + r < M && n0 + c < N) Y[static_cast<size_t>(m0 + r) * N + n0 + c] = __float2half(Cs[r][c]);
  }
}

}  // namespace

torch::Tensor qkv_proj(torch::Tensor x, torch::Tensor w) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "x and w must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == torch::kHalf && w.scalar_type() == torch::kHalf, "x and w must be float16");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "expected x[M,K] and w[N,K]");
  x = x.contiguous();
  w = w.contiguous();
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(K > 0 && K % 8 == 0, "K must be a positive multiple of 8 (16-byte cp.async chunks)");
  TORCH_CHECK(M <= INT_MAX && N <= INT_MAX && K <= INT_MAX, "sizes must fit in int32");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "x and w must be 16-byte aligned");
  auto y = torch::empty({M, N}, x.options());
  if (M == 0 || N == 0) return y;
  const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN), static_cast<unsigned>((M + BM - 1) / BM));
  qkv_tile_kernel<<<grid, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(w.data_ptr<at::Half>()),
      reinterpret_cast<half*>(y.data_ptr<at::Half>()),
      static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qkv_proj", &qkv_proj, "Y = X @ W^T in fp16 (two-stage cp.async pipeline + WMMA tensor cores)");
}
