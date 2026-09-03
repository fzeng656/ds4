#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
#include "mlx/backend/metal/kernels/quantized_nax.h"
using namespace metal;

// Baseline BM32 kernel retained for small batches and explicit rollback.
instantiate_kernel("qn_gather_rhs_nax_bf16_gs64_b4_bm32", affine_gather_qmm_rhs_nax, bfloat, 64, 4, 32, 64, 64, 2, 2, true)

// Expert-aligned BM32 gather QMM adapted from MLX Metal NAX. Each tid.y is
// one descriptor and therefore exactly one expert. This removes the baseline
// kernel's per-tile expert segmentation/recompute while retaining the same
// NAX BM32 geometry.
kernel void qn_gather_rhs_nax_bf16_gs64_b4_bm32_aligned(
    const device bfloat* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device bfloat* scales [[buffer(2)]],
    const device bfloat* biases [[buffer(3)]],
    const device uint32_t* tile_start [[buffer(4)]],
    const device uint32_t* tile_expert [[buffer(5)]],
    const device uint32_t* tile_rows [[buffer(6)]],
    device bfloat* y [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& K [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int group_size = 64;
  constexpr int bits = 4;
  constexpr int BM = 32;
  constexpr int BK = 64;
  constexpr int BN = 64;
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = BK + 16 / sizeof(bfloat);

  using loader_w_t = QuantizedBlockLoader<
      bfloat, BN, BK, BK_padded, true,
      WM * WN * SIMD_SIZE, group_size, bits>;
  threadgroup bfloat Ws[BN * BK_padded];

  const uint tile = tid.y;
  const int y_row = int(tile_start[tile]);
  const short tgp_bm = short(tile_rows[tile]);
  if (tgp_bm <= 0) return;
  const uint32_t index = tile_expert[tile];
  const int y_col = int(tid.x) * BN;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int K_it = K / BK;
  const size_t stride_w = size_t(N) * K_w;
  const size_t stride_s = size_t(N) * K_g;

  auto wl = (const device uint8_t*)w;
  x += size_t(y_row) * K;
  y += size_t(y_row) * N + y_col;
  wl += size_t(y_col) * K_w;
  scales += size_t(y_col) * K_g;
  biases += size_t(y_col) * K_g;

  constexpr short SM = BM / WM; // 16
  constexpr short SN = BN / WN; // 32
  constexpr short SK = 32;
  constexpr short TM = SM / 16; // 1
  constexpr short TN = SN / 16; // 2
  constexpr short TK = SK / 16; // 2
  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);
  const short sgp_sm = min(SM, short(max(0, int(tgp_bm) - int(tm))));
  const bool full_m = (sgp_sm == SM);

  using AccumType = float;
  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();
  const device bfloat* xn = x + tm * K;
  thread loader_w_t loader_w(
      wl + size_t(index) * stride_w,
      scales + size_t(index) * stride_s,
      biases + size_t(index) * stride_s,
      K, Ws, simd_group_id, simd_lane_id);

  dispatch_bool(full_m, [&](auto kFullM) {
    for (int k = 0; k < K_it; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_w.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      STEEL_PRAGMA_NO_UNROLL
      for (int kk1 = 0; kk1 < BK; kk1 += SK) {
        NAXTile<bfloat, TM, TK> Atile;
        NAXTile<bfloat, TN, TK> Btile;
        volatile int compiler_barrier;
        if constexpr (kFullM.value) Atile.load(xn + kk1, K);
        else Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
        Btile.template load<bfloat, BK_padded, 1>(Ws + tn * BK_padded + kk1);
        tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
        (void)compiler_barrier;
      }
      xn += BK;
      loader_w.next();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if constexpr (kFullM.value) Dtile.store(y + tm * N + tn, N);
    else if (sgp_sm > 0) Dtile.store_slice(y + tm * N + tn, N, short2(0,0), short2(SN,sgp_sm));
  });
}
