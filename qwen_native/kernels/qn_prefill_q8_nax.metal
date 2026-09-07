#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
#include "mlx/backend/metal/kernels/quantized_nax.h"
using namespace metal;
instantiate_kernel("qn_prefill_q8_nax_bf16_gs64", affine_qmm_t_nax, bfloat, 64, 8, true, false, 64, 64, 64, 2, 2)
inline float qn_bf16_to_f32(bfloat x){ return float(x); }

kernel void qn_prefill_q8_nax_f32_bf16params_gs64(
    const device uint32_t* w32 [[buffer(0)]],
    const device bfloat* scales [[buffer(1)]],
    const device bfloat* biases [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& x_batch_ndims [[buffer(8)]],
    const constant int* x_shape [[buffer(9)]],
    const constant int64_t* x_strides [[buffer(10)]],
    const constant int& w_batch_ndims [[buffer(11)]],
    const constant int* w_shape [[buffer(12)]],
    const constant int64_t* w_strides [[buffer(13)]],
    const constant int64_t* s_strides [[buffer(14)]],
    const constant int64_t* b_strides [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)x_batch_ndims;(void)x_shape;(void)x_strides;(void)w_batch_ndims;(void)w_shape;(void)w_strides;(void)s_strides;(void)b_strides;
  constexpr int BM=64,BN=64,BK=64,WM=2,WN=2;
  constexpr int BKP=BK+16/sizeof(float);
  threadgroup float Ws[BN*BKP];
  const int yrow=int(tid.y)*BM,ycol=int(tid.x)*BN;
  const short SM=BM/WM,SN=BN/WN,SK=32,TM=SM/16,TN=SN/16,TK=SK/16;
  const short tm=SM*(simd_gid/WN),tn=SN*(simd_gid%WN);
  const short sm=min(SM,short(max(0,M-(yrow+tm))));
  NAXTile<float,TM,TN>D;D.clear();
  const device uint8_t* w=(const device uint8_t*)w32;
  const int KG=K/64;
  const device float* xp=x+(size_t)(yrow+tm)*K;
  for(int k0=0;k0<K;k0+=BK){
    for(uint idx=lid;idx<BN*BK;idx+=WM*WN*SIMD_SIZE){uint rr=idx/BK,cc=idx-rr*BK;uint orow=ycol+rr;if(orow<(uint)N){float sc=float(scales[(size_t)orow*KG+k0/64]);float bi=float(biases[(size_t)orow*KG+k0/64]);Ws[rr*BKP+cc]=float(w[(size_t)orow*K+k0+cc])*sc+bi;}else Ws[rr*BKP+cc]=0.0f;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_NO_UNROLL
    for(int kk=0;kk<BK;kk+=SK){NAXTile<float,TM,TK>A;NAXTile<float,TN,TK>B;volatile int compiler_barrier;if(sm==SM)A.load(xp+k0+kk,K);else if(sm>0)A.load_safe(xp+k0+kk,K,short2(SK,sm));B.template load<float,BKP,1>(Ws+tn*BKP+kk);if(sm>0)tile_matmad_nax(D,A,metal::bool_constant<false>{},B,metal::bool_constant<true>{});(void)compiler_barrier;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if(sm==SM)D.store(y+(size_t)(yrow+tm)*N+ycol+tn,N);else if(sm>0)D.store_safe(y+(size_t)(yrow+tm)*N+ycol+tn,N,short2(SN,sm));
}
