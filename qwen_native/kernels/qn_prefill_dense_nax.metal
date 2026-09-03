#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused_nax.h"
using namespace metal;
instantiate_kernel("qn_prefill_bf16_dense_nt_bm128_bn128_bk64", gemm, bfloat, 128, 128, 64, 4, 4, false, true, float)
