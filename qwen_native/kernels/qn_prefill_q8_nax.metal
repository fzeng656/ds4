#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
#include "mlx/backend/metal/kernels/quantized_nax.h"
using namespace metal;
instantiate_kernel("qn_prefill_q8_nax_bf16_gs64", affine_qmm_t_nax, bfloat, 64, 8, true, false, 64, 64, 64, 2, 2)
