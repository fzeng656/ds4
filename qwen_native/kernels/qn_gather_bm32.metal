// Regeneration wrapper for the M5/Metal-4 BM32 sorted-MoE gather QMM kernel.
// Requires MLX headers at build time; runtime uses only the compiled metallib.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
#include "mlx/backend/metal/kernels/quantized_nax.h"
using namespace metal;
instantiate_kernel("qn_gather_rhs_nax_bf16_gs64_b4_bm32", affine_gather_qmm_rhs_nax, bfloat, 64, 4, 32, 64, 64, 2, 2, true)
