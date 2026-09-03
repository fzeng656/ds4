#!/bin/zsh
set -euo pipefail
HERE=${0:A:h}
MLX_INCLUDE=${MLX_INCLUDE:-/opt/homebrew/Cellar/omlx/0.6.1/libexec/lib/python3.11/site-packages/mlx/include}
xcrun -sdk macosx metal -std=metal4.0 -mmacosx-version-min=26.2 -I "$MLX_INCLUDE" -c "$HERE/qn_prefill_dense_nax.metal" -o /tmp/qn_prefill_dense_nax.air
xcrun -sdk macosx metal -std=metal4.0 -mmacosx-version-min=26.2 -I "$MLX_INCLUDE" -c "$HERE/qn_prefill_q8_nax.metal" -o /tmp/qn_prefill_q8_nax.air
xcrun -sdk macosx metallib /tmp/qn_prefill_dense_nax.air /tmp/qn_prefill_q8_nax.air -o "$HERE/qn_prefill_nax.metallib"
