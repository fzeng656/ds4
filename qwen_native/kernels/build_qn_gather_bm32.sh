#!/bin/zsh
set -euo pipefail
HERE="${0:A:h}"
if [[ -z "${MLX_INCLUDE:-}" ]]; then
  MLX_INCLUDE="$(python3 - <<'PY'
import pathlib, mlx
for root in mlx.__path__:
    p = pathlib.Path(root) / 'include'
    if (p / 'mlx/backend/metal/kernels/quantized_nax.h').exists():
        print(p)
        break
else:
    raise SystemExit('MLX include directory not found; set MLX_INCLUDE')
PY
)"
fi
xcrun -sdk macosx metal -std=metal4.0 -mmacosx-version-min=26.2 -I "$MLX_INCLUDE" -c "$HERE/qn_gather_bm32.metal" -o /tmp/qn_gather_bm32.air
xcrun -sdk macosx metallib /tmp/qn_gather_bm32.air -o "$HERE/qn_gather_bm32.metallib"
