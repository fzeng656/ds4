#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
ROOT=${SCRIPT_DIR:h:h}
OUT=${1:-"$ROOT/build/qwen-native"}
BIN="$OUT/bin"
SHARE="$OUT/share/qwen-native"

mkdir -p "$BIN" "$SHARE"

COMMON=(
  -O3 -ffast-math -fobjc-arc -Wall -Wextra -Wno-unused-function -Wno-unused-variable
  -DQN_DISABLE_SOURCE_METALLIB_FALLBACK=1
  -I "$ROOT/qwen_native"
  -framework Foundation -framework Metal -framework MetalPerformanceShaders
)
SOURCES=(
  "$ROOT/qwen_native/qn_runtime.m"
  "$ROOT/qwen_native/qn_manifest.m"
  "$ROOT/qwen_native/qn_model_io.m"
  "$ROOT/qwen_native/qn_gdn_layer.m"
  "$ROOT/qwen_native/qn_ple_layer.m"
  "$ROOT/qwen_native/qn_qwen4_layer.m"
  "$ROOT/qwen_native/qn_qwen4_model.m"
)

clang "${COMMON[@]}" "$ROOT/qwen_native/tools/qn_generate_ids.m" "${SOURCES[@]}" -o "$BIN/qn_generate_ids"
clang -O2 -fobjc-arc -Wall -Wextra -DQN_DISABLE_SOURCE_METALLIB_FALLBACK=1 -I "$ROOT/qwen_native" \
  "$ROOT/qwen_native/tools/qn_runtime_probe.m" "$ROOT/qwen_native/qn_runtime.m" \
  -framework Foundation -framework Metal -o "$BIN/qn_runtime_probe"

cp "$ROOT/qwen_native/kernels/qn_gather_bm32.metallib" "$SHARE/qn_gather_bm32.metallib"
cp "$ROOT/NOTICE" "$SHARE/NOTICE"
cp "$ROOT/LICENSE" "$SHARE/LICENSE"
cp "$ROOT/LICENSE-APACHE-2.0" "$SHARE/LICENSE-APACHE-2.0"

printf 'qwen-native production bundle: %s\n' "$OUT"
printf '  runtime probe: QN_RUNTIME_MODE=stable %s\n' "$BIN/qn_runtime_probe"
printf '  inference:     QN_RUNTIME_MODE=stable %s MODEL_DIR MANIFEST NGRAM MAX_TOKENS TOKEN_ID...\n' "$BIN/qn_generate_ids"
