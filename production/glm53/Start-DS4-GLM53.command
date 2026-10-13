#!/bin/zsh
set -e

PORT=8001
PID="$(lsof -t -iTCP:$PORT -sTCP:LISTEN 2>/dev/null | head -1 || true)"
if [[ -n "$PID" ]]; then
  echo "Stopping current DS4 PID $PID..."
  kill -INT "$PID" 2>/dev/null || true
  for i in {1..40}; do
    kill -0 "$PID" 2>/dev/null || break
    sleep 0.25
  done
fi

cd /Users/mikuru/ds4-glm53-sparse-mla-20260929

export DS4_GLM53_SPARSE_MLA_NAX=1
export DS4_GLM53_SPARSE_MLA_NAX_V2=1
export DS4_GLM53_KDA_PERCORE120=1
unset DS4_GLM53_POST_STEER_FILE DS4_GLM53_POST_STEER_SCALE

echo "GLM 5.3 Flash DogContext Q2 | Metal resident | 262K ctx | KV disk cap 32 GiB | port 8001"
echo "Model: /Users/mikuru/models/GLM-5.3-Flash-DogContext-Q2-ds4/GLM-5.3-Flash-Uncensored-IQ2-imatrix-MTP-ds4.gguf"

caffeinate -ims ./ds4-server   -m /Users/mikuru/models/GLM-5.3-Flash-DogContext-Q2-ds4/GLM-5.3-Flash-Uncensored-IQ2-imatrix-MTP-ds4.gguf   --metal   --vision /Users/mikuru/models/GLM-5.3-Flash-Official-Q2-ds4/GLM-5.3-Flash-Vision-Encoder.gguf   --ctx 262144   --kv-disk-dir /Users/mikuru/.ds4/kv-glm53   --kv-disk-space-mb 32768   --kv-cache-continued-interval-tokens 4096 --kv-cache-boundary-align-tokens 256   --host 0.0.0.0   --port 8001   --cors
