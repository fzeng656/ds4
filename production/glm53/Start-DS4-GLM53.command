#!/bin/zsh
set -e

PORT=8000
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
export DS4_GLM53_POST_STEER_FILE=/Users/mikuru/ds4-glm53-sparse-mla-20260929/dir-steering/ourablit-glm53-post/refusal-post.f32
export DS4_GLM53_POST_STEER_SCALE=0.4

nohup caffeinate -ims ./ds4-server   -m /Users/mikuru/models/GLM-5.3-Flash-Official-Q2-ds4/GLM-5.3-Flash-Q2.gguf   --metal   --vision /Users/mikuru/models/GLM-5.3-Flash-Official-Q2-ds4/GLM-5.3-Flash-Vision-Encoder.gguf   --ctx 262144   --kv-disk-dir /Users/mikuru/.ds4/kv-glm53   --kv-disk-space-mb 65536   --kv-cache-continued-interval-tokens 4096   --host 0.0.0.0   --port 8000   --cors   >/tmp/ds4-glm53-production.log 2>&1 &

NEWPID=$!
echo "Starting GLM-5.3 PID $NEWPID..."
sleep 3
lsof -nP -iTCP:8000 -sTCP:LISTEN 2>/dev/null || true
tail -20 /tmp/ds4-glm53-production.log
