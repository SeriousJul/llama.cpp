#!/usr/bin/env bash
# Window 2: decompose the base "unaccounted" band that window 1 showed is NOT graphs.
# All rows run with graphs off (GGML_CUDA_DISABLE_GRAPHS) so the 16 MiB graph cost is gone.
#   idle-c160k : model + KV + compute reserved, no inference at all
#   idle-c8k   : same, small ctx -> does the band scale with allocation size?
#   ngl0       : nothing offloaded -> ggml/cuBLAS init floor
#   pp-only    : one 1331-token prefill, 1 generated token -> what pp leaves behind
set -uo pipefail
REPO=/home/seriousjul/src/llama.cpp
SRV=$REPO/build-012/bin/llama-server
MODEL=/home/seriousjul/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ4_XS.gguf
SRC=$REPO/specs/artifacts/011-sweep
OUT=$REPO/specs/artifacts/012-sweep
PORT=36101
SVC=llama-server.service
LOG=$OUT/window2.log; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
restart_svc() { systemctl --user start "$SVC" && say "service restarted (trap)"; }
trap restart_svc EXIT INT TERM

P=$(python3 -c 'import json;print(json.dumps(open("'"$SRC"'/012-prompt.txt").read()))')

run() { # tag ctx ngl n_predict
  local tag=$1 ctx=$2 ngl=$3 n=$4
  local f="$OUT/w2-$tag.log"
  say "run $tag (ctx=$ctx ngl=$ngl n=$n) clocks=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits)"
  setsid env GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_GRAPH_PROBE=1 "$SRV" --model "$MODEL" \
    --host 127.0.0.1 --port "$PORT" --ctx-size "$ctx" --batch-size 2048 --ubatch-size 512 \
    --n-gpu-layers "$ngl" --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    --no-kv-unified --fit off --threads 16 --threads-batch 16 -v > "$f" 2>&1 &
  local sp=$!
  for _ in $(seq 120); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done
  if (( n > 0 )); then
    curl -s "http://127.0.0.1:$PORT/completion" \
      -d "{\"prompt\": $P, \"n_predict\": $n, \"temperature\": 0, \"ignore_eos\": true}" > "$OUT/w2-$tag.req"
    sleep 2
  else
    sleep 5
  fi
  say "  used_now=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MiB"
  kill -TERM -- -"$sp" 2>/dev/null
  for _ in $(seq 30); do kill -0 "$sp" 2>/dev/null || break; sleep 1; done
  say "  $tag: $(grep 'CUDA0' "$f" | grep '=' | tail -1 | sed 's/.*breakdown_print: //')"
}

systemctl --user stop "$SVC"
for _ in $(seq 20); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); (( used < 1000 )) && break; sleep 2; done
say "window open used=$used lib=$(md5sum $REPO/build-012/bin/libggml-cuda.so | cut -c1-12)"

run idle-c160k 160000 all 0
run idle-c8k    8192 all 0
run ngl0-c160k 160000 0   0
run pp-only   160000 all  1

say "window closed"
