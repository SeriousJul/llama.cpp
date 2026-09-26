#!/usr/bin/env bash
# Window 4: which phase owns the one captured graph. If pp captures its own graph, the
# pp-only row must show an instantiate; if only decode does, the 1-token-prompt row must.
set -uo pipefail
REPO=/home/seriousjul/src/llama.cpp
SRV=$REPO/build-012/bin/llama-server
MODEL=/home/seriousjul/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ4_XS.gguf
SRC=$REPO/specs/artifacts/011-sweep
OUT=$REPO/specs/artifacts/012-sweep
PORT=36103
SVC=llama-server.service
LOG=$OUT/window4.log; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
restart_svc() { systemctl --user start "$SVC" && say "service restarted (trap)"; }
trap restart_svc EXIT INT TERM

PBIG=$(python3 -c 'import json;print(json.dumps(open("'"$SRC"'/012-prompt.txt").read()))')

run() { # tag prompt n
  local tag=$1 prompt=$2 n=$3
  local f="$OUT/w4-$tag.log"
  say "run $tag (n=$n, prompt ${#prompt} chars) clocks=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits)"
  setsid env GGML_CUDA_GRAPH_PROBE=1 "$SRV" --model "$MODEL" \
    --host 127.0.0.1 --port "$PORT" --ctx-size 160000 --batch-size 2048 --ubatch-size 512 \
    --n-gpu-layers all --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    --no-kv-unified --fit off --threads 16 --threads-batch 16 -v > "$f" 2>&1 &
  local sp=$!
  for _ in $(seq 120); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done
  curl -s "http://127.0.0.1:$PORT/completion" \
    -d "{\"prompt\": $prompt, \"n_predict\": $n, \"temperature\": 0, \"ignore_eos\": true}" > "$OUT/w4-$tag.req"
  kill -TERM -- -"$sp" 2>/dev/null
  for _ in $(seq 30); do kill -0 "$sp" 2>/dev/null || break; sleep 1; done
  grep 'graph-probe' "$f" > "$OUT/w4-$tag.probe"
  say "  $tag: instantiates=$(grep -c 'post instantiate' "$f") captures_open=$(grep -c 'pre capture' "$f")"
  sed 's/^/    /' "$OUT/w4-$tag.probe" | tail -4 | tee -a "$LOG"
  say "  $tag: $(grep -E 'prompt eval time|eval time' "$f" | tail -2 | sed 's/.*task 0 | //' | tr '\n' '|')"
}

systemctl --user stop "$SVC"
for _ in $(seq 20); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); (( used < 1000 )) && break; sleep 2; done
say "window open used=$used lib=$(md5sum $REPO/build-012/bin/libggml-cuda.so | cut -c1-12)"

run pp-only      "$PBIG" 1     # 1331-token prefill, one generated token
run tg-only      '"Hi"'  8     # one-token prompt, 8 decode steps
run pp-tg-long   "$PBIG" 64

say "window closed"
