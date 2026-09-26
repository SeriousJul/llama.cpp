#!/usr/bin/env bash
# Window 3: does the production spec-decode shape change 012's graph number? Verify
# batch 1..8 makes several graph shapes, which is where a "pp graph + tg graph
# coexist" reclaim would have to come from. Graphs ON vs OFF, same request.
set -uo pipefail
REPO=/home/seriousjul/src/llama.cpp
SRV=$REPO/build-012/bin/llama-server
MODEL=/home/seriousjul/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ4_XS.gguf
DRAFT=/home/seriousjul/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2-GGUF/snapshots/51962825493a48b846b40126d35c799ac4093d8/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
SRC=$REPO/specs/artifacts/011-sweep
OUT=$REPO/specs/artifacts/012-sweep
PORT=36102
SVC=llama-server.service
LOG=$OUT/window3.log; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
restart_svc() { systemctl --user start "$SVC" && say "service restarted (trap)"; }
trap restart_svc EXIT INT TERM

ls -l "$DRAFT" 2>/dev/null || DRAFT=$(find /home/seriousjul/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2-GGUF -name "*Q4_K_M.gguf" | head -1)
[[ -f "$DRAFT" ]] || { say "draft model missing"; exit 1; }
P=$(python3 -c 'import json;print(json.dumps(open("'"$SRC"'/012-prompt.txt").read()))')

run() { # tag graphs n_predict
  local tag=$1 graphs=$2 n=$3
  local f="$OUT/w3-$tag.log"
  local enva=(GGML_CUDA_GRAPH_PROBE=1)
  [[ $graphs == OFF ]] && enva=(GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_GRAPH_PROBE=1)
  say "run $tag (graphs=$graphs n=$n) clocks=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits)"
  setsid env "${enva[@]}" "$SRV" --model "$MODEL" --spec-draft-model "$DRAFT" --spec-type draft-dflash \
    --spec-draft-n-max 7 --spec-draft-ngl all --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 \
    --host 127.0.0.1 --port "$PORT" --ctx-size 160000 --batch-size 2048 --ubatch-size 512 \
    --n-gpu-layers all --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    --no-kv-unified --fit off --threads 16 --threads-batch 16 -v > "$f" 2>&1 &
  local sp=$!
  for _ in $(seq 180); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done
  curl -s "http://127.0.0.1:$PORT/completion" \
    -d "{\"prompt\": $P, \"n_predict\": $n, \"temperature\": 0, \"ignore_eos\": true}" > "$OUT/w3-$tag.req"
  say "  used_now=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MiB"
  kill -TERM -- -"$sp" 2>/dev/null
  for _ in $(seq 30); do kill -0 "$sp" 2>/dev/null || break; sleep 1; done
  grep 'graph-probe' "$f" > "$OUT/w3-$tag.probe"
  say "  $tag: $(grep 'CUDA0' "$f" | grep '=' | tail -1 | sed 's/.*breakdown_print: //')"
  say "  $tag: $(grep -E 'eval time' "$f" | tail -1 | sed 's/.*print_timing: //') captures=$(grep -c 'post instantiate' "$f")"
}

systemctl --user stop "$SVC"
for _ in $(seq 20); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); (( used < 1000 )) && break; sleep 2; done
say "window open used=$used lib=$(md5sum $REPO/build-012/bin/libggml-cuda.so | cut -c1-12)"

run prod-ON  ON  512
run prod-OFF OFF 512
run prod-OFF2 OFF 512
run prod-ON2  ON  512

say "window closed"
