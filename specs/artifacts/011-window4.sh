#!/usr/bin/env bash
# Window 4: 011 true-overflow rows + missing OFF baselines; 012 differential with a real request.
set -uo pipefail
REPO=/home/seriousjul/src/llama.cpp
CLI=$REPO/build-004/bin/llama-cli
TOK=$REPO/build-004/bin/llama-tokenize
SRV=$REPO/build-004/bin/llama-server
MODEL=/home/seriousjul/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ4_XS.gguf
OUT=$REPO/specs/artifacts/011-sweep
SVC=llama-server.service
LOG=$OUT/window4.log; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
restart_svc() { systemctl --user start "$SVC" && say "service restarted (trap)"; }
trap restart_svc EXIT INT TERM

# ~270K token prompt
python3 -c "print(('the quick brown fox jumps over the lazy dog while sensors record the ambient telemetry of the plant floor ' * 11000), end='')" > "$OUT/prompt270k.txt"
T=$("$TOK" -m "$MODEL" --stdin --show-count < "$OUT/prompt270k.txt" 2>&1 | tail -1)
T150=$("$TOK" -m "$MODEL" --stdin --show-count < "$OUT/prompt260k.txt" 2>&1 | tail -1)
say "token counts: prompt260k: $T150 ; prompt270k: $T"

COMMON=(-m "$MODEL" -ngl 99 -fa on -b 2048 -ub 512 -t 16 --temp 0 --no-display-prompt -fit off --ignore-eos -st)
run() { # mode kv ctx timeout promptfile tag
  local mode=$1 kv=$2 ctx=$3 tmo=$4 pf=$5 tag=$6
  local enva=(); [[ $mode == ON ]] && enva=(GGML_CUDA_ENABLE_UNIFIED_MEMORY=1) || enva=(-u GGML_CUDA_ENABLE_UNIFIED_MEMORY)
  local f="$OUT/w4-$tag.txt"
  say "w4 $tag start (tmo $tmo)"
  timeout "$tmo" env "${enva[@]}" "$CLI" "${COMMON[@]}" -c "$ctx" -ctk "$kv" -ctv "$kv" -n 128 -f "$pf" > "$f" 2>&1
  local rc=$?
  say "w4 $tag rc=$rc $(grep -oE 'Prompt: *[0-9.]+ t/s \| Generation: *[0-9.]+ t/s' "$f" | tail -1)"
}

systemctl --user stop "$SVC"
for _ in $(seq 20); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); (( used < 1000 )) && break; sleep 2; done
say "window open used=$used"

run OFF q8_0 160000 900 "$OUT/prompt260k.txt" OFF-q8-c160k-p150k
run ON  q8_0 160000 900 "$OUT/prompt260k.txt" ON-q8-c160k-p150k
run OFF q8_0 262144 900 "$OUT/prompt270k.txt" OFF-q8-c262k-p270k
run ON  q8_0 262144 900 "$OUT/prompt270k.txt" ON-q8-c262k-p270k
run ON  f16  262144 1800 "$OUT/prompt270k.txt" ON-f16-c262k-p270k

say "012 differential"
for b in 2048 512; do
  setsid "$SRV" --model "$MODEL" --host 127.0.0.1 --port 36099 \
    --ctx-size 160000 --batch-size "$b" --ubatch-size 512 --n-gpu-layers all \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    --no-kv-unified --fit off --threads 16 --threads-batch 16 \
    > "$OUT/012d-b$b.log" 2>&1 &
  SP=$!
  for _ in $(seq 60); do curl -sf http://127.0.0.1:36099/health >/dev/null && break; sleep 1; done
  curl -s "http://127.0.0.1:36099/completion" -d '{"prompt": "Hello", "n_predict": 8}' > /dev/null
  sleep 2
  kill -TERM -- -"$SP" 2>/dev/null
  for _ in $(seq 30); do kill -0 "$SP" 2>/dev/null || break; sleep 1; done
  say "b=$b: $(grep -E 'sched_reserve.*compute buffer size' "$OUT/012d-b$b.log" | head -2 | tr '\n' ' ')"
done
say "window closed"
