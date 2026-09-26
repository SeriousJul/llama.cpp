#!/usr/bin/env bash
# Window 3: fixed CLI matrix (overflow regime) + 012 pp-share differential. Per-call timeouts.
set -uo pipefail
REPO=/home/seriousjul/src/llama.cpp
CLI=$REPO/build-004/bin/llama-cli
SRV=$REPO/build-004/bin/llama-server
MODEL=/home/seriousjul/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ4_XS.gguf
OUT=$REPO/specs/artifacts/011-sweep
SVC=llama-server.service
LOG=$OUT/window3.log; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
restart_svc() { systemctl --user start "$SVC" && say "service restarted (trap)"; }
trap restart_svc EXIT INT TERM

COMMON=(-m "$MODEL" -ngl 99 -fa on -b 2048 -ub 512 -t 16 --temp 0 --no-display-prompt -fit off --ignore-eos -st)

systemctl --user stop "$SVC"
for _ in $(seq 20); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); (( used < 1000 )) && break; sleep 2; done
say "window open, used=$used"

# config: mode kv ctx timeout
for cfg in "OFF f16 262144 60" "ON f16 262144 1200" "OFF f16 163840 600" "ON f16 163840 1200" "OFF q8_0 262144 60" "ON q8_0 262144 1200"; do
  read -r mode kv ctx tmo <<< "$cfg"
  enva=(); [[ $mode == ON ]] && enva=(GGML_CUDA_ENABLE_UNIFIED_MEMORY=1) || enva=(-u GGML_CUDA_ENABLE_UNIFIED_MEMORY)
  f="$OUT/cli3-$mode-$kv-$ctx.txt"
  say "cli3 $mode $kv ctx=$ctx start (timeout $tmo)"
  timeout "$tmo" env "${enva[@]}" "$CLI" "${COMMON[@]}" -c "$ctx" -ctk "$kv" -ctv "$kv" -n 128 \
    -f "$OUT/prompt260k.txt" > "$f" 2>&1
  rc=$?
  say "cli3 $mode $kv ctx=$ctx rc=$rc size=$(wc -c < "$f")"
done

say "012 differential: compute bytes pp2048 vs pp512-bounded"
for b in 2048 512; do
  setsid "$SRV" --model "$MODEL" --host 127.0.0.1 --port 36099 \
    --ctx-size 160000 --batch-size "$b" --ubatch-size 512 --n-gpu-layers all \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    --no-kv-unified --fit off --threads 16 --threads-batch 16 \
    > "$OUT/012-diff-b$b.log" 2>&1 &
  SP=$!
  for _ in $(seq 120); do curl -sf http://127.0.0.1:36099/health >/dev/null && break; sleep 2; done
  grep -E "compute buffer size" "$OUT/012-diff-b$b.log" | tail -2 >> "$LOG"
  kill -TERM -- -"$SP" 2>/dev/null
  for _ in $(seq 20); do kill -0 "$SP" 2>/dev/null || break; sleep 1; done
  say "b=$b captured"
done
say "window closed"
